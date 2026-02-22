"""
OpenAI-compatible STT server wrapping NVIDIA's nemotron-speech-streaming-en-0.6b model.

Usage:
    python server.py [--host 0.0.0.0] [--port 8000]
"""

import argparse
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger("stt-server")
logging.basicConfig(level=logging.INFO)

MODEL_NAME = os.getenv("NEMOTRON_MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b")
MODEL_ID = os.getenv("NEMOTRON_MODEL_ID", "nemotron-speech-streaming")
TARGET_SAMPLE_RATE = 16000

asr_model = None


def load_model():
    global asr_model
    logger.info("Loading model %s ...", MODEL_NAME)
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    asr_model.eval()

    if torch.cuda.is_available():
        asr_model = asr_model.cuda()
        logger.info("Model on CUDA")
    elif torch.backends.mps.is_available():
        try:
            asr_model = asr_model.to("mps")
            logger.info("Model on MPS")
        except Exception:
            logger.info("MPS unavailable for this model, using CPU")
    else:
        logger.info("Model on CPU")

    logger.info("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Nemotron STT Server", lifespan=lifespan)


def load_audio(audio_bytes: bytes, filename: str) -> np.ndarray:
    """Load audio bytes, resample to 16kHz mono, return float32 numpy array."""
    suffix = os.path.splitext(filename)[1] if filename else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    try:
        data, sample_rate = sf.read(tmp_in_path, dtype="float32")
    except Exception:
        import torchaudio

        waveform, sample_rate = torchaudio.load(tmp_in_path)
        data = waveform.numpy()
        if data.ndim == 2:
            data = data.mean(axis=0)
    finally:
        os.unlink(tmp_in_path)

    if data.ndim > 1:
        data = data.mean(axis=-1) if data.shape[-1] <= data.shape[0] else data.mean(axis=0)

    if sample_rate != TARGET_SAMPLE_RATE:
        import torchaudio

        waveform = torch.tensor(data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )
        waveform = resampler(waveform)
        data = waveform.squeeze(0).numpy()

    return data


def direct_transcribe(audio: np.ndarray) -> str:
    """Run full-file transcription using direct model forward pass."""
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(asr_model.device)
    audio_len = torch.tensor([audio.shape[0]], dtype=torch.long).to(asr_model.device)

    with torch.no_grad():
        processed, processed_len = asr_model.preprocessor(
            input_signal=audio_tensor,
            length=audio_len,
        )
        encoded, encoded_len = asr_model.encoder(
            audio_signal=processed,
            length=processed_len,
        )
        hypotheses = asr_model.decoding.rnnt_decoder_predictions_tensor(
            encoded,
            encoded_len,
            return_hypotheses=False,
        )

    first_hypothesis = hypotheses[0]
    return first_hypothesis.text if hasattr(first_hypothesis, "text") else str(first_hypothesis)


def streaming_transcribe(audio: np.ndarray):
    """Yield incremental transcript deltas using conformer_stream_step."""
    model = asr_model
    device = model.device

    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
    audio_len = torch.tensor([audio.shape[0]], dtype=torch.long).to(device)

    with torch.no_grad():
        processed, _processed_len = model.preprocessor(
            input_signal=audio_tensor,
            length=audio_len,
        )

        streaming_cfg = model.encoder.streaming_cfg
        chunk_size = streaming_cfg.chunk_size
        shift_size = streaming_cfg.shift_size

        chunk_frames = chunk_size[0] if isinstance(chunk_size, (list, tuple)) else chunk_size
        shift_frames = shift_size[0] if isinstance(shift_size, (list, tuple)) else shift_size

        pre_encode_cache = streaming_cfg.pre_encode_cache_size
        pre_cache_frames = (
            pre_encode_cache[0]
            if isinstance(pre_encode_cache, (list, tuple))
            else pre_encode_cache
        )

        total_frames = processed.shape[2]
        previous_text = ""
        previous_hypotheses = None

        cache_last_channel, cache_last_time, cache_last_channel_len = (
            model.encoder.get_initial_cache_state(batch_size=1)
        )

        if pre_cache_frames > 0:
            pad = torch.zeros(
                processed.shape[0],
                processed.shape[1],
                pre_cache_frames,
                device=device,
                dtype=processed.dtype,
            )
            processed = torch.cat([pad, processed], dim=2)
            total_frames = processed.shape[2]

        offset = 0
        while offset < total_frames:
            end = min(offset + chunk_frames, total_frames)
            chunk = processed[:, :, offset:end]
            chunk_len = torch.tensor([chunk.shape[2]], dtype=torch.long).to(device)

            result = model.conformer_stream_step(
                processed_signal=chunk,
                processed_signal_length=chunk_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                previous_hypotheses=previous_hypotheses,
                return_transcription=True,
            )

            (
                _greedy_preds,
                all_hypotheses,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                best_hypothesis,
            ) = result

            if best_hypothesis and len(best_hypothesis) > 0:
                hypothesis = best_hypothesis[0]
                current_text = (
                    hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)
                )
            elif isinstance(all_hypotheses, list) and len(all_hypotheses) > 0:
                first = all_hypotheses[0]
                if isinstance(first, str):
                    current_text = first
                elif hasattr(first, "text"):
                    current_text = first.text
                else:
                    current_text = str(first)
            else:
                current_text = ""

            previous_hypotheses = best_hypothesis

            if current_text and current_text != previous_text:
                delta = current_text[len(previous_text) :]
                if delta:
                    yield delta
                previous_text = current_text

            offset += shift_frames


async def sse_generator(audio: np.ndarray):
    """Generate SSE events from streaming transcription."""
    full_text = ""
    for delta in streaming_transcribe(audio):
        full_text += delta
        event = {"type": "transcript.text.delta", "delta": delta}
        yield f"data: {json.dumps(event)}\n\n"

    done_event = {"type": "transcript.text.done", "text": full_text.strip()}
    yield f"data: {json.dumps(done_event)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    response_format: Optional[str] = Form("json"),
    stream: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
):
    del model, language, temperature, prompt

    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    is_stream = stream is not None and stream.lower() in ("true", "1", "yes")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio = load_audio(audio_bytes, file.filename or "audio.wav")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {error}")

    if is_stream:
        return StreamingResponse(
            sse_generator(audio),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        text = direct_transcribe(audio)
    except Exception as error:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {error}")

    if response_format == "text":
        return PlainTextResponse(content=text)
    if response_format == "verbose_json":
        return JSONResponse(
            content={
                "text": text,
                "task": "transcribe",
                "language": "en",
                "duration": None,
            }
        )

    return JSONResponse(content={"text": text})


@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "nvidia",
                }
            ],
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": asr_model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nemotron STT Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()

    uvicorn.run(app, host=arguments.host, port=arguments.port)
