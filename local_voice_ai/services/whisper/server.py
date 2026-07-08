"""Minimal OpenAI-compatible STT server backed by ``faster-whisper``.

Replaces the ``vox-box`` dependency, which pins ``aiofiles==23.2.1`` and can
never share a venv with ``livekit-agents`` — the reason the whisper extra was
uninstallable alongside ``[ml]``. Exposes only what
``livekit.plugins.openai.STT`` needs, mirroring the nemotron service:

  - ``POST /v1/audio/transcriptions`` → {"text": ...} (or SSE deltas)
  - ``GET  /v1/models``               → list of one model
  - ``GET  /health``                  → readiness probe

The model is loaded once at startup and reused across requests.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

logger = logging.getLogger("whisper")
logging.basicConfig(level=logging.INFO)

MODEL_NAME = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-small")
DEVICE = os.getenv("DEVICE", "cpu")

_model = None


def _load_model() -> None:
    global _model
    # faster-whisper (CTranslate2) supports cpu/cuda only — anything else
    # (e.g. mps) falls back to cpu.
    device = "cuda" if DEVICE == "cuda" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info("loading %s (device=%s, compute_type=%s)", MODEL_NAME, device, compute_type)
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    _model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
    logger.info("whisper model ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Whisper STT Server", lifespan=lifespan)


def _sse_generator(segments):
    full_text = ""
    for segment in segments:
        delta = segment.text
        full_text += delta
        event = {"type": "transcript.text.delta", "delta": delta}
        yield f"data: {json.dumps(event)}\n\n"

    done_event = {"type": "transcript.text.done", "text": full_text.strip()}
    yield f"data: {json.dumps(done_event)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),
    response_format: str | None = Form("json"),
    stream: str | None = Form(None),
    language: str | None = Form(None),
    temperature: str | None = Form(None),
    prompt: str | None = Form(None),
):
    del model  # the loaded model is fixed at startup

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    is_stream = stream is not None and stream.lower() in ("true", "1", "yes")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        segments, _info = _model.transcribe(
            io.BytesIO(audio_bytes),
            language=language or None,
            temperature=float(temperature) if temperature else 0.0,
            initial_prompt=prompt or None,
        )
    except Exception as error:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {error}") from error

    if is_stream:
        return StreamingResponse(
            _sse_generator(segments),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ``segments`` is a lazy generator; consuming it runs the actual decode.
    text = "".join(segment.text for segment in segments).strip()

    if response_format == "text":
        return PlainTextResponse(content=text)
    if response_format == "verbose_json":
        return JSONResponse(
            content={
                "text": text,
                "task": "transcribe",
                "language": _info.language,
                "duration": _info.duration,
            }
        )

    return JSONResponse(content={"text": text})


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "systran",
                }
            ],
        }
    )


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper STT Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
