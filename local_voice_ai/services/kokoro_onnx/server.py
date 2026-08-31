"""OpenAI-compatible TTS server backed by ``kokoro-onnx``.

This server uses the same Kokoro 82M voices as the PyTorch server. The FP16
ONNX graph uses much less resident memory, which makes the complete voice stack
fit on small unified-memory devices such as Jetson Orin Nano.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("kokoro-onnx")
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.getenv("KOKORO_MODEL_ID", "kokoro-onnx")
DEFAULT_VOICE = os.getenv("KOKORO_DEFAULT_VOICE", "af_nova")
LANG = os.getenv("KOKORO_LANG", "en-us")

_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODEL_URL = os.getenv("KOKORO_ONNX_MODEL_URL", f"{_RELEASE}/kokoro-v1.0.fp16.onnx")
VOICES_URL = os.getenv("KOKORO_ONNX_VOICES_URL", f"{_RELEASE}/voices-v1.0.bin")

_kokoro = None
_synthesis_lock = asyncio.Lock()


def _cache_dir() -> Path:
    base = os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "kokoro-onnx"


def _fetch(url: str) -> Path:
    """Download a file once and return its cache path."""
    dest = _cache_dir() / os.path.basename(url)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    logger.info("downloading %s", url)
    urllib.request.urlretrieve(url, partial)
    partial.rename(dest)
    logger.info("saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def _load_model() -> None:
    global _kokoro
    from kokoro_onnx import Kokoro  # type: ignore[import-not-found]

    model_path = _fetch(MODEL_URL)
    voices_path = _fetch(VOICES_URL)
    logger.info("loading %s", model_path.name)
    _kokoro = Kokoro(str(model_path), str(voices_path))
    logger.info("kokoro-onnx ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Kokoro ONNX TTS Server", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    response_format: str | None = "mp3"
    speed: float | None = 1.0


def _synthesize(text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
    if _kokoro is None:
        raise RuntimeError("model not loaded")
    samples, sample_rate = _kokoro.create(text, voice=voice, speed=speed, lang=LANG)
    return np.asarray(samples, dtype=np.float32), sample_rate


def _encode(audio: np.ndarray, sample_rate: int, fmt: str) -> tuple[bytes, str]:
    fmt = (fmt or "mp3").lower()
    buf = io.BytesIO()

    if fmt in {"mp3", "opus", "aac", "flac"}:
        try:
            sf.write(buf, audio, sample_rate, format=fmt.upper())
            return buf.getvalue(), f"audio/{fmt}"
        except Exception:
            buf = io.BytesIO()

    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


def _pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


async def _stream_pcm(text: str, voice: str, speed: float):
    if _kokoro is None:
        raise RuntimeError("model not loaded")
    # espeak-ng has process-global state. Serializing phonemization prevents
    # concurrent requests from producing duplicated or reordered speech.
    async with _synthesis_lock:
        async for audio, _sample_rate in _kokoro.create_stream(
            text,
            voice=voice,
            speed=speed,
            lang=LANG,
        ):
            yield _pcm16(audio)


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest) -> Response:
    if _kokoro is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.input:
        raise HTTPException(status_code=400, detail="input is required")

    voice = req.voice or DEFAULT_VOICE
    if (req.response_format or "").lower() == "pcm":
        return StreamingResponse(
            _stream_pcm(req.input, voice, float(req.speed or 1.0)),
            media_type="audio/pcm",
            headers={"X-Sample-Rate": "24000"},
        )
    try:
        audio, sample_rate = _synthesize(req.input, voice, float(req.speed or 1.0))
    except Exception as exc:
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    data, media_type = _encode(audio, sample_rate, req.response_format or "mp3")
    return Response(content=data, media_type=media_type)


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "hexgrad",
                }
            ],
        }
    )


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": _kokoro is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kokoro ONNX TTS Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8880)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
