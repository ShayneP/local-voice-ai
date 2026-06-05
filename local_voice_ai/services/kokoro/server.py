"""Minimal OpenAI-compatible TTS server backed by the ``kokoro`` PyPI package.

Replaces the ``ghcr.io/remsky/kokoro-fastapi-cpu`` image with a small in-tree
service that exposes only what ``livekit.plugins.openai.TTS`` needs:

  - ``POST /v1/audio/speech``  → audio bytes
  - ``GET  /v1/models``         → list of one model
  - ``GET  /health``            → readiness probe

The model is loaded once at startup and reused across requests.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logger = logging.getLogger("kokoro")
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.getenv("KOKORO_MODEL_ID", "kokoro")
LANG_CODE = os.getenv("KOKORO_LANG_CODE", "a")  # 'a' = American English
DEFAULT_VOICE = os.getenv("KOKORO_DEFAULT_VOICE", "af_nova")
SAMPLE_RATE = 24000

_pipeline = None


def _load_pipeline() -> None:
    global _pipeline
    logger.info("loading kokoro pipeline (lang=%s)", LANG_CODE)
    from kokoro import KPipeline  # type: ignore[import-not-found]

    _pipeline = KPipeline(lang_code=LANG_CODE)
    logger.info("kokoro pipeline ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_pipeline()
    yield


app = FastAPI(title="Kokoro TTS Server", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0


def _synthesize(text: str, voice: str, speed: float) -> np.ndarray:
    if _pipeline is None:
        raise RuntimeError("pipeline not loaded")
    chunks: list[np.ndarray] = []
    for _, _, audio in _pipeline(text, voice=voice, speed=speed):
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def _encode(audio: np.ndarray, fmt: str) -> tuple[bytes, str]:
    fmt = (fmt or "mp3").lower()
    buf = io.BytesIO()

    if fmt in {"mp3", "opus", "aac", "flac"}:
        try:
            sf.write(buf, audio, SAMPLE_RATE, format=fmt.upper())
            return buf.getvalue(), f"audio/{fmt}"
        except Exception:
            buf = io.BytesIO()  # fall through to wav

    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest) -> Response:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.input:
        raise HTTPException(status_code=400, detail="input is required")

    voice = req.voice or DEFAULT_VOICE
    try:
        audio = _synthesize(req.input, voice, float(req.speed or 1.0))
    except Exception as exc:
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    data, media_type = _encode(audio, req.response_format or "mp3")
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
    return {"status": "ok", "model_loaded": _pipeline is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kokoro TTS Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8880)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
