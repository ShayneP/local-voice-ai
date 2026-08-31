"""Low-memory Kokoro ONNX server behavior."""

from __future__ import annotations

import numpy as np
import pytest

from local_voice_ai.services.kokoro_onnx import server


class _FakeKokoro:
    def create(
        self, text: str, *, voice: str, speed: float, lang: str
    ) -> tuple[np.ndarray, int]:
        assert text == "hello"
        assert voice == "af_nova"
        assert speed == 1.0
        assert lang == "en-us"
        return np.array([0.0, 0.25, -0.25], dtype=np.float32), 24_000

    async def create_stream(self, text: str, *, voice: str, speed: float, lang: str):
        audio, sample_rate = self.create(
            text,
            voice=voice,
            speed=speed,
            lang=lang,
        )
        yield audio[:2], sample_rate
        yield audio[2:], sample_rate


def test_uses_fp16_model() -> None:
    assert server.MODEL_URL.endswith("kokoro-v1.0.fp16.onnx")


def test_fetch_caches_download(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads = 0

    def download(url: str, destination) -> None:
        nonlocal downloads
        downloads += 1
        destination.write_bytes(b"model")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(server.urllib.request, "urlretrieve", download)

    first = server._fetch("https://example.test/model.onnx")
    second = server._fetch("https://example.test/model.onnx")

    assert first == second
    assert first.read_bytes() == b"model"
    assert downloads == 1


@pytest.mark.asyncio
async def test_speech_returns_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_kokoro", _FakeKokoro())

    response = await server.speech(
        server.SpeechRequest(
            input="hello",
            voice="af_nova",
            speed=1.0,
            response_format="wav",
        )
    )

    assert response.media_type == "audio/wav"
    assert response.body.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_speech_streams_raw_pcm_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_kokoro", _FakeKokoro())

    response = await server.speech(
        server.SpeechRequest(
            input="hello",
            voice="af_nova",
            speed=1.0,
            response_format="pcm",
        )
    )
    chunks = [chunk async for chunk in response.body_iterator]

    assert response.media_type == "audio/pcm"
    assert len(chunks) == 2
    assert b"".join(chunks) == server._pcm16(
        np.array([0.0, 0.25, -0.25], dtype=np.float32)
    )
