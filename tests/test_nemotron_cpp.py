"""Native Nemotron launcher behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from local_voice_ai.services.nemotron_cpp.launcher import (
    artifact_for_language,
    ensure_model,
    server_argv,
)


def test_model_download_is_atomic_and_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    source.write_bytes(b"small test model")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "cache" / "model.gguf"

    result = ensure_model(
        url=source.as_uri(),
        expected_sha256=expected,
        destination=destination,
    )

    assert result == destination
    assert destination.read_bytes() == b"small test model"
    assert not destination.with_suffix(".gguf.part").exists()


def test_verified_cached_model_does_not_touch_source(tmp_path: Path) -> None:
    destination = tmp_path / "model.gguf"
    destination.write_bytes(b"cached")
    expected = hashlib.sha256(destination.read_bytes()).hexdigest()

    assert (
        ensure_model(
            url="file:///does/not/exist",
            expected_sha256=expected,
            destination=destination,
        )
        == destination
    )


def test_server_uses_selected_backend_and_low_latency_right_context(tmp_path: Path) -> None:
    model = tmp_path / "nemotron.gguf"

    argv = server_argv(
        model,
        host="127.0.0.1",
        port=8000,
        right_context=1,
        gpu=-1,
        binary="/opt/nemo-speech/bin/nemo-speech",
    )

    assert argv[:2] == ["/opt/nemo-speech/bin/nemo-speech", "serve"]
    assert argv[argv.index("--asr-model") + 1] == str(model)
    assert argv[argv.index("--asr.backend.gpu") + 1] == "-1"
    assert argv[argv.index("--asr.streaming.rnnt_right_context") + 1] == "1"


def test_english_uses_the_accuracy_optimized_model() -> None:
    assert "streaming-en" in artifact_for_language("en-US").filename


@pytest.mark.parametrize("language", ["auto", "fr-FR", "ja-JP"])
def test_non_english_uses_multilingual_nemotron_35(language: str) -> None:
    artifact = artifact_for_language(language)

    assert artifact.filename == "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
    assert "nemotron-3.5-asr-streaming-0.6b" in artifact.url
