"""Download the pinned Nemotron GGUF and replace this process with NeMo-Speech.cpp."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("nemotron-cpp")
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class ModelArtifact:
    filename: str
    url: str
    sha256: str


ENGLISH_MODEL = ModelArtifact(
    filename="nemotron-speech-streaming-en-0.6b.q8_0.gguf",
    url=(
        "https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/resolve/"
        "ebe59e5a817142986528bbbee5dba8db7b38ed50/"
        "nemotron-speech-streaming-en-0.6b.q8_0.gguf"
    ),
    sha256="d9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d",
)
MULTILINGUAL_MODEL = ModelArtifact(
    filename="nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
    url=(
        "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/"
        "1c8deaecc64b91f034d73e08dd8b64625eb3395d/"
        "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
    ),
    sha256="a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae",
)

# Backward-compatible names for callers that customize the download URL.
MODEL_FILENAME = ENGLISH_MODEL.filename
MODEL_URL = ENGLISH_MODEL.url
MODEL_SHA256 = ENGLISH_MODEL.sha256


def artifact_for_language(language: str, selection: str = "auto") -> ModelArtifact:
    """Select the English-specialized or multilingual streaming Q8 model."""
    selected = selection.strip().lower()
    if selected == "english":
        return ENGLISH_MODEL
    if selected == "multilingual":
        return MULTILINGUAL_MODEL
    if selected != "auto":
        raise ValueError("NEMOTRON_CPP_MODEL must be auto, english, or multilingual")
    language = language.strip().lower()
    return ENGLISH_MODEL if language == "en" or language.startswith("en-") else MULTILINGUAL_MODEL


def cache_dir() -> Path:
    base = os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "nemo-speech"


def model_path(filename: str = MODEL_FILENAME) -> Path:
    return cache_dir() / filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(
    *,
    url: str = MODEL_URL,
    expected_sha256: str = MODEL_SHA256,
    destination: Path | None = None,
) -> Path:
    """Return a verified local model, downloading it atomically when absent."""
    destination = destination or model_path()
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    logger.info("downloading %s", url)
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual_sha256 = _sha256(partial)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Nemotron model checksum mismatch: expected {expected_sha256}, "
                f"received {actual_sha256}"
            )
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    logger.info("saved %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
    return destination


def server_argv(
    model: Path,
    *,
    host: str,
    port: int,
    right_context: int,
    gpu: int,
    binary: str = "nemo-speech",
) -> list[str]:
    return [
        binary,
        "serve",
        "--asr-model",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
        "--no-ui",
        "--asr.backend.gpu",
        str(gpu),
        "--asr.streaming.rnnt_right_context",
        str(right_context),
    ]


def resolve_binary() -> str:
    """Find the nemo-speech binary, or explain how to install it.

    ``run.py`` installs the runtime under ``.local-voice-ai/runtime`` and hands
    the path down as NEMO_SPEECH_BIN. Running the supervisor directly skips
    that, and an unset variable used to reach execvp as a bare "nemo-speech"
    and die with a bare ENOENT, so look in the install directory too.
    """
    configured = os.getenv("NEMO_SPEECH_BIN")
    if configured:
        found = shutil.which(configured) or (configured if Path(configured).is_file() else None)
        if found is None:
            raise SystemExit(f"NEMO_SPEECH_BIN was set but not found: {configured}")
        return found

    on_path = shutil.which("nemo-speech")
    if on_path:
        return on_path

    installed = sorted(Path(".local-voice-ai/runtime").glob("*/bin/nemo-speech"))
    if installed:
        return str(installed[-1].resolve())

    raise SystemExit(
        "the native speech runtime (nemo-speech) is not installed. Run "
        "`python run.py start` to install it, or set NEMO_SPEECH_BIN to an "
        "existing binary. To use the Python speech service instead, set "
        "STT_PROVIDER=nemotron."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch native streaming Nemotron Speech")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--right-context",
        type=int,
        choices=(0, 1, 6, 13),
        default=int(os.getenv("NEMOTRON_CPP_RIGHT_CONTEXT", "1")),
        help="80 ms frames of right context; 1 gives about 160 ms model latency",
    )
    args = parser.parse_args(argv)

    language = os.getenv("STT_LANGUAGE", "en")
    artifact = artifact_for_language(
        language,
        os.getenv("NEMOTRON_CPP_MODEL", "auto"),
    )

    model = ensure_model(
        url=os.getenv("NEMOTRON_CPP_MODEL_URL", artifact.url),
        expected_sha256=os.getenv("NEMOTRON_CPP_MODEL_SHA256", artifact.sha256),
        destination=model_path(artifact.filename),
    )
    device = os.getenv("STT_DEVICE") or os.getenv("DEVICE", "cpu")
    configured_gpu = os.getenv("NEMOTRON_CPP_GPU")
    gpu = int(configured_gpu) if configured_gpu else (-1 if device == "cpu" else 0)
    command = server_argv(
        model,
        host=args.host,
        port=args.port,
        right_context=args.right_context,
        gpu=gpu,
        binary=resolve_binary(),
    )
    logger.info("starting NeMo-Speech.cpp")
    os.execvp(command[0], command)
    return 0  # pragma: no cover - execvp either replaces the process or raises


if __name__ == "__main__":
    raise SystemExit(main())
