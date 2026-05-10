"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest


_OUR_ENV_PREFIXES = (
    "LIVEKIT_", "LLAMA_", "STT_", "TTS_", "MANAGE_", "WEB_",
    "DEVICE", "NEMOTRON_", "VOXBOX_", "FRONTEND_DIR", "KOKORO_", "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe project-owned env vars before each test so defaults are predictable."""
    for key in list(os.environ):
        if key.startswith(_OUR_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
