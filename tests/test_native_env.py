"""Tests for the settings the native ``serve`` path has to supply itself.

``run.py`` prepares an environment before starting the stack: it resolves a
hardware profile and installs the native speech runtime. Running
``python -m local_voice_ai serve`` directly has no such caller, and each thing
it failed to carry over broke differently — a profile's context size silently
reverting to the default, and a missing speech binary dying on a bare ENOENT.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from local_voice_ai import __main__ as entry
from local_voice_ai.profiles import HardwareInfo, UserSelection, save_selection
from local_voice_ai.services.nemotron_cpp.launcher import resolve_binary

_CUDA = HardwareInfo(
    system="Linux",
    machine="x86_64",
    platform_key="nvidia-cuda",
    device_name="NVIDIA GPU",
    accelerator="cuda",
    memory_topology="discrete",
    total_memory_gib=32.0,
    accelerator_memory_gib=12.0,
)


@pytest.fixture
def saved_balanced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A saved 'balanced' selection, with hardware detection pinned."""
    path = tmp_path / ".local-voice-ai.toml"
    save_selection(
        path,
        UserSelection(profile="balanced", detected_platform="nvidia-cuda"),
    )
    monkeypatch.setattr(entry, "detect_hardware", lambda: _CUDA)
    return path


class TestProfileDefaults:
    def test_applies_the_profiles_context_size(
        self, saved_balanced: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLAMA_CTX_SIZE", raising=False)

        entry._apply_profile_defaults(saved_balanced)

        # 'balanced' is the larger-context profile; the Config default is 4096.
        assert os.environ["LLAMA_CTX_SIZE"] == "16384"

    def test_does_not_override_an_existing_value(
        self, saved_balanced: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLAMA_CTX_SIZE", "2048")

        entry._apply_profile_defaults(saved_balanced)

        assert os.environ["LLAMA_CTX_SIZE"] == "2048"

    def test_missing_selection_is_not_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLAMA_CTX_SIZE", raising=False)

        entry._apply_profile_defaults(tmp_path / "absent.toml")

        assert "LLAMA_CTX_SIZE" not in os.environ

    def test_unreadable_selection_is_not_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broken = tmp_path / "broken.toml"
        broken.write_text("this is not a selection\n", encoding="utf-8")
        monkeypatch.setattr(entry, "detect_hardware", lambda: _CUDA)

        entry._apply_profile_defaults(broken)  # warns, does not raise


class TestResolveBinary:
    def test_prefers_the_configured_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binary = tmp_path / "nemo-speech"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
        monkeypatch.setenv("NEMO_SPEECH_BIN", str(binary))

        assert resolve_binary() == str(binary)

    def test_configured_but_missing_is_a_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_SPEECH_BIN", "/nope/nemo-speech")

        with pytest.raises(SystemExit, match="NEMO_SPEECH_BIN"):
            resolve_binary()

    def test_falls_back_to_the_installed_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # What `run.py start` leaves behind, which the native path must find.
        installed = tmp_path / ".local-voice-ai/runtime/nemo-speech-0.1.0-linux-x86_64-cuda/bin"
        installed.mkdir(parents=True)
        binary = installed / "nemo-speech"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
        monkeypatch.delenv("NEMO_SPEECH_BIN", raising=False)
        monkeypatch.setattr("shutil.which", lambda _name: None)
        monkeypatch.chdir(tmp_path)

        assert resolve_binary() == str(binary.resolve())

    def test_nothing_installed_explains_how_to_install(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEMO_SPEECH_BIN", raising=False)
        monkeypatch.setattr("shutil.which", lambda _name: None)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit, match=re.escape("run.py start")):
            resolve_binary()
