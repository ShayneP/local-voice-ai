"""Tests for ``local_voice_ai.config.Config``.

The most important invariant: ``manage_X`` defaults to True when the matching
base URL points at the local machine, and False when it points elsewhere. An
explicit ``MANAGE_X`` env var overrides the auto-detected value either way.
"""

from __future__ import annotations

import pytest

from local_voice_ai.config import Config, _is_loopback


class TestIsLoopback:
    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:7880",
        "http://localhost:8000",
        "ws://0.0.0.0:1234",
        "http://[::1]:5000",
        "ws://127.0.0.1",
    ])
    def test_loopback_urls(self, url: str) -> None:
        assert _is_loopback(url) is True

    @pytest.mark.parametrize("url", [
        "https://api.openai.com/v1",
        "wss://my-project.livekit.cloud",
        "http://192.168.1.5:8000",
        "http://nemotron:8000/v1",  # docker service name → not loopback
    ])
    def test_external_urls(self, url: str) -> None:
        assert _is_loopback(url) is False

    def test_malformed_url(self) -> None:
        # urlparse tolerates almost anything; the function must not raise
        assert _is_loopback("not a url") in (True, False)


class TestManageDefaults:
    def test_all_loopback_defaults_to_managed(self) -> None:
        cfg = Config.from_env()
        assert cfg.manage_livekit
        assert cfg.manage_llama
        assert cfg.manage_stt
        assert cfg.manage_tts

    def test_external_livekit_disables_management(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIVEKIT_URL", "wss://my-project.livekit.cloud")
        cfg = Config.from_env()
        assert cfg.manage_livekit is False
        assert cfg.manage_llama and cfg.manage_stt and cfg.manage_tts

    def test_external_llama_disables_management(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLAMA_BASE_URL", "https://api.openai.com/v1")
        cfg = Config.from_env()
        assert cfg.manage_llama is False
        assert cfg.manage_livekit and cfg.manage_stt and cfg.manage_tts

    def test_external_stt_disables_management(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STT_BASE_URL", "https://api.example.com/v1")
        cfg = Config.from_env()
        assert cfg.manage_stt is False

    def test_external_tts_disables_management(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TTS_BASE_URL", "https://api.example.com/v1")
        cfg = Config.from_env()
        assert cfg.manage_tts is False


class TestManageOverride:
    def test_force_disable_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MANAGE_LLAMA", "0")
        cfg = Config.from_env()
        assert cfg.manage_llama is False  # forced off even though URL is loopback

    def test_force_enable_via_env_overrides_external_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLAMA_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("MANAGE_LLAMA", "1")
        cfg = Config.from_env()
        assert cfg.manage_llama is True

    @pytest.mark.parametrize("raw,expected", [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("", False),
    ])
    def test_boolean_parsing(
        self, raw: str, expected: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MANAGE_TTS", raw)
        cfg = Config.from_env()
        assert cfg.manage_tts is expected


class TestLlamaOffline:
    """LLAMA_OFFLINE is tri-state: unset → None (auto-detect at spec build),
    otherwise an explicit bool that wins over auto-detection."""

    def test_unset_means_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLAMA_OFFLINE", raising=False)
        assert Config.from_env().llama_offline is None

    @pytest.mark.parametrize("raw,expected", [
        ("1", True), ("true", True), ("YES", True),
        ("0", False), ("false", False), ("no", False),
    ])
    def test_explicit_value(
        self, raw: str, expected: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLAMA_OFFLINE", raw)
        assert Config.from_env().llama_offline is expected

    def test_model_path_default_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLAMA_MODEL_PATH", raising=False)
        assert Config.from_env().llama_model_path == ""

    def test_model_path_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLAMA_MODEL_PATH", "/models/foo.gguf")
        assert Config.from_env().llama_model_path == "/models/foo.gguf"


class TestWakeWord:
    def test_disabled_by_default(self) -> None:
        cfg = Config.from_env()
        assert cfg.wake_word is False
        assert cfg.wake_word_threshold == 0.5

    def test_enabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WAKE_WORD", "1")
        monkeypatch.setenv("WAKE_WORD_THRESHOLD", "0.3")
        cfg = Config.from_env()
        assert cfg.wake_word is True
        assert cfg.wake_word_threshold == 0.3

    def test_passed_to_agent_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WAKE_WORD", "1")
        env = Config.from_env().agent_env()
        assert env["WAKE_WORD"] == "1"
        assert "WAKE_WORD_MODEL" in env and "WAKE_WORD_THRESHOLD" in env


class TestSttProviderDefaults:
    def test_nemotron_default_model(self) -> None:
        cfg = Config.from_env()
        assert cfg.stt_provider == "nemotron"
        assert cfg.stt_model == "nemotron-speech-streaming"

    def test_whisper_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STT_PROVIDER", "whisper")
        cfg = Config.from_env()
        assert cfg.stt_provider == "whisper"
        assert cfg.stt_model == "Systran/faster-whisper-small"

    def test_explicit_stt_model_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STT_MODEL", "custom-model")
        cfg = Config.from_env()
        assert cfg.stt_model == "custom-model"


class TestAgentEnv:
    def test_agent_env_carries_all_provider_urls(self) -> None:
        cfg = Config.from_env()
        env = cfg.agent_env()
        for required in (
            "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
            "LLAMA_BASE_URL", "LLAMA_MODEL", "LLAMA_API_KEY",
            "STT_BASE_URL", "STT_MODEL", "STT_API_KEY", "STT_PROVIDER",
            "TTS_BASE_URL", "TTS_VOICE", "TTS_API_KEY",
        ):
            assert required in env, f"agent_env missing {required}"
