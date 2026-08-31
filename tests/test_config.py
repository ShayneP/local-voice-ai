"""Tests for ``local_voice_ai.config.Config``.

The most important invariant: ``manage_X`` defaults to True when the matching
base URL points at the local machine, and False when it points elsewhere. An
explicit ``MANAGE_X`` env var overrides the auto-detected value either way.
"""

from __future__ import annotations

import pytest

from local_voice_ai.config import Config, _is_loopback


class TestIsLoopback:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:7880",
            "http://localhost:8000",
            "ws://0.0.0.0:1234",
            "http://[::1]:5000",
            "ws://127.0.0.1",
        ],
    )
    def test_loopback_urls(self, url: str) -> None:
        assert _is_loopback(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.openai.com/v1",
            "wss://my-project.livekit.cloud",
            "http://192.168.1.5:8000",
            "http://nemotron:8000/v1",  # docker service name → not loopback
        ],
    )
    def test_external_urls(self, url: str) -> None:
        assert _is_loopback(url) is False

    def test_malformed_url(self) -> None:
        # urlparse tolerates almost anything; the function must not raise
        assert _is_loopback("not a url") in (True, False)


class TestManageDefaults:
    def test_compact_context_is_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLAMA_CTX_SIZE", raising=False)

        assert Config.from_env().llama_ctx_size == 4096

    def test_llama_parallelism_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLAMA_PARALLEL", "1")
        assert Config.from_env().llama_parallel == 1

    def test_all_loopback_defaults_to_managed(self) -> None:
        cfg = Config.from_env()
        assert cfg.manage_livekit
        assert cfg.manage_llama
        assert cfg.manage_stt
        assert cfg.manage_tts

    def test_sequential_startup_is_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SEQUENTIAL_STARTUP", raising=False)
        assert Config.from_env().sequential_startup is False

        monkeypatch.setenv("SEQUENTIAL_STARTUP", "1")
        assert Config.from_env().sequential_startup is True

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

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1", True),
            ("true", True),
            ("YES", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("", False),
        ],
    )
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

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1", True),
            ("true", True),
            ("YES", True),
            ("0", False),
            ("false", False),
            ("no", False),
        ],
    )
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
        assert cfg.stt_provider == "nemotron-cpp"
        assert cfg.stt_model == "nemotron-speech-streaming"
        assert cfg.stt_language == "en"

    def test_whisper_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STT_PROVIDER", "whisper")
        cfg = Config.from_env()
        assert cfg.stt_provider == "whisper"
        assert cfg.stt_model == "Systran/faster-whisper-small"

    def test_explicit_stt_model_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STT_MODEL", "custom-model")
        cfg = Config.from_env()
        assert cfg.stt_model == "custom-model"

    def test_language_reaches_the_streaming_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STT_LANGUAGE", "fr-FR")

        cfg = Config.from_env()

        assert cfg.stt_language == "fr-FR"
        assert cfg.agent_env()["STT_LANGUAGE"] == "fr-FR"


class TestLowMemoryRuntimeOptions:
    def test_tts_provider_defaults_to_torch_kokoro(self) -> None:
        assert Config.from_env().tts_provider == "kokoro"

    def test_empty_idle_process_setting_preserves_livekit_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENT_IDLE_PROCESSES", "")

        assert Config.from_env().agent_idle_processes is None

    def test_low_memory_options_reach_children(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TTS_PROVIDER", "kokoro-onnx")
        monkeypatch.setenv("TURN_DETECTION", "vad")
        monkeypatch.setenv("AGENT_IDLE_PROCESSES", "1")

        cfg = Config.from_env()

        assert cfg.tts_provider == "kokoro-onnx"
        assert cfg.turn_detection == "vad"
        assert cfg.agent_idle_processes == 1
        assert cfg.agent_env()["TURN_DETECTION"] == "vad"
        assert cfg.agent_env()["AGENT_IDLE_PROCESSES"] == "1"

    def test_stt_device_can_differ_from_the_llm_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEVICE", "cuda")
        monkeypatch.setenv("STT_DEVICE", "cpu")

        cfg = Config.from_env()

        assert cfg.device == "cuda"
        assert cfg.stt_device == "cpu"


class TestAgentEnv:
    def test_agent_env_carries_all_provider_urls(self) -> None:
        cfg = Config.from_env()
        env = cfg.agent_env()
        for required in (
            "LIVEKIT_URL",
            "LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET",
            "LLAMA_BASE_URL",
            "LLAMA_MODEL",
            "LLAMA_API_KEY",
            "STT_BASE_URL",
            "STT_MODEL",
            "STT_API_KEY",
            "STT_PROVIDER",
            "TTS_BASE_URL",
            "TTS_PROVIDER",
            "TTS_VOICE",
            "TTS_API_KEY",
            "TURN_DETECTION",
        ):
            assert required in env, f"agent_env missing {required}"


class TestAdvertisedLiveKitUrl:
    """LIVEKIT_PUBLIC_URL separates what browsers are told from what we manage.

    Before it existed, pointing LIVEKIT_URL at a LAN address to reach remote
    clients also turned the managed server off, and the stack reported itself
    ready with nothing listening.
    """

    def test_defaults_to_the_internal_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LIVEKIT_PUBLIC_URL", raising=False)
        cfg = Config.from_env()
        assert cfg.advertised_livekit_url == cfg.livekit_url

    def test_public_url_does_not_disable_management(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIVEKIT_URL", "ws://127.0.0.1:7880")
        monkeypatch.setenv("LIVEKIT_PUBLIC_URL", "ws://192.168.1.40:7880")
        cfg = Config.from_env()
        assert cfg.advertised_livekit_url == "ws://192.168.1.40:7880"
        assert cfg.manage_livekit is True

    def test_node_ip_follows_the_public_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LIVEKIT_NODE_IP", raising=False)
        monkeypatch.setenv("LIVEKIT_PUBLIC_URL", "ws://192.168.1.40:7880")
        assert Config.from_env().livekit_node_ip == "192.168.1.40"

    def test_explicit_node_ip_still_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIVEKIT_PUBLIC_URL", "ws://192.168.1.40:7880")
        monkeypatch.setenv("LIVEKIT_NODE_IP", "10.0.0.9")
        assert Config.from_env().livekit_node_ip == "10.0.0.9"

    def test_hostname_leaves_node_ip_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # --node-ip needs an address, so a DNS name must not be passed through.
        monkeypatch.delenv("LIVEKIT_NODE_IP", raising=False)
        monkeypatch.setenv("LIVEKIT_PUBLIC_URL", "ws://voice.example.com:7880")
        assert Config.from_env().livekit_node_ip == "127.0.0.1"
