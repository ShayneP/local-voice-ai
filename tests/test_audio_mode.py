"""Tests for the audio-in half-cascade mode (LLM ingests audio, no STT)."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from livekit import rtc

from local_voice_ai.config import Config
from local_voice_ai.realtime import AudioLLM
from local_voice_ai.__main__ import _build_specs


class TestAudioModeConfig:
    def test_default_is_off(self) -> None:
        cfg = Config.from_env()
        assert cfg.llm_audio_input is False
        assert cfg.manage_stt is True

    def test_enabling_drops_stt_and_repoints_llama(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_AUDIO_INPUT", "1")
        cfg = Config.from_env()
        assert cfg.llm_audio_input is True
        # STT child is not managed in audio mode.
        assert cfg.manage_stt is False
        # The llama child serves the audio model, addressed by the audio alias.
        assert cfg.llama_hf_repo == "unsloth/gemma-4-12b-it-GGUF"
        assert cfg.llama_model == "gemma-4-audio"
        assert cfg.llama_model_alias == "gemma-4-audio"
        assert cfg.agent_env()["LLM_AUDIO_INPUT"] == "1"

    def test_generic_llama_envs_ignored_in_audio_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # docker-compose sets these for the text default; audio mode must ignore them.
        monkeypatch.setenv("LLM_AUDIO_INPUT", "1")
        monkeypatch.setenv("LLAMA_MODEL", "qwen3-4b")
        monkeypatch.setenv("LLAMA_MODEL_ALIAS", "qwen3-4b")
        monkeypatch.setenv("AUDIO_LLM_ALIAS", "my-audio")
        cfg = Config.from_env()
        assert cfg.llama_model == "my-audio"
        assert cfg.llama_model_alias == "my-audio"


class TestAudioModeSpecs:
    def test_text_mode_spawns_stt(self) -> None:
        cfg = Config.from_env()
        names = [s.name for s in _build_specs(cfg)]
        assert "nemotron" in names or "whisper" in names

    def test_audio_mode_no_stt_child(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_AUDIO_INPUT", "1")
        cfg = Config.from_env()
        names = [s.name for s in _build_specs(cfg)]
        assert "nemotron" not in names and "whisper" not in names
        assert {"livekit", "llama", "kokoro", "agent"} <= set(names)

    def test_audio_mode_llama_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_AUDIO_INPUT", "1")
        cfg = Config.from_env()
        llama = next(s for s in _build_specs(cfg) if s.name == "llama")
        argv = llama.argv
        assert "--jinja" in argv  # required by the Gemma-4 chat template
        assert "--reasoning-budget" in argv and argv[argv.index("--reasoning-budget") + 1] == "0"
        assert "-hf" in argv and "unsloth/gemma-4-12b-it-GGUF:Q4_K_M" in argv
        assert "--mmproj-url" in argv


class TestAudioLLMPlugin:
    def test_half_cascade_capabilities(self) -> None:
        m = AudioLLM(base_url="http://127.0.0.1:11434/v1", model="gemma-4-audio")
        caps = m.capabilities
        assert caps.audio_output is False  # text out -> framework routes to TTS
        assert caps.turn_detection is False  # rely on the AgentSession VAD
        assert caps.user_transcription is True  # surfaced via parallel transcription
        assert m.provider == "llama-audio"

    def test_push_and_commit_builds_16k_mono_wav(self) -> None:
        m = AudioLLM(base_url="http://127.0.0.1:11434/v1", model="gemma-4-audio")
        s = m.session()
        # 200 ms of 48 kHz mono input -> resampled to 16 kHz mono
        frame = rtc.AudioFrame(
            data=np.zeros(9600, dtype=np.int16).tobytes(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=9600,
        )
        s.push_audio(frame)
        s.commit_audio()
        assert s._pending_wav is not None
        w = wave.open(io.BytesIO(s._pending_wav))
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getnframes() > 0

    def test_build_messages_shape(self) -> None:
        m = AudioLLM(
            base_url="http://127.0.0.1:11434/v1",
            model="gemma-4-audio",
            instructions="be brief",
        )
        s = m.session()
        messages = s._build_audio_messages(b"RIFFfake", "hello there", sys_override=None)
        assert messages[0] == {"role": "system", "content": "be brief"}
        last = messages[-1]
        assert last["role"] == "user"
        types = [p["type"] for p in last["content"]]
        assert "text" in types and "input_audio" in types  # transcript grounds the turn
        audio_part = next(p for p in last["content"] if p["type"] == "input_audio")
        assert audio_part["input_audio"]["format"] == "wav"

    def test_history_replayed_in_messages(self) -> None:
        m = AudioLLM(base_url="http://127.0.0.1:11434/v1", model="gemma-4-audio")
        s = m.session()
        s._history = [
            {"role": "user", "content": "My name is Shane."},
            {"role": "assistant", "content": "Nice to meet you, Shane!"},
        ]
        messages = s._build_audio_messages(b"RIFFfake", "What is my name?", sys_override=None)
        # prior turns must appear before the current (audio) turn
        assert {"role": "user", "content": "My name is Shane."} in messages
        assert {"role": "assistant", "content": "Nice to meet you, Shane!"} in messages
        assert messages[-1]["content"][-1]["type"] == "input_audio"

    def test_text_input_enters_history_and_builds_text_request(self) -> None:
        import asyncio

        m = AudioLLM(base_url="http://127.0.0.1:11434/v1", model="gemma-4-audio", instructions="be brief")
        s = m.session()
        # Simulate the framework's typed-text path: it reads chat_ctx, appends the
        # user message, and calls update_chat_ctx before generate_reply.
        ctx = s.chat_ctx
        ctx.add_message(role="user", content="What is two plus three?")
        asyncio.run(s.update_chat_ctx(ctx))
        assert s._history[-1] == {"role": "user", "content": "What is two plus three?"}
        # A text turn (no audio) replays history as plain text — no input_audio parts.
        messages = s._build_text_messages(sys_override=None)
        assert messages[0]["role"] == "system"
        assert messages[-1] == {"role": "user", "content": "What is two plus three?"}
        assert all(isinstance(msg["content"], str) for msg in messages)
