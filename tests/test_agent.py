"""Voice-agent configuration that must hold across LiveKit releases."""

from __future__ import annotations

from local_voice_ai import agent


def test_low_memory_mode_uses_explicit_vad_turn_detection(monkeypatch) -> None:
    monkeypatch.setattr(agent, "MultilingualModel", None)

    assert agent._turn_detection_mode() == "vad"


def test_multilingual_mode_constructs_the_turn_detector(monkeypatch) -> None:
    detector = object()
    monkeypatch.setattr(agent, "MultilingualModel", lambda: detector)

    assert agent._turn_detection_mode() is detector


def test_native_nemotron_receives_configured_language(monkeypatch) -> None:
    monkeypatch.setenv("STT_LANGUAGE", "fr-FR")

    assert agent._stt_language() == "fr-FR"
