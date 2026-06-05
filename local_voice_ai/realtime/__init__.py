"""Local audio-in LLM integration for LiveKit Agents (half-cascade).

Exposes :class:`AudioLLM`, a ``livekit.agents.llm.RealtimeModel`` that feeds the
user's microphone audio directly into an audio-capable LLM served by an
OpenAI-compatible endpoint (llama.cpp ``llama-server`` with a multimodal
projector), returning text. Paired with a separate TTS (Kokoro) this forms the
LiveKit "half-cascade" pipeline and removes the dedicated STT stage.
"""

from .audio_llm import AudioLLM

__all__ = ["AudioLLM"]
