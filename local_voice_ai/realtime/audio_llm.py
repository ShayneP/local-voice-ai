"""``AudioLLM`` — a LiveKit ``RealtimeModel`` backed by an audio-capable LLM.

Design (half-cascade, text-out):
  * The user's mic audio arrives frame-by-frame via ``push_audio``. We resample
    each frame to 16 kHz mono and accumulate it.
  * The LiveKit ``AgentSession`` (with a Silero VAD and ``turn_detection`` left to
    the framework) detects end-of-turn and calls ``commit_audio`` then
    ``generate_reply``. We snapshot the accumulated PCM into a single WAV.
  * ``generate_reply`` returns a Future immediately and, in the background, POSTs
    the WAV as an OpenAI ``input_audio`` content part to ``llama-server``'s
    ``/v1/chat/completions`` (``stream=True``), forwarding ``content`` deltas into
    a text stream. ``reasoning_content`` deltas are ignored (Gemma-4 "thinking"
    is disabled server-side with ``--reasoning-budget 0``).
  * Because ``capabilities.audio_output`` is False and a ``tts=`` is configured,
    the framework automatically routes the text stream to TTS (Kokoro).

This is a *batch-per-turn pseudo-stream*: llama.cpp has no streaming audio input
(it processes whole utterances in ~30 s chunks), so there are no interim
transcripts and the request is sent once the turn completes.

User transcripts & memory: each turn is transcribed FIRST (by the same model),
then the reply request sends that transcript as text *alongside* the audio. The
text is needed because Gemma-4 misreads self-referential audio questions ("what is
my name?") from audio alone. The transcript also drives UI captions
(``input_audio_transcription_completed``) and is appended, with the reply, to a
session-owned history that is replayed on later turns for multi-turn context.

Tradeoff: transcribing before the reply adds ~0.5s to first-token latency.

Typed text input is also supported: when the user types (no audio), the framework
appends the message to our chat_ctx via ``update_chat_ctx`` and calls
``generate_reply`` with no pending audio; we then send a plain text chat-completion
built from history.

Known limitations (prototype):
  * Transcripts are whole-utterance (post-VAD), not interim/streaming.
  * Tool/function calling is not wired (audio tool-calling is immature).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import wave
from dataclasses import dataclass

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm import (
    ChatContext,
    GenerationCreatedEvent,
    MessageGeneration,
    RealtimeCapabilities,
    ToolContext,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

logger = logging.getLogger("audio-llm")

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
# llama.cpp / libmtmd processes audio in fixed ~30 s chunks; cap the buffer so a
# long turn (or accumulated silence) can't blow past that or grow unbounded.
MAX_UTTERANCE_SECONDS = 30
_MAX_PCM_BYTES = MAX_UTTERANCE_SECONDS * TARGET_SAMPLE_RATE * 2  # int16 mono

_TRANSCRIBE_PROMPT = (
    "Transcribe the user's speech verbatim. Reply with only the exact words you "
    "hear, with no quotes, labels, or commentary."
)

_MAX_HISTORY_MESSAGES = 20  # cap replayed conversation turns


@dataclass
class _Opts:
    base_url: str
    model: str
    api_key: str
    instructions: str
    temperature: float
    max_tokens: int


class AudioLLM(llm.RealtimeModel):
    """RealtimeModel that sends each user turn as audio to an OpenAI-compatible
    audio LLM and returns text (half-cascade; pair with a ``tts=``)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "no-key-needed",
        instructions: str = "",
        temperature: float = 0.4,
        max_tokens: int = 512,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,  # use the AgentSession's Silero VAD
                user_transcription=True,  # surfaced via a parallel transcription call
                auto_tool_reply_generation=False,
                audio_output=False,  # text out -> framework routes to TTS
                manual_function_calls=False,
                mutable_chat_context=True,
                mutable_instructions=True,
                mutable_tools=False,
                per_response_tool_choice=False,
                supports_say=False,
            )
        )
        self._opts = _Opts(
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key,
            instructions=instructions,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._http_session = http_session
        self._owns_session = http_session is None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "llama-audio"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    def session(self) -> "AudioLLMSession":
        return AudioLLMSession(self)

    async def aclose(self) -> None:
        if self._owns_session and self._http_session is not None:
            await self._http_session.close()
            self._http_session = None


class AudioLLMSession(llm.RealtimeSession):
    def __init__(self, realtime_model: AudioLLM) -> None:
        super().__init__(realtime_model)
        self._model = realtime_model
        self._opts = realtime_model._opts

        self._instructions = self._opts.instructions

        self._resampler: rtc.AudioResampler | None = None
        self._resampler_in_rate: int | None = None

        self._pcm = bytearray()  # accumulated 16 kHz mono int16 for the live turn
        self._pending_wav: bytes | None = None  # snapshot taken at commit_audio

        self._gen_task: asyncio.Task | None = None
        # Session-owned conversation history (OpenAI message dicts). We maintain
        # this ourselves rather than relying on the framework to re-sync chat_ctx
        # into the session before each turn: every generate_reply is a stateless
        # HTTP call, so the model only remembers what we replay here.
        self._history: list[dict] = []

    # --- context / config ------------------------------------------------
    @property
    def chat_ctx(self) -> ChatContext:
        # Expose our session-owned history so the framework (and the text-input
        # path) sees the full conversation.
        ctx = ChatContext.empty()
        for m in self._history:
            ctx.add_message(role=m["role"], content=m["content"])
        return ctx

    @property
    def tools(self) -> ToolContext:
        return ToolContext.empty()

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        # Rebuild history from the framework's view. This is how typed text input
        # enters the session: the framework appends the user's message to our
        # chat_ctx and calls this immediately before generate_reply().
        history: list[dict] = []
        for item in chat_ctx.items:
            if getattr(item, "type", None) != "message" or item.role not in ("user", "assistant"):
                continue
            text = item.text_content
            if text:
                history.append({"role": item.role, "content": text})
        self._history = history

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        pass  # tool calling not wired for the audio path

    def update_options(self, *, tool_choice: NotGivenOr = NOT_GIVEN) -> None:
        pass

    # --- audio input -----------------------------------------------------
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self._pcm.extend(self._to_mono16k(frame))
        if len(self._pcm) > _MAX_PCM_BYTES:
            del self._pcm[: len(self._pcm) - _MAX_PCM_BYTES]

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def commit_audio(self) -> None:
        # Drain any samples the streaming resampler is still holding, then start
        # the next turn with a fresh resampler.
        if self._resampler is not None and hasattr(self._resampler, "flush"):
            for f in self._resampler.flush():
                self._pcm.extend(f.data.tobytes())
            self._resampler = None
            self._resampler_in_rate = None
        if not self._pcm:
            self._pending_wav = None
            return
        self._pending_wav = _pcm_to_wav(bytes(self._pcm))
        self._pcm.clear()

    def clear_audio(self) -> None:
        self._pcm.clear()
        self._pending_wav = None

    def _to_mono16k(self, frame: rtc.AudioFrame) -> bytes:
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        if frame.sample_rate == TARGET_SAMPLE_RATE:
            return pcm.tobytes()
        if self._resampler is None or self._resampler_in_rate != frame.sample_rate:
            self._resampler = rtc.AudioResampler(
                frame.sample_rate, TARGET_SAMPLE_RATE, num_channels=TARGET_CHANNELS
            )
            self._resampler_in_rate = frame.sample_rate
        mono = rtc.AudioFrame(
            data=pcm.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=TARGET_CHANNELS,
            samples_per_channel=len(pcm),
        )
        return b"".join(f.data.tobytes() for f in self._resampler.push(mono))

    # --- generation ------------------------------------------------------
    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr = NOT_GIVEN,
        tools: NotGivenOr = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.Future()
        sys_override = instructions if utils.is_given(instructions) else None
        self._gen_task = asyncio.create_task(self._run_generation(fut, sys_override))
        return fut

    async def _run_generation(
        self, fut: asyncio.Future[GenerationCreatedEvent], sys_override: str | None
    ) -> None:
        text_ch = utils.aio.Chan[str]()
        msg_ch = utils.aio.Chan[MessageGeneration]()
        fn_ch = utils.aio.Chan[llm.FunctionCall]()
        rid = utils.shortuuid("audio-llm-")

        modalities: asyncio.Future[list] = asyncio.Future()
        modalities.set_result(["text"])
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        audio_ch.close()  # text-only: no audio frames

        msg_ch.send_nowait(
            MessageGeneration(
                message_id=rid,
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        msg_ch.close()  # exactly one message per response

        gen_ev = GenerationCreatedEvent(
            message_stream=msg_ch,
            function_stream=fn_ch,
            user_initiated=True,
            response_id=rid,
        )
        if not fut.done():
            fut.set_result(gen_ev)
        self.emit("generation_created", gen_ev)
        fn_ch.close()  # no tool calls on the audio path

        wav = self._pending_wav
        self._pending_wav = None
        reply_text = ""
        transcript = ""
        try:
            if wav:
                # Audio turn. Transcribe FIRST: Gemma-4 misreads self-referential
                # audio questions ("what is my name?") from audio alone, so we send
                # the transcript as text alongside the audio. The transcript also
                # drives UI captions and history. Costs ~0.5s before the reply.
                transcript = await self._transcribe(wav)
                if transcript:
                    self.emit(
                        "input_audio_transcription_completed",
                        llm.InputTranscriptionCompleted(
                            item_id=utils.shortuuid("user-"),
                            transcript=transcript,
                            is_final=True,
                        ),
                    )
                messages = self._build_audio_messages(wav, transcript, sys_override)
            else:
                # Typed text turn: the framework appended the user's message to our
                # history via update_chat_ctx just before calling generate_reply.
                if not self._history:
                    logger.warning("generate_reply with neither audio nor text; empty reply")
                    return
                messages = self._build_text_messages(sys_override)
            reply_text = await self._stream_completion(messages, text_ch)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("audio LLM generation failed")
            self.emit(
                "error",
                llm.RealtimeModelError(
                    timestamp=time.time(),
                    label=self._model.label,
                    error=exc,
                    recoverable=True,
                ),
            )
        finally:
            text_ch.close()
        # Record the turn. Audio turns: the user transcript isn't in history yet, so
        # append it. Text turns: the user message is already in history (added via
        # update_chat_ctx), so append only the reply. Skipped on interrupt.
        if wav and transcript:
            self._history.append({"role": "user", "content": transcript})
        if reply_text:
            self._history.append({"role": "assistant", "content": reply_text})
        if len(self._history) > _MAX_HISTORY_MESSAGES:
            del self._history[: len(self._history) - _MAX_HISTORY_MESSAGES]

    async def _stream_completion(
        self, messages: list[dict], text_ch: utils.aio.Chan[str]
    ) -> str:
        payload = {
            "model": self._opts.model,
            "messages": messages,
            "stream": True,
            "temperature": self._opts.temperature,
            "max_tokens": self._opts.max_tokens,
        }
        parts: list[str] = []
        session = self._model._ensure_session()
        async with session.post(self._url, json=payload, headers=self._headers) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                content = delta.get("content")  # ignore reasoning_content
                if content:
                    text_ch.send_nowait(content)
                    parts.append(content)
        return "".join(parts)

    def _system_prefix(self, sys_override: str | None) -> list[dict]:
        system = sys_override or self._instructions
        return [{"role": "system", "content": system}] if system else []

    def _build_audio_messages(
        self, wav: bytes, transcript: str, sys_override: str | None
    ) -> list[dict]:
        messages = self._system_prefix(sys_override)
        messages.extend(self._history)
        # Current user turn: the audio plus its transcript as text. The text grounds
        # self-referential questions the model otherwise mishears from audio alone;
        # the audio preserves prosody/nuance for understanding.
        b64 = base64.b64encode(wav).decode("ascii")
        content: list[dict] = []
        if transcript:
            content.append({"type": "text", "text": transcript})
        content.append({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}})
        messages.append({"role": "user", "content": content})
        return messages

    def _build_text_messages(self, sys_override: str | None) -> list[dict]:
        # The typed user message is already the tail of self._history.
        return self._system_prefix(sys_override) + list(self._history)

    @property
    def _url(self) -> str:
        return f"{self._opts.base_url}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._opts.api_key}"}

    async def _transcribe(self, wav: bytes) -> str:
        b64 = base64.b64encode(wav).decode("ascii")
        payload = {
            "model": self._opts.model,
            "stream": False,
            "temperature": 0.0,
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _TRANSCRIBE_PROMPT},
                        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    ],
                }
            ],
        }
        session = self._model._ensure_session()
        async with session.post(self._url, json=payload, headers=self._headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()

    # --- lifecycle / interrupts -----------------------------------------
    def interrupt(self) -> None:
        if self._gen_task is not None and not self._gen_task.done():
            self._gen_task.cancel()

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list,
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass

    async def aclose(self) -> None:
        if self._gen_task is not None and not self._gen_task.done():
            self._gen_task.cancel()
            try:
                await self._gen_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(TARGET_CHANNELS)
        w.setsampwidth(2)  # int16
        w.setframerate(TARGET_SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()
