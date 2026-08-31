"""LiveKit adapter for native streaming Nemotron."""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from livekit import rtc
from livekit.agents import vad
from livekit.agents.stt import SpeechEventType

from local_voice_ai.nemotron_stt import NemotronSTT


class _EndpointStream:
    def __init__(self) -> None:
        self._events: asyncio.Queue[vad.VADEvent | None] = asyncio.Queue()
        self._ended = False

    def push_frame(self, _frame: rtc.AudioFrame) -> None:
        if self._ended:
            return
        self._ended = True
        self._events.put_nowait(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=0,
                timestamp=0.0,
                speech_duration=0.05,
                silence_duration=0.3,
            )
        )

    def end_input(self) -> None:
        self._events.put_nowait(None)

    async def aclose(self) -> None:
        self._events.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self) -> vad.VADEvent:
        event = await self._events.get()
        if event is None:
            raise StopAsyncIteration
        return event


class _EndpointVAD:
    def stream(self) -> _EndpointStream:
        return _EndpointStream()


def test_native_adapter_advertises_real_streaming() -> None:
    recognizer = NemotronSTT(base_url="http://127.0.0.1:8000/v1")

    assert recognizer.capabilities.streaming is True
    assert recognizer.capabilities.interim_results is True
    assert recognizer.realtime_url == "ws://127.0.0.1:8000/v1/realtime"
    assert recognizer.session_update()["session"] == {
        "sample_rate": 16_000,
        "language": "en",
        "automatic_punctuation": True,
        "endpointing_ms": 300,
    }


@pytest.mark.asyncio
async def test_microphone_frames_produce_interim_and_final_transcripts() -> None:
    received: dict[str, object] = {}

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        received["update"] = await socket.receive_json()
        audio = await socket.receive()
        received["audio_type"] = audio.type
        received["audio_bytes"] = len(audio.data)
        commit = await socket.receive()
        received["commit"] = json.loads(commit.data)
        await socket.send_json(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "turn-1",
                "delta": "Hello",
            }
        )
        await socket.send_json(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "turn-1",
                "transcript": "Hello Jetson.",
            }
        )
        await socket.close()
        return socket

    app = web.Application()
    app.router.add_get("/v1/realtime", realtime)
    async with TestServer(app) as server:
        recognizer = NemotronSTT(base_url=str(server.make_url("/v1")))
        stream = recognizer.stream()
        frame = rtc.AudioFrame.create(sample_rate=16_000, num_channels=1, samples_per_channel=800)
        stream.push_frame(frame)
        stream.end_input()
        events = [event async for event in stream]
        await recognizer.aclose()

    assert received["update"] == recognizer.session_update()
    assert received["audio_type"].name == "BINARY"
    assert received["audio_bytes"] == 1600
    assert received["commit"] == {"type": "input_audio_buffer.commit"}
    assert [event.type for event in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.RECOGNITION_USAGE,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events[1].alternatives[0].text == "Hello"
    assert events[2].alternatives[0].text == "Hello Jetson."


@pytest.mark.asyncio
async def test_vad_endpoint_commits_a_live_stream_without_ending_input() -> None:
    committed = asyncio.Event()

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.receive_json()
        await socket.receive()
        commit = await socket.receive_json()
        assert commit == {"type": "input_audio_buffer.commit"}
        committed.set()
        await socket.send_json(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "turn-live",
                "delta": "Streaming",
            }
        )
        await socket.send_json(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "turn-live",
                "transcript": "Streaming works.",
            }
        )
        await asyncio.Future()

    app = web.Application()
    app.router.add_get("/v1/realtime", realtime)
    async with TestServer(app) as server:
        recognizer = NemotronSTT(
            base_url=str(server.make_url("/v1")),
            vad_model=_EndpointVAD(),  # type: ignore[arg-type]
        )
        stream = recognizer.stream()
        frame = rtc.AudioFrame.create(
            sample_rate=16_000,
            num_channels=1,
            samples_per_channel=800,
        )
        stream.push_frame(frame)
        events = [await asyncio.wait_for(anext(stream), timeout=2) for _ in range(5)]
        await asyncio.wait_for(committed.wait(), timeout=2)
        await recognizer.aclose()

    assert [event.type for event in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.RECOGNITION_USAGE,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events[2].alternatives[0].text == "Streaming works."
