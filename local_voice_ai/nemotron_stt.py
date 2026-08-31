"""LiveKit streaming STT adapter for the local NeMo-Speech.cpp WebSocket."""

from __future__ import annotations

import asyncio
import json
import weakref

import aiohttp
import httpx
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    vad,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

SAMPLE_RATE = 16_000


class NemotronSTT(stt.STT):
    """Feed microphone frames directly to Nemotron's cache-aware recognizer."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "nemotron-speech-streaming",
        api_key: str = "no-key-needed",
        language: str = "en",
        endpointing_ms: int = 300,
        vad_model: vad.VAD | None = None,
        ws_session: aiohttp.ClientSession | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=True,
            )
        )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._language = LanguageCode(language)
        self._endpointing_ms = endpointing_ms
        self._vad_model = vad_model
        self._ws_session = ws_session
        self._owns_ws_session = ws_session is None
        self._http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_http_client = http_client is None
        self._streams: weakref.WeakSet[NemotronSpeechStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "nemo-speech.cpp"

    @property
    def realtime_url(self) -> str:
        scheme = "wss" if self._base_url.startswith("https://") else "ws"
        rest = self._base_url.split("://", 1)[-1]
        return f"{scheme}://{rest}/realtime"

    def session_update(self, language: LanguageCode | None = None) -> dict[str, object]:
        selected_language = language or self._language
        return {
            "type": "session.update",
            "session": {
                "sample_rate": SAMPLE_RATE,
                "language": selected_language.language,
                "automatic_punctuation": True,
                "endpointing_ms": self._endpointing_ms,
            },
        }

    def _ensure_ws_session(self) -> aiohttp.ClientSession:
        if self._ws_session is None:
            self._ws_session = aiohttp.ClientSession()
        return self._ws_session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        selected = LanguageCode(language) if is_given(language) else self._language
        wav = rtc.combine_audio_frames(buffer).to_wav_bytes()
        try:
            response = await self._http_client.post(
                f"{self._base_url}/audio/transcriptions",
                headers=self._headers(),
                data={"model": self._model, "language": selected.language},
                files={"file": ("audio.wav", wav, "audio/wav")},
                timeout=httpx.Timeout(30.0, connect=conn_options.timeout),
            )
            response.raise_for_status()
            payload = response.json()
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=response.headers.get("x-request-id", ""),
                alternatives=[
                    stt.SpeechData(text=str(payload.get("text", "")), language=selected)
                ],
            )
        except httpx.TimeoutException:
            raise APITimeoutError() from None
        except httpx.HTTPStatusError as error:
            raise APIStatusError(
                str(error),
                status_code=error.response.status_code,
                body=error.response.text,
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise APIConnectionError(str(error)) from error

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> NemotronSpeechStream:
        selected = LanguageCode(language) if is_given(language) else self._language
        stream = NemotronSpeechStream(
            recognizer=self,
            language=selected,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        await asyncio.gather(
            *(stream.aclose() for stream in list(self._streams)),
            return_exceptions=True,
        )
        if self._owns_ws_session and self._ws_session is not None:
            await self._ws_session.close()
        if self._owns_http_client:
            await self._http_client.aclose()


class NemotronSpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        recognizer: NemotronSTT,
        language: LanguageCode,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            stt=recognizer,
            conn_options=conn_options,
            sample_rate=SAMPLE_RATE,
        )
        self._recognizer = recognizer
        self._language = language
        self._current_text = ""
        self._in_speech = False
        self._audio_since_final = 0.0
        self._uncommitted_audio = 0.0
        self._commit_pending = False
        self._commit_lock = asyncio.Lock()

    async def _run(self) -> None:
        session = self._recognizer._ensure_ws_session()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(
                    self._recognizer.realtime_url,
                    headers=self._recognizer._headers(),
                ),
                timeout=self._conn_options.timeout,
            )
        except (aiohttp.ClientError, TimeoutError) as error:
            raise APIConnectionError(str(error)) from error

        input_done = asyncio.Event()
        vad_stream = (
            self._recognizer._vad_model.stream()
            if self._recognizer._vad_model is not None
            else None
        )
        await ws.send_json(self._recognizer.session_update(self._language))
        self._report_connection_acquired(0.0, False)

        async def commit_audio() -> None:
            async with self._commit_lock:
                if self._commit_pending or self._uncommitted_audio <= 0:
                    return
                await ws.send_json({"type": "input_audio_buffer.commit"})
                self._commit_pending = True
                self._uncommitted_audio = 0.0

        async def send_audio() -> None:
            try:
                async for item in self._input_ch:
                    if isinstance(item, rtc.AudioFrame):
                        self._audio_since_final += item.duration
                        self._uncommitted_audio += item.duration
                        await ws.send_bytes(item.data.tobytes())
                        if vad_stream is not None:
                            vad_stream.push_frame(item)
                    else:
                        await commit_audio()
            finally:
                if vad_stream is not None:
                    vad_stream.end_input()
                input_done.set()

        async def detect_end_of_speech() -> None:
            assert vad_stream is not None
            async for event in vad_stream:
                if event.type == vad.VADEventType.END_OF_SPEECH:
                    await commit_audio()

        async def receive_events() -> None:
            while True:
                message = await ws.receive()
                if message.type == aiohttp.WSMsgType.TEXT:
                    self._handle_event(json.loads(message.data))
                    continue
                if message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                }:
                    if input_done.is_set():
                        return
                    raise APIConnectionError("Nemotron realtime connection closed")
                if message.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(str(ws.exception()))

        tasks = [asyncio.create_task(send_audio()), asyncio.create_task(receive_events())]
        if vad_stream is not None:
            tasks.append(asyncio.create_task(detect_end_of_speech()))
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if vad_stream is not None:
                await vad_stream.aclose()
            await ws.close()

    def _handle_event(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        request_id = str(event.get("item_id", ""))
        if event_type == "conversation.item.input_audio_transcription.delta":
            delta = str(event.get("delta", ""))
            if not delta:
                return
            self._current_text += delta
            if not self._in_speech:
                self._event_ch.send_nowait(stt.SpeechEvent(stt.SpeechEventType.START_OF_SPEECH))
                self._in_speech = True
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[
                        stt.SpeechData(text=self._current_text, language=self._language)
                    ],
                )
            )
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            self._commit_pending = False
            transcript = str(event.get("transcript", "")).strip()
            if transcript:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=request_id,
                        alternatives=[stt.SpeechData(text=transcript, language=self._language)],
                    )
                )
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        request_id=request_id,
                        recognition_usage=stt.RecognitionUsage(
                            audio_duration=self._audio_since_final
                        ),
                    )
                )
            if self._in_speech:
                self._event_ch.send_nowait(stt.SpeechEvent(stt.SpeechEventType.END_OF_SPEECH))
            self._current_text = ""
            self._audio_since_final = 0.0
            self._in_speech = False
            return

        if event_type == "error":
            detail = event.get("error", event)
            raise APIStatusError(str(detail), status_code=500, body=event)
