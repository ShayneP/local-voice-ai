"""LiveKit Agents worker.

Moved verbatim from ``livekit_agent/src/agent.py``. The only change is that the
default base URLs are loopback (``127.0.0.1``) instead of Docker service names —
the supervisor spawns the inference children on loopback ports, so this is
correct for both single-image deployment and bare-metal local runs.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero

_TURN_DETECTION = os.getenv("TURN_DETECTION", "multilingual").strip().lower()
if _TURN_DETECTION == "multilingual":
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
else:
    MultilingualModel = None  # type: ignore[misc,assignment]

logger = logging.getLogger("agent")

load_dotenv(".env.local")

INSTRUCTIONS = (
    "You are a helpful voice AI assistant. The user is interacting with you via "
    "voice, even if you perceive the conversation as text. You eagerly assist "
    "users with their questions by providing information from your extensive "
    "knowledge. Your responses are concise, to the point, and without any "
    "complex formatting or punctuation including emojis, asterisks, or other "
    "symbols. You are curious, friendly, and have a sense of humor."
)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    @function_tool()
    async def multiply_numbers(
        self,
        context: RunContext,
        number1: int,
        number2: int,
    ) -> dict[str, Any]:
        """Multiply two numbers.

        Args:
            number1: The first number to multiply.
            number2: The second number to multiply.
        """
        return f"The product of {number1} and {number2} is {number1 * number2}."


_idle_processes = os.getenv("AGENT_IDLE_PROCESSES")
server = AgentServer(**({"num_idle_processes": int(_idle_processes)} if _idle_processes else {}))


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def _turn_detection_mode():
    # Passing None disables automatic turn completion in current LiveKit
    # Agents. The low-memory profile needs explicit VAD-only endpointing.
    return MultilingualModel() if MultilingualModel is not None else "vad"


def _stt_language() -> str:
    return os.getenv("STT_LANGUAGE", "en")


@server.rtc_session()
async def my_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    llama_model = os.getenv("LLAMA_MODEL", "gemma-4-e2b")
    llama_base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    llama_api_key = os.getenv("LLAMA_API_KEY", "no-key-needed")

    stt_provider = os.getenv("STT_PROVIDER", "nemotron-cpp").lower()
    if stt_provider == "whisper":
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "Systran/faster-whisper-small"
    else:
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "nemotron-speech-streaming"

    stt_base_url = os.getenv("STT_BASE_URL", default_stt_base_url)
    stt_model = os.getenv("STT_MODEL", default_stt_model)
    stt_api_key = os.getenv("STT_API_KEY", "no-key-needed")

    tts_base_url = os.getenv("TTS_BASE_URL", "http://127.0.0.1:8880/v1")
    tts_provider = os.getenv("TTS_PROVIDER", "kokoro").lower()
    tts_voice = os.getenv("TTS_VOICE", "af_nova")
    tts_api_key = os.getenv("TTS_API_KEY", "no-key-needed")

    logger.info(
        "agent session: stt=%s/%s llm=%s/%s tts=%s",
        stt_provider,
        stt_model,
        llama_base_url,
        llama_model,
        tts_base_url,
    )

    wake_word = os.getenv("WAKE_WORD", "").strip().lower() in {"1", "true", "yes", "on"}
    wake_word_model = os.getenv("WAKE_WORD_MODEL", "/app/models/wakeword/hey_livekit.onnx")
    wake_word_threshold = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))

    audio_input = os.getenv("LLM_AUDIO_INPUT", "").strip().lower() in {"1", "true", "yes", "on"}

    session_vad = ctx.proc.userdata["vad"]

    # Kokoro TTS is shared by both paths below. The model name selects the wire
    # protocol the openai TTS plugin uses: only {"tts-1", "tts-1-hd"} use the
    # raw-audio-bytes stream that the Kokoro server speaks. Any other name
    # (e.g. "kokoro") routes the plugin into the gpt-4o-mini-tts SSE reader,
    # which parses Kokoro's binary audio body as text, pushes zero frames, and
    # raises "no audio frames were pushed". Kokoro ignores the model field, so
    # "tts-1" is purely a protocol selector here.
    tts = openai.TTS(
        base_url=tts_base_url,
        model="tts-1",
        voice=tts_voice,
        api_key=tts_api_key,
        # Raw PCM lets the ONNX server deliver playable chunks without
        # buffering a complete WAV or MP3 response first.
        response_format="pcm" if tts_provider == "kokoro-onnx" else "mp3",
    )

    if audio_input:
        # Half-cascade: the user's mic audio goes straight to the audio-capable
        # LLM (llama-server + multimodal projector), which returns text; Kokoro
        # TTS speaks it. There is no STT, and the semantic turn detector is
        # replaced by plain VAD (the turn detector operates on STT text, which
        # we no longer produce).
        from .realtime import AudioLLM

        logger.info(
            "audio-in half-cascade: mic -> %s (%s) -> TTS, STT disabled",
            llama_base_url, llama_model,
        )
        session = AgentSession(
            llm=AudioLLM(
                base_url=llama_base_url,
                model=llama_model,
                api_key=llama_api_key,
                instructions=INSTRUCTIONS,
                temperature=0.4,
                max_tokens=512,
            ),
            tts=tts,
            vad=session_vad,
            turn_detection="vad",
        )
    else:
        if stt_provider == "nemotron-cpp":
            from .nemotron_stt import NemotronSTT

            stt = NemotronSTT(
                base_url=stt_base_url,
                model=stt_model,
                api_key=stt_api_key,
                language=_stt_language(),
                endpointing_ms=int(os.getenv("NEMOTRON_ENDPOINTING_MS", "300")),
                # NeMo-Speech.cpp requires the client to commit each utterance.
                # Reuse Silero's weights in a second stream so real microphone
                # noise does not make endpointing depend on a fragile RMS cutoff.
                vad_model=session_vad,
            )
        else:
            stt = openai.STT(base_url=stt_base_url, model=stt_model, api_key=stt_api_key)

        session = AgentSession(
            stt=stt,
            llm=openai.LLM(base_url=llama_base_url, model=llama_model, api_key=llama_api_key),
            tts=tts,
            turn_detection=_turn_detection_mode(),
            vad=session_vad,
            preemptive_generation=True,
        )

    await session.start(agent=Assistant(), room=ctx.room)
    await ctx.connect()

    if wake_word:
        # Join deaf, wait for the wake phrase, then wake up and greet.
        from .wakeword import wait_for_wake_word

        session.input.set_audio_enabled(False)
        participant = await ctx.wait_for_participant()
        try:
            await wait_for_wake_word(participant, wake_word_model, wake_word_threshold)
        except Exception:
            # Fail open: a broken detector shouldn't brick the assistant.
            logger.exception("wake word detection failed; enabling audio input")
        session.input.set_audio_enabled(True)
        session.generate_reply(
            instructions=(
                "You just woke up because the user said the wake phrase. "
                "Greet them very briefly and ask how you can help."
            )
        )
    else:
        # Speak first so the user knows the audio path works.
        session.generate_reply(
            instructions=(
                "Greet the user warmly in one short sentence and invite them to ask you anything."
            )
        )


if __name__ == "__main__":
    cli.run_app(server)
