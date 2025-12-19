import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)
from livekit.plugins import silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")

    session = AgentSession(
        stt=openai.STT(
            base_url="http://whisper:80/v1",
            model="Systran/faster-whisper-small",
            api_key="no-key-needed"
        ),
        llm=openai.LLM(
            base_url="http://ollama:11434/v1",
            model=ollama_model,
            api_key="no-key-needed"
        ),
        tts=openai.TTS(
            base_url="http://kokoro:8880/v1",
            model="kokoro",
            voice="af_nova",
            api_key="no-key-needed"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)
