"""Environment-driven configuration for the local-voice-ai supervisor.

A single ``Config`` object is constructed at startup from environment variables
and shared with every subsystem (supervisor, agent, FastAPI routes).

The "manage X" flags decide whether the supervisor will spawn a given service
as a child process. They default to ``True`` when the matching base URL is a
loopback address (or unset), and ``False`` otherwise — pointing any base URL
at a remote endpoint automatically disables the local child.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_loopback(url: str) -> bool:
    """Return True if ``url`` points at the local machine."""
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    return host in {"", "localhost", "127.0.0.1", "0.0.0.0", "::1"}


@dataclass
class Config:
    # --- Web (FastAPI in the supervisor process) -------------------------
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    frontend_dir: Optional[str] = None  # path to a Next.js static export dir

    # --- LiveKit ---------------------------------------------------------
    livekit_url: str = "ws://127.0.0.1:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"
    livekit_bind_port: int = 7880
    livekit_rtc_port: int = 7881  # WebRTC over TCP (ICE/TCP fallback)
    livekit_udp_port: int = 7882  # WebRTC over UDP (preferred media transport)
    # IP the managed dev server advertises in ICE candidates. 127.0.0.1 is
    # reachable both from a browser on the host (via Docker-published ports) and
    # from the in-container agent (via loopback). Override (LIVEKIT_NODE_IP) when
    # running the server on a remote host reached over the network.
    livekit_node_ip: str = "127.0.0.1"
    manage_livekit: bool = True

    # --- LLM (llama.cpp by default) -------------------------------------
    llama_base_url: str = "http://127.0.0.1:11434/v1"
    llama_model: str = "qwen3-4b"
    llama_api_key: str = "no-key-needed"
    llama_hf_repo: str = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    llama_model_alias: str = "qwen3-4b"
    llama_ctx_size: int = 16384
    llama_n_gpu_layers: int = 0
    llama_bind_port: int = 11434
    manage_llama: bool = True

    # --- STT (Nemotron by default) --------------------------------------
    stt_provider: str = "nemotron"  # "nemotron" | "whisper"
    stt_base_url: str = "http://127.0.0.1:8000/v1"
    stt_model: str = "nemotron-speech-streaming"
    stt_api_key: str = "no-key-needed"
    stt_bind_port: int = 8000
    manage_stt: bool = True

    # Nemotron-specific
    nemotron_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    nemotron_model_id: str = "nemotron-speech-streaming"

    # Whisper (vox-box) specific
    voxbox_hf_repo_id: str = "Systran/faster-whisper-small"
    voxbox_device: str = "cpu"

    # --- TTS (Kokoro) ---------------------------------------------------
    tts_base_url: str = "http://127.0.0.1:8880/v1"
    tts_voice: str = "af_nova"
    tts_api_key: str = "no-key-needed"
    tts_bind_port: int = 8880
    manage_tts: bool = True

    # --- Device ---------------------------------------------------------
    device: str = "cpu"  # cpu | cuda | mps

    # --- Misc -----------------------------------------------------------
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Build the config from ``os.environ`` with sane defaults."""
        livekit_url = os.getenv("LIVEKIT_URL", cls.livekit_url)
        llama_base_url = os.getenv("LLAMA_BASE_URL", cls.llama_base_url)
        stt_base_url = os.getenv("STT_BASE_URL")
        tts_base_url = os.getenv("TTS_BASE_URL", cls.tts_base_url)

        stt_provider = os.getenv("STT_PROVIDER", cls.stt_provider).lower()
        if stt_base_url is None:
            # Default STT URL depends on provider
            stt_base_url = (
                "http://127.0.0.1:8000/v1"
                if stt_provider != "whisper"
                else "http://127.0.0.1:8000/v1"
            )

        default_stt_model = (
            "Systran/faster-whisper-small"
            if stt_provider == "whisper"
            else "nemotron-speech-streaming"
        )

        return cls(
            web_host=os.getenv("WEB_HOST", cls.web_host),
            web_port=int(os.getenv("WEB_PORT", str(cls.web_port))),
            frontend_dir=os.getenv("FRONTEND_DIR"),
            #
            livekit_url=livekit_url,
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", cls.livekit_api_key),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", cls.livekit_api_secret),
            livekit_bind_port=int(os.getenv("LIVEKIT_BIND_PORT", str(cls.livekit_bind_port))),
            livekit_rtc_port=int(os.getenv("LIVEKIT_RTC_PORT", str(cls.livekit_rtc_port))),
            livekit_udp_port=int(os.getenv("LIVEKIT_UDP_PORT", str(cls.livekit_udp_port))),
            livekit_node_ip=os.getenv("LIVEKIT_NODE_IP", cls.livekit_node_ip),
            manage_livekit=_env_bool("MANAGE_LIVEKIT", _is_loopback(livekit_url)),
            #
            llama_base_url=llama_base_url,
            llama_model=os.getenv("LLAMA_MODEL", cls.llama_model),
            llama_api_key=os.getenv("LLAMA_API_KEY", cls.llama_api_key),
            llama_hf_repo=os.getenv("LLAMA_HF_REPO", cls.llama_hf_repo),
            llama_model_alias=os.getenv("LLAMA_MODEL_ALIAS", cls.llama_model_alias),
            llama_ctx_size=int(os.getenv("LLAMA_CTX_SIZE", str(cls.llama_ctx_size))),
            llama_n_gpu_layers=int(os.getenv("LLAMA_N_GPU_LAYERS", str(cls.llama_n_gpu_layers))),
            llama_bind_port=int(os.getenv("LLAMA_BIND_PORT", str(cls.llama_bind_port))),
            manage_llama=_env_bool("MANAGE_LLAMA", _is_loopback(llama_base_url)),
            #
            stt_provider=stt_provider,
            stt_base_url=stt_base_url,
            stt_model=os.getenv("STT_MODEL", default_stt_model),
            stt_api_key=os.getenv("STT_API_KEY", cls.stt_api_key),
            stt_bind_port=int(os.getenv("STT_BIND_PORT", str(cls.stt_bind_port))),
            manage_stt=_env_bool("MANAGE_STT", _is_loopback(stt_base_url)),
            nemotron_model_name=os.getenv("NEMOTRON_MODEL_NAME", cls.nemotron_model_name),
            nemotron_model_id=os.getenv("NEMOTRON_MODEL_ID", cls.nemotron_model_id),
            voxbox_hf_repo_id=os.getenv("VOXBOX_HF_REPO_ID", cls.voxbox_hf_repo_id),
            voxbox_device=os.getenv("VOXBOX_DEVICE", cls.voxbox_device),
            #
            tts_base_url=tts_base_url,
            tts_voice=os.getenv("TTS_VOICE", cls.tts_voice),
            tts_api_key=os.getenv("TTS_API_KEY", cls.tts_api_key),
            tts_bind_port=int(os.getenv("TTS_BIND_PORT", str(cls.tts_bind_port))),
            manage_tts=_env_bool("MANAGE_TTS", _is_loopback(tts_base_url)),
            #
            device=os.getenv("DEVICE", cls.device).lower(),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
        )

    def agent_env(self) -> dict[str, str]:
        """Environment variables to pass to the agent worker subprocess."""
        return {
            "LIVEKIT_URL": self.livekit_url,
            "LIVEKIT_API_KEY": self.livekit_api_key,
            "LIVEKIT_API_SECRET": self.livekit_api_secret,
            "LLAMA_BASE_URL": self.llama_base_url,
            "LLAMA_MODEL": self.llama_model,
            "LLAMA_API_KEY": self.llama_api_key,
            "STT_PROVIDER": self.stt_provider,
            "STT_BASE_URL": self.stt_base_url,
            "STT_MODEL": self.stt_model,
            "STT_API_KEY": self.stt_api_key,
            "TTS_BASE_URL": self.tts_base_url,
            "TTS_VOICE": self.tts_voice,
            "TTS_API_KEY": self.tts_api_key,
        }
