"""Environment-driven configuration for the local-voice-ai supervisor.

A single ``Config`` object is constructed at startup from environment variables
and shared with every subsystem (supervisor, agent, FastAPI routes).

The "manage X" flags decide whether the supervisor will spawn a given service
as a child process. They default to ``True`` when the matching base URL is a
loopback address (or unset), and ``False`` otherwise — pointing any base URL
at a remote endpoint automatically disables the local child.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from urllib.parse import urlparse


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_bool_opt(name: str) -> bool | None:
    """Like ``_env_bool`` but returns ``None`` when the var is unset, so callers
    can distinguish "not configured" (auto) from an explicit true/false."""
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_opt(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw else None


def _env_csv(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(value.strip() for value in raw.split(",") if value.strip())


_DEFAULT_BIND_HOST = "127.0.0.1"

# Wildcard binds aren't connectable addresses — probe readiness over loopback.
_WILDCARD_HOSTS = {"0.0.0.0", "::", ""}


def probe_host(bind_host: str) -> str:
    """The address to reach a child that is listening on ``bind_host``."""
    return "127.0.0.1" if bind_host in _WILDCARD_HOSTS else bind_host


def _is_loopback(url: str) -> bool:
    """Return True if ``url`` points at the local machine."""
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    return host in {"", "localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _literal_ip(url: str) -> str:
    """The literal IP address in ``url``, or "" when it has none.

    LiveKit's ``--node-ip`` needs an address rather than a name, so a DNS
    hostname yields "" and leaves LIVEKIT_NODE_IP to supply the address.
    """
    if not url:
        return ""
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return ""
    return host


@dataclass
class Config:
    # --- Web (FastAPI in the supervisor process) -------------------------
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    frontend_dir: str | None = None  # path to a Next.js static export dir
    # Extra browser origins allowed to call the token and status APIs. Localhost
    # client apps are allowed separately by the API's narrow origin regex.
    client_origins: tuple[str, ...] = ()
    # Expose the OpenAI-compatible /v1/* surface on the web port, proxied to the
    # LLM/STT/TTS children, so one URL serves everything and the children can
    # stay on loopback. Off by default: web_host is already 0.0.0.0 and the
    # backends have no auth, so enabling this publishes unauthenticated
    # inference to whatever network the web port is reachable from.
    gateway: bool = False

    # --- LiveKit ---------------------------------------------------------
    # Where this process and the agent reach LiveKit. It also decides whether
    # the supervisor runs its own server (see manage_livekit), so it should
    # stay on loopback whenever the managed dev server is wanted.
    livekit_url: str = "ws://127.0.0.1:7880"
    # Where browsers are told to connect, when that differs — a LAN address for
    # remote clients, say. Empty means "same as livekit_url". Keeping this
    # separate matters: overloading livekit_url for the advertised address used
    # to switch the managed server off as a side effect.
    livekit_public_url: str = ""
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
    llama_model: str = "gemma-4-e2b"
    llama_api_key: str = "no-key-needed"
    # Quantization-aware-trained quant — holds up much better at 4-bit than
    # post-hoc quantization. The :tag suffix selects the quant within the repo.
    llama_hf_repo: str = "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL"
    # Path to a local .gguf. When set, llama-server loads it directly with -m
    # instead of resolving --hf-repo against Hugging Face (works fully offline).
    llama_model_path: str = ""
    # Pass --offline to llama-server: use only cached files, never hit the
    # network. Lets a previously-downloaded --hf-repo model start with no
    # internet. ``None`` (the default) means auto: enable --offline when the
    # model is already cached, otherwise allow the first-run download.
    # See https://github.com/ShayneP/local-voice-ai/issues/9
    llama_offline: bool | None = None
    llama_model_alias: str = "gemma-4-e2b"
    llama_ctx_size: int = 4096
    llama_parallel: int = 4
    llama_n_gpu_layers: int = 0
    llama_bind_port: int = 11434
    llama_bind_host: str = _DEFAULT_BIND_HOST
    manage_llama: bool = True

    # --- Audio-in half-cascade (LLM ingests audio directly; no STT) ------
    # When enabled, the local llama child serves an audio-capable GGUF (with a
    # multimodal projector) and the agent feeds mic audio straight to it via a
    # custom RealtimeModel; the dedicated STT child is not spawned. Output is
    # text, spoken by the normal TTS (Kokoro) — LiveKit's "half-cascade".
    # Requires a llama.cpp build >= b9518 for the Gemma-4 "gemma4uv" projector.
    llm_audio_input: bool = False
    audio_llm_hf_repo: str = "unsloth/gemma-4-12b-it-GGUF"
    audio_llm_quant: str = "Q4_K_M"
    audio_llm_alias: str = "gemma-4-audio"
    audio_llm_mmproj_url: str = (
        "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/mmproj-BF16.gguf"
    )

    # --- STT (native streaming Nemotron by default) ---------------------
    stt_provider: str = "nemotron-cpp"  # "nemotron-cpp" | "nemotron" | "whisper"
    stt_base_url: str = "http://127.0.0.1:8000/v1"
    stt_model: str = "nemotron-speech-streaming"
    stt_language: str = "en"
    stt_api_key: str = "no-key-needed"
    stt_bind_port: int = 8000
    stt_bind_host: str = _DEFAULT_BIND_HOST
    manage_stt: bool = True

    # Nemotron-specific
    nemotron_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    nemotron_model_id: str = "nemotron-speech-streaming"
    # fp16 halves the weights with no measured transcript change on GPU.
    nemotron_fp16: bool = True
    # Nemotron transcribes numbers verbatim ("four one five..."); normalise
    # them to digits the way whisper already does internally.
    nemotron_itn: bool = True

    # Whisper (faster-whisper) specific
    whisper_model: str = "Systran/faster-whisper-small"
    # Optional STT-only override. For example, Jetson runs Whisper on CPU while
    # llama.cpp continues to use CUDA through LLAMA_N_GPU_LAYERS.
    stt_device: str = ""

    # --- TTS (Kokoro) ---------------------------------------------------
    tts_provider: str = "kokoro"  # "kokoro" | "kokoro-onnx"
    tts_base_url: str = "http://127.0.0.1:8880/v1"
    tts_voice: str = "af_nova"
    tts_api_key: str = "no-key-needed"
    tts_bind_port: int = 8880
    tts_bind_host: str = _DEFAULT_BIND_HOST
    manage_tts: bool = True

    # --- Wake word (off by default) --------------------------------------
    # When enabled the agent joins deaf and only starts listening after the
    # wake phrase ("hey livekit") is detected on the user's microphone.
    wake_word: bool = False
    wake_word_model: str = "/app/models/wakeword/hey_livekit.onnx"
    wake_word_threshold: float = 0.5

    # --- Device ---------------------------------------------------------
    device: str = "cpu"  # cpu | cuda | mps

    # --- Misc -----------------------------------------------------------
    # The multilingual semantic turn detector is more accurate but needs
    # another model. Small systems can use VAD-only endpointing instead.
    turn_detection: str = "multilingual"  # "multilingual" | "vad"
    # None preserves LiveKit's environment-specific default.
    agent_idle_processes: int | None = None
    # Load managed children one at a time. This reduces temporary memory peaks
    # on small unified-memory devices such as Jetson Orin Nano.
    sequential_startup: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Config:
        """Build the config from ``os.environ`` with sane defaults."""
        # Managed inference children listen on loopback, so nothing is reachable
        # off-box by default. BIND_HOST widens all three at once (0.0.0.0 for the
        # whole LAN, or a specific NIC address to pick one interface); the
        # per-service *_BIND_HOST vars override it individually. These endpoints
        # have no auth — only widen this on a network you trust.
        bind_host = os.getenv("BIND_HOST", _DEFAULT_BIND_HOST)
        livekit_url = os.getenv("LIVEKIT_URL", cls.livekit_url)
        livekit_public_url = os.getenv("LIVEKIT_PUBLIC_URL", cls.livekit_public_url)
        llama_base_url = os.getenv("LLAMA_BASE_URL", cls.llama_base_url)
        stt_base_url = os.getenv("STT_BASE_URL")
        tts_base_url = os.getenv("TTS_BASE_URL", cls.tts_base_url)

        # Audio-in half-cascade. When on, the llama child serves the audio model
        # and STT is dropped; reuse the llama_* fields so the rest of the stack
        # (base URL, model name) is unchanged.
        llm_audio_input = _env_bool("LLM_AUDIO_INPUT", cls.llm_audio_input)
        audio_llm_hf_repo = os.getenv("AUDIO_LLM_HF_REPO", cls.audio_llm_hf_repo)
        audio_llm_quant = os.getenv("AUDIO_LLM_QUANT", cls.audio_llm_quant)
        audio_llm_alias = os.getenv("AUDIO_LLM_ALIAS", cls.audio_llm_alias)
        audio_llm_mmproj_url = os.getenv("AUDIO_LLM_MMPROJ_URL", cls.audio_llm_mmproj_url)
        if llm_audio_input:
            # Audio mode addresses the model by AUDIO_LLM_ALIAS and serves it from
            # AUDIO_LLM_HF_REPO; the generic LLAMA_MODEL/ALIAS/HF_REPO are ignored.
            llama_model = audio_llm_alias
            llama_model_alias = audio_llm_alias
            llama_hf_repo = audio_llm_hf_repo
        else:
            llama_model = os.getenv("LLAMA_MODEL", cls.llama_model)
            llama_model_alias = os.getenv("LLAMA_MODEL_ALIAS", cls.llama_model_alias)
            llama_hf_repo = os.getenv("LLAMA_HF_REPO", cls.llama_hf_repo)

        stt_provider = os.getenv("STT_PROVIDER", cls.stt_provider).lower()
        if stt_base_url is None:
            stt_base_url = "http://127.0.0.1:8000/v1"

        default_stt_model = (
            "Systran/faster-whisper-small"
            if stt_provider == "whisper"
            else "nemotron-speech-streaming"
        )

        return cls(
            web_host=os.getenv("WEB_HOST", cls.web_host),
            web_port=int(os.getenv("WEB_PORT", str(cls.web_port))),
            frontend_dir=os.getenv("FRONTEND_DIR"),
            client_origins=_env_csv("CLIENT_ORIGINS"),
            gateway=_env_bool("GATEWAY", cls.gateway),
            #
            livekit_url=livekit_url,
            livekit_public_url=livekit_public_url,
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", cls.livekit_api_key),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", cls.livekit_api_secret),
            livekit_bind_port=int(os.getenv("LIVEKIT_BIND_PORT", str(cls.livekit_bind_port))),
            livekit_rtc_port=int(os.getenv("LIVEKIT_RTC_PORT", str(cls.livekit_rtc_port))),
            livekit_udp_port=int(os.getenv("LIVEKIT_UDP_PORT", str(cls.livekit_udp_port))),
            # ICE candidates must carry an address the browser can reach, so
            # default it to the advertised host instead of making a LAN setup
            # remember a second variable.
            livekit_node_ip=os.getenv("LIVEKIT_NODE_IP")
            or _literal_ip(livekit_public_url or livekit_url)
            or cls.livekit_node_ip,
            manage_livekit=_env_bool("MANAGE_LIVEKIT", _is_loopback(livekit_url)),
            #
            llama_base_url=llama_base_url,
            llama_model=llama_model,
            llama_api_key=os.getenv("LLAMA_API_KEY", cls.llama_api_key),
            llama_hf_repo=llama_hf_repo,
            llama_model_path=os.getenv("LLAMA_MODEL_PATH", cls.llama_model_path),
            llama_offline=_env_bool_opt("LLAMA_OFFLINE"),
            llama_model_alias=llama_model_alias,
            llama_ctx_size=int(os.getenv("LLAMA_CTX_SIZE", str(cls.llama_ctx_size))),
            llama_parallel=int(os.getenv("LLAMA_PARALLEL", str(cls.llama_parallel))),
            llama_n_gpu_layers=int(os.getenv("LLAMA_N_GPU_LAYERS", str(cls.llama_n_gpu_layers))),
            llama_bind_port=int(os.getenv("LLAMA_BIND_PORT", str(cls.llama_bind_port))),
            llama_bind_host=os.getenv("LLAMA_BIND_HOST", bind_host),
            manage_llama=_env_bool("MANAGE_LLAMA", _is_loopback(llama_base_url)),
            llm_audio_input=llm_audio_input,
            audio_llm_hf_repo=audio_llm_hf_repo,
            audio_llm_quant=audio_llm_quant,
            audio_llm_alias=audio_llm_alias,
            audio_llm_mmproj_url=audio_llm_mmproj_url,
            #
            stt_provider=stt_provider,
            stt_base_url=stt_base_url,
            stt_model=os.getenv("STT_MODEL", default_stt_model),
            stt_language=os.getenv("STT_LANGUAGE", cls.stt_language),
            stt_api_key=os.getenv("STT_API_KEY", cls.stt_api_key),
            stt_bind_port=int(os.getenv("STT_BIND_PORT", str(cls.stt_bind_port))),
            stt_bind_host=os.getenv("STT_BIND_HOST", bind_host),
            manage_stt=_env_bool("MANAGE_STT", _is_loopback(stt_base_url) and not llm_audio_input),
            nemotron_model_name=os.getenv("NEMOTRON_MODEL_NAME", cls.nemotron_model_name),
            nemotron_model_id=os.getenv("NEMOTRON_MODEL_ID", cls.nemotron_model_id),
            nemotron_fp16=_env_bool("NEMOTRON_FP16", cls.nemotron_fp16),
            nemotron_itn=_env_bool("NEMOTRON_ITN", cls.nemotron_itn),
            whisper_model=os.getenv("WHISPER_MODEL", cls.whisper_model),
            stt_device=os.getenv("STT_DEVICE", cls.stt_device).lower(),
            #
            wake_word=_env_bool("WAKE_WORD", cls.wake_word),
            wake_word_model=os.getenv("WAKE_WORD_MODEL", cls.wake_word_model),
            wake_word_threshold=float(
                os.getenv("WAKE_WORD_THRESHOLD", str(cls.wake_word_threshold))
            ),
            #
            tts_provider=os.getenv("TTS_PROVIDER", cls.tts_provider).lower(),
            tts_base_url=tts_base_url,
            tts_voice=os.getenv("TTS_VOICE", cls.tts_voice),
            tts_api_key=os.getenv("TTS_API_KEY", cls.tts_api_key),
            tts_bind_port=int(os.getenv("TTS_BIND_PORT", str(cls.tts_bind_port))),
            tts_bind_host=os.getenv("TTS_BIND_HOST", bind_host),
            manage_tts=_env_bool("MANAGE_TTS", _is_loopback(tts_base_url)),
            #
            device=os.getenv("DEVICE", cls.device).lower(),
            turn_detection=os.getenv("TURN_DETECTION", cls.turn_detection).lower(),
            agent_idle_processes=_env_int_opt("AGENT_IDLE_PROCESSES"),
            sequential_startup=_env_bool("SEQUENTIAL_STARTUP", cls.sequential_startup),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
        )

    @property
    def advertised_livekit_url(self) -> str:
        """The LiveKit address to hand to browsers."""
        return self.livekit_public_url or self.livekit_url

    def agent_env(self) -> dict[str, str]:
        """Environment variables to pass to the agent worker subprocess."""
        env = {
            "LIVEKIT_URL": self.livekit_url,
            "LIVEKIT_API_KEY": self.livekit_api_key,
            "LIVEKIT_API_SECRET": self.livekit_api_secret,
            "LLAMA_BASE_URL": self.llama_base_url,
            "LLAMA_MODEL": self.llama_model,
            "LLAMA_API_KEY": self.llama_api_key,
            "STT_PROVIDER": self.stt_provider,
            "STT_BASE_URL": self.stt_base_url,
            "STT_MODEL": self.stt_model,
            "STT_LANGUAGE": self.stt_language,
            "STT_API_KEY": self.stt_api_key,
            "WAKE_WORD": "1" if self.wake_word else "0",
            "WAKE_WORD_MODEL": self.wake_word_model,
            "WAKE_WORD_THRESHOLD": str(self.wake_word_threshold),
            "TTS_BASE_URL": self.tts_base_url,
            "TTS_PROVIDER": self.tts_provider,
            "TTS_VOICE": self.tts_voice,
            "TTS_API_KEY": self.tts_api_key,
            "LLM_AUDIO_INPUT": "1" if self.llm_audio_input else "0",
            "TURN_DETECTION": self.turn_detection,
        }
        if self.agent_idle_processes is not None:
            env["AGENT_IDLE_PROCESSES"] = str(self.agent_idle_processes)
        return env
