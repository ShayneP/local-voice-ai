<div align="center">
  <img src="./frontend/.github/assets/template-light.webp" alt="App Icon" width="80" />
  <h1>Local Voice AI</h1>
  <p>This project's goal is to enable anyone to easily build a powerful, private, local voice AI agent.</p>
  <p>A real-time voice AI assistant — STT, LLM, TTS — running in <strong>one container</strong>, supervised by a single Python parent process. Powered by <a href="https://docs.livekit.io/agents?utm_source=local-voice-ai">LiveKit Agents</a>.</p>
</div>

## Overview

Everything runs as managed children of one Python supervisor (`python -m local_voice_ai serve`):

- **LiveKit server** (Go binary subprocess) for WebRTC signaling — skipped if `LIVEKIT_URL` points at LiveKit Cloud.
- **llama.cpp** (`llama-server` binary subprocess) for the LLM — skipped if `LLAMA_BASE_URL` points elsewhere.
- **Nemotron STT** or **Whisper (vox-box)** — Python uvicorn child, OpenAI-compatible.
- **Kokoro TTS** — Python uvicorn child, OpenAI-compatible.
- **LiveKit Agents worker** — the orchestrator child.
- **FastAPI** in the supervisor itself, serving `POST /api/connection-details` (token minting) and the statically-exported Next.js frontend.

Children speak HTTP only over `127.0.0.1`. The image exposes three ports: `8080` (web), `7880`, `7881` (LiveKit WebRTC, only if running locally).

> **Optional [audio-in mode (Gemma 4)](#audio-in-mode-gemma-4):** feed microphone audio straight into an audio-native LLM and drop the STT stage entirely — a LiveKit half-cascade.

## Getting started

```bash
docker compose up --build
```

Open <http://localhost:8080> and click the start button.

The first build pulls upstream binaries (llama-server, livekit-server) and downloads the Nemotron + LLM weights on first request — expect tens of GB on first boot.

### GPU (NVIDIA)

```bash
LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda \
PYTHON_BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
LLAMA_N_GPU_LAYERS=35 \
docker compose up --build
```

### Apple Silicon

The CPU image works as-is. `llama-server` uses Metal automatically through its bundled binary.

## Audio-in mode (Gemma 4)

[Gemma 4 12B](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) accepts **audio as input natively** — it's encoder-free, projecting raw audio straight into the token space (distinct from Gemma 3n's USM audio encoder). That lets us collapse STT and the LLM into one model: the user's microphone audio goes **straight to the LLM**, which transcribes and reasons in a single pass and returns text; Kokoro still speaks the reply. In LiveKit terms this is a [half-cascade](https://docs.livekit.io/agents/models/pipelines/#half-cascade) — an audio-in "realtime" model paired with a standalone TTS.

```
default:   mic → STT (Nemotron) → LLM (Qwen, text) → TTS (Kokoro) → speaker
audio-in:  mic → [ Gemma 4 audio LLM: transcribe + reason ] → text → TTS (Kokoro) → speaker
                 ↑ no STT service; Silero VAD segments turns
```

Enable it with `LLM_AUDIO_INPUT=1`. The supervisor then **does not spawn the STT child** and serves an audio-capable GGUF (`unsloth/gemma-4-12b-it-GGUF` by default) with its multimodal projector, adding `--jinja` and `--reasoning-budget 0`.

### Running it

**GPU (NVIDIA) — everything in one container:**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

**macOS (Apple Silicon) — hybrid:** Docker Desktop has no Metal passthrough, so a 12B *inside* the container would be CPU-bound and far too slow. Run the audio LLM **natively** (Metal) and the rest in Docker:

```bash
./scripts/run-native-audio-llm.sh                                    # native llama-server on :11500 (Metal)
docker compose -f docker-compose.yml -f docker-compose.audio.yml up  # livekit + kokoro + agent
```

The container reaches the host LLM via `host.docker.internal`; a non-loopback `LLAMA_BASE_URL` automatically tells the supervisor not to spawn its own llama child.

> Plain CPU `docker compose up` with `LLM_AUDIO_INPUT=1` is **not** recommended — a 12B multimodal model is minutes-per-turn on CPU.

### Requirements

- **llama.cpp ≥ b9518** for the Gemma 4 "unified" audio projector (`gemma4uv`). Older builds fail with `unknown projector type: gemma4uv`; the `:server-cuda` image and Homebrew `llama.cpp` latest both qualify.
- **~16 GB VRAM** for the 12B at Q4 (model ~7 GB + KV cache + Kokoro), or Apple Silicon with ≥ 24 GB unified memory.

### Design nuances

The audio path is a custom LiveKit `RealtimeModel` (`local_voice_ai/realtime/audio_llm.py`), **not** the usual `openai.LLM()` — the OpenAI LLM plugin is text-only and silently drops audio content. A few non-obvious decisions make it usable as a real-time assistant:

- **Batch-per-turn, not streaming.** llama.cpp has no streaming audio input (it processes whole utterances in ~30 s chunks), so Silero VAD segments each turn, the audio is encoded to a WAV, and sent as one OpenAI `input_audio` request. Consequence: **no interim transcripts** while you speak.
- **Thinking is disabled** (`--reasoning-budget 0`). Gemma 4 emits a hidden reasoning channel by default, which added ~4.5 s before the actual answer; disabling it drops time-to-first-spoken-token to ~1 s. (The plugin also ignores any `reasoning_content` deltas.)
- **Transcribe-first grounding.** Gemma 4 *misreads self-referential questions from audio alone* — asked "what is my name?" by voice it answers about itself. So each turn is transcribed first (by the same model) and the transcript is sent **as text alongside the audio**: the text grounds the question, the audio preserves prosody. That one transcription also:
  - drives the **UI captions** (`input_audio_transcription_completed`), and
  - feeds a **session-owned conversation history** replayed on later turns — which is what gives the agent multi-turn memory (the model only "remembers" what we replay back to it).
  - Cost: ~0.5 s added to first-token latency, since transcription runs before the reply.
- **Half-cascade output.** Gemma 4 is text-out only, so Kokoro TTS speaks the reply.
- **Typed text works too.** A message typed in the UI (no audio) is routed through the chat context as a normal text completion, so you can freely mix talking and typing in one conversation.
- **Tool/function calling works.** Function tools are advertised on each request; streamed `tool_calls` are emitted to the framework, which executes them and feeds the result back through the chat context for a final spoken answer — for audio and typed turns alike.

### Configuration

| Variable               | Default                          | Purpose                                            |
| ---------------------- | -------------------------------- | -------------------------------------------------- |
| `LLM_AUDIO_INPUT`      | `0`                              | `1` enables audio-in mode (skips the STT child)    |
| `AUDIO_LLM_HF_REPO`    | `unsloth/gemma-4-12b-it-GGUF`    | audio-capable GGUF repo                            |
| `AUDIO_LLM_QUANT`      | `Q4_K_M`                         | quantization tag                                   |
| `AUDIO_LLM_ALIAS`      | `gemma-4-audio`                  | model name the agent addresses                     |
| `AUDIO_LLM_MMPROJ_URL` | unsloth BF16 mmproj              | multimodal projector (BF16 recommended for Gemma)  |

### Limitations

- No interim/streaming transcripts (captions appear per-utterance, just before the reply).
- Spoken-turn latency is ~1.5 s end-to-end (VAD end-of-turn + transcribe + reply); typed turns are faster.

## Swapping in cloud providers

Each service has a single "manage" decision driven by its base URL — point it at a remote endpoint and the local subprocess is skipped:

| Goal                              | Set                                                                                  |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| Use LiveKit Cloud                 | `LIVEKIT_URL=wss://your-project.livekit.cloud` (+ `LIVEKIT_API_KEY` / `…_SECRET`)   |
| Use OpenAI for the LLM            | `LLAMA_BASE_URL=https://api.openai.com/v1`, `LLAMA_MODEL=gpt-4o-mini`, `LLAMA_API_KEY=sk-…` |
| Use a remote OpenAI-compatible STT| `STT_BASE_URL=…`, `STT_MODEL=…`, `STT_API_KEY=…`                                     |
| Use a remote OpenAI-compatible TTS| `TTS_BASE_URL=…`, `TTS_API_KEY=…`                                                    |

The supervisor logs which children it manages on startup.

## Local development (no Docker)

```bash
# Python side
uv pip install -e ".[ml,dev]"
python -m local_voice_ai serve

# Frontend side, in another shell (only needed if you're editing the UI)
cd frontend && pnpm install && pnpm run dev
```

## Architecture

```
┌──────────────────────── single container ────────────────────────┐
│  python -m local_voice_ai serve                                  │
│  │                                                                │
│  ├── child: livekit-server     (skipped if LIVEKIT_URL external) │
│  ├── child: llama-server       (skipped if LLAMA_BASE_URL ext.)  │
│  ├── child: nemotron | whisper (skipped if STT_BASE_URL ext.)    │
│  ├── child: kokoro             (skipped if TTS_BASE_URL ext.)    │
│  ├── child: livekit-agents worker                                │
│  └── in-process: FastAPI on :8080                                 │
│        ├── POST /api/connection-details  (token minting)         │
│        └── GET  /*                       (static frontend)       │
└───────────────────────────────────────────────────────────────────┘
```

## Project structure

```
.
├─ local_voice_ai/         # Python package: supervisor + agent + services
│  ├─ __main__.py          # python -m local_voice_ai serve
│  ├─ supervisor.py        # async process supervisor
│  ├─ config.py            # env-driven config + manage-X flags
│  ├─ api.py               # FastAPI: token route + static frontend
│  ├─ agent.py             # LiveKit Agents worker
│  ├─ realtime/
│  │  └─ audio_llm.py      # audio-in RealtimeModel (Gemma 4 half-cascade)
│  └─ services/
│     ├─ nemotron/server.py
│     └─ kokoro/server.py
├─ frontend/               # Next.js (configured for static export)
├─ scripts/
│  └─ run-native-audio-llm.sh   # native (Metal) audio LLM for the macOS hybrid
├─ Dockerfile              # multi-stage build
├─ docker-compose.yml      # base service (STT → LLM → TTS)
├─ docker-compose.gpu.yml  # GPU audio-in overlay (single container)
├─ docker-compose.audio.yml # macOS/Metal audio-in hybrid overlay
└─ pyproject.toml          # one Python package, one venv
```

## Environment variables

See `.env` for the full list. The most important ones:

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` — local-default; override for cloud.
- `LLAMA_BASE_URL`, `LLAMA_MODEL`, `LLAMA_HF_REPO`, `LLAMA_N_GPU_LAYERS`
- `STT_PROVIDER` (`nemotron`|`whisper`), `STT_BASE_URL`, `STT_MODEL`
- `TTS_BASE_URL`, `TTS_VOICE`
- `WEB_PORT` (default `8080`)
- `MANAGE_LIVEKIT`, `MANAGE_LLAMA`, `MANAGE_STT`, `MANAGE_TTS` — explicit overrides for the auto-detected "is the URL external?" logic.
- `LLM_AUDIO_INPUT`, `AUDIO_LLM_HF_REPO`, `AUDIO_LLM_QUANT`, `AUDIO_LLM_ALIAS`, `AUDIO_LLM_MMPROJ_URL` — see [Audio-in mode (Gemma 4)](#audio-in-mode-gemma-4).

## Credits

- LiveKit: <https://livekit.io/>
- LiveKit Agents: <https://docs.livekit.io/agents/>
- NVIDIA Nemotron Speech: <https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b>
- Gemma 4 (audio-in mode): <https://huggingface.co/unsloth/gemma-4-12b-it-GGUF>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- Kokoro TTS: <https://github.com/hexgrad/kokoro>
- VoxBox (Whisper fallback): <https://pypi.org/project/vox-box/>
