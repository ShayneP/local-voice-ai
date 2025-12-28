<div align="center">
  <img src="./frontend/.github/assets/template-light.webp" alt="App Icon" width="80" />
  <h1>Local Voice AI</h1>
  <p>This project's goal is to enable anyone to easily build a powerful, private, local voice AI agent.</p>
  <p>A full-stack, Dockerized AI voice assistant with speech, text, and voice synthesis delivered via WebRTC powered by <a href="https://docs.livekit.io/agents?utm_source=local-voice-ai">LiveKit Agents</a>.</p>
</div>

## Overview

This repo contains everything needed to run a real-time AI voice assistant locally using:

- **LiveKit** for WebRTC realtime audio + rooms.
- **LiveKit Agents (Python)** to orchestrate the STT → LLM → TTS pipeline.
- **Whisper (via VoxBox)** for speech-to-text.
- **Ollama** for running local LLMs.
- **Kokoro** for text-to-speech voice synthesis.
- **Next.js + Tailwind** frontend UI.
- Fully containerized via Docker Compose.

## Getting Started

Windows uses the PowerShell command; Linux and OSX use the bash command. Both will prompt you to choose CPU or Nvidia GPU.

Windows:
```bash
./compose-up.ps1
```

Mac / Linux:
```bash
./compose-up.sh
```

Once it's up, visit [http://localhost:3000](http://localhost:3000) in your browser to start chatting.

### Notes on models and resources

- The default LLM is `qwen3-vl:4b`, because it's small and it supports tools.
- If you want to use a different LLM, you can change the `OLLAMA_MODEL` environment variable.
- If you want to use a different STT, you can change the `VOXBOX_HF_REPO_ID` environment variable.
- You can even swap out the URLs to use cloud models if you want (see `livekit_agent/src/agent.py`).
- The first time you run it, it needs to download a lot of stuff (often ~40GB) of models and supporting libraries (like CUDA) for the different local inference providers. CPU-only is much smaller and faster for the initial install.
- Installing takes a while. On an i9-14900hx it takes about 10 minutes to get everything ready.
- Once it's all downloaded though, the whole suite itself fits in ~8GB of VRAM.
## Architecture

Each service is containerized and communicates over a shared Docker network:

- `livekit`: WebRTC signaling server
- `livekit_agent`: Python agent (LiveKit Agents SDK)
- `whisper`: Speech-to-text (VoxBox + Whisper)
- `ollama`: Local LLM provider
- `kokoro`: TTS engine
- `frontend`: Next.js client UI

## Agent

The agent entrypoint is `livekit_agent/src/agent.py`. It uses the LiveKit Agents OpenAI-compatible plugins to talk to local inference services:

- `openai.STT` → the VoxBox Whisper container
- `openai.LLM` → the Ollama container
- `openai.TTS` → the Kokoro container
- `silero.VAD` for voice activity detection

## Environment variables

Example env files:

- `.env` (used by Docker Compose)
- `frontend/.env.example`
- `livekit_agent/.env.example`

For local (non-Docker) development, use `.env.local` files:

- `frontend/.env.local`
- `livekit_agent/.env.local`

### LiveKit URLs (important)

The LiveKit URL is used in two different contexts:

- `LIVEKIT_URL` is the internal, server-to-server address (e.g. `ws://livekit:7880`) used by containers like the agent.
- `NEXT_PUBLIC_LIVEKIT_URL` is the browser-reachable LiveKit address returned by the frontend API (e.g. `ws://localhost:7880`).

The frontend only signs tokens; it does not connect to LiveKit directly. The browser connects using the `serverUrl` returned by `/api/connection-details`, so make sure `NEXT_PUBLIC_LIVEKIT_URL` points to a reachable LiveKit endpoint.

## Development

Use `.env.local` files in both `frontend` and `livekit_agent` dirs to set the dev environment variables for the project. This way, you can run either of those with `pnpm dev` or `uv run python src/agent.py dev` and test them without needing to build the Docker projects. You can just run either locally and hotreload your changes.

## Project structure

```
.
├─ frontend/        # Next.js UI client
├─ inference/       # Local inference services (ollama/whisper/kokoro)
├─ livekit/         # LiveKit server config
├─ livekit_agent/   # Python voice agent (LiveKit Agents)
├─ docker-compose.yml
└─ docker-compose.gpu.yml
```

## Requirements

- Docker + Docker Compose
- No GPU required (CPU works)
- Recommended RAM: 12GB+

## Credits

- Built with LiveKit: https://livekit.io/
- Uses LiveKit Agents: https://docs.livekit.io/agents/
- STT via VoxBox + Whisper: https://github.com/gpustack/vox-box
- Local LLM via Ollama: https://ollama.com/
- TTS via Kokoro: https://github.com/remsky/kokoro
