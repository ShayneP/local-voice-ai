<div align="center">
  <img src="./voice-assistant-frontend/.github/assets/app-icon.png" alt="App Icon" width="80" />
  <h1>🧠 Local Voice Agent</h1>
  <p>A full-stack, Dockerized AI voice assistant with speech, text, and voice synthesis powered by <a href="https://livekit.io?utm_source=demo">LiveKit</a>.</p>
</div>

[Demo Video](https://github.com/user-attachments/assets/67a76e94-aacb-4087-b09c-d4e46d8e695e)

## 🧩 Overview

This repo contains everything needed to run a real-time AI voice assistant locally using:

- 🎙️ **LiveKit Agents** for STT ↔ LLM ↔ TTS
- 🧠 **llama.cpp** for running local LLMs
- 🗣️ **Kokoro** for TTS voice synthesis
- 👂 **Whisper (via VoxBox)** for speech-to-text
- 🔍 **RAG** powered by Sentence Transformers and FAISS
- 💬 **Next.js + Tailwind** frontend UI
- 🐳 Fully containerized via Docker Compose

## 🏁 Quick Start

```bash
./test.sh
```

This script:
- Cleans up existing containers
- Builds all services
- Launches the full stack (agent, LLM, STT, TTS, frontend, and signaling server)

Once it's up, visit [http://localhost:3000](http://localhost:3000) in your browser to start chatting.

## 📦 Architecture

Each service is containerized and communicates over a shared Docker network:
- `livekit`: WebRTC signaling server
- `agent`: Custom Python agent with LiveKit SDK
- `whisper`: Speech-to-text using `vox-box` and Whisper model
- `llama_cpp`: Local LLM provider (e.g., `ggml-org/gemma-3-1b-it-GGUF`)
- `kokoro`: TTS engine for speaking responses
- `frontend`: React-based client using LiveKit components

## 🧠 Agent Instructions

Your agent lives in [`agent/myagent.py`](./agent/myagent.py). It uses:
- `openai.STT` → routes to Whisper
- `openai.LLM` → routes to llama.cpp
- `groq.TTS` → routes to Kokoro
- `silero.VAD` → for voice activity detection
- `SentenceTransformer` → embeds documents and queries for RAG
- `FAISS` → performs similarity search for knowledge retrieval

The agent supports Retrieval-Augmented Generation (RAG) by loading documents from the `agent/docs` directory. These documents are embedded using the all-MiniLM-L6-v2 model and indexed using FAISS for fast similarity search. During conversations, relevant document snippets are automatically retrieved to enhance the agent's responses.

All metrics from each component are logged for debugging.

## 🔐 Environment Variables

You can find environment examples in:
- [`/.env`](./.env)
- [`/agent/.env`](./agent/.env)
- [`/voice-assistant-frontend/.env.example`](./voice-assistant-frontend/.env.example)

These provide keys and internal URLs for each service. Most keys are placeholders for local dev use.

## 🧪 Testing & Dev

To test or redeploy:

```bash
docker-compose down -v --remove-orphans
docker-compose up --build
```

The services will restart and build fresh containers.

## 🧰 Project Structure

```
.
├── agent/                     # Python voice agent
├── llama_cpp/                 # LLM serving
├── whisper/                   # Whisper via vox-box
├── livekit/                   # Signaling server
├── voice-assistant-frontend/ # Next.js UI client
└── docker-compose.yml         # Brings it all together
```

## 📷 Screenshots

![UI Screenshot](./voice-assistant-frontend/.github/assets/frontend-screenshot.jpeg)

## 🛠️ Requirements

- Docker + Docker Compose
- No GPU required (uses CPU-based models)
- Recommended RAM: 12GB+

## 🙌 Credits

- Built with ❤️ by [LiveKit](https://livekit.io/)
- Uses [LiveKit Agents](https://docs.livekit.io/agents/)
- Local LLMs via [llama.cpp](https://github.com/ggml-org/llama.cpp)
- TTS via [Kokoro](https://github.com/remsky/kokoro)
