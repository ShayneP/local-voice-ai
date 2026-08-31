<div align="center">
  <img src="./frontend/.github/assets/template-light.webp" alt="Local Voice Agent" width="80" />
  <h1>Local Voice Agent</h1>
  <p>A private, low-latency voice assistant that runs on your hardware.</p>
  <p>Powered by <a href="https://docs.livekit.io/agents?utm_source=local-voice-ai">LiveKit Agents</a>.</p>
</div>

Local Voice Agent combines speech recognition, a language model, and speech
generation in one supervised application. It selects a model stack that fits
the available hardware and memory.

> [!TIP]
> The Jetson profile supports real-time voice conversations on a Jetson Orin Nano.

The application includes:

- A browser voice interface.
- Local streaming speech recognition with Nemotron Q8.
- Local language models through llama.cpp.
- Local speech generation with Kokoro.
- Automatic setup for CPU, NVIDIA, Apple Silicon, and Jetson.
- A remote-client mode for devices that run without a local browser.

> **Optional [audio-in mode (Gemma 4)](#audio-in-mode-gemma-4):** feed microphone audio straight into an audio-native LLM and drop the STT stage entirely — a LiveKit half-cascade.

## Requirements

Clone this repository before you start:

```bash
git clone https://github.com/ShayneP/local-voice-ai.git
cd local-voice-ai
```

The setup launcher needs Python 3.10 or later.

Install the additional tools for your platform:

| Platform       | Requirement                                                  |
| -------------- | ------------------------------------------------------------ |
| Linux CPU      | Docker Engine with Docker Compose                            |
| Desktop NVIDIA | Docker Engine, Docker Compose, and NVIDIA Container Toolkit  |
| Jetson Orin    | JetPack 6.2, L4T 36.4, and the NVIDIA Docker runtime         |
| Apple Silicon  | Python 3.11–3.13, `uv`, `livekit-server`, and `llama-server` |

On Apple Silicon, install the native server tools with Homebrew:

```bash
brew install livekit llama.cpp
uv sync --extra ml --extra dev
```

The first start needs an internet connection. Later starts reuse downloaded
model files and native components. Docker also reuses its image layers.

## Quick start

Start the setup launcher:

```bash
python3 run.py
```

The launcher shows the detected hardware, memory budget, and recommended
models. Accept the recommendation or select a different profile.

When the application is ready, open <http://localhost:8080>. When the browser
requests microphone access, permit it.

For a non-interactive start, run:

```bash
python3 run.py start --profile auto --yes
```

## Model profiles

Automatic selection uses the device type to select a runtime. It then uses the
memory budget to select a model profile.

| Profile           | Memory target | Language model | Context | Speech recognition | Voice       |
| ----------------- | ------------: | -------------- | ------: | ------------------ | ----------- |
| `lean`            |  About 4.7 GB | Qwen3 1.7B     |      4K | Nemotron Q8        | Kokoro ONNX |
| `jetson-realtime` |  About 4.7 GB | Qwen3 1.7B     |      4K | Nemotron Q8        | Kokoro ONNX |
| `compact`         |  About 5.5 GB | Gemma 4 E2B    |      4K | Nemotron Q8        | Kokoro      |
| `balanced`        |  About 6.5 GB | Gemma 4 E2B    |     16K | Nemotron Q8        | Kokoro      |

The memory values are planning targets, not hard limits. The automatic mode
keeps memory available for the operating system and active conversations.

All profiles use the native streaming Nemotron Q8 runtime. The launcher selects
the CPU, CUDA, or Metal runtime for the device.

The default language is English. For English, the application uses the
English-specific Nemotron model. For another supported language, it uses
Nemotron 3.5.

Set the language in `.env.local`:

```env
STT_LANGUAGE=fr-FR
```

If the speaker language can change, use `STT_LANGUAGE=auto`. This value selects
the multilingual model. Whisper remains available as a manual fallback:

```env
STT_PROVIDER=whisper
```

Whisper waits for a complete utterance before transcription. Nemotron sends
partial transcripts while the user speaks, so Nemotron has lower voice latency.

To set a memory budget, use `--memory-gb`:

```bash
python3 run.py start --profile auto --memory-gb 5.5 --yes
```

## Audio-in mode (Gemma 4)

[Gemma 4 12B](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) accepts **audio as input natively** — it's encoder-free, projecting raw audio straight into the token space (distinct from Gemma 3n's USM audio encoder). That lets us collapse STT and the LLM into one model: the user's microphone audio goes **straight to the LLM**, which transcribes and reasons in a single pass and returns text; Kokoro still speaks the reply. In LiveKit terms this is a [half-cascade](https://docs.livekit.io/agents/models/pipelines/#half-cascade) — an audio-in "realtime" model paired with a standalone TTS.

```
default:   mic → STT (Nemotron) → LLM (text) → TTS (Kokoro) → speaker
audio-in:  mic → [ Gemma 4 audio LLM: transcribe + reason ] → text → TTS (Kokoro) → speaker
                 ↑ no STT service; Silero VAD segments turns
```

Enable it with `LLM_AUDIO_INPUT=1`. The supervisor then **does not spawn the STT child** and serves an audio-capable GGUF (`unsloth/gemma-4-12b-it-GGUF` by default) with its multimodal projector, adding `--jinja` and `--reasoning-budget 0`.

### Running it

**GPU (NVIDIA) — everything in one container:**

```bash
LLM_AUDIO_INPUT=1 docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

**macOS (Apple Silicon) — hybrid:** Docker Desktop has no Metal passthrough, so a 12B *inside* the container would be CPU-bound and far too slow. Run the audio LLM **natively** (Metal) and the rest in Docker:

```bash
./scripts/run-native-audio-llm.sh                                    # native llama-server on :11500 (Metal)
docker compose -f docker-compose.yml -f docker-compose.audio.yml up  # livekit + kokoro + agent
```

The container reaches the host LLM via `host.docker.internal`; a non-loopback `LLAMA_BASE_URL` automatically tells the supervisor not to spawn its own llama child.

> Plain CPU `docker compose up` with `LLM_AUDIO_INPUT=1` is **not** recommended — a 12B multimodal model is minutes-per-turn on CPU.

### Audio-in requirements

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

### Audio-in configuration

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

## Use a Jetson as the voice server

The recommended Jetson setup runs the voice stack on the Jetson and the browser
interface on a laptop. This gives the browser a `localhost` address for
microphone access.

The Jetson setup needs approximately 29 GB of free disk space. The first build
compiles native components, so it takes longer than later builds.

### 1. Configure the Jetson address

On the Jetson, find its LAN address:

```bash
ip -4 -brief address
```

Create `.env.local` in the repository root. Replace the example address with
the Jetson address:

```env
LIVEKIT_URL=ws://192.168.1.40:7880
LIVEKIT_NODE_IP=192.168.1.40
MANAGE_LIVEKIT=1
```

### 2. Permit local network traffic

The laptop needs these ports on the Jetson:

| Port   | Protocol | Use                           |
| ------ | -------- | ----------------------------- |
| `8080` | TCP      | Connection details and status |
| `7880` | TCP      | LiveKit connection            |
| `7881` | TCP      | WebRTC fallback media         |
| `7882` | UDP      | WebRTC media                  |

If UFW is active, permit only the local subnet. Replace the example subnet with
your local subnet:

```bash
sudo ufw status
sudo ufw allow proto tcp from 192.168.1.0/24 to any port 7880,7881,8080 comment 'local voice ai'
sudo ufw allow proto udp from 192.168.1.0/24 to any port 7882 comment 'local voice ai media'
```

CAUTION: Do not expose these ports to the public internet. The default service
uses development credentials.

### 3. Start the Jetson

```bash
python3 run.py start --profile auto --memory-gb 5.5 --yes
```

Wait until the launcher reports that all services are ready.

### 4. Start the laptop client

Install Node.js 20 on the laptop. Then run:

```bash
git clone https://github.com/ShayneP/local-voice-ai.git
cd local-voice-ai
corepack enable
python3 run.py client --server 192.168.1.40
```

Open <http://localhost:3000>. The client installs its frontend packages on the
first start.

## Common commands

| Command                                 | Purpose                               |
| --------------------------------------- | ------------------------------------- |
| `python3 run.py`                        | Configure and start the application   |
| `python3 run.py configure`              | Select a different profile            |
| `python3 run.py plan`                   | Show the selected runtime and models  |
| `python3 run.py status`                 | Show service readiness                |
| `python3 run.py logs`                   | Follow the application logs           |
| `python3 run.py down`                   | Stop the Docker application           |
| `python3 run.py client --server <host>` | Run the interface for a remote server |

The launcher saves the selected profile in `.local-voice-ai.toml`. This file is
local to the device and is not committed to Git.

## Configuration

Put device-specific configuration in `.env.local`. This file overrides the
selected profile and the defaults in `.env`.

Common values include:

| Value             | Purpose                                            |
| ----------------- | -------------------------------------------------- |
| `LIVEKIT_URL`     | LiveKit server address                             |
| `LIVEKIT_NODE_IP` | LAN address advertised by a managed LiveKit server |
| `LLAMA_MODEL`     | Model name used by the agent                       |
| `LLAMA_HF_REPO`   | GGUF model repository and quantization             |
| `STT_PROVIDER`    | Speech engine. The default is `nemotron-cpp`       |
| `STT_LANGUAGE`    | Speech language. The default is `en`               |
| `TTS_VOICE`       | Kokoro voice name                                  |
| `WAKE_WORD=1`     | Require “Hey LiveKit” before the agent listens     |
| `WEB_PORT`        | Browser interface port. The default is `8080`      |

See [`.env`](./.env) for the complete list.

### Use an external service

Set a remote base URL to replace one local service. The supervisor does not
start the matching local process.

| Service            | Configuration                                          |
| ------------------ | ------------------------------------------------------ |
| LiveKit Cloud      | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` |
| Language model     | `LLAMA_BASE_URL`, `LLAMA_MODEL`, `LLAMA_API_KEY`       |
| Speech recognition | `STT_BASE_URL`, `STT_MODEL`, `STT_API_KEY`             |
| Speech generation  | `TTS_BASE_URL`, `TTS_API_KEY`                          |

Store API keys in `.env.local`. Do not commit this file.

## Troubleshooting

### A model shows several gigabytes during startup

The startup value is the model cache size on disk. It is not the memory used by
the process.

### The laptop cannot connect to the Jetson

From the laptop, request the Jetson status:

```bash
curl -fsS http://192.168.1.40:8080/api/status | python3 -m json.tool
```

If this command times out, make sure that the firewall permits the laptop
subnet.

### The interface connects without audio

Make sure that UDP port `7882` is open between the laptop and the Jetson.

### A service does not become ready

Show the current status and logs:

```bash
python3 run.py status
python3 run.py logs
```

## Local development

Local development needs Python 3.11–3.13, `uv`, Node.js 20, pnpm,
`livekit-server`, and `llama-server`.

Install the Python environment:

```bash
uv sync --extra ml --extra dev
.venv/bin/python -m local_voice_ai.agent download-files
```

Start the application:

```bash
.venv/bin/python -m local_voice_ai serve
```

This reads `.env.local`, then the saved profile in `.local-voice-ai.toml`, then
`.env`, so it starts with the same settings `python3 run.py` would use.

### Serve it to other computers

`serve` binds the web port to every interface, but it tells browsers to connect
to LiveKit on loopback, which no other machine can reach. Name the address they
should use instead:

```bash
# .env.local
LIVEKIT_PUBLIC_URL=ws://192.168.1.40:7880
```

That is the only variable needed: the ICE address follows it, and LiveKit is
still started here because `LIVEKIT_URL` remains on loopback. Set `LIVEKIT_URL`
itself only to use a LiveKit you run elsewhere, such as LiveKit Cloud.

`serve` does not host the web interface, so connect from the other computer
with `python3 run.py client --server 192.168.1.40`, which needs Node.js and
pnpm there but not Docker.

If you change the frontend, start its development server in another terminal:

```bash
corepack enable
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend dev
```

Run the automated tests:

```bash
.venv/bin/python -m pytest -q
pnpm --dir frontend build
```

## Security

The default configuration is for local development and trusted private
networks. It does not provide authentication for local model endpoints.

- Keep `.env.local` out of Git.
- Limit firewall rules to the local subnet.
- Do not publish the LiveKit or model ports directly to the internet.
- Use authentication and TLS before you expose the application through a
  public service.

## Credits

- [LiveKit](https://livekit.io/)
- [LiveKit Agents](https://docs.livekit.io/agents/)
- [NVIDIA Nemotron Speech](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [NVIDIA Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [NVIDIA NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Gemma 4](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF)
- [Gemma 4 12B (audio-in mode)](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF)
- [Kokoro](https://github.com/hexgrad/kokoro)
- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

Questions and feature requests are welcome through [GitHub Issues](https://github.com/ShayneP/local-voice-ai/issues).
