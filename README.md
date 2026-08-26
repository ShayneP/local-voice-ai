<div align="center">
  <img src="./frontend/.github/assets/template-light.webp" alt="Local Voice Agent" width="80" />
  <h1>Local Voice Agent</h1>
  <p>A private, low-latency voice assistant that runs on your hardware.</p>
  <p>Powered by <a href="https://docs.livekit.io/agents?utm_source=local-voice-ai">LiveKit Agents</a>.</p>
</div>

Local Voice Agent combines speech recognition, a language model, and speech
generation in one supervised application. It selects a model stack that fits
the available hardware and memory.

The application includes:

- A browser voice interface.
- Local speech recognition with Nemotron or Whisper.
- Local language models through llama.cpp.
- Local speech generation with Kokoro.
- Automatic setup for CPU, NVIDIA, Apple Silicon, and Jetson.
- A remote-client mode for devices that run without a local browser.

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
model files. Docker also reuses its image layers.

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
| `lean`            |  About 4.7 GB | Qwen3 1.7B     |      4K | Whisper Small      | Kokoro ONNX |
| `jetson-realtime` |  About 4.7 GB | Qwen3 1.7B     |      4K | Streaming Nemotron | Kokoro ONNX |
| `compact`         |  About 5.5 GB | Gemma 4 E2B    |      4K | Nemotron           | Kokoro      |
| `balanced`        |  About 6.5 GB | Gemma 4 E2B    |     16K | Nemotron           | Kokoro      |

The memory values are planning targets, not hard limits. The automatic mode
keeps memory available for the operating system and active conversations.

To set a memory budget, use `--memory-gb`:

```bash
python3 run.py start --profile auto --memory-gb 5.5 --yes
```

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
| `STT_PROVIDER`    | `nemotron`, `nemotron-cpp`, or `whisper`           |
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
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Gemma 4](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF)
- [Kokoro](https://github.com/hexgrad/kokoro)
- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

Questions and feature requests are welcome through [GitHub Issues](https://github.com/ShayneP/local-voice-ai/issues).
