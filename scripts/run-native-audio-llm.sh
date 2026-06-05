#!/usr/bin/env bash
# Run the audio-in LLM (Gemma-4-12B) natively with GPU/Metal acceleration, for
# use with: docker compose -f docker-compose.yml -f docker-compose.audio.yml up
#
# Why native: a 12B multimodal model is far too slow on CPU (and Docker Desktop
# on macOS has no Metal passthrough), so the LLM runs on the host while the rest
# of the stack stays containerized.
#
# Requires llama.cpp >= b9518 (the Gemma-4 "gemma4uv" multimodal projector). If a
# recent llama-server isn't provided via LLAMA_SERVER_BIN, a pinned macOS-arm64
# build is fetched into ./.bin.
set -euo pipefail

PORT="${AUDIO_LLM_PORT:-11500}"
REPO="${AUDIO_LLM_HF_REPO:-unsloth/gemma-4-12b-it-GGUF}"
QUANT="${AUDIO_LLM_QUANT:-Q4_K_M}"
MMPROJ_URL="${AUDIO_LLM_MMPROJ_URL:-https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/mmproj-BF16.gguf}"
BUILD="${LLAMA_BUILD:-b9518}"

BIN="${LLAMA_SERVER_BIN:-}"
if [ -z "$BIN" ]; then
  DIR=".bin/llama-${BUILD}"
  if [ ! -x "$DIR/llama-server" ]; then
    echo "fetching llama.cpp ${BUILD} (macOS arm64) into ${DIR}..."
    mkdir -p "$DIR"
    curl -fsSL -o "/tmp/llama-${BUILD}.tar.gz" \
      "https://github.com/ggml-org/llama.cpp/releases/download/${BUILD}/llama-${BUILD}-bin-macos-arm64.tar.gz"
    tar -xzf "/tmp/llama-${BUILD}.tar.gz" --strip-components=1 -C "$DIR"
  fi
  BIN="$DIR/llama-server"
fi

echo "serving ${REPO}:${QUANT} on 0.0.0.0:${PORT} (Metal, reasoning disabled)"
# --jinja: required by the Gemma-4 chat template.
# --reasoning-budget 0: skip "thinking" so the spoken reply isn't delayed.
# --host 0.0.0.0: reachable from the container via host.docker.internal.
exec "$BIN" \
  -hf "${REPO}:${QUANT}" \
  --mmproj-url "${MMPROJ_URL}" \
  --jinja --reasoning-budget 0 --alias gemma-4-audio \
  --host 0.0.0.0 --port "${PORT}" -c 8192
