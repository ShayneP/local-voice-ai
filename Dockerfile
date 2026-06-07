# syntax=docker/dockerfile:1.6
#
# Single-image build for local-voice-ai.
#
# Stages:
#   frontend  → produces a Next.js static export at /app/out
#   binaries  → references upstream images for the livekit-server and llama-server binaries
#   runtime   → Python 3.11 with all deps + the binaries + the frontend
#
# Build args:
#   --build-arg LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda  (for GPU)
#   --build-arg PYTHON_BASE=python:3.11-slim                        (or nvidia/cuda...)

ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server
ARG LIVEKIT_IMAGE=livekit/livekit-server:latest
ARG PYTHON_BASE=python:3.11-slim

# ---------------- frontend ----------------
FROM node:20-slim AS frontend
WORKDIR /app
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm run build

# ---------------- binary sources ----------------
FROM ${LLAMA_IMAGE} AS llama-bin
FROM ${LIVEKIT_IMAGE} AS livekit-bin

# ---------------- runtime ----------------
FROM ${PYTHON_BASE} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    HF_HOME=/models \
    XDG_CACHE_HOME=/models \
    # Ubuntu-based CUDA bases (GPU build) ship a PEP 668 "externally managed"
    # system Python that refuses pip/uv system installs. We deliberately install
    # into the system env (single-purpose image), so opt out. No-op on the
    # default python:3.11-slim base, which isn't externally managed.
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_BREAK_SYSTEM_PACKAGES=1

# System libs needed by the inference stack and the binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

# The default base (python:3.11-slim) ships Python, but CUDA bases
# (nvidia/cuda:*-runtime, used for GPU builds) do not — install it when absent.
RUN command -v python3 >/dev/null 2>&1 || ( \
        apt-get update \
        && apt-get install -y --no-install-recommends python3 python3-pip \
        && rm -rf /var/lib/apt/lists/* )
RUN command -v python >/dev/null 2>&1 || ln -s "$(command -v python3)" /usr/local/bin/python

WORKDIR /app

# Install Python deps via uv for speed and a reproducible env
RUN python -m pip install --no-cache-dir uv

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Copy project metadata first for layer caching
COPY pyproject.toml ./
COPY local_voice_ai ./local_voice_ai

# Install: torch (with explicit index for CPU/CUDA selection) + the [ml] extras
# in a single resolution pass so versions are consistent.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-strategy unsafe-best-match \
        --extra-index-url ${TORCH_INDEX_URL} \
        ".[ml]"

# Drop in the binaries from upstream images.
#
# llama-server is dynamically linked against shared libraries that ship next to
# it in the upstream image's /app dir (libllama*.so, libggml*.so, libmtmd*.so),
# plus the libggml-cpu-*.so / libggml-cuda.so backends it dlopen()s at runtime.
# Its RUNPATH is the absolute build path /app/build/bin (which doesn't exist
# here) and it has no $ORIGIN entry, so copying just the binary leaves the
# loader unable to find libllama-server-impl.so. Copy the whole library set into
# a dedicated dir and register it with ldconfig so both the link-time NEEDED
# libs and the runtime-dlopen'd backends resolve. Registering via ldconfig
# (rather than LD_LIBRARY_PATH) keeps the CUDA/driver search paths the nvidia
# base image configures for the GPU build untouched.
COPY --from=llama-bin /app/ /usr/local/lib/llama/
RUN ln -s /usr/local/lib/llama/llama-server /usr/local/bin/llama-server \
    && echo /usr/local/lib/llama > /etc/ld.so.conf.d/llama.conf \
    && ldconfig
COPY --from=livekit-bin /livekit-server /usr/local/bin/livekit-server

# Drop in the static-exported frontend
COPY --from=frontend /app/out /app/frontend/out
ENV FRONTEND_DIR=/app/frontend/out

# Pre-download VAD + turn detector weights so cold start is faster
RUN python -m local_voice_ai.agent download-files || true

EXPOSE 8080 7880 7881
VOLUME ["/models"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "local_voice_ai", "serve"]
