# syntax=docker/dockerfile:1.5
ARG WHISPER_BASE=python:3.10-slim
FROM ${WHISPER_BASE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && if ! command -v python3 >/dev/null 2>&1; then \
        apt-get install -y --no-install-recommends python3 python3-pip; \
    fi \
    && if ! command -v pip3 >/dev/null 2>&1; then \
        apt-get install -y --no-install-recommends python3-pip; \
    fi \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install vox-box

WORKDIR /app

ENV DATA_DIR=/data

CMD ["sh", "-c", "vox-box start --huggingface-repo-id ${VOXBOX_HF_REPO_ID:-Systran/faster-whisper-small} --data-dir ${DATA_DIR} --device ${VOXBOX_DEVICE:-cpu}"]
