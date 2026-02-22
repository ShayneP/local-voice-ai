# syntax=docker/dockerfile:1.5
ARG NEMOTRON_BASE=python:3.10-slim
FROM ${NEMOTRON_BASE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      ffmpeg \
      libsndfile1 \
    && if ! command -v python3 >/dev/null 2>&1; then \
      apt-get install -y --no-install-recommends python3 python3-pip python3-venv; \
    fi \
    && if ! command -v pip3 >/dev/null 2>&1; then \
      apt-get install -y --no-install-recommends python3-pip; \
    fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --index-url ${TORCH_INDEX_URL} torch torchaudio \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install "nemo-toolkit[asr]" huggingface_hub

COPY server.py .

EXPOSE 8000

CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "8000"]
