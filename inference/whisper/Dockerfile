FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir vox-box

WORKDIR /app

ENV DATA_DIR=/data

CMD ["sh", "-c", "vox-box start --huggingface-repo-id ${VOXBOX_HF_REPO_ID:-Systran/faster-whisper-small} --data-dir ${DATA_DIR}"]
