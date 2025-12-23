#!/bin/sh
set -e

OLLAMA_MODEL="${OLLAMA_MODEL:-gemma3:4b}"
OLLAMA_PREWARM="${OLLAMA_PREWARM:-true}"

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama server to be ready..."
# Wait for ollama to be ready by checking the API
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:11434/api/tags | grep -q "$OLLAMA_MODEL"; then
    echo "Downloading ${OLLAMA_MODEL} model..."
    ollama pull "$OLLAMA_MODEL"
fi

if [ "$OLLAMA_PREWARM" != "false" ]; then
    echo "Warming up ${OLLAMA_MODEL} model..."
    warmup_payload=$(printf '{"model":"%s","prompt":"warmup","stream":false,"keep_alive":"10m","options":{"num_predict":1}}' "$OLLAMA_MODEL")
    curl -s http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d "$warmup_payload" >/dev/null 2>&1 || true
fi

echo "Setup complete, keeping container running..."
wait $OLLAMA_PID
