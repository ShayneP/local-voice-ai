#!/bin/sh
set -e

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

if ! curl -s http://localhost:11434/api/tags | grep -q "gemma3:4b"; then
    echo "Downloading gemma3:4b model..."
    ollama pull gemma3:4b
fi

echo "Setup complete, keeping container running..."
wait $OLLAMA_PID