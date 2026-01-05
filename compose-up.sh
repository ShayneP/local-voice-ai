#!/usr/bin/env bash
set -euo pipefail

mode=""
if [[ $# -gt 0 ]]; then
  case "$1" in
    cpu|gpu)
      mode="$1"
      shift
      ;;
  esac
fi

if [[ -z "$mode" ]]; then
  echo "Select target:"
  echo "  1) CPU"
  echo "  2) GPU"
  read -r -p "Enter choice (1/2): " choice
  case "$choice" in
    1) mode="cpu" ;;
    2) mode="gpu" ;;
    *) echo "Invalid choice. Use 1 for CPU or 2 for GPU." >&2; exit 1 ;;
  esac
fi

# macOS-specific settings
if [[ "$(uname -s)" == "Darwin" ]]; then
  export LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-16384}"
fi

compose_files=(-f docker-compose.yml)
if [[ "$mode" == "gpu" ]]; then
  compose_files+=(-f docker-compose.gpu.yml)
elif [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  compose_files+=(-f docker-compose.macos.yml)
fi

echo "Running: docker compose ${compose_files[*]} up $*"
docker compose "${compose_files[@]}" up "$@"
