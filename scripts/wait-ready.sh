#!/usr/bin/env bash
# Displays service status and waits until all are ready

set -uo pipefail

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SERVICES=("livekit" "whisper" "llama_cpp" "kokoro" "livekit_agent" "frontend")

is_ready() {
  local name="$1"
  local json
  json=$(docker compose ps --format json "$name" 2>/dev/null)
  [[ -z "$json" ]] && return 1
  
  # Check State field - running is good
  if echo "$json" | grep -q '"State":"running"'; then
    # If there's a healthcheck, it must be healthy
    if echo "$json" | grep -q '"Health":"unhealthy"'; then
      return 1
    fi
    return 0
  fi
  return 1
}

clear_lines() {
  for ((i = 0; i < $1; i++)); do
    printf "\033[A\033[K"
  done
}

main() {
  local all_ready=false
  local iteration=0
  local line_count=$((${#SERVICES[@]} + 2))

  echo ""
  echo -e "${BOLD}Waiting for services...${NC}"
  echo ""
  for name in "${SERVICES[@]}"; do
    printf "  ${YELLOW}○${NC} %-14s\n" "$name"
  done
  echo ""

  while [[ "$all_ready" == "false" ]]; do
    sleep 1
    ((iteration++)) || true
    clear_lines "$line_count"

    echo -e "${BOLD}Waiting for services...${NC} (${iteration}s)"
    echo ""

    all_ready=true
    for name in "${SERVICES[@]}"; do
      if is_ready "$name"; then
        printf "  ${GREEN}●${NC} %-14s\n" "$name"
      else
        printf "  ${YELLOW}○${NC} %-14s\n" "$name"
        all_ready=false
      fi
    done
    echo ""
  done

  clear_lines "$line_count"
  echo -e "${BOLD}${GREEN}All services ready!${NC} (${iteration}s)"
  echo ""
  for name in "${SERVICES[@]}"; do
    printf "  ${GREEN}●${NC} %-14s\n" "$name"
  done
  echo ""
  echo -e "Open ${BLUE}http://localhost:3000${NC} to start chatting."
  echo ""
}

main "$@"
