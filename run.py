#!/usr/bin/env python3
"""Interactive launcher for the local-voice-ai stack.

Picks a deployment profile (CPU / GPU / Mac-Metal) and runs the right
``docker compose`` invocation, so nobody has to remember the
``-f docker-compose.yml -f docker-compose.gpu.yml ... up -d --build`` incantation.

Usage:
    python run.py            # interactive menu
    python run.py gpu        # start the GPU (NVIDIA/CUDA) Gemma stack
    python run.py cpu        # start the CPU baseline (qwen3-4b + STT)
    python run.py mac        # start the Mac/Metal half-cascade
    python run.py status     # show what's running (incl. GPU vs CPU)
    python run.py down        # stop everything

Stdlib only -no pip install needed, runs the same on Windows and macOS.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Profile:
    def __init__(self, key: str, label: str, files: list[str], blurb: str, native_hint: str = "") -> None:
        self.key = key
        self.label = label
        self.files = files  # compose -f args, in order
        self.blurb = blurb
        self.native_hint = native_hint  # printed before start if set


PROFILES: dict[str, Profile] = {
    "gpu": Profile(
        "gpu",
        "GPU (NVIDIA/CUDA)",
        ["docker-compose.yml", "docker-compose.gpu.yml"],
        "Gemma 12B audio half-cascade, fully on the GPU",
    ),
    "cpu": Profile(
        "cpu",
        "CPU (default)",
        ["docker-compose.yml"],
        "qwen3-4b + Nemotron STT, runs anywhere",
    ),
    "mac": Profile(
        "mac",
        "Mac (Metal / native LLM)",
        ["docker-compose.yml", "docker-compose.audio.yml"],
        "Gemma 12B on the host via Metal, rest in containers",
        native_hint="Start the native LLM first in another terminal:\n    ./scripts/run-native-audio-llm.sh",
    ),
}


# --- shelling out ---------------------------------------------------------
def _compose(files: list[str], *args: str, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose"]
    for f in files:
        cmd += ["-f", f]
    cmd += list(args)
    return subprocess.run(
        cmd, cwd=ROOT, text=True,
        capture_output=capture,
        # Docker output is UTF-8; don't let Windows' cp1252 default crash the read.
        encoding="utf-8", errors="replace",
    )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Capture a command's text output, tolerant of non-cp1252 bytes on Windows."""
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _require_docker() -> bool:
    if shutil.which("docker") is None:
        print("error: `docker` not found on PATH. Install Docker Desktop / Engine first.")
        return False
    probe = _run(["docker", "info"])
    if probe.returncode != 0:
        print("error: the Docker daemon isn't reachable. Is Docker Desktop running?")
        return False
    return True


# --- hardware detection ---------------------------------------------------
def recommended_profile() -> str:
    if platform.system() == "Darwin":
        return "mac"
    if shutil.which("nvidia-smi"):
        probe = _run(["nvidia-smi", "-L"])
        if probe.returncode == 0 and "GPU" in probe.stdout:
            return "gpu"
    return "cpu"


# --- actions --------------------------------------------------------------
def start(profile: Profile) -> int:
    if not _require_docker():
        return 1
    if profile.native_hint:
        print(f"\nNOTE: {profile.native_hint}\n")
    print(f"Starting profile '{profile.key}' ({profile.label})...")
    print("(using --build; Docker's cache makes this near-instant when nothing changed)\n")
    rc = _compose(profile.files, "up", "-d", "--build").returncode
    if rc == 0:
        print("\nStack is up. Check it with:  python run.py status")
        print("Open the app at:  http://localhost:8080   (use localhost, not 0.0.0.0)")
    return rc


def down() -> int:
    if not _require_docker():
        return 1
    print("Stopping the stack...")
    # The base file is enough to resolve the project and tear down every
    # container, regardless of which profile started it.
    return _compose(["docker-compose.yml"], "down").returncode


def status() -> int:
    if not _require_docker():
        return 1
    ps = _compose(["docker-compose.yml"], "ps", capture=True)
    print(ps.stdout.strip() or "No containers for this project are running.")

    # Report the inference backend from the container's resolved env -the
    # deterministic source of whether you're on GPU/Gemma, CPU/qwen, or a native
    # host LLM. (Reading it from env beats grepping hours of logs.)
    cid = _compose(["docker-compose.yml"], "ps", "-q", "app", capture=True).stdout.strip()
    print("\n--- inference backend ---")
    if not cid:
        print("app container: not running.")
        return ps.returncode
    env = _app_env(cid)
    base = env.get("LLAMA_BASE_URL", "")
    audio = env.get("LLM_AUDIO_INPUT", "").strip().lower() in {"1", "true", "yes", "on"}
    layers = int(env.get("LLAMA_N_GPU_LAYERS", "0") or 0)
    if base and not any(h in base for h in ("127.0.0.1", "localhost", "0.0.0.0")):
        # Non-loopback base URL -> the supervisor doesn't manage llama; the LLM
        # runs elsewhere (e.g. the native Metal build on the Mac host).
        print(f"model: native/external LLM at {base}  (e.g. Metal host build)")
    else:
        model = env.get("AUDIO_LLM_ALIAS", "gemma-4-audio") if audio else env.get("LLAMA_MODEL", "qwen3-4b")
        mode = "GPU" if layers > 0 else "CPU"
        print(f"model: {model}   gpu_layers: {layers}   ->  {mode}")
    # VRAM, when an NVIDIA GPU is present (memory is reliable even when SM% isn't).
    if shutil.which("nvidia-smi"):
        smi = _run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"])
        if smi.returncode == 0:
            print(f"GPU memory: {smi.stdout.strip()}")
    return ps.returncode


def _app_env(cid: str) -> dict[str, str]:
    """Resolved environment of the app container, as a dict."""
    out = _run(["docker", "inspect", "-f", "{{range .Config.Env}}{{println .}}{{end}}", cid])
    env: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


# --- menu -----------------------------------------------------------------
def menu() -> int:
    rec = recommended_profile()
    order = ["gpu", "cpu", "mac"]
    print("\nlocal-voice-ai launcher\n")
    items: list[tuple[str, str]] = []
    for key in order:
        p = PROFILES[key]
        tag = "  (recommended for this machine)" if key == rec else ""
        items.append((key, f"Start: {p.label} - {p.blurb}{tag}"))
    items.append(("status", "Status - what's running (GPU vs CPU)"))
    items.append(("down", "Stop everything"))
    items.append(("quit", "Quit"))

    for i, (_, label) in enumerate(items, 1):
        print(f"  {i}. {label}")
    print()

    try:
        raw = input(f"Pick [1-{len(items)}] (default {order.index(rec) + 1} = {rec}): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    if not raw:
        choice = rec
    else:
        if not raw.isdigit() or not (1 <= int(raw) <= len(items)):
            print("Not a valid choice.")
            return 1
        choice = items[int(raw) - 1][0]

    if choice == "quit":
        return 0
    if choice == "status":
        return status()
    if choice == "down":
        return down()
    return start(PROFILES[choice])


def main(argv: list[str]) -> int:
    if not argv:
        return menu()
    cmd = argv[0].lower()
    if cmd in ("down", "stop"):
        return down()
    if cmd in ("status", "ps"):
        return status()
    if cmd in PROFILES:
        return start(PROFILES[cmd])
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    print(f"unknown command: {cmd!r}\n")
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
