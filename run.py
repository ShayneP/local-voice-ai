#!/usr/bin/env python3
"""Configure and run Local Voice Agent with a hardware-aware profile.

Examples:
    python run.py                         # first-run wizard, then start
    python run.py configure               # revisit the wizard
    python run.py plan                    # show the resolved plan
    python run.py start --profile auto --yes
    python run.py start --profile compact --memory-gb 5.5 --yes
    python run.py client --server 192.168.1.40
    python run.py status
    python run.py logs
    python run.py down

The launcher and its profile engine use only the Python standard library, so
the setup decision happens before project dependencies or model weights exist.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

from local_voice_ai.launcher import choose_profile, render_plan
from local_voice_ai.native_runtime import install_runtime
from local_voice_ai.profiles import (
    DEFAULT_CATALOG_PATH,
    HardwareInfo,
    ProfileCatalog,
    ResolvedProfile,
    UserSelection,
    detect_hardware,
    load_catalog,
    load_selection,
    resolve_profile,
    save_selection,
)

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
SELECTION_PATH = ROOT / ".local-voice-ai.toml"
LOCAL_ENV_PATH = ROOT / ".env.local"
NATIVE_RUNTIME_DIR = ROOT / ".local-voice-ai" / "runtime"


def read_env_file(path: Path) -> dict[str, str]:
    """Read the simple KEY=VALUE subset used by this project's env files."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _client_backend_url(value: str) -> str:
    """Normalize a client server argument without accepting embedded credentials."""
    value = value.strip()
    if not value:
        raise ValueError("the server address is empty")

    had_scheme = "://" in value
    candidate = value if had_scheme else f"http://{value}"
    try:
        parsed = urllib.parse.urlsplit(candidate)
        # Accessing port validates malformed values such as ':not-a-port'.
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid server address: {value}") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("the server must be an HTTP or HTTPS host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("the server address must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("the server address must not contain a query or fragment")

    netloc = parsed.netloc
    if not had_scheme and port is None:
        netloc += ":8080"
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit((parsed.scheme, netloc, path, "", ""))


def _client_api_url(backend_url: str, path: str) -> str:
    return f"{backend_url.rstrip('/')}/{path.lstrip('/')}"


def _client_connection_details(backend_url: str) -> dict[str, object]:
    request = urllib.request.Request(
        _client_api_url(backend_url, "/api/connection-details"),
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError("the server returned invalid connection details")
    return data


def _is_loopback_host(host: str | None) -> bool:
    return host in {"localhost", "127.0.0.1", "::1"}


def _client(server: str, port: int) -> int:
    try:
        backend_url = _client_backend_url(server)
        details = _client_connection_details(backend_url)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"Cannot connect to Local Voice AI at {server}: {exc}")
        return 1

    advertised_url = details.get("serverUrl")
    try:
        advertised_host = urllib.parse.urlsplit(str(advertised_url)).hostname
        backend_host = urllib.parse.urlsplit(backend_url).hostname
    except ValueError:
        advertised_host = None
        backend_host = None
    if not _is_loopback_host(backend_host) and _is_loopback_host(advertised_host):
        print(
            "The server is reachable, but LiveKit advertises a loopback address.\n"
            f"On {backend_host}, set LIVEKIT_URL and LIVEKIT_NODE_IP to that machine's "
            "LAN address, then restart it."
        )
        return 1

    pnpm = shutil.which("pnpm")
    if pnpm is None:
        print("pnpm was not found. Install Node.js and enable Corepack, then try again.")
        return 1

    environment = dict(os.environ)
    environment["NEXT_PUBLIC_BACKEND_URL"] = backend_url
    if not (FRONTEND / "node_modules").is_dir():
        print("Installing frontend dependencies on this computer…")
        installed = _run(
            [pnpm, "--dir", str(FRONTEND), "install", "--frozen-lockfile"],
            environment=environment,
        )
        if installed.returncode != 0:
            return installed.returncode

    print(f"Connecting this computer to {backend_url}")
    print(f"Open http://localhost:{port} and allow microphone access.\n")
    try:
        return _run(
            [
                pnpm,
                "--dir",
                str(FRONTEND),
                "dev",
                "--hostname",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            environment=environment,
        ).returncode
    except KeyboardInterrupt:
        print("\nClient stopped.")
        return 0


def runtime_environment(
    profile: ResolvedProfile,
    *,
    local_overrides: Mapping[str, str] | None = None,
    shell_environment: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Merge profile defaults with intentional per-machine and shell overrides.

    Precedence is shell > .env.local > profile. This keeps one-off commands and
    existing custom configuration authoritative without mutating either file.
    """
    profile_values = dict(profile.environment)
    environment = dict(profile_values)
    environment.update(local_overrides or {})
    environment.update(os.environ if shell_environment is None else shell_environment)
    overrides = {
        key: environment[key]
        for key, value in profile_values.items()
        if environment.get(key) != value
    }
    return environment, overrides


def compose_command(profile: ResolvedProfile, *arguments: str) -> list[str]:
    command = ["docker", "compose"]
    for compose_file in profile.platform.compose_files:
        command.extend(["-f", compose_file])
    command.extend(arguments)
    return command


def selection_for(
    profile: ResolvedProfile,
    *,
    memory_budget_was_set: bool = False,
) -> UserSelection:
    return UserSelection(
        profile=profile.model.key,
        detected_platform=profile.hardware.platform_key,
        memory_budget_gib=profile.memory_budget_gib if memory_budget_was_set else None,
        detected_memory_gib=profile.hardware.memory_capacity_gib,
    )


def _run(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    capture: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(environment) if environment is not None else None,
        check=False,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _native_python() -> Path:
    candidates = [
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    return next(
        (candidate for candidate in candidates if candidate.is_file()), Path(sys.executable)
    )


def _native_dependency_error(python: Path, stt_provider: str = "nemotron-cpp") -> str | None:
    imports = "import uvicorn, dotenv, torch, kokoro; import livekit.agents"
    if stt_provider == "nemotron":
        imports += "; import nemo.collections.asr"
    elif stt_provider == "whisper":
        imports += "; import faster_whisper"
    probe = _run(
        [
            str(python),
            "-c",
            imports,
        ],
        capture=True,
        timeout=60,
    )
    if probe.returncode == 0:
        return None
    detail = (probe.stderr or probe.stdout).strip().splitlines()
    suffix = f" ({detail[-1]})" if detail else ""
    return f"native Python dependencies are incomplete{suffix}"


def _native_python_version_error(python: Path) -> str | None:
    probe = _run(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        capture=True,
        timeout=10,
    )
    version = probe.stdout.strip()
    try:
        major, minor = (int(part) for part in version.split(".", 1))
    except ValueError:
        return f"could not determine the Python version for {python}"
    if (major, minor) < (3, 11) or (major, minor) >= (3, 14):
        return (
            f"the application runtime requires Python 3.11-3.13; "
            f"{python} is Python {version} (it can run the setup launcher only)"
        )
    return None


def prepare_native_stt(
    profile: ResolvedProfile,
    environment: dict[str, str],
) -> str | None:
    """Install the pinned native STT runtime when a native profile needs it."""
    if (
        profile.platform.runtime != "native"
        or environment.get("STT_PROVIDER", "nemotron-cpp") != "nemotron-cpp"
    ):
        return None

    configured = environment.get("NEMO_SPEECH_BIN")
    search_path = environment.get("PATH")
    if configured:
        existing = shutil.which(configured, path=search_path)
        if existing is None:
            return f"configured NEMO_SPEECH_BIN was not found: {configured}"
        environment["NEMO_SPEECH_BIN"] = existing
        return None

    print("Installing the native streaming speech runtime…")
    try:
        binary = install_runtime(
            NATIVE_RUNTIME_DIR,
            system=profile.hardware.system,
            machine=profile.hardware.machine,
            backend=profile.hardware.accelerator,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return f"could not install the native streaming speech runtime: {exc}"

    environment["NEMO_SPEECH_BIN"] = str(binary)
    environment["PATH"] = os.pathsep.join(
        part for part in (str(binary.parent), search_path or "") if part
    )
    return None


def _matches_release(version: str, supported: Sequence[str]) -> bool:
    return any(version == release or version.startswith(release + ".") for release in supported)


def _docker_failure_reason(stderr: str) -> str:
    """Explain why ``docker info`` failed, in terms of what to fix.

    A running daemon that refuses this user looks identical to a stopped one
    unless the message is read, and "the daemon is not reachable" sends people
    off restarting a service that was never down.
    """
    lowered = (stderr or "").lower()
    if "permission denied" in lowered:
        return (
            "Docker is running but this user cannot reach it. Add yourself to "
            "the 'docker' group (sudo usermod -aG docker $USER), then log out "
            "and back in"
        )
    if "is the docker daemon running" in lowered or "cannot connect" in lowered:
        return "the Docker daemon is not running. Start it (sudo systemctl start docker)"
    return "the Docker daemon is not reachable"


def _docker_runtime_names(raw: str) -> set[str]:
    try:
        runtimes = json.loads(raw)
    except json.JSONDecodeError:
        return set()
    if not isinstance(runtimes, dict):
        return set()
    return {str(name) for name in runtimes}


def jetson_container_errors(
    profile: ResolvedProfile,
    docker_runtimes: set[str],
) -> list[str]:
    """Validate the host facts on which the pinned Jetson image depends."""
    if not profile.hardware.platform_key.startswith("jetson"):
        return []

    hardware = profile.hardware
    platform_profile = profile.platform
    errors: list[str] = []
    if hardware.jetpack_version is None and hardware.l4t_version is None:
        errors.append(
            "the JetPack/L4T release could not be detected from nvidia-jetpack or "
            "/etc/nv_tegra_release"
        )
    if (
        hardware.jetpack_version is not None
        and platform_profile.supported_jetpack_versions
        and not _matches_release(
            hardware.jetpack_version,
            platform_profile.supported_jetpack_versions,
        )
    ):
        supported = ", ".join(platform_profile.supported_jetpack_versions)
        errors.append(
            f"JetPack {hardware.jetpack_version} is unsupported by the Jetson image. "
            f"The image supports {supported}"
        )
    if (
        hardware.l4t_version is not None
        and platform_profile.supported_l4t_versions
        and not _matches_release(
            hardware.l4t_version,
            platform_profile.supported_l4t_versions,
        )
    ):
        supported = ", ".join(platform_profile.supported_l4t_versions)
        errors.append(
            f"L4T {hardware.l4t_version} is unsupported by the Jetson image. "
            f"The image supports {supported}"
        )
    if "nvidia" not in docker_runtimes:
        errors.append(
            "Docker's nvidia runtime is unavailable. Install or repair nvidia-container-runtime"
        )
    return errors


def preflight(profile: ResolvedProfile, environment: Mapping[str, str]) -> list[str]:
    """Return actionable startup blockers without changing the machine."""
    errors: list[str] = []
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    required_disk_gib = profile.model.download_gib + profile.platform.disk_overhead_gib
    if free_gib < required_disk_gib:
        errors.append(
            f"only {free_gib:.1f} GB disk space is free; allow at least "
            f"{required_disk_gib:.1f} GB for weights and build overhead"
        )

    if profile.platform.runtime == "docker":
        if shutil.which("docker") is None:
            errors.append("Docker was not found on PATH")
            return errors
        try:
            docker_info = _run(
                ["docker", "info", "--format", "{{json .Runtimes}}"],
                capture=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            errors.append("Docker did not respond within 15 seconds")
        else:
            if docker_info.returncode != 0:
                errors.append(_docker_failure_reason(docker_info.stderr))
            elif profile.hardware.platform_key.startswith("jetson"):
                errors.extend(
                    jetson_container_errors(
                        profile,
                        _docker_runtime_names(docker_info.stdout),
                    )
                )
        try:
            compose_version = _run(["docker", "compose", "version"], capture=True, timeout=15)
        except subprocess.TimeoutExpired:
            errors.append("Docker Compose did not respond within 15 seconds")
        else:
            if compose_version.returncode != 0:
                errors.append("the Docker Compose plugin is unavailable")
        if profile.hardware.platform_key == "nvidia-cuda" and shutil.which("nvidia-smi") is None:
            errors.append("nvidia-smi is unavailable; install the NVIDIA driver first")
        return errors

    python = _native_python()
    version_error = _native_python_version_error(python)
    dependency_error: str | None = None
    if version_error:
        errors.append(version_error)
    else:
        dependency_error = _native_dependency_error(
            python,
            environment.get("STT_PROVIDER", "nemotron-cpp"),
        )
        if dependency_error:
            if profile.hardware.platform_key.startswith("jetson"):
                errors.append(
                    dependency_error
                    + "; install the NVIDIA PyTorch build matched to this JetPack release, "
                    "then install the project dependencies without replacing it"
                )
            else:
                errors.append(dependency_error + "; run `uv sync --extra ml --extra dev` first")

    search_path = environment.get("PATH")
    binaries = ["livekit-server", "llama-server"]
    if environment.get("STT_PROVIDER", "nemotron-cpp") == "nemotron-cpp":
        binaries.append(environment.get("NEMO_SPEECH_BIN", "nemo-speech"))
    for binary in binaries:
        if shutil.which(binary, path=search_path) is None:
            errors.append(f"{binary} was not found on PATH")

    accelerator_probe: list[str] | None = None
    accelerator_label = ""
    if profile.hardware.platform_key.startswith("jetson"):
        accelerator_probe = [
            str(python),
            "-c",
            "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
        ]
        accelerator_label = "JetPack-matched CUDA PyTorch"
    elif profile.hardware.platform_key == "apple-silicon":
        accelerator_probe = [
            str(python),
            "-c",
            "import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)",
        ]
        accelerator_label = "Metal-enabled PyTorch"

    if accelerator_probe is not None and version_error is None and dependency_error is None:
        try:
            result = _run(accelerator_probe, capture=True, timeout=30)
        except subprocess.TimeoutExpired:
            errors.append(f"{accelerator_label} probe timed out")
        else:
            if result.returncode != 0:
                errors.append(f"{accelerator_label} is not available in {python}")
    return errors


def _status_url(environment: Mapping[str, str]) -> str:
    return f"http://127.0.0.1:{environment.get('WEB_PORT', '8080')}/api/status"


def _fetch_status(environment: Mapping[str, str]) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(_status_url(environment), timeout=2) as response:
            data = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return data if isinstance(data, dict) else None


def _status_line(status: Mapping[str, object]) -> str:
    children = status.get("children")
    if not isinstance(children, list):
        return "starting"
    parts: list[str] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        marker = "ready" if child.get("ready") else "starting"
        detail = f" · {child['detail']}" if child.get("detail") else ""
        parts.append(f"{child.get('name', 'service')} {marker}{detail}")
    return " | ".join(parts) or "starting"


def _wait_until_ready(environment: Mapping[str, str], timeout: float = 1200) -> int:
    print("Waiting for the models and services to become ready…")
    deadline = time.monotonic() + timeout
    previous = ""
    try:
        while time.monotonic() < deadline:
            status = _fetch_status(environment)
            if status is not None:
                line = _status_line(status)
                if line != previous:
                    print(f"  {line}")
                    previous = line
                if status.get("ready"):
                    port = environment.get("WEB_PORT", "8080")
                    print(f"\nReady: http://localhost:{port}")
                    return 0
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStartup continues in the background. Use `python run.py status` to check it.")
        return 0
    print("Startup is still incomplete after 20 minutes. Use `python run.py logs` for details.")
    return 1


def _start(profile: ResolvedProfile, *, build: bool = True) -> int:
    environment, overrides = runtime_environment(
        profile,
        local_overrides=read_env_file(LOCAL_ENV_PATH),
        shell_environment=os.environ,
    )
    if overrides:
        formatted = ", ".join(f"{key}={value}" for key, value in sorted(overrides.items()))
        print(f"Using local overrides: {formatted}")

    runtime_error = prepare_native_stt(profile, environment)
    if runtime_error:
        print(f"\nCannot start yet:\n  - {runtime_error}")
        return 1

    blockers = preflight(profile, environment)
    if blockers:
        print("\nCannot start yet:")
        for blocker in blockers:
            print(f"  - {blocker}")
        return 1

    if profile.platform.runtime == "docker":
        arguments = ["up", "-d"]
        if build:
            arguments.append("--build")
        result = _run(compose_command(profile, *arguments), environment=environment)
        if result.returncode != 0:
            return result.returncode
        return _wait_until_ready(environment)

    print("Starting the native supervisor. Press Ctrl+C to stop it.\n")
    return _run(
        [str(_native_python()), "-m", "local_voice_ai", "serve"],
        environment=environment,
    ).returncode


def _show_status(profile: ResolvedProfile | None) -> int:
    if profile is not None:
        environment, _ = runtime_environment(
            profile,
            local_overrides=read_env_file(LOCAL_ENV_PATH),
            shell_environment=os.environ,
        )
    else:
        environment = read_env_file(LOCAL_ENV_PATH)
        environment.update(os.environ)

    status = _fetch_status(environment)
    if status is not None:
        print("Ready" if status.get("ready") else "Starting")
        print(_status_line(status))
        return 0

    if shutil.which("docker") is not None:
        result = _run(
            ["docker", "compose", "-f", "docker-compose.yml", "ps"],
            environment=environment,
        )
        if result.returncode == 0:
            return 0
    print("Local Voice Agent is not reachable at " + _status_url(environment))
    return 1


def _down() -> int:
    if shutil.which("docker") is None:
        print("Docker is not installed. Native runs stop when their foreground process exits.")
        return 1
    return _run(["docker", "compose", "-f", "docker-compose.yml", "down"]).returncode


def _logs() -> int:
    if shutil.which("docker") is None:
        print("Native logs are printed by the foreground `python run.py start` process.")
        return 1
    return _run(["docker", "compose", "-f", "docker-compose.yml", "logs", "-f", "app"]).returncode


def _add_profile_arguments(parser: argparse.ArgumentParser, *, include_yes: bool) -> None:
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help="auto, compact, or balanced (default: saved choice or auto)",
    )
    parser.add_argument(
        "--memory-gb",
        type=float,
        help="memory available to the inference stack; overrides the detected budget",
    )
    if include_yes:
        parser.add_argument("--yes", action="store_true", help="accept the automatic choice")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    start = subparsers.add_parser("start", help="configure if needed, then start")
    _add_profile_arguments(start, include_yes=True)
    start.add_argument("--no-build", action="store_true", help="reuse the existing image")

    client = subparsers.add_parser(
        "client", help="run the voice UI here and connect it to another machine"
    )
    client.add_argument(
        "--server",
        required=True,
        metavar="HOST_OR_URL",
        help="server host (uses port 8080) or full HTTP URL",
    )
    client.add_argument("--port", type=int, default=3000, help="local UI port (default: 3000)")

    configure = subparsers.add_parser("configure", help="choose and save a startup profile")
    _add_profile_arguments(configure, include_yes=True)

    plan = subparsers.add_parser("plan", help="show the detected hardware and selected models")
    _add_profile_arguments(plan, include_yes=False)

    subparsers.add_parser("status", help="show per-service readiness")
    subparsers.add_parser("logs", help="follow Docker supervisor logs")
    subparsers.add_parser("down", help="stop the Docker stack")
    return parser


def _saved_profile(
    catalog: ProfileCatalog,
    hardware: HardwareInfo,
    selection: UserSelection | None,
) -> ResolvedProfile | None:
    if selection is None or selection.detected_platform != hardware.platform_key:
        return None
    if (
        selection.detected_memory_gib is not None
        and abs(selection.detected_memory_gib - hardware.memory_capacity_gib) > 0.25
    ):
        return None
    return resolve_profile(
        catalog,
        hardware,
        requested=selection.profile,
        memory_budget_gib=selection.memory_budget_gib,
    )


def _select_profile(
    args: argparse.Namespace,
    catalog: ProfileCatalog,
    hardware: HardwareInfo,
    saved: UserSelection | None,
) -> tuple[ResolvedProfile | None, bool, bool]:
    """Return profile, whether it was shown, and whether memory was explicit."""
    requested = getattr(args, "profile", None)
    memory = getattr(args, "memory_gb", None)
    if requested is not None:
        profile = resolve_profile(
            catalog,
            hardware,
            requested=requested,
            memory_budget_gib=memory,
        )
        return profile, False, memory is not None

    if args.command in {"start", "plan"}:
        saved_profile = _saved_profile(catalog, hardware, saved)
        if saved_profile is not None and memory is None:
            return saved_profile, False, saved.memory_budget_gib is not None

    if args.command == "plan" or getattr(args, "yes", False) or not sys.stdin.isatty():
        return (
            resolve_profile(catalog, hardware, memory_budget_gib=memory),
            False,
            memory is not None,
        )

    if saved is not None and saved.detected_platform != hardware.platform_key:
        print(
            f"Hardware changed from {saved.detected_platform} to {hardware.platform_key}; "
            "choosing a new profile."
        )
    elif (
        saved is not None
        and saved.detected_memory_gib is not None
        and abs(saved.detected_memory_gib - hardware.memory_capacity_gib) > 0.25
    ):
        print(
            f"Available hardware memory changed from {saved.detected_memory_gib:.1f} GB "
            f"to {hardware.memory_capacity_gib:.1f} GB; choosing a new profile."
        )
    profile = choose_profile(catalog, hardware, memory_budget_gib=memory)
    memory_was_set = bool(
        profile and abs(profile.memory_budget_gib - hardware.inference_budget_gib) > 0.01
    )
    return profile, True, memory_was_set


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        raw_argv = ["start"]
    parser = _parser()
    args = parser.parse_args(raw_argv)

    if args.command == "down":
        return _down()
    if args.command == "logs":
        return _logs()
    if args.command == "client":
        return _client(args.server, args.port)

    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    hardware = detect_hardware()
    try:
        saved = load_selection(SELECTION_PATH)
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read {SELECTION_PATH.name}: {exc}")

    saved_profile = _saved_profile(catalog, hardware, saved)
    if args.command == "status":
        return _show_status(saved_profile)

    try:
        profile, was_shown, memory_was_set = _select_profile(args, catalog, hardware, saved)
    except ValueError as exc:
        parser.error(str(exc))
    if profile is None:
        return 0

    if not was_shown:
        print(render_plan(profile))

    if args.command == "plan":
        return 0

    save_selection(
        SELECTION_PATH,
        selection_for(profile, memory_budget_was_set=memory_was_set),
    )
    print(f"\nSaved {profile.model.label} to {SELECTION_PATH.name}.")
    if args.command == "configure":
        return 0

    return _start(profile, build=not args.no_build)


if __name__ == "__main__":
    raise SystemExit(main())
