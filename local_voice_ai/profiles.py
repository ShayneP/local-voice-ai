"""Hardware detection and data-driven startup profile resolution.

This module intentionally uses only the Python standard library. The repository
launcher imports it before the project's virtual environment necessarily exists.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CATALOG_PATH = Path(__file__).with_name("profiles.json")


@dataclass(frozen=True)
class HardwareInfo:
    system: str
    machine: str
    platform_key: str
    device_name: str
    accelerator: str
    memory_topology: str
    total_memory_gib: float
    accelerator_memory_gib: float | None = None
    jetpack_version: str | None = None
    l4t_version: str | None = None

    @property
    def memory_capacity_gib(self) -> float:
        if self.memory_topology == "discrete" and self.accelerator_memory_gib is not None:
            return self.accelerator_memory_gib
        return self.total_memory_gib

    @property
    def inference_budget_gib(self) -> float:
        """Conservative memory available to the whole inference stack."""
        if self.memory_topology == "discrete" and self.accelerator_memory_gib is not None:
            # Leave room for the display server, driver allocations, and allocator
            # fragmentation. A percentage scales better than a fixed reserve.
            return round(self.accelerator_memory_gib * 0.92, 2)

        # Apple Silicon and Jetson share RAM with the OS. Preserve at least 2 GB
        # and at least 25% of the machine, whichever is greater.
        reserve = max(2.0, self.total_memory_gib * 0.25)
        return round(max(0.0, self.total_memory_gib - reserve), 2)


@dataclass(frozen=True)
class ModelProfile:
    key: str
    rank: int
    label: str
    description: str
    minimum_budget_gib: float
    target_memory_gib: float
    download_gib: float
    llm: str
    stt: str
    tts: str
    environment: Mapping[str, str]
    platforms: tuple[str, ...] = ()

    def supports(self, platform_key: str) -> bool:
        """Return whether this model stack can run on ``platform_key``."""
        return not self.platforms or platform_key in self.platforms


@dataclass(frozen=True)
class PlatformProfile:
    key: str
    label: str
    description: str
    runtime: str
    compose_files: tuple[str, ...]
    max_profile: str | None
    environment: Mapping[str, str]
    container_download_gib: float = 0.0
    disk_overhead_gib: float = 3.0
    supported_jetpack_versions: tuple[str, ...] = ()
    supported_l4t_versions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfileCatalog:
    models: Mapping[str, ModelProfile]
    platforms: Mapping[str, PlatformProfile]

    def ordered_models(self, platform_key: str | None = None) -> list[ModelProfile]:
        models = self.models.values()
        if platform_key is not None:
            models = (model for model in models if model.supports(platform_key))
        return sorted(models, key=lambda profile: profile.rank)


@dataclass(frozen=True)
class ResolvedProfile:
    hardware: HardwareInfo
    platform: PlatformProfile
    model: ModelProfile
    memory_budget_gib: float
    environment: Mapping[str, str]
    automatic: bool
    warning: str | None = None


@dataclass(frozen=True)
class UserSelection:
    profile: str
    detected_platform: str
    memory_budget_gib: float | None = None
    detected_memory_gib: float | None = None


def _string_map(raw: object, field_name: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a table")
    return {str(key): str(value) for key, value in raw.items()}


def _string_tuple(raw: object, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be an array")
    return tuple(str(value) for value in raw)


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> ProfileCatalog:
    with path.open(encoding="utf-8") as source:
        raw = json.load(source)

    models: dict[str, ModelProfile] = {}
    for key, values in raw.get("models", {}).items():
        models[key] = ModelProfile(
            key=key,
            rank=int(values["rank"]),
            label=str(values["label"]),
            description=str(values["description"]),
            minimum_budget_gib=float(values["minimum_budget_gib"]),
            target_memory_gib=float(values["target_memory_gib"]),
            download_gib=float(values["download_gib"]),
            llm=str(values["llm"]),
            stt=str(values["stt"]),
            tts=str(values["tts"]),
            environment=_string_map(values.get("environment"), f"models.{key}.environment"),
            platforms=_string_tuple(values.get("platforms"), f"models.{key}.platforms"),
        )

    platforms: dict[str, PlatformProfile] = {}
    for key, values in raw.get("platforms", {}).items():
        compose_files = values.get("compose_files", [])
        if not isinstance(compose_files, list):
            raise ValueError(f"platforms.{key}.compose_files must be an array")
        platforms[key] = PlatformProfile(
            key=key,
            label=str(values["label"]),
            description=str(values["description"]),
            runtime=str(values["runtime"]),
            compose_files=tuple(str(item) for item in compose_files),
            max_profile=str(values["max_profile"]) if values.get("max_profile") else None,
            environment=_string_map(values.get("environment"), f"platforms.{key}.environment"),
            container_download_gib=float(values.get("container_download_gib", 0.0)),
            disk_overhead_gib=float(values.get("disk_overhead_gib", 3.0)),
            supported_jetpack_versions=_string_tuple(
                values.get("supported_jetpack_versions"),
                f"platforms.{key}.supported_jetpack_versions",
            ),
            supported_l4t_versions=_string_tuple(
                values.get("supported_l4t_versions"),
                f"platforms.{key}.supported_l4t_versions",
            ),
        )

    if not models or not platforms:
        raise ValueError("profile catalog must define models and platforms")
    return ProfileCatalog(models=models, platforms=platforms)


def _default_command_output(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _default_read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return None


def _total_memory_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass

    # Windows does not provide os.sysconf. Keep ctypes at this narrow boundary
    # so importing the launcher remains portable.
    if platform.system() == "Windows":  # pragma: no cover - exercised on Windows
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [  # noqa: RUF012 - required ctypes class declaration
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
    return 0


def _nvidia_device(raw: str | None) -> tuple[str, float] | None:
    if not raw:
        return None
    first_line = raw.splitlines()[0]
    try:
        name, memory_mib = first_line.rsplit(",", 1)
        return name.strip(), round(float(memory_mib.strip()) / 1024, 2)
    except (ValueError, TypeError):
        return None


def _jetpack_version(raw: str | None) -> str | None:
    if not raw:
        return None
    match = re.search(r"(?<!\d)(\d+\.\d+(?:\.\d+)?)(?!\d)", raw)
    return match.group(1) if match else None


def _l4t_version(raw: str | None) -> str | None:
    if not raw:
        return None
    release = re.search(r"\bR(\d+)\b", raw)
    revision = re.search(r"\bREVISION:\s*([\d.]+)", raw)
    if release is None or revision is None:
        return None
    return f"{release.group(1)}.{revision.group(1)}"


def detect_hardware(
    *,
    system: str | None = None,
    machine: str | None = None,
    total_memory_bytes: int | None = None,
    command_output: Callable[[list[str]], str | None] = _default_command_output,
    read_text: Callable[[str], str | None] = _default_read_text,
) -> HardwareInfo:
    system = system or platform.system()
    machine = machine or platform.machine()
    memory_bytes = _total_memory_bytes() if total_memory_bytes is None else total_memory_bytes
    total_memory_gib = round(memory_bytes / 1024**3, 2)

    device_tree_model = read_text("/proc/device-tree/model") if system == "Linux" else None
    if device_tree_model and "jetson" in device_tree_model.lower():
        platform_key = "jetson-orin-nano" if "orin nano" in device_tree_model.lower() else "jetson"
        jetpack_version = _jetpack_version(command_output(["dpkg-query", "-W", "nvidia-jetpack"]))
        l4t_version = _l4t_version(read_text("/etc/nv_tegra_release"))
        return HardwareInfo(
            system=system,
            machine=machine,
            platform_key=platform_key,
            device_name=device_tree_model.strip("\x00"),
            accelerator="cuda",
            memory_topology="shared",
            total_memory_gib=total_memory_gib,
            jetpack_version=jetpack_version,
            l4t_version=l4t_version,
        )

    if system == "Darwin" and machine.lower() in {"arm64", "aarch64"}:
        return HardwareInfo(
            system=system,
            machine=machine,
            platform_key="apple-silicon",
            device_name="Apple Silicon",
            accelerator="metal",
            memory_topology="shared",
            total_memory_gib=total_memory_gib,
        )

    nvidia = _nvidia_device(
        command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    if nvidia is not None:
        device_name, accelerator_memory_gib = nvidia
        return HardwareInfo(
            system=system,
            machine=machine,
            platform_key="nvidia-cuda",
            device_name=device_name,
            accelerator="cuda",
            memory_topology="discrete",
            total_memory_gib=total_memory_gib,
            accelerator_memory_gib=accelerator_memory_gib,
        )

    return HardwareInfo(
        system=system,
        machine=machine,
        platform_key="linux-cpu",
        device_name=f"{machine or 'generic'} CPU",
        accelerator="cpu",
        memory_topology="system",
        total_memory_gib=total_memory_gib,
    )


def resolve_profile(
    catalog: ProfileCatalog,
    hardware: HardwareInfo,
    *,
    requested: str = "auto",
    memory_budget_gib: float | None = None,
) -> ResolvedProfile:
    try:
        platform_profile = catalog.platforms[hardware.platform_key]
    except KeyError as exc:
        raise ValueError(f"no platform profile for {hardware.platform_key!r}") from exc

    budget = hardware.inference_budget_gib if memory_budget_gib is None else memory_budget_gib
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("memory budget must be a finite number greater than zero")

    ordered = catalog.ordered_models(hardware.platform_key)
    if not ordered:
        raise ValueError(f"no model profiles support {hardware.platform_key!r}")
    automatic = requested == "auto"
    if automatic:
        eligible = [profile for profile in ordered if profile.minimum_budget_gib <= budget]
        model = eligible[-1] if eligible else ordered[0]
        if platform_profile.max_profile:
            cap = catalog.models[platform_profile.max_profile]
            if model.rank > cap.rank:
                model = cap
    else:
        try:
            model = catalog.models[requested]
        except KeyError as exc:
            choices = ", ".join(catalog.models)
            raise ValueError(f"unknown profile {requested!r}; choose one of: {choices}") from exc
        if not model.supports(hardware.platform_key):
            raise ValueError(
                f"profile {requested!r} does not support {hardware.platform_key!r}"
            )

    warnings: list[str] = []
    if model.minimum_budget_gib > budget:
        warnings.append(
            f"{model.label} exceeds the {budget:.1f} GB budget "
            f"(needs about {model.minimum_budget_gib:.1f} GB)"
        )
    if not automatic and platform_profile.max_profile:
        cap = catalog.models[platform_profile.max_profile]
        if model.rank > cap.rank:
            warnings.append(f"{platform_profile.label} normally recommends at most {cap.label}")

    environment = dict(model.environment)
    environment.update(platform_profile.environment)
    return ResolvedProfile(
        hardware=hardware,
        platform=platform_profile,
        model=model,
        memory_budget_gib=round(budget, 2),
        environment=environment,
        automatic=automatic,
        warning="; ".join(warnings) or None,
    )


def save_selection(path: Path, selection: UserSelection) -> None:
    lines = [
        "version = 1",
        f"profile = {json.dumps(selection.profile)}",
        f"detected_platform = {json.dumps(selection.detected_platform)}",
    ]
    if selection.detected_memory_gib is not None:
        lines.append(f"detected_memory_gib = {selection.detected_memory_gib:g}")
    if selection.memory_budget_gib is not None:
        lines.append(f"memory_budget_gib = {selection.memory_budget_gib:g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_selection(path: Path) -> UserSelection | None:
    if not path.is_file():
        return None
    raw: dict[str, object] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid selection line {line_number} in {path}")
        key, encoded_value = (part.strip() for part in line.split("=", 1))
        try:
            raw[key] = json.loads(encoded_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid selection value on line {line_number} in {path}") from exc
    if int(raw.get("version", 0)) != 1:
        raise ValueError(f"unsupported selection version in {path}")
    memory = raw.get("memory_budget_gib")
    detected_memory = raw.get("detected_memory_gib")
    return UserSelection(
        profile=str(raw["profile"]),
        detected_platform=str(raw["detected_platform"]),
        memory_budget_gib=float(memory) if memory is not None else None,
        detected_memory_gib=float(detected_memory) if detected_memory is not None else None,
    )
