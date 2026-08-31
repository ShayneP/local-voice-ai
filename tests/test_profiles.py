"""Hardware detection and startup profile selection behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_voice_ai.profiles import (
    DEFAULT_CATALOG_PATH,
    HardwareInfo,
    UserSelection,
    detect_hardware,
    load_catalog,
    load_selection,
    resolve_profile,
    save_selection,
)

GiB = 1024**3


def _commands(outputs: dict[str, str]):
    def command_output(argv: list[str]) -> str | None:
        return outputs.get(" ".join(argv))

    return command_output


def _files(contents: dict[str, str]):
    def read_text(path: str) -> str | None:
        return contents.get(path)

    return read_text


class TestHardwareDetection:
    def test_jetson_orin_nano_is_shared_cuda_memory(self) -> None:
        hardware = detect_hardware(
            system="Linux",
            machine="aarch64",
            total_memory_bytes=8 * GiB,
            command_output=_commands({"dpkg-query -W nvidia-jetpack": "nvidia-jetpack\t6.2.1+b38"}),
            read_text=_files(
                {
                    "/proc/device-tree/model": "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit\x00",
                    "/etc/nv_tegra_release": (
                        "# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, "
                        "EABI: aarch64"
                    ),
                }
            ),
        )

        assert hardware.platform_key == "jetson-orin-nano"
        assert hardware.accelerator == "cuda"
        assert hardware.memory_topology == "shared"
        assert hardware.total_memory_gib == 8.0
        assert hardware.inference_budget_gib == 6.0
        assert hardware.jetpack_version == "6.2.1"
        assert hardware.l4t_version == "36.4.3"

    def test_apple_silicon_is_shared_metal_memory(self) -> None:
        hardware = detect_hardware(
            system="Darwin",
            machine="arm64",
            total_memory_bytes=16 * GiB,
            command_output=_commands({}),
            read_text=_files({}),
        )

        assert hardware.platform_key == "apple-silicon"
        assert hardware.accelerator == "metal"
        assert hardware.memory_topology == "shared"
        assert hardware.inference_budget_gib == 12.0

    def test_desktop_nvidia_uses_reported_vram(self) -> None:
        hardware = detect_hardware(
            system="Linux",
            machine="x86_64",
            total_memory_bytes=32 * GiB,
            command_output=_commands(
                {
                    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits": (
                        "NVIDIA GeForce RTX 3060, 12288\n"
                    ),
                }
            ),
            read_text=_files({}),
        )

        assert hardware.platform_key == "nvidia-cuda"
        assert hardware.memory_topology == "discrete"
        assert hardware.accelerator_memory_gib == 12.0
        assert hardware.inference_budget_gib == pytest.approx(11.04)

    def test_cpu_fallback_does_not_claim_accelerator_memory(self) -> None:
        hardware = detect_hardware(
            system="Linux",
            machine="x86_64",
            total_memory_bytes=16 * GiB,
            command_output=_commands({}),
            read_text=_files({}),
        )

        assert hardware.platform_key == "linux-cpu"
        assert hardware.accelerator == "cpu"
        assert hardware.memory_topology == "system"
        assert hardware.accelerator_memory_gib is None


class TestProfileResolution:
    @pytest.fixture
    def catalog(self):
        return load_catalog(DEFAULT_CATALOG_PATH)

    def test_catalog_uses_python_310_compatible_json(self) -> None:
        assert DEFAULT_CATALOG_PATH.suffix == ".json"

    def test_eight_gb_shared_memory_recommends_realtime_jetson(self, catalog) -> None:
        hardware = HardwareInfo(
            system="Linux",
            machine="aarch64",
            platform_key="jetson-orin-nano",
            device_name="Jetson Orin Nano",
            accelerator="cuda",
            memory_topology="shared",
            total_memory_gib=8.0,
        )

        resolved = resolve_profile(catalog, hardware)

        assert resolved.model.key == "jetson-realtime"
        assert resolved.memory_budget_gib == 6.0
        assert resolved.environment["DEVICE"] == "cuda"
        assert resolved.environment["LLAMA_CTX_SIZE"] == "4096"
        assert resolved.environment["LLAMA_PARALLEL"] == "1"
        assert resolved.environment["LLAMA_MODEL"] == "qwen3-1.7b"
        assert resolved.model.stt == "Nemotron Speech 0.6B Q8 (native streaming)"
        assert resolved.environment["LLAMA_HF_REPO"] == ("unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL")
        assert resolved.environment["JETSON_LLAMA_BASE_URL"] == ("http://127.0.0.1:11435/v1")
        assert resolved.environment["JETSON_LLAMA_BIND_PORT"] == "11435"
        assert resolved.environment["SEQUENTIAL_STARTUP"] == "1"
        assert resolved.environment["TTS_PROVIDER"] == "kokoro-onnx"
        assert resolved.environment["TURN_DETECTION"] == "vad"
        assert resolved.environment["AGENT_IDLE_PROCESSES"] == "1"
        assert resolved.environment["STT_PROVIDER"] == "nemotron-cpp"
        assert resolved.platform.runtime == "docker"
        assert resolved.platform.compose_files == (
            "docker-compose.yml",
            "docker-compose.jetson.yml",
        )
        assert resolved.platform.supported_jetpack_versions == ("6.2",)
        assert resolved.platform.supported_l4t_versions == ("36.4",)

    def test_jetson_only_profile_is_not_offered_on_apple_silicon(self, catalog) -> None:
        hardware = HardwareInfo(
            system="Darwin",
            machine="arm64",
            platform_key="apple-silicon",
            device_name="Apple Silicon",
            accelerator="metal",
            memory_topology="shared",
            total_memory_gib=16.0,
        )

        assert "jetson-realtime" not in {
            model.key for model in catalog.ordered_models(hardware.platform_key)
        }
        with pytest.raises(ValueError, match="does not support"):
            resolve_profile(catalog, hardware, requested="jetson-realtime")

    def test_sixteen_gb_apple_silicon_recommends_balanced(self, catalog) -> None:
        hardware = HardwareInfo(
            system="Darwin",
            machine="arm64",
            platform_key="apple-silicon",
            device_name="Apple Silicon",
            accelerator="metal",
            memory_topology="shared",
            total_memory_gib=16.0,
        )

        resolved = resolve_profile(catalog, hardware)

        assert resolved.model.key == "balanced"
        assert resolved.environment["DEVICE"] == "mps"
        assert resolved.environment["LLAMA_N_GPU_LAYERS"] == "999"

    def test_cpu_profile_is_capped_at_compact(self, catalog) -> None:
        hardware = HardwareInfo(
            system="Linux",
            machine="x86_64",
            platform_key="linux-cpu",
            device_name="CPU",
            accelerator="cpu",
            memory_topology="shared",
            total_memory_gib=64.0,
        )

        resolved = resolve_profile(catalog, hardware)

        assert resolved.model.key == "compact"
        assert resolved.environment["STT_PROVIDER"] == "nemotron-cpp"

    def test_all_general_profiles_use_native_streaming_nemotron(self, catalog) -> None:
        for profile_name in ("lean", "compact", "balanced"):
            profile = catalog.models[profile_name]

            assert profile.environment["STT_PROVIDER"] == "nemotron-cpp"
            assert "Q8" in profile.stt

    def test_explicit_profile_wins_and_warns_when_over_budget(self, catalog) -> None:
        hardware = HardwareInfo(
            system="Darwin",
            machine="arm64",
            platform_key="apple-silicon",
            device_name="Apple Silicon",
            accelerator="metal",
            memory_topology="shared",
            total_memory_gib=8.0,
        )

        resolved = resolve_profile(catalog, hardware, requested="balanced")

        assert resolved.model.key == "balanced"
        assert "exceeds" in resolved.warning.lower()

    def test_manual_memory_budget_changes_auto_recommendation(self, catalog) -> None:
        hardware = HardwareInfo(
            system="Darwin",
            machine="arm64",
            platform_key="apple-silicon",
            device_name="Apple Silicon",
            accelerator="metal",
            memory_topology="shared",
            total_memory_gib=16.0,
        )

        resolved = resolve_profile(catalog, hardware, memory_budget_gib=5.5)

        assert resolved.model.key == "compact"
        assert resolved.memory_budget_gib == 5.5

    @pytest.mark.parametrize("budget", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_memory_budget_is_rejected(self, catalog, budget: float) -> None:
        hardware = HardwareInfo(
            system="Linux",
            machine="aarch64",
            platform_key="jetson-orin-nano",
            device_name="Jetson Orin Nano",
            accelerator="cuda",
            memory_topology="shared",
            total_memory_gib=8.0,
        )

        with pytest.raises(ValueError, match="finite"):
            resolve_profile(catalog, hardware, memory_budget_gib=budget)


class TestSavedSelection:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / ".local-voice-ai.toml"
        selection = UserSelection(
            profile="compact",
            memory_budget_gib=5.5,
            detected_platform="jetson-orin-nano",
            detected_memory_gib=8.0,
        )

        save_selection(path, selection)

        assert load_selection(path) == selection

    def test_missing_selection_returns_none(self, tmp_path: Path) -> None:
        assert load_selection(tmp_path / "missing.toml") is None

    def test_existing_flat_toml_selection_remains_readable(self, tmp_path: Path) -> None:
        path = tmp_path / ".local-voice-ai.toml"
        path.write_text(
            "version = 1\n"
            'profile = "compact"\n'
            'detected_platform = "jetson-orin-nano"\n'
            "detected_memory_gib = 8\n"
            "memory_budget_gib = 5.5\n",
            encoding="utf-8",
        )

        assert load_selection(path) == UserSelection(
            profile="compact",
            detected_platform="jetson-orin-nano",
            memory_budget_gib=5.5,
            detected_memory_gib=8.0,
        )
