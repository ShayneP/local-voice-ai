"""Behavior of the dependency-free first-run terminal flow."""

from __future__ import annotations

from io import StringIO

from local_voice_ai.launcher import choose_profile, render_plan
from local_voice_ai.profiles import (
    DEFAULT_CATALOG_PATH,
    HardwareInfo,
    load_catalog,
    resolve_profile,
)


def _jetson() -> HardwareInfo:
    return HardwareInfo(
        system="Linux",
        machine="aarch64",
        platform_key="jetson-orin-nano",
        device_name="NVIDIA Jetson Orin Nano",
        accelerator="cuda",
        memory_topology="shared",
        total_memory_gib=8.0,
        jetpack_version="6.2.1",
        l4t_version="36.4.3",
    )


def test_plan_explains_shared_memory_and_models() -> None:
    resolved = resolve_profile(load_catalog(DEFAULT_CATALOG_PATH), _jetson())

    rendered = render_plan(resolved)

    assert "8.0 GB unified/shared" in rendered
    assert "6.0 GB inference budget" in rendered
    assert "Jetson Realtime" in rendered
    assert "Qwen3 1.7B" in rendered
    assert "Nemotron Speech 0.6B Q8" in rendered
    assert "Kokoro" in rendered
    assert "JetPack 6.2.1" in rendered
    assert "L4T 36.4.3" in rendered
    assert "docker" in rendered
    assert "5.9 GB compressed" in rendered


def test_enter_accepts_recommendation() -> None:
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    output = StringIO()

    selected = choose_profile(
        catalog,
        _jetson(),
        input_fn=lambda _prompt: "",
        output=output,
    )

    assert selected is not None
    assert selected.model.key == "jetson-realtime"
    assert "Press Enter to accept" in output.getvalue()


def test_change_menu_can_select_another_profile() -> None:
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    answers = iter(["c", "4"])

    selected = choose_profile(
        catalog,
        _jetson(),
        input_fn=lambda _prompt: next(answers),
        output=StringIO(),
    )

    assert selected is not None
    assert selected.model.key == "balanced"
    assert selected.warning


def test_memory_override_recomputes_recommendation() -> None:
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    apple = HardwareInfo(
        system="Darwin",
        machine="arm64",
        platform_key="apple-silicon",
        device_name="Apple Silicon",
        accelerator="metal",
        memory_topology="shared",
        total_memory_gib=16.0,
    )
    answers = iter(["m", "5.5", ""])

    selected = choose_profile(
        catalog,
        apple,
        input_fn=lambda _prompt: next(answers),
        output=StringIO(),
    )

    assert selected is not None
    assert selected.model.key == "compact"
    assert selected.memory_budget_gib == 5.5
