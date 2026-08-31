"""Presentation helpers for the dependency-free terminal launcher."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

from .profiles import HardwareInfo, ProfileCatalog, ResolvedProfile, resolve_profile


def render_plan(profile: ResolvedProfile) -> str:
    hardware = profile.hardware
    if hardware.memory_topology == "shared":
        memory = (
            f"{hardware.total_memory_gib:.1f} GB unified/shared; "
            f"{profile.memory_budget_gib:.1f} GB inference budget"
        )
    elif hardware.memory_topology == "discrete":
        memory = (
            f"{hardware.accelerator_memory_gib:.1f} GB VRAM; "
            f"{profile.memory_budget_gib:.1f} GB inference budget"
        )
    else:
        memory = (
            f"{hardware.total_memory_gib:.1f} GB system RAM; "
            f"{profile.memory_budget_gib:.1f} GB inference budget"
        )

    lines = [
        "",
        "Local Voice Agent setup",
        "=" * 54,
        f"Detected   {hardware.device_name} ({hardware.system}/{hardware.machine})",
    ]
    release = " · ".join(
        detail
        for detail in (
            f"JetPack {hardware.jetpack_version}" if hardware.jetpack_version else "",
            f"L4T {hardware.l4t_version}" if hardware.l4t_version else "",
        )
        if detail
    )
    if release:
        lines.append(f"Platform   {release}")
    lines.extend(
        [
            f"Memory     {memory}",
            f"Runtime    {profile.platform.label} · {profile.platform.runtime}",
            "",
            f"Recommended  {profile.model.label}"
            if profile.automatic
            else f"Selected     {profile.model.label}",
            f"  LLM       {profile.model.llm}",
            f"  Speech    {profile.model.stt}",
            f"  Voice     {profile.model.tts}",
            f"  Target    about {profile.model.target_memory_gib:.1f} GB resident memory",
            f"  Weights   about {profile.model.download_gib:.1f} GB on first run",
        ]
    )
    if profile.platform.container_download_gib:
        lines.append(
            f"  Image     about {profile.platform.container_download_gib:.1f} GB compressed "
            "on first build"
        )
    lines.append(f"  Why       {profile.model.description}")
    if profile.warning:
        lines.extend(["", f"Warning: {profile.warning}"])
    return "\n".join(lines)


def choose_profile(
    catalog: ProfileCatalog,
    hardware: HardwareInfo,
    *,
    memory_budget_gib: float | None = None,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> ResolvedProfile | None:
    """Offer one obvious recommendation with profile and memory escape hatches."""
    budget = memory_budget_gib
    while True:
        recommendation = resolve_profile(catalog, hardware, memory_budget_gib=budget)
        output.write(render_plan(recommendation) + "\n\n")
        output.write("Press Enter to accept, [c] change profile, [m] set memory, [q] quit: ")
        output.flush()
        try:
            choice = input_fn("").strip().lower()
        except (EOFError, KeyboardInterrupt):
            output.write("\n")
            return None

        if choice == "":
            return recommendation
        if choice == "q":
            return None
        if choice == "c":
            output.write("\n")
            models = catalog.ordered_models(hardware.platform_key)
            for index, model in enumerate(models, 1):
                output.write(f"  {index}. {model.label} · about {model.target_memory_gib:.1f} GB\n")
            output.write("Choose profile: ")
            output.flush()
            try:
                raw = input_fn("").strip()
                index = int(raw) - 1
                if index < 0:
                    raise ValueError
                selected = models[index]
            except (ValueError, IndexError):
                output.write("That is not a valid profile.\n")
                continue
            except (EOFError, KeyboardInterrupt):
                output.write("\n")
                return None
            resolved = resolve_profile(
                catalog,
                hardware,
                requested=selected.key,
                memory_budget_gib=budget,
            )
            output.write(render_plan(resolved) + "\n")
            return resolved
        if choice == "m":
            output.write("Memory available to inference in GB: ")
            output.flush()
            try:
                raw_budget = float(input_fn("").strip())
                if raw_budget <= 0 or raw_budget > hardware.memory_capacity_gib:
                    raise ValueError
                budget = raw_budget
            except ValueError:
                output.write(
                    f"Enter a number greater than 0 and no more than "
                    f"{hardware.memory_capacity_gib:.1f}.\n"
                )
            except (EOFError, KeyboardInterrupt):
                output.write("\n")
                return None
            continue
        output.write("Choose Enter, c, m, or q.\n")
