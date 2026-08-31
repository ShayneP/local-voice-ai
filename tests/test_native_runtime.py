"""Selection and installation of the packaged NeMo-Speech.cpp runtime."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from local_voice_ai.native_runtime import RuntimeArchive, install_archive, select_archive


@pytest.mark.parametrize(
    ("system", "machine", "backend", "asset_fragment"),
    [
        ("Linux", "x86_64", "cpu", "linux-x86_64-cpu"),
        ("Linux", "amd64", "cuda", "linux-x86_64-cuda"),
        ("Linux", "aarch64", "cuda", "linux-aarch64-cuda12"),
        ("Darwin", "arm64", "metal", "macos-aarch64-metal"),
    ],
)
def test_select_archive_matches_platform_backend(
    system: str,
    machine: str,
    backend: str,
    asset_fragment: str,
) -> None:
    archive = select_archive(system=system, machine=machine, backend=backend)

    assert asset_fragment in archive.asset


def test_install_archive_verifies_and_extracts_binary(tmp_path: Path) -> None:
    source = tmp_path / "runtime.tar.gz"
    payload = b"native runtime"
    with tarfile.open(source, "w:gz") as archive:
        info = tarfile.TarInfo("nemo-speech-test/bin/nemo-speech")
        info.mode = 0o755
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    runtime = RuntimeArchive(
        asset=source.name,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    destination = tmp_path / "installed"

    binary = install_archive(runtime, destination, base_url=tmp_path.as_uri())

    assert binary == destination / "bin" / "nemo-speech"
    assert binary.read_bytes() == payload
    assert binary.stat().st_mode & 0o111


def test_install_archive_rejects_wrong_checksum(tmp_path: Path) -> None:
    source = tmp_path / "runtime.tar.gz"
    source.write_bytes(b"not a runtime")
    runtime = RuntimeArchive(asset=source.name, sha256="0" * 64)

    with pytest.raises(RuntimeError, match="checksum"):
        install_archive(runtime, tmp_path / "installed", base_url=tmp_path.as_uri())
