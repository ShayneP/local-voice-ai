"""Install the pinned NeMo-Speech.cpp runtime for the current platform."""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

NEMO_SPEECH_VERSION = "0.1.0"
RELEASE_BASE_URL = (
    f"https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v{NEMO_SPEECH_VERSION}"
)


@dataclass(frozen=True)
class RuntimeArchive:
    asset: str
    sha256: str


_ARCHIVES = {
    ("linux", "x86_64", "cpu"): RuntimeArchive(
        asset="nemo-speech-0.1.0-linux-x86_64-cpu.tar.gz",
        sha256="0f74131d631ad2c694cf0ec53490866bb6461147959589a69fb6fc231944065b",
    ),
    ("linux", "aarch64", "cpu"): RuntimeArchive(
        asset="nemo-speech-0.1.0-linux-aarch64-cpu.tar.gz",
        sha256="0e4112255d566de7bdd142f239e984995c4447103ba8feb41f2bb5c559d561d3",
    ),
    ("linux", "x86_64", "cuda"): RuntimeArchive(
        asset="nemo-speech-0.1.0-linux-x86_64-cuda.tar.gz",
        sha256="e68628f396489c98fb353e070efaea5bc4977409ae7734fce56c251a79e29147",
    ),
    ("linux", "aarch64", "cuda"): RuntimeArchive(
        asset="nemo-speech-0.1.0-linux-aarch64-cuda12.tar.gz",
        sha256="20ab68b0f8b4ace1e30b6605859935b1a059fc149d7b24f7a4b4d95cbf2f0c2d",
    ),
    ("darwin", "aarch64", "metal"): RuntimeArchive(
        asset="nemo-speech-0.1.0-macos-aarch64-metal.tar.gz",
        sha256="f1dff4f9dd9c96214f8cb78b982812459132df8a4ad1a42409fd94de4a366244",
    ),
}


def _machine_key(machine: str) -> str:
    normalized = machine.strip().lower()
    if normalized in {"amd64", "x64"}:
        return "x86_64"
    if normalized in {"arm64", "arm64/v8"}:
        return "aarch64"
    return normalized


def _backend_key(backend: str) -> str:
    normalized = backend.strip().lower()
    return "metal" if normalized == "mps" else normalized


def select_archive(
    *,
    system: str | None = None,
    machine: str | None = None,
    backend: str,
) -> RuntimeArchive:
    key = (
        (system or platform.system()).strip().lower(),
        _machine_key(machine or platform.machine()),
        _backend_key(backend),
    )
    try:
        return _ARCHIVES[key]
    except KeyError as exc:
        raise ValueError(
            f"NeMo-Speech.cpp {NEMO_SPEECH_VERSION} has no runtime for {key[0]}/{key[1]}/{key[2]}"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if member_path != root and root not in member_path.parents:
            raise RuntimeError(f"unsafe path in NeMo-Speech.cpp archive: {member.name}")
        if member.issym() or member.islnk():
            link_path = (member_path.parent / member.linkname).resolve()
            if link_path != root and root not in link_path.parents:
                raise RuntimeError(f"unsafe link in NeMo-Speech.cpp archive: {member.name}")
    archive.extractall(destination)


def install_archive(
    archive: RuntimeArchive,
    destination: Path,
    *,
    base_url: str = RELEASE_BASE_URL,
) -> Path:
    """Install a verified release archive atomically and return its executable."""
    binary = destination / "bin" / "nemo-speech"
    if binary.is_file():
        return binary

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="nemo-speech-", dir=destination.parent))
    try:
        download = temporary / archive.asset
        url = f"{base_url.rstrip('/')}/{archive.asset}"
        with urllib.request.urlopen(url) as response, download.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if _sha256(download) != archive.sha256:
            raise RuntimeError("NeMo-Speech.cpp runtime checksum mismatch")

        extracted = temporary / "extracted"
        extracted.mkdir()
        with tarfile.open(download, "r:gz") as bundle:
            _safe_extract(bundle, extracted)

        roots = [path for path in extracted.iterdir() if path.is_dir()]
        if len(roots) != 1 or not (roots[0] / "bin" / "nemo-speech").is_file():
            raise RuntimeError("NeMo-Speech.cpp runtime archive has an unexpected layout")
        roots[0].replace(destination)
        binary.chmod(binary.stat().st_mode | 0o111)
        return binary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def install_runtime(
    cache_dir: Path,
    *,
    system: str | None = None,
    machine: str | None = None,
    backend: str,
) -> Path:
    archive = select_archive(system=system, machine=machine, backend=backend)
    destination = cache_dir / archive.asset.removesuffix(".tar.gz")
    return install_archive(archive, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default=platform.system())
    parser.add_argument("--machine", default=platform.machine())
    parser.add_argument("--backend", required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args(argv)

    archive = select_archive(
        system=args.system,
        machine=args.machine,
        backend=args.backend,
    )
    binary = install_archive(archive, args.prefix)
    print(binary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
