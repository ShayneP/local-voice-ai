"""Tests for llama.cpp child-spec construction, focused on offline startup.

The invariant (issue #9): a model that has already been downloaded must be
able to start with no internet. ``--offline`` is added automatically when the
--hf-repo model is found in the cache; ``LLAMA_OFFLINE`` overrides the
auto-detection either way; ``LLAMA_MODEL_PATH`` bypasses Hugging Face entirely.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_voice_ai.__main__ import _build_specs, _llama_cache_dir, _llama_repo_cached
from local_voice_ai.config import Config
from local_voice_ai.supervisor import ChildSpec

REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the llama cache at a fresh tmp dir and clear offline overrides."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for var in ("LLAMA_CACHE", "LLAMA_OFFLINE", "LLAMA_MODEL_PATH", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _seed_manifest(cache_root: Path, repo: str = REPO, tag: str = "latest") -> None:
    """Create the manifest file llama-server writes after an --hf-repo download."""
    cache = cache_root / "llama.cpp"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"manifest={repo.replace('/', '=')}={tag}.json").write_text("{}")


def _llama_spec() -> ChildSpec:
    cfg = Config.from_env()
    return next(s for s in _build_specs(cfg) if s.name == "llama")


class TestCacheDir:
    def test_llama_cache_wins(self) -> None:
        env = {"LLAMA_CACHE": "/x/llama", "XDG_CACHE_HOME": "/y"}
        assert _llama_cache_dir(env) == Path("/x/llama")

    def test_xdg_cache_home(self) -> None:
        assert _llama_cache_dir({"XDG_CACHE_HOME": "/y"}) == Path("/y/llama.cpp")

    def test_home_fallback(self) -> None:
        assert _llama_cache_dir({}) == Path.home() / ".cache" / "llama.cpp"


class TestRepoCached:
    def test_missing_cache_dir(self, tmp_path: Path) -> None:
        env = {"XDG_CACHE_HOME": str(tmp_path / "nope")}
        assert _llama_repo_cached(REPO, env) is False

    def test_empty_cache(self, tmp_path: Path) -> None:
        (tmp_path / "llama.cpp").mkdir()
        assert _llama_repo_cached(REPO, {"XDG_CACHE_HOME": str(tmp_path)}) is False

    def test_manifest_present(self, tmp_path: Path) -> None:
        _seed_manifest(tmp_path)
        assert _llama_repo_cached(REPO, {"XDG_CACHE_HOME": str(tmp_path)}) is True

    def test_manifest_with_quant_tag(self, tmp_path: Path) -> None:
        _seed_manifest(tmp_path, tag="Q4_K_M")
        env = {"XDG_CACHE_HOME": str(tmp_path)}
        assert _llama_repo_cached(f"{REPO}:Q4_K_M", env) is True
        assert _llama_repo_cached(REPO, env) is False  # :latest not downloaded

    def test_other_repo_not_cached(self, tmp_path: Path) -> None:
        _seed_manifest(tmp_path)
        assert _llama_repo_cached("foo/Bar-GGUF", {"XDG_CACHE_HOME": str(tmp_path)}) is False

    def test_gguf_fallback_without_manifest(self, tmp_path: Path) -> None:
        cache = tmp_path / "llama.cpp"
        cache.mkdir()
        gguf = "unsloth_Qwen3-4B-Instruct-2507-GGUF_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        (cache / gguf).write_text("x")
        assert _llama_repo_cached(REPO, {"XDG_CACHE_HOME": str(tmp_path)}) is True


class TestOfflineResolution:
    def test_first_run_downloads(self) -> None:
        # Nothing cached yet → no --offline, normal --hf-repo download path.
        argv = _llama_spec().argv
        assert "--hf-repo" in argv
        assert "--offline" not in argv

    def test_cached_model_auto_offline(self, _isolated_cache: Path) -> None:
        _seed_manifest(_isolated_cache)
        argv = _llama_spec().argv
        assert "--hf-repo" in argv
        assert "--offline" in argv

    def test_env_forces_offline_even_when_not_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLAMA_OFFLINE", "1")
        assert "--offline" in _llama_spec().argv

    def test_env_disables_auto_offline(
        self, _isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_manifest(_isolated_cache)
        monkeypatch.setenv("LLAMA_OFFLINE", "0")
        assert "--offline" not in _llama_spec().argv

    def test_local_model_path_bypasses_hf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLAMA_MODEL_PATH", "/models/foo.gguf")
        argv = _llama_spec().argv
        assert argv[argv.index("-m") + 1] == "/models/foo.gguf"
        assert "--hf-repo" not in argv
        assert "--offline" not in argv  # -m never touches the network anyway

    def test_local_model_path_with_explicit_offline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLAMA_MODEL_PATH", "/models/foo.gguf")
        monkeypatch.setenv("LLAMA_OFFLINE", "1")
        assert "--offline" in _llama_spec().argv

    def test_cache_env_passed_to_child(self, _isolated_cache: Path) -> None:
        # The dir we probe must be the dir the child will actually use.
        spec = _llama_spec()
        assert spec.env["XDG_CACHE_HOME"] == str(_isolated_cache)
