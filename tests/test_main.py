"""Tests for llama.cpp child-spec construction, focused on offline startup.

The invariant (issue #9): a model that has already been downloaded must be
able to start with no internet. ``--offline`` is added automatically when the
--hf-repo model is found in the cache; ``LLAMA_OFFLINE`` overrides the
auto-detection either way; ``LLAMA_MODEL_PATH`` bypasses Hugging Face entirely.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import sys
from contextlib import closing
from pathlib import Path
from textwrap import dedent

import httpx
import pytest

import local_voice_ai.__main__ as main_mod
from local_voice_ai.__main__ import (
    _build_specs,
    _hf_hub_dir,
    _llama_cache_dir,
    _llama_repo_cached,
    _serve,
)
from local_voice_ai.config import Config
from local_voice_ai.supervisor import ChildSpec

# Must match Config.llama_hf_repo (checked below) — the :tag selects the quant.
REPO = "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL"
BARE_REPO, TAG = REPO.rsplit(":", 1)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the llama cache at a fresh tmp dir and clear offline overrides."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for var in ("LLAMA_CACHE", "LLAMA_OFFLINE", "LLAMA_MODEL_PATH", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _seed_manifest(cache_root: Path, repo: str = BARE_REPO, tag: str = TAG) -> None:
    """Create the manifest file legacy llama-server wrote after a download."""
    cache = cache_root / "llama.cpp"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"manifest={repo.replace('/', '=')}={tag}.json").write_text("{}")


def _seed_hub(cache_root: Path, repo: str = BARE_REPO, gguf: str | None = None) -> None:
    """Mirror the HF hub layout current llama-server downloads into
    (verified against a real b9909 download)."""
    snap = (
        cache_root / "huggingface" / "hub" / f"models--{repo.replace('/', '--')}"
        / "snapshots" / "0123abc"
    )
    snap.mkdir(parents=True, exist_ok=True)
    (snap / (gguf or f"gemma-4-E2B-it-qat-{TAG}.gguf")).write_text("x")


def _llama_spec() -> ChildSpec:
    cfg = Config.from_env()
    return next(s for s in _build_specs(cfg) if s.name == "llama")


def test_repo_constant_matches_config_default() -> None:
    # The cache-seeding tests only prove auto-offline works if this constant
    # is the repo the default config actually uses.
    assert Config.from_env().llama_hf_repo == REPO


class TestCacheDir:
    def test_llama_cache_wins(self) -> None:
        env = {"LLAMA_CACHE": "/x/llama", "XDG_CACHE_HOME": "/y"}
        assert _llama_cache_dir(env) == Path("/x/llama")

    def test_xdg_cache_home(self) -> None:
        assert _llama_cache_dir({"XDG_CACHE_HOME": "/y"}) == Path("/y/llama.cpp")

    def test_home_fallback(self) -> None:
        assert _llama_cache_dir({}) == Path.home() / ".cache" / "llama.cpp"


class TestHubDir:
    def test_hf_home_wins(self) -> None:
        env = {"HF_HOME": "/models", "XDG_CACHE_HOME": "/y"}
        assert _hf_hub_dir(env) == Path("/models/hub")

    def test_xdg_fallback(self) -> None:
        assert _hf_hub_dir({"XDG_CACHE_HOME": "/y"}) == Path("/y/huggingface/hub")

    def test_home_fallback(self) -> None:
        assert _hf_hub_dir({}) == Path.home() / ".cache" / "huggingface" / "hub"


class TestRepoCachedHubLayout:
    def test_exact_repo_and_tag(self, tmp_path: Path) -> None:
        _seed_hub(tmp_path)
        assert _llama_repo_cached(REPO, {"XDG_CACHE_HOME": str(tmp_path)}) is True

    def test_wrong_quant_tag_misses(self, tmp_path: Path) -> None:
        _seed_hub(tmp_path)
        env = {"XDG_CACHE_HOME": str(tmp_path)}
        assert _llama_repo_cached(f"{BARE_REPO}:Q8_0", env) is False

    def test_untagged_repo_matches_any_gguf(self, tmp_path: Path) -> None:
        _seed_hub(tmp_path)
        assert _llama_repo_cached(BARE_REPO, {"XDG_CACHE_HOME": str(tmp_path)}) is True

    def test_other_repo_misses(self, tmp_path: Path) -> None:
        _seed_hub(tmp_path)
        assert _llama_repo_cached("foo/Bar-GGUF", {"XDG_CACHE_HOME": str(tmp_path)}) is False

    def test_hf_home_layout(self, tmp_path: Path) -> None:
        # Production passes HF_HOME=/models; hub lives at /models/hub.
        snap = (
            tmp_path / "hub" / f"models--{BARE_REPO.replace('/', '--')}"
            / "snapshots" / "abc"
        )
        snap.mkdir(parents=True)
        (snap / f"gemma-4-E2B-it-qat-{TAG}.gguf").write_text("x")
        assert _llama_repo_cached(REPO, {"HF_HOME": str(tmp_path)}) is True


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

    def test_untagged_repo_uses_latest(self, tmp_path: Path) -> None:
        _seed_manifest(tmp_path, tag="latest")
        env = {"XDG_CACHE_HOME": str(tmp_path)}
        assert _llama_repo_cached(BARE_REPO, env) is True  # no tag → :latest
        assert _llama_repo_cached(REPO, env) is False  # :UD-Q4_K_XL not downloaded

    def test_other_repo_not_cached(self, tmp_path: Path) -> None:
        _seed_manifest(tmp_path)
        assert _llama_repo_cached("foo/Bar-GGUF", {"XDG_CACHE_HOME": str(tmp_path)}) is False

    def test_gguf_fallback_without_manifest(self, tmp_path: Path) -> None:
        cache = tmp_path / "llama.cpp"
        cache.mkdir()
        gguf = "unsloth_gemma-4-E2B-it-qat-GGUF_gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
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

    def test_reasoning_disabled_for_voice(self) -> None:
        # Thinking models must answer directly — reasoning is dead air on voice.
        argv = _llama_spec().argv
        assert argv[argv.index("--reasoning") + 1] == "off"

    def test_cache_env_passed_to_child(self, _isolated_cache: Path) -> None:
        # The dir we probe must be the dir the child will actually use.
        spec = _llama_spec()
        assert spec.env["XDG_CACHE_HOME"] == str(_isolated_cache)


def _free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# HTTP stub that plays a slow-starting child: sleeps first (simulating a model
# download), then serves 200s so the readiness probe passes.
_SLOW_HTTP_STUB = dedent(
    """
    import sys, time, http.server, socketserver
    time.sleep(float(sys.argv[2]))
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        def log_message(self, *a, **k): pass
    class S(socketserver.TCPServer):
        allow_reuse_address = True
    with S(('127.0.0.1', int(sys.argv[1])), H) as srv:
        srv.serve_forever()
    """
).strip()


class TestServeFirstBoot:
    """The web server must be up (and /api/status must report per-child
    readiness) while children are still starting — that's the whole point of
    the first-boot splash."""

    @pytest.mark.asyncio
    async def test_status_available_before_children_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        web_port, child_port = _free_port(), _free_port()
        spec = ChildSpec(
            name="slow",
            argv=[sys.executable, "-c", _SLOW_HTTP_STUB, str(child_port), "1.5"],
            ready_url=f"http://127.0.0.1:{child_port}/",
            ready_timeout=30.0,
        )
        monkeypatch.setattr(main_mod, "_build_specs", lambda cfg: [spec])
        monkeypatch.setenv("WEB_PORT", str(web_port))
        cfg = Config.from_env()

        serve_task = asyncio.create_task(_serve(cfg))
        status_url = f"http://127.0.0.1:{web_port}/api/status"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                # Web must answer while the child is still sleeping.
                deadline = asyncio.get_running_loop().time() + 10
                while True:
                    assert not serve_task.done(), "serve exited during startup"
                    try:
                        first = (await client.get(status_url)).json()
                        break
                    except httpx.RequestError:
                        assert asyncio.get_running_loop().time() < deadline
                        await asyncio.sleep(0.05)

                assert first["ready"] is False
                assert first["children"] == [
                    {"name": "slow", "ready": False, "running": True, "restarts": 0}
                ]

                # ...and flip to ready once the child passes its probe.
                while True:
                    data = (await client.get(status_url)).json()
                    if data["ready"]:
                        break
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.1)
                assert data["children"][0]["ready"] is True
        finally:
            # SIGTERM exercises the real coordinated-shutdown path.
            os.kill(os.getpid(), signal.SIGTERM)
            with contextlib.suppress(asyncio.CancelledError):
                rc = await asyncio.wait_for(serve_task, timeout=10)
                assert rc == 0


class TestWhisperSpec:
    def test_whisper_provider_uses_in_tree_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STT_PROVIDER", "whisper")
        cfg = Config.from_env()
        spec = next(s for s in _build_specs(cfg) if s.name == "whisper")
        assert "local_voice_ai.services.whisper.server" in spec.argv
        assert spec.env["WHISPER_MODEL"] == "Systran/faster-whisper-small"
        assert spec.ready_url == "http://127.0.0.1:8000/health"

    def test_nemotron_is_default(self) -> None:
        cfg = Config.from_env()
        names = [s.name for s in _build_specs(cfg)]
        assert "nemotron" in names and "whisper" not in names
