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
    _load_env_files,
    _serve,
    _startup_line,
    make_status_provider,
)
from local_voice_ai.config import Config
from local_voice_ai.supervisor import ChildSpec, Supervisor

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


class TestStatusDetails:
    """make_status_provider augments not-ready children with the bytes their
    model download occupies so far (the '6 loading bars' data source)."""

    def _provider(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        cfg = Config.from_env()
        sup = Supervisor(_build_specs(cfg))  # nothing spawned: all not-ready
        return make_status_provider(sup, cfg), cfg

    def test_no_detail_before_download_starts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, _ = self._provider(tmp_path, monkeypatch)
        assert all("detail" not in c for c in provider())

    def test_detail_reports_downloaded_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, cfg = self._provider(tmp_path, monkeypatch)
        repo_dir = (
            tmp_path / "hub"
            / f"models--{cfg.llama_hf_repo.split(':')[0].replace('/', '--')}"
            / "blobs"
        )
        repo_dir.mkdir(parents=True)
        (repo_dir / "x.incomplete").write_bytes(b"\0" * 2_000_000)
        llama = next(c for c in provider() if c["name"] == "llama")
        assert llama["detail"] == "2 MB"

    def test_detail_does_not_count_hugging_face_symlinks_twice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, cfg = self._provider(tmp_path, monkeypatch)
        repo_dir = (
            tmp_path / "hub"
            / f"models--{cfg.llama_hf_repo.split(':')[0].replace('/', '--')}"
        )
        blob = repo_dir / "blobs" / "model"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"\0" * 2_000_000)
        snapshot = repo_dir / "snapshots" / "revision" / "model.gguf"
        snapshot.parent.mkdir(parents=True)
        snapshot.symlink_to(blob)

        llama = next(c for c in provider() if c["name"] == "llama")
        assert llama["detail"] == "2 MB"

    def test_onnx_tts_progress_uses_its_own_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        monkeypatch.setenv("TTS_PROVIDER", "kokoro-onnx")
        cfg = Config.from_env()
        sup = Supervisor(_build_specs(cfg))
        model = tmp_path / "kokoro-onnx" / "model.onnx.part"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"\0" * 2_000_000)

        kokoro = next(
            c for c in make_status_provider(sup, cfg)() if c["name"] == "kokoro"
        )

        assert kokoro["detail"] == "2 MB"

    def test_startup_line_format(self) -> None:
        line = _startup_line([
            {"name": "llama", "ready": False, "detail": "1.2 GB"},
            {"name": "kokoro", "ready": True},
        ])
        assert line == "llama … 1.2 GB | kokoro ✓"


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

    def test_whisper_can_run_on_cpu_while_llama_uses_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STT_PROVIDER", "whisper")
        monkeypatch.setenv("DEVICE", "cuda")
        monkeypatch.setenv("STT_DEVICE", "cpu")

        spec = next(s for s in _build_specs(Config.from_env()) if s.name == "whisper")

        assert spec.env["DEVICE"] == "cpu"

    def test_nemotron_is_default(self) -> None:
        cfg = Config.from_env()
        names = [s.name for s in _build_specs(cfg)]
        assert "nemotron" in names and "whisper" not in names

    def test_native_nemotron_uses_quantized_streaming_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STT_PROVIDER", "nemotron-cpp")

        spec = next(s for s in _build_specs(Config.from_env()) if s.name == "nemotron")

        assert "local_voice_ai.services.nemotron_cpp.launcher" in spec.argv
        assert spec.ready_url == "http://127.0.0.1:8000/ready"

    def test_sequential_startup_loads_stt_before_llama(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEQUENTIAL_STARTUP", "1")
        names = [spec.name for spec in _build_specs(Config.from_env())]

        assert names.index("nemotron") < names.index("llama")


class TestTtsSpec:
    def test_onnx_provider_uses_low_memory_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TTS_PROVIDER", "kokoro-onnx")

        spec = next(s for s in _build_specs(Config.from_env()) if s.name == "kokoro")

        assert "local_voice_ai.services.kokoro_onnx.server" in spec.argv


class TestAgentSpec:
    def test_readiness_waits_for_agent_http_server(self) -> None:
        spec = next(s for s in _build_specs(Config.from_env()) if s.name == "agent")

        assert spec.ready_url == "http://127.0.0.1:8081/"


class TestBindHost:
    """Inference children bind loopback unless explicitly widened."""

    def _hosts(self) -> dict[str, str]:
        cfg = Config.from_env()
        specs = _build_specs(cfg)
        return {
            s.name: s.argv[s.argv.index("--host") + 1]
            for s in specs
            if "--host" in s.argv
        }

    def test_defaults_to_loopback(self) -> None:
        hosts = self._hosts()
        assert hosts["llama"] == "127.0.0.1"
        assert hosts["nemotron"] == "127.0.0.1"
        assert hosts["kokoro"] == "127.0.0.1"

    def test_bind_host_widens_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BIND_HOST", "0.0.0.0")
        assert set(self._hosts().values()) == {"0.0.0.0"}

    def test_per_service_overrides_global(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BIND_HOST", "0.0.0.0")
        monkeypatch.setenv("TTS_BIND_HOST", "127.0.0.1")
        hosts = self._hosts()
        assert hosts["llama"] == "0.0.0.0"
        assert hosts["nemotron"] == "0.0.0.0"
        assert hosts["kokoro"] == "127.0.0.1"  # TTS stays local

    def test_wildcard_bind_probes_over_loopback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 0.0.0.0 is not a connectable address, so readiness must use loopback.
        monkeypatch.setenv("BIND_HOST", "0.0.0.0")
        specs = _build_specs(Config.from_env())
        for spec in specs:
            if spec.ready_url:
                assert "0.0.0.0" not in spec.ready_url

    def test_specific_address_is_probed_directly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLAMA_BIND_HOST", "192.168.1.50")
        spec = _llama_spec()
        assert spec.ready_url == "http://192.168.1.50:11434/v1/models"

    def test_llama_readiness_requires_the_expected_model(self) -> None:
        spec = _llama_spec()
        assert spec.response_check is not None

        expected = httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "gemma-4-e2b"}]},
        )
        ollama = httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "qwen3:1.7b"}]},
        )

        assert spec.response_check(expected) is True
        assert spec.response_check(ollama) is False

    def test_llama_voice_server_skips_vision_and_uses_configured_slots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLAMA_PARALLEL", "1")
        argv = _llama_spec().argv

        assert "--no-mmproj" in argv
        assert argv[argv.index("--parallel") + 1] == "1"


class TestEnvFileLoading:
    """Bare-metal runs never read .env — only Docker did, via compose's
    env_file. These cover the precedence that makes local overrides work."""

    def test_env_file_is_loaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("LLAMA_MODEL=from-env\n")
        monkeypatch.delenv("LLAMA_MODEL", raising=False)
        _load_env_files()
        assert os.environ["LLAMA_MODEL"] == "from-env"

    def test_env_local_beats_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("LLAMA_MODEL=from-env\n")
        (tmp_path / ".env.local").write_text("LLAMA_MODEL=from-local\n")
        monkeypatch.delenv("LLAMA_MODEL", raising=False)
        _load_env_files()
        assert os.environ["LLAMA_MODEL"] == "from-local"

    def test_real_env_beats_both_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An explicit `FOO=bar python -m local_voice_ai serve` must still win,
        # or every one-off override on the command line would be ignored.
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("LLAMA_MODEL=from-env\n")
        (tmp_path / ".env.local").write_text("LLAMA_MODEL=from-local\n")
        monkeypatch.setenv("LLAMA_MODEL", "from-shell")
        _load_env_files()
        assert os.environ["LLAMA_MODEL"] == "from-shell"

    def test_missing_files_are_not_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _load_env_files()  # no .env or .env.local present
