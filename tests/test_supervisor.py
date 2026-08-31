"""Tests for the async process supervisor.

We exercise the four behaviors that matter:
  1. Spawn N children, each waits for a readiness URL → all_ready() returns.
  2. A child that never becomes ready trips the timeout.
  3. A child that crashes after becoming ready gets restarted.
  4. Exceeding ``max_restarts`` signals the supervisor to stop.

Each test uses a tiny inline HTTP server child (no project deps).
"""

from __future__ import annotations

import asyncio
import socket
import sys
from contextlib import closing
from textwrap import dedent

import httpx
import pytest

from local_voice_ai.supervisor import ChildSpec, Supervisor


def _free_port() -> int:
    """Return a port that's currently unbound (best-effort)."""
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_HTTP_STUB = dedent(
    """
    import sys, http.server, socketserver
    port = int(sys.argv[1])
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        def log_message(self, *a, **k): pass
    class S(socketserver.TCPServer):
        allow_reuse_address = True  # restarted child must be able to rebind
    with S(('127.0.0.1', port), H) as srv:
        srv.serve_forever()
    """
).strip()


def _http_child(name: str, port: int, *, ready_timeout: float = 10.0,
                max_restarts: int = 5) -> ChildSpec:
    return ChildSpec(
        name=name,
        argv=[sys.executable, "-c", _HTTP_STUB, str(port)],
        ready_url=f"http://127.0.0.1:{port}/",
        ready_timeout=ready_timeout,
        max_restarts=max_restarts,
    )


class TestSpawnAndReady:
    @pytest.mark.asyncio
    async def test_two_children_become_ready(self) -> None:
        port_a, port_b = _free_port(), _free_port()
        sup = Supervisor([_http_child("a", port_a), _http_child("b", port_b)])
        try:
            await sup.start_all()
            async with httpx.AsyncClient(timeout=2.0) as c:
                ra = await c.get(f"http://127.0.0.1:{port_a}/")
                rb = await c.get(f"http://127.0.0.1:{port_b}/")
            assert ra.status_code == 200
            assert rb.status_code == 200
        finally:
            await sup.shutdown(timeout=3.0)

    @pytest.mark.asyncio
    async def test_sequential_startup_waits_before_starting_next_child(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        specs = [
            ChildSpec(name="first", argv=["first"]),
            ChildSpec(name="second", argv=["second"]),
        ]
        sup = Supervisor(specs, sequential_startup=True)
        events: list[str] = []

        async def record_start(child) -> None:
            events.append(f"start:{child.spec.name}")

        async def record_ready(child) -> None:
            events.append(f"ready:{child.spec.name}")

        monkeypatch.setattr(sup, "_start", record_start)
        monkeypatch.setattr(sup, "_await_ready", record_ready)

        try:
            await sup.start_all()
        finally:
            await sup.shutdown(timeout=1.0)

        assert events == [
            "start:first",
            "ready:first",
            "start:second",
            "ready:second",
        ]

    @pytest.mark.asyncio
    async def test_successful_http_response_must_pass_the_child_check(self) -> None:
        port = _free_port()
        spec = ChildSpec(
            name="expected-service",
            argv=[sys.executable, "-c", _HTTP_STUB, str(port)],
            ready_url=f"http://127.0.0.1:{port}/",
            ready_timeout=0.5,
            response_check=lambda response: response.text == "expected",
        )
        sup = Supervisor([spec])

        try:
            with pytest.raises(TimeoutError):
                await sup.start_all()
            assert sup.status()[0]["ready"] is False
        finally:
            await sup.shutdown(timeout=3.0)

    @pytest.mark.asyncio
    async def test_status_reflects_readiness(self) -> None:
        port = _free_port()
        sup = Supervisor([_http_child("a", port)])

        # Before spawn: not running, not ready — the first-boot UI's view.
        assert sup.status() == [
            {"name": "a", "ready": False, "running": False, "restarts": 0}
        ]

        try:
            await sup.start_all()
            assert sup.status() == [
                {"name": "a", "ready": True, "running": True, "restarts": 0}
            ]
        finally:
            await sup.shutdown(timeout=3.0)

    @pytest.mark.asyncio
    async def test_shutdown_terminates_children(self) -> None:
        port = _free_port()
        sup = Supervisor([_http_child("a", port)])
        await sup.start_all()
        try:
            child = sup._children[0]
            pid = child.process.pid  # type: ignore[union-attr]
            assert child.process and child.process.returncode is None
        finally:
            await sup.shutdown(timeout=3.0)

        assert child.process and child.process.returncode is not None
        # PID should no longer be a live process
        with pytest.raises(ProcessLookupError):
            import os
            os.kill(pid, 0)


class TestReadinessTimeout:
    @pytest.mark.asyncio
    async def test_child_that_never_responds_trips_timeout(self) -> None:
        # Spawn a child that exits immediately — readiness probe will never succeed.
        spec = ChildSpec(
            name="dead",
            argv=[sys.executable, "-c", "import sys; sys.exit(0)"],
            ready_url="http://127.0.0.1:1/",  # port 1 won't be reachable either
            ready_timeout=2.0,
        )
        sup = Supervisor([spec])
        with pytest.raises((RuntimeError, TimeoutError)):
            await sup.start_all()
        await sup.shutdown(timeout=1.0)


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_child_crash_after_ready_is_restarted(self) -> None:
        port = _free_port()
        sup = Supervisor([_http_child("a", port, max_restarts=3)])
        await sup.start_all()
        try:
            child = sup._children[0]
            assert child.process is not None
            first_pid = child.process.pid

            # Kill the process to simulate a crash.
            child.process.terminate()
            await child.process.wait()

            # Give the supervisor a moment to notice and restart.
            for _ in range(40):  # up to ~10s with the backoff
                await asyncio.sleep(0.25)
                if child.process and child.process.returncode is None and child.process.pid != first_pid:
                    break
            assert child.process and child.process.pid != first_pid, "child was not restarted"

            # And it should be reachable again after the restart.
            async with httpx.AsyncClient(timeout=3.0) as c:
                for _ in range(20):
                    try:
                        r = await c.get(f"http://127.0.0.1:{port}/")
                        if r.status_code == 200:
                            break
                    except httpx.RequestError:
                        await asyncio.sleep(0.25)
                else:
                    pytest.fail("restarted child never became reachable")
        finally:
            await sup.shutdown(timeout=3.0)

    @pytest.mark.asyncio
    async def test_exceeding_max_restarts_signals_stop(self) -> None:
        port = _free_port()
        # Set max_restarts=0 — first crash should immediately set stop_event.
        sup = Supervisor([_http_child("a", port, max_restarts=0)])
        await sup.start_all()
        try:
            child = sup._children[0]
            assert child.process is not None
            child.process.terminate()
            await child.process.wait()
            # Wait for the watch task to run.
            for _ in range(20):
                await asyncio.sleep(0.1)
                if sup.stopping:
                    break
            assert sup.stopping, "supervisor did not enter stopping state"
        finally:
            await sup.shutdown(timeout=3.0)


# A stub that starts healthy and starts returning 503 once a sentinel file
# appears — the shape of a model process that is alive but can no longer serve.
_FLIPPABLE_STUB = dedent(
    """
    import sys, os, http.server, socketserver
    port = int(sys.argv[1]); flag = sys.argv[2]
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if os.path.exists(flag):
                self.send_response(503); self.end_headers(); self.wfile.write(b'degraded')
            else:
                self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        def log_message(self, *a, **k): pass
    class S(socketserver.TCPServer):
        allow_reuse_address = True
    with S(('127.0.0.1', port), H) as srv:
        srv.serve_forever()
    """
).strip()


class TestUnhealthyRestart:
    """A child stuck unhealthy must be restarted even though it never exits.

    This is the CUDA-OOM failure mode: the STT process keeps its socket open
    and answers, but every inference fails, so _watch_exit never fires.
    """

    def _spec(self, port: int, flag: str) -> ChildSpec:
        return ChildSpec(
            name="flappy",
            argv=[sys.executable, "-c", _FLIPPABLE_STUB, str(port), flag],
            ready_url=f"http://127.0.0.1:{port}/",
            ready_timeout=10.0,
            max_restarts=3,
            health_interval=0.25,
            health_failures=2,
        )

    @pytest.mark.asyncio
    async def test_unhealthy_child_is_terminated_and_restarted(self, tmp_path) -> None:
        port = _free_port()
        flag = str(tmp_path / "degraded")
        sup = Supervisor([self._spec(port, flag)])
        await sup.start_all()
        try:
            child = sup._children[0]
            first_pid = child.process.pid

            # Flip the stub unhealthy; the monitor should notice and recycle it.
            open(flag, "w").close()
            for _ in range(80):
                await asyncio.sleep(0.25)
                if child.process and child.process.pid != first_pid:
                    break
            assert child.process.pid != first_pid, "unhealthy child was not restarted"
            assert child.restart_count >= 1
        finally:
            await sup.shutdown()

    @pytest.mark.asyncio
    async def test_healthy_child_is_not_restarted(self, tmp_path) -> None:
        # The monitor must not recycle a child that keeps answering, or every
        # long-running service would be killed on a timer.
        port = _free_port()
        sup = Supervisor([self._spec(port, str(tmp_path / "never"))])
        await sup.start_all()
        try:
            child = sup._children[0]
            first_pid = child.process.pid
            await asyncio.sleep(2.0)  # many health intervals
            assert child.process.pid == first_pid
            assert child.restart_count == 0
        finally:
            await sup.shutdown()

    @pytest.mark.asyncio
    async def test_health_interval_zero_disables_monitor(self, tmp_path) -> None:
        port = _free_port()
        flag = str(tmp_path / "degraded")
        spec = self._spec(port, flag)
        spec.health_interval = 0
        sup = Supervisor([spec])
        await sup.start_all()
        try:
            child = sup._children[0]
            first_pid = child.process.pid
            open(flag, "w").close()
            await asyncio.sleep(1.5)
            assert child.process.pid == first_pid, "monitor ran despite being disabled"
        finally:
            await sup.shutdown()
