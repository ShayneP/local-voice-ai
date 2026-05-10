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
