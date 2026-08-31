"""Tests for the FastAPI app: token minting + static frontend serving."""

from __future__ import annotations

import base64
import json
import pathlib
import socket
import tempfile
from collections.abc import Iterator
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from local_voice_ai.api import _GATEWAY_ROUTES, _upstream_url, build_app
from local_voice_ai.config import Config


def _decode_jwt_payload(token: str) -> dict:
    payload_b64 = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload_b64 + "==").decode())


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret-secret-secret-thirty-two-chars")
    monkeypatch.setenv("LIVEKIT_URL", "ws://127.0.0.1:7880")
    return Config.from_env()


@pytest.fixture
def client(cfg: Config) -> TestClient:
    return TestClient(build_app(cfg))


class TestHealth:
    def test_healthz_returns_ok(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestStatus:
    def test_no_provider_reports_ready(self, client: TestClient) -> None:
        # Without a supervisor (tests, bare API) the stack is trivially ready.
        r = client.get("/api/status")
        assert r.status_code == 200
        assert r.json() == {"ready": True, "children": [], "wake_word": False}

    def test_wake_word_flag_surfaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WAKE_WORD", "1")
        client = TestClient(build_app(Config.from_env()))
        assert client.get("/api/status").json()["wake_word"] is True

    def test_reports_children_not_ready(self, cfg: Config) -> None:
        children = [
            {"name": "llama", "ready": False, "running": True, "restarts": 0},
            {"name": "kokoro", "ready": True, "running": True, "restarts": 0},
        ]
        client = TestClient(build_app(cfg, status_provider=lambda: children))
        data = client.get("/api/status").json()
        assert data["ready"] is False
        assert data["children"] == children

    def test_ready_when_all_children_ready(self, cfg: Config) -> None:
        children = [
            {"name": "llama", "ready": True, "running": True, "restarts": 0},
            {"name": "agent", "ready": True, "running": True, "restarts": 1},
        ]
        client = TestClient(build_app(cfg, status_provider=lambda: children))
        assert client.get("/api/status").json()["ready"] is True


class TestConnectionDetails:
    def test_mints_token_with_empty_body(self, client: TestClient) -> None:
        r = client.post("/api/connection-details", json={})
        assert r.status_code == 200
        data = r.json()
        assert set(data) == {"serverUrl", "roomName", "participantName", "participantToken"}
        assert data["serverUrl"] == "ws://127.0.0.1:7880"
        assert data["roomName"].startswith("voice_assistant_room_")
        assert data["participantName"] == "user"

    def test_jwt_carries_correct_issuer_and_grants(self, client: TestClient) -> None:
        r = client.post("/api/connection-details", json={})
        payload = _decode_jwt_payload(r.json()["participantToken"])
        assert payload["iss"] == "devkey"
        assert payload["sub"].startswith("voice_assistant_user_")
        assert "video" in payload
        # AccessToken.with_grants serializes VideoGrants with camelCase keys.
        video = payload["video"]
        assert video.get("roomJoin") is True
        assert video.get("canPublish") is True
        assert video.get("canSubscribe") is True

    def test_token_has_expiry(self, client: TestClient) -> None:
        r = client.post("/api/connection-details", json={})
        payload = _decode_jwt_payload(r.json()["participantToken"])
        # 15-minute TTL → exp - nbf should be 900s.
        assert payload["exp"] - payload["nbf"] == 900

    def test_agent_dispatch_included_when_requested(self, client: TestClient) -> None:
        r = client.post(
            "/api/connection-details",
            json={"room_config": {"agents": [{"agent_name": "my-agent"}]}},
        )
        assert r.status_code == 200
        payload = _decode_jwt_payload(r.json()["participantToken"])
        assert "roomConfig" in payload

    def test_missing_agent_name_does_not_attach_room_config(self, client: TestClient) -> None:
        r = client.post("/api/connection-details", json={})
        payload = _decode_jwt_payload(r.json()["participantToken"])
        assert "roomConfig" not in payload

    def test_malformed_body_still_returns_a_token(self, client: TestClient) -> None:
        # The Next.js route swallowed JSON errors silently; ours should too.
        r = client.post("/api/connection-details", content=b"not json")
        assert r.status_code == 200

    def test_each_call_produces_a_fresh_room(self, client: TestClient) -> None:
        rooms = {client.post("/api/connection-details", json={}).json()["roomName"] for _ in range(8)}
        # Random ints in [0, 9999] → collisions are statistically possible but rare;
        # we want at least most of the rooms to be unique.
        assert len(rooms) >= 6


class TestClientOrigins:
    def test_local_client_can_request_a_connection_token(self, client: TestClient) -> None:
        response = client.options(
            "/api/connection-details",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-sandbox-id",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_unconfigured_remote_web_origin_is_not_allowed(self, client: TestClient) -> None:
        response = client.options(
            "/api/connection-details",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert "access-control-allow-origin" not in response.headers

    def test_configured_client_origin_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLIENT_ORIGINS", "https://voice.example, https://backup.example")
        client = TestClient(build_app(Config.from_env()))

        response = client.options(
            "/api/connection-details",
            headers={
                "Origin": "https://voice.example",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://voice.example"


class TestStaticFrontend:
    @pytest.fixture
    def frontend_dir(self) -> Iterator[pathlib.Path]:
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td)
            (out / "index.html").write_text("<h1>HOME</h1>")
            (out / "favicon.ico").write_bytes(b"\x00\x00")
            (out / "_next").mkdir()
            (out / "_next" / "static.js").write_text("// stub")
            yield out

    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch, frontend_dir: pathlib.Path) -> TestClient:
        monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "secret-secret-secret-thirty-two-chars")
        monkeypatch.setenv("FRONTEND_DIR", str(frontend_dir))
        return TestClient(build_app(Config.from_env()))

    def test_serves_index(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert "HOME" in r.text

    def test_serves_static_asset(self, client: TestClient) -> None:
        r = client.get("/_next/static.js")
        assert r.status_code == 200
        assert "stub" in r.text

    def test_spa_fallback_for_unknown_route(self, client: TestClient) -> None:
        r = client.get("/some/client-side/route")
        assert r.status_code == 200
        assert "HOME" in r.text

    def test_api_route_still_wins_over_spa_fallback(self, client: TestClient) -> None:
        r = client.post("/api/connection-details", json={})
        assert r.status_code == 200
        assert "participantToken" in r.json()

    def test_healthz_still_wins_over_spa_fallback(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def _closed_port() -> int:
    """A port with nothing listening, for exercising the unreachable path."""
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestGatewayDisabled:
    """The /v1 surface must stay absent unless explicitly enabled: the web port
    defaults to 0.0.0.0 and the backends have no auth."""

    def test_off_by_default(self) -> None:
        assert Config.from_env().gateway is False

    def test_routes_absent(self, client: TestClient) -> None:
        assert client.get("/v1/models").status_code == 404
        assert client.post("/v1/chat/completions").status_code == 404


class TestGatewayRouting:
    def test_upstream_url_strips_duplicate_v1(self) -> None:
        assert (
            _upstream_url("http://127.0.0.1:11434/v1", "/v1/chat/completions")
            == "http://127.0.0.1:11434/v1/chat/completions"
        )

    def test_upstream_url_tolerates_trailing_slash(self) -> None:
        assert (
            _upstream_url("http://127.0.0.1:8000/v1/", "/v1/audio/transcriptions")
            == "http://127.0.0.1:8000/v1/audio/transcriptions"
        )

    def test_audio_and_chat_use_different_backends(self) -> None:
        assert _GATEWAY_ROUTES["/v1/chat/completions"] == "llm"
        assert _GATEWAY_ROUTES["/v1/audio/transcriptions"] == "stt"
        assert _GATEWAY_ROUTES["/v1/audio/speech"] == "tts"

    def test_enabled_registers_routes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GATEWAY", "1")
        app = build_app(Config.from_env())
        paths = {r.path for r in app.routes if r.path.startswith("/v1")}
        assert paths == set(_GATEWAY_ROUTES) | {"/v1/models"}


class TestGatewayBackendDown:
    """A backend that isn't up must produce a clear error, not a hang or a 500."""

    @pytest.fixture
    def dead_client(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        port = _closed_port()
        monkeypatch.setenv("GATEWAY", "1")
        monkeypatch.setenv("LLAMA_BASE_URL", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setenv("STT_BASE_URL", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setenv("TTS_BASE_URL", f"http://127.0.0.1:{port}/v1")
        # Context manager so the lifespan runs and the shared client exists.
        with TestClient(build_app(Config.from_env())) as c:
            yield c

    def test_unreachable_backend_returns_502(self, dead_client: TestClient) -> None:
        resp = dead_client.post("/v1/chat/completions", json={"model": "x", "messages": []})
        assert resp.status_code == 502
        assert "unreachable" in resp.json()["detail"]

    def test_models_degrades_to_empty_list(self, dead_client: TestClient) -> None:
        # Discovery shouldn't fail outright just because a backend is still
        # loading — clients poll this while the stack warms up.
        resp = dead_client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json() == {"object": "list", "data": []}
