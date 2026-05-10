"""Tests for the FastAPI app: token minting + static frontend serving."""

from __future__ import annotations

import base64
import json
import pathlib
import tempfile
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from local_voice_ai.api import build_app
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
