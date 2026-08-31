"""FastAPI app served from the supervisor process.

Three responsibilities:
  1. ``POST /api/connection-details`` — mints a LiveKit access token. This is
     the Python port of ``frontend/app/api/connection-details/route.ts``.
  2. ``/v1/*`` — an OpenAI-compatible gateway proxying to the LLM/STT/TTS
     children, when ``Config.gateway`` is set. Off by default.
  3. ``GET /*`` — serves the statically-exported Next.js frontend, when
     ``Config.frontend_dir`` is set.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from livekit import api as lk_api
from starlette.background import BackgroundTask

from .config import Config

logger = logging.getLogger("api")

# A frontend running on the user's own computer gets a secure-enough browser
# context at localhost, so microphone access works without TLS. Keep this
# narrower than CORS "*": arbitrary websites must not be able to mint tokens
# for a Local Voice AI server on the user's LAN.
_LOCAL_CLIENT_ORIGIN = r"^https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$"

# Which backend serves each OpenAI route. Everything else under /v1 is a 404 —
# an explicit table beats a catch-all that would silently forward typos to the
# LLM and return a confusing upstream error.
_GATEWAY_ROUTES: dict[str, str] = {
    "/v1/chat/completions": "llm",
    "/v1/completions": "llm",
    "/v1/embeddings": "llm",
    "/v1/audio/transcriptions": "stt",
    "/v1/audio/translations": "stt",
    "/v1/audio/speech": "tts",
}

# Connection-scoped headers must not be forwarded across a proxy hop; httpx
# recomputes length/host for the upstream request.
_DROP_REQUEST_HEADERS = frozenset(
    {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "upgrade"}
)
_DROP_RESPONSE_HEADERS = frozenset(
    {"connection", "keep-alive", "transfer-encoding", "upgrade"}
)


def _backend_base_urls(cfg: Config) -> dict[str, str]:
    """Backend name → its OpenAI base URL (each already ends in ``/v1``)."""
    return {
        "llm": cfg.llama_base_url,
        "stt": cfg.stt_base_url,
        "tts": cfg.tts_base_url,
    }


def _upstream_url(base_url: str, path: str) -> str:
    """Map a gateway path onto a backend base URL.

    Both sides carry the ``/v1`` prefix, so it is stripped from the path to
    avoid ``/v1/v1/chat/completions``.
    """
    return base_url.rstrip("/") + path[len("/v1") :]


def _mint_token(cfg: Config, agent_name: str | None) -> dict[str, Any]:
    participant_name = "user"
    participant_identity = f"voice_assistant_user_{random.randint(0, 9999)}"
    room_name = f"voice_assistant_room_{random.randint(0, 9999)}"

    token = (
        lk_api.AccessToken(cfg.livekit_api_key, cfg.livekit_api_secret)
        .with_identity(participant_identity)
        .with_name(participant_name)
        .with_ttl(timedelta(minutes=15))
        .with_grants(
            lk_api.VideoGrants(
                room=room_name,
                room_join=True,
                can_publish=True,
                can_publish_data=True,
                can_subscribe=True,
            )
        )
    )

    if agent_name:
        token = token.with_room_config(
            lk_api.RoomConfiguration(agents=[lk_api.RoomAgentDispatch(agent_name=agent_name)])
        )

    return {
        "serverUrl": cfg.advertised_livekit_url,
        "roomName": room_name,
        "participantName": participant_name,
        "participantToken": token.to_jwt(),
    }


def _gateway_lifespan(cfg: Config):
    """Own the proxy's HTTP client for the app's lifetime.

    One shared AsyncClient: opening a connection per request would add a
    handshake to every token of a stream. Returns None when the gateway is off
    so the app keeps FastAPI's default lifespan.
    """
    if not cfg.gateway:
        return None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # No read timeout: a long generation or a cold model load can outlast
        # any figure we'd pick, and the caller hanging up cancels it anyway.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0),
            follow_redirects=False,
        ) as client:
            app.state.gateway_client = client
            yield

    return lifespan


def _mount_gateway(app: FastAPI, cfg: Config) -> None:
    """Add the OpenAI-compatible ``/v1/*`` proxy routes to ``app``.

    Registered before the SPA catch-all so ``/v1/...`` never falls through to
    index.html. One shared AsyncClient is kept for the app's lifetime: opening a
    connection per request would add a handshake to every token of a stream.
    """
    bases = _backend_base_urls(cfg)

    def _client() -> httpx.AsyncClient:
        return app.state.gateway_client

    async def _proxy(request: Request, backend: str) -> StreamingResponse:
        client: httpx.AsyncClient = _client()
        url = _upstream_url(bases[backend], request.url.path)
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _DROP_REQUEST_HEADERS
        }
        upstream = client.build_request(
            request.method,
            url,
            headers=headers,
            params=request.query_params,
            content=await request.body(),
        )
        try:
            # stream=True so SSE tokens reach the caller as they are produced
            # rather than being buffered until the generation completes.
            response = await client.send(upstream, stream=True)
        except httpx.RequestError as exc:
            logger.warning("gateway: %s backend unreachable at %s (%s)", backend, url, exc)
            raise HTTPException(
                status_code=502, detail=f"{backend} backend unreachable"
            ) from exc

        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers={
                k: v
                for k, v in response.headers.items()
                if k.lower() not in _DROP_RESPONSE_HEADERS
            },
            background=BackgroundTask(response.aclose),
        )

    for path, backend in _GATEWAY_ROUTES.items():
        # Bind `backend` per iteration; a closure over the loop variable would
        # send every route to whichever backend happened to be last.
        def _make(backend: str = backend):
            async def handler(request: Request) -> StreamingResponse:
                return await _proxy(request, backend)

            return handler

        app.post(path)(_make())

    @app.get("/v1/models")
    async def gateway_models() -> JSONResponse:
        """Union of every backend's model list.

        A backend that is still loading (or configured but not running) is
        skipped rather than failing the whole listing, so clients can discover
        the LLM while STT is still warming up.
        """
        client: httpx.AsyncClient = _client()
        entries: list[dict[str, Any]] = []
        for backend, base in bases.items():
            try:
                resp = await client.get(f"{base.rstrip('/')}/models", timeout=5.0)
                resp.raise_for_status()
                payload = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                logger.debug("gateway: %s model listing unavailable (%s)", backend, exc)
                continue
            for entry in payload.get("data", []):
                if isinstance(entry, dict) and entry.get("id"):
                    # Tag the role so a client hitting one URL can tell which
                    # id is the chat model and which transcribes.
                    entries.append({**entry, "backend": backend})
        return JSONResponse({"object": "list", "data": entries})


def build_app(
    cfg: Config,
    status_provider: Callable[[], list[dict[str, Any]]] | None = None,
) -> FastAPI:
    app = FastAPI(title="local-voice-ai", version="0.1.0", lifespan=_gateway_lifespan(cfg))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.client_origins),
        allow_origin_regex=_LOCAL_CLIENT_ORIGIN,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Sandbox-Id"],
    )

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        """Per-child readiness, polled by the frontend's first-boot splash.

        The web server starts before the children are ready (first boot can
        spend a long time downloading model weights), so this is how the UI
        knows whether the stack is usable yet.
        """
        children = status_provider() if status_provider is not None else []
        return {
            "ready": all(c["ready"] for c in children),
            "children": children,
            # Lets the frontend hint "say the wake phrase" when enabled.
            "wake_word": cfg.wake_word,
        }

    @app.post("/api/connection-details")
    async def connection_details(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}

        agent_name: str | None = None
        try:
            agent_name = body.get("room_config", {}).get("agents", [{}])[0].get("agent_name")
        except (AttributeError, IndexError, TypeError):
            agent_name = None

        try:
            data = _mint_token(cfg, agent_name)
        except Exception as exc:
            logger.exception("token minting failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    if cfg.gateway:
        _mount_gateway(app, cfg)

    if cfg.frontend_dir:
        # SPA-style: serve static export, falling back to index.html for unknown paths.
        static = StaticFiles(directory=cfg.frontend_dir, html=True)

        @app.get("/{path:path}")
        async def spa(path: str, request: Request) -> Any:
            try:
                return await static.get_response(path or "index.html", request.scope)
            except Exception:
                return FileResponse(f"{cfg.frontend_dir}/index.html")

    return app
