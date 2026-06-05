"""FastAPI app served from the supervisor process.

Two responsibilities:
  1. ``POST /api/connection-details`` — mints a LiveKit access token. This is
     the Python port of ``frontend/app/api/connection-details/route.ts``.
  2. ``GET /*`` — serves the statically-exported Next.js frontend, when
     ``Config.frontend_dir`` is set.
"""

from __future__ import annotations

import logging
import random
from datetime import timedelta
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from livekit import api as lk_api

from .config import Config

logger = logging.getLogger("api")


def _mint_token(cfg: Config, agent_name: Optional[str]) -> dict[str, Any]:
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
        "serverUrl": cfg.livekit_url,
        "roomName": room_name,
        "participantName": participant_name,
        "participantToken": token.to_jwt(),
    }


def build_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="local-voice-ai", version="0.1.0")

    @app.post("/api/connection-details")
    async def connection_details(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}

        agent_name: Optional[str] = None
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
