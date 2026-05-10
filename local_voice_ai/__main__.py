"""Entry point: ``python -m local_voice_ai [serve|download-models|console]``.

The default ``serve`` command:
  1. Builds child specs based on the config (skipping any service whose base
     URL is external).
  2. Spawns all children, waits for readiness.
  3. Starts the FastAPI app (token route + static frontend) on the same loop.
  4. Blocks on SIGTERM/SIGINT, then shuts everything down cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

import uvicorn

from .api import build_app
from .config import Config
from .supervisor import ChildSpec, Supervisor, configure_logging

logger = logging.getLogger("main")


def _build_specs(cfg: Config) -> list[ChildSpec]:
    specs: list[ChildSpec] = []
    py = sys.executable

    # --- LiveKit server (Go binary) ----------------------------------
    if cfg.manage_livekit:
        livekit_bin = os.getenv("LIVEKIT_BIN", "livekit-server")
        specs.append(
            ChildSpec(
                name="livekit",
                argv=[
                    livekit_bin,
                    "--dev",
                    "--bind", "0.0.0.0",
                    "--port", str(cfg.livekit_bind_port),
                    "--rtc-port", str(cfg.livekit_rtc_port),
                ],
                ready_url=None,  # LiveKit dev server has no consistent /health
                ready_timeout=30.0,
            )
        )

    # --- llama.cpp server (C++ binary) -------------------------------
    if cfg.manage_llama:
        llama_bin = os.getenv("LLAMA_BIN", "llama-server")
        specs.append(
            ChildSpec(
                name="llama",
                argv=[
                    llama_bin,
                    "--host", "127.0.0.1",
                    "--port", str(cfg.llama_bind_port),
                    "--hf-repo", cfg.llama_hf_repo,
                    "--alias", cfg.llama_model_alias,
                    "--ctx-size", str(cfg.llama_ctx_size),
                    "--n-gpu-layers", str(cfg.llama_n_gpu_layers),
                ],
                env={"HF_HOME": os.getenv("HF_HOME", "/models"), "XDG_CACHE_HOME": os.getenv("XDG_CACHE_HOME", "/models")},
                ready_url=f"http://127.0.0.1:{cfg.llama_bind_port}/v1/models",
                ready_timeout=900.0,  # first-run model download can be slow
            )
        )

    # --- STT (Nemotron or Whisper) -----------------------------------
    if cfg.manage_stt:
        if cfg.stt_provider == "whisper":
            specs.append(
                ChildSpec(
                    name="whisper",
                    argv=[
                        "vox-box", "start",
                        "--huggingface-repo-id", cfg.voxbox_hf_repo_id,
                        "--data-dir", os.getenv("VOXBOX_DATA_DIR", "/data"),
                        "--device", cfg.voxbox_device,
                        "--host", "127.0.0.1",
                        "--port", str(cfg.stt_bind_port),
                    ],
                    ready_url=f"http://127.0.0.1:{cfg.stt_bind_port}/v1/models",
                    ready_timeout=600.0,
                )
            )
        else:
            specs.append(
                ChildSpec(
                    name="nemotron",
                    argv=[
                        py, "-m", "local_voice_ai.services.nemotron.server",
                        "--host", "127.0.0.1",
                        "--port", str(cfg.stt_bind_port),
                    ],
                    env={
                        "NEMOTRON_MODEL_NAME": cfg.nemotron_model_name,
                        "NEMOTRON_MODEL_ID": cfg.nemotron_model_id,
                        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                    },
                    ready_url=f"http://127.0.0.1:{cfg.stt_bind_port}/health",
                    ready_timeout=600.0,
                )
            )

    # --- TTS (Kokoro) ------------------------------------------------
    if cfg.manage_tts:
        specs.append(
            ChildSpec(
                name="kokoro",
                argv=[
                    py, "-m", "local_voice_ai.services.kokoro.server",
                    "--host", "127.0.0.1",
                    "--port", str(cfg.tts_bind_port),
                ],
                ready_url=f"http://127.0.0.1:{cfg.tts_bind_port}/v1/models",
                ready_timeout=600.0,
            )
        )

    # --- Agent worker ------------------------------------------------
    specs.append(
        ChildSpec(
            name="agent",
            argv=[py, "-m", "local_voice_ai.agent", "start"],
            env=cfg.agent_env(),
            ready_url=None,
            ready_timeout=30.0,
        )
    )

    return specs


async def _serve(cfg: Config) -> int:
    specs = _build_specs(cfg)
    supervisor = Supervisor(specs)

    logger.info(
        "supervisor managing %d children (livekit=%s llama=%s stt=%s tts=%s)",
        len(specs),
        cfg.manage_livekit, cfg.manage_llama, cfg.manage_stt, cfg.manage_tts,
    )

    try:
        await supervisor.start_all()
    except Exception:
        logger.exception("startup failed; shutting down")
        await supervisor.shutdown()
        return 1

    app = build_app(cfg)
    uv_config = uvicorn.Config(
        app,
        host=cfg.web_host,
        port=cfg.web_port,
        log_level=cfg.log_level.lower(),
        access_log=False,
    )
    uv_server = uvicorn.Server(uv_config)

    web_task = asyncio.create_task(uv_server.serve(), name="web")
    sup_task = asyncio.create_task(supervisor.run_until_signal(), name="supervisor")

    done, _ = await asyncio.wait({web_task, sup_task}, return_when=asyncio.FIRST_COMPLETED)

    # Whatever finished first triggers a coordinated shutdown.
    uv_server.should_exit = True
    if not sup_task.done():
        await supervisor.shutdown()
    for task in (web_task, sup_task):
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    return 0


def _download_models(cfg: Config) -> int:
    """Pre-download VAD, turn-detector, Nemotron weights so first run is warm."""
    logger.info("downloading agent prewarm models (silero VAD, turn detector)")
    # Reuse livekit-agents' built-in download-files command
    import subprocess
    rc = subprocess.call([sys.executable, "-m", "local_voice_ai.agent", "download-files"])
    if rc != 0:
        return rc

    if cfg.manage_stt and cfg.stt_provider == "nemotron":
        logger.info("downloading nemotron model %s", cfg.nemotron_model_name)
        import nemo.collections.asr as nemo_asr  # type: ignore[import]
        nemo_asr.models.ASRModel.from_pretrained(cfg.nemotron_model_name)

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="local_voice_ai")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="run the full supervised stack (default)")
    sub.add_parser("download-models", help="pre-download model weights")
    sub.add_parser("console", help="run the agent in interactive console mode")

    args = parser.parse_args(argv)
    cfg = Config.from_env()
    configure_logging(cfg.log_level)

    cmd = args.cmd or "serve"
    if cmd == "serve":
        return asyncio.run(_serve(cfg))
    if cmd == "download-models":
        return _download_models(cfg)
    if cmd == "console":
        os.execv(
            sys.executable,
            [sys.executable, "-m", "local_voice_ai.agent", "console"],
        )
    parser.error(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
