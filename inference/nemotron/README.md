# Nemotron STT Service

This folder contains an OpenAI-compatible FastAPI wrapper for
`nvidia/nemotron-speech-streaming-en-0.6b`.

It exposes:

- `POST /v1/audio/transcriptions`
- `GET /v1/models`
- `GET /health`

The service is used by default in the root `docker-compose.yml` as the STT backend.

## Env vars

- `NEMOTRON_MODEL_NAME` (default: `nvidia/nemotron-speech-streaming-en-0.6b`)
- `NEMOTRON_MODEL_ID` (default: `nemotron-speech-streaming`)

## Notes

- First startup downloads model weights and can take a while.
- GPU builds are configured via `docker-compose.gpu.yml`.
- The root compose file maps host port `11435` to container `8000`.
