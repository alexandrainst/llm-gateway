# MVP single-GPU orchestrator (realtime + batch)

## Run locally (MVP)
1. git clone this repo (put files as above)
2. Copy env example and adjust if needed: `cp .env.vllm.example .env`
3. Copy auth keys example and adjust tokens/scopes: `cp config/example.keys.json config/auth.keys.json`
4. docker-compose up --build

Services:
- orchestrator: http://localhost:8000
- runtime-sim: http://localhost:8001
- redis: localhost:6379

## Realtime demo (requires auth)
Token must have `realtime` scope:

curl -X POST \
  -H "Authorization: Bearer example-realtime-token" \
  -H "Content-Type: application/json" \
  "http://localhost:8000/v1/completions" \
  -d '{"model":"synquid/gemma-3-27b-it-FP8","prompt":"hello","max_tokens":16}'

Cold start may return 503 with Retry-After; retry after the delay.

## Batch demo (requires auth)
Token must have `batch` scope:

curl -X POST \
  -H "Authorization: Bearer example-batch-token" \
  -H "Content-Type: application/json" \
  "http://localhost:8000/v1/batch/completions" \
  -d '{"model":"mistralai/Mistral-7B-Instruct-v0.1","prompt":"Write a haiku about snow.","max_tokens":32,"priority":1}'

curl -H "Authorization: Bearer example-batch-token" \
  "http://localhost:8000/v1/batch/completions/<id>"

## Notes:
- runtime-sim simulates load times and inference; replace runtime service with your actual vLLM/SGLang container.
- In production, implement `download_model()` to snapshot_download HF models into /models, convert/quantize, and record real `load_est_s` in Redis model:<id> metadata.
- prefetching uses vmtouch/dd; for large models use page-cache priming as described in the earlier notes.

## Authentication (scoped)

- Keys load from a static JSON at startup; see `config/example.keys.json`.
- `AUTH_KEYS_FILE` must point to a container path (Compose mounts `./config/auth.keys.json` to `/app-config/auth.keys.json`).
- Scopes: `realtime`, `batch`, `upload`, `eval`, `monitor`.
- UI: `/dashboard` is public; UI data endpoints require a token with `monitor`. Use `?key=YOUR_TOKEN`.
- Reload keys: edit the JSON file and restart the orchestrator container.

## Use a Patched vLLM Runtime Image (optional)

If the published `vllm/vllm-openai:latest` has a Python dependency issue (e.g., NCCL), use the simple Dockerfile to rebuild on top of it and update only NCCL:

1) Build a minimal patched image (updates `nvidia-nccl-cu12`):

```
docker build -f runtime/Dockerfile \
  --build-arg NCCL_VERSION=2.21.5 \
  -t vllm-openai:nccl-patched .
```

2) Point the orchestrator to your patched image (set in `.env`):

```
RUNTIME_TYPE=vllm
RUNTIME_IMAGE=vllm-openai:nccl-patched
```

This avoids rebuilding the heavy vLLM image from source and only updates the NCCL wheel in the Python environment used by the server.
