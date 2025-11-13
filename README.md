# LLM Gateway Orchestrator (realtime + batch)

Single-node FastAPI service that keeps a realtime model hot while scheduling batch jobs to a GPU runtime container via Docker and Redis.

## Requirements
- Docker + Docker Compose with NVIDIA Container Toolkit
- Host Hugging Face cache directory exported via `HOST_HF_CACHE=/path/to/cache` (Compose binds it into containers at `/host_hf_cache`)
- `.env` based on `.env.vllm.example`
- Scoped auth keys file at `config/auth.keys.json` (copy `config/example.keys.json`)

## Quickstart
1. `cp .env.vllm.example .env`
2. `cp config/example.keys.json config/auth.keys.json`
3. `docker compose up --build`

Services (listening on `0.0.0.0`): API `http://<host>:8000`, UI `http://<host>:8000/dashboard`, metrics `http://<host>:8000/metrics` (requires `monitor` scope), Redis `redis://<host>:6379`.

## Configuration
| Variable | Purpose |
| --- | --- |
| `RUNTIME_TYPE` | `vllm` (default) or `sglang`; controls runtime image/args |
| `RUNTIME_IMAGE`, `RUNTIME_ARGS` | Docker image and CLI args passed to `llm_runtime` |
| `REALTIME_MODEL` | Model to keep hot for `/v1/completions` & `/v1/chat/completions` |
| `MAX_CONCURRENT_BATCH` | Upper bound on simultaneous batch requests |
| `AUTH_KEYS_FILE` | Container path to scoped key JSON (mounted via Compose) |
| `EUROEVAL_DEBUG_LOG` | Optional path for EuroEval adapter debug output |

Update `.env` and restart the orchestrator container to apply changes.

## Auth Scopes
| Scope | Grants |
| --- | --- |
| `realtime` | `/v1/completions`, `/v1/chat/completions`, `/v1/models` |
| `batch` | `/v1/batch/*`, `/v1/jobs/*` |
| `eval` | `/eval/*` endpoints from `eval_manager` |
| `upload` | `/v1/upload/*` TUS endpoints |
| `monitor` | `/dashboard`, `/metrics`, queue/eval UI JSON |

Scopes load once at startup; restart the container after editing `config/auth.keys.json`.

## Usage Tracking
- Every authenticated realtime or batch inference call updates per-key counters (requests, successes/errors, prompt/completion tokens) stored in Redis hashes under `key_usage:<key_id>`.
- Internal eval submissions that originate without a user key log against `INTERNAL_EVAL_KEY_ID` (default `__internal_eval__`) so ops traffic appears in the same usage dashboards.
- Daily buckets `key_usage:<key_id>::YYYYMMDD` retain roughly 90 days of history for simple day-over-day/month-over-month reporting.
- Streaming responses automatically request `stream_options.include_usage` so vLLM/SGLang emit usage blocks; ensure the runtime images you deploy support this flag.
- Operators with a `monitor` token can hit `GET /monitor/key_usage?limit=50&bucket=YYYYMMDD` to fetch the same data via the API, or inspect Redis directly (`redis-cli --scan --pattern 'key_usage:*'` and `HGETALL key_usage:<key_id>` for raw hashes).

## API Usage
Realtime:
```
curl -sS -X POST \
  -H "Authorization: Bearer $REALTIME_TOKEN" \
  -H "Content-Type: application/json" \
  http://localhost:8000/v1/completions \
  -d '{"model":"synquid/gemma-3-27b-it-FP8","prompt":"hello","max_tokens":16}'
```

Batch:
```
curl -sS -X POST \
  -H "Authorization: Bearer $BATCH_TOKEN" \
  -H "Content-Type: application/json" \
  http://localhost:8000/v1/batch/completions \
  -d '{"model":"mistralai/Mistral-7B-Instruct-v0.1","prompt":"Write a haiku","max_tokens":32,"priority":1}'
```
Poll the `status_url` from the 202 response until `status` becomes `completed`.

## Runtime & Logs
- Orchestrator starts/stops the `llm_runtime` container as needed; use `docker logs orchestrator` and `docker logs llm_runtime` for diagnostics.
- Runtime crashes persist short logs under `/host_hf_cache/runtime_failures`.
- EuroEval adapter writes to `$EUROEVAL_DEBUG_LOG` (defaults to `./euroeval_debug.log`).
- Build custom runtime images with `runtime/Dockerfile` (e.g., patched NCCL) and point `RUNTIME_IMAGE` to the new tag.

## Evaluations
`eval-repo/run_eval.py` implements the evaluation protocol:
- `python eval-repo/run_eval.py list [--base-model]`
- `python eval-repo/run_eval.py prepare <eval_name> <model> [options]`
- `python eval-repo/run_eval.py score <eval_name>` (stdin JSON)

See `eval-repo/PROTOCOL.md` for request/response schema details.

## Troubleshooting
- 503 with `Retry-After`: runtime is cold-starting; wait and retry.
- 401: token missing required scope (see table above).
- Batch job stuck: check Redis (`job:<id>` hash) and orchestrator logs for failure flags.
- No eval output: inspect `$EUROEVAL_DEBUG_LOG` and EuroEval stdout for tracebacks.
