# orchestrator/app.py
import os, time, uuid, json, asyncio, subprocess, socket, collections, math, re, random, hashlib
import contextlib
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Response, Depends, APIRouter, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import redis.asyncio as aioredis
import logging
import httpx
import docker
from docker.types import DeviceRequest
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from model_utils import detect_is_base_model, CHAT_TEMPLATE_FILENAMES
from eval_manager import EvalManager, router as eval_router

log = structlog.get_logger()

# Config from env
# Auth: static scoped keys loaded from JSON file at startup (no fallback API_KEY)
AUTH_KEYS_FILE = os.getenv("AUTH_KEYS_FILE")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
# Path INSIDE orchestrator container where host HF cache is mounted
# Use the container-side mount path for the host HF cache consistently.
# Do not override via env to avoid confusion between host vs container paths.
HOST_HF_CACHE = "/host_hf_cache"
# Optional: explicit host path for HF cache (as seen by Docker daemon)
HOST_HF_CACHE_HOST = os.getenv("HOST_HF_CACHE_HOST", None)
RUNTIME_TYPE = os.getenv("RUNTIME_TYPE", "sglang")  # "sglang" or "vllm"
RUNTIME_IMAGE = os.getenv("RUNTIME_IMAGE", "lmsysorg/sglang:b200-cu129" if RUNTIME_TYPE == "sglang" else "vllm/vllm-openai:latest")
RUNTIME_ARGS = os.getenv("RUNTIME_ARGS", "--tp 2 --cuda-graph-max-bs 16" if RUNTIME_TYPE == "sglang" else "--tensor-parallel-size 2")
RUNTIME_PORT = int(os.getenv("RUNTIME_PORT", "30000"))
RUNTIME_API_PATH = os.getenv("RUNTIME_API_PATH", "/v1/completions")  # Both support OpenAI API
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "synquid/gemma-3-27b-it-FP8")
# Optional host path to persist vLLM torch.compile cache across restarts
VLLM_CACHE_HOST = os.getenv("VLLM_CACHE_HOST", "/home/alex-admin/.cache/vllm")
REALTIME_COOLDOWN_S = int(os.getenv("REALTIME_COOLDOWN_S", "120"))
MIN_HOT_TTL_S = int(os.getenv("MIN_HOT_TTL_S", "600"))
L_MAX_S = int(os.getenv("L_MAX_S", "600"))
P_MAX = float(os.getenv("P_MAX", "0.3"))
TOKENS_B = float(os.getenv("TOKENS_B", "3"))
REFILL_PER_SEC = float(os.getenv("TOKENS_REFILL_PER_SEC", "0.000555"))
MAX_CONCURRENT_BATCH = int(os.getenv("MAX_CONCURRENT_BATCH", "64"))
PREEMPT_CANCEL_TIMEOUT_S = float(os.getenv("PREEMPT_CANCEL_TIMEOUT_S", "3.0"))

USAGE_HASH_PREFIX = os.getenv("USAGE_HASH_PREFIX", "key_usage:")
USAGE_BUCKET_TTL_S = int(os.getenv("USAGE_BUCKET_TTL_S", str(90 * 86400)))
ENABLE_USAGE_DAILY_BUCKETS = os.getenv("ENABLE_USAGE_DAILY_BUCKETS", "1") not in ("0", "false", "False")
try:
    AUTH_KEYS_WATCH_INTERVAL_S = max(0.5, float(os.getenv("AUTH_KEYS_WATCH_INTERVAL_S", "1.5")))
except Exception:
    AUTH_KEYS_WATCH_INTERVAL_S = 1.5

# Readiness timeouts (seconds)
DEFAULT_READY_TIMEOUT_S = int(os.getenv("DEFAULT_READY_TIMEOUT_S", "120"))
VLLM_READY_TIMEOUT_S = int(os.getenv("VLLM_READY_TIMEOUT_S", "900"))

BATCH_ZSET = "z:batchjobs"
JOB_HASH_PREFIX = "job:"
MODEL_META_PREFIX = "model:"

CONTAINER_NAME = "llm_runtime"

app = FastAPI()
app.state.planner_task = None
app.state.auth_keys_watch_task = None
redis = None
docker_client = docker.from_env()

state = {
    "last_rt_ts": 0.0,
    "tokens": TOKENS_B,
    "lambda_ewma": 0.0,
    "alpha": 0.3,
    "current_model": None,
    "preempt": False,
    "loading_task": None,  # Track if model is already loading
    "loading_model": None,  # Track which model is being loaded
    # Runtime activity tracking
    "inflight_realtime": 0,
    "inflight_batch": 0,
    "last_runtime_req_ts": 0.0,
    "last_runtime_ok_ts": 0.0,
    "last_runtime_err_ts": 0.0,
    "last_warmup_ts": 0.0,
    "rt_cooldown_start_ts": 0.0,
}


def get_request_principal(request: Request) -> Optional[Dict[str, Any]]:
    return getattr(request.state, "auth_principal", None)


def _usage_hash_key(key_id: str, bucket: Optional[str] = None) -> str:
    if bucket:
        return f"{USAGE_HASH_PREFIX}{key_id}::{bucket}"
    return f"{USAGE_HASH_PREFIX}{key_id}"


async def _apply_usage_mutations(
    hash_key: str,
    *,
    count_increments: Dict[str, int],
    float_increments: Dict[str, float],
    metadata: Dict[str, Any],
    ttl: Optional[int] = None,
):
    if not redis:
        return
    pipe = redis.pipeline(transaction=False)
    for field, delta in count_increments.items():
        if delta:
            pipe.hincrby(hash_key, field, delta)
    for field, delta in float_increments.items():
        if delta:
            pipe.hincrbyfloat(hash_key, field, float(delta))
    if metadata:
        pipe.hset(hash_key, mapping={k: str(v) for k, v in metadata.items()})
    if ttl:
        pipe.expire(hash_key, ttl)
    try:
        await pipe.execute()
    except Exception as exc:
        log.warning("record_key_usage_failed", key=hash_key, error=str(exc))


async def record_key_usage(
    key_id: Optional[str],
    *,
    source: str,
    scope: Optional[str],
    model: Optional[str],
    usage: Optional[Dict[str, Any]],
    status: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    if not key_id or not redis:
        return
    now_ts = int(now())
    counts = {"request_total": 1}
    source_field = f"{source}_requests"
    counts[source_field] = counts.get(source_field, 0) + 1
    if status == "success":
        counts["success_total"] = 1
    else:
        counts["error_total"] = 1
    floats: Dict[str, float] = {}
    if isinstance(usage, dict):
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = usage.get(field)
            if isinstance(val, (int, float)):
                floats[field] = floats.get(field, 0.0) + float(val)
    info_fields: Dict[str, Any] = {
        "last_ts": now_ts,
        "last_status": status,
        "last_source": source,
    }
    if scope:
        info_fields["last_scope"] = scope
    if model:
        info_fields["last_model"] = model
    if metadata:
        info_fields.update({k: v for k, v in metadata.items() if v is not None})

    await _apply_usage_mutations(
        _usage_hash_key(key_id),
        count_increments=counts,
        float_increments=floats,
        metadata=info_fields,
    )

    if ENABLE_USAGE_DAILY_BUCKETS:
        bucket = time.strftime("%Y%m%d", time.gmtime(now_ts))
        await _apply_usage_mutations(
            _usage_hash_key(key_id, bucket=bucket),
            count_increments=counts,
            float_increments=floats,
            metadata={"bucket": bucket, **info_fields},
            ttl=USAGE_BUCKET_TTL_S,
        )


def _looks_like_runtime_oom(exit_code: int, logs: str, *, oom_killed: bool = False) -> bool:
    """Determine if a failure likely stems from an OOM condition."""
    if oom_killed:
        return True
    if exit_code == 137:  # SIGKILL (can be OOM or manual kill)
        # Require log corroboration to avoid false positives when we stop containers ourselves.
        if not logs:
            return False
    elif not logs:
        return False
    lowered = logs.lower()
    oom_markers = (
        "cuda out of memory",
        "out of memory",
        "resource exhausted",
        "hip error out of memory",
    )
    return any(marker in lowered for marker in oom_markers)


def persist_runtime_failure_log(model: Optional[str], reason: str, logs: str) -> Optional[str]:
    """Persist recent runtime logs for postmortem analysis."""
    if not logs:
        return None
    model_id = model or "unknown_model"
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model_id)
    safe_reason = re.sub(r"[^A-Za-z0-9._-]", "_", reason or "unknown_reason")
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    base_dir = Path(os.getenv("RUNTIME_FAILURE_LOG_DIR", "/host_hf_cache/runtime_failures"))
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        log_path = base_dir / f"{timestamp}_{safe_model}_{safe_reason}.log"
        log_path.write_text(logs, encoding="utf-8", errors="ignore")
        return str(log_path)
    except Exception as exc:
        log.warning("runtime_failure_log_write_failed", error=str(exc))
        return None

# -------- Reduce access log noise (healthz and UI polling) --------
class _AccessFilter(logging.Filter):
    IGNORE_PREFIXES = (
        "GET /healthz",
        "GET /ui/",
        "GET /metrics",
    )
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            req = getattr(record, "request_line", "")
        except Exception:
            req = ""
        msg = record.getMessage()
        line = req if isinstance(req, str) and req else (msg if isinstance(msg, str) else str(msg))
        for p in self.IGNORE_PREFIXES:
            if p in line:
                return False
        return True

if os.getenv("FILTER_ACCESS_LOG", "1") not in ("0", "false", "False"):
    try:
        logging.getLogger("uvicorn.access").addFilter(_AccessFilter())
    except Exception:
        pass

# ============ Prometheus Metrics ============
# Queue metrics
queue_depth = Gauge('llm_queue_depth', 'Current queue depth', ['type'])
jobs_total = Counter('llm_jobs_total', 'Total jobs processed', ['status', 'type'])

# Model metrics  
model_switches = Counter('llm_model_switches_total', 'Total model switches')
model_load_time = Histogram('llm_model_load_time_seconds', 'Model load time', ['model'])
current_model_info = Gauge('llm_current_model_info', 'Current model (1 for active)', ['model'])

# Performance metrics
request_duration = Histogram('llm_request_duration_seconds', 'Request duration', ['endpoint', 'status'])
batch_segment_duration = Histogram('llm_batch_segment_duration_seconds', 'Batch segment duration')

# Resource metrics
token_bucket = Gauge('llm_token_bucket_remaining', 'Remaining tokens for realtime protection')
runtime_ready_metric = Gauge('llm_runtime_ready', 'Runtime ready status (1=ready, 0=not ready)')

# Error metrics
model_failures = Counter('llm_model_failures_total', 'Model failures', ['model', 'reason'])

# ============ Authentication (Scoped Keys) ============
security = HTTPBearer(auto_error=False)

# In-memory key store: token -> {"id": str, "scopes": set[str]}
AUTH_KEYS: Dict[str, Dict[str, Union[str, set[str]]]] = {}
AUTH_KEYS_LAST_MTIME: Optional[float] = None

def load_auth_keys_from_file(path: str) -> None:
    """Load scoped auth keys from a JSON file.
    Schema: {"keys": [{"id": str, "token": str, "scopes": [str, ...]}]}
    """
    global AUTH_KEYS
    if not path or not os.path.exists(path):
        raise RuntimeError("AUTH_KEYS_FILE is required and must point to a readable JSON file")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("keys"), list):
        raise RuntimeError("AUTH_KEYS_FILE must contain an object with a 'keys' list")
    tmp: Dict[str, Dict[str, Union[str, set[str]]]] = {}
    for entry in data["keys"]:
        if not isinstance(entry, dict):
            continue
        kid = entry.get("id")
        token = entry.get("token")
        scopes = entry.get("scopes", [])
        if not kid or not token or not isinstance(scopes, list):
            continue
        tmp[str(token)] = {"id": str(kid), "scopes": set(map(str, scopes)), "token": str(token)}
    if not tmp:
        raise RuntimeError("AUTH_KEYS_FILE contains no valid keys")
    AUTH_KEYS = tmp

def _get_auth_keys_mtime(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    try:
        return os.path.getmtime(path)
    except OSError:
        return None

async def _watch_auth_keys_file():
    """Poll AUTH_KEYS_FILE for changes and hot-reload when it updates."""
    global AUTH_KEYS_LAST_MTIME
    path = AUTH_KEYS_FILE
    if not path:
        log.warning("auth_keys_watch_disabled_no_path")
        return
    interval = AUTH_KEYS_WATCH_INTERVAL_S
    log.info("auth_keys_watch_started", path=path, interval_s=interval)
    try:
        while True:
            try:
                await asyncio.sleep(interval)
                current_mtime = _get_auth_keys_mtime(path)
                if current_mtime is None:
                    if AUTH_KEYS_LAST_MTIME is not None:
                        log.warning("auth_keys_file_missing", path=path)
                        AUTH_KEYS_LAST_MTIME = None
                    continue
                if AUTH_KEYS_LAST_MTIME is None:
                    AUTH_KEYS_LAST_MTIME = current_mtime
                    continue
                if current_mtime != AUTH_KEYS_LAST_MTIME:
                    # Small delay allows writers to finish file replacements
                    await asyncio.sleep(0.05)
                    try:
                        load_auth_keys_from_file(path)
                        AUTH_KEYS_LAST_MTIME = current_mtime
                        log.info("auth_keys_hot_reloaded", path=path)
                    except Exception as exc:
                        log.warning("auth_keys_reload_failed", error=str(exc), path=path)
            except asyncio.CancelledError:
                raise
            except Exception as inner_exc:
                log.warning("auth_keys_watch_error", error=str(inner_exc))
                await asyncio.sleep(max(1.0, interval))
    except asyncio.CancelledError:
        log.info("auth_keys_watch_stopped")
    except Exception as exc:
        log.error("auth_keys_watch_crashed", error=str(exc))
        raise
def make_require_scopes(required: set[str]):
    async def _dep(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ):
        if not credentials or not credentials.credentials:
            raise HTTPException(status_code=401, detail="Missing authentication")
        token = credentials.credentials
        principal = AUTH_KEYS.get(token)
        if not principal:
            raise HTTPException(status_code=401, detail="Invalid token")
        scopes = principal.get("scopes", set())
        if not isinstance(scopes, set) or not required.issubset(scopes):
            raise HTTPException(status_code=403, detail="Insufficient scope")
        principal_copy = {
            "id": principal.get("id"),
            "scopes": scopes,
            "token": principal.get("token", token),
        }
        request.state.auth_principal = principal_copy
        return principal_copy

    return _dep

# -------- Routers --------
# v1: OpenAI-compatible and batch APIs
v1_router = APIRouter(prefix="/v1")

# uploads/UI routers
upload_router = APIRouter()
ui_router = APIRouter()

@app.get("/healthz")
async def healthz():
    """Simple liveness endpoint (no auth, fast path)."""
    return {"status": "ok"}

# Mount routers

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list[str] = None
    # Optional logprobs fields (OpenAI compat)
    # For vLLM Completions, logprobs should be an integer top-K
    logprobs: Optional[Union[int, bool]] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[dict] = None
    guided_json: Optional[Any] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[Any] = None
    guided_grammar: Optional[Any] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list[str] = None
    # Optional logprobs fields for chat
    # vLLM Chat expects logprobs: bool and top_logprobs: int
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[dict] = None
    guided_json: Optional[Any] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[Any] = None
    guided_grammar: Optional[Any] = None

class JobSubmit(BaseModel):
    model: str  # Changed from model_id for consistency
    input: str
    priority: int = 1
    estimate_s: float = 5.0

class BatchCompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list[str] = None
    # Batch-specific fields
    priority: int = 1
    logprobs: Optional[Union[int, bool]] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[dict] = None
    guided_json: Optional[Any] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[Any] = None
    guided_grammar: Optional[Any] = None

class BatchChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list[str] = None
    # Batch-specific fields
    priority: int = 1
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[dict] = None
    guided_json: Optional[Any] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[Any] = None
    guided_grammar: Optional[Any] = None

# ---------- redis connect ----------
async def redis_connect():
    """Connect to Redis with retries to tolerate startup ordering."""
    global redis
    attempts = int(os.getenv("REDIS_CONNECT_ATTEMPTS", "60"))
    delay_s = float(os.getenv("REDIS_CONNECT_BACKOFF_S", "0.5"))
    last_err = None
    for i in range(1, attempts + 1):
        try:
            redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            await redis.ping()
            log.info("redis_connected", url=REDIS_URL, attempt=i)
            return
        except Exception as e:
            last_err = e
            log.warning("redis_connect_retry", attempt=i, error=str(e))
            await asyncio.sleep(delay_s)
    # If we get here, fail startup; Compose will restart based on policy
    raise RuntimeError(f"Unable to connect to Redis after {attempts} attempts: {last_err}")

# ---------- utilities ----------
def now():
    return time.time()

async def cleanup_stuck_jobs():
    """Clean up any jobs stuck in 'running' state from previous orchestrator run."""
    cleaned_count = 0
    
    # Clean up stuck batch jobs (hashes under job:*)
    cursor = 0
    while True:
        cursor, keys = await redis.scan(cursor, match=f"{JOB_HASH_PREFIX}*", count=200)
        for key in keys:
            try:
                job_data = await redis.hgetall(key)
                if job_data and job_data.get("status") == "running":
                    await redis.hset(key, mapping={
                        "status": "error",
                        "error": "Orchestrator restarted",
                        "finished_ts": str(now())
                    })
                    cleaned_count += 1
                    log.info("cleaned_stuck_batch_job", key=key)
            except Exception as e:
                log.warning("cleanup_batch_job_error", key=key, error=str(e))
        if cursor == 0:
            break
    
    # Clean up stuck eval jobs
    cursor = 0
    while True:
        cursor, keys = await redis.scan(cursor, match="eval_job:*", count=100)
        for key in keys:
            try:
                eval_data = json.loads(await redis.get(key))
                if eval_data.get("status") == "running":
                    eval_data["status"] = "failed"
                    eval_data["error"] = "Orchestrator restarted during evaluation"
                    await redis.set(key, json.dumps(eval_data))
                    cleaned_count += 1
                    log.info("cleaned_stuck_eval_job", key=key)
            except Exception as e:
                log.error(f"Error cleaning eval job {key}: {e}")
        if cursor == 0:
            break
    
    if cleaned_count > 0:
        log.info("cleaned_stuck_jobs_total", count=cleaned_count)

    # Remove orphaned or inconsistent items from batch queue (zset)
    try:
        queued = await redis.zrange(BATCH_ZSET, 0, -1)
        removed = 0
        canceled = 0
        # Build set of active eval jobs (those still running)
        active_eval_jobs: set[str] = set()
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match="eval_job:*", count=200)
            for key in keys:
                try:
                    data_raw = await redis.get(key)
                    if not data_raw:
                        continue
                    data = json.loads(data_raw)
                    if data.get("status") == "running":
                        active_eval_jobs.add(key.split(":", 1)[1])
                except Exception:
                    pass
            if cursor == 0:
                break
        # Sweep zset
        for jid in queued:
            h = await redis.hgetall(JOB_HASH_PREFIX + jid)
            if not h:
                await redis.zrem(BATCH_ZSET, jid)
                removed += 1
                continue
            # Drop any non-queued items lingering in zset
            if h.get("status") and h.get("status") != "queued":
                await redis.zrem(BATCH_ZSET, jid)
                removed += 1
                continue
            # Cancel orphan eval-submitted jobs when their parent eval is not running
            if "eval_name" in h:
                evj = h.get("eval_job_id")
                if not evj or evj not in active_eval_jobs:
                    await redis.zrem(BATCH_ZSET, jid)
                    await redis.hset(JOB_HASH_PREFIX + jid, mapping={
                        "status": "canceled",
                        "error": "orphaned_on_startup",
                        "finished_ts": str(now())
                    })
                    canceled += 1
        if removed or canceled:
            log.info("cleanup_batch_queue_sweep", removed=removed, canceled=canceled)
    except Exception as e:
        log.warning("cleanup_batch_queue_error", error=str(e))

def ewma_update_interarrival(delta_t):
    if delta_t <= 0:
        return
    if state["lambda_ewma"] == 0:
        ewma_dt = delta_t
    else:
        prev_dt = 1.0 / state["lambda_ewma"] if state["lambda_ewma"] > 0 else delta_t
        ewma_dt = state["alpha"] * delta_t + (1 - state["alpha"]) * prev_dt
    state["lambda_ewma"] = 1.0 / ewma_dt

def refill_tokens():
    state["tokens"] = min(TOKENS_B, state["tokens"] + REFILL_PER_SEC * 1.0)

# ---------- runtime container control ----------
def start_runtime_container_sync(model_id, extra_args=None):
    """
    Start the runtime container (blocking call returning container object).
    It maps host HF cache into /root/.cache/huggingface and publishes RUNTIME_PORT on host.
    Returns container object on success, None on failure.
    """
    # stop any existing one first
    try:
        old = docker_client.containers.get(CONTAINER_NAME)
        try:
            old.remove(force=True)
        except Exception:
            pass
    except docker.errors.NotFound:
        pass
    
    # Check if this is an uploaded model and use local path if so
    # Use a synchronous Redis client for this blocking function
    import redis as sync_redis
    model_path = model_id
    try:
        sync_redis_client = sync_redis.from_url(REDIS_URL, decode_responses=True)
        model_meta = sync_redis_client.hgetall(f"{MODEL_META_PREFIX}{model_id}")
        sync_redis_client.close()
        
        if model_meta and model_meta.get("type") == "uploaded":
            # For uploaded models, use the local path in the container
            model_dir = f"models--{model_id.replace('/', '--')}"
            model_path = f"/root/.cache/huggingface/hub/{model_dir}/snapshots/main"
            log.info("using_local_path_for_uploaded_model", model=model_id, path=model_path)
    except Exception as e:
        log.error("error_checking_model_type", model=model_id, error=str(e))

    # Build command based on runtime type
    if RUNTIME_TYPE == "vllm":
        # For vLLM docker images, the entrypoint is already set, just pass args
        cmd = [
            "--model", model_path,
            "--host", "0.0.0.0",
            "--port", str(RUNTIME_PORT),  # Explicitly set port
            "--served-model-name", model_id  # Use original model ID for API
        ]
    else:  # sglang
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model", model_path,
            "--host", "0.0.0.0",
            "--port", str(RUNTIME_PORT)  # Explicitly set port
        ]
    
    # Add chat template automatically for vLLM if available and not already specified
    def _maybe_get_chat_template_arg(host_hf_cache_path: str) -> list[str]:
        try:
            base_dir = os.path.join(host_hf_cache_path, "hub", f"models--{model_id.replace('/', '--')}")
            snapshots_dir = os.path.join(base_dir, "snapshots")
            candidates: list[str] = []
            if os.path.isdir(snapshots_dir):
                main_dir = os.path.join(snapshots_dir, "main")
                if os.path.isdir(main_dir):
                    candidates.append(main_dir)
                hashed = []
                for name in os.listdir(snapshots_dir):
                    if name == "main":
                        continue
                    p = os.path.join(snapshots_dir, name)
                    if os.path.isdir(p):
                        try:
                            hashed.append((os.path.getmtime(p), p))
                        except Exception:
                            hashed.append((0, p))
                hashed.sort(key=lambda x: x[0], reverse=True)
                candidates.extend([p for _, p in hashed])
            for snap in candidates:
                for fname in CHAT_TEMPLATE_FILENAMES:
                    path = os.path.join(snap, fname)
                    if not os.path.isfile(path):
                        continue
                    try:
                        if not open(path, "r", encoding="utf-8").read().strip():
                            continue
                    except Exception:
                        continue
                    # Map host path to runtime path inside container
                    try:
                        rel = os.path.relpath(path, host_hf_cache_path)
                        in_container = os.path.join("/root/.cache/huggingface", rel)
                        return ["--chat-template", in_container]
                    except Exception:
                        return []
        except Exception as e:
            log.warning("chat_template_lookup_failed", model=model_id, error=str(e))
        return []

    # Add extra args or default args
    if extra_args:
        cmd += extra_args.split()
    else:
        cmd += RUNTIME_ARGS.split()
    
    # device request for GPUs
    dev_req = DeviceRequest(count=-1, capabilities=[["gpu"]])

    # Determine host path for HF cache mount so runtime container can reuse it
    def _detect_host_hf_cache() -> str:
        # 1) Honor explicit override if provided
        if HOST_HF_CACHE_HOST:
            return HOST_HF_CACHE_HOST
        # 2) Inspect this orchestrator container's mounts to find source for /host_hf_cache
        try:
            self_ctr = docker_client.containers.get("orchestrator")
            mounts = self_ctr.attrs.get("Mounts", [])
            for m in mounts:
                # Docker SDK uses 'Destination' for target path in container
                if m.get("Destination") == HOST_HF_CACHE:
                    src = m.get("Source")
                    if src:
                        return src
        except Exception as e:
            log.warning("hf_cache_host_detect_failed", error=str(e))
        # 3) Fallback to a common default
        return "/home/alex-admin/.cache/huggingface"

    host_hf_cache_path = _detect_host_hf_cache()
    log.info("using_host_hf_cache", host_path=host_hf_cache_path, container_path="/root/.cache/huggingface")

    if RUNTIME_TYPE == "vllm":
        try:
            existing_args = set(cmd)
            has_chat_arg = "--chat-template" in existing_args
            if not has_chat_arg:
                chat_args = _maybe_get_chat_template_arg(host_hf_cache_path)
                if chat_args:
                    cmd += chat_args
                    log.info("using_chat_template_for_runtime", model=model_id, chat_template=chat_args[-1])
        except Exception as e:
            log.warning("add_chat_template_arg_failed", model=model_id, error=str(e))
    
    # Log the command for debugging
    log.info("runtime_command", runtime_type=RUNTIME_TYPE, cmd=cmd, model_path=model_path)

    volumes = {
        # Map host HF cache into runtime container under its default HF location
        host_hf_cache_path: {"bind": "/root/.cache/huggingface", "mode": "rw"}
    }
    # Persist vLLM torch.compile cache to speed subsequent startups
    if RUNTIME_TYPE == "vllm" and VLLM_CACHE_HOST:
        try:
            volumes[VLLM_CACHE_HOST] = {"bind": "/root/.cache/vllm", "mode": "rw"}
        except Exception:
            pass

    # create container - join the same network as orchestrator
    # First, get the network the orchestrator is on
    network_name = "llm-gateway_default"  # Default compose network name
    try:
        orchestrator = docker_client.containers.get("orchestrator")
        networks = list(orchestrator.attrs['NetworkSettings']['Networks'].keys())
        if networks:
            network_name = networks[0]
        log.info("starting_runtime_container", model_id=model_id, network=network_name)
    except Exception as e:
        log.warning("could_not_determine_orchestrator_network", error=str(e), fallback_network=network_name)
    
    # Build container kwargs
    container_kwargs = {
        "detach": True,
        "name": CONTAINER_NAME,
        "device_requests": [dev_req],
        "volumes": volumes,
        "network": network_name,  # Use same network as orchestrator
        "shm_size": "32g",
        "ipc_mode": "host",
        "labels": {"gpu-owner": "orchestrator"},
        "stderr": True, 
        "stdout": True,
        "restart_policy": {"Name": "no"},
        "environment": {
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",  # Disable progress bars in non-TTY environment
            "TQDM_DISABLE": "1",  # Also disable tqdm globally
            "VLLM_SLEEP_WHEN_IDLE": "1",  # Reduce busy-spin CPU usage when idle
        }
    }
    
    # For vLLM, override entrypoint to ensure our args are used correctly
    if RUNTIME_TYPE == "vllm":
        container_kwargs["entrypoint"] = "python3 -m vllm.entrypoints.openai.api_server"
        container_kwargs["command"] = cmd
    else:
        container_kwargs["command"] = cmd
    
    try:
        container = docker_client.containers.run(
            RUNTIME_IMAGE,
            **container_kwargs
        )
        return container
    except Exception as e:
        log.error("failed_to_start_runtime_container", model_id=model_id, error=str(e))
        return None

def stop_runtime_container_sync():
    try:
        c = docker_client.containers.get(CONTAINER_NAME)
        c.remove(force=True)
    except docker.errors.NotFound:
        pass
    except Exception:
        pass

def is_runtime_ready():
    """Check if runtime container is running and ready"""
    try:
        container = docker_client.containers.get(CONTAINER_NAME)
        if container.status != "running":
            return False
        # Container exists and is running
        return True
    except docker.errors.NotFound:
        return False

def get_runtime_host():
    """Get the hostname for the runtime container - use container name in compose network"""
    return CONTAINER_NAME

async def send_warmup_request():
    """Send a warmup request to the runtime to prime it.
    Tries the inferred endpoint first, then falls back to the other if it fails.
    """
    try:
        # Rate-limit warmup to avoid redundant calls
        min_interval = float(os.getenv("WARMUP_MIN_INTERVAL_S", "30"))
        tnow = now()
        lastwu = float(state.get("last_warmup_ts", 0.0))
        if lastwu and (tnow - lastwu) < min_interval:
            log.debug("warmup_skipped_recent", since_s=tnow - lastwu, min_interval_s=min_interval)
            return
        runtime_host = get_runtime_host()
        runtime_url = f"http://{runtime_host}:{RUNTIME_PORT}"
        # Get the actual model name from the runtime
        async with httpx.AsyncClient(timeout=30) as client:
            models_response = await client.get(f"{runtime_url}/v1/models")
            if models_response.status_code == 200:
                models_data = models_response.json()
                if models_data.get("data"):
                    model_name = models_data["data"][0]["id"]
                else:
                    model_name = state.get("current_model", "model")
            else:
                model_name = state.get("current_model", "model")

            # Determine base vs instruction using local HF cache
            current_model_id = state.get("current_model") or state.get("loading_model") or model_name
            is_base_guess = detect_is_base_model(current_model_id)

            # Prepare both payloads
            comp_payload = {"model": model_name, "prompt": "Hello", "max_tokens": 1, "temperature": 0}
            chat_payload = {"model": model_name, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1, "temperature": 0}

            first = ("/v1/completions", comp_payload) if is_base_guess else ("/v1/chat/completions", chat_payload)
            second = ("/v1/chat/completions", chat_payload) if is_base_guess else ("/v1/completions", comp_payload)

            # Try first
            endpoint, payload = first
            resp = await client.post(f"{runtime_url}{endpoint}", json=payload)
            if resp.status_code == 200:
                log.info("warmup_request_sent", model=model_name, endpoint=endpoint, is_base=is_base_guess)
                state["last_warmup_ts"] = now()
                return
            log.warning("warmup_request_failed", model=model_name, endpoint=endpoint, status=resp.status_code, is_base=is_base_guess)

            # Fallback to the other endpoint
            endpoint2, payload2 = second
            resp2 = await client.post(f"{runtime_url}{endpoint2}", json=payload2)
            if resp2.status_code == 200:
                log.info("warmup_request_fallback_ok", model=model_name, endpoint=endpoint2, first_endpoint=endpoint, is_base_guess=is_base_guess)
                state["last_warmup_ts"] = now()
            else:
                log.warning("warmup_request_both_failed", model=model_name, first_endpoint=endpoint, first_status=resp.status_code, second_endpoint=endpoint2, second_status=resp2.status_code, is_base_guess=is_base_guess)
    except Exception as e:
        log.warning("warmup_request_error", error=str(e))

async def wait_for_runtime_ready(timeout=600):
    """Wait until the runtime reports ready. Returns (ok, failure_reason)."""
    kv_tune_attempted = False  # Prevent multiple auto-tune retries per startup
    runtime_host = get_runtime_host()
    deadline = time.time() + timeout
    log.info("waiting_for_runtime", host=runtime_host, port=RUNTIME_PORT, timeout=timeout)

    last_container_check = 0
    # Rate-limit HTTP readiness probe logs
    log_interval_s = float(os.getenv("READY_PROBE_LOG_INTERVAL_S", "15"))
    last_http_log = 0.0
    while time.time() < deadline:
        # Allow cooperative abort of a non-realtime load
        try:
            if state.get("load_abort_requested"):
                log.info("runtime_ready_wait_aborted")
                return False, None
        except Exception:
            pass
        # Check container health every 5 seconds
        if time.time() - last_container_check > 5:
            try:
                container = docker_client.containers.get(CONTAINER_NAME)
                if container.status == "exited":
                    # Container crashed - check exit code and logs
                    state_attrs = container.attrs.get('State', {})
                    exit_code = state_attrs.get('ExitCode', -1)
                    oom_killed = bool(state_attrs.get('OOMKilled'))
                    model = state.get("loading_model") or state.get("current_model")
                    logs = container.logs(tail=200).decode('utf-8')
                    # Detect vLLM KV cache insufficient error and auto-tune max sequence length once
                    if (RUNTIME_TYPE == "vllm" and not kv_tune_attempted 
                        and ("estimated maximum model length is" in logs)):
                        try:
                            m = re.search(r"estimated maximum model length is\s+(\d+)", logs)
                            if m:
                                suggested = int(m.group(1))
                                margin = int(os.getenv("MAX_MODEL_LEN_MARGIN_TOKENS", "4096"))
                                tuned = max(1024, suggested - margin)
                                kv_tune_attempted = True
                                log.warning(
                                    "kv_cache_insufficient_autotune",
                                    model=model, suggested_max_model_len=suggested, tuned_max_model_len=tuned,
                                )
                                # Restart with tuned max model len
                                await asyncio.get_running_loop().run_in_executor(
                                    None,
                                    lambda: start_runtime_container_sync(
                                        model,
                                        extra_args=f"{RUNTIME_ARGS} --max-model-len {tuned}"
                                    )
                                )
                                # Continue waiting within same timeout window
                                last_container_check = time.time()
                                await asyncio.sleep(1.0)
                                continue
                        except Exception as _e:
                            # Fall through to generic handling
                            pass
                    
                    # Exit code 137 = SIGKILL (often OOM killer)
                    # Exit code 139 = SIGSEGV (segfault)
                    # Exit code 1 = general error

                    failure_reason = "runtime_crashed"
                    logs_snippet = logs[-500:]

                    if _looks_like_runtime_oom(exit_code, logs, oom_killed=oom_killed):
                        failure_reason = "runtime_oom"
                        log.error("runtime_oom", model=model, exit_code=exit_code, logs=logs_snippet)
                        try:
                            if model:
                                model_failures.labels(model=model, reason="oom").inc()
                        except Exception:
                            pass
                        if model:
                            await redis.set(f"model_oom:{model}", str(now()), ex=86400)
                    elif "has no SGlang implementation" in logs or "not compatible with SGLang" in logs:
                        failure_reason = "model_incompatible"
                        log.error("model_not_supported", model=model, exit_code=exit_code, logs=logs_snippet)
                        try:
                            if model:
                                model_failures.labels(model=model, reason="incompatible").inc()
                        except Exception:
                            pass
                        if model:
                            await redis.set(f"model_incompatible:{model}", str(now()), ex=2592000)
                    elif exit_code == 139:
                        failure_reason = "runtime_segfault"
                        log.error("runtime_segfault", model=model, exit_code=exit_code, logs=logs_snippet)
                        try:
                            if model:
                                model_failures.labels(model=model, reason="segfault").inc()
                        except Exception:
                            pass
                        if model:
                            await redis.set(f"model_failure:{model}", str(now()), ex=21600)
                    else:
                        log.error("runtime_crashed", model=model, exit_code=exit_code, logs=logs_snippet)
                        try:
                            if model:
                                model_failures.labels(model=model, reason="crashed").inc()
                        except Exception:
                            pass
                        if model:
                            await redis.set(f"model_failure:{model}", str(now()), ex=3600)

                    log_path = persist_runtime_failure_log(model, failure_reason, logs)
                    if log_path:
                        log.info("runtime_failure_log_saved", model=model, reason=failure_reason, path=log_path)

                    return False, failure_reason
            except docker.errors.NotFound:
                pass
            last_container_check = time.time()

        if is_runtime_ready():
            # Try an HTTP check to ensure it's actually responding
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(f"http://{runtime_host}:{RUNTIME_PORT}/v1/models", timeout=5.0)
                    if r.status_code == 200:
                        log.info("runtime_ready", host=runtime_host)
                        # Send warmup request to prime the runtime
                        await send_warmup_request()
                        return True, None
            except Exception as e:
                now_ts = time.time()
                if (now_ts - last_http_log) >= log_interval_s or last_http_log == 0.0:
                    log.debug("runtime_not_responding_yet", host=runtime_host, error=str(e))
                    last_http_log = now_ts
        await asyncio.sleep(1.0)
    
    log.error("runtime_timeout", host=runtime_host, timeout=timeout)
    return False, "runtime_timeout"

async def http_health_ok(timeout_s: float = 3.0) -> bool:
    """Lightweight HTTP health probe without logging/warmup."""
    try:
        runtime_host = get_runtime_host()
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(f"http://{runtime_host}:{RUNTIME_PORT}/v1/models")
            return r.status_code == 200
    except Exception:
        return False

async def runtime_proxy(
    endpoint: str,
    payload: dict,
    timeout=120,
    source: str = "unknown",
    usage_context: Optional[Dict[str, Any]] = None,
):
    """Proxy OpenAI-compatible requests directly to runtime and optionally track usage."""

    from fastapi.responses import StreamingResponse

    runtime_host = get_runtime_host()
    url = f"http://{runtime_host}:{RUNTIME_PORT}{endpoint}"

    async def _record_usage_if_needed(
        usage_payload: Optional[Dict[str, Any]],
        status: str,
    ):
        if not usage_context:
            return
        await record_key_usage(
            usage_context.get("key_id"),
            source=usage_context.get("source", source),
            scope=usage_context.get("scope"),
            model=usage_context.get("model") or payload.get("model"),
            usage=usage_payload,
            status=status,
            metadata=usage_context.get("metadata"),
        )

    # Ensure streaming requests ask runtime to emit usage blocks
    if payload.get("stream", False):
        stream_options = payload.get("stream_options")
        if isinstance(stream_options, dict):
            if not stream_options.get("include_usage"):
                stream_options = dict(stream_options)
                stream_options["include_usage"] = True
                payload["stream_options"] = stream_options
        else:
            payload["stream_options"] = {"include_usage": True}

    # Update inflight counters
    if source == "realtime":
        state["inflight_realtime"] = max(0, state.get("inflight_realtime", 0)) + 1
    elif source == "batch":
        state["inflight_batch"] = max(0, state.get("inflight_batch", 0)) + 1
    state["last_runtime_req_ts"] = now()

    if payload.get("stream", False):
        async def stream_generator():
            usage_payload: Optional[Dict[str, Any]] = None
            remainder = ""
            usage_recorded = False
            track_usage = bool(usage_context)

            def _process_chunk_text(text: str):
                nonlocal remainder, usage_payload
                remainder += text
                while "\n\n" in remainder:
                    event, remainder = remainder.split("\n\n", 1)
                    if not event.strip():
                        continue
                    data_lines = []
                    for line in event.split("\n"):
                        line = line.strip("\r")
                        if line.startswith("data:"):
                            data_lines.append(line[5:].strip())
                    if not data_lines:
                        continue
                    data_blob = "\n".join(data_lines).strip()
                    if not data_blob or data_blob == "[DONE]":
                        continue
                    try:
                        parsed = json.loads(data_blob)
                    except Exception:
                        continue
                    if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict):
                        usage_payload = parsed["usage"]

            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream("POST", url, json=payload, timeout=timeout) as r:
                        if r.status_code >= 400:
                            state["last_runtime_err_ts"] = now()
                            try:
                                error_payload = await r.aread()
                                try:
                                    error_payload = json.loads(error_payload)
                                except Exception:
                                    error_payload = error_payload.decode("utf-8", errors="replace")
                            except Exception:
                                error_payload = "stream_error"
                            if not usage_recorded:
                                await _record_usage_if_needed(None, status="error")
                                usage_recorded = True
                            error_id = f"rt-{uuid.uuid4().hex[:8]}"
                            log.error(
                                "runtime_stream_error",
                                error_id=error_id,
                                endpoint=endpoint,
                                status_code=r.status_code,
                                error=error_payload,
                                model=payload.get("model"),
                            )
                            raise HTTPException(
                                status_code=r.status_code,
                                detail={
                                    "error": "runtime_stream_error",
                                    "error_id": error_id,
                                    "upstream_status": r.status_code,
                                    "upstream_error": error_payload,
                                    "endpoint": endpoint,
                                    "model": payload.get("model"),
                                },
                            )
                        async for chunk in r.aiter_bytes():
                            if track_usage and chunk:
                                try:
                                    _process_chunk_text(chunk.decode("utf-8", errors="ignore"))
                                except Exception:
                                    pass
                            yield chunk
                state["last_runtime_ok_ts"] = now()
            except HTTPException:
                if not usage_recorded:
                    await _record_usage_if_needed(None, status="error")
                    usage_recorded = True
                raise
            except Exception:
                state["last_runtime_err_ts"] = now()
                if not usage_recorded:
                    await _record_usage_if_needed(None, status="error")
                    usage_recorded = True
                raise
            else:
                if not usage_recorded:
                    await _record_usage_if_needed(usage_payload, status="success")
                    usage_recorded = True
            finally:
                if source == "realtime":
                    state["inflight_realtime"] = max(0, state.get("inflight_realtime", 0) - 1)
                elif source == "batch":
                    state["inflight_batch"] = max(0, state.get("inflight_batch", 0) - 1)

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Non-streaming request
    usage_recorded = False
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=timeout)
            if r.status_code >= 400:
                state["last_runtime_err_ts"] = now()
                try:
                    error_detail: Any = r.json()
                except Exception:
                    error_detail = r.text
                if not usage_recorded:
                    await _record_usage_if_needed(None, status="error")
                    usage_recorded = True
                error_id = f"rt-{uuid.uuid4().hex[:8]}"
                # Log rich context for visibility into runtime failures
                log.error(
                    "runtime_nonstream_error",
                    error_id=error_id,
                    endpoint=endpoint,
                    status_code=r.status_code,
                    error=error_detail,
                    model=payload.get("model"),
                )
                detail: dict[str, Any] = {
                    "error": "runtime_error",
                    "error_id": error_id,
                    "upstream_status": r.status_code,
                    "upstream_error": error_detail,
                    "endpoint": endpoint,
                    "model": payload.get("model"),
                }
                raise HTTPException(status_code=r.status_code, detail=detail)
            state["last_runtime_ok_ts"] = now()
            result = r.json()
            if not usage_recorded:
                usage_block = result.get("usage") if isinstance(result, dict) else None
                await _record_usage_if_needed(usage_block, status="success")
                usage_recorded = True
            return result
    except HTTPException:
        raise
    except httpx.TimeoutException as e:
        # Request timed out - likely due to high GPU load
        state["last_runtime_err_ts"] = now()
        error_id = f"rt-{uuid.uuid4().hex[:8]}"
        log.error(
            "runtime_proxy_timeout",
            error_id=error_id,
            endpoint=endpoint,
            timeout_s=timeout,
            error=str(e),
            model=payload.get("model"),
        )
        if not usage_recorded:
            await _record_usage_if_needed(None, status="error")
            usage_recorded = True
        detail: Dict[str, Any] = {
            "error": "runtime_proxy_timeout",
            "error_id": error_id,
            "message": (
                f"Request to runtime timed out after {timeout}s. "
                "The model may be overloaded due to high GPU utilization or processing "
                "a large batch of concurrent requests."
            ),
            "endpoint": endpoint,
            "model": payload.get("model"),
        }
        raise HTTPException(status_code=504, detail=detail)
    except Exception as e:
        # Network/serialization or other unexpected proxy-layer failure
        state["last_runtime_err_ts"] = now()
        error_id = f"rt-{uuid.uuid4().hex[:8]}"
        log.error(
            "runtime_proxy_exception",
            error_id=error_id,
            endpoint=endpoint,
            error=str(e) or repr(e),
            error_type=type(e).__name__,
            model=payload.get("model"),
            max_tokens=payload.get("max_tokens"),
            stream=payload.get("stream"),
        )
        if not usage_recorded:
            await _record_usage_if_needed(None, status="error")
            usage_recorded = True
        detail: Dict[str, Any] = {
            "error": "runtime_proxy_error",
            "error_id": error_id,
            "message": str(e) or repr(e),
            "endpoint": endpoint,
            "model": payload.get("model"),
        }
        raise HTTPException(status_code=500, detail=detail)
    finally:
        if source == "realtime":
            state["inflight_realtime"] = max(0, state.get("inflight_realtime", 0) - 1)
        elif source == "batch":
            state["inflight_batch"] = max(0, state.get("inflight_batch", 0) - 1)

# ---------- prefetch (page-cache) ----------
def prefetch_pagecache_blocking(model_id):
    """Prefetch model files to page cache using vmtouch or dd"""
    # HuggingFace hub structure: hub/models--org--name/snapshots/hash/
    # Replace "/" with "--" for the HF cache directory structure
    model_path_pattern = model_id.replace("/", "--")
    # Use the container mount path, not the HOST_HF_CACHE env var
    model_dir = os.path.join("/host_hf_cache", "hub", f"models--{model_path_pattern}")
    
    if not os.path.exists(model_dir):
        log.warning("model_not_found_for_prefetch", model_id=model_id, path=model_dir)
        return
    
    log.debug("prefetching_model_to_cache", model_id=model_id, path=model_dir)
    
    # Try vmtouch first (most efficient)
    from shutil import which
    if which("vmtouch"):
        try:
            # Touch all files in the model directory
            result = subprocess.run(["vmtouch", "-t", model_dir], 
                                  capture_output=True, text=True, check=False)
            if result.returncode == 0:
                log.info("vmtouch_success", model_id=model_id, output=result.stdout[:200])
            return
        except Exception as e:
            log.warning("vmtouch_failed", error=str(e))
    
    # Fallback: read model files to warm cache
    total_size = 0
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(('.bin', '.safetensors', '.pt', '.pth', '.gguf')):
                p = os.path.join(root, f)
                try:
                    size = os.path.getsize(p)
                    total_size += size
                    # Read in chunks to avoid memory issues
                    with open(p, 'rb') as fp:
                        while fp.read(1024 * 1024 * 10):  # 10MB chunks
                            pass
                except Exception:
                    pass
    
    log.debug("prefetch_complete", model_id=model_id, bytes_read=total_size)

async def prefetch_pagecache_async(model_id):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: prefetch_pagecache_blocking(model_id))

# ---------- Storage hierarchy management ----------
def check_model_on_disk(model_id: str) -> bool:
    """Check if model exists on disk in HuggingFace cache"""
    model_path_pattern = model_id.replace("/", "--")
    # Use the container mount path
    model_dir = os.path.join("/host_hf_cache", "hub", f"models--{model_path_pattern}")
    
    if not os.path.exists(model_dir):
        return False
    
    # Check if snapshots directory exists and has content
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        return False
    
    # Check if there's at least one snapshot with model files
    for snapshot in os.listdir(snapshots_dir):
        snapshot_path = os.path.join(snapshots_dir, snapshot)
        if os.path.isdir(snapshot_path):
            # Look for model files
            for f in os.listdir(snapshot_path):
                if f.endswith(('.bin', '.safetensors', '.pt', '.pth', '.gguf', '.json')):
                    return True
    
    return False

async def download_model_async(model_id: str) -> bool:
    """Download model from HuggingFace Hub"""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
        
        log.info("downloading_model", model_id=model_id)
        loop = asyncio.get_running_loop()
        
        # Get HF token from environment
        token = os.environ.get("HF_TOKEN")
        
        def download_blocking():
            return snapshot_download(
                repo_id=model_id,
                cache_dir="/host_hf_cache",  # Use container mount path
                token=token,
                local_dir_use_symlinks=False,
                resume_download=True
            )
        
        path = await loop.run_in_executor(None, download_blocking)
        log.info("model_downloaded", model_id=model_id, path=path)
        return True
    except (RepositoryNotFoundError, GatedRepoError) as e:
        # Definitive failure: mark model as failed and drop queued jobs
        log.error("model_access_denied", model_id=model_id, error=str(e), 
                 hint="Need HuggingFace token or model doesn't exist/is gated")
        try:
            await redis.set(f"model_failure:{model_id}", json.dumps({"reason": "access_denied", "ts": now()}))
            await drop_queued_jobs_for_model(model_id, reason="model_access_denied")
        except Exception:
            pass
        return False
    except Exception as e:
        error_msg = str(e)
        if ("401" in error_msg) or ("403" in error_msg) or ("authentication" in error_msg.lower()) or ("token" in error_msg.lower()):
            log.error("model_auth_required", model_id=model_id, error=error_msg,
                     hint="Set HF_TOKEN environment variable or check access")
            try:
                await redis.set(f"model_failure:{model_id}", json.dumps({"reason": "auth_required", "ts": now()}))
                await drop_queued_jobs_for_model(model_id, reason="model_auth_required")
            except Exception:
                pass
        else:
            log.error("model_download_failed", model_id=model_id, error=error_msg)
        return False

## Note: detect_is_base_model is imported from model_utils.
## Remove duplicate local implementation to avoid diverging behavior.

def get_model_size_on_disk(model_id: str) -> int:
    """Get size of model on disk in bytes"""
    model_path_pattern = model_id.replace("/", "--")
    model_dir = os.path.join(HOST_HF_CACHE, "hub", f"models--{model_path_pattern}")
    
    if not os.path.exists(model_dir):
        return 0
    
    total_size = 0
    for root, _, files in os.walk(model_dir):
        for f in files:
            try:
                total_size += os.path.getsize(os.path.join(root, f))
            except:
                pass
    
    return total_size

def get_disk_usage() -> Tuple[int, int, float]:
    """Get disk usage for HF cache: (used_bytes, total_bytes, percent_used)"""
    stat = shutil.disk_usage(HOST_HF_CACHE)
    return stat.used, stat.total, (stat.used / stat.total * 100)

async def ensure_disk_space(required_gb: float = 100.0) -> bool:
    """Ensure enough disk space by evicting old models if needed"""
    required_bytes = required_gb * 1024 * 1024 * 1024
    used, total, percent = get_disk_usage()
    free = total - used
    
    if free > required_bytes:
        return True
    
    log.warning("insufficient_disk_space", free_gb=free/(1024**3), required_gb=required_gb)
    
    # TODO: Implement LRU eviction of models (except pinned realtime model)
    # For now, just return False if not enough space
    return False

async def analyze_queue_storage_needs() -> Dict[str, Dict]:
    """Analyze queue and return storage requirements for each model"""
    if not redis:
        return {}
    
    needs = {}
    queued = await redis.zrange(BATCH_ZSET, 0, -1)
    
    for jid in queued:
        h = await redis.hgetall(JOB_HASH_PREFIX + jid)
        if not h:
            continue
        
        model_id = h.get("model")  # Batch jobs store it as "model"
        if not model_id:
            continue
        
        if model_id not in needs:
            needs[model_id] = {
                "job_count": 0,
                "total_priority": 0,
                "on_disk": check_model_on_disk(model_id),
                "size_bytes": get_model_size_on_disk(model_id) if check_model_on_disk(model_id) else 0,
                "in_page_cache": False,  # Would need more complex check
                "is_current": state.get("current_model") == model_id,
            }
        
        needs[model_id]["job_count"] += 1
        needs[model_id]["total_priority"] += int(h.get("priority", 1))

    return needs

async def get_model_block_reason(model_id: str) -> Optional[str]:
    """Return the active failure flag for a model, if any."""
    if not redis:
        return None
    keys = (
        f"model_oom:{model_id}",
        f"model_incompatible:{model_id}",
        f"model_failure:{model_id}",
    )
    try:
        flags = await redis.mget(*keys)
    except Exception:
        return None
    if not flags:
        return None
    if flags[0]:
        return "model_oom"
    if len(flags) > 1 and flags[1]:
        return "model_incompatible"
    if len(flags) > 2 and flags[2]:
        return "model_failure"
    return None

async def proactive_storage_management():
    """Proactively manage storage based on queue analysis"""
    needs = await analyze_queue_storage_needs()
    
    if not needs:
        return
    
    # Sort models by priority/job count
    models_by_importance = sorted(
        needs.items(),
        key=lambda x: x[1]["total_priority"] * x[1]["job_count"],
        reverse=True
    )
    
    # Download missing models in order of importance
    for model_id, info in models_by_importance:
        block_reason = await get_model_block_reason(model_id)
        if block_reason:
            log.debug("storage_skip_flagged_model", model_id=model_id, reason=block_reason)
            continue
        if not info["on_disk"] and not info["is_current"]:
            # Check disk space before downloading
            if await ensure_disk_space(100):  # Assume 100GB needed
                log.info("proactive_download_starting", model_id=model_id, job_count=info["job_count"])
                asyncio.create_task(download_model_async(model_id))
                break  # Download one at a time
    
    # Prefetch high-priority models that might be used soon
    for model_id, info in models_by_importance[:2]:  # Top 2 models
        block_reason = await get_model_block_reason(model_id)
        if block_reason:
            log.debug("prefetch_skip_flagged_model", model_id=model_id, reason=block_reason)
            continue
        if info["on_disk"] and not info["is_current"] and not info["in_page_cache"]:
            log.debug("proactive_prefetch", model_id=model_id)  # Changed to debug level
            asyncio.create_task(prefetch_pagecache_async(model_id))
            break  # Prefetch one at a time

# ---------- API & job endpoints ----------
@app.on_event("startup")
async def startup():
    global AUTH_KEYS_LAST_MTIME
    # Load scoped auth keys upfront
    load_auth_keys_from_file(AUTH_KEYS_FILE)
    AUTH_KEYS_LAST_MTIME = _get_auth_keys_mtime(AUTH_KEYS_FILE)
    app.state.auth_keys_watch_task = asyncio.create_task(_watch_auth_keys_file())

    await redis_connect()
    
    # Clean up any stuck jobs from previous run
    await cleanup_stuck_jobs()
    
    # Initialize eval system
    app.state.eval_manager = EvalManager(redis, None)  # batch_manager=None for now
    
    # Check if runtime container is already running
    if is_runtime_ready():
        try:
            runtime_host = get_runtime_host()
            async with httpx.AsyncClient() as client:
                r = await client.get(f"http://{runtime_host}:{RUNTIME_PORT}/v1/models", timeout=5.0)
                if r.status_code == 200:
                    models = r.json()
                    if models.get("data"):
                        model_id = models["data"][0]["id"]
                        state["current_model"] = model_id
                        log.info("found_existing_runtime", model_id=model_id)
                        
                        # Check if we should switch models based on pending work
                        batch_job_count = await redis.zcard(BATCH_ZSET) if redis else 0
                        
                        if batch_job_count == 0 and model_id != REALTIME_MODEL:
                            # No batch jobs, switch to realtime model
                            log.info("no_batch_jobs_switching_to_realtime", 
                                   current=model_id, target=REALTIME_MODEL)
                            asyncio.create_task(load_realtime_model_async())
                        elif batch_job_count > 0:
                            # Have batch jobs - check if current model can handle them
                            jobs_for_current = 0
                            all_jobs = await redis.zrange(BATCH_ZSET, 0, -1)
                            for jid in all_jobs[:100]:  # Sample first 100
                                job_data = await redis.hgetall(JOB_HASH_PREFIX + jid)
                                if job_data and job_data.get("model") == model_id:
                                    jobs_for_current += 1
                            
                            if jobs_for_current == 0:
                                # Current model has no jobs, should switch to one that does
                                log.info("current_model_has_no_jobs", 
                                       model=model_id, batch_jobs=batch_job_count)
                                # Let the batch planner handle switching to the right model
                            else:
                                log.info("keeping_current_model", 
                                       model=model_id, jobs_for_model=jobs_for_current)
        except Exception as e:
            log.warning("could_not_determine_model_from_runtime", error=str(e))
    else:
        # No runtime running - check what we need
        batch_job_count = await redis.zcard(BATCH_ZSET) if redis else 0
        if batch_job_count == 0:
            # No batch jobs - start realtime model
            log.info("starting_default_realtime_model", model=REALTIME_MODEL)
            asyncio.create_task(load_realtime_model_async())
        else:
            # Have batch jobs - let the planner decide which model to load
            log.info("batch_jobs_pending_planner_will_decide", count=batch_job_count)
    
    app.state.planner_task = asyncio.create_task(batch_planner_loop())

async def load_realtime_model_async():
    """Load and warm up the realtime model on startup"""
    try:
        # Kick off prefetch in the background so it doesn't block startup
        loop = asyncio.get_running_loop()
        loop.create_task(prefetch_pagecache_async(REALTIME_MODEL))
        t0 = time.time()
        await asyncio.get_running_loop().run_in_executor(None, lambda: start_runtime_container_sync(REALTIME_MODEL))
        # vLLM needs more time to start up
        startup_timeout = VLLM_READY_TIMEOUT_S if RUNTIME_TYPE == "vllm" else DEFAULT_READY_TIMEOUT_S
        ok, failure_reason = await wait_for_runtime_ready(timeout=startup_timeout)
        if ok:
            state["current_model"] = REALTIME_MODEL
            try:
                model_load_time.labels(model=REALTIME_MODEL).observe(time.time() - t0)
                # Count as a switch if we came from something else
                # (on cold start previous may be None)
                model_switches.inc()
            except Exception:
                pass
            log.info("realtime_model_loaded_on_startup", model=REALTIME_MODEL)
        elif failure_reason:
            log.error("realtime_model_failed_to_start", reason=failure_reason)
    except Exception as e:
        log.error("failed_to_load_realtime_on_startup", error=str(e))

async def ensure_realtime_model_loading():
    """Ensure realtime model is loading - returns existing task if already in progress"""
    # If recent startup failure recorded for realtime model, avoid restart loop
    try:
        if redis is not None:
            recent_fail = await redis.get(f"model_failure:{REALTIME_MODEL}")
            recent_oom = await redis.get(f"model_oom:{REALTIME_MODEL}")
            incompatible = await redis.get(f"model_incompatible:{REALTIME_MODEL}")
            if recent_fail or recent_oom or incompatible:
                log.warning("realtime_start_blocked_due_to_recent_failure", 
                            model=REALTIME_MODEL,
                            failure=bool(recent_fail), oom=bool(recent_oom), incompatible=bool(incompatible))
                return None
    except Exception:
        pass
    # If a model is currently loading, handle preemption semantics
    if state.get("loading_task") and not state["loading_task"].done():
        loading_model = state.get("loading_model")
        log.info("another_model_already_loading", loading_model=loading_model)
        # If a non-realtime model is loading, request an abort and wait briefly
        if loading_model and loading_model != REALTIME_MODEL:
            try:
                abort_done = state.get("load_abort_done_event")
                if not isinstance(abort_done, asyncio.Event):
                    abort_done = asyncio.Event()
                    state["load_abort_done_event"] = abort_done
                else:
                    # Reset to unsignaled for a fresh abort handshake
                    try:
                        abort_done.clear()
                    except Exception:
                        pass
                state["load_abort_requested"] = True
                try:
                    await asyncio.wait_for(abort_done.wait(), timeout=PREEMPT_CANCEL_TIMEOUT_S)
                    log.info("load_abort_completed", aborted_model=loading_model)
                except asyncio.TimeoutError:
                    log.warning("load_abort_timeout", model=loading_model, timeout_s=PREEMPT_CANCEL_TIMEOUT_S)
                finally:
                    # Clear abort request before starting realtime load to avoid aborting it
                    state["load_abort_requested"] = False
            except Exception as e:
                log.warning("load_abort_handshake_error", error=str(e))
        else:
            # Realtime already loading; reuse that task
            return state["loading_task"]
    
    # Create new loading task
    async def load_model_async():
        state["preempt"] = True
        state["loading_model"] = REALTIME_MODEL
        state["rt_cooldown_start_ts"] = now()
        # Handshake: request drain of active batch tasks before stopping runtime
        try:
            # Create or reuse a drain-done event for this preempt
            drain_done = state.get("preempt_drain_done_event")
            if not isinstance(drain_done, asyncio.Event):
                drain_done = asyncio.Event()
                state["preempt_drain_done_event"] = drain_done
            state["preempt_drain_requested"] = True
            # Allow scheduler a brief window to cancel and requeue inflight tasks
            try:
                await asyncio.wait_for(drain_done.wait(), timeout=PREEMPT_CANCEL_TIMEOUT_S)
                log.info("preempt_drain_completed")
            except asyncio.TimeoutError:
                log.warning("preempt_drain_timeout", timeout_s=PREEMPT_CANCEL_TIMEOUT_S)
        except Exception as e:
            log.warning("preempt_drain_handshake_error", error=str(e))
        try:
            log.info("starting_realtime_model_load", model=REALTIME_MODEL)
            loop = asyncio.get_running_loop()
            loop.create_task(prefetch_pagecache_async(REALTIME_MODEL))
            await asyncio.get_running_loop().run_in_executor(None, stop_runtime_container_sync)
            t0 = time.time()
            await asyncio.get_running_loop().run_in_executor(None, lambda: start_runtime_container_sync(REALTIME_MODEL))
            load_timeout = VLLM_READY_TIMEOUT_S if RUNTIME_TYPE == "vllm" else DEFAULT_READY_TIMEOUT_S
            ok, failure_reason = await wait_for_runtime_ready(timeout=load_timeout)
            if ok:
                prev_model = state.get("current_model")
                state["current_model"] = REALTIME_MODEL
                try:
                    model_load_time.labels(model=REALTIME_MODEL).observe(time.time() - t0)
                    if prev_model != REALTIME_MODEL:
                        model_switches.inc()
                except Exception:
                    pass
                log.info("realtime_model_ready", model=REALTIME_MODEL)
                state["rt_cooldown_start_ts"] = 0.0
            elif failure_reason:
                log.error("realtime_model_load_failed", reason=failure_reason)
        except Exception as e:
            log.error("realtime_model_load_failed", error=str(e))
        finally:
            state["preempt"] = False
            state["preempt_drain_requested"] = False
            state["load_abort_requested"] = False
            try:
                dd = state.get("preempt_drain_done_event")
                if isinstance(dd, asyncio.Event):
                    # Reset the event for future use
                    dd.clear()
            except Exception:
                pass
            state["loading_task"] = None
            state["loading_model"] = None
    
    # Start new loading task
    state["loading_task"] = asyncio.create_task(load_model_async())
    return state["loading_task"]

@app.on_event("shutdown")
async def shutdown():
    if app.state.planner_task:
        app.state.planner_task.cancel()
        await asyncio.sleep(0.1)
    watch_task = getattr(app.state, "auth_keys_watch_task", None)
    if watch_task:
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watch_task

# ---------- UI Endpoints ----------
@ui_router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the dashboard UI from a static HTML file only."""
    static_path = Path(__file__).parent / "static" / "dashboard.html"
    if static_path.exists():
        return FileResponse(static_path, media_type="text/html")
    return HTMLResponse("Dashboard file not found (orchestrator/static/dashboard.html)", status_code=500) 

@ui_router.get("/ui/summary")
async def ui_summary(auth: bool = Depends(make_require_scopes({"monitor"}))):
    """Summarized state for the dashboard."""
    current = state.get("current_model")
    # Container status + quick HTTP health
    container_status = "not_found"
    try:
        c = docker_client.containers.get(CONTAINER_NAME)
        container_status = c.attrs.get("State", {}).get("Status", getattr(c, "status", "unknown"))
    except docker.errors.NotFound:
        container_status = "not_found"

    http_ok = False
    try:
        runtime_host = get_runtime_host()
        async with httpx.AsyncClient(timeout=0.5) as client:
            r = await client.get(f"http://{runtime_host}:{RUNTIME_PORT}/v1/models")
            http_ok = (r.status_code == 200)
    except Exception:
        http_ok = False

    ready = (container_status == "running" and http_ok)
    tokens_val = float(state.get("tokens", 0.0))
    lam = float(state.get("lambda_ewma", 0.0))
    last_ts = float(state.get("last_rt_ts", 0.0))
    last_age = max(0.0, now() - last_ts) if last_ts > 0 else -1.0
    depth = await redis.zcard(BATCH_ZSET) if redis else 0

    # Best guess for next/target model based on queue head (if any)
    next_model = None
    if redis and depth > 0:
        try:
            head = await redis.zrange(BATCH_ZSET, 0, 0)
            if head:
                head_job = await redis.hgetall(JOB_HASH_PREFIX + head[0])
                next_model = head_job.get("model") if head_job else None
        except Exception:
            next_model = None

    # Loading state
    loading_model = state.get("loading_model")
    switching = bool(state.get("loading_task"))

    # Derive a friendly runtime status string
    if ready:
        runtime_status = "ready"
    elif switching and loading_model:
        runtime_status = f"loading {loading_model}"
    elif container_status in ("created", "restarting"):
        runtime_status = "starting"
    elif container_status == "running":
        runtime_status = "initializing"
    elif container_status == "exited":
        runtime_status = "stopped"
    elif container_status == "not_found":
        runtime_status = "not running"
    else:
        runtime_status = container_status or "unknown"

    return {
        "current_model": current,
        "runtime_ready": bool(ready),
        "runtime_status": runtime_status,
        "container_status": container_status,
        "runtime_http_ok": http_ok,
        "loading_model": loading_model,
        "switching": switching,
        "next_model": next_model,
        "inflight_realtime": int(state.get("inflight_realtime", 0)),
        "inflight_batch": int(state.get("inflight_batch", 0)),
        "inflight_total": int(state.get("inflight_realtime", 0)) + int(state.get("inflight_batch", 0)),
        "last_runtime_req_age_s": (now() - state.get("last_runtime_req_ts", 0.0)) if state.get("last_runtime_req_ts", 0.0) else None,
        "last_runtime_ok_age_s": (now() - state.get("last_runtime_ok_ts", 0.0)) if state.get("last_runtime_ok_ts", 0.0) else None,
        "last_runtime_err_age_s": (now() - state.get("last_runtime_err_ts", 0.0)) if state.get("last_runtime_err_ts", 0.0) else None,
        "tokens": tokens_val,
        "lambda_ewma": lam,
        "last_rt_age_s": last_age,
        "queue_depth": depth,
    }

@ui_router.get("/ui/queue")
async def ui_queue(limit: int = 100, auth: bool = Depends(make_require_scopes({"monitor"}))):
    """Return a sample of queued jobs (and statuses)."""
    ids = await redis.zrange(BATCH_ZSET, 0, max(0, limit - 1)) if redis else []
    items = []
    for jid in ids:
        h = await redis.hgetall(JOB_HASH_PREFIX + jid)
        if not h:
            continue
        items.append({
            "id": jid,
            "model": h.get("model"),
            "status": h.get("status"),
            "priority": int(h.get("priority", 0)) if h.get("priority") else 0,
            "type": h.get("type", "completion"),
            "submit_ts": float(h.get("submit_ts", 0.0)) if h.get("submit_ts") else None,
        })
    return {"items": items, "queued_total": await redis.zcard(BATCH_ZSET)}

@ui_router.get("/ui/evals")
async def ui_evals(limit: int = 50, auth: bool = Depends(make_require_scopes({"monitor"}))):
    """List evaluation jobs by scanning eval_job:* keys (best-effort)."""
    items = []
    if not redis:
        return {"items": items}
    cursor = 0
    scanned = 0
    results = []
    while True:
        cursor, keys = await redis.scan(cursor, match="eval_job:*", count=200)
        for k in keys:
            try:
                data = await redis.get(k)
                if not data:
                    continue
                job = json.loads(data)
                results.append({
                    "job_id": job.get("job_id"),
                    "model": job.get("model"),
                    "status": job.get("status"),
                    "created_at": job.get("created_at"),
                })
            except Exception:
                continue
            scanned += 1
        if cursor == 0 or len(results) >= limit:
            break
    # Sort by created_at desc
    def _key(j):
        return j.get("created_at") or ""
    results.sort(key=_key, reverse=True)
    return {"items": results[:limit]}

@ui_router.get("/ui/runtime/logs")
async def ui_runtime_logs(
    tail: int = 200,
    since_s: float | None = None,
    auth: bool = Depends(make_require_scopes({"monitor"})),
):
    """Return recent runtime container logs for the UI (auth-protected).
    Params:
      - tail: number of lines from the end (default 200)
      - since_s: only logs since now - since_s seconds
    """
    try:
        c = docker_client.containers.get(CONTAINER_NAME)
        status = c.attrs.get("State", {}).get("Status", getattr(c, "status", "unknown"))
        kwargs = {"tail": tail, "timestamps": True}
        if since_s and since_s > 0:
            import time as _time
            kwargs["since"] = int(_time.time() - float(since_s))
        raw = c.logs(**kwargs)
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        # Soft limit to avoid huge payloads
        if len(text) > 120_000:
            text = text[-120_000:]
        return {"container_status": status, "text": text, "tail": tail}
    except docker.errors.NotFound:
        return {"container_status": "not_found", "text": "(runtime container not found)", "tail": tail}
    except Exception as e:
        return {"container_status": "error", "text": f"(error fetching logs: {e})", "tail": tail}

@ui_router.get("/ui/eval_matrix")
async def ui_eval_matrix(auth: bool = Depends(make_require_scopes({"monitor"}))):
    """Return an aggregated matrix of latest eval results.
    Rows = models, Columns = evals (including nested expansions, e.g., euroeval datasets).
    Each column exposes up to two metrics chosen by priority.
    """
    base = Path("/data/eval_results")
    if not base.exists():
        return {"columns": [], "rows": [], "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # Helpers
    metric_priority = [
        "accuracy", "acc", "strict_accuracy", "macro_f1", "f1",
        "exact_match", "em",
        # Prefer RULER aggregates in this order
        "avg", "wavg_inc", "score",
        # Other common metrics
        "bleu", "rougeL", "pearson", "spearman"
    ]

    def pick_metrics(candidates: set[str]) -> list[str]:
        # Choose up to 2 by priority; if fewer available, fill by sorted name
        chosen = []
        for m in metric_priority:
            if m in candidates:
                chosen.append(m)
            if len(chosen) == 2:
                return chosen
        # Fill from remaining (deterministic order)
        remaining = sorted([m for m in candidates if m not in chosen])
        for m in remaining:
            chosen.append(m)
            if len(chosen) == 2:
                break
        return chosen

    # Collect per-model results and available column metrics
    columns: dict[str, dict] = {}  # key -> {label, metrics_set:set}
    rows_map: dict[str, dict] = {}  # model -> {col_key -> {metric->value}}

    def _iter_jsonl_tail(path: Path, block_size: int = 1024 * 1024):
        try:
            with open(path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                pos = f.tell()
                leftover = b''
                while pos > 0:
                    step = min(block_size, pos)
                    pos -= step
                    f.seek(pos)
                    chunk = f.read(step)
                    data = chunk + leftover
                    lines = data.split(b"\n")
                    leftover = lines[0]
                    # yield full lines from end to start
                    for ln in reversed(lines[1:]):
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            yield ln.decode('utf-8', errors='ignore')
                        except Exception:
                            continue
                # any remaining line at start
                if leftover and leftover.strip():
                    try:
                        yield leftover.decode('utf-8', errors='ignore')
                    except Exception:
                        pass
        except Exception:
            return

    for model_dir in base.iterdir():
        if not model_dir.is_dir():
            continue
        # Prefer consolidated JSONL with history; pick the latest per eval_name by scanning tail-first
        jsonl = model_dir / "results.jsonl"
        latest_by_eval: dict[str, dict] = {}
        if jsonl.exists():
            for line in _iter_jsonl_tail(jsonl):
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                ev = entry.get("eval_name")
                if not ev or ev in latest_by_eval:
                    # already have latest for this eval
                    continue
                latest_by_eval[ev] = entry
                # Optional stop condition if file is huge: if we collected many evals
                if len(latest_by_eval) >= 200:
                    break
        
        if not latest_by_eval:
            # Fallback to previous behavior: look at latest job directory
            latest_dir = None
            latest_link = model_dir / "latest"
            try:
                if latest_link.exists():
                    latest_dir = latest_link.resolve()
            except Exception:
                latest_dir = None
            if latest_dir is None or not latest_dir.exists():
                subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
                if not subdirs:
                    continue
                latest_dir = sorted(subdirs)[-1]
            for f in latest_dir.glob("*_results.json"):
                try:
                    with open(f, "r") as fh:
                        data = json.load(fh)
                    ev = data.get("eval_name") or f.stem.replace("_results", "")
                    latest_by_eval[ev] = data
                except Exception:
                    continue

        # Now aggregate latest_by_eval into matrix rows/columns
        for ev, data in latest_by_eval.items():
            model_name = data.get("model") or model_dir.name
            metrics = data.get("metrics", {}) or {}
            if not isinstance(metrics, dict) or not metrics:
                continue
            rows_map.setdefault(model_name, {})

            if ev == "euroeval":
                by_ds: dict[str, dict] = {}
                for k, v in metrics.items():
                    if not isinstance(v, (int, float)):
                        continue
                    if "_" not in k:
                        continue
                    ds, m = k.split("_", 1)
                    if not ds:
                        continue
                    by_ds.setdefault(ds, {})[m] = v
                for ds, mvals in by_ds.items():
                    col_key = f"euroeval:{ds}"
                    col_label = f"euroeval • {ds}"
                    c = columns.setdefault(col_key, {"label": col_label, "metrics_set": set()})
                    c["metrics_set"].update(list(mvals.keys()))
                    rows_map[model_name].setdefault(col_key, {})
                    rows_map[model_name][col_key].update(mvals)
            elif ev == "lm-harness":
                by_task: dict[str, dict] = {}
                for k, v in metrics.items():
                    if not isinstance(v, (int, float)):
                        continue
                    if ":" not in k:
                        continue
                    task, m = k.split(":", 1)
                    if not task:
                        continue
                    by_task.setdefault(task, {})[m] = v
                for task, mvals in by_task.items():
                    col_key = f"lm-harness:{task}"
                    col_label = f"lm-harness • {task}"
                    c = columns.setdefault(col_key, {"label": col_label, "metrics_set": set()})
                    c["metrics_set"].update(list(mvals.keys()))
                    rows_map[model_name].setdefault(col_key, {})
                    rows_map[model_name][col_key].update(mvals)
            else:
                col_key = ev
                col_label = ev
                numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                if not numeric:
                    continue
                c = columns.setdefault(col_key, {"label": col_label, "metrics_set": set()})
                c["metrics_set"].update(list(numeric.keys()))
                rows_map[model_name].setdefault(col_key, {})
                rows_map[model_name][col_key].update(numeric)

    # Finalize columns with chosen metrics
    ordered_cols = []
    for key, meta in columns.items():
        metrics_list = pick_metrics(set(meta.get("metrics_set", set())))
        ordered_cols.append({"key": key, "label": meta.get("label", key), "metrics": metrics_list})
    # Stable order: euroeval groups first (alphabetical), then others
    def col_sort(c):
        return (0 if c["key"].startswith("euroeval:") else 1, c["label"]) 
    ordered_cols.sort(key=col_sort)

    # Build rows
    rows = []
    for model in sorted(rows_map.keys()):
        entry = {"model": model, "data": {}}
        for col in ordered_cols:
            col_key = col["key"]
            src = rows_map[model].get(col_key, {})
            # Only include chosen metrics
            entry["data"][col_key] = {m: src.get(m) for m in col["metrics"] if m in src}
        rows.append(entry)

    return {
        "columns": ordered_cols,
        "rows": rows,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

@v1_router.get("/models")
async def list_models(auth: bool = Depends(make_require_scopes({"realtime"}))):
    """OpenAI-compatible models endpoint - returns only the realtime model"""
    return {
        "object": "list",
        "data": [
            {
                "id": REALTIME_MODEL,
                "object": "model",
                "created": 1686935002,
                "owned_by": "system",
            }
        ]
    }

@app.get("/metrics")
async def metrics(auth: bool = Depends(make_require_scopes({"monitor"}))):
    """Prometheus metrics endpoint"""
    # Update runtime ready metric
    runtime_ready_metric.set(1 if is_runtime_ready() else 0)
    
    # Update token bucket metric
    token_bucket.set(state.get("tokens", 0))
    
    # Update current model metric
    current = state.get("current_model")
    if current:
        # Clear all model labels first
        current_model_info._metrics.clear()
        current_model_info.labels(model=current).set(1)
    
    return Response(generate_latest(), media_type="text/plain")


@app.get("/monitor/key_usage")
async def monitor_key_usage(
    limit: int = 100,
    bucket: Optional[str] = None,
    cursor: str = "0",
    auth: bool = Depends(make_require_scopes({"monitor"})),
):
    """Return per-key usage aggregates for operators."""
    if not redis:
        return {"items": [], "cursor": "0", "bucket": bucket}
    limit = max(1, min(limit, 500))
    scan_cursor = 0
    try:
        scan_cursor = int(cursor)
    except Exception:
        scan_cursor = 0
    pattern = f"{USAGE_HASH_PREFIX}*"
    if bucket:
        pattern = f"{USAGE_HASH_PREFIX}*::{bucket}"

    def _as_int(val: Optional[str]) -> Optional[int]:
        if val is None:
            return None
        try:
            return int(float(val))
        except Exception:
            return None

    def _as_float(val: Optional[str]) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None

    items = []
    while True:
        scan_cursor, keys = await redis.scan(scan_cursor, match=pattern, count=limit * 2)
        for key in keys:
            data = await redis.hgetall(key)
            if not data:
                continue
            suffix = key[len(USAGE_HASH_PREFIX):]
            key_bucket = None
            key_id = suffix
            if "::" in suffix:
                key_id, key_bucket = suffix.split("::", 1)
            entry = {
                "key_id": key_id,
                "bucket": key_bucket,
                "request_total": _as_int(data.get("request_total")) or 0,
                "realtime_requests": _as_int(data.get("realtime_requests")) or 0,
                "batch_requests": _as_int(data.get("batch_requests")) or 0,
                "success_total": _as_int(data.get("success_total")) or 0,
                "error_total": _as_int(data.get("error_total")) or 0,
                "prompt_tokens": _as_float(data.get("prompt_tokens")) or 0.0,
                "completion_tokens": _as_float(data.get("completion_tokens")) or 0.0,
                "total_tokens": _as_float(data.get("total_tokens")) or 0.0,
                "last_model": data.get("last_model"),
                "last_source": data.get("last_source"),
                "last_scope": data.get("last_scope"),
                "last_status": data.get("last_status"),
                "last_ts": _as_float(data.get("last_ts")),
            }
            items.append(entry)
            if len(items) >= limit:
                break
        if len(items) >= limit or scan_cursor == 0:
            break
    return {"items": items, "cursor": str(scan_cursor), "bucket": bucket}

@v1_router.post("/completions")
async def completions(
    req: CompletionRequest,
    response: Response,
    request: Request,
    auth: Dict[str, Any] = Depends(make_require_scopes({"realtime"})),
):
    """OpenAI-compatible completions endpoint"""
    principal = get_request_principal(request)
    usage_context = None
    if principal and principal.get("id"):
        usage_context = {
            "key_id": principal["id"],
            "source": "realtime",
            "scope": "realtime",
            "metadata": {"endpoint": "/v1/completions"},
        }
    # Update realtime arrival tracking for EWMA + hot TTL
    try:
        t_now = now()
        prev = state.get("last_rt_ts", 0.0) or 0.0
        if prev > 0:
            ewma_update_interarrival(t_now - prev)
        state["last_rt_ts"] = t_now
    except Exception:
        # Non-fatal; keep request path resilient
        pass
    # Check if request is for realtime model
    if req.model != REALTIME_MODEL:
        # If not realtime model, could enqueue as batch or reject
        raise HTTPException(
            status_code=400,
            detail={
                "error": "model_not_available",
                "message": f"Model '{req.model}' not available for realtime on this endpoint.",
                "requested_model": req.model,
                "available_realtime_model": REALTIME_MODEL,
            },
        )
    
    # Check if realtime model is hot
    if state.get("current_model") == REALTIME_MODEL and is_runtime_ready():
        try:
            payload = req.dict(exclude_none=True)
            # Adjust logprobs semantics for completions: vLLM expects integer top-K
            lp = payload.get("logprobs")
            tlp = payload.get("top_logprobs")
            if lp is not None:
                if isinstance(lp, bool):
                    if lp:
                        payload["logprobs"] = int(tlp) if isinstance(tlp, int) and tlp > 0 else 8
                    else:
                        payload.pop("logprobs", None)
                elif isinstance(lp, int) and lp <= 0:
                    payload.pop("logprobs", None)
            # Remove top_logprobs for completions
            if "top_logprobs" in payload:
                payload.pop("top_logprobs", None)
            return await runtime_proxy(
                "/v1/completions",
                payload,
                source="realtime",
                usage_context=usage_context,
            )
        except HTTPException as e:
            # Preserve upstream HTTPException details (including structured runtime error info)
            raise e
        except httpx.ConnectError as e:
            # Runtime is actually down
            log.error("completions_proxy_connect_error", error=str(e))
            state["current_model"] = None
            # Fall through to cold start
        except Exception as e:
            # Request-level error, runtime is still up
            error_id = f"rt-{uuid.uuid4().hex[:8]}"
            log.error(
                "completions_proxy_error",
                error_id=error_id,
                error=str(e) or repr(e),
                error_type=type(e).__name__,
                model=req.model,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "gateway_proxy_error",
                    "error_id": error_id,
                    "message": str(e) or repr(e),
                    "endpoint": "/v1/completions",
                    "model": req.model,
                },
            )
    
    # Model needs to be loaded - return 503 with dynamic Retry-After
    # Initialize cooldown start if not set yet
    if not state.get("rt_cooldown_start_ts"):
        state["rt_cooldown_start_ts"] = now()
    tnow = now()
    start = float(state.get("rt_cooldown_start_ts", tnow))
    remaining = (start + REALTIME_COOLDOWN_S) - tnow
    retry_after = int(math.ceil(remaining)) if remaining > 0 else 0
    if retry_after < 10:
        retry_after = 10
    response.status_code = 503
    response.headers["Retry-After"] = str(retry_after)
    log.info("realtime_model_cold", model=REALTIME_MODEL, retry_after_s=retry_after)

    # Start loading in background (or reuse existing loading task)
    asyncio.create_task(ensure_realtime_model_loading())

    return {
        "error": "Model is loading, please retry",
        "type": "model_cold_start",
        "retry_after": retry_after,
        "estimated_ready_in_seconds": retry_after,
    }

@v1_router.post("/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    response: Response,
    request: Request,
    auth: Dict[str, Any] = Depends(make_require_scopes({"realtime"})),
):
    """OpenAI-compatible chat completions endpoint"""
    principal = get_request_principal(request)
    usage_context = None
    if principal and principal.get("id"):
        usage_context = {
            "key_id": principal["id"],
            "source": "realtime",
            "scope": "realtime",
            "metadata": {"endpoint": "/v1/chat/completions"},
        }
    # Update realtime arrival tracking for EWMA + hot TTL
    try:
        t_now = now()
        prev = state.get("last_rt_ts", 0.0) or 0.0
        if prev > 0:
            ewma_update_interarrival(t_now - prev)
        state["last_rt_ts"] = t_now
    except Exception:
        # Non-fatal; keep request path resilient
        pass
    # Check if request is for realtime model
    if req.model != REALTIME_MODEL:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "model_not_available",
                "message": f"Model '{req.model}' not available for realtime on this endpoint.",
                "requested_model": req.model,
                "available_realtime_model": REALTIME_MODEL,
            },
        )
    
    # Check if realtime model is hot
    if state.get("current_model") == REALTIME_MODEL and is_runtime_ready():
        try:
            payload = req.dict(exclude_none=True)
            return await runtime_proxy(
                "/v1/chat/completions",
                payload,
                source="realtime",
                usage_context=usage_context,
            )
        except HTTPException as e:
            # Preserve upstream HTTPException details (including structured runtime error info)
            raise e
        except httpx.ConnectError as e:
            # Runtime is actually down
            log.error("chat_completions_proxy_connect_error", error=str(e))
            state["current_model"] = None
            # Fall through to cold start
        except Exception as e:
            # Request-level error, runtime is still up
            err_detail = getattr(e, "detail", None)
            error_id = f"rt-{uuid.uuid4().hex[:8]}"
            log.error(
                "chat_completions_proxy_error",
                error_id=error_id,
                error=str(e) or repr(e),
                error_type=type(e).__name__,
                detail=err_detail,
                model=req.model,
                max_tokens=payload.get("max_tokens"),
                messages_count=len(payload.get("messages", [])),
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "gateway_proxy_error",
                    "error_id": error_id,
                    "message": str(e) or "runtime_proxy_error",
                    "endpoint": "/v1/chat/completions",
                    "model": req.model,
                },
            )

    # Model needs to be loaded - return 503 with dynamic Retry-After
    if not state.get("rt_cooldown_start_ts"):
        state["rt_cooldown_start_ts"] = now()
    tnow = now()
    start = float(state.get("rt_cooldown_start_ts", tnow))
    remaining = (start + REALTIME_COOLDOWN_S) - tnow
    retry_after = int(math.ceil(remaining)) if remaining > 0 else 0
    if retry_after < 10:
        retry_after = 10
    response.status_code = 503
    response.headers["Retry-After"] = str(retry_after)
    log.info("realtime_model_cold", model=REALTIME_MODEL, retry_after_s=retry_after)

    # Start loading in background (or reuse existing loading task)
    asyncio.create_task(ensure_realtime_model_loading())

    return {
        "error": "Model is loading, please retry",
        "type": "model_cold_start",
        "retry_after": retry_after,
        "estimated_ready_in_seconds": retry_after,
    }

@v1_router.post("/batch/completions")
async def batch_completions(
    req: BatchCompletionRequest,
    response: Response,
    request: Request,
    auth: Dict[str, Any] = Depends(make_require_scopes({"batch"})),
):
    """Submit a batch completion request"""
    # Generate completion ID
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    submit_ts = now()
    principal = get_request_principal(request)
    key_id = principal.get("id") if principal else None
    
    # Store in Redis with TTL of 7 days (604800 seconds)
    jobkey = JOB_HASH_PREFIX + completion_id
    jobdata = {
        "id": completion_id,
        "type": "completion",
        "model": req.model,
        "prompt": req.prompt,
        "max_tokens": str(req.max_tokens),
        "temperature": str(req.temperature),
        "top_p": str(req.top_p),
        "n": str(req.n),
        "stop": json.dumps(req.stop) if req.stop else "null",
        "logprobs": json.dumps(req.logprobs) if req.logprobs is not None else "null",
        "top_logprobs": json.dumps(req.top_logprobs) if req.top_logprobs is not None else "null",
        "response_format": json.dumps(req.response_format) if req.response_format else "null",
        "guided_json": json.dumps(req.guided_json) if req.guided_json is not None else "null",
        "guided_regex": json.dumps(req.guided_regex) if req.guided_regex is not None else "null",
        "guided_choice": json.dumps(req.guided_choice) if req.guided_choice is not None else "null",
        "guided_grammar": json.dumps(req.guided_grammar) if req.guided_grammar is not None else "null",
        "priority": str(req.priority),
        "status": "queued",
        "submit_ts": str(submit_ts),
        "created": str(int(submit_ts))
    }
    if key_id:
        jobdata["key_id"] = key_id
    
    await redis.hset(jobkey, mapping=jobdata)
    await redis.expire(jobkey, 604800)  # 7 days TTL
    await redis.zadd(BATCH_ZSET, {completion_id: submit_ts})
    # Clear any prior definitive failure to allow re-attempt on new submission
    try:
        await redis.delete(f"model_failure:{req.model}")
    except Exception:
        pass
    
    # Return 202 Accepted with ID and status URL
    response.status_code = 202
    return {
        "id": completion_id,
        "status": "queued",
        "status_url": f"/v1/batch/completions/{completion_id}"
    }

@v1_router.get("/batch/completions/{completion_id}")
async def get_batch_completion(completion_id: str, auth: bool = Depends(make_require_scopes({"batch"}))):
    """Get status/result of a batch completion"""
    jobkey = JOB_HASH_PREFIX + completion_id
    data = await redis.hgetall(jobkey)
    
    if not data:
        raise HTTPException(status_code=404, detail="Completion not found")
    
    status = data.get("status", "unknown")
    
    # If still queued or running, return status
    if status in ["queued", "running"]:
        return {
            "id": completion_id,
            "status": status,
            "created": int(data.get("created", 0)),
            "position_estimate": await redis.zrank(BATCH_ZSET, completion_id) if status == "queued" else None
        }
    
    # If completed, return full OpenAI-compatible response
    if status == "completed":
        result = json.loads(data.get("result", "{}"))
        # Don't delete on retrieval - let TTL handle it
        return result
    
    # If error
    if status == "error":
        return {
            "id": completion_id,
            "status": "error",
            "error": data.get("error", "Unknown error")
        }
    
    return {"id": completion_id, "status": status}

@v1_router.post("/batch/chat/completions")
async def batch_chat_completions(
    req: BatchChatCompletionRequest,
    response: Response,
    request: Request,
    auth: Dict[str, Any] = Depends(make_require_scopes({"batch"})),
):
    """Submit a batch chat completion request"""
    # Generate completion ID
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    submit_ts = now()
    principal = get_request_principal(request)
    key_id = principal.get("id") if principal else None
    
    # Store in Redis with TTL
    jobkey = JOB_HASH_PREFIX + completion_id
    jobdata = {
        "id": completion_id,
        "type": "chat_completion",
        "model": req.model,
        "messages": json.dumps(req.messages),
        "max_tokens": str(req.max_tokens),
        "temperature": str(req.temperature),
        "top_p": str(req.top_p),
        "n": str(req.n),
        "stop": json.dumps(req.stop) if req.stop else "null",
        "logprobs": json.dumps(req.logprobs) if req.logprobs is not None else "null",
        "top_logprobs": json.dumps(req.top_logprobs) if req.top_logprobs is not None else "null",
        "response_format": json.dumps(req.response_format) if req.response_format else "null",
        "guided_json": json.dumps(req.guided_json) if req.guided_json is not None else "null",
        "guided_regex": json.dumps(req.guided_regex) if req.guided_regex is not None else "null",
        "guided_choice": json.dumps(req.guided_choice) if req.guided_choice is not None else "null",
        "guided_grammar": json.dumps(req.guided_grammar) if req.guided_grammar is not None else "null",
        "priority": str(req.priority),
        "status": "queued",
        "submit_ts": str(submit_ts),
        "created": str(int(submit_ts))
    }
    if key_id:
        jobdata["key_id"] = key_id
    
    await redis.hset(jobkey, mapping=jobdata)
    await redis.expire(jobkey, 604800)  # 7 days TTL
    await redis.zadd(BATCH_ZSET, {completion_id: submit_ts})
    # Clear prior failure on new submission for this model
    try:
        await redis.delete(f"model_failure:{req.model}")
    except Exception:
        pass
    
    # Return 202 Accepted
    response.status_code = 202
    return {
        "id": completion_id,
        "status": "queued",
        "status_url": f"/v1/batch/chat/completions/{completion_id}"
    }

@v1_router.get("/batch/chat/completions/{completion_id}")
async def get_batch_chat_completion(completion_id: str, auth: bool = Depends(make_require_scopes({"batch"}))):
    """Get status/result of a batch chat completion"""
    jobkey = JOB_HASH_PREFIX + completion_id
    data = await redis.hgetall(jobkey)
    
    if not data:
        raise HTTPException(status_code=404, detail="Chat completion not found")
    
    status = data.get("status", "unknown")
    
    # If still queued or running, return status
    if status in ["queued", "running"]:
        return {
            "id": completion_id,
            "status": status,
            "created": int(data.get("created", 0)),
            "position_estimate": await redis.zrank(BATCH_ZSET, completion_id) if status == "queued" else None
        }
    
    # If completed, return full response
    if status == "completed":
        result = json.loads(data.get("result", "{}"))
        return result
    
    # If error
    if status == "error":
        return {
            "id": completion_id,
            "status": "error",
            "error": data.get("error", "Unknown error")
        }
    
    return {"id": completion_id, "status": status}


@v1_router.post("/jobs")
async def submit_job(
    j: JobSubmit,
    request: Request,
    auth: Dict[str, Any] = Depends(make_require_scopes({"batch"})),
):
    job_id = str(uuid.uuid4())
    submit_ts = now()
    jobkey = JOB_HASH_PREFIX + job_id
    principal = get_request_principal(request)
    key_id = principal.get("id") if principal else None
    jobdata = {
        "job_id": job_id,
        "model": j.model,
        "input": j.input,
        "priority": str(j.priority),
        "estimate_s": str(j.estimate_s),
        "submit_ts": str(submit_ts),
        "status": "queued"
    }
    if key_id:
        jobdata["key_id"] = key_id
    await redis.hset(jobkey, mapping=jobdata)
    await redis.zadd(BATCH_ZSET, {job_id: submit_ts})
    # Clear prior failure on new submission for this model
    try:
        await redis.delete(f"model_failure:{j.model}")
    except Exception:
        pass
    return {"job_id": job_id}

@v1_router.get("/jobs/{job_id}")
async def get_job(job_id: str, auth: bool = Depends(make_require_scopes({"batch"}))):
    jobkey = JOB_HASH_PREFIX + job_id
    data = await redis.hgetall(jobkey)
    if not data:
        raise HTTPException(status_code=404, detail="job not found")
    return data

# ---------- batch planner ----------
async def batch_planner_loop():
    last_storage_check = 0
    last_queue_state = None  # Track queue state to detect changes
    last_ranking_log = 0  # Track when we last logged ranking
    
    while True:
        try:
            await asyncio.sleep(1.0)
            refill_tokens()
            
            # Run proactive storage management every 10 seconds
            current_time = now()
            if current_time - last_storage_check >= 10:
                asyncio.create_task(proactive_storage_management())
                last_storage_check = current_time

            queued = await redis.zrange(BATCH_ZSET, 0, -1)
            try:
                queue_depth.labels(type="batch").set(len(queued))
            except Exception:
                pass
            if not queued:
                eval_mgr = getattr(app.state, "eval_manager", None)
                pending_submitters = getattr(eval_mgr, "submission_in_progress", 0) if eval_mgr else 0
                if pending_submitters:
                    log.debug(
                        "batch_queue_empty_waiting_for_submission",
                        pending_submitters=pending_submitters,
                    )
                    await asyncio.sleep(1.0)
                    continue
                # If there are no batch jobs and we're not on the realtime model, switch back proactively
                if state.get("current_model") != REALTIME_MODEL:
                    loading_task = state.get("loading_task")
                    loading_model = state.get("loading_model")
                    already_loading_rt = bool(loading_task) and (not loading_task.done()) and (loading_model == REALTIME_MODEL)
                    if not already_loading_rt:
                        log.info(
                            "no_batch_jobs_switching_to_realtime_proactive",
                            from_model=state.get("current_model"),
                            to_model=REALTIME_MODEL,
                        )
                        try:
                            await ensure_realtime_model_loading()
                        except Exception as e:
                            log.error("failed_switch_to_realtime_proactive", error=str(e))
                await asyncio.sleep(1.0)
                continue
            
            # Check if queue has changed
            current_queue_state = tuple(queued)  # Make it hashable
            queue_changed = (current_queue_state != last_queue_state)
            last_queue_state = current_queue_state

            # If a model switch is in progress, avoid claiming/probing until it completes
            lt = state.get("loading_task")
            if (lt and not lt.done()) or state.get("preempt"):
                log.debug("switch_in_progress_skip_scheduling", loading_model=state.get("loading_model"))
                await asyncio.sleep(0.5)
                continue

            jobs_by_model = {}
            for jid in queued:
                h = await redis.hgetall(JOB_HASH_PREFIX + jid)
                if not h:
                    await redis.zrem(BATCH_ZSET, jid)
                    continue
                # Only consider jobs that are queued
                if h.get("status") and h.get("status") != "queued":
                    await redis.zrem(BATCH_ZSET, jid)
                    continue
                mid = h.get("model")
                priority = int(h.get("priority", "1"))
                est = float(h.get("estimate_s", "5.0"))
                jobs_by_model.setdefault(mid, []).append((jid, priority, est))

            # Create priority-ranked list of models
            # Score = job_count * average_priority (simple and effective)
            model_ranking = []
            for mid, items in jobs_by_model.items():
                # Skip models that recently failed (download, OOM, or incompatible)
                failure_key = f"model_failure:{mid}"
                oom_key = f"model_oom:{mid}"
                incompatible_key = f"model_incompatible:{mid}"

                try:
                    failure, oom, incompatible = await redis.mget(failure_key, oom_key, incompatible_key)
                except Exception:
                    failure = oom = incompatible = None

                # If prior failure was a download/availability issue but the model is now on disk,
                # clear the transient failure flag so the scheduler can proceed.
                if failure and check_model_on_disk(mid):
                    try:
                        await redis.delete(failure_key)
                        failure = None
                    except Exception:
                        pass

                if failure or oom or incompatible:
                    block_reason = "model_failure"
                    if oom:
                        block_reason = "model_oom"
                    elif incompatible:
                        block_reason = "model_incompatible"

                    log.debug(
                        "skipping_failed_model",
                        model=mid,
                        failure=bool(failure),
                        oom=bool(oom),
                        incompatible=bool(incompatible),
                    )

                    if items:
                        try:
                            await drop_queued_jobs_for_model(mid, reason=block_reason)
                        except Exception as drop_err:
                            log.warning("drop_jobs_for_flagged_model_failed", model=mid, error=str(drop_err))
                    continue
                    
                job_count = len(items)
                total_priority = sum(p for (_, p, _) in items)
                avg_priority = total_priority / job_count if job_count > 0 else 1
                score = job_count * avg_priority
                model_ranking.append((mid, score, job_count))
            
            # Sort by score descending
            model_ranking.sort(key=lambda x: x[1], reverse=True)
            
            # Proactively download top 2 models that aren't on disk (in background)
            for i, (mid, score, count) in enumerate(model_ranking[:2]):
                if not check_model_on_disk(mid):
                    log.info("proactive_model_download", model=mid, rank=i+1, score=score)
                    # Don't await - let it download in background
                    asyncio.create_task(download_model_async(mid))
            
            # Log the ranking only if changed or every 30 seconds
            if model_ranking and (queue_changed or current_time - last_ranking_log >= 30):
                log.debug("model_ranking", 
                         ranking=[(m, f"score={s:.1f}", f"jobs={c}") 
                                 for m, s, c in model_ranking[:3]])  # Top 3
                last_ranking_log = current_time
            
            # Simple strategy: stick with current model if it has ANY jobs
            current_model = state.get("current_model")
            if current_model in jobs_by_model:
                best_model = current_model
                # Only log periodically to avoid spam
                if queue_changed or current_time - last_ranking_log >= 30:
                    log.debug("sticking_with_current_model", 
                             model=current_model, 
                             job_count=len(jobs_by_model[current_model]))
            elif model_ranking:
                # Select highest-ranked model
                best_model = model_ranking[0][0]
            else:
                best_model = None

            if not best_model:
                await asyncio.sleep(1.0)
                continue

            # Check if we need to switch models and if we're allowed to
            need_switch = (best_model != current_model)
            
            if need_switch:
                # Check if we can leave the current model
                can_leave_rt = True
                reasons_blocked = []
                
                if state["tokens"] < 1:
                    can_leave_rt = False
                    reasons_blocked.append(f"no_tokens (have {state['tokens']})")
                    
                if current_model == REALTIME_MODEL:
                    time_since_rt = now() - state["last_rt_ts"]
                    if time_since_rt < MIN_HOT_TTL_S:
                        can_leave_rt = False
                        reasons_blocked.append(f"min_hot_ttl ({time_since_rt:.0f}s < {MIN_HOT_TTL_S}s)")

                if not can_leave_rt:
                    # Log why we're blocked (only once per minute to avoid spam)
                    if int(now()) % 60 == 0:
                        log.debug("model_switch_blocked", 
                                 from_model=current_model,
                                 to_model=best_model,
                                 reasons=reasons_blocked)
                    await asyncio.sleep(1.0)
                    continue
                    
                # We can switch!
                log.info("switching_to_new_model", 
                        from_model=current_model, 
                        to_model=best_model,
                        score=model_ranking[0][1] if model_ranking else 0,
                        job_count=len(jobs_by_model[best_model]))

            # Claim ALL jobs for the current model (no L-second limit)
            claimed = []
            all_queued = await redis.zrange(BATCH_ZSET, 0, -1)
            for jid in all_queued:
                h = await redis.hgetall(JOB_HASH_PREFIX + jid)
                if not h:
                    await redis.zrem(BATCH_ZSET, jid)
                    continue
                if h.get("model") != best_model:
                    continue
                removed = await redis.zrem(BATCH_ZSET, jid)
                if removed:
                    await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "running", "started_ts": str(now())})
                    claimed.append(jid)

            if not claimed:
                await asyncio.sleep(0.5)
                continue
            
            # CRITICAL: Ensure model exists on disk before evicting realtime
            if not check_model_on_disk(best_model):
                log.info("batch_model_not_on_disk_downloading", model=best_model)
                success = await download_model_async(best_model)
                if not success:
                    log.error("batch_model_download_failed", model=best_model)
                    
                    # Mark this model as failed so we don't retry it immediately
                    # Store failure timestamp in Redis to implement backoff
                    failure_key = f"model_failure:{best_model}"
                    await redis.set(failure_key, str(now()), ex=3600)  # Remember failure for 1 hour
                    
                    # Requeue all claimed jobs with their original submit time
                    for jid in claimed:
                        job_data = await redis.hgetall(JOB_HASH_PREFIX + jid)
                        original_ts = float(job_data.get("submit_ts", now()))
                        await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "queued"})
                        await redis.zadd(BATCH_ZSET, {jid: original_ts})
                    await asyncio.sleep(10)
                    continue
            
            # Only restart runtime if we're switching models
            if best_model != state.get("current_model"):
                # Only spend token if we're actually going to evict realtime
                if state.get("current_model") == REALTIME_MODEL:
                    state["tokens"] = max(0.0, state["tokens"] - 1.0)
                    log.info("batch_evicting_realtime", tokens_remaining=state["tokens"])

                # Expose loading state for observability while we switch to the batch model
                state["loading_model"] = best_model

                async def _load_batch_model():
                    failure_reason = None

                    async def _requeue_claimed_jobs():
                        for jid in claimed:
                            job_data = await redis.hgetall(JOB_HASH_PREFIX + jid)
                            if not job_data:
                                continue
                            original_ts = float(job_data.get("submit_ts", now()))
                            await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "queued"})
                            await redis.zadd(BATCH_ZSET, {jid: original_ts})

                    try:
                        # If a preempt abort was requested before starting, exit early
                        if state.get("load_abort_requested"):
                            await _requeue_claimed_jobs()
                            return False, None

                        # Prefetch model to page cache before evicting realtime
                        log.info("prefetching_batch_model", model=best_model)
                        await prefetch_pagecache_async(best_model)

                        # Check abort again before stopping current runtime
                        if state.get("load_abort_requested"):
                            await _requeue_claimed_jobs()
                            return False, None

                        # Stop current runtime and start new one
                        loop = asyncio.get_running_loop()
                        log.info("stopping_runtime_for_batch", current_model=state.get("current_model"))
                        await loop.run_in_executor(None, stop_runtime_container_sync)

                        t0 = time.time()
                        try:
                            started = await loop.run_in_executor(
                                None, lambda: start_runtime_container_sync(best_model)
                            )
                        except Exception:
                            failure_reason = "runtime_start_exception"
                            await _requeue_claimed_jobs()
                            return False, failure_reason

                        if started is None:
                            failure_reason = "runtime_start_failed"
                            await _requeue_claimed_jobs()
                            return False, failure_reason

                        # If abort requested after starting, stop batch runtime and exit
                        if state.get("load_abort_requested"):
                            await asyncio.get_running_loop().run_in_executor(None, stop_runtime_container_sync)
                            await _requeue_claimed_jobs()
                            return False, None

                        ok, failure_reason = await wait_for_runtime_ready(timeout=900)
                        if not ok:
                            await _requeue_claimed_jobs()
                            if not failure_reason:
                                failure_reason = "runtime_not_ready"
                            return False, failure_reason

                        prev_model = state.get("current_model")
                        state["current_model"] = best_model
                        try:
                            model_load_time.labels(model=best_model).observe(time.time() - t0)
                            if prev_model != best_model:
                                model_switches.inc()
                        except Exception:
                            pass
                        return True, None
                    finally:
                        # Clear loading fields regardless of success
                        state["loading_task"] = None
                        state["loading_model"] = None
                        # Signal abort completion if requested
                        if state.get("load_abort_requested"):
                            ev = state.get("load_abort_done_event")
                            try:
                                if isinstance(ev, asyncio.Event) and not ev.is_set():
                                    ev.set()
                            except Exception:
                                pass

                # Create a task to reflect switching state, then await it to proceed
                state["loading_task"] = asyncio.create_task(_load_batch_model())
                ok_switch, failure_reason = await state["loading_task"]
                if not ok_switch:
                    if failure_reason:
                        try:
                            await drop_queued_jobs_for_model(best_model, reason=failure_reason)
                        except Exception as e:
                            log.error("drop_jobs_after_start_failure_error", model=best_model, error=str(e))
                    await asyncio.sleep(1.0)
                    continue
            else:
                # Already on the right model
                lt = state.get("loading_task")
                if lt and not lt.done():
                    # Skip probing while a switch/load is in progress
                    await asyncio.sleep(0.5)
                    continue
                # Lightweight HTTP probe only
                ok = await http_health_ok(timeout_s=3.0)
                if not ok:
                    log.warning("runtime_not_responding", model=best_model)
                    # Do not force restart here; let dedicated switch paths handle it
                    pass

            # While running jobs, prefetch the next model in background
            if len(model_ranking) > 1:
                next_model = None
                for mid, _, _ in model_ranking[1:]:
                    if mid != best_model:
                        next_model = mid
                        break
                if next_model and check_model_on_disk(next_model):
                    block_reason = await get_model_block_reason(next_model)
                    if block_reason:
                        log.debug("skip_prefetch_flagged_model", current=best_model, next=next_model, reason=block_reason)
                    else:
                        log.info("prefetching_next_model_pagecache", current=best_model, next=next_model)
                        asyncio.create_task(prefetch_pagecache_async(next_model))

            # Process jobs with a bounded number of inflight requests; keep pipeline full
            completed_count = 0
            error_count = 0
            requeued_count = 0
            active_by_eval = {}
            active_tasks = {}

            async def process_batch_job(jid):
                nonlocal completed_count, error_count, requeued_count
                # Check for preemption early
                if state.get("preempt"):
                    job_data = await redis.hgetall(JOB_HASH_PREFIX + jid)
                    if job_data:
                        original_ts = float(job_data.get("submit_ts", now()))
                        await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "queued"})
                        await redis.zadd(BATCH_ZSET, {jid: original_ts})
                        requeued_count += 1
                    return None
                
                jobh = await redis.hgetall(JOB_HASH_PREFIX + jid)
                if not jobh:
                    return None

                job_type = jobh.get("type", "completion")
                eval_name_local = jobh.get("eval_name") or None
                key_id = jobh.get("key_id")
                job_usage_context: Optional[Dict[str, Any]] = None
                if key_id:
                    metadata = {"job_id": jid}
                    if eval_name_local:
                        metadata["eval_name"] = eval_name_local
                    job_usage_context = {
                        "key_id": key_id,
                        "source": "batch",
                        "scope": "batch",
                        "metadata": metadata,
                    }
                reasoning_flag = jobh.get("reasoning_mode") in ("1", "true", "True")

                def _decode_json_field(val: Optional[str]):
                    if val is None or val == "" or val == "null":
                        return None
                    try:
                        return json.loads(val)
                    except Exception:
                        return val

                def _coerce_positive_int(val: Any) -> Optional[int]:
                    try:
                        if isinstance(val, bool) or val is None:
                            return None
                        if isinstance(val, (int, float)):
                            iv = int(val)
                        elif isinstance(val, str):
                            if not val.strip():
                                return None
                            iv = int(float(val)) if ("." in val and val.replace('.', '', 1).isdigit()) else int(val)
                        else:
                            return None
                        return iv if iv > 0 else None
                    except Exception:
                        return None

                async def _call_with_retries(
                    endpoint: str,
                    request_data: dict,
                    usage_context: Optional[Dict[str, Any]] = None,
                ):
                    """Call runtime with retries for eval jobs only; returns (result, error_info)."""
                    max_attempts = 4
                    base_backoff = 0.5
                    attempt = 0
                    last_err = None
                    while attempt < max_attempts:
                        try:
                            res = await runtime_proxy(
                                endpoint,
                                request_data,
                                source="batch",
                                usage_context=usage_context,
                            )
                            return res, None
                        except asyncio.CancelledError:
                            # Propagate cancellation immediately
                            raise
                        except httpx.HTTPStatusError as e:
                            status = getattr(e, 'response', None).status_code if getattr(e, 'response', None) else None
                            retry_after = None
                            try:
                                if e.response is not None:
                                    ra = e.response.headers.get('Retry-After')
                                    if ra:
                                        retry_after = float(ra)
                            except Exception:
                                retry_after = None
                            # Retry on 429 or 5xx
                            if status == 429 or (status is not None and status >= 500):
                                attempt += 1
                                last_err = ("http_status", status, str(e))
                                if attempt >= max_attempts:
                                    break
                                delay = retry_after if retry_after is not None else (base_backoff * (2 ** (attempt - 1)))
                                delay += random.uniform(0, 0.2)
                                await asyncio.sleep(delay)
                                continue
                            # Non-retryable 4xx
                            msg_tail = str(e)[:300]
                            return None, {"error_type": "http_status", "error_code": status, "error_message": msg_tail, "retry_count": attempt}
                        except httpx.RequestError as e:
                            attempt += 1
                            last_err = ("network_error", None, str(e))
                            if attempt >= max_attempts:
                                break
                            delay = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                            await asyncio.sleep(delay)
                            continue
                        except Exception as e:
                            attempt += 1
                            last_err = ("exception", None, str(e))
                            if attempt >= max_attempts:
                                break
                            delay = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                            await asyncio.sleep(delay)
                            continue
                    etype, code, msg = last_err if last_err else ("unknown", None, "")
                    return None, {"error_type": etype, "error_code": code, "error_message": (msg or "")[:300], "retry_count": attempt}
                try:
                    if job_type == "completion":
                        request_data = {
                            "model": jobh.get("model", best_model),
                            "prompt": jobh.get("prompt", ""),
                            "max_tokens": int(jobh.get("max_tokens", 100)),
                            "temperature": float(jobh.get("temperature", 0.7)),
                            "top_p": float(jobh.get("top_p", 1.0)),
                            "n": int(jobh.get("n", 1)),
                        }
                        # Optional: stop
                        try:
                            stop_val = json.loads(jobh.get("stop", "null"))
                            if stop_val not in (None, [], ""):
                                request_data["stop"] = stop_val
                        except Exception:
                            pass
                        # Optional: echo (needed for prompt logprobs when max_tokens=0)
                        ech = jobh.get("echo")
                        if ech is not None and ech != "":
                            try:
                                request_data["echo"] = True if ech in ("1", "true", "True") else False
                            except Exception:
                                pass
                        # Optional: logprobs/top_logprobs (vLLM completions expects integer logprobs)
                        lp = _decode_json_field(jobh.get("logprobs"))
                        tlp = _decode_json_field(jobh.get("top_logprobs"))
                        # Determine top-k from either integer or boolean+top_logprobs
                        topk = None
                        if isinstance(lp, bool):
                            if lp:
                                topk = _coerce_positive_int(tlp) or 8
                        elif lp is not None:
                            topk = _coerce_positive_int(lp)
                        if topk is not None:
                            request_data["logprobs"] = topk
                        # Structured outputs
                        response_format = _decode_json_field(jobh.get("response_format"))
                        if isinstance(response_format, dict):
                            request_data["response_format"] = response_format
                        for guided_key in ("guided_json", "guided_regex", "guided_choice", "guided_grammar"):
                            guided_val = _decode_json_field(jobh.get(guided_key))
                            if guided_val is not None and guided_val != "":
                                request_data[guided_key] = guided_val
                        res, err = await _call_with_retries(
                            "/v1/completions",
                            request_data,
                            usage_context=job_usage_context,
                        )
                        if err is not None:
                            await redis.hset(JOB_HASH_PREFIX + jid, mapping={
                                "status": "error",
                                "error": err.get("error_message", ""),
                                "error_type": err.get("error_type", ""),
                                "error_code": str(err.get("error_code")) if err.get("error_code") is not None else "",
                                "retry_count": str(err.get("retry_count", 0)),
                                "finished_ts": str(now())
                            })
                            error_count += 1
                            try:
                                jobs_total.labels(status="error", type=job_type).inc()
                            except Exception:
                                pass
                            return eval_name_local
                    else:
                        request_data = {
                            "model": jobh.get("model", best_model), 
                            "messages": json.loads(jobh.get("messages", "[]")),
                            "max_tokens": int(jobh.get("max_tokens", 100)),
                            "temperature": float(jobh.get("temperature", 0.7)),
                            "top_p": float(jobh.get("top_p", 1.0)),
                            "n": int(jobh.get("n", 1)),
                        }
                        try:
                            stop_val = json.loads(jobh.get("stop", "null"))
                            if stop_val not in (None, [], ""):
                                request_data["stop"] = stop_val
                        except Exception:
                            pass
                        # Optional: logprobs/top_logprobs (chat expects bool + int)
                        lp = _decode_json_field(jobh.get("logprobs"))
                        if isinstance(lp, bool):
                            request_data["logprobs"] = lp
                        elif isinstance(lp, (int, float)):
                            request_data["logprobs"] = bool(lp)
                        elif isinstance(lp, str) and lp:
                            request_data["logprobs"] = lp.lower() in ("1", "true", "yes")
                        tlp = _decode_json_field(jobh.get("top_logprobs"))
                        topk = _coerce_positive_int(tlp)
                        if topk is not None:
                            request_data["top_logprobs"] = topk
                        # Structured outputs
                        response_format = _decode_json_field(jobh.get("response_format"))
                        if isinstance(response_format, dict):
                            request_data["response_format"] = response_format
                        for guided_key in ("guided_json", "guided_regex", "guided_choice", "guided_grammar"):
                            guided_val = _decode_json_field(jobh.get(guided_key))
                            if guided_val is not None and guided_val != "":
                                request_data[guided_key] = guided_val
                        res, err = await _call_with_retries(
                            "/v1/chat/completions",
                            request_data,
                            usage_context=job_usage_context,
                        )
                        if err is not None:
                            await redis.hset(JOB_HASH_PREFIX + jid, mapping={
                                "status": "error",
                                "error": err.get("error_message", ""),
                                "error_type": err.get("error_type", ""),
                                "error_code": str(err.get("error_code")) if err.get("error_code") is not None else "",
                                "retry_count": str(err.get("retry_count", 0)),
                                "finished_ts": str(now())
                            })
                            error_count += 1
                            try:
                                jobs_total.labels(status="error", type=job_type).inc()
                            except Exception:
                                pass
                            return eval_name_local
                    
                    if jobh.get("eval_name"):
                        debug_log = "/tmp/eval_debug.log"
                        with open(debug_log, "a") as f:
                            f.write(f"\n[{now()}] Job {jid} for eval {jobh.get('eval_name')}\n")
                            f.write(f"Request max_tokens: {jobh.get('max_tokens', 'not set')}\n")
                            if "choices" in res and res["choices"]:
                                content = res["choices"][0].get("message", {}).get("content", "") if "message" in res["choices"][0] else res["choices"][0].get("text", "")
                                f.write(f"Response content length: {len(content)}\n")
                                f.write(f"Response content (first 200 chars): {content[:200]!r}\n")
                                f.write(f"Finish reason: {res['choices'][0].get('finish_reason', 'unknown')}\n")
                            else:
                                f.write(f"No choices in response: {json.dumps(res)[:500]}\n")
                    if reasoning_flag and isinstance(res, dict):
                        choices = res.get("choices")
                        if isinstance(choices, list):
                            for choice in choices:
                                if isinstance(choice, dict) and "logprobs" in choice:
                                    choice["logprobs"] = None
                    
                    await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "completed", "result": json.dumps(res), "finished_ts": str(now())})
                    completed_count += 1
                    try:
                        jobs_total.labels(status="completed", type=job_type).inc()
                    except Exception:
                        pass
                    return eval_name_local
                except asyncio.CancelledError:
                    # Requeue this in-flight job due to preemption cancellation
                    try:
                        jobh2 = await redis.hgetall(JOB_HASH_PREFIX + jid)
                        if jobh2:
                            original_ts = float(jobh2.get("submit_ts", now()))
                            await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "queued"})
                            await redis.zadd(BATCH_ZSET, {jid: original_ts})
                            requeued_count += 1
                        log.info("job_cancelled_requeued_due_to_preempt", job_id=jid)
                    except Exception:
                        pass
                    return eval_name_local
                except Exception as e:
                    await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "error", "error": str(e), "finished_ts": str(now())})
                    error_count += 1
                    try:
                        jobs_total.labels(status="error", type=job_type).inc()
                    except Exception:
                        pass
                    return eval_name_local

            # Keep up to MAX_CONCURRENT_BATCH inflight; refill as tasks finish
            seg_t0 = time.time()
            pending = collections.deque(claimed)
            active = set()
            def _dec_active(eval_name: Optional[str]):
                if not eval_name:
                    return
                active_by_eval[eval_name] = max(0, active_by_eval.get(eval_name, 0) - 1)

            async def _try_schedule_one() -> bool:
                if not pending:
                    return False
                scanned = 0
                while scanned < len(pending) and len(active) < MAX_CONCURRENT_BATCH and not state.get("preempt"):
                    jid = pending.popleft()
                    jobh = await redis.hgetall(JOB_HASH_PREFIX + jid)
                    eval_name_local = jobh.get("eval_name") if jobh else None
                    limit = None
                    if jobh and eval_name_local:
                        try:
                            mc = jobh.get("eval_max_concurrency")
                            if mc is not None and mc != "":
                                limit = int(mc)
                        except Exception:
                            limit = None
                    if limit is not None and eval_name_local and active_by_eval.get(eval_name_local, 0) >= limit:
                        pending.append(jid)
                        scanned += 1
                        continue
                    # schedule
                    if eval_name_local:
                        active_by_eval[eval_name_local] = active_by_eval.get(eval_name_local, 0) + 1
                    t = asyncio.create_task(process_batch_job(jid))
                    def _done_cb(fut, name=eval_name_local):
                        _dec_active(name)
                        active_tasks.pop(fut, None)
                    t.add_done_callback(_done_cb)
                    active.add(t)
                    active_tasks[t] = (jid, eval_name_local)
                    return True
                return False

            # Prime
            while pending and len(active) < MAX_CONCURRENT_BATCH and not state.get("preempt"):
                scheduled = await _try_schedule_one()
                if not scheduled:
                    break
            # Loop until drained
            while active:
                # If a preempt drain was requested, cancel all active tasks now
                if state.get("preempt") and state.get("preempt_drain_requested"):
                    for t in list(active):
                        try:
                            t.cancel()
                        except Exception:
                            pass
                done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                active -= done
                # Refill
                while pending and len(active) < MAX_CONCURRENT_BATCH and not state.get("preempt"):
                    scheduled = await _try_schedule_one()
                    if not scheduled:
                        break
                # If preempting, requeue not-started jobs
                if state.get("preempt") and pending:
                    # Requeue not-started jobs
                    for jid in list(pending):
                        job_data = await redis.hgetall(JOB_HASH_PREFIX + jid)
                        if job_data:
                            original_ts = float(job_data.get("submit_ts", now()))
                            await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "queued"})
                            await redis.zadd(BATCH_ZSET, {jid: original_ts})
                            requeued_count += 1
                    pending.clear()

                # If drain requested and nothing is active anymore, signal done
                if not active and state.get("preempt_drain_requested"):
                    ev = state.get("preempt_drain_done_event")
                    try:
                        if isinstance(ev, asyncio.Event) and not ev.is_set():
                            ev.set()
                    except Exception:
                        pass

            # After batch work is done, return to realtime model
            try:
                batch_segment_duration.observe(time.time() - seg_t0)
            except Exception:
                pass
            if state.get("preempt"):
                log.info(
                    "batch_segment_aborted",
                    model=best_model,
                    claimed=len(claimed),
                    requeued=requeued_count,
                    completed=completed_count,
                    errors=error_count,
                )
            else:
                log.info(
                    "batch_segment_complete",
                    model=best_model,
                    jobs_processed=completed_count,
                    errors=error_count,
                )
            
            # Return to realtime model if no more batch work
            # Double-check remaining jobs to avoid racing with prepares finishing
            remaining_jobs = await redis.zcard(BATCH_ZSET)
            if remaining_jobs == 0:
                await asyncio.sleep(1.0)
                remaining_jobs = await redis.zcard(BATCH_ZSET)
            # If drain requested and we are fully idle (no active segment), also signal done
            if remaining_jobs == 0 and state.get("preempt_drain_requested"):
                ev = state.get("preempt_drain_done_event")
                try:
                    if isinstance(ev, asyncio.Event) and not ev.is_set():
                        ev.set()
                except Exception:
                    pass
            if remaining_jobs == 0 and state.get("current_model") != REALTIME_MODEL:
                log.info("no_batch_jobs_remaining_loading_realtime")
                try:
                    await ensure_realtime_model_loading()
                    log.info("returned_to_realtime", model=REALTIME_MODEL)
                except Exception as e:
                    log.error("failed_to_return_to_realtime", error=str(e))

            # loop again
        except asyncio.CancelledError:
            break
        except Exception as e:
            import traceback
            log.error("batch_planner_error", 
                     error=str(e),
                     error_type=type(e).__name__,
                     traceback=traceback.format_exc())
            await asyncio.sleep(1.0)

# ============ TUS Upload Implementation ============
import aiofiles
import tarfile
import base64
import zstandard as zstd
from pathlib import Path
from fastapi import Request, Header

TUS_VERSION = "1.0.0"
TUS_MAX_SIZE = 200 * 1024 * 1024 * 1024  # 200GB
UPLOAD_TEMP_DIR = "/tmp/model_uploads"

# Ensure temp dir exists
Path(UPLOAD_TEMP_DIR).mkdir(parents=True, exist_ok=True)

@upload_router.options("/upload")
async def tus_options(auth: bool = Depends(make_require_scopes({"upload"}))):
    """TUS discovery endpoint"""
    return Response(
        status_code=204,
        headers={
            "Tus-Version": TUS_VERSION,
            "Tus-Max-Size": str(TUS_MAX_SIZE),
            "Tus-Extension": "creation"
        }
    )

@upload_router.post("/upload")
async def tus_create(
    request: Request,
    auth: bool = Depends(make_require_scopes({"upload"})),
    upload_length: int = Header(None),
    upload_metadata: str = Header(None),
    tus_resumable: str = Header(TUS_VERSION)
):
    """TUS upload creation endpoint"""
    if tus_resumable != TUS_VERSION:
        return Response(
            status_code=412,
            headers={"Tus-Version": TUS_VERSION}
        )
    
    if not upload_length:
        raise HTTPException(status_code=400, detail="Upload-Length required")
    
    if upload_length > TUS_MAX_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {TUS_MAX_SIZE}")
    
    # Parse metadata (base64 encoded key-value pairs)
    metadata = {}
    if upload_metadata:
        for item in upload_metadata.split(","):
            if " " in item:
                key, value_b64 = item.strip().split(" ", 1)
                try:
                    metadata[key] = base64.b64decode(value_b64).decode('utf-8')
                except:
                    pass
    
    model_name = metadata.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name required in metadata")
    
    # Validate model name (alphanumeric, hyphens, underscores, one slash)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+(/[a-zA-Z0-9_-]+)?$', model_name):
        raise HTTPException(status_code=400, detail="Invalid model name format")
    
    # Create upload ID and temp file path
    upload_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_TEMP_DIR, f"{upload_id}.tar.zst")
    
    # Store upload state in Redis
    sha256_expected = metadata.get("sha256")
    await redis.hset(f"upload:{upload_id}", mapping={
        "model_name": model_name,
        "size": str(upload_length),
        "offset": "0",
        "temp_path": temp_path,
        "status": "uploading",
        "created": str(int(now())),
        "sha256_expected": sha256_expected or ""
    })
    await redis.expire(f"upload:{upload_id}", 86400)  # 24h TTL
    
    log.info("upload_created", upload_id=upload_id, model=model_name, size=upload_length)
    
    return Response(
        status_code=201,
        headers={
            "Location": f"/upload/{upload_id}",
            "Tus-Resumable": TUS_VERSION
        }
    )

@upload_router.head("/upload/{upload_id}")
async def tus_head(
    upload_id: str,
    auth: bool = Depends(make_require_scopes({"upload"})),
    tus_resumable: str = Header(..., alias="Tus-Resumable")
):
    """TUS resumption endpoint - returns current offset"""
    if tus_resumable != TUS_VERSION:
        return Response(
            status_code=412,
            headers={"Tus-Version": TUS_VERSION}
        )

    upload_data = await redis.hgetall(f"upload:{upload_id}")
    if not upload_data:
        raise HTTPException(status_code=404)
    
    # Derive the truth from the on-disk file size and keep Redis in sync
    temp_path = upload_data["temp_path"]
    actual = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
    stored = int(upload_data.get("offset", 0))
    if actual != stored:
        try:
            await redis.hset(f"upload:{upload_id}", "offset", str(actual))
        except Exception:
            pass

    return Response(headers={
        "Upload-Offset": str(actual),
        "Upload-Length": upload_data["size"],
        "Tus-Resumable": TUS_VERSION,
        "Cache-Control": "no-store",
    })

@upload_router.patch("/upload/{upload_id}")
async def tus_patch(
    upload_id: str,
    request: Request,
    auth: bool = Depends(make_require_scopes({"upload"})),
    upload_offset: int = Header(..., alias="Upload-Offset"),
    content_type: str = Header(..., alias="Content-Type"),
    tus_resumable: str = Header(..., alias="Tus-Resumable"),
    background_tasks: BackgroundTasks = None
):
    """TUS chunk upload endpoint"""
    if tus_resumable != TUS_VERSION:
        return Response(
            status_code=412,
            headers={"Tus-Version": TUS_VERSION}
        )
    if content_type != "application/offset+octet-stream":
        raise HTTPException(status_code=415)
    
    upload_data = await redis.hgetall(f"upload:{upload_id}")
    if not upload_data:
        raise HTTPException(status_code=404)
    
    # Truth = file size; keep Redis consistent before comparing
    temp_path = upload_data["temp_path"]
    current_offset = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
    if str(current_offset) != upload_data.get("offset", "0"):
        try:
            await redis.hset(f"upload:{upload_id}", "offset", str(current_offset))
        except Exception:
            pass

    # If client is behind/ahead, tell them where to resume
    if upload_offset != current_offset:
        return Response(
            status_code=409,
            headers={
                "Upload-Offset": str(current_offset),
                "Tus-Resumable": TUS_VERSION,
            },
        )

    # Read the chunk
    chunk = await request.body()
    total = int(upload_data["size"])

    # Guard: don't allow overflow beyond declared Upload-Length
    if upload_offset + len(chunk) > total:
        return Response(status_code=413)

    # Append
    async with aiofiles.open(temp_path, "ab") as f:
        await f.write(chunk)
    new_offset = current_offset + len(chunk)

    # Persist new offset (best-effort)
    try:
        await redis.hset(f"upload:{upload_id}", "offset", str(new_offset))
    except Exception:
        pass

    # If complete, finalize asynchronously so we can return immediately
    if new_offset >= total:
        if background_tasks is not None:
            background_tasks.add_task(_finalize_upload, upload_id, upload_data)
        else:
            # Fallback (shouldn't happen under FastAPI, but harmless)
            asyncio.create_task(_finalize_upload(upload_id, upload_data))

    return Response(status_code=204, headers={
        "Upload-Offset": str(new_offset),
        "Tus-Resumable": TUS_VERSION,
    })

async def _finalize_upload(upload_id: str, upload_data: dict):
    """Extract and install uploaded model"""
    temp_path = upload_data["temp_path"]
    model_name = upload_data["model_name"]
    
    try:
        # Compute SHA256 of uploaded tarball and optionally verify against provided checksum
        def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                while True:
                    b = f.read(chunk_size)
                    if not b:
                        break
                    h.update(b)
            return h.hexdigest()
        sha256_actual = _sha256_file(temp_path)
        sha256_expected = upload_data.get("sha256_expected") or ""
        try:
            await redis.hset(f"upload:{upload_id}", mapping={"sha256": sha256_actual})
        except Exception:
            pass
        if sha256_expected:
            if sha256_actual.lower() != sha256_expected.lower():
                log.error("upload_checksum_mismatch", model=model_name, expected=sha256_expected, actual=sha256_actual)
                await redis.hset(f"upload:{upload_id}", mapping={"status": "failed", "error": "checksum_mismatch"})
                raise ValueError("Uploaded file checksum mismatch")
        log.info("upload_checksum", model=model_name, sha256=sha256_actual, verified=bool(sha256_expected))

        # Validate uploaded size matches expectation
        try:
            expected_size = int(upload_data.get("size", "0"))
            actual_size = os.path.getsize(temp_path)
            if actual_size != expected_size:
                raise ValueError(f"Uploaded size mismatch: expected={expected_size} actual={actual_size}")
        except Exception as e:
            # Size mismatch or missing; fail fast
            raise

        # Target directory in HF cache format (container-side mount path)
        model_dir = f"models--{model_name.replace('/', '--')}"
        target_path = os.path.join(HOST_HF_CACHE, "hub", model_dir, "snapshots", "main")
        
        # Clean existing if present
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        os.makedirs(target_path, exist_ok=True)
        
        log.info("extracting_model", model=model_name, path=target_path)
        
        # Helpers for safe extraction
        def _strip_leading_dir(name: str) -> str:
            name = name.lstrip("./")
            parts = name.split("/", 1)
            return parts[1] if len(parts) > 1 else parts[0]

        def _safe_join(base: str, *paths: str) -> str:
            candidate = os.path.normpath(os.path.join(base, *paths))
            # Ensure within base
            if os.path.commonpath([base, candidate]) != os.path.abspath(base):
                raise ValueError("Path traversal detected during extraction")
            return candidate

        # Decompress and safely extract
        dctx = zstd.ZstdDecompressor()
        with open(temp_path, 'rb') as compressed:
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as tar:
                    for member in tar:
                        # Normalize and sanitize path
                        name = _strip_leading_dir(member.name)
                        if not name:
                            continue
                        if os.path.isabs(name) or ".." in name.split("/"):
                            continue

                        # Skip symlinks, hardlinks, devices, and fifos
                        if member.issym() or member.islnk() or member.ischr() or member.isblk() or member.isfifo():
                            continue

                        dest_path = _safe_join(target_path, name)

                        if member.isdir():
                            os.makedirs(dest_path, exist_ok=True)
                            continue
                        if member.isreg():
                            # Ensure directory exists
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            src = tar.extractfile(member)
                            if src is None:
                                continue
                            with open(dest_path, 'wb') as out:
                                shutil.copyfileobj(src, out, length=1024 * 1024)
                            # Set conservative file perms
                            try:
                                os.chmod(dest_path, 0o644)
                            except Exception:
                                pass
                            continue
                        # Ignore any other types
                        continue
        
        # Validate model files exist
        # Search recursively for config.json (sometimes not at top-level)
        has_config = False
        for root, _, files in os.walk(target_path):
            if "config.json" in files:
                has_config = True
                break
        if not has_config:
            raise ValueError("Missing config.json in extracted model directory")
        
        # Check for model weights
        has_weights = False
        for root, _, files in os.walk(target_path):
            if any(f.endswith((".safetensors", ".bin", ".gguf")) for f in files):
                has_weights = True
                break
        if not has_weights:
            raise ValueError("No model weights found")
        
        # Mark complete
        await redis.hset(f"upload:{upload_id}", mapping={
            "status": "completed",
            "completed": str(int(now()))
        })
        
        # Store model metadata
        size = 0
        for root, _, files in os.walk(target_path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    size += os.path.getsize(fp)
                except Exception:
                    pass
        await redis.hset(f"{MODEL_META_PREFIX}{model_name}", mapping={
            "type": "uploaded",
            "size_gb": str(size / (1024**3)),
            "uploaded_at": str(int(now())),
            "load_est_s": "30"
        })
        # Clear transient failure flags (e.g., prior download failure) now that the model is present on disk
        try:
            await redis.delete(f"model_failure:{model_name}")
        except Exception:
            pass
        
        log.info("model_uploaded", model=model_name, size_gb=size/(1024**3))
        
    except Exception as e:
        log.error("upload_failed", model=model_name, error=str(e))
        await redis.hset(f"upload:{upload_id}", "status", "failed")
        raise
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Mount routers after all endpoints have been defined
app.include_router(v1_router)
app.include_router(upload_router)
app.include_router(eval_router, dependencies=[Depends(make_require_scopes({"eval"}))])  # /eval endpoints
app.include_router(ui_router)

async def drop_queued_jobs_for_model(model_id: str, reason: str = "model_failure"):
    """Remove queued jobs for a model and mark them as error for visibility."""
    try:
        queued = await redis.zrange(BATCH_ZSET, 0, -1)
        dropped = 0
        for jid in queued:
            try:
                h = await redis.hgetall(JOB_HASH_PREFIX + jid)
                if not h:
                    continue
                if h.get("model") != model_id:
                    continue
                status = h.get("status")
                if status not in (None, "queued", "running"):
                    continue
                await redis.hset(JOB_HASH_PREFIX + jid, mapping={
                    "status": "error",
                    "error": reason,
                    "finished_ts": str(now())
                })
                await redis.zrem(BATCH_ZSET, jid)
                dropped += 1
            except Exception:
                continue
        if dropped:
            log.warning("dropped_jobs_for_failed_model", model=model_id, dropped=dropped, reason=reason)
    except Exception as e:
        log.error("drop_jobs_failed", model=model_id, error=str(e))
