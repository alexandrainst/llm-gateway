# orchestrator/app.py
import os, time, uuid, json, asyncio, subprocess, socket
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from fastapi import FastAPI, HTTPException, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import redis.asyncio as aioredis
import httpx
import docker
from docker.types import DeviceRequest
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from eval_manager import EvalManager, router as eval_router

log = structlog.get_logger()

# Config from env
API_KEY = os.getenv("API_KEY", None)  # Optional API key for auth
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
HOST_HF_CACHE = os.getenv("HOST_HF_CACHE", "/host_hf_cache")
RUNTIME_TYPE = os.getenv("RUNTIME_TYPE", "sglang")  # "sglang" or "vllm"
RUNTIME_IMAGE = os.getenv("RUNTIME_IMAGE", "lmsysorg/sglang:b200-cu129" if RUNTIME_TYPE == "sglang" else "vllm/vllm-openai:latest")
RUNTIME_ARGS = os.getenv("RUNTIME_ARGS", "--tp 2 --cuda-graph-max-bs 16" if RUNTIME_TYPE == "sglang" else "--tensor-parallel-size 2")
RUNTIME_PORT = int(os.getenv("RUNTIME_PORT", "30000"))
RUNTIME_API_PATH = os.getenv("RUNTIME_API_PATH", "/v1/completions")  # Both support OpenAI API
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "synquid/gemma-3-27b-it-FP8")
MIN_HOT_TTL_S = int(os.getenv("MIN_HOT_TTL_S", "600"))
L_MAX_S = int(os.getenv("L_MAX_S", "600"))
P_MAX = float(os.getenv("P_MAX", "0.3"))
TOKENS_B = float(os.getenv("TOKENS_B", "2"))
REFILL_PER_SEC = float(os.getenv("TOKENS_REFILL_PER_SEC", "0.000555"))

BATCH_ZSET = "z:batchjobs"
JOB_HASH_PREFIX = "job:"
MODEL_META_PREFIX = "model:"

CONTAINER_NAME = "llm_runtime"

app = FastAPI()
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
    "loading_model": None  # Track which model is being loaded
}

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

# ============ Authentication ============
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if configured"""
    if not API_KEY:
        # No API key configured, allow all requests
        return True
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authentication")
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

@app.get("/healthz")
async def healthz():
    """Simple liveness endpoint (no auth, fast path)."""
    return {"status": "ok"}

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list[str] = None
    
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list[str] = None

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

# ---------- redis connect ----------
async def redis_connect():
    global redis
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    await redis.ping()

# ---------- utilities ----------
def now():
    return time.time()

async def cleanup_stuck_jobs():
    """Clean up any jobs stuck in 'running' state from previous orchestrator run."""
    cleaned_count = 0
    
    # Clean up stuck batch jobs
    cursor = 0
    while True:
        cursor, keys = await redis.scan(cursor, match="batch_job:*", count=100)
        for key in keys:
            job_data = await redis.hgetall(key)
            if job_data.get("status") == "running":
                # Mark as error - the orchestrator restarted while it was running
                await redis.hset(key, mapping={
                    "status": "error",
                    "error": "Orchestrator restarted",
                    "finished_ts": str(now())
                })
                cleaned_count += 1
                log.info(f"Cleaned up stuck batch job: {key.decode() if isinstance(key, bytes) else key}")
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
                    log.info(f"Cleaned up stuck eval job: {key.decode() if isinstance(key, bytes) else key}")
            except Exception as e:
                log.error(f"Error cleaning eval job {key}: {e}")
        if cursor == 0:
            break
    
    if cleaned_count > 0:
        log.info(f"Cleaned up {cleaned_count} stuck jobs from previous run")

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
    
    # Add extra args or default args
    if extra_args:
        cmd += extra_args.split()
    else:
        cmd += RUNTIME_ARGS.split()
    
    # Log the command for debugging
    log.info("runtime_command", runtime_type=RUNTIME_TYPE, cmd=cmd, model_path=model_path)

    # device request for GPUs
    dev_req = DeviceRequest(count=-1, capabilities=[["gpu"]])

    volumes = {
        # map the host HF cache into the runtime container so it reuses cached downloads
        # Use the actual host path, not the orchestrator's container mount path
        "/home/alex-admin/.cache/huggingface": {"bind": "/root/.cache/huggingface", "mode": "rw"}
    }

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
            "TQDM_DISABLE": "1"  # Also disable tqdm globally
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
    """Send a warmup request to the runtime to prime it"""
    try:
        runtime_host = get_runtime_host()
        runtime_url = f"http://{runtime_host}:{RUNTIME_PORT}"
        
        # Get the actual model name from the runtime
        async with httpx.AsyncClient(timeout=30) as client:
            # First get the model name from the runtime
            models_response = await client.get(f"{runtime_url}/v1/models")
            if models_response.status_code == 200:
                models_data = models_response.json()
                if models_data.get("data"):
                    model_name = models_data["data"][0]["id"]
                else:
                    model_name = state.get("current_model", "model")
            else:
                model_name = state.get("current_model", "model")
            
            # Check if this is a base model by looking for chat_template in tokenizer config
            # (Same logic as eval_manager uses)
            is_base_model = False
            current_model = state.get("current_model") or state.get("loading_model")
            if current_model:
                # Check if it's an uploaded model first
                import redis as sync_redis
                try:
                    sync_redis_client = sync_redis.from_url(REDIS_URL, decode_responses=True)
                    model_meta = sync_redis_client.hgetall(f"{MODEL_META_PREFIX}{current_model}")
                    sync_redis_client.close()
                    
                    if model_meta and model_meta.get("type") == "uploaded":
                        # For uploaded models, check tokenizer_config.json
                        model_dir = f"/host_hf_cache/hub/models--{current_model.replace('/', '--')}"
                        tokenizer_config_path = f"{model_dir}/snapshots/main/tokenizer_config.json"
                        if os.path.exists(tokenizer_config_path):
                            with open(tokenizer_config_path) as f:
                                config = json.load(f)
                                is_base_model = config.get("chat_template") is None
                except Exception:
                    pass
            
            # Send appropriate warmup based on model type
            if is_base_model:
                # Base model - use completion endpoint
                warmup_data = {
                    "model": model_name,
                    "prompt": "Hello",
                    "max_tokens": 1,
                    "temperature": 0
                }
                endpoint = "/v1/completions"
            else:
                # Instruction model - use chat endpoint
                warmup_data = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1,
                    "temperature": 0
                }
                endpoint = "/v1/chat/completions"
            
            response = await client.post(
                f"{runtime_url}{endpoint}",
                json=warmup_data
            )
            if response.status_code == 200:
                log.info("warmup_request_sent", model=model_name, is_base=is_base_model)
            else:
                log.warning("warmup_request_failed", status=response.status_code, model=model_name, is_base=is_base_model)
    except Exception as e:
        log.warning("warmup_request_error", error=str(e))

async def wait_for_runtime_ready(timeout=600):
    runtime_host = get_runtime_host()
    deadline = time.time() + timeout
    log.info("waiting_for_runtime", host=runtime_host, port=RUNTIME_PORT, timeout=timeout)
    
    last_container_check = 0
    while time.time() < deadline:
        # Check container health every 5 seconds
        if time.time() - last_container_check > 5:
            try:
                container = docker_client.containers.get(CONTAINER_NAME)
                if container.status == "exited":
                    # Container crashed - check exit code and logs
                    exit_code = container.attrs.get('State', {}).get('ExitCode', -1)
                    model = state.get("loading_model") or state.get("current_model")
                    logs = container.logs(tail=200).decode('utf-8')
                    
                    # Exit code 137 = SIGKILL (often OOM killer)
                    # Exit code 139 = SIGSEGV (segfault)
                    # Exit code 1 = general error
                    
                    if exit_code == 137 or "out of memory" in logs.lower() or "oom" in logs.lower() or "CUDA out of memory" in logs:
                        log.error("runtime_oom", model=model, exit_code=exit_code, logs=logs[-500:])
                        # Mark model as OOM for 24 hours
                        if model:
                            await redis.set(f"model_oom:{model}", str(now()), ex=86400)
                    elif "has no SGlang implementation" in logs or "not compatible with SGLang" in logs:
                        log.error("model_not_supported", model=model, exit_code=exit_code, logs=logs[-500:])
                        # Mark model as incompatible permanently (30 days)
                        if model:
                            await redis.set(f"model_incompatible:{model}", str(now()), ex=2592000)
                    elif exit_code == 139:
                        log.error("runtime_segfault", model=model, exit_code=exit_code, logs=logs[-500:])
                        # Segfault might be model-specific, mark for 6 hours
                        if model:
                            await redis.set(f"model_failure:{model}", str(now()), ex=21600)
                    else:
                        log.error("runtime_crashed", model=model, exit_code=exit_code, logs=logs[-500:])
                        # Mark model as failed for 1 hour (might be transient)
                        if model:
                            await redis.set(f"model_failure:{model}", str(now()), ex=3600)
                    
                    return False
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
                        return True
            except Exception as e:
                log.debug("runtime_not_responding_yet", host=runtime_host, error=str(e))
        await asyncio.sleep(1.0)
    
    log.error("runtime_timeout", host=runtime_host, timeout=timeout)
    return False

async def runtime_proxy(endpoint: str, payload: dict, timeout=120):
    """Proxy OpenAI-compatible requests directly to runtime"""
    from fastapi.responses import StreamingResponse
    runtime_host = get_runtime_host()
    url = f"http://{runtime_host}:{RUNTIME_PORT}{endpoint}"
    
    # Check if this is a streaming request
    if payload.get("stream", False):
        # For streaming, we need to proxy the stream
        async def stream_generator():
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, json=payload, timeout=timeout) as r:
                    r.raise_for_status()
                    async for chunk in r.aiter_bytes():
                        yield chunk
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # Non-streaming request
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()

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
        log.error("model_access_denied", model_id=model_id, error=str(e), 
                 hint="Need HuggingFace token or model doesn't exist/is gated")
        return False
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower() or "token" in error_msg.lower():
            log.error("model_auth_required", model_id=model_id, error=error_msg,
                     hint="Set HF_TOKEN environment variable")
        else:
            log.error("model_download_failed", model_id=model_id, error=error_msg)
        return False

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
                "is_current": state.get("current_model") == model_id
            }
        
        needs[model_id]["job_count"] += 1
        needs[model_id]["total_priority"] += int(h.get("priority", 1))
    
    return needs

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
        if not info["on_disk"] and not info["is_current"]:
            # Check disk space before downloading
            if await ensure_disk_space(100):  # Assume 100GB needed
                log.info("proactive_download_starting", model_id=model_id, job_count=info["job_count"])
                asyncio.create_task(download_model_async(model_id))
                break  # Download one at a time
    
    # Prefetch high-priority models that might be used soon
    for model_id, info in models_by_importance[:2]:  # Top 2 models
        if info["on_disk"] and not info["is_current"] and not info["in_page_cache"]:
            log.debug("proactive_prefetch", model_id=model_id)  # Changed to debug level
            asyncio.create_task(prefetch_pagecache_async(model_id))
            break  # Prefetch one at a time

# ---------- API & job endpoints ----------
@app.on_event("startup")
async def startup():
    await redis_connect()
    
    # Clean up any stuck jobs from previous run
    await cleanup_stuck_jobs()
    
    # Initialize eval system
    app.state.eval_manager = EvalManager(redis, None)  # batch_manager=None for now
    app.include_router(eval_router)  # Mount at /eval
    
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
        await prefetch_pagecache_async(REALTIME_MODEL)
        await asyncio.get_running_loop().run_in_executor(None, lambda: start_runtime_container_sync(REALTIME_MODEL))
        # vLLM needs more time to start up
        startup_timeout = 300 if RUNTIME_TYPE == "vllm" else 120
        ok = await wait_for_runtime_ready(timeout=startup_timeout)
        if ok:
            state["current_model"] = REALTIME_MODEL
            log.info("realtime_model_loaded_on_startup", model=REALTIME_MODEL)
    except Exception as e:
        log.error("failed_to_load_realtime_on_startup", error=str(e))

async def ensure_realtime_model_loading():
    """Ensure realtime model is loading - returns existing task if already in progress"""
    # If already loading the realtime model, return the existing task
    if state["loading_task"] and state["loading_model"] == REALTIME_MODEL:
        if not state["loading_task"].done():
            log.info("realtime_model_already_loading")
            return state["loading_task"]
    
    # Create new loading task
    async def load_model_async():
        state["preempt"] = True
        state["loading_model"] = REALTIME_MODEL
        try:
            log.info("starting_realtime_model_load", model=REALTIME_MODEL)
            await prefetch_pagecache_async(REALTIME_MODEL)
            await asyncio.get_running_loop().run_in_executor(None, stop_runtime_container_sync)
            await asyncio.get_running_loop().run_in_executor(None, lambda: start_runtime_container_sync(REALTIME_MODEL))
            ok = await wait_for_runtime_ready(timeout=60)
            if ok:
                state["current_model"] = REALTIME_MODEL
                log.info("realtime_model_ready", model=REALTIME_MODEL)
        except Exception as e:
            log.error("realtime_model_load_failed", error=str(e))
        finally:
            state["preempt"] = False
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

@app.get("/v1/models")
async def list_models(auth: bool = Depends(verify_api_key)):
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
async def metrics(auth: bool = Depends(verify_api_key)):
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

@app.post("/v1/completions")
async def completions(req: CompletionRequest, response: Response, auth: bool = Depends(verify_api_key)):
    """OpenAI-compatible completions endpoint"""
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
        raise HTTPException(status_code=400, detail=f"Model {req.model} not available for realtime. Use /v1/jobs for batch processing.")
    
    # Check if realtime model is hot
    if state.get("current_model") == REALTIME_MODEL and is_runtime_ready():
        try:
            return await runtime_proxy("/v1/completions", req.dict())
        except httpx.ConnectError as e:
            # Runtime is actually down
            log.error("completions_proxy_connect_error", error=str(e))
            state["current_model"] = None
            # Fall through to cold start
        except Exception as e:
            # Request-level error, runtime is still up
            log.error("completions_proxy_error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    # Model needs to be loaded - return 503 with Retry-After
    response.status_code = 503
    response.headers["Retry-After"] = "30"  # Suggest retry in 30 seconds
    log.info("realtime_model_cold", model=REALTIME_MODEL)
    
    # Start loading in background (or reuse existing loading task)
    asyncio.create_task(ensure_realtime_model_loading())
    
    return {"error": "Model is loading, please retry", "type": "model_cold_start", "retry_after": 30}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, response: Response, auth: bool = Depends(verify_api_key)):
    """OpenAI-compatible chat completions endpoint"""
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
        raise HTTPException(status_code=400, detail=f"Model {req.model} not available for realtime. Use /v1/jobs for batch processing.")
    
    # Check if realtime model is hot
    if state.get("current_model") == REALTIME_MODEL and is_runtime_ready():
        try:
            return await runtime_proxy("/v1/chat/completions", req.dict())
        except httpx.ConnectError as e:
            # Runtime is actually down
            log.error("chat_completions_proxy_connect_error", error=str(e))
            state["current_model"] = None
            # Fall through to cold start
        except Exception as e:
            # Request-level error, runtime is still up
            log.error("chat_completions_proxy_error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    # Model needs to be loaded - return 503 with Retry-After
    response.status_code = 503
    response.headers["Retry-After"] = "30"  # Suggest retry in 30 seconds
    log.info("realtime_model_cold", model=REALTIME_MODEL)
    
    # Start loading in background (or reuse existing loading task)
    asyncio.create_task(ensure_realtime_model_loading())
    
    return {"error": "Model is loading, please retry", "type": "model_cold_start", "retry_after": 30}

@app.post("/v1/batch/completions")
async def batch_completions(req: BatchCompletionRequest, response: Response, auth: bool = Depends(verify_api_key)):
    """Submit a batch completion request"""
    # Generate completion ID
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    submit_ts = now()
    
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
        "priority": str(req.priority),
        "status": "queued",
        "submit_ts": str(submit_ts),
        "created": str(int(submit_ts))
    }
    
    await redis.hset(jobkey, mapping=jobdata)
    await redis.expire(jobkey, 604800)  # 7 days TTL
    await redis.zadd(BATCH_ZSET, {completion_id: submit_ts})
    
    # Return 202 Accepted with ID and status URL
    response.status_code = 202
    return {
        "id": completion_id,
        "status": "queued",
        "status_url": f"/v1/batch/completions/{completion_id}"
    }

@app.get("/v1/batch/completions/{completion_id}")
async def get_batch_completion(completion_id: str, auth: bool = Depends(verify_api_key)):
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

@app.post("/v1/batch/chat/completions")
async def batch_chat_completions(req: BatchChatCompletionRequest, response: Response, auth: bool = Depends(verify_api_key)):
    """Submit a batch chat completion request"""
    # Generate completion ID
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    submit_ts = now()
    
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
        "priority": str(req.priority),
        "status": "queued",
        "submit_ts": str(submit_ts),
        "created": str(int(submit_ts))
    }
    
    await redis.hset(jobkey, mapping=jobdata)
    await redis.expire(jobkey, 604800)  # 7 days TTL
    await redis.zadd(BATCH_ZSET, {completion_id: submit_ts})
    
    # Return 202 Accepted
    response.status_code = 202
    return {
        "id": completion_id,
        "status": "queued",
        "status_url": f"/v1/batch/chat/completions/{completion_id}"
    }

@app.get("/v1/batch/chat/completions/{completion_id}")
async def get_batch_chat_completion(completion_id: str, auth: bool = Depends(verify_api_key)):
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


@app.post("/v1/jobs")
async def submit_job(j: JobSubmit):
    job_id = str(uuid.uuid4())
    submit_ts = now()
    jobkey = JOB_HASH_PREFIX + job_id
    jobdata = {
        "job_id": job_id,
        "model": j.model,
        "input": j.input,
        "priority": str(j.priority),
        "estimate_s": str(j.estimate_s),
        "submit_ts": str(submit_ts),
        "status": "queued"
    }
    await redis.hset(jobkey, mapping=jobdata)
    await redis.zadd(BATCH_ZSET, {job_id: submit_ts})
    return {"job_id": job_id}

@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
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
            if not queued:
                await asyncio.sleep(1.0)
                continue
            
            # Check if queue has changed
            current_queue_state = tuple(queued)  # Make it hashable
            queue_changed = (current_queue_state != last_queue_state)
            last_queue_state = current_queue_state

            jobs_by_model = {}
            for jid in queued:
                h = await redis.hgetall(JOB_HASH_PREFIX + jid)
                if not h:
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
                
                if await redis.get(failure_key) or await redis.get(oom_key) or await redis.get(incompatible_key):
                    log.debug("skipping_failed_model", model=mid, 
                             failure=bool(await redis.get(failure_key)),
                             oom=bool(await redis.get(oom_key)),
                             incompatible=bool(await redis.get(incompatible_key)))
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
                
                # Prefetch model to page cache before evicting realtime
                log.info("prefetching_batch_model", model=best_model)
                await prefetch_pagecache_async(best_model)
                
                # Only NOW stop the runtime container
                log.info("stopping_runtime_for_batch", current_model=state.get("current_model"))
                await asyncio.get_running_loop().run_in_executor(None, stop_runtime_container_sync)
                try:
                    await asyncio.get_running_loop().run_in_executor(None, lambda: start_runtime_container_sync(best_model))
                except Exception as e:
                    # requeue claimed jobs with original submit time
                    for jid in claimed:
                        job_data = await redis.hgetall(JOB_HASH_PREFIX + jid)
                        original_ts = float(job_data.get("submit_ts", now()))
                        await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "queued"})
                        await redis.zadd(BATCH_ZSET, {jid: original_ts})
                    await asyncio.sleep(1.0)
                    continue

                ok = await wait_for_runtime_ready(timeout=900)
                if not ok:
                    for jid in claimed:
                        job_data = await redis.hgetall(JOB_HASH_PREFIX + jid)
                        original_ts = float(job_data.get("submit_ts", now()))
                        await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "queued"})
                        await redis.zadd(BATCH_ZSET, {jid: original_ts})
                    await asyncio.sleep(1.0)
                    continue

                state["current_model"] = best_model
            else:
                # Already on the right model, just check it's ready
                ok = await wait_for_runtime_ready(timeout=10)
                if not ok:
                    log.warning("runtime_not_responding", model=best_model)
                    # Force restart
                    await asyncio.get_running_loop().run_in_executor(None, stop_runtime_container_sync)
                    await asyncio.get_running_loop().run_in_executor(None, lambda: start_runtime_container_sync(best_model))
                    ok = await wait_for_runtime_ready(timeout=900)
                    if not ok:
                        for jid in claimed:
                            job_data = await redis.hgetall(JOB_HASH_PREFIX + jid)
                            original_ts = float(job_data.get("submit_ts", now()))
                            await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "queued"})
                            await redis.zadd(BATCH_ZSET, {jid: original_ts})
                        await asyncio.sleep(1.0)
                        continue

            # While running jobs, prefetch the next model in background
            if len(model_ranking) > 1:
                next_model = None
                for mid, _, _ in model_ranking[1:]:
                    if mid != best_model:
                        next_model = mid
                        break
                if next_model and check_model_on_disk(next_model):
                    log.info("prefetching_next_model_pagecache", current=best_model, next=next_model)
                    asyncio.create_task(prefetch_pagecache_async(next_model))

            # Process claimed jobs concurrently with max concurrency using semaphore
            MAX_CONCURRENT_BATCH = 200
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCH)
            
            async def process_batch_job(jid):
                """Process a single batch job with semaphore-controlled concurrency"""
                async with semaphore:
                    # Check for preemption
                    if state.get("preempt"):
                        # Requeue this job
                        job_data = await redis.hgetall(JOB_HASH_PREFIX + jid)
                        if job_data:
                            original_ts = float(job_data.get("submit_ts", now()))
                            await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "queued"})
                            await redis.zadd(BATCH_ZSET, {jid: original_ts})
                        return
                    
                    jobh = await redis.hgetall(JOB_HASH_PREFIX + jid)
                    if not jobh:
                        return
                    
                    # Reconstruct the request based on job type
                    job_type = jobh.get("type", "completion")
                    try:
                        if job_type == "completion":
                            # Build completions request
                            request_data = {
                                "model": jobh.get("model", best_model),
                                "prompt": jobh.get("prompt", ""),
                                "max_tokens": int(jobh.get("max_tokens", 100)),
                                "temperature": float(jobh.get("temperature", 0.7)),
                                "top_p": float(jobh.get("top_p", 1.0)),
                                "n": int(jobh.get("n", 1)),
                                "stop": json.loads(jobh.get("stop", "null"))
                            }
                            res = await runtime_proxy("/v1/completions", request_data)
                        else:  # chat_completion
                            # Build chat completions request
                            request_data = {
                                "model": jobh.get("model", best_model), 
                                "messages": json.loads(jobh.get("messages", "[]")),
                                "max_tokens": int(jobh.get("max_tokens", 100)),
                                "temperature": float(jobh.get("temperature", 0.7)),
                                "top_p": float(jobh.get("top_p", 1.0)),
                                "n": int(jobh.get("n", 1)),
                                "stop": json.loads(jobh.get("stop", "null"))
                            }
                            res = await runtime_proxy("/v1/chat/completions", request_data)
                        
                        # Debug logging for eval responses
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
                        
                        await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "completed", "result": json.dumps(res), "finished_ts": str(now())})
                    except Exception as e:
                        await redis.hset(JOB_HASH_PREFIX + jid, mapping={"status": "error", "error": str(e), "finished_ts": str(now())})
            
            # Start all jobs concurrently - semaphore will limit actual concurrency
            tasks = [asyncio.create_task(process_batch_job(jid)) for jid in claimed]
            
            # Wait for all tasks to complete or preemption
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                # If cancelled due to preemption, tasks will handle their own cleanup
                pass

            # After batch work is done, return to realtime model
            log.info("batch_segment_complete", model=best_model, jobs_processed=len(claimed))
            
            # Return to realtime model if no more batch work
            remaining_jobs = await redis.zcard(BATCH_ZSET)
            if remaining_jobs == 0 and state.get("current_model") != REALTIME_MODEL:
                log.info("no_batch_jobs_remaining_loading_realtime")
                await asyncio.get_running_loop().run_in_executor(None, stop_runtime_container_sync)
                try:
                    await asyncio.get_running_loop().run_in_executor(None, lambda: start_runtime_container_sync(REALTIME_MODEL))
                    ok = await wait_for_runtime_ready(timeout=60)
                    if ok:
                        state["current_model"] = REALTIME_MODEL
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

@app.options("/upload")
async def tus_options():
    """TUS discovery endpoint"""
    return Response(
        status_code=204,
        headers={
            "Tus-Version": TUS_VERSION,
            "Tus-Max-Size": str(TUS_MAX_SIZE),
            "Tus-Extension": "creation"
        }
    )

@app.post("/upload")
async def tus_create(
    request: Request,
    auth: bool = Depends(verify_api_key),
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
    await redis.hset(f"upload:{upload_id}", mapping={
        "model_name": model_name,
        "size": str(upload_length),
        "offset": "0",
        "temp_path": temp_path,
        "status": "uploading",
        "created": str(int(now()))
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

@app.head("/upload/{upload_id}")
async def tus_head(
    upload_id: str,
    auth: bool = Depends(verify_api_key),
    tus_resumable: str = Header(TUS_VERSION)
):
    """TUS resumption endpoint - returns current offset"""
    upload_data = await redis.hgetall(f"upload:{upload_id}")
    if not upload_data:
        raise HTTPException(status_code=404)
    
    return Response(
        headers={
            "Upload-Offset": upload_data["offset"],
            "Upload-Length": upload_data["size"],
            "Tus-Resumable": TUS_VERSION,
            "Cache-Control": "no-store"
        }
    )

@app.patch("/upload/{upload_id}")
async def tus_patch(
    upload_id: str,
    request: Request,
    auth: bool = Depends(verify_api_key),
    upload_offset: int = Header(...),
    content_type: str = Header(...),
    tus_resumable: str = Header(TUS_VERSION)
):
    """TUS chunk upload endpoint"""
    if content_type != "application/offset+octet-stream":
        raise HTTPException(status_code=415)
    
    upload_data = await redis.hgetall(f"upload:{upload_id}")
    if not upload_data:
        raise HTTPException(status_code=404)
    
    # Check offset matches
    current_offset = int(upload_data["offset"])
    if upload_offset != current_offset:
        raise HTTPException(status_code=409, detail="Offset mismatch")
    
    # Read and append chunk
    chunk = await request.body()
    temp_path = upload_data["temp_path"]
    
    async with aiofiles.open(temp_path, "ab") as f:
        await f.write(chunk)
    
    new_offset = current_offset + len(chunk)
    await redis.hset(f"upload:{upload_id}", "offset", str(new_offset))
    
    # Check if complete
    if new_offset >= int(upload_data["size"]):
        await _finalize_upload(upload_id, upload_data)
    
    return Response(
        status_code=204,
        headers={
            "Upload-Offset": str(new_offset),
            "Tus-Resumable": TUS_VERSION
        }
    )

async def _finalize_upload(upload_id: str, upload_data: dict):
    """Extract and install uploaded model"""
    temp_path = upload_data["temp_path"]
    model_name = upload_data["model_name"]
    
    try:
        # Target directory in HF cache format
        model_dir = f"models--{model_name.replace('/', '--')}"
        target_path = os.path.join(HOST_HF_CACHE, "hub", model_dir, "snapshots", "main")
        
        # Clean existing if present
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        os.makedirs(target_path, exist_ok=True)
        
        log.info("extracting_model", model=model_name, path=target_path)
        
        # Decompress and extract
        dctx = zstd.ZstdDecompressor()
        with open(temp_path, 'rb') as compressed:
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as tar:
                    for member in tar:
                        # Security check
                        if os.path.isabs(member.name) or ".." in member.name:
                            continue
                        
                        # Strip leading directory if present
                        parts = member.name.split('/', 1)
                        if len(parts) > 1:
                            member.name = parts[1]
                        
                        tar.extract(member, target_path)
        
        # Validate model files exist
        if not os.path.exists(os.path.join(target_path, "config.json")):
            raise ValueError("Missing config.json")
        
        # Check for model weights
        has_weights = any(
            f.endswith(('.safetensors', '.bin', '.gguf'))
            for f in os.listdir(target_path)
        )
        if not has_weights:
            raise ValueError("No model weights found")
        
        # Mark complete
        await redis.hset(f"upload:{upload_id}", mapping={
            "status": "completed",
            "completed": str(int(now()))
        })
        
        # Store model metadata
        size = sum(
            os.path.getsize(os.path.join(target_path, f))
            for f in os.listdir(target_path)
        )
        await redis.hset(f"{MODEL_META_PREFIX}{model_name}", mapping={
            "type": "uploaded",
            "size_gb": str(size / (1024**3)),
            "uploaded_at": str(int(now())),
            "load_est_s": "30"
        })
        
        log.info("model_uploaded", model=model_name, size_gb=size/(1024**3))
        
    except Exception as e:
        log.error("upload_failed", model=model_name, error=str(e))
        await redis.hset(f"upload:{upload_id}", "status", "failed")
        raise
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
