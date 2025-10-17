"""
Evaluation Manager for orchestrating model evaluations.
Coordinates between training repos, eval repos, and the gateway.
"""

import asyncio
import json
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Any
from enum import Enum
from model_utils import detect_is_base_model
import structlog
from fastapi import APIRouter, HTTPException, Depends, Request
import math

logger = structlog.get_logger()


async def run_eval_command(eval_name: str, eval_repo_path: Path, cmd_args: List[str], 
                          stdin_data: Optional[str] = None) -> tuple[int, bytes, bytes]:
    """
    Run an eval command in its own virtual environment if it exists.
    Creates the venv if needed and the eval has dependencies.
    
    Returns: (returncode, stdout, stderr)
    """
    # Ensure eval_repo_path is a Path object
    if isinstance(eval_repo_path, str):
        eval_repo_path = Path(eval_repo_path)
    
    eval_dir = eval_repo_path / "evals" / eval_name
    venv_path = eval_dir / ".venv"
    
    # Check if eval needs a venv (has dependencies)
    has_deps = (eval_dir / "pyproject.toml").exists() or (eval_dir / "requirements.txt").exists()
    
    if has_deps:
        # Ensure venv exists (and repair if broken)
        async def create_or_repair_venv(recreate: bool = False) -> bool:
            try:
                # If requested, remove any existing venv first
                if recreate and venv_path.exists():
                    try:
                        shutil.rmtree(venv_path)
                    except Exception:
                        pass
                args = ["/usr/local/bin/uv", "venv", str(venv_path)]
                venv_result = await asyncio.create_subprocess_exec(
                    *args,
                    cwd=str(eval_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = await venv_result.communicate()
                if venv_result.returncode != 0:
                    logger.error(f"Failed to {'re' if recreate else ''}create venv: {stderr.decode(errors='replace')}")
                    return False
                logger.info(f"{'Re' if recreate else 'C'}reated venv successfully for {eval_name}")
                return True
            except Exception as e:
                logger.error(f"Exception creating venv: {e}")
                return False

        # Create if missing
        if not venv_path.exists():
            ok_venv = await create_or_repair_venv(recreate=False)
            if not ok_venv:
                # Fail early if venv cannot be created
                return 1, b"", f"Failed to create venv for {eval_name}".encode()

        # Detect broken interpreter symlinks and repair
        py3 = venv_path / "bin" / "python3"
        py = venv_path / "bin" / "python"
        def _broken(p: Path) -> bool:
            # Broken if the path lexists but does not exist (dangling symlink)
            try:
                return p.is_symlink() and not p.exists()
            except Exception:
                return False
        if _broken(py3) or _broken(py):
            logger.warning(f"Detected broken venv interpreter for {eval_name}; reinstalling venv")
            ok_fix = await create_or_repair_venv(recreate=True)
            if not ok_fix:
                return 1, b"", f"Failed to repair venv for {eval_name}".encode()

        # Install dependencies using uv pip (pip-less venvs supported)
        uv_env = {**os.environ, "VIRTUAL_ENV": str(venv_path), "PIP_DISABLE_PIP_VERSION_CHECK": "1"}

        if (eval_dir / "pyproject.toml").exists():
            logger.info(f"Syncing dependencies for {eval_name} (pyproject.toml via uv pip)...")
            install_result = await asyncio.create_subprocess_exec(
                "/usr/local/bin/uv", "pip", "install", "-e", ".",
                cwd=str(eval_dir), env=uv_env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = await install_result.communicate()
            if install_result.returncode != 0:
                err_text = err.decode(errors='replace')
                logger.error(f"uv pip install -e . failed for {eval_name}: {err_text}")
                # If failure indicates broken interpreter, repair and retry once
                if "Broken symlink" in err_text or "Failed to inspect Python interpreter" in err_text:
                    logger.info(f"Repairing venv for {eval_name} and retrying uv pip install -e .")
                    ok_fix = await create_or_repair_venv(recreate=True)
                    if not ok_fix:
                        return 1, b"", f"Failed to repair venv for {eval_name}".encode()
                    install_result = await asyncio.create_subprocess_exec(
                        "/usr/local/bin/uv", "pip", "install", "-e", ".",
                        cwd=str(eval_dir), env=uv_env,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    out, err = await install_result.communicate()
                    if install_result.returncode == 0:
                        logger.info(f"uv pip install -e . succeeded after venv repair for {eval_name}")
                    else:
                        err_text = err.decode(errors='replace')
                        logger.error(f"uv pip install -e . failed after repair for {eval_name}: {err_text}")
                # Fallback: install only declared dependencies from pyproject without packaging the project
                try:
                    import tomllib
                    with open(eval_dir / "pyproject.toml", "rb") as f:
                        pyproj = tomllib.load(f)
                    deps = pyproj.get("project", {}).get("dependencies", [])
                    if deps:
                        logger.info(f"Falling back to installing declared dependencies for {eval_name} via uv pip: {len(deps)} packages")
                        fallback = await asyncio.create_subprocess_exec(
                            "/usr/local/bin/uv", "pip", "install", *deps,
                            cwd=str(eval_dir), env=uv_env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                        fout, ferr = await fallback.communicate()
                        if fallback.returncode != 0:
                            ferr_text = ferr.decode(errors='replace')
                            logger.error(f"uv pip install deps fallback failed for {eval_name}: {ferr_text}")
                            if "Broken symlink" in ferr_text or "Failed to inspect Python interpreter" in ferr_text:
                                logger.info(f"Repairing venv for {eval_name} and retrying deps fallback")
                                ok_fix = await create_or_repair_venv(recreate=True)
                                if not ok_fix:
                                    return 1, b"", f"Failed to repair venv for {eval_name}".encode()
                                fallback = await asyncio.create_subprocess_exec(
                                    "/usr/local/bin/uv", "pip", "install", *deps,
                                    cwd=str(eval_dir), env=uv_env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                                )
                                fout, ferr = await fallback.communicate()
                                if fallback.returncode != 0:
                                    logger.error(f"uv pip install deps fallback failed after repair for {eval_name}: {ferr.decode(errors='replace')}")
                    else:
                        logger.warning(f"No dependencies found in pyproject for {eval_name}; skipping install fallback")
                except Exception as fe:
                    logger.error(f"Failed to parse pyproject for deps fallback ({eval_name}): {fe}")
        elif (eval_dir / "requirements.txt").exists():
            logger.info(f"Syncing dependencies for {eval_name} (requirements.txt via uv pip)...")
            install_result = await asyncio.create_subprocess_exec(
                "/usr/local/bin/uv", "pip", "install", "-r", "requirements.txt",
                cwd=str(eval_dir), env=uv_env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = await install_result.communicate()
            if install_result.returncode != 0:
                err_text = err.decode(errors='replace')
                logger.error(f"uv pip install -r requirements.txt failed for {eval_name}: {err_text}")
                if "Broken symlink" in err_text or "Failed to inspect Python interpreter" in err_text:
                    logger.info(f"Repairing venv for {eval_name} and retrying uv pip -r requirements.txt")
                    ok_fix = await create_or_repair_venv(recreate=True)
                    if not ok_fix:
                        return 1, b"", f"Failed to repair venv for {eval_name}".encode()
                    install_result = await asyncio.create_subprocess_exec(
                        "/usr/local/bin/uv", "pip", "install", "-r", "requirements.txt",
                        cwd=str(eval_dir), env=uv_env,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    out, err = await install_result.communicate()
                    if install_result.returncode != 0:
                        logger.error(f"uv pip install -r requirements.txt failed after repair for {eval_name}: {err.decode(errors='replace')}")

        # Use the venv's Python
        python_exec = venv_path / "bin" / "python"
        if not python_exec.exists():
            return 1, b"", f"Venv interpreter missing for {eval_name}".encode()
        python_cmd = str(python_exec)
    else:
        # No dependencies, use system Python
        python_cmd = "python"
    
    # Build command with the appropriate Python
    full_cmd = [python_cmd] + cmd_args
    
    logger.info(f"Running eval command for {eval_name}: {' '.join(full_cmd)}")
    logger.info(f"Working directory: {str(eval_dir) if has_deps else 'default'}")
    logger.info(f"Python command: {python_cmd}, Has deps: {has_deps}")
    
    # Run the command
    if stdin_data:
        result = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE
        )
        stdout, stderr = await result.communicate(input=stdin_data.encode())
    else:
        result = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
    
    return result.returncode, stdout, stderr


def request_to_redis_job(request: Dict[str, Any], model: str, comp_id: str, **extra_fields) -> Dict[str, str]:
    """
    Convert an OpenAI-format request to Redis job format for batch processing.
    
    The batch system expects individual fields stored as strings in Redis,
    not a nested JSON request object.
    """
    job_data = {
        "completion_id": comp_id,
        "id": comp_id,  # Some code paths use "id" 
        "model": model,
        "status": "queued",
        "created_at": str(int(datetime.utcnow().timestamp())),
        "submit_ts": str(datetime.utcnow().timestamp()),
        "created": str(int(datetime.utcnow().timestamp())),
        **extra_fields  # For eval_name, etc.
    }
    
    # Determine type and flatten fields based on request format
    if "messages" in request:
        job_data["type"] = "chat_completion"
        job_data["messages"] = json.dumps(request["messages"])
    elif "prompt" in request:
        job_data["type"] = "completion"
        job_data["prompt"] = request["prompt"]
    else:
        raise ValueError("Request must have either 'messages' or 'prompt'")
    
    # Common fields (stored as strings for Redis)
    job_data["max_tokens"] = str(request.get("max_tokens", 100))
    job_data["temperature"] = str(request.get("temperature", 0.7))
    job_data["top_p"] = str(request.get("top_p", 1.0))
    job_data["n"] = str(request.get("n", 1))
    job_data["stop"] = json.dumps(request.get("stop"))
    job_data["priority"] = str(request.get("priority", 0))
    # Optional: logprobs/top_logprobs for scoring
    # Preserve integer top-K if provided; else coerce boolean to "1"/"0"
    if "logprobs" in request and request["logprobs"] is not None:
        lp_val = request.get("logprobs")
        try:
            # Prefer integer K for completions API compatibility
            if isinstance(lp_val, int):
                job_data["logprobs"] = str(int(lp_val))
            else:
                job_data["logprobs"] = "1" if bool(lp_val) else "0"
        except Exception:
            job_data["logprobs"] = "1" if bool(lp_val) else "0"
    if "top_logprobs" in request and request["top_logprobs"] is not None:
        try:
            job_data["top_logprobs"] = str(int(request.get("top_logprobs")))
        except Exception:
            # Ignore malformed values
            pass
    # Optional: echo (needed for prompt logprobs when max_tokens=0)
    if "echo" in request and request["echo"] is not None:
        job_data["echo"] = "1" if bool(request.get("echo")) else "0"
    
    return job_data


def redis_job_to_request(job_data: Dict[str, str]) -> Dict[str, Any]:
    """
    Reconstruct an OpenAI-format request from Redis job data.
    
    Inverse of request_to_redis_job.
    """
    request = {
        "model": job_data.get("model", ""),
        "max_tokens": int(job_data.get("max_tokens", 100)),
        "temperature": float(job_data.get("temperature", 0.7)),
        "top_p": float(job_data.get("top_p", 1.0)),
        "n": int(job_data.get("n", 1)),
    }
    
    # Handle stop sequences
    stop_str = job_data.get("stop", "null")
    if stop_str and stop_str != "null":
        request["stop"] = json.loads(stop_str)
    
    # Add type-specific fields
    job_type = job_data.get("type", "")
    if job_type == "chat_completion":
        messages_str = job_data.get("messages", "[]")
        request["messages"] = json.loads(messages_str)
    elif job_type == "completion":
        request["prompt"] = job_data.get("prompt", "")
    
    # Optional logprobs fields
    lp = job_data.get("logprobs")
    if lp is not None and lp != "":
        try:
            # If numeric string, return int top-K; otherwise boolean
            request["logprobs"] = int(lp) if lp.isdigit() else (True if lp in ("1", "true", "True") else False)
        except Exception:
            pass
    tlp = job_data.get("top_logprobs")
    if tlp is not None and tlp != "":
        try:
            request["top_logprobs"] = int(tlp)
        except Exception:
            pass
    # Optional echo
    ech = job_data.get("echo")
    if ech is not None and ech != "":
        try:
            request["echo"] = True if ech in ("1", "true", "True") else False
        except Exception:
            pass

    return request


class EvalStatus(str, Enum):
    PENDING = "pending"
    SUBMITTING = "submitting"
    RUNNING = "running"
    SCORING = "scoring"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalManager:
    """Manages evaluation jobs across multiple eval repositories."""
    
    def __init__(self, redis_client, batch_manager):
        self.redis = redis_client
        self.batch_manager = batch_manager
        self.eval_repo_path = os.getenv("EVAL_REPO_PATH", "/home/alex-admin/llm-gateway/eval-repo")
        self.data_dir = Path("/data/eval_results")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Track active evaluations
        self.active_evals: Dict[str, asyncio.Task] = {}
    
    async def submit_eval(self, model: str, evals: List[str] | None = None, prepare_args: Dict | None = None) -> str:
        """
        Submit a model for evaluation.
        
        Args:
            model: Model name/path to evaluate
            
        Returns:
            job_id for tracking the evaluation
        """
        job_id = f"eval_{uuid.uuid4().hex[:12]}"
        
        # Create initial job state
        job_data = {
            "job_id": job_id,
            "model": model,
            "status": EvalStatus.PENDING,
            "created_at": datetime.utcnow().isoformat(),
            "eval_status": {},
            "completion_ids": [],
            "eval_groups": {},
            # Optional overrides
            "requested_evals": evals or [],
            "prepare_args": prepare_args or {},
        }
        
        # Store in Redis with 30 day TTL
        await self.redis.setex(
            f"eval_job:{job_id}",
            30 * 24 * 3600,
            json.dumps(job_data)
        )
        
        # Start async evaluation task
        task = asyncio.create_task(
            self._run_eval(job_id, model)
        )
        self.active_evals[job_id] = task
        
        return job_id
    
    async def get_eval_status(self, job_id: str) -> Optional[Dict]:
        """Get the current status of an evaluation job."""
        data = await self.redis.get(f"eval_job:{job_id}")
        if data:
            return json.loads(data)
        return None
    
    async def _run_eval(self, job_id: str, model: str):
        """Run the full evaluation workflow."""
        try:
            # Update status to submitting
            await self._update_job_status(job_id, EvalStatus.SUBMITTING)
            
            # Phase 1: Submit requests to eval repo
            submission_result = await self._submit_requests(job_id, model)
            
            if not submission_result:
                await self._update_job_status(job_id, EvalStatus.FAILED, 
                                            error="Failed to submit requests")
                return
            
            # Store completion IDs and eval groups
            await self._update_job_data(job_id, {
                "completion_ids": submission_result.get("completion_ids", []),
                "eval_groups": submission_result.get("eval_groups", {}),
                "metadata": submission_result.get("metadata", {})
            })
            
            # Update status to running
            await self._update_job_status(job_id, EvalStatus.RUNNING)
            
            # Monitor completions per eval group
            eval_groups = submission_result.get("eval_groups", {})
            for eval_name, completion_ids in eval_groups.items():
                # Start monitoring this eval's completions
                asyncio.create_task(
                    self._monitor_eval_completions(job_id, eval_name, completion_ids)
                )
            
        except Exception as e:
            logger.error(f"Eval {job_id} failed: {e}")
            await self._update_job_status(job_id, EvalStatus.FAILED, error=str(e))
        finally:
            # Clean up from active evals
            self.active_evals.pop(job_id, None)
    
    async def _check_is_base_model(self, model: str) -> bool:
        """Detect base vs instruction. Runs in a thread to avoid blocking the event loop.
        Also ensures minimal tokenizer/config files are fetched into the cache if missing.
        """
        try:
            loop = asyncio.get_running_loop()
            is_base = await loop.run_in_executor(None, detect_is_base_model, model)
            logger.info(f"Model {model} is_base: {is_base}")
            return is_base
        except Exception as e:
            logger.warning(f"Detection error for {model}: {e}; defaulting to instruct")
            return False
    
    async def _submit_requests(self, job_id: str, model: str) -> Optional[Dict]:
        """
        Phase 1: Get requests from eval repo and submit them as batch jobs.
        
        Returns dict with completion_ids, eval_groups, and metadata.
        """
        # Load overrides from job record if present
        try:
            job_raw = await self.redis.get(f"eval_job:{job_id}")
            job_cfg = json.loads(job_raw) if job_raw else {}
            requested_evals = job_cfg.get("requested_evals") or []
            prepare_args = job_cfg.get("prepare_args") or {}
        except Exception:
            requested_evals = []
            prepare_args = {}
        # Check if model is a base model
        is_base_model = await self._check_is_base_model(model)
        
        # Determine evals to run: overrides or defaults from repo list
        if requested_evals:
            eval_names = [str(e) for e in requested_evals]
        else:
            list_cmd = ["python", f"{self.eval_repo_path}/run_eval.py", "list"]
            if is_base_model:
                list_cmd.append("--base-model")
            try:
                result = await asyncio.create_subprocess_exec(
                    *list_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = await result.communicate()
                if result.returncode != 0:
                    logger.error(f"Failed to list evals: {stderr.decode()}")
                    eval_names = ["example_eval"]
                else:
                    list_result = json.loads(stdout.decode())
                    eval_names = list_result.get("evals", ["example_eval"])
            except Exception as e:
                logger.error(f"Error getting eval list: {e}")
                eval_names = ["example_eval"]
        
        logger.info(f"Running evals: {eval_names}")
        
        all_completion_ids: list[str] = []
        eval_groups: dict[str, list[str]] = {}
        metadata = {
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
            "eval_suite": "standard",
        }

        # Prepare and submit requests for all evals concurrently so the planner sees work ASAP
        sem = asyncio.Semaphore(int(os.getenv("EVAL_PREP_CONCURRENCY", "3")))

        async def prepare_and_submit(eval_name: str):
            async with sem:
                try:
                    cmd_args = [
                        f"{self.eval_repo_path}/run_eval.py",
                        "prepare", eval_name, model
                    ]
                    if is_base_model:
                        cmd_args.append("--base-model")
                    # Optional: apply prepare_args (support max_prompts)
                    try:
                        mp = prepare_args.get("max_prompts")
                        if mp is not None:
                            cmd_args.extend(["--max-prompts", str(int(mp))])
                    except Exception:
                        pass

                    rc, out, err = await run_eval_command(eval_name, self.eval_repo_path, cmd_args)
                    if rc != 0:
                        # Persist prepare stdout/stderr for troubleshooting
                        try:
                            job = await self.get_eval_status(job_id)
                            created_at = job.get("created_at") if job else None
                            ts = datetime.fromisoformat(created_at).strftime("%Y-%m-%d_%H-%M-%S") if created_at else datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
                            safe_model = model.replace("/", "--").replace(":", "_")
                            log_dir = self.data_dir / safe_model / ts
                            log_dir.mkdir(parents=True, exist_ok=True)
                            log_file = log_dir / f"{eval_name}_prepare.log"
                            with open(log_file, "w", encoding="utf-8") as f:
                                f.write(f"Command: {' '.join(cmd_args)}\n")
                                f.write(f"Return code: {rc}\n\n")
                                if out:
                                    try:
                                        f.write("[STDOUT]\n")
                                        f.write(out.decode(errors='replace'))
                                        f.write("\n\n")
                                    except Exception:
                                        pass
                                if err:
                                    try:
                                        f.write("[STDERR]\n")
                                        f.write(err.decode(errors='replace'))
                                        f.write("\n")
                                    except Exception:
                                        pass
                            logger.error(f"prepare requests failed for {eval_name}: {err.decode(errors='replace')}", log_path=str(log_file))
                        except Exception as le:
                            logger.error(f"prepare requests failed for {eval_name} and could not write log: {le}")
                        return eval_name, []

                    prepare_result = json.loads(out.decode())
                    requests = prepare_result.get("requests", [])
                    eval_meta = prepare_result.get("metadata", {})
                    # Extract optional per-eval max concurrency (scalar)
                    eval_max_conc = None
                    try:
                        mc = eval_meta.get("max_concurrency")
                        if mc is not None:
                            eval_max_conc = int(mc)
                    except Exception:
                        eval_max_conc = None

                    completion_ids_local: list[str] = []
                    for request in requests:
                        comp_id = f"comp_{uuid.uuid4().hex[:12]}"
                        request["model"] = model
                        extra = {
                            "eval_name": eval_name,
                            "eval_job_id": job_id,
                        }
                        if eval_max_conc is not None:
                            extra["eval_max_concurrency"] = str(eval_max_conc)
                        job_data = request_to_redis_job(
                            request,
                            model,
                            comp_id,
                            **extra,
                        )
                        await self.redis.hset(f"job:{comp_id}", mapping=job_data)
                        await self.redis.expire(f"job:{comp_id}", 7 * 24 * 3600)
                        await self.redis.zadd("z:batchjobs", {comp_id: datetime.utcnow().timestamp()})
                        completion_ids_local.append(comp_id)

                    # Store per-eval metadata for scoring
                    await self.redis.setex(
                        f"eval_metadata:{job_id}:{eval_name}",
                        7 * 24 * 3600,
                        json.dumps(eval_meta)
                    )

                    logger.info(f"Submitted {len(completion_ids_local)} requests for {eval_name}")
                    return eval_name, completion_ids_local
                except Exception as e:
                    import traceback
                    logger.error(f"Error processing eval {eval_name}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return eval_name, []

        tasks = [asyncio.create_task(prepare_and_submit(name)) for name in eval_names]
        for task in asyncio.as_completed(tasks):
            name, cids = await task
            if cids:
                eval_groups[name] = cids
                all_completion_ids.extend(cids)

        metadata["total_requests"] = len(all_completion_ids)
        
        return {
            "completion_ids": all_completion_ids,
            "eval_groups": eval_groups,
            "metadata": metadata
        }
    
    async def _monitor_eval_completions(self, job_id: str, eval_name: str, 
                                       completion_ids: List[str]):
        """Monitor completions for a specific eval and trigger scoring when ready."""
        try:
            # Update eval-specific status
            await self._update_eval_status(job_id, eval_name, {
                "status": "running",
                "total": len(completion_ids),
                "completed": 0,
                "failed": 0
            })
            
            # Poll for completion status
            while True:
                completed = 0
                failed = 0
                
                for comp_id in completion_ids:
                    # Check batch job status (stored as hash)
                    job_data = await self.redis.hgetall(f"job:{comp_id}")
                    if job_data:
                        job = job_data  # Already a dict from hgetall
                        status = job.get("status")
                        
                        if status == "completed":
                            completed += 1
                        elif status in ["failed", "error"]:
                            failed += 1
                
                # Update progress
                await self._update_eval_status(job_id, eval_name, {
                    "status": "running",
                    "progress": f"{completed}/{len(completion_ids)}",
                    "completed": completed,
                    "failed": failed
                })
                
                # Check if all are done
                if completed + failed >= len(completion_ids):
                    break
                
                # Wait before next check
                await asyncio.sleep(5)
            
            # All completions done, trigger scoring
            await self._update_eval_status(job_id, eval_name, {"status": "scoring"})
            
            score_result = await self._score_results(job_id, eval_name, completion_ids)
            
            if score_result:
                await self._update_eval_status(job_id, eval_name, {
                    "status": "completed",
                    "results": score_result.get("metrics", {}),
                    "details": score_result.get("details", {})
                })
                
                # Save to persistent storage
                await self._save_eval_result(job_id, eval_name, score_result)
            else:
                await self._update_eval_status(job_id, eval_name, {
                    "status": "failed",
                    "error": "Scoring failed"
                })
                
        except Exception as e:
            logger.error(f"Error monitoring eval {eval_name} for job {job_id}: {e}")
            await self._update_eval_status(job_id, eval_name, {
                "status": "failed",
                "error": str(e)
            })
    
    async def _score_results(self, job_id: str, eval_name: str, 
                            completion_ids: List[str]) -> Optional[Dict]:
        """
        Phase 2: Gather results and call eval repo to score them.
        
        Returns dict with metrics and details.
        """
        try:
            # Gather all requests and responses using completion IDs as keys
            # This ensures request[i] matches response[i] regardless of completion order
            request_response_pairs = {}
            http_error_summary = {"total_errors": 0, "by_code": {}, "by_type": {}, "examples": []}
            
            for comp_id in completion_ids:
                # Get job data from Redis hash (same format as batch endpoints use)
                job_data = await self.redis.hgetall(f"job:{comp_id}")
                if not job_data:
                    logger.warning(f"Missing job data for {comp_id}")
                    request_response_pairs[comp_id] = (
                        {},
                        {"error": "Job data not found"}
                    )
                    continue
                
                # Reconstruct the request from flattened Redis fields
                request = redis_job_to_request(job_data)
                
                # Get the response if available
                status = job_data.get("status", "unknown")
                if status == "completed":
                    result_str = job_data.get("result", "{}")
                    response = json.loads(result_str)
                    
                    # Debug logging
                    with open("/tmp/eval_debug.log", "a") as f:
                        f.write(f"\n[Score phase] Comp ID: {comp_id}\n")
                        f.write(f"Status: {status}\n")
                        if "choices" in response and response["choices"]:
                            content = response["choices"][0].get("message", {}).get("content", "")
                            f.write(f"Retrieved content length: {len(content)}\n")
                            f.write(f"Retrieved content (first 100): {content[:100]!r}\n")
                            f.write(f"Finish reason: {response['choices'][0].get('finish_reason', 'unknown')}\n")
                        else:
                            f.write(f"Response structure issue: {json.dumps(response)[:200]}\n")
                elif status == "error":
                    # Capture HTTP-layer error info if available
                    err_msg = job_data.get("error", "Unknown error")
                    err_code = job_data.get("error_code")
                    err_type = job_data.get("error_type")
                    retry_count = job_data.get("retry_count")
                    response = {"error": err_msg, "error_code": err_code, "error_type": err_type, "retry_count": retry_count}
                    try:
                        http_error_summary["total_errors"] += 1
                        if err_code:
                            http_error_summary["by_code"][str(err_code)] = http_error_summary["by_code"].get(str(err_code), 0) + 1
                        if err_type:
                            http_error_summary["by_type"][str(err_type)] = http_error_summary["by_type"].get(str(err_type), 0) + 1
                        if len(http_error_summary["examples"]) < 5:
                            http_error_summary["examples"].append({
                                "comp_id": comp_id,
                                "error_code": err_code,
                                "error_type": err_type,
                                "message": (err_msg or "")[:200],
                            })
                    except Exception:
                        pass
                else:
                    response = {"error": f"Job status: {status}"}
                
                request_response_pairs[comp_id] = (request, response)
            
            # Now extract in the original order
            requests = []
            responses = []
            for comp_id in completion_ids:
                if comp_id in request_response_pairs:
                    req, resp = request_response_pairs[comp_id]
                    requests.append(req)
                    responses.append(resp)
                else:
                    # This shouldn't happen, but handle it
                    requests.append({})
                    responses.append({"error": "Missing completion data"})
            
            # Get eval metadata stored during prepare phase
            eval_metadata_key = f"eval_metadata:{job_id}:{eval_name}"
            metadata_json = await self.redis.get(eval_metadata_key)
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            # Prepare input for scoring
            score_input = {
                "requests": requests,
                "responses": responses,
                "metadata": metadata
            }

            logger.info(
                "score_input_metadata",
                eval_name=eval_name,
                has_dataset_metadata=bool(metadata.get("dataset_metadata")) if isinstance(metadata, dict) else False,
                metadata_keys=list(metadata.keys()) if isinstance(metadata, dict) else None,
            )
            
            # Debug: Log sample of what we're sending to scoring
            with open("/tmp/eval_debug.log", "a") as f:
                f.write(f"\n[Sending to score] Eval: {eval_name}\n")
                f.write(f"Number of requests: {len(requests)}\n")
                f.write(f"Number of responses: {len(responses)}\n")
                if responses and len(responses) > 0:
                    f.write(f"Sample response structure: {json.dumps(responses[0], indent=2)[:500]}\n")
                    # Check response format consistency
                    has_choices = sum(1 for r in responses if isinstance(r, dict) and "choices" in r)
                    has_content = sum(1 for r in responses if isinstance(r, dict) and "choices" in r and r["choices"] and "message" in r["choices"][0])
                    has_text = sum(1 for r in responses if isinstance(r, dict) and "choices" in r and r["choices"] and "text" in r["choices"][0])
                    f.write(f"Responses with choices: {has_choices}/{len(responses)}\n")
                    f.write(f"Responses with message.content: {has_content}/{len(responses)}\n")
                    f.write(f"Responses with text: {has_text}/{len(responses)}\n")
            
            # Save requests and responses for debugging
            await self._save_eval_interactions(job_id, eval_name, requests, responses, metadata)
            
            # Call eval repo to score
            cmd_args = [
                f"{self.eval_repo_path}/run_eval.py",
                "score", eval_name
            ]
            
            returncode, stdout, stderr = await run_eval_command(
                eval_name, self.eval_repo_path, cmd_args,
                stdin_data=json.dumps(score_input)
            )
            
            if returncode != 0:
                logger.error(f"score_results failed for {eval_name}: {stderr.decode()}")
                return None
            
            # Parse JSON output and add raw requests/responses
            score_result = json.loads(stdout.decode())
            score_result["requests"] = requests
            score_result["responses"] = responses
            if http_error_summary["total_errors"] > 0:
                score_result.setdefault("details", {})["http_errors"] = http_error_summary
            return score_result
            
        except Exception as e:
            logger.error(f"Error scoring results for {eval_name}: {e}")
            return None
    
    async def _save_eval_interactions(self, job_id: str, eval_name: str,
                                     requests: List[Dict], responses: List[Dict], metadata: Dict,
                                     annotations: Optional[List[Optional[Dict[str, Any]]]] = None):
        """Save requests and responses for debugging and analysis.

        Args:
            job_id: Eval job identifier.
            eval_name: Name of the evaluation.
            requests: List of request payloads.
            responses: List of response payloads.
            metadata: Metadata captured during request preparation.
            annotations: Optional per-example judgement metadata (aligned with requests).
        """
        try:
            job_data = await self.get_eval_status(job_id)
            if not job_data:
                return
            
            model = job_data.get("model", "unknown")
            # Sanitize model name for filesystem
            safe_model = model.replace("/", "--").replace(":", "_")
            
            # Use consistent timestamp from job creation
            created_at = job_data.get("created_at", datetime.utcnow().isoformat())
            job_timestamp = datetime.fromisoformat(created_at).strftime("%Y-%m-%d_%H-%M-%S")
            eval_job_dir = self.data_dir / safe_model / job_timestamp
            eval_job_dir.mkdir(parents=True, exist_ok=True)
            
            analysis_rows: Optional[List[Dict[str, Any]]] = [] if annotations is not None else None

            def _json_safe(value: Any) -> Any:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    return value
                if hasattr(value, "item") and callable(getattr(value, "item")):
                    try:
                        return value.item()  # type: ignore[no-any-return]
                    except Exception:
                        pass
                if isinstance(value, list):
                    return [_json_safe(v) for v in value]
                if isinstance(value, tuple):
                    return [_json_safe(v) for v in value]
                if isinstance(value, dict):
                    return {str(k): _json_safe(v) for k, v in value.items()}
                return str(value)

            # Save human-readable interactions file
            interactions_file = eval_job_dir / f"{eval_name}_interactions.txt"
            with open(interactions_file, "w", encoding="utf-8") as f:
                f.write(f"=" * 80 + "\n")
                f.write(f"EVAL: {eval_name}\n")
                f.write(f"MODEL: {model}\n")
                f.write(f"TIMESTAMP: {created_at}\n")
                f.write(f"TOTAL EXAMPLES: {len(requests)}\n")
                f.write(f"=" * 80 + "\n\n")
                
                # Group by subdataset if available (for EuroEval)
                grouped = {}
                for i, (req, resp) in enumerate(zip(requests, responses)):
                    subdataset = req.get("_dataset", "default")
                    if subdataset not in grouped:
                        grouped[subdataset] = []
                    grouped[subdataset].append((i, req, resp))
                
                # Write each group
                for subdataset, examples in grouped.items():
                    if subdataset != "default":
                        f.write(f"\n{'='*60}\n")
                        f.write(f"SUBDATASET: {subdataset} ({len(examples)} examples)\n")
                        f.write(f"{'='*60}\n\n")
                    
                    for idx, req, resp in examples:
                        f.write(f"--- Example {idx + 1} ---\n")
                        annotation = annotations[idx] if annotations and idx < len(annotations) else None
                        
                        # Write prompt/messages
                        if "messages" in req:
                            for msg in req["messages"]:
                                role = msg.get("role", "unknown")
                                content = msg.get("content", "")
                                f.write(f"[{role.upper()}]:\n{content}\n\n")
                        elif "prompt" in req:
                            f.write(f"[PROMPT]:\n{req['prompt']}\n\n")
                        
                        # Write response
                        f.write(f"[RESPONSE]:\n")
                        response_text = ""
                        finish_reason = "unknown"
                        if isinstance(resp, dict):
                            if "choices" in resp and len(resp["choices"]) > 0:
                                choice = resp["choices"][0]
                                # Handle both chat (message.content) and completion (text) formats
                                content = None
                                if "message" in choice:
                                    message = choice.get("message")
                                    if message and "content" in message:
                                        content = message["content"]
                                        if not content:
                                            logger.warning("empty_message_content", 
                                                         eval_name=eval_name,
                                                         example_idx=i,
                                                         subdataset=subdataset,
                                                         finish_reason=choice.get("finish_reason", "unknown"))
                                    else:
                                        logger.warning("missing_message_content", 
                                                     eval_name=eval_name,
                                                     example_idx=i,
                                                     subdataset=subdataset,
                                                     message_keys=list(message.keys()) if message else None)
                                        content = ""
                                elif "text" in choice:
                                    content = choice["text"]
                                    if not content:
                                        logger.warning("empty_text_response", 
                                                     eval_name=eval_name,
                                                     example_idx=i,
                                                     subdataset=subdataset,
                                                     finish_reason=choice.get("finish_reason", "unknown"))
                                else:
                                    logger.error("unknown_response_format", 
                                               eval_name=eval_name,
                                               example_idx=i,
                                               subdataset=subdataset,
                                               choice_keys=list(choice.keys()))
                                    content = ""
                                
                                finish_reason = choice.get("finish_reason", "unknown")
                                response_text = content or ""
                                f.write(f"{response_text}\n")
                                f.write(f"(finish_reason: {finish_reason})\n")
                            elif "text" in resp:
                                response_text = resp["text"]
                                f.write(f"{response_text}\n")
                            elif "error" in resp:
                                response_text = f"ERROR: {resp['error']}"
                                f.write(f"{response_text}\n")
                            else:
                                response_text = str(resp)
                                f.write(f"{response_text}\n")
                        else:
                            response_text = str(resp)
                            f.write(f"{response_text}\n")

                        if annotation:
                            status = annotation.get("is_correct")
                            if status is True:
                                status_text = "CORRECT"
                            elif status is False:
                                status_text = "INCORRECT"
                            else:
                                status_text = "UNKNOWN"
                            f.write(f"\n[RESULT]:\n")
                            f.write(f"Status: {status_text}\n")
                            if "reference" in annotation and annotation["reference"] is not None:
                                f.write(f"Expected: {annotation['reference']}\n")
                            if "prediction" in annotation and annotation["prediction"] is not None:
                                f.write(f"Prediction: {annotation['prediction']}\n")
                        
                        if analysis_rows is not None:
                            analysis_rows.append(
                                {
                                    "index": idx,
                                    "dataset": annotation.get("dataset") if annotation else subdataset,
                                    "request": _json_safe(req),
                                    "response": _json_safe(resp),
                                    "response_text": response_text,
                                    "finish_reason": finish_reason,
                                    "prediction": annotation.get("prediction") if annotation else None,
                                    "reference": annotation.get("reference") if annotation else None,
                                    "is_correct": annotation.get("is_correct") if annotation else None,
                                }
                            )
                        
                        f.write(f"\n{'-'*60}\n\n")

            if analysis_rows is not None:
                analysis_file = eval_job_dir / f"{eval_name}_analysis.jsonl"
                with open(analysis_file, "w", encoding="utf-8") as analysis_f:
                    for row in analysis_rows:
                        analysis_f.write(json.dumps(row) + "\n")
                logger.info(f"Saved structured analysis to {analysis_file}")
            
            logger.info(f"Saved {len(requests)} interactions to {interactions_file}")
            
        except Exception as e:
            logger.error(f"Error saving eval interactions: {e}")
    
    async def _save_eval_result(self, job_id: str, eval_name: str, result: Dict):
        """Save evaluation result to persistent JSONL storage."""
        job_data = await self.get_eval_status(job_id)
        if not job_data:
            return
        
        model = job_data.get("model", "unknown")
        # Sanitize model name for filesystem
        safe_model = model.replace("/", "--").replace(":", "_")
        
        # Use the job's creation timestamp for consistent directory naming
        # This ensures all evals from the same job go to the same directory
        created_at = job_data.get("created_at", datetime.utcnow().isoformat())
        # Parse ISO timestamp and format for directory name
        job_timestamp = datetime.fromisoformat(created_at).strftime("%Y-%m-%d_%H-%M-%S")
        eval_job_dir = self.data_dir / safe_model / job_timestamp
        eval_job_dir.mkdir(parents=True, exist_ok=True)
        
        # Create/update a "latest" symlink for easy access
        latest_link = self.data_dir / safe_model / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(job_timestamp)
        
        # Save compact summary
        summary_entry = {
            "job_id": job_id,
            "eval_name": eval_name,
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": result.get("metrics", {}),
            "metadata": job_data.get("metadata", {}),
            "details": result.get("details", {})
        }
        
        # Write to eval-specific results file
        eval_results_file = eval_job_dir / f"{eval_name}_results.json"
        with open(eval_results_file, "w") as f:
            json.dump(summary_entry, f, indent=2)
        
        # Also append to the main model results.jsonl for backward compatibility
        model_dir = self.data_dir / safe_model
        model_dir.mkdir(exist_ok=True)
        results_file = model_dir / "results.jsonl"
        with open(results_file, "a") as f:
            f.write(json.dumps(summary_entry) + "\n")
        
        # Save detailed responses for debugging (if there are errors or if requested)
        metrics = result.get("metrics", {})
        if metrics.get("n_errors", 0) > 0 or os.environ.get("SAVE_ALL_EVAL_RESPONSES"):
            # Save each request-response pair in the eval-specific directory
            debug_file = eval_job_dir / f"{eval_name}_debug.jsonl"
            requests = result.get("requests", [])
            responses = result.get("responses", [])
            
            with open(debug_file, "w") as f:
                for i, (req, resp) in enumerate(zip(requests, responses)):
                    debug_entry = {
                        "index": i,
                        "job_id": job_id,
                        "eval_name": eval_name,
                        "request": req,
                        "response": resp
                    }
                    f.write(json.dumps(debug_entry) + "\n")
            
            logger.info(f"Saved debug responses to {debug_file}")
        
        logger.info(f"Saved eval result for {model}/{eval_name} to {results_file}")

        # Update interactions file with correctness annotations and structured analysis
        requests_for_output = result.get("requests", [])
        responses_for_output = result.get("responses", [])
        metadata_for_output = result.get("metadata", {})
        if not isinstance(metadata_for_output, dict):
            metadata_for_output = {}
        annotations: Optional[List[Optional[Dict[str, Any]]]] = None

        if requests_for_output and responses_for_output:
            total_examples = len(requests_for_output)
            per_dataset_details = result.get("details", {})
            dataset_metadata = metadata_for_output.get("dataset_metadata", {}) if isinstance(metadata_for_output, dict) else {}
            logger.info(
                "interaction_annotation_inputs",
                eval_name=eval_name,
                total_examples=total_examples,
                has_dataset_metadata=bool(dataset_metadata),
                metadata_keys=list(metadata_for_output.keys()) if isinstance(metadata_for_output, dict) else None,
                details_datasets=list(per_dataset_details.keys()) if isinstance(per_dataset_details, dict) else None,
            )

            if dataset_metadata and isinstance(per_dataset_details, dict):
                annotations = [None] * total_examples
                logger.info(
                    "initialised_annotations",
                    eval_name=eval_name,
                    total_examples=total_examples,
                )
                for dataset_name, dataset_info in dataset_metadata.items():
                    per_dataset = per_dataset_details.get(dataset_name, {})
                    per_example = per_dataset.get("per_example") if isinstance(per_dataset, dict) else None
                    if not per_example:
                        logger.info(
                            "missing_per_example_records",
                            eval_name=eval_name,
                            dataset=dataset_name,
                            available_keys=list(per_dataset.keys()) if isinstance(per_dataset, dict) else None,
                        )
                        continue
                    start_index = dataset_info.get("start_index", 0)
                    for offset, record in enumerate(per_example):
                        idx = start_index + offset
                        if idx >= total_examples:
                            continue
                        logger.info(
                            "annotation_entry",
                            eval_name=eval_name,
                            dataset=dataset_name,
                            idx=idx,
                        )
                        annotations[idx] = {
                            "dataset": dataset_name,
                            "prediction": record.get("prediction"),
                            "reference": record.get("reference"),
                            "is_correct": record.get("is_correct"),
                        }

        await self._save_eval_interactions(
            job_id,
            eval_name,
            requests_for_output,
            responses_for_output,
            metadata_for_output,
            annotations=annotations,
        )
    
    async def _update_job_status(self, job_id: str, status: EvalStatus, **kwargs):
        """Update the overall job status."""
        job_data = await self.get_eval_status(job_id)
        if job_data:
            job_data["status"] = status
            job_data.update(kwargs)
            await self.redis.setex(
                f"eval_job:{job_id}",
                30 * 24 * 3600,
                json.dumps(job_data)
            )
    
    async def _update_job_data(self, job_id: str, data: Dict):
        """Update job data fields."""
        job_data = await self.get_eval_status(job_id)
        if job_data:
            job_data.update(data)
            await self.redis.setex(
                f"eval_job:{job_id}",
                30 * 24 * 3600,
                json.dumps(job_data)
            )
    
    async def _update_eval_status(self, job_id: str, eval_name: str, status: Dict):
        """Update status for a specific eval within a job."""
        job_data = await self.get_eval_status(job_id)
        if job_data:
            if "eval_status" not in job_data:
                job_data["eval_status"] = {}
            job_data["eval_status"][eval_name] = status
            
            # Check if all evals are complete
            all_complete = all(
                eval_status.get("status") in ["completed", "failed"]
                for eval_status in job_data["eval_status"].values()
            )
            
            if all_complete:
                # Update overall job status
                has_failures = any(
                    eval_status.get("status") == "failed"
                    for eval_status in job_data["eval_status"].values()
                )
                job_data["status"] = EvalStatus.FAILED if has_failures else EvalStatus.COMPLETED
            
            await self.redis.setex(
                f"eval_job:{job_id}",
                30 * 24 * 3600,
                json.dumps(job_data)
            )


# ============ FastAPI Router ============

# Create router
router = APIRouter(prefix="/eval", tags=["evaluation"])


def get_eval_manager(request: Request) -> EvalManager:
    """Get eval manager from app state."""
    if not hasattr(request.app.state, "eval_manager"):
        raise HTTPException(status_code=503, detail="Eval system not initialized")
    return request.app.state.eval_manager


@router.post("/submit")
async def submit_evaluation(
    request_body: Dict,
    request: Request,
    eval_manager: EvalManager = Depends(get_eval_manager)
):
    """Submit a model for evaluation.
    Optional overrides:
      - evals: list[str] of eval module names to run
      - prepare_args: dict of adapter-specific prepare flags (e.g., {"max_prompts": 50})
    """
    model = request_body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Model name required")
    # Parse optional overrides
    evals = request_body.get("evals") or []
    if evals and not isinstance(evals, list):
        raise HTTPException(status_code=400, detail="'evals' must be a list of strings")
    try:
        evals = [str(x) for x in (evals or [])]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid 'evals' list")
    prepare_args = request_body.get("prepare_args") or {}
    if prepare_args and not isinstance(prepare_args, dict):
        raise HTTPException(status_code=400, detail="'prepare_args' must be an object")

    # Submit evaluation job
    job_id = await eval_manager.submit_eval(model, evals=evals, prepare_args=prepare_args)

    return {
        "job_id": job_id,
        "model": model,
        "status": "submitted",
        "message": f"Evaluation job {job_id} submitted successfully",
        "requested_evals": evals,
        "prepare_args": prepare_args,
    }


def _sanitize_for_json(obj):
    """Recursively replace non-finite floats (NaN/Inf) with None to satisfy strict JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj

@router.get("/{job_id}")
async def get_eval_status(
    job_id: str,
    eval_manager: EvalManager = Depends(get_eval_manager)
):
    """Get status of an evaluation job."""
    status = await eval_manager.get_eval_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Evaluation job {job_id} not found")
    # Sanitize NaN/Inf values that break strict JSON responses
    return _sanitize_for_json(status)


@router.post("/rescore")
async def rescore_evaluation(
    request_body: Dict,
    eval_manager: EvalManager = Depends(get_eval_manager)
):
    """Re-score eval results from stored responses (no new inference).
    Body options:
      - { job_id: str, eval_name?: str }
      - { model: str, eval_name?: str }  # uses latest eval job for that model
    """
    job_id = (request_body.get("job_id") or "").strip() or None
    model_filter = (request_body.get("model") or "").strip() or None

    # If job_id not provided, find latest eval_job for model
    if not job_id:
        if not model_filter:
            raise HTTPException(status_code=400, detail="Provide either job_id or model")
        # Scan eval jobs and pick latest by created_at
        cursor = 0
        latest_job_id: str | None = None
        latest_ts = None
        while True:
            cursor, keys = await eval_manager.redis.scan(cursor, match="eval_job:*", count=200)
            for k in keys:
                try:
                    raw = await eval_manager.redis.get(k)
                    if not raw:
                        continue
                    jd = json.loads(raw)
                    if jd.get("model") != model_filter:
                        continue
                    ts_raw = jd.get("created_at") or jd.get("created")
                    ts_val = None
                    if isinstance(ts_raw, str):
                        try:
                            ts_val = datetime.fromisoformat(ts_raw)
                        except Exception:
                            ts_val = None
                    # Fallback: if missing/invalid, keep but don't override a valid latest
                    if latest_ts is None and ts_val is None and latest_job_id is None:
                        latest_job_id = k.split(":", 1)[1]
                    elif ts_val is not None and (latest_ts is None or ts_val > latest_ts):
                        latest_ts = ts_val
                        latest_job_id = k.split(":", 1)[1]
                except Exception:
                    continue
            if cursor == 0:
                break
        if not latest_job_id:
            raise HTTPException(status_code=404, detail=f"No eval job found for model '{model_filter}'")
        job_id = latest_job_id

    # Load job data
    job_key = f"eval_job:{job_id}"
    job_data_raw = await eval_manager.redis.get(job_key)
    if not job_data_raw:
        raise HTTPException(status_code=404, detail=f"Evaluation job {job_id} not found")
    try:
        job_data = json.loads(job_data_raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Corrupt eval job record")

    model = job_data.get("model")
    eval_groups = job_data.get("eval_groups", {})
    eval_statuses = job_data.get("eval_status", {})

    # Decide which evals to rescore
    only_eval = request_body.get("eval_name")
    eval_names = [only_eval] if only_eval else list(eval_groups.keys() or eval_statuses.keys())
    if not eval_names:
        raise HTTPException(status_code=400, detail="No evals to rescore for this job")

    results: Dict[str, Dict] = {}
    errors: Dict[str, str] = {}

    for eval_name in eval_names:
        cids = eval_groups.get(eval_name) or (eval_statuses.get(eval_name, {}).get("completion_ids") or [])
        if not cids:
            errors[eval_name] = "No completion IDs found"
            continue
        # Gather requests/responses
        request_response_pairs: Dict[str, tuple] = {}
        for comp_id in cids:
            jobh = await eval_manager.redis.hgetall(f"job:{comp_id}")
            if not jobh:
                continue
            req = redis_job_to_request(jobh)
            if jobh.get("status") == "completed":
                try:
                    resp = json.loads(jobh.get("result", "{}"))
                except Exception:
                    resp = {"error": "bad_result_json"}
            else:
                resp = {"error": f"status={jobh.get('status')}"}
            request_response_pairs[comp_id] = (req, resp)
        # Preserve original order
        requests = []
        responses = []
        for comp_id in cids:
            if comp_id in request_response_pairs:
                r, s = request_response_pairs[comp_id]
                requests.append(r)
                responses.append(s)

        # Load eval metadata
        meta_key = f"eval_metadata:{job_id}:{eval_name}"
        meta_raw = await eval_manager.redis.get(meta_key)
        metadata = json.loads(meta_raw) if meta_raw else {}

        score_input = {"requests": requests, "responses": responses, "metadata": metadata}

        # Call scoring in the eval's venv
        cmd_args = [
            f"{eval_manager.eval_repo_path}/run_eval.py",
            "score", eval_name
        ]
        rc, stdout, stderr = await run_eval_command(
            eval_name, eval_manager.eval_repo_path, cmd_args,
            stdin_data=json.dumps(score_input)
        )
        if rc != 0:
            errors[eval_name] = stderr.decode(errors='replace')[:500]
            continue
        try:
            score_result = json.loads(stdout.decode())
        except Exception as e:
            errors[eval_name] = f"parse_error: {e}"
            continue

        # Save to disk and update eval status
        await eval_manager._save_eval_result(job_id, eval_name, score_result)
        es = eval_statuses.get(eval_name, {})
        es.update({
            "status": "completed",
            "results": score_result.get("metrics", {}),
            "details": score_result.get("details", {}),
        })
        job_data.setdefault("eval_status", {})[eval_name] = es
        results[eval_name] = score_result.get("metrics", {})

    # Persist updated job data
    await eval_manager.redis.setex(job_key, 30 * 24 * 3600, json.dumps(job_data))

    return {
        "job_id": job_id,
        "model": model,
        "rescored": list(results.keys()),
        "errors": errors,
        "metrics": results,
    }


@router.get("/")
async def list_evaluations(
    model: Optional[str] = None,
    limit: int = 100,
    eval_manager: EvalManager = Depends(get_eval_manager)
):
    """List evaluation jobs, optionally filtered by model."""
    # TODO: Implement listing from Redis and/or JSONL files
    # For now, return empty list
    return {
        "evaluations": [],
        "total": 0,
        "model_filter": model,
        "limit": limit
    }
