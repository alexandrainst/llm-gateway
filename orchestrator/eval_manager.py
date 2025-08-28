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
from typing import Dict, List, Optional, Any
from enum import Enum
import structlog
from fastapi import APIRouter, HTTPException, Depends, Request

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
        # Create venv if it doesn't exist
        if not venv_path.exists():
            logger.info(f"Creating venv for {eval_name} at {venv_path}")
            logger.info(f"Working directory: {eval_dir}")
            # Create venv with uv (use full path)
            try:
                venv_result = await asyncio.create_subprocess_exec(
                    "/usr/local/bin/uv", "venv", str(venv_path),
                    cwd=str(eval_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = await venv_result.communicate()
                if venv_result.returncode != 0:
                    logger.error(f"Failed to create venv: {stderr.decode()}")
                else:
                    logger.info(f"Created venv successfully")
            except Exception as e:
                logger.error(f"Exception creating venv: {e}")
                raise
            
            # Install dependencies
            if (eval_dir / "pyproject.toml").exists():
                logger.info(f"Installing dependencies for {eval_name} from pyproject.toml...")
                install_result = await asyncio.create_subprocess_exec(
                    "/usr/local/bin/uv", "pip", "install", "-e", ".",
                    cwd=str(eval_dir),
                    env={**os.environ, "VIRTUAL_ENV": str(venv_path)},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await install_result.communicate()
            elif (eval_dir / "requirements.txt").exists():
                logger.info(f"Installing dependencies for {eval_name} from requirements.txt...")
                install_result = await asyncio.create_subprocess_exec(
                    "/usr/local/bin/uv", "pip", "install", "-r", "requirements.txt",
                    cwd=str(eval_dir),
                    env={**os.environ, "VIRTUAL_ENV": str(venv_path)},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await install_result.communicate()
        
        # Use the venv's Python
        python_cmd = str(venv_path / "bin" / "python")
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
    
    async def submit_eval(self, model: str, user_token: str) -> str:
        """
        Submit a model for evaluation.
        
        Args:
            model: Model name/path to evaluate
            user_token: API token for authentication
            
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
            "eval_groups": {}
        }
        
        # Store in Redis with 30 day TTL
        await self.redis.setex(
            f"eval_job:{job_id}",
            30 * 24 * 3600,
            json.dumps(job_data)
        )
        
        # Start async evaluation task
        task = asyncio.create_task(
            self._run_eval(job_id, model, user_token)
        )
        self.active_evals[job_id] = task
        
        return job_id
    
    async def get_eval_status(self, job_id: str) -> Optional[Dict]:
        """Get the current status of an evaluation job."""
        data = await self.redis.get(f"eval_job:{job_id}")
        if data:
            return json.loads(data)
        return None
    
    async def _run_eval(self, job_id: str, model: str, user_token: str):
        """Run the full evaluation workflow."""
        try:
            # Update status to submitting
            await self._update_job_status(job_id, EvalStatus.SUBMITTING)
            
            # Phase 1: Submit requests to eval repo
            submission_result = await self._submit_requests(job_id, model, user_token)
            
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
                    self._monitor_eval_completions(job_id, eval_name, completion_ids, user_token)
                )
            
        except Exception as e:
            logger.error(f"Eval {job_id} failed: {e}")
            await self._update_job_status(job_id, EvalStatus.FAILED, error=str(e))
        finally:
            # Clean up from active evals
            self.active_evals.pop(job_id, None)
    
    def _check_is_base_model(self, model: str) -> bool:
        """
        Check if a model is a base model by looking for chat_template in tokenizer_config.json.
        """
        # Convert model ID to HuggingFace cache path format
        model_path_pattern = model.replace("/", "--")
        cache_dir = Path("/host_hf_cache/hub")
        
        # Look for the model directory
        model_dirs = list(cache_dir.glob(f"models--{model_path_pattern}"))
        if not model_dirs:
            logger.warning(f"Model directory not found for {model}")
            return False
        
        # Check snapshots for tokenizer_config.json
        snapshots_dir = model_dirs[0] / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                tokenizer_config = snapshot / "tokenizer_config.json"
                if tokenizer_config.exists():
                    try:
                        with open(tokenizer_config) as f:
                            config = json.load(f)
                            # If chat_template is None or missing, it's a base model
                            has_chat_template = config.get("chat_template") is not None
                            logger.info(f"Model {model} has_chat_template: {has_chat_template}")
                            return not has_chat_template
                    except Exception as e:
                        logger.warning(f"Error reading tokenizer config: {e}")
        
        # Default to assuming it's an instruction model
        logger.info(f"Could not determine if {model} is base model, assuming instruction model")
        return False
    
    async def _submit_requests(self, job_id: str, model: str, user_token: str) -> Optional[Dict]:
        """
        Phase 1: Get requests from eval repo and submit them as batch jobs.
        
        Returns dict with completion_ids, eval_groups, and metadata.
        """
        # Check if model is a base model
        is_base_model = self._check_is_base_model(model)
        
        # Ask eval repo which evals to run, passing base model flag
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
                eval_names = ["example_eval"]  # Fallback
            else:
                list_result = json.loads(stdout.decode())
                eval_names = list_result.get("evals", ["example_eval"])
        except Exception as e:
            logger.error(f"Error getting eval list: {e}")
            eval_names = ["example_eval"]  # Fallback
        
        logger.info(f"Running evals: {eval_names}")
        
        all_completion_ids = []
        eval_groups = {}
        metadata = {
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
            "eval_suite": "standard",
        }
        
        for eval_name in eval_names:
            try:
                # Phase 1a: Get requests from eval
                cmd_args = [
                    f"{self.eval_repo_path}/run_eval.py",
                    "prepare", eval_name, model
                ]
                if is_base_model:
                    cmd_args.append("--base-model")
                
                returncode, stdout, stderr = await run_eval_command(
                    eval_name, self.eval_repo_path, cmd_args
                )
                
                if returncode != 0:
                    logger.error(f"prepare requests failed for {eval_name}: {stderr.decode()}")
                    continue
                
                prepare_result = json.loads(stdout.decode())
                requests = prepare_result.get("requests", [])
                eval_metadata = prepare_result.get("metadata", {})
                
                # Phase 1b: Submit requests as batch jobs (directly to Redis)
                completion_ids = []
                for request in requests:
                    # Generate completion ID
                    comp_id = f"comp_{uuid.uuid4().hex[:12]}"
                    
                    # Ensure model is set in request
                    request["model"] = model
                    
                    # Convert to Redis job format using helper
                    job_data = request_to_redis_job(
                        request, 
                        model, 
                        comp_id,
                        eval_name=eval_name  # Extra metadata for eval tracking
                    )
                    
                    # Store job in Redis with 7 day TTL
                    await self.redis.hset(f"job:{comp_id}", mapping=job_data)
                    await self.redis.expire(f"job:{comp_id}", 7 * 24 * 3600)
                    
                    # Add to batch queue with current timestamp as score
                    await self.redis.zadd("z:batchjobs", {comp_id: datetime.utcnow().timestamp()})
                    
                    completion_ids.append(comp_id)
                    
                    logger.debug(f"Submitted batch job {comp_id} for eval {eval_name}")
                
                all_completion_ids.extend(completion_ids)
                eval_groups[eval_name] = completion_ids
                
                # Store eval metadata for scoring later
                await self.redis.setex(
                    f"eval_metadata:{job_id}:{eval_name}",
                    7 * 24 * 3600,
                    json.dumps(eval_metadata)
                )
                
                logger.info(f"Submitted {len(completion_ids)} requests for {eval_name}")
                
            except Exception as e:
                import traceback
                logger.error(f"Error processing eval {eval_name}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        metadata["total_requests"] = len(all_completion_ids)
        
        return {
            "completion_ids": all_completion_ids,
            "eval_groups": eval_groups,
            "metadata": metadata
        }
    
    async def _monitor_eval_completions(self, job_id: str, eval_name: str, 
                                       completion_ids: List[str], user_token: str):
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
            
            score_result = await self._score_results(job_id, eval_name, completion_ids, user_token)
            
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
                            completion_ids: List[str], user_token: str) -> Optional[Dict]:
        """
        Phase 2: Gather results and call eval repo to score them.
        
        Returns dict with metrics and details.
        """
        try:
            # Gather all requests and responses using completion IDs as keys
            # This ensures request[i] matches response[i] regardless of completion order
            request_response_pairs = {}
            
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
                    response = {"error": job_data.get("error", "Unknown error")}
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
            return score_result
            
        except Exception as e:
            logger.error(f"Error scoring results for {eval_name}: {e}")
            return None
    
    async def _save_eval_interactions(self, job_id: str, eval_name: str, 
                                     requests: List[Dict], responses: List[Dict], metadata: Dict):
        """Save requests and responses for debugging and analysis."""
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
                                f.write(f"{content}\n")
                                f.write(f"(finish_reason: {finish_reason})\n")
                            elif "text" in resp:
                                f.write(f"{resp['text']}\n")
                            elif "error" in resp:
                                f.write(f"ERROR: {resp['error']}\n")
                            else:
                                f.write(f"{resp}\n")
                        else:
                            f.write(f"{resp}\n")
                        
                        f.write(f"\n{'-'*60}\n\n")
            
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
    """Submit a model for evaluation."""
    model = request_body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Model name required")
    
    # Get user token from request or use default
    user_token = request_body.get("api_key") or os.getenv("API_KEY", "changeme")
    
    # Submit evaluation job
    job_id = await eval_manager.submit_eval(model, user_token)
    
    return {
        "job_id": job_id,
        "model": model,
        "status": "submitted",
        "message": f"Evaluation job {job_id} submitted successfully"
    }


@router.get("/{job_id}")
async def get_eval_status(
    job_id: str,
    eval_manager: EvalManager = Depends(get_eval_manager)
):
    """Get status of an evaluation job."""
    status = await eval_manager.get_eval_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Evaluation job {job_id} not found")
    
    return status


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