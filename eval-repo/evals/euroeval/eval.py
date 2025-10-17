"""
EuroEval batch processing adapter.

This module provides two-phase batch processing for EuroEval benchmarks:
- Phase 1: Collect all model API requests from EuroEval
- Phase 2: Score responses using EuroEval's evaluation logic
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
try:
    import tomllib
except Exception:  # pragma: no cover
    try:
        import tomli as tomllib
    except Exception:
        tomllib = None

# Disable tqdm progress bars in non-interactive environments
os.environ["TQDM_DISABLE"] = "1"

# Debug log location (default to working tree so host can inspect it)
# Prefer an explicit path; otherwise write into the eval directory (which is bind-mounted)
DEFAULT_DEBUG_BASENAME = "euroeval_debug.log"
_default_target = Path.cwd() / DEFAULT_DEBUG_BASENAME
# If cwd is read-only (e.g. /app inside container), fall back to /tmp
if not os.access(_default_target.parent, os.W_OK):
    _default_target = Path("/tmp") / DEFAULT_DEBUG_BASENAME
DEBUG_LOG_PATH = Path(os.getenv("EUROEVAL_DEBUG_LOG", _default_target))


def _write_debug(message: str) -> None:
    """Append a line to the EuroEval debug log, creating directories if needed."""
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(message)
    except OSError:
        # As a last resort, fall back to /tmp
        fallback = Path("/tmp") / DEFAULT_DEBUG_BASENAME
        fallback.parent.mkdir(parents=True, exist_ok=True)
        with fallback.open("a", encoding="utf-8") as log:
            log.write(message)

# Add EuroEval to path
euroeval_path = Path(__file__).parent.parent.parent.parent / "EuroEval" / "src"
sys.path.insert(0, str(euroeval_path))

from euroeval import Benchmarker
from euroeval.enums import TaskGroup
from euroeval.generation import register_generation_observer

# Load configuration
CONFIG_FILE = Path(__file__).parent / "eval.toml"
if tomllib is not None and CONFIG_FILE.exists():
    with open(CONFIG_FILE, "rb") as f:
        CONFIG = tomllib.load(f)
else:
    CONFIG = {}


def prepare_requests(model_name: str, **kwargs) -> Dict[str, Any]:
    """
    Phase 1: Collect all requests that EuroEval would make to the model API.
    
    Args:
        model_name: Name of the model to evaluate
        **kwargs: Additional configuration (datasets, etc.)
    
    Returns:
        Dict with 'requests' list and 'metadata' dict
    """
    # Get configuration from config file
    euroeval_config = CONFIG.get("euroeval", {})
    
    # Get datasets from kwargs, then config, then defaults
    datasets = kwargs.get("datasets", euroeval_config.get("datasets", []))
    if isinstance(datasets, str):
        datasets = [datasets]
    
    # Get test split settings from config
    evaluate_test_split = kwargs.get("evaluate_test_split", euroeval_config.get("evaluate_test_split", True))
    
    # Check if this is a base model
    is_base_model = kwargs.get("is_base_model", False)
    
    # Determine few-shot setting based on model type
    # Base models: always use few-shot (True)
    # Instruction models: always use zero-shot (False)
    # Allow override via kwargs if explicitly provided
    if "few_shot" in kwargs:
        few_shot = kwargs["few_shot"]
    else:
        few_shot = True if is_base_model else False
    
    # Debug logging
    _write_debug("\n[PREPARE_REQUESTS] Starting phase 1\n")
    _write_debug(f"  - Model: {model_name}\n")
    _write_debug(f"  - Datasets: {datasets}\n")
    _write_debug(f"  - Is base model: {is_base_model}\n")
    _write_debug(
        f"  - Few-shot: {few_shot} (base={is_base_model}, instruct={not is_base_model})\n"
    )
    
    # Collect requests from all datasets
    all_requests = []
    dataset_metadata = {}
    
    for dataset in datasets:
        _write_debug(f"\n[PREPARE_REQUESTS] Processing dataset: {dataset}\n")
        
        # Create temporary file for request collection for this dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            requests_file = f.name
        
        try:
            # Set gateway environment variables for phase 1
            os.environ["EUROEVAL_GATEWAY_MODE"] = "phase1"
            os.environ["EUROEVAL_GATEWAY_REQUESTS_FILE"] = requests_file
            
            # Debug: log that we're setting the environment
            _write_debug("\n[PREPARE_REQUESTS] Set EUROEVAL_GATEWAY_MODE=phase1\n")
            _write_debug(
                f"[PREPARE_REQUESTS] Set EUROEVAL_GATEWAY_REQUESTS_FILE={requests_file}\n"
            )
            
            # Set base model flag if provided
            if is_base_model:
                os.environ["EUROEVAL_IS_BASE_MODEL"] = "true"
            
            # Configure benchmarker to collect requests
            # Don't pass api_base or api_key - this will prevent LiteLLM from being selected
            benchmarker = Benchmarker(
                dataset=dataset,
                save_results=False,
                cache_dir=kwargs.get("cache_dir", "/tmp/euroeval_cache"),
                progress_bar=False,
                verbose=False,
                num_iterations=1,
                force=True,
                raise_errors=True,
                is_base_model=kwargs.get("is_base_model", False),
                evaluate_test_split=evaluate_test_split,
                few_shot=few_shot
            )
            
            # Run benchmarker to collect requests
            _write_debug(
                f"[PREPARE_REQUESTS] Running benchmarker.benchmark() for {dataset}...\n"
            )
            
            try:
                # Pass the actual model name - our gateway module will use it
                results = benchmarker.benchmark(model=[model_name])
                _write_debug(
                    f"[PREPARE_REQUESTS] Benchmark returned {len(results) if results else 0} results for {dataset}\n"
                )
            except Exception as e:
                # Log the full exception with traceback, then surface the failure
                import traceback
                _write_debug(
                    f"[PREPARE_REQUESTS] Benchmark failed for {dataset} with error: {str(e)}\n"
                )
                _write_debug(
                    f"[PREPARE_REQUESTS] Traceback:\n{traceback.format_exc()}\n"
                )
                raise
            
            # Read collected requests for this dataset
            dataset_requests = []
            if Path(requests_file).exists():
                _write_debug(
                    f"[PREPARE_REQUESTS] Reading requests from {requests_file} for {dataset}\n"
                )
                
                with open(requests_file, 'r') as f:
                    for line in f:
                        request_data = json.loads(line)
                        
                        # Check if this is a base model request (has "prompt" field) or chat model (has "messages")
                        if "messages" in request_data:
                            # Chat completion request
                            formatted_request = {
                                "messages": request_data["messages"],
                                "max_tokens": request_data["generation_kwargs"].get("max_tokens", 512),
                                "temperature": request_data["generation_kwargs"].get("temperature", 0.0),
                                "_dataset": dataset,  # Track which dataset this request is from
                            }
                        elif "prompt" in request_data:
                            # Text completion request (base model)
                            formatted_request = {
                                "prompt": request_data["prompt"],
                                "max_tokens": request_data["generation_kwargs"].get("max_tokens", 512),
                                "temperature": request_data["generation_kwargs"].get("temperature", 0.0),
                                "_dataset": dataset,  # Track which dataset this request is from
                            }
                        else:
                            # Fallback - assume chat if we have messages in some other format
                            formatted_request = {
                                "messages": request_data.get("messages", []),
                                "max_tokens": request_data["generation_kwargs"].get("max_tokens", 512),
                                "temperature": request_data["generation_kwargs"].get("temperature", 0.0),
                                "_dataset": dataset,  # Track which dataset this request is from
                            }
                        
                        # Add other generation kwargs if present (include logprobs)
                        for key in [
                            "top_p",
                            "frequency_penalty",
                            "presence_penalty",
                            "stop",
                            "logprobs",
                            "top_logprobs",
                        ]:
                            if key in request_data["generation_kwargs"]:
                                formatted_request[key] = request_data["generation_kwargs"][key]
                        
                        dataset_requests.append(formatted_request)
            
            _write_debug(
                f"[PREPARE_REQUESTS] Collected {len(dataset_requests)} requests from {dataset}\n"
            )
            
            # Track metadata for this dataset
            dataset_metadata[dataset] = {
                "num_requests": len(dataset_requests),
                "start_index": len(all_requests),
                "end_index": len(all_requests) + len(dataset_requests)
            }
            
            # Add this dataset's requests to the overall list
            all_requests.extend(dataset_requests)
            
        finally:
            # Clean up temp file for this dataset
            if Path(requests_file).exists():
                Path(requests_file).unlink()
    
    # Clean up environment
    if "EUROEVAL_GATEWAY_MODE" in os.environ:
        del os.environ["EUROEVAL_GATEWAY_MODE"]
    if "EUROEVAL_GATEWAY_REQUESTS_FILE" in os.environ:
        del os.environ["EUROEVAL_GATEWAY_REQUESTS_FILE"]
    if "EUROEVAL_IS_BASE_MODEL" in os.environ:
        del os.environ["EUROEVAL_IS_BASE_MODEL"]
    
    _write_debug(
        f"\n[PREPARE_REQUESTS] Total collected: {len(all_requests)} requests from {len(datasets)} datasets\n"
    )
    
    # Return in protocol format
    return {
        "requests": all_requests,
        "metadata": {
            "datasets": datasets,
            "dataset_metadata": dataset_metadata,
            "model_name": model_name,
            "is_base_model": is_base_model,
            "total_requests": len(all_requests)
        }
    }


def score_results(requests: List[Dict], responses: List[Dict], metadata: Dict) -> Dict[str, Any]:
    """
    Phase 2: Score model responses using EuroEval's evaluation logic.
    
    Args:
        requests: List of request dicts
        responses: List of response dicts (parallel to requests)
        metadata: Metadata from phase 1
    
    Returns:
        Dict with 'metrics' and optional 'details'
    """
    # Get configuration from config file
    euroeval_config = CONFIG.get("euroeval", {})
    evaluate_test_split = euroeval_config.get("evaluate_test_split", True)
    
    # Get is_base_model from metadata
    is_base_model = metadata.get("is_base_model", False)
    
    # Use same logic as prepare_requests for few-shot setting
    # Base models: always use few-shot (True)
    # Instruction models: always use zero-shot (False)
    few_shot = True if is_base_model else False
    # Handle both single dataset (old format) and multiple datasets (new format)
    if "datasets" in metadata:
        datasets = metadata["datasets"]
        dataset_metadata = metadata.get("dataset_metadata", {})
    else:
        # Backward compatibility with single dataset
        datasets = [metadata.get("dataset", "angry-tweets")]
        dataset_metadata = {
            datasets[0]: {
                "start_index": 0,
                "end_index": len(requests),
                "num_requests": len(requests)
            }
        }

    if not dataset_metadata:
        # Fallback: reconstruct index ranges from request order
        reconstructed = {}
        for idx, request in enumerate(requests):
            dataset_name = request.get("_dataset", "default")
            entry = reconstructed.setdefault(
                dataset_name,
                {"start_index": idx, "end_index": idx, "num_requests": 0},
            )
            entry["num_requests"] += 1
            entry["end_index"] = idx + 1
        if reconstructed:
            dataset_metadata = reconstructed
            if not datasets or datasets == ["default"]:
                datasets = list(reconstructed.keys())

    if isinstance(metadata, dict):
        metadata.setdefault("datasets", datasets)
        metadata["dataset_metadata"] = dataset_metadata

    # Debug log what we received
    with open("/tmp/euroeval_debug.log", "a") as log:
        log.write(f"\n[SCORE_RESULTS] Called with:\n")
        log.write(f"  - {len(requests)} requests\n")
        log.write(f"  - {len(responses)} responses\n")
        log.write(f"  - datasets: {datasets}\n")
        log.write(f"  - dataset_metadata: {dataset_metadata}\n")
    
    # Process each dataset separately
    all_metrics = {}
    all_details = {}
    
    def _to_serialisable(value: Any) -> Any:
        """Convert EuroEval outputs to JSON-friendly structures."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        # Handle numpy scalar types
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, list):
            return [_to_serialisable(v) for v in value]
        if isinstance(value, tuple):
            return [_to_serialisable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _to_serialisable(v) for k, v in value.items()}
        return str(value)

    def _normalize_str(value: str) -> str:
        return value.strip().lower()

    def _list_strings_equal(first: List[str], second: List[str]) -> Optional[bool]:
        if len(first) != len(second):
            return False
        return all(_normalize_str(a) == _normalize_str(b) for a, b in zip(first, second))

    def _build_label_maps(dataset_config):
        label_to_prompt = {}
        prompt_to_label = {}
        mapping = getattr(dataset_config, "prompt_label_mapping", None)
        if isinstance(mapping, dict):
            for label, prompt in mapping.items():
                if isinstance(label, str) and isinstance(prompt, str):
                    label_norm = _normalize_str(label)
                    prompt_norm = _normalize_str(prompt)
                    label_to_prompt[label_norm] = prompt
                    prompt_to_label[prompt_norm] = label_norm
        return label_to_prompt, prompt_to_label

    def _canonicalize_value(value: Any, prompt_to_label: Dict[str, str]) -> Any:
        if isinstance(value, str):
            norm = _normalize_str(value)
            return prompt_to_label.get(norm, norm)
        if isinstance(value, list):
            converted = [_canonicalize_value(item, prompt_to_label) for item in value]
            return converted
        if isinstance(value, tuple):
            return tuple(_canonicalize_value(item, prompt_to_label) for item in value)
        return value

    def _format_label_for_display(value: Any, canonical: Any, label_to_prompt: Dict[str, str]) -> Any:
        if isinstance(canonical, str):
            prompt = label_to_prompt.get(_normalize_str(canonical))
            if prompt:
                return prompt
        return value

    def _qa_collect_texts(prediction: Any, reference: Any) -> tuple[List[str], List[str]]:
        pred_texts: List[str] = []
        if isinstance(prediction, dict):
            if isinstance(prediction.get("prediction_text"), str):
                pred_texts.append(prediction["prediction_text"])
            elif isinstance(prediction.get("text"), str):
                pred_texts.append(prediction["text"])
        elif isinstance(prediction, str):
            pred_texts.append(prediction)

        ref_texts: List[str] = []
        if isinstance(reference, dict):
            answers = reference.get("answers")
            if isinstance(answers, dict):
                texts = answers.get("text")
                if isinstance(texts, list):
                    ref_texts.extend([t for t in texts if isinstance(t, str)])
            if isinstance(reference.get("text"), list):
                ref_texts.extend([t for t in reference["text"] if isinstance(t, str)])
        elif isinstance(reference, list):
            ref_texts.extend([t for t in reference if isinstance(t, str)])
        elif isinstance(reference, str):
            ref_texts.append(reference)

        return pred_texts, ref_texts

    def _qa_is_correct(prediction: Any, reference: Any) -> Optional[bool]:
        pred_texts, ref_texts = _qa_collect_texts(prediction, reference)
        if not pred_texts or not ref_texts:
            return None
        best_em = False
        for pred_text in pred_texts:
            norm_pred = _squad_normalize(pred_text)
            for ref_text in ref_texts:
                norm_ref = _squad_normalize(ref_text)
                if not norm_pred and not norm_ref:
                    continue
                if norm_pred == norm_ref:
                    return True
                # No partial credit here; fall back to F1 threshold later if needed
        return False

    def _squad_normalize(text: str) -> str:
        if not isinstance(text, str):
            return ""

        def remove_articles(s: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", s)

        def white_space_fix(s: str) -> str:
            return " ".join(s.split())

        def remove_punc(s: str) -> str:
            return re.sub(r"[^a-z0-9\s]", "", s)

        def lower(s: str) -> str:
            return s.lower()

        return white_space_fix(remove_articles(remove_punc(lower(text))))

    def _squad_f1(prediction: str, ground_truth: str) -> float:
        pred_tokens = _squad_normalize(prediction).split()
        truth_tokens = _squad_normalize(ground_truth).split()
        if not pred_tokens and not truth_tokens:
            return 1.0
        if not pred_tokens or not truth_tokens:
            return 0.0

        common = {}
        for token in pred_tokens:
            common[token] = common.get(token, 0) + 1
        overlap = 0
        for token in truth_tokens:
            if common.get(token, 0) > 0:
                overlap += 1
                common[token] -= 1

        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_tokens)
        recall = overlap / len(truth_tokens)
        return 2 * precision * recall / (precision + recall)

    def _qa_best_scores(prediction: Any, reference: Any) -> Optional[Dict[str, float]]:
        pred_texts, ref_texts = _qa_collect_texts(prediction, reference)
        if not pred_texts or not ref_texts:
            return None

        best_em = 0.0
        best_f1 = 0.0
        for pred_text in pred_texts:
            for ref_text in ref_texts:
                em = float(_squad_normalize(pred_text) == _squad_normalize(ref_text))
                f1 = _squad_f1(pred_text, ref_text)
                if em > best_em or (em == best_em and f1 > best_f1):
                    best_em = em
                    best_f1 = f1
        return {"exact_match": best_em, "f1": best_f1}

    def _compute_is_correct(dataset_config, prediction: Any, reference: Any) -> Optional[bool]:
        """Best-effort correctness check on canonicalised values."""
        if prediction is None or reference is None:
            return None

        group = dataset_config.task.task_group

        if group == TaskGroup.QUESTION_ANSWERING:
            scores = _qa_best_scores(prediction, reference)
            if not scores:
                return None
            return scores["exact_match"] >= 0.5

        # Normalise simple scalar comparisons
        if isinstance(prediction, str) and isinstance(reference, str):
            return _normalize_str(prediction) == _normalize_str(reference)

        if isinstance(prediction, (int, float, bool)) and isinstance(reference, (int, float, bool)):
            return prediction == reference

        if isinstance(prediction, list) and isinstance(reference, list):
            if all(isinstance(item, str) for item in prediction) and all(isinstance(item, str) for item in reference):
                return _list_strings_equal(prediction, reference)
            return prediction == reference

        if isinstance(prediction, dict) or isinstance(reference, dict):
            return None

        if prediction == reference:
            return True

        # For free-form tasks we can't easily determine correctness.
        if group in {TaskGroup.TEXT_TO_TEXT, TaskGroup.QUESTION_ANSWERING}:
            return None

        return False

    for dataset in datasets:
        dataset_info = dataset_metadata.get(dataset, {})
        start_idx = dataset_info.get("start_index", 0)
        end_idx = dataset_info.get("end_index", len(responses))
        
        # Get responses for this dataset
        dataset_responses = responses[start_idx:end_idx]
        dataset_requests = requests[start_idx:end_idx]
        
        with open("/tmp/euroeval_debug.log", "a") as log:
            log.write(f"\n[SCORE_RESULTS] Processing dataset: {dataset}\n")
            log.write(f"  - Start index: {start_idx}\n")
            log.write(f"  - End index: {end_idx}\n")
            log.write(f"  - Num responses: {len(dataset_responses)}\n")
        
        if len(dataset_responses) == 0:
            with open("/tmp/euroeval_debug.log", "a") as log:
                log.write(f"[SCORE_RESULTS] No responses for dataset {dataset}, skipping\n")
            continue
        
        # Create temporary response file for this dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            responses_file = f.name
            
            # Write responses in format expected by modified EuroEval
            for idx, response in enumerate(dataset_responses):
                # Extract content from response
                if isinstance(response, dict):
                    if "choices" in response and len(response["choices"]) > 0:
                        choice = response["choices"][0]
                        # Handle both chat (message.content) and completion (text) formats
                        if "message" in choice:
                            content = choice.get("message", {}).get("content", "")
                        elif "text" in choice:
                            content = choice.get("text", "")
                        else:
                            content = ""
                    elif "text" in response:
                        # Fallback for non-standard format
                        content = response["text"]
                    else:
                        content = str(response)
                else:
                    content = str(response)
                
                # Ensure content is never None
                if content is None:
                    content = ""
                
                # Write in EuroEval gateway format
                response_data = {
                    "request_id": f"{dataset}_{idx}",
                    "content": content,
                    "finish_reason": "stop",
                }
                
                # Add logprobs if available
                if isinstance(response, dict) and "choices" in response:
                    choice = response["choices"][0] if response["choices"] else {}
                    if "logprobs" in choice:
                        response_data["logprobs"] = choice["logprobs"]
                
                f.write(json.dumps(response_data) + "\n")
        
        try:
            # Set gateway environment variables for phase 2
            os.environ["EUROEVAL_GATEWAY_MODE"] = "phase2"
            os.environ["EUROEVAL_GATEWAY_RESPONSES_FILE"] = responses_file
            
            # Set base model flag if provided in metadata
            if metadata.get("is_base_model", False):
                os.environ["EUROEVAL_IS_BASE_MODEL"] = "true"
            
            # Debug logging to file
            with open("/tmp/euroeval_debug.log", "a") as log:
                log.write(f"\n[DEBUG] Phase 2 for dataset: {dataset}\n")
                log.write(f"[DEBUG] Responses file: {responses_file}\n")
                log.write(f"[DEBUG] Number of responses: {len(dataset_responses)}\n")
            
            # Configure benchmarker for scoring
            # Don't pass api_base or api_key - this will prevent LiteLLM from being selected
            benchmarker = Benchmarker(
                dataset=dataset,
                save_results=False,
                cache_dir="/tmp/euroeval_cache",
                progress_bar=False,
                verbose=False,
                num_iterations=1,
                force=True,
                raise_errors=False,
                is_base_model=metadata.get("is_base_model", False),
                evaluate_test_split=evaluate_test_split,
                few_shot=few_shot
            )
            
            # Run benchmarker with responses
            with open("/tmp/euroeval_debug.log", "a") as log:
                log.write(f"[DEBUG] Running benchmarker.benchmark() for {dataset}...\n")

            # Use the model name from metadata
            model_name = metadata.get("model_name", "unknown-model")
            per_example_records: List[Dict[str, Any]] = []

            def _observer(*, dataset_config, predictions, references, **kwargs):
                if dataset_config.name != dataset:
                    return
                label_to_prompt, prompt_to_label = _build_label_maps(dataset_config)
                try:
                    with open("/tmp/euroeval_debug.log", "a") as log:
                        log.write(
                            f"[PER_EXAMPLE_OBSERVER] dataset={dataset_config.name} preds={len(predictions)} refs={len(references)}\n"
                        )
                except Exception:
                    pass
                for pred, ref in zip(predictions, references):
                    canonical_pred = _canonicalize_value(pred, prompt_to_label)
                    canonical_ref = _canonicalize_value(ref, prompt_to_label)
                    correctness = _compute_is_correct(dataset_config, canonical_pred, canonical_ref)
                    display_pred = _format_label_for_display(pred, canonical_pred, label_to_prompt)
                    display_ref = _format_label_for_display(ref, canonical_ref, label_to_prompt)
                    qa_scores: Optional[Dict[str, float]] = None
                    if dataset_config.task.task_group == TaskGroup.QUESTION_ANSWERING:
                        if isinstance(pred, dict) and isinstance(pred.get("prediction_text"), str):
                            display_pred = pred["prediction_text"]
                        if isinstance(ref, dict):
                            answers = ref.get("answers")
                            if isinstance(answers, dict):
                                texts = answers.get("text")
                                if isinstance(texts, list) and texts:
                                    display_ref = texts[0]
                        qa_scores = _qa_best_scores(canonical_pred, canonical_ref)
                    record: Dict[str, Any] = {
                        "prediction": _to_serialisable(display_pred),
                        "reference": _to_serialisable(display_ref),
                        "is_correct": correctness,
                    }
                    if qa_scores:
                        record["answer_exact_match"] = qa_scores.get("exact_match")
                        record["answer_f1"] = qa_scores.get("f1")
                    if display_pred != pred:
                        record["prediction_raw"] = _to_serialisable(pred)
                    if display_ref != ref:
                        record["reference_raw"] = _to_serialisable(ref)
                    if isinstance(canonical_pred, str) and isinstance(display_pred, str) and _normalize_str(display_pred) != _normalize_str(canonical_pred):
                        record["prediction_canonical"] = _to_serialisable(canonical_pred)
                    if isinstance(canonical_ref, str) and isinstance(display_ref, str) and _normalize_str(display_ref) != _normalize_str(canonical_ref):
                        record["reference_canonical"] = _to_serialisable(canonical_ref)
                    per_example_records.append(record)

            register_generation_observer(_observer)
            try:
                results = benchmarker.benchmark(model=[model_name])
            finally:
                register_generation_observer(None)
            
            # Debug logging for results
            with open("/tmp/euroeval_debug.log", "a") as log:
                log.write(f"[DEBUG] Benchmark returned {len(results) if results else 0} results for {dataset}\n")
                log.write(f"[DEBUG] Collected {len(per_example_records)} per-example records for {dataset}\n")
            
            # Parse results into metrics for this dataset
            dataset_metrics = {}
            dataset_details = {}
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, "results"):
                    result_dict = result.results
                    
                    # Extract aggregated scores
                    if "total" in result_dict and isinstance(result_dict["total"], dict):
                        total = result_dict["total"]
                        for key, value in total.items():
                            if not key.endswith("_se"):  # Skip standard error metrics
                                metric_name = key.replace("test_", "") if key.startswith("test_") else key
                                dataset_metrics[metric_name] = value
                            else:
                                # Store standard errors in details
                                metric_name = key.replace("test_", "") if key.startswith("test_") else key
                                dataset_details[metric_name] = value
                    
                    # Store raw results in details
                    if "raw" in result_dict:
                        dataset_details["raw_results"] = result_dict["raw"]
                
                # Add metadata to details
                if hasattr(result, "dataset"):
                    dataset_details["dataset_name"] = result.dataset
                if hasattr(result, "task"):
                    dataset_details["task"] = result.task
                if hasattr(result, "dataset_languages"):
                    dataset_details["languages"] = result.dataset_languages
            
            # Store metrics and details with dataset prefix
            for key, value in dataset_metrics.items():
                all_metrics[f"{dataset}_{key}"] = value
            if per_example_records:
                dataset_details["per_example"] = per_example_records
            else:
                with open("/tmp/euroeval_debug.log", "a") as log:
                    log.write(f"[DEBUG] No per-example records stored for {dataset}\n")
            all_details[dataset] = dataset_details
            
        finally:
            # Clean up temp file for this dataset
            if Path(responses_file).exists():
                Path(responses_file).unlink()
    
    # Clean up environment
    if "EUROEVAL_GATEWAY_MODE" in os.environ:
        del os.environ["EUROEVAL_GATEWAY_MODE"]
    if "EUROEVAL_GATEWAY_RESPONSES_FILE" in os.environ:
        del os.environ["EUROEVAL_GATEWAY_RESPONSES_FILE"]
    if "EUROEVAL_IS_BASE_MODEL" in os.environ:
        del os.environ["EUROEVAL_IS_BASE_MODEL"]
    
    return {
        "metrics": all_metrics,
        "details": all_details,
        "metadata": metadata,
    }


# For testing/debugging
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EuroEval batch processing")
    subparsers = parser.add_subparsers(dest='command')
    
    # Prepare command
    prepare_parser = subparsers.add_parser('prepare')
    prepare_parser.add_argument('model', help='Model name')
    prepare_parser.add_argument('--dataset', default='angry-tweets', help='Dataset to evaluate')
    
    # Score command
    score_parser = subparsers.add_parser('score')
    score_parser.add_argument('--input', required=True, help='JSON file with requests and responses')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        result = prepare_requests(args.model, dataset=args.dataset)
        print(json.dumps(result, indent=2))
    
    elif args.command == 'score':
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        result = score_results(
            data['requests'],
            data['responses'],
            data.get('metadata', {})
        )
        print(json.dumps(result, indent=2))
