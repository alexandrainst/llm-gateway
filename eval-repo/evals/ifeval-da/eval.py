"""
IFEval-DA evaluation with two-phase protocol for batch processing.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add the ifeval-da src to path for the evaluation library
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Download required NLTK data if not present
import nltk
nltk_data_dir = Path(__file__).parent / ".venv" / "nltk_data"
nltk_data_dir.mkdir(parents=True, exist_ok=True)
nltk.data.path.insert(0, str(nltk_data_dir))
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=str(nltk_data_dir))
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=str(nltk_data_dir))

from ifeval_da import evaluation_lib_key_based as evaluation_lib

REASONING_END_TOKENS = ("</think>", "</reason>", "</reasoning>", "</analysis>", "</chain_of_thought>")
REASONING_START_TOKENS = ("<think>", "<reason>", "<reasoning>", "<analysis>", "<chain_of_thought>")


def _strip_reasoning_content(text: str | None) -> str:
    """Drop reasoning traces (<think>...</think>) so scoring sees the final answer."""
    if not text:
        return ""
    lowered = text.lower()
    for marker in REASONING_END_TOKENS:
        idx = lowered.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].lstrip()
    for marker in REASONING_START_TOKENS:
        if marker in lowered:
            return ""
    return text


def prepare_requests(model_name: str, **kwargs) -> Dict:
    """
    Prepare batch requests for IFEval-DA evaluation.
    
    Args:
        model_name: Name of the model to evaluate
        **kwargs: Additional options (e.g., max_prompts, use_danish)
        
    Returns:
        Dictionary with:
            - requests: List of chat completion requests
            - metadata: Data needed for scoring (prompt keys, instruction IDs, etc.)
    """
    # Load input data (prefer Danish, fallback to English)
    data_file = Path(__file__).parent / "data" / "danish.jsonl"
    if not data_file.exists() or kwargs.get("use_english", False):
        data_file = Path(__file__).parent / "data" / "english.jsonl"
    
    if not data_file.exists():
        raise FileNotFoundError(f"No data file found at {data_file}")
    
    # Load prompts
    prompts = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    
    # Optionally limit number of prompts for testing
    max_prompts = kwargs.get("max_prompts", len(prompts))
    if max_prompts < len(prompts):
        prompts = prompts[:max_prompts]
    
    # Create requests for batch submission
    requests = []
    metadata = {
        "dataset": data_file.name,
        "prompt_data": [],  # Store full prompt data for scoring
    }
    
    for prompt_data in prompts:
        # Create the request
        request = {
            "messages": [{"role": "user", "content": prompt_data.get("prompt", "")}],
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        requests.append(request)
        
        # Store metadata for scoring
        metadata["prompt_data"].append({
            "key": prompt_data.get("key"),
            "instruction_id_list": prompt_data.get("instruction_id_list", []),
            "prompt": prompt_data.get("prompt", ""),
            "kwargs": prompt_data.get("kwargs", [])
        })
    
    return {
        "requests": requests,
        "metadata": metadata
    }


def score_results(requests: List[Dict], responses: List[Dict], metadata: Dict) -> Dict:
    """
    Score the IFEval-DA results.
    
    Args:
        requests: Original requests sent
        responses: Responses received (in same order)
        metadata: The metadata returned from prepare_requests
        
    Returns:
        Dictionary with metrics and detailed results
    """
    prompt_data_list = metadata.get("prompt_data", [])
    
    # Create response dictionary for evaluation library
    # The evaluation library expects a dict mapping keys to lists of responses
    response_dict = {}
    none_responses = []
    
    for i, (prompt_data, response) in enumerate(zip(prompt_data_list, responses)):
        key = prompt_data.get("key")
        if key:
            # Extract the actual response text
            response_text = None
            if isinstance(response, dict):
                # Handle different response formats
                if "choices" in response:
                    # OpenAI format
                    try:
                        response_text = response["choices"][0]["message"]["content"]
                    except (KeyError, IndexError):
                        response_text = None
                elif "content" in response:
                    # Simple format
                    response_text = response["content"]
                elif "error" in response:
                    # Error response
                    response_text = None
            
            if response_text is None:
                none_responses.append(key)
            else:
                response_text = _strip_reasoning_content(response_text)
                response_dict[key] = [response_text]  # Wrap in list as expected
    
    # Run both strict and loose evaluation
    strict_results = []
    loose_results = []
    
    for prompt_data in prompt_data_list:
        key = prompt_data.get("key")
        if key in response_dict:
            # Create input example for evaluation
            input_example = evaluation_lib.InputExample(
                key=key,
                instruction_id_list=prompt_data.get("instruction_id_list", []),
                prompt=prompt_data.get("prompt", ""),
                kwargs=prompt_data.get("kwargs", [])
            )
            
            # Strict evaluation
            strict_result = evaluation_lib.test_instruction_following_strict(
                input_example, response_dict
            )
            strict_results.append({
                "key": key,
                "follow_all_instructions": strict_result.follow_all_instructions,
                "follow_instruction_list": strict_result.follow_instruction_list,
                "instruction_id_list": prompt_data.get("instruction_id_list", [])
            })
            
            # Loose evaluation  
            loose_result = evaluation_lib.test_instruction_following_loose(
                input_example, response_dict
            )
            loose_results.append({
                "key": key,
                "follow_all_instructions": loose_result.follow_all_instructions,
                "follow_instruction_list": loose_result.follow_instruction_list,
                "instruction_id_list": prompt_data.get("instruction_id_list", [])
            })
    
    # Calculate overall metrics
    strict_accuracy = sum(1 for r in strict_results if r["follow_all_instructions"]) / len(strict_results) if strict_results else 0
    loose_accuracy = sum(1 for r in loose_results if r["follow_all_instructions"]) / len(loose_results) if loose_results else 0
    
    # Calculate instruction-level accuracies
    strict_inst_correct = sum(sum(r["follow_instruction_list"]) for r in strict_results)
    loose_inst_correct = sum(sum(r["follow_instruction_list"]) for r in loose_results)
    total_instructions = sum(len(r["follow_instruction_list"]) for r in strict_results)
    
    strict_inst_accuracy = strict_inst_correct / total_instructions if total_instructions > 0 else 0
    loose_inst_accuracy = loose_inst_correct / total_instructions if total_instructions > 0 else 0
    
    # Calculate per-category breakdown
    category_stats_strict = {}
    category_stats_loose = {}
    
    for result in strict_results:
        for inst_id, followed in zip(result["instruction_id_list"], result["follow_instruction_list"]):
            category = inst_id.split(":")[0] if ":" in inst_id else inst_id
            if category not in category_stats_strict:
                category_stats_strict[category] = {"correct": 0, "total": 0}
            category_stats_strict[category]["total"] += 1
            if followed:
                category_stats_strict[category]["correct"] += 1
    
    for result in loose_results:
        for inst_id, followed in zip(result["instruction_id_list"], result["follow_instruction_list"]):
            category = inst_id.split(":")[0] if ":" in inst_id else inst_id
            if category not in category_stats_loose:
                category_stats_loose[category] = {"correct": 0, "total": 0}
            category_stats_loose[category]["total"] += 1
            if followed:
                category_stats_loose[category]["correct"] += 1
    
    # Convert to accuracy percentages
    category_accuracy_strict = {
        cat: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        for cat, stats in category_stats_strict.items()
    }
    category_accuracy_loose = {
        cat: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        for cat, stats in category_stats_loose.items()
    }
    
    return {
        "metrics": {
            "dataset": metadata.get("dataset", "unknown"),
            "n_prompts": len(prompt_data_list),
            "n_responses": len(responses),
            "n_errors": len(none_responses),
            
            # Main accuracy metrics
            "strict_accuracy": strict_accuracy,
            "loose_accuracy": loose_accuracy,
            "strict_inst_accuracy": strict_inst_accuracy,
            "loose_inst_accuracy": loose_inst_accuracy,
            
            # Detailed stats
            "strict_correct": sum(1 for r in strict_results if r["follow_all_instructions"]),
            "loose_correct": sum(1 for r in loose_results if r["follow_all_instructions"]),
            "total_instructions": total_instructions,
            "strict_inst_correct": strict_inst_correct,
            "loose_inst_correct": loose_inst_correct,
            
            # Per-category breakdown
            "category_accuracy_strict": category_accuracy_strict,
            "category_accuracy_loose": category_accuracy_loose,
            "category_stats_strict": category_stats_strict,
            "category_stats_loose": category_stats_loose
        },
        "details": {
            "strict_results": strict_results[:5],  # Sample for inspection
            "loose_results": loose_results[:5],
            "failed_keys": none_responses[:10] if none_responses else []
        }
    }
