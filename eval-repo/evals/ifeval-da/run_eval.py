#!/usr/bin/env python3
"""
IFEval-DA integration for LLM Eval Service
This uses the service's broker and eval SDK for proper integration.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add the ifeval-da src to path for the evaluation library
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app.eval_sdk import EvalClient
from ifeval_da import evaluation_lib_key_based as evaluation_lib

# Ensure NLTK data is available (required by ifeval-da)
try:
    import nltk
    import ssl
    
    # Handle SSL issues that sometimes occur with NLTK downloads
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Check if punkt_tab is available, download if not
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading required NLTK data...")
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            # Fallback to regular punkt if punkt_tab fails
            try:
                nltk.download('punkt', quiet=True)
            except:
                print("Warning: Could not download NLTK data. Evaluation may fail.")
except ImportError:
    print("Warning: NLTK not available. Some evaluation features may not work.")
except Exception as e:
    print(f"Warning: Error setting up NLTK: {e}")


async def run(service_url: str, job_dir: str, metadata_path: str):
    """
    Run the IFEval-DA evaluation using the service's broker
    
    Args:
        service_url: URL of the eval service API (with broker)
        job_dir: Directory to write outputs to
        metadata_path: Path to metadata.json file
    
    Returns:
        Dict with evaluation results
    """
    # Initialize eval client with service token
    token = os.environ.get("API_TOKEN", "changeme")
    client = EvalClient(service_url, token)
    
    # Check if this is a base model (which doesn't support instruction following)
    # Get model info from service
    import httpx
    try:
        # Extract base URL from service_url (remove /api/broker if present)
        base_url = service_url.replace("/api/broker", "")
        with httpx.Client() as http_client:
            resp = http_client.get(f"{base_url}/get_model_info", timeout=5.0)
            if resp.status_code == 200:
                model_info = resp.json()
                model_path = model_info.get("model_path", "unknown")
                
                # Try to detect if it's a base model
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    is_base_model = tokenizer.chat_template is None
                    print(f"Model type check: {'base' if is_base_model else 'instruction-tuned'} model detected")
                    
                    if is_base_model:
                        error_msg = "Base model - no instruction following"
                        print(f"\n❌ ERROR: {error_msg}")
                        
                        # Save error metrics
                        metrics = {
                            "error": error_msg,
                            "model": model_path,
                            "status": "SKIPPED"
                        }
                        os.makedirs(job_dir, exist_ok=True)
                        with open(os.path.join(job_dir, "metrics.json"), "w") as f:
                            json.dump(metrics, f, indent=2)
                        
                        # Exit with error
                        sys.exit(1)
                except Exception as e:
                    print(f"Warning: Could not check model type: {e}")
                    # Continue anyway - let it fail later if it's a base model
    except Exception as e:
        print(f"Warning: Could not get model info: {e}")
        # Continue anyway
    
    # Health check - try a single request first to ensure connectivity
    print(f"Checking connection to {service_url}...")
    try:
        test_response = await client.chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
    except Exception as e:
        print(f"\n❌ ERROR: Cannot connect to service at {service_url}")
        print(f"  Details: {e}")
        print(f"\n  Make sure to start a server first:")
        print(f"    python cli.py mock dummy  # For mock server")  
        print(f"    python cli.py serve       # For full service")
        print(f"    python cli.py run ifeval-da --mock  # With auto mock")
        raise RuntimeError(f"Service not available at {service_url}") from e
    
    # Load input data (prefer Danish, fallback to English)
    data_file = Path(__file__).parent / "data" / "danish.jsonl"
    if not data_file.exists():
        data_file = Path(__file__).parent / "data" / "english.jsonl"
    
    if not data_file.exists():
        raise FileNotFoundError(f"No data file found at {data_file}")
    
    # Load prompts
    prompts = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    
    # For testing/development, optionally limit number of prompts
    max_prompts = int(os.environ.get("IFEVAL_MAX_PROMPTS", len(prompts)))
    if max_prompts < len(prompts):
        prompts = prompts[:max_prompts]
        print(f"Limited to {max_prompts} prompts for testing")
    
    print(f"Loaded {len(prompts)} prompts from {data_file.name}")
    
    # Generate responses using the broker
    # The broker will handle rate limiting and concurrency globally
    start_time = time.time()
    
    # Create tasks for all prompts
    tasks = []
    for prompt_data in prompts:
        prompt_text = prompt_data.get("prompt", "")
        # Use the eval SDK's chat method which goes through the broker
        task = client.chat(
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
            max_tokens=2048
        )
        tasks.append(task)
    
    print(f"Generating {len(tasks)} responses via broker...")
    
    # Execute all tasks - broker handles concurrency limits
    responses_raw = await asyncio.gather(*tasks, return_exceptions=True)
    
    generation_time = time.time() - start_time
    
    # Process responses and combine with prompt data
    responses = []
    errors = 0
    for prompt_data, response in zip(prompts, responses_raw):
        result = prompt_data.copy()
        
        if isinstance(response, Exception):
            result["response"] = ""
            result["error"] = str(response)
            errors += 1
        else:
            # Extract content from chat completion response
            try:
                content = response["choices"][0]["message"]["content"]
                result["response"] = content
            except (KeyError, IndexError, TypeError) as e:
                result["response"] = ""
                result["error"] = f"Failed to parse response: {e}"
                errors += 1
        
        responses.append(result)
    
    if errors > 0:
        print(f"Warning: {errors} responses failed")
    
    # Save responses
    os.makedirs(job_dir, exist_ok=True)
    responses_file = os.path.join(job_dir, "responses.jsonl")
    
    with open(responses_file, "w", encoding="utf-8") as f:
        for response in responses:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(responses)} responses to {responses_file}")
    
    # Run evaluation
    print("Running evaluation...")
    eval_start = time.time()
    
    # Create response dictionary for evaluation
    # The evaluation library expects a dict mapping keys to lists of responses
    response_dict = {}
    none_responses = []
    for r in responses:
        if "key" in r:
            if r.get("response") is None:
                none_responses.append(r.get("key", "unknown"))
            else:
                response_dict[r["key"]] = [r["response"]]  # Wrap in list
    
    if none_responses:
        error_msg = f"Got None responses for {len(none_responses)} prompts: {none_responses[:5]}"
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg)
    
    # Run both strict and loose evaluation
    strict_results = []
    loose_results = []
    
    for prompt_data in prompts:
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
    
    eval_time = time.time() - eval_start
    
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
    
    # Save evaluation results
    strict_file = os.path.join(job_dir, "eval_results_strict.jsonl")
    loose_file = os.path.join(job_dir, "eval_results_loose.jsonl")
    
    with open(strict_file, "w") as f:
        for result in strict_results:
            f.write(json.dumps(result) + "\n")
    
    with open(loose_file, "w") as f:
        for result in loose_results:
            f.write(json.dumps(result) + "\n")
    
    # Prepare comprehensive metrics
    metrics = {
        "dataset": data_file.name,
        "n_prompts": len(prompts),
        "n_responses": len(responses),
        "n_errors": errors,
        "generation_time_s": generation_time,
        "eval_time_s": eval_time,
        "total_time_s": generation_time + eval_time,
        
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
    }
    
    # Save metrics
    metrics_file = os.path.join(job_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("IFEVAL-DA EVALUATION COMPLETE")
    print("="*60)
    print(f"Dataset: {data_file.name}")
    print(f"Prompts: {len(prompts)}")
    print(f"Successful responses: {len(responses) - errors}")
    print(f"Generation time: {generation_time:.1f}s")
    print(f"Evaluation time: {eval_time:.1f}s")
    print(f"\nOVERALL RESULTS:")
    print(f"  Strict accuracy: {strict_accuracy:.1%} ({metrics['strict_correct']}/{len(strict_results)})")
    print(f"  Loose accuracy:  {loose_accuracy:.1%} ({metrics['loose_correct']}/{len(loose_results)})")
    print(f"\nINSTRUCTION-LEVEL:")
    print(f"  Strict: {strict_inst_accuracy:.1%} ({strict_inst_correct}/{total_instructions})")
    print(f"  Loose:  {loose_inst_accuracy:.1%} ({loose_inst_correct}/{total_instructions})")
    
    if category_accuracy_strict:
        print(f"\nCATEGORY BREAKDOWN (Strict):")
        for cat in sorted(category_accuracy_strict.keys()):
            stats = category_stats_strict[cat]
            acc = category_accuracy_strict[cat]
            print(f"  {cat:20s}: {acc:6.1%} ({stats['correct']}/{stats['total']})")
    
    print("="*60)
    
    return {
        "metrics": metrics,
        "strict_results": strict_results[:5],  # Return sample for verification
        "loose_results": loose_results[:5]
    }


async def main():
    """CLI entry point - parses args and calls run()"""
    parser = argparse.ArgumentParser(description="Run IFEval-DA evaluation")
    parser.add_argument("--service-url", required=True, help="URL of the eval service API")
    parser.add_argument("--job-dir", required=True, help="Directory to write outputs")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    args = parser.parse_args()
    
    result = await run(args.service_url, args.job_dir, args.metadata)
    return result


if __name__ == "__main__":
    asyncio.run(main())