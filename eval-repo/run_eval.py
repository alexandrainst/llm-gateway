#!/usr/bin/env python3
"""
Minimal eval runner for two-phase batch evaluation protocol.

Usage:
    # List which evals to run
    python run_eval.py list
    
    # Phase 1: Prepare requests
    python run_eval.py prepare <eval_name> <model_name> [options]
    
    # Phase 2: Score results  
    python run_eval.py score <eval_name> < results.json
"""

import sys
import json
try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    try:
        import tomli as tomllib  # Backport for Python <3.11
    except Exception:
        tomllib = None  # Fallback: config loading will return None
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional


def load_eval_config(eval_name: str) -> Optional[Dict[str, Any]]:
    """Load eval configuration from eval.toml if it exists."""
    config_path = Path(__file__).parent / "evals" / eval_name / "eval.toml"
    if config_path.exists() and tomllib is not None:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return None


def load_eval_module(eval_name: str):
    """Load an eval module by name."""
    eval_path = Path(__file__).parent / "evals" / eval_name
    
    # Try to load eval.py first, then __init__.py
    module_file = eval_path / "eval.py"
    if not module_file.exists():
        module_file = eval_path / "__init__.py"
    
    if not module_file.exists():
        raise ImportError(f"No eval.py or __init__.py found for eval '{eval_name}'")
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(f"evals.{eval_name}", module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"evals.{eval_name}"] = module
    spec.loader.exec_module(module)
    
    # Verify required functions exist
    if not hasattr(module, 'prepare_requests'):
        raise ImportError(f"Eval '{eval_name}' missing prepare_requests function")
    if not hasattr(module, 'score_results'):
        raise ImportError(f"Eval '{eval_name}' missing score_results function")
    
    return module


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_eval.py <action> [args...]", file=sys.stderr)
        print("Actions: list, prepare, score", file=sys.stderr)
        sys.exit(1)
    
    action = sys.argv[1]
    
    # Handle 'list' action - returns which evals to run
    if action == "list":
        # Check for optional --base-model flag
        is_base_model = "--base-model" in sys.argv
        
        config_path = Path(__file__).parent / "config.json"
        
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                all_evals = config.get("default_evals", [])
        else:
            all_evals = ["example_eval"]
        
        # Filter evals based on model type if needed
        if is_base_model:
            filtered_evals = []
            for eval_name in all_evals:
                eval_config = load_eval_config(eval_name)
                if eval_config and "eval" in eval_config:
                    requires_instruction = eval_config["eval"].get("requires_instruction_model", False)
                    if not requires_instruction:
                        filtered_evals.append(eval_name)
                else:
                    # If no config, include by default
                    filtered_evals.append(eval_name)
            evals = filtered_evals
        else:
            evals = all_evals
        
        print(json.dumps({"evals": evals}))
        return
    
    # Other actions need eval_name
    if len(sys.argv) < 3:
        print(f"Usage: python run_eval.py {action} <eval_name> [args...]", file=sys.stderr)
        sys.exit(1)
    
    eval_name = sys.argv[2]
    
    try:
        eval_module = load_eval_module(eval_name)
    except ImportError as e:
        print(f"Error loading eval '{eval_name}': {e}", file=sys.stderr)
        sys.exit(1)
    
    if action == "prepare":
        # Prepare requests for batch submission
        if len(sys.argv) < 4:
            print("Usage: python run_eval.py prepare <eval_name> <model_name> [--max-prompts N] [--base-model]", file=sys.stderr)
            sys.exit(1)
        
        model_name = sys.argv[3]
        
        # Parse optional arguments
        kwargs = {}
        i = 4
        while i < len(sys.argv):
            if sys.argv[i] == "--max-prompts" and i + 1 < len(sys.argv):
                kwargs["max_prompts"] = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--use-english":
                kwargs["use_english"] = True
                i += 1
            elif sys.argv[i] == "--base-model":
                kwargs["is_base_model"] = True
                i += 1
            else:
                print(f"Unknown argument: {sys.argv[i]}", file=sys.stderr)
                i += 1
        
        try:
            result = eval_module.prepare_requests(model_name, **kwargs)
            # Attach optional concurrency hint from eval.toml
            try:
                eval_config = load_eval_config(eval_name)
                max_conc = None
                if eval_config and "eval" in eval_config:
                    mc = eval_config["eval"].get("max_concurrency")
                    if mc is not None:
                        max_conc = int(mc)
                if max_conc is not None:
                    if "metadata" not in result or not isinstance(result.get("metadata"), dict):
                        result["metadata"] = {}
                    result["metadata"]["max_concurrency"] = max_conc
            except Exception:
                # Non-fatal: ignore malformed config and continue
                pass
            print(json.dumps(result))
        except Exception as e:
            print(f"Error preparing requests: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif action == "score":
        # Score results from batch completion
        try:
            input_data = json.loads(sys.stdin.read())
        except json.JSONDecodeError as e:
            print(f"Error parsing input JSON: {e}", file=sys.stderr)
            sys.exit(1)
        
        if "requests" not in input_data or "responses" not in input_data:
            print("Input must contain 'requests' and 'responses' fields", file=sys.stderr)
            sys.exit(1)
        
        try:
            result = eval_module.score_results(
                input_data["requests"],
                input_data["responses"],
                input_data.get("metadata", {})
            )
            print(json.dumps(result))
        except Exception as e:
            print(f"Error scoring results: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        print(f"Unknown action: {action}", file=sys.stderr)
        print("Valid actions: list, prepare, score", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
