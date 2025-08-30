#!/usr/bin/env python3
"""
Unified launcher for RULER (Danish adaptation) using an OpenAI-compatible API.

Responsibilities:
- Discover served model from /v1/models (OpenAI-compatible endpoint), unless --model is given.
- Auto-select a chat template for instruct models (default: meta-llama3; overridable).
- Auto-select tokenizer for prepare: HF if --model_local_path provided, otherwise OpenAI/tiktoken.
- Prepare datasets on-demand for each task and sequence length.
- Call inference via OpenAI Chat Completions and write predictions.
- Evaluate predictions and write summary CSV.

Environment variables:
- OPENAI_BASE_URL: e.g., http://127.0.0.1:8000/v1 (vLLM OpenAI server)
- OPENAI_API_KEY: set to any non-empty value for local vLLM; actual key for OpenAI

Example:
  export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
  export OPENAI_API_KEY=dummy
  python scripts/launch.py \
    --benchmark synthetic \
    --seq-lengths 4096,8192 \
    --num-samples 100 \
    --exclude-qa
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


def get_models_via_openai() -> list:
    try:
        from openai import OpenAI
    except Exception:
        return []
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "dummy")
    if not base_url:
        return []
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        models = client.models.list()
        return [m.id for m in getattr(models, 'data', [])]
    except Exception:
        return []



from typing import Optional, Tuple


def detect_tokenizer(model_name: str, model_local_path: Optional[str]) -> Tuple[str, str]:
    """Return (tokenizer_type, tokenizer_path/id).
    Prefer HF tokenizer: local path if provided; otherwise the discovered model id.
    Fallback to OpenAI/tiktoken only if neither is available.
    """
    if model_local_path:
        mpath = Path(model_local_path)
        if (mpath / "tokenizer.model").exists():
            return "spm", str(mpath / "tokenizer.model")
        if mpath.exists():
            return "hf", str(mpath)
    if model_name:
        # Let transformers download the tokenizer by repo id
        return "hf", model_name
    return "openai", "cl100k_base"


def load_tasks(benchmark: str, exclude_qa: bool) -> list[str]:
    config_path = Path(__file__).parent / f"{benchmark}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        tasks_cfg = yaml.safe_load(f)
    tasks = list(tasks_cfg.keys())
    if exclude_qa:
        def is_qa(name: str) -> bool:
            conf = tasks_cfg.get(name, {})
            return (name.startswith("qa") or conf.get("task") == "qa")
        tasks = [t for t in tasks if not is_qa(t)]
    return tasks


def ensure_dataset(task: str, seq_len: int, num_samples: int, benchmark: str,
                   tokenizer_type: str, tokenizer_path: str,
                   data_dir: Path, remove_newline_tab: bool = False) -> None:
    out_dir = data_dir / task
    out_file = out_dir / "validation.jsonl"
    if out_file.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(Path(__file__).parent / "data/prepare.py"),
        "--save_dir", str(data_dir),
        "--benchmark", benchmark,
        "--task", task,
        "--tokenizer_path", tokenizer_path,
        "--tokenizer_type", tokenizer_type,
        "--max_seq_length", str(seq_len),
        "--num_samples", str(num_samples),
    ]
    if remove_newline_tab:
        cmd.append("--remove_newline_tab")
    subprocess.run(cmd, check=True)


def run_predictions(task: str, benchmark: str, data_dir: Path, pred_dir: Path,
                    model_name: str, temperature: float, top_k: int, top_p: float) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(Path(__file__).parent / "pred/call_api.py"),
        "--data_dir", str(data_dir),
        "--save_dir", str(pred_dir),
        "--benchmark", benchmark,
        "--task", task,
        "--server_type", "openai",
        "--model_name_or_path", model_name,
        "--temperature", str(temperature),
        "--top_k", str(top_k),
        "--top_p", str(top_p),
    ]
    subprocess.run(cmd, check=True)


def run_eval(pred_dir: Path, benchmark: str) -> None:
    cmd = [
        sys.executable, str(Path(__file__).parent / "eval/evaluate.py"),
        "--data_dir", str(pred_dir),
        "--benchmark", benchmark,
    ]
    subprocess.run(cmd, check=True)


def summarize_lengths(base_dir: Path) -> None:
    cmd = [
        sys.executable, str(Path(__file__).parent / "eval/summarize_lengths.py"),
        "--base_dir", str(base_dir),
    ]
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="synthetic")
    p.add_argument("--tasks", default="all", help="all or comma-separated task ids from YAML")
    p.add_argument("--exclude-qa", action="store_true")
    p.add_argument("--seq-lengths", default="4096,8192,16384,32768")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--model", default=None, help="served model id for OpenAI API; if omitted, discover via /v1/models")
    p.add_argument("--model_local_path", default=None, help="HF model dir for tokenizer (optional)")
    # No manual chat template; server-side tokenizer applies it
    p.add_argument("--save_root", default="benchmark_root")
    p.add_argument("--remove_newline_tab", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Discover model if not provided
    model_name = args.model
    if not model_name:
        models = get_models_via_openai()
        if not models:
            raise RuntimeError("Could not discover model via /v1/models; set OPENAI_BASE_URL and/or pass --model")
        # Pick the first model, or refine selection here
        model_name = models[0]
        print(f"Discovered model: {model_name}")

    # Select tokenizer for prepare
    tok_type, tok_path = detect_tokenizer(model_name, args.model_local_path)
    print(f"Tokenizer: type={tok_type} path={tok_path}")
    if tok_type == "openai":
        print("Warning: using OpenAI/tiktoken for token budgeting; counts may differ slightly from the served HF tokenizer. Pass --model_local_path for exact budgeting.")

    # Build task list
    if args.tasks == "all":
        tasks = load_tasks(args.benchmark, exclude_qa=args.exclude_qa)
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"Tasks: {tasks}")

    # Iterate
    seq_lengths = [int(s) for s in args.seq_lengths.split(",") if s]
    root = Path(args.save_root) / model_name / args.benchmark

    for L in seq_lengths:
        data_dir = root / str(L) / "data"
        pred_dir = root / str(L) / "pred"
        data_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            ensure_dataset(
                task=task, seq_len=L, num_samples=args.num_samples, benchmark=args.benchmark,
                tokenizer_type=tok_type, tokenizer_path=tok_path,
                data_dir=data_dir, remove_newline_tab=args.remove_newline_tab,
            )
            run_predictions(
                task=task, benchmark=args.benchmark, data_dir=data_dir, pred_dir=pred_dir,
                model_name=model_name, temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p,
            )

        run_eval(pred_dir=pred_dir, benchmark=args.benchmark)

    # Summarize across lengths (Avg and wAvg inc)
    try:
        summarize_lengths(root)
    except Exception as e:
        print(f"[warn] Failed to summarize across lengths: {e}")

    print("Launch completed.")


if __name__ == "__main__":
    main()
