"""
RULER (Danish adaptation) batch adapter.

Implements two-phase protocol used by the orchestrator:
- prepare_requests(model_name, **kwargs) -> list of OpenAI-compatible requests
- score_results(requests, responses, metadata) -> metrics + details

We re-use the provided RULER scripts to generate datasets and evaluate predictions,
while converting to/from the gateway's request/response format.
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple
try:
    import tomllib
except Exception:  # pragma: no cover
    try:
        import tomli as tomllib
    except Exception:
        tomllib = None
import subprocess
import re


# Paths within this eval module
THIS_DIR = Path(__file__).parent
SCRIPTS_DIR = THIS_DIR / "scripts"

# Load configuration
CONFIG_FILE = THIS_DIR / "eval.toml"
if tomllib is not None and CONFIG_FILE.exists():
    with open(CONFIG_FILE, "rb") as f:
        CONFIG = tomllib.load(f)
else:
    CONFIG = {}


def _sanitize_model_dir(name: str) -> str:
    # Avoid nested directories when using HF repo ids
    return name.replace("/", "--")


def _load_tasks_from_yaml(benchmark: str, exclude_qa: bool) -> List[str]:
    import yaml
    cfg_path = SCRIPTS_DIR / f"{benchmark}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        tasks_cfg = yaml.safe_load(f)
    tasks = list(tasks_cfg.keys())
    if exclude_qa:
        def is_qa(name: str) -> bool:
            conf = tasks_cfg.get(name, {})
            return name.startswith("qa") or conf.get("task") == "qa"
        tasks = [t for t in tasks if not is_qa(t)]
    return tasks


def _merge_task_config(benchmark: str, task_name: str) -> Dict[str, Any]:
    """Merge base constants with customized YAML for a task to get tokens_to_generate etc."""
    sys.path.append(str(SCRIPTS_DIR))
    import yaml
    # Base constants
    base = __import__(f"data.{benchmark}.constants", fromlist=["TASKS"]).TASKS
    # Custom config
    with open(SCRIPTS_DIR / f"{benchmark}.yaml", "r", encoding="utf-8") as f:
        tasks_custom = yaml.safe_load(f)
    cfg = dict(tasks_custom.get(task_name, {}))
    base_cfg = base[cfg.get("task")]
    merged = dict(base_cfg)
    merged.update(cfg)
    return merged


def _ensure_dataset(benchmark: str, task: str, tokenizer_type: str, tokenizer_path: str,
                    seq_len: int, num_samples: int, data_dir: Path, remove_newline_tab: bool = False) -> None:
    out_dir = data_dir / task
    out_file = out_dir / "validation.jsonl"
    if out_file.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "data/prepare.py"),
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


def prepare_requests(model_name: str, **kwargs) -> Dict[str, Any]:
    """
    Phase 1: Generate RULER datasets and return OpenAI-style requests.

    Returns:
      {
        "requests": [ {messages|prompt...}, ... ],
        "metadata": { ... mapping to reconstruct per-task/per-length splits ... }
      }
    """
    ruler_cfg = CONFIG.get("ruler", {})
    benchmark = ruler_cfg.get("benchmark", "synthetic")
    seq_lengths = kwargs.get("seq_lengths") or ruler_cfg.get("seq_lengths", [4096])
    num_samples = int(kwargs.get("num_samples") or ruler_cfg.get("num_samples", 200))
    exclude_qa = bool(kwargs.get("exclude_qa") or ruler_cfg.get("exclude_qa", True))

    # Determine tasks
    tasks_arg = kwargs.get("tasks") or ruler_cfg.get("tasks", "all")
    if isinstance(tasks_arg, str) and tasks_arg != "all":
        tasks = [t.strip() for t in tasks_arg.split(",") if t.strip()]
    else:
        tasks = _load_tasks_from_yaml(benchmark, exclude_qa=exclude_qa)

    # Tokenizer for dataset generation: prefer OpenAI/tiktoken to avoid HF download
    tok_type, tok_path = ("openai", "cl100k_base")

    # Work directory strategy:
    # - Prefer external work dir via EVAL_WORK_DIR env (no cleanup by default)
    # - Else create a temp directory (cleanup by default unless EVAL_KEEP_WORK is set)
    env_work = os.getenv("EVAL_WORK_DIR")
    if env_work:
        root = Path(env_work) / "ruler-da" / _sanitize_model_dir(model_name) / benchmark
        cleanup = False
    else:
        root = Path(tempfile.mkdtemp(prefix=f"ruler-da_{_sanitize_model_dir(model_name)}_{benchmark}_"))
        cleanup = os.getenv("EVAL_KEEP_WORK", "0") not in ("1", "true", "True")

    all_requests: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "benchmark": benchmark,
        "seq_lengths": seq_lengths,
        "tasks": tasks,
        "base_dir": str(root),
        "work_dir": str(root),
        "cleanup": bool(cleanup),
        "dataset_slices": {},  # key: f"{L}:{task}" -> slice indices
    }

    for L in seq_lengths:
        data_dir = root / str(L) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            # Ensure dataset exists
            _ensure_dataset(
                benchmark=benchmark, task=task,
                tokenizer_type=tok_type, tokenizer_path=tok_path,
                seq_len=int(L), num_samples=num_samples,
                data_dir=data_dir,
            )

            # Determine tokens_to_generate per task for request max_tokens
            tcfg = _merge_task_config(benchmark, task)
            max_new = int(tcfg.get("tokens_to_generate", 128))

            # Read dataset and turn into chat requests
            ds_file = data_dir / task / "validation.jsonl"
            start_idx = len(all_requests)
            with open(ds_file, "r", encoding="utf-8") as f:
                for line in f:
                    dp = json.loads(line)
                    prompt = dp.get("input", "")
                    req: Dict[str, Any] = {
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_new,
                        "temperature": 0.0,
                        "_ruler": {
                            "seq_len": int(L),
                            "task": task,
                            "index": dp.get("index", -1),
                            "length": dp.get("length", -1),
                        }
                    }
                    all_requests.append(req)
            end_idx = len(all_requests)
            meta["dataset_slices"][f"{L}:{task}"] = {
                "start_index": start_idx,
                "end_index": end_idx,
                "num_requests": end_idx - start_idx,
            }

    return {
        "requests": all_requests,
        "metadata": meta,
    }


def _write_predictions_for_slice(pred_dir: Path, task: str, requests: List[Dict[str, Any]], responses: List[Dict[str, Any]], references: List[List[str]] | None = None):
    """Write predictions JSONL file that RULER's evaluator expects for a single task.
    We rely on the fact that prepare_requests produced requests for this slice contiguously.
    """
    pred_file = pred_dir / f"{task}.jsonl"
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_file, "w", encoding="utf-8") as fout:
        for i, (req, resp) in enumerate(zip(requests, responses)):
            # Extract model text from response (chat or completion)
            text = ""
            if isinstance(resp, dict):
                try:
                    if "choices" in resp and resp["choices"]:
                        choice = resp["choices"][0]
                        if "message" in choice:
                            text = choice.get("message", {}).get("content", "") or ""
                        elif "text" in choice:
                            text = choice.get("text", "") or ""
                except Exception:
                    text = ""

            # Ground-truth references from dataset if provided
            outs = []
            if references is not None and i < len(references):
                ref = references[i]
                # Normalize to a list of strings
                if isinstance(ref, list):
                    outs = [str(x) for x in ref]
                elif ref is not None:
                    outs = [str(ref)]

            # Minimal schema; evaluator expects 'outputs' to contain references
            out = {
                "index": i,
                "input": req.get("messages", [{}])[0].get("content", ""),
                "outputs": outs,
                "pred": text,
                "others": {"id": i},
                "truncation": -1,
                "length": req.get("_ruler", {}).get("length", -1),
            }
            fout.write(json.dumps(out) + "\n")


def _parse_num(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except Exception:
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
        return float(m.group(0)) if m else 0.0


def _parse_summary_csv(path: Path) -> Dict[str, float]:
    """Parse summary.csv or summary-<task>.csv written by evaluate.py (pandas CSV with header row).
    Returns a mapping {task_name: score}.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return {}

    if not rows:
        return {}

    # Split into cells
    cells = [r.split(",") for r in rows]

    # Drop a leading pandas header row like "0,1,2,..." if present
    def _is_numeric_header(row: List[str]) -> bool:
        return all(c.strip().isdigit() for c in row if c.strip() != "")

    if cells and _is_numeric_header(cells[0]):
        cells = cells[1:]

    # Find Tasks and Score rows
    tasks: List[str] | None = None
    scores: List[str] | None = None
    for r in cells:
        if not r:
            continue
        key = r[0].strip().lower()
        if key == "tasks":
            tasks = r[1:]
        elif key == "score":
            scores = r[1:]

    if not tasks or not scores:
        return {}

    # Align lengths conservatively
    n = min(len(tasks), len(scores))
    return {tasks[i]: _parse_num(scores[i]) for i in range(n)}


def score_results(requests: List[Dict[str, Any]], responses: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 2: Write predictions per (seq_len, task) slice and invoke RULER evaluator.
    Returns per-length metrics and a combined details dict.
    """
    benchmark = metadata.get("benchmark", "synthetic")
    base_dir = Path(metadata.get("base_dir") or metadata.get("work_dir") or Path(tempfile.mkdtemp(prefix="ruler-da_score_")))
    will_cleanup = bool(metadata.get("cleanup", False)) and os.getenv("EVAL_KEEP_WORK", "0") not in ("1", "true", "True")
    dataset_slices = metadata.get("dataset_slices", {})

    # Group slices by seq_len -> [tasks]
    by_len: Dict[int, List[str]] = {}
    for key in dataset_slices.keys():
        seq_str, task = key.split(":", 1)
        by_len.setdefault(int(seq_str), []).append(task)

    metrics: Dict[str, Any] = {}
    details: Dict[str, Any] = {"lengths": {}, "slices": dataset_slices}
    per_length_summary: List[Tuple[int, float]] = []  # (L, avg)

    for L, tasks in by_len.items():
        # Collect predictions into pred_dir
        pred_dir = base_dir / str(L) / "pred"
        pred_dir.mkdir(parents=True, exist_ok=True)

        # For each task, extract appropriate request/response sublists and write one JSONL file
        for task in tasks:
            sl = dataset_slices.get(f"{L}:{task}", {})
            s, e = int(sl.get("start_index", 0)), int(sl.get("end_index", 0))
            # Load references from the dataset file to include in predictions
            data_dir = base_dir / str(L) / "data"
            ds_file = data_dir / task / "validation.jsonl"
            refs: List[List[str]] | None = None
            try:
                with open(ds_file, "r", encoding="utf-8") as f:
                    refs = []
                    for line in f:
                        dp = json.loads(line)
                        r = dp.get("outputs")
                        if r is None:
                            # Fallback single reference
                            r = [dp.get("output", "")]
                        # Ensure list of strings
                        if isinstance(r, list):
                            refs.append([str(x) for x in r])
                        else:
                            refs.append([str(r)])
            except Exception:
                refs = None
            _write_predictions_for_slice(pred_dir, task, requests[s:e], responses[s:e], refs)

        # Run evaluator for this length over all tasks present in pred_dir
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "eval/evaluate.py"),
            "--data_dir", str(pred_dir),
            "--benchmark", benchmark,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # On failure, record stderr and continue
            details.setdefault("errors", []).append({"seq_len": L, "stderr": e.stderr[-1000:] if e.stderr else ""})
            continue

        # Parse summary.csv for this length
        summary_csv = pred_dir / "summary.csv"
        if not summary_csv.exists():
            # If only one task per file, evaluator writes summary-<task>.csv. Parse each and aggregate.
            import glob
            per_task_files = list(glob.glob(str(pred_dir / "summary-*.csv")))
            if per_task_files:
                per_task_map: Dict[str, float] = {}
                vals: List[float] = []
                for p in per_task_files:
                    parsed = _parse_summary_csv(Path(p))
                    for t, v in parsed.items():
                        per_task_map[t] = v
                        vals.append(v)
                if vals:
                    avg_len = sum(vals) / len(vals)
                    metrics[f"ruler_{L}_avg"] = avg_len
                    per_length_summary.append((int(L), float(avg_len)))
                    for t, v in per_task_map.items():
                        metrics[f"ruler_{L}:{t}"] = v
                details["lengths"][str(L)] = {"per_task": per_task_map, "from_files": per_task_files}
            continue

        try:
            per_task = _parse_summary_csv(summary_csv)
            if per_task:
                vals = list(per_task.values())
                for t, v in per_task.items():
                    metrics[f"ruler_{L}:{t}"] = v
                avg_len = sum(vals) / len(vals)
                metrics[f"ruler_{L}_avg"] = avg_len
                per_length_summary.append((int(L), float(avg_len)))
                details["lengths"][str(L)] = {"per_task": per_task}
            else:
                details.setdefault("errors", []).append({"seq_len": L, "parse_error": "empty_or_unparsed_summary"})
        except Exception as e:
            details.setdefault("errors", []).append({"seq_len": L, "parse_error": str(e)})

    # Compute overall summary across lengths: Avg and wAvg (inc)
    if per_length_summary:
        per_length_summary.sort(key=lambda x: x[0])
        avgs = [a for _, a in per_length_summary]
        overall_avg = sum(avgs) / len(avgs)
        n = len(avgs)
        weights = list(range(1, n + 1))
        wavg_inc = sum(a * w for a, w in zip(avgs, weights)) / sum(weights)
        # Standard names expected by UI
        metrics["avg"] = overall_avg
        metrics["wavg_inc"] = wavg_inc
        metrics["score"] = overall_avg  # alias for compatibility
        details["overall"] = {
            "per_length": per_length_summary,
            "avg": overall_avg,
            "wavg_inc": wavg_inc,
        }

    result = {"metrics": metrics, "details": details}

    # Best-effort cleanup of temporary work dir created at prepare time
    try:
        if will_cleanup:
            shutil.rmtree(base_dir, ignore_errors=True)
    except Exception:
        pass

    return result
