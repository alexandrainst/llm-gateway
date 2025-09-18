"""Gateway adapter for running lm-evaluation-harness through the orchestrator two-phase
batch protocol.

Phase 1 (``prepare_requests``) materialises deterministic harness requests and returns
OpenAI-compatible payloads that the gateway enqueues. Phase 2 (``score_results``)
replays lm-eval's aggregation on the collected responses to emit metrics that match the
lm-evaluation-harness output while retaining the structured summary used by the UI.
"""
from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # Python >=3.11
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[assignment]

# Optional compatibility shim for older ``datasets`` releases shipped in some envs
try:
    from datasets import features as _ds_features

    if not hasattr(_ds_features, "List") and hasattr(_ds_features, "Sequence"):
        _ds_features.List = _ds_features.Sequence  # Backwards-compatible alias
except Exception:  # pragma: no cover
    pass

from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.evaluator_utils import (
    TaskOutput,
    consolidate_group_results,
    consolidate_results,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
)

SCHEMA_VERSION = "1.0"
CONFIG_PATH = Path(__file__).with_suffix(".toml")

DEFAULT_CONFIG: Dict[str, Any] = {
    "eval": {
        "tasks": ["ai2_arc", "mmlu_pro", "squad_completion", "triviaqa"],
        "limit": 0,
        "logprobs_top_k": 5,
        "fewshot_seed": 1234,
        "max_concurrency": 4,
    },
    "task": {
        "ai2_arc": {"num_fewshot": 0},
        "mmlu_pro": {"num_fewshot": 5},
        "squad_completion": {"num_fewshot": 0},
        "triviaqa": {"num_fewshot": 0, "requires_instruction_model": False},
    },
}


@dataclass
class RequestManifestEntry:
    task: str
    doc_id: int
    instance_idx: int
    request_type: str
    repeat_index: int


def _deep_update(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``incoming`` into ``base`` and return the result."""
    for key, value in incoming.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _load_config() -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if tomllib is None or not CONFIG_PATH.exists():
        return cfg
    with CONFIG_PATH.open("rb") as fh:
        loaded = tomllib.load(fh)
    return _deep_update(cfg, loaded)


def _serialise_config_block(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serialisable copy of a config mapping."""
    def _coerce(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (dict, list, tuple)):
            return json.loads(json.dumps(obj, default=str))
        return obj

    return {k: _coerce(v) for k, v in data.items()}


def _ensure_numeric(value: Any) -> Any:
    try:
        if isinstance(value, bool):  # bool is subclass of int; keep as-is
            return value
        if isinstance(value, (int, float)):
            return float(value)
        if hasattr(value, "__float__"):
            return float(value)  # type: ignore[arg-type]
    except Exception:
        pass
    return value


def _jsonify(obj: Any) -> Any:
    """json.dumps/json.loads round-trip with fallbacks for numpy/path types."""
    def _default(o: Any):
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:  # pragma: no cover
                pass
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)

    return json.loads(json.dumps(obj, default=_default))


def _build_task_settings(config: Dict[str, Any], is_base_model: bool) -> List[Tuple[str, Dict[str, Any]]]:
    eval_cfg = config.get("eval", {})
    tasks = eval_cfg.get("tasks", []) or []
    task_cfg = config.get("task", {})

    resolved: List[Tuple[str, Dict[str, Any]]] = []
    for name in tasks:
        settings = deepcopy(task_cfg.get(name, {}))
        if is_base_model and settings.get("requires_instruction_model"):
            continue
        resolved.append((name, settings))
    return resolved


def _configure_task(task, task_settings: Dict[str, Any], global_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply overrides to a harness task and return the applied values."""
    applied: Dict[str, Any] = {}

    if "num_fewshot" in task_settings:
        num_fs = int(task_settings["num_fewshot"])
        task.set_config("num_fewshot", num_fs)
        applied["num_fewshot"] = num_fs

    seed = task_settings.get("fewshot_seed", global_cfg.get("fewshot_seed"))
    if seed is not None:
        task.set_fewshot_seed(int(seed))
        applied["fewshot_seed"] = int(seed)

    if "repeats" in task_settings:
        repeats = int(task_settings["repeats"])
        task.set_config("repeats", repeats)
        applied["repeats"] = repeats

    if "apply_chat_template" in task_settings:
        applied["apply_chat_template"] = bool(task_settings["apply_chat_template"])

    limit = task_settings.get("limit", global_cfg.get("limit"))
    applied["limit"] = None if (limit is None or int(limit) <= 0) else int(limit)

    return applied


def _instance_to_request(instance, logprobs_top_k: int, repeat_index: int) -> Tuple[Dict[str, Any], RequestManifestEntry]:
    req_type = instance.request_type
    if req_type not in {"loglikelihood", "generate_until"}:
        raise ValueError(f"Unsupported request type: {req_type}")

    if req_type == "loglikelihood":
        context, continuation = instance.arguments
        payload = {
            "prompt": context + continuation,
            "max_tokens": 0,
            "temperature": 0.0,
            "logprobs": max(1, int(logprobs_top_k)),
            "echo": True,
        }
    else:  # generate_until
        context, gen_kwargs = instance.arguments
        max_tokens = int(
            gen_kwargs.get("max_gen_toks")
            or gen_kwargs.get("max_length")
            or gen_kwargs.get("max_tokens")
            or 512
        )
        payload = {
            "prompt": context,
            "max_tokens": max_tokens,
            "temperature": float(gen_kwargs.get("temperature", 0.0)),
            "top_p": float(gen_kwargs.get("top_p", 1.0)),
        }
        if "stop" in gen_kwargs:
            payload["stop"] = gen_kwargs["stop"]
        elif "until" in gen_kwargs:
            payload["stop"] = gen_kwargs["until"]

    manifest = RequestManifestEntry(
        task=instance.task_name,
        doc_id=int(instance.doc_id) if instance.doc_id is not None else -1,
        instance_idx=int(instance.idx),
        request_type=req_type,
        repeat_index=repeat_index,
    )
    return payload, manifest


def prepare_requests(model_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Phase 1: produce gateway-ready request payloads for the configured harness tasks."""
    is_base_model = bool(kwargs.get("is_base_model", False))
    config = _load_config()
    eval_cfg = config.get("eval", {})

    task_plan = _build_task_settings(config, is_base_model)
    if not task_plan:
        return {"requests": [], "metadata": {"schema": SCHEMA_VERSION, "tasks": []}}

    manager = TaskManager()

    requests: List[Dict[str, Any]] = []
    manifest_entries: List[Dict[str, Any]] = []
    task_meta: Dict[str, Any] = {}
    alias_map: Dict[str, List[str]] = {}
    child_tasks: List[str] = []

    logprobs_top_k = int(eval_cfg.get("logprobs_top_k", 5))

    for alias, settings in task_plan:
        try:
            sub_dict = get_task_dict([alias], task_manager=manager)
        except Exception as exc:
            raise ValueError(f"Unknown lm-harness task or group '{alias}'") from exc

        task_outputs = get_task_list(sub_dict)
        if not task_outputs:
            raise ValueError(f"No tasks resolved for alias '{alias}'")

        for task_output in task_outputs:
            task_obj = task_output.task
            child_name = task_output.task_name
            applied = _configure_task(task_obj, settings, eval_cfg)

            task_obj.build_all_requests(
                limit=applied.get("limit"),
                cache_requests=False,
                rewrite_requests_cache=False,
                system_instruction=settings.get("system_instruction"),
                apply_chat_template=bool(settings.get("apply_chat_template", False)),
                fewshot_as_multiturn=bool(settings.get("fewshot_as_multiturn", False)),
            )

            task_meta[child_name] = {
                "applied_settings": applied,
                "config": _serialise_config_block(task_obj.dump_config()),
                "alias": alias,
            }
            alias_map.setdefault(alias, []).append(child_name)
            child_tasks.append(child_name)

            for inst in task_obj.instances:
                repeats = max(1, int(getattr(inst, "repeats", 1) or 1))
                for repeat_idx in range(repeats):
                    payload, man_entry = _instance_to_request(inst, logprobs_top_k, repeat_idx)
                    requests.append(payload)
                    manifest_entries.append({
                        "task": child_name,
                        "doc_id": man_entry.doc_id,
                        "instance_idx": man_entry.instance_idx,
                        "request_type": man_entry.request_type,
                        "repeat_index": man_entry.repeat_index,
                    })

    metadata = {
        "schema": SCHEMA_VERSION,
        "model": model_name,
        "is_base_model": is_base_model,
        "tasks": child_tasks,
        "task_metadata": task_meta,
        "task_aliases": alias_map,
        "manifest": manifest_entries,
        "config": {
            "logprobs_top_k": logprobs_top_k,
            "max_concurrency": eval_cfg.get("max_concurrency", 4),
            "fewshot_seed": eval_cfg.get("fewshot_seed"),
        },
    }

    return {"requests": requests, "metadata": metadata}


def _reconstruct_tasks(metadata: Dict[str, Any]) -> Dict[str, TaskOutput]:
    task_names = metadata.get("tasks", [])
    if not task_names:
        return {}

    manager = TaskManager()
    task_dict = get_task_dict(task_names, task_manager=manager)

    eval_cfg = metadata.get("task_metadata", {})
    global_cfg = metadata.get("config", {})

    for task_name, info in eval_cfg.items():
        task = task_dict[task_name]
        applied = info.get("applied_settings", {})
        settings = dict(applied)
        if "apply_chat_template" not in settings:
            settings["apply_chat_template"] = False
        _configure_task(task, settings, {"limit": applied.get("limit"), "fewshot_seed": global_cfg.get("fewshot_seed")})

        if "fewshot_seed" in applied:
            task.set_fewshot_seed(int(applied["fewshot_seed"]))
        elif global_cfg.get("fewshot_seed") is not None:
            task.set_fewshot_seed(int(global_cfg["fewshot_seed"]))

        task.build_all_requests(
            limit=applied.get("limit"),
            cache_requests=False,
            rewrite_requests_cache=False,
            system_instruction=None,
            apply_chat_template=bool(applied.get("apply_chat_template", False)),
            fewshot_as_multiturn=False,
        )

    return task_dict


def _extract_loglikelihood(response: Dict[str, Any], instance) -> Tuple[Optional[Tuple[float, bool]], Optional[str]]:
    try:
        choice = response.get("choices", [])[0]
        logprobs = choice.get("logprobs") if isinstance(choice, dict) else None
        if not logprobs:
            return None, "missing_logprobs"

        offsets = logprobs.get("text_offset") or []
        token_logprobs = logprobs.get("token_logprobs") or []
        top_logprobs = logprobs.get("top_logprobs") or []

        context, continuation = instance.arguments
        context_len = len(context)

        total = 0.0
        is_greedy = True
        found = False
        for offset, lp, candidates in zip(offsets, token_logprobs, top_logprobs):
            if offset is None or lp is None or offset < context_len:
                continue
            found = True
            total += float(lp)
            if isinstance(candidates, dict) and candidates:
                best_lp = max(candidates.values())
                if lp < best_lp - 1e-6:
                    is_greedy = False
        if not found:
            return None, "no_continuation_tokens"
        return (total, is_greedy), None
    except Exception as exc:  # pragma: no cover
        return None, f"loglikelihood_parse_error:{exc}"


def _extract_generation(response: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    try:
        choice = response.get("choices", [])[0]
        if not isinstance(choice, dict):
            return None, "malformed_choice"
        if "message" in choice:
            content = choice.get("message", {}).get("content", "")
        else:
            content = choice.get("text", "")
        return content, None
    except Exception as exc:  # pragma: no cover
        return None, f"generation_parse_error:{exc}"



def score_results(
    requests: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    manifest = metadata.get("manifest", [])
    del requests  # Requests can be reconstructed from metadata

    if len(responses) != len(manifest):
        raise ValueError("Responses length does not match manifest")

    task_dict = _reconstruct_tasks(metadata)
    if not task_dict:
        return {"metrics": {}, "details": {"errors": ["no_tasks"]}}

    task_outputs = get_task_list(task_dict)
    errors: List[str] = []

    lookup: Dict[Tuple[str, int, int], Any] = {}
    for task_output in task_outputs:
        for inst in task_output.task.instances:
            lookup[(task_output.task_name, int(inst.doc_id), int(inst.idx))] = inst

    for entry, response in zip(manifest, responses):
        key = (entry["task"], entry["doc_id"], entry["instance_idx"])
        inst = lookup.get(key)
        if inst is None:
            errors.append(f"missing_instance:{key}")
            continue

        req_type = entry.get("request_type")
        if req_type == "loglikelihood":
            value, err = _extract_loglikelihood(response, inst)
        elif req_type == "generate_until":
            value, err = _extract_generation(response)
        else:
            errors.append(f"unsupported_request_type:{req_type}")
            continue

        if err:
            errors.append(f"{entry['task']}:{entry['doc_id']}:{err}")
            continue
        inst.resps.append(value)

    for key, inst in lookup.items():
        expected = max(1, int(getattr(inst, "repeats", 1) or 1))
        if len(inst.resps) < expected:
            errors.append(f"incomplete_responses:{key}:{len(inst.resps)}/{expected}")
        elif len(inst.resps) > expected:
            inst.resps = inst.resps[:expected]

    for task_output in task_outputs:
        task_output.task.apply_filters()

    for task_output in task_outputs:
        task = task_output.task
        if not task.instances:
            continue

        instances_by_doc = defaultdict(list)
        for inst in task.instances:
            instances_by_doc[inst.doc_id].append(inst)

        filter_keys = list(task.instances[0].filtered_resps.keys()) or ["none"]
        for filter_key in filter_keys:
            for doc_id, instances in instances_by_doc.items():
                try:
                    results = task.process_results(
                        instances[0].doc,
                        [inst.filtered_resps.get(filter_key, inst.resps) for inst in instances],
                    )
                except Exception as exc:
                    errors.append(f"{task_output.task_name}:{doc_id}:process_results_failed:{exc}")
                    continue
                for metric, value in results.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)
        task_output.calculate_aggregate_metric(bootstrap_iters=0)

    results, samples, configs, versions, num_fewshot, higher_is_better = consolidate_results(task_outputs)
    subtask_list = get_subtask_list(task_dict)
    if bool(results):
        results, versions, show_group_table, *_ = consolidate_group_results(results, versions, task_dict)
    else:
        show_group_table = False

    task_agg, _ = prepare_print_tasks(task_dict, results)

    alias_map = metadata.get("task_aliases", {})
    aggregated_aliases = {"mmlu_pro"}

    metrics: Dict[str, Any] = {}
    primary: Dict[str, float] = {}
    handled: set[str] = set()

    def _select_primary(metric_map: Dict[str, float]) -> Optional[str]:
        ordered = ["exact_match", "acc", "contains", "f1", "score"]
        for key in ordered:
            if key in metric_map:
                return key
        if metric_map:
            return next(iter(metric_map.keys()))
        return None

    for alias, children in alias_map.items():
        if alias in aggregated_aliases:
            totals: Dict[str, float] = {}
            weights: Dict[str, float] = {}
            for child in children:
                handled.add(child)
                data = task_agg.get(child, {})
                sample_count = float(data.get("samples", 0) or 0)
                for metric_name, value in data.items():
                    if metric_name in ("alias", "samples"):
                        continue
                    base = metric_name.split(",", 1)[0]
                    val_num = _ensure_numeric(value)
                    if not isinstance(val_num, (int, float)):
                        continue
                    val = float(val_num)
                    totals.setdefault(base, 0.0)
                    weights.setdefault(base, 0.0)
                    if sample_count > 0:
                        totals[base] += val * sample_count
                        weights[base] += sample_count
                    else:
                        totals[base] += val
                        weights[base] += 1.0
            aggregated_metrics: Dict[str, float] = {}
            for metric_name, total in totals.items():
                weight = weights.get(metric_name, 0.0)
                if weight <= 0:
                    continue
                aggregated_metrics[metric_name] = _ensure_numeric(total / weight)
                metrics[f"{alias}:{metric_name}"] = aggregated_metrics[metric_name]
            primary_key = _select_primary(aggregated_metrics)
            if primary_key is not None:
                primary[alias] = aggregated_metrics[primary_key]
        else:
            for child in children:
                handled.add(child)
                data = task_agg.get(child, {})
                child_metrics: Dict[str, float] = {}
                for metric_name, value in data.items():
                    if metric_name in ("alias", "samples"):
                        continue
                    base = metric_name.split(",", 1)[0]
                    val = _ensure_numeric(value)
                    metrics[f"{child}:{base}"] = val
                    child_metrics[base] = val
                primary_key = _select_primary(child_metrics)
                if primary_key is not None:
                    primary[child] = child_metrics[primary_key]

    remaining = set(task_agg.keys()) - handled
    for task_name in remaining:
        data = task_agg.get(task_name, {})
        child_metrics: Dict[str, float] = {}
        for metric_name, value in data.items():
            if metric_name in ("alias", "samples"):
                continue
            base = metric_name.split(",", 1)[0]
            val = _ensure_numeric(value)
            metrics[f"{task_name}:{base}"] = val
            child_metrics[base] = val
        primary_key = _select_primary(child_metrics)
        if primary_key is not None:
            primary[task_name] = child_metrics[primary_key]

    if primary:
        avg_score = _ensure_numeric(sum(primary.values()) / len(primary))
        metrics["avg"] = avg_score
        metrics["score"] = avg_score

    tasks_detail: Dict[str, Dict[str, int]] = {}
    for task_name, data in task_agg.items():
        samples_count = data.get("samples")
        tasks_detail[task_name] = {"n_docs": int(samples_count)} if samples_count is not None else {}

    details: Dict[str, Any] = {
        "tasks": tasks_detail,
    }
    grouped = [alias for alias in alias_map if alias in aggregated_aliases]
    if grouped:
        details["grouped"] = grouped
    if errors:
        details["errors"] = errors

    return {"metrics": metrics, "details": details}
