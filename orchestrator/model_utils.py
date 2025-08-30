import os
import json
from typing import Any


def _load_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _looks_like_instruct_from_tokens(cfg: dict[str, Any]) -> bool:
    """Heuristic: presence of common chat special tokens implies an instruct/chat model."""
    if not isinstance(cfg, dict):
        return False
    # tokenizer_config.json may expose these under various keys
    specials = []
    # transformers saves additional_special_tokens as a list
    specials.extend(cfg.get("additional_special_tokens", []) or [])
    # Sometimes nested under special_tokens_map
    stm = cfg.get("special_tokens_map", {}) or {}
    specials.extend(stm.get("additional_special_tokens", []) or [])
    # tokenizer.json often contains an added_tokens_decoder mapping
    added = cfg.get("added_tokens_decoder") or {}
    if isinstance(added, dict):
        specials.extend([v.get("content") for v in added.values() if isinstance(v, dict)])
    # Known chat tokens/templates
    chat_markers = [
        "<|im_start|>", "<|im_end|>", "<|system|>", "<|user|>", "<|assistant|>",
        "[INST]", "[/INST]", "### Instruction", "### Response", "USER:", "ASSISTANT:",
    ]
    specials_str = "\n".join([s for s in specials if isinstance(s, str)])
    return any(marker in specials_str for marker in chat_markers)


def _ensure_tokenizer_files(model_id: str) -> None:
    """Download minimal tokenizer/config files into the shared HF cache so detection can proceed
    without fetching full weights. Safe to call multiple times (uses cache).
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return
    cache_root = os.getenv("HOST_HF_CACHE", "/host_hf_cache")
    token = os.getenv("HF_TOKEN")
    allow = [
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "added_tokens.json",
        "generation_config.json",
        "*/tokenizer_config.json",
        "*/tokenizer.json",
        "*/generation_config.json",
    ]
    try:
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_root,
            token=token,
            local_dir_use_symlinks=False,
            allow_patterns=allow,
            resume_download=True,
        )
    except Exception:
        # Ignore download errors here; detection will fall back to name heuristics
        return


def detect_is_base_model(model_id: str) -> bool:
    """Detect base vs. instruction model from local HF cache.

    Strategy (ordered):
    1) If any snapshot contains a non-empty chat_template (tokenizer_config.json or generation_config.json) → instruct (return False).
    2) If tokenizer.json/… show common chat special tokens → instruct (return False).
    3) If model_id name strongly suggests instruct (e.g., "-Instruct", "-it", "-chat") → instruct (return False).
    4) If we still can't tell, default to instruct (return False) rather than base to avoid forcing chatless flows.

    Returns True only when we can reasonably infer a base model (explicit chat_template is missing/null across snapshots and no instruct markers found).
    """
    try:
        cache_root = os.getenv("HOST_HF_CACHE", "/host_hf_cache")
        base_dir = os.path.join(cache_root, "hub", f"models--{model_id.replace('/', '--')}")
        snapshots_dir = os.path.join(base_dir, "snapshots")
        candidates: list[str] = []
        if os.path.isdir(snapshots_dir):
            # Prefer uploads snapshot "main"
            main_dir = os.path.join(snapshots_dir, "main")
            if os.path.isdir(main_dir):
                candidates.append(main_dir)
            # Then hashed snapshots, newest first
            hashed = []
            for name in os.listdir(snapshots_dir):
                if name == "main":
                    continue
                p = os.path.join(snapshots_dir, name)
                if os.path.isdir(p):
                    try:
                        hashed.append((os.path.getmtime(p), p))
                    except Exception:
                        hashed.append((0, p))
            hashed.sort(key=lambda x: x[0], reverse=True)
            candidates.extend([p for _, p in hashed])
        else:
            # No cache yet: download minimal tokenizer/config files and retry
            _ensure_tokenizer_files(model_id)
            if os.path.isdir(snapshots_dir):
                main_dir = os.path.join(snapshots_dir, "main")
                if os.path.isdir(main_dir):
                    candidates.append(main_dir)
                hashed = []
                for name in os.listdir(snapshots_dir):
                    if name == "main":
                        continue
                    p = os.path.join(snapshots_dir, name)
                    if os.path.isdir(p):
                        try:
                            hashed.append((os.path.getmtime(p), p))
                        except Exception:
                            hashed.append((0, p))
                hashed.sort(key=lambda x: x[0], reverse=True)
                candidates.extend([p for _, p in hashed])

        # 1) Look for explicit chat_template first
        for snap in candidates:
            for fname in ("tokenizer_config.json", "generation_config.json"):
                cfg = _load_json(os.path.join(snap, fname))
                if cfg is None:
                    continue
                if "chat_template" in cfg:
                    tmpl = cfg.get("chat_template")
                    if isinstance(tmpl, str) and tmpl.strip():
                        return False  # instruct/chat model
                    # If explicitly null/empty, keep checking other signals

        # 2) Heuristics from tokens in tokenizer.json or tokenizer_config.json
        for snap in candidates:
            tok_cfg = _load_json(os.path.join(snap, "tokenizer_config.json")) or {}
            if _looks_like_instruct_from_tokens(tok_cfg):
                return False
            tok_json = _load_json(os.path.join(snap, "tokenizer.json")) or {}
            if _looks_like_instruct_from_tokens(tok_json):
                return False

        # 3) Name heuristics
        name = model_id.lower()
        instruct_markers = ["instruct", "-it", "-chat", "-assistant", "-sft", "-qa"]
        if any(m in name for m in instruct_markers):
            return False

        # If we have HF cache and still couldn't confirm instruct, tentatively assume base.
        if os.path.isdir(snapshots_dir):
            return True
        # As a last resort, try to fetch minimal tokenizer and test once more
        _ensure_tokenizer_files(model_id)
        if os.path.isdir(snapshots_dir):
            # Recurse once to re-run detection on newly downloaded configs
            try:
                return detect_is_base_model(model_id)
            except Exception:
                pass
        return False
    except Exception:
        # On any error, assume instruct to avoid under-detecting chat models.
        return False
