import os
import json
from typing import Any

CHAT_TEMPLATE_FILENAMES = ("chat_template.jinja",)


def _load_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _looks_like_instruct_from_tokens(cfg: dict[str, Any]) -> bool:
    """Deprecated: no longer used. Kept for compatibility if needed later."""
    return False


def _ensure_tokenizer_files(model_id: str) -> None:
    """Download minimal tokenizer/config files into the shared HF cache so detection can proceed
    without fetching full weights. Safe to call multiple times (uses cache).
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return
    # Always use the container-side mount path for the HF cache
    cache_root = "/host_hf_cache"
    token = os.getenv("HF_TOKEN")
    # Only fetch tokenizer_config.json and chat_template files; detection relies on them
    allow = [
        "tokenizer_config.json",
        "*/tokenizer_config.json",
        "chat_template.jinja",
        "*/chat_template.jinja",
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


def _has_standalone_chat_template(snapshot_dir: str) -> bool:
    for name in CHAT_TEMPLATE_FILENAMES:
        path = os.path.join(snapshot_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                if f.read().strip():
                    return True
        except Exception:
            continue
    return False


def detect_is_base_model(model_id: str) -> bool:
    """Detect base vs instruction using a single rule:

    - If tokenizer_config.json contains a non-empty "chat_template" in any cached snapshot,
      treat as instruction (return False).
    - Otherwise, treat as base (return True).

    If the cache is missing, we attempt to fetch tokenizer_config.json; on failure we still
    return True (base) per the simplified rule.
    """
    try:
        # Always use the container-side mount path for the HF cache
        cache_root = "/host_hf_cache"
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

        # Look for explicit chat_template in tokenizer_config.json
        for snap in candidates:
            cfg = _load_json(os.path.join(snap, "tokenizer_config.json"))
            if cfg is None:
                cfg = {}
            tmpl = cfg.get("chat_template")
            if isinstance(tmpl, str) and tmpl.strip():
                return False  # instruct/chat model
            if _has_standalone_chat_template(snap):
                return False

        # No instruct signal found in available snapshots.
        if os.path.isdir(snapshots_dir):
            return True

        # As a last resort, try to fetch minimal tokenizer and test once more
        _ensure_tokenizer_files(model_id)
        if os.path.isdir(snapshots_dir):
            # Re-run quickly without recursion to avoid deep calls
            main_dir = os.path.join(snapshots_dir, "main")
            snaps = [main_dir] if os.path.isdir(main_dir) else []
            for name in os.listdir(snapshots_dir):
                if name != "main":
                    p = os.path.join(snapshots_dir, name)
                    if os.path.isdir(p):
                        snaps.append(p)
            for snap in snaps:
                cfg = _load_json(os.path.join(snap, "tokenizer_config.json"))
                if cfg and isinstance(cfg.get("chat_template"), str) and cfg.get("chat_template").strip():
                    return False
                if _has_standalone_chat_template(snap):
                    return False
            return True
        # Cache still missing; default to base per simplified rule
        return True
    except Exception:
        # On any error, assume base per simplified rule.
        return True
