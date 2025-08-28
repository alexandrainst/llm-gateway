import os
import json

def detect_is_base_model(model_id: str) -> bool:
    """Detect if a model is a base (no chat template) by inspecting tokenizer_config.json
    in the local HF cache for both uploaded (snapshots/main) and HF-downloaded (snapshots/<hash>) cases.

    Returns True for base models (no chat_template), False for instruction/chat models.
    Defaults to True (base) if undeterminable.
    """
    try:
        base_dir = os.path.join("/host_hf_cache", "hub", f"models--{model_id.replace('/', '--')}")
        snapshots_dir = os.path.join(base_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return True  # default conservative

        # Prefer uploads path first
        candidates = []
        main_dir = os.path.join(snapshots_dir, "main")
        if os.path.isdir(main_dir):
            candidates.append(main_dir)

        # Include hashed snapshot dirs sorted by mtime desc
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

        for snap in candidates:
            tok_cfg = os.path.join(snap, "tokenizer_config.json")
            if os.path.exists(tok_cfg):
                try:
                    with open(tok_cfg, "r") as f:
                        cfg = json.load(f)
                    # Base model if chat_template missing or None
                    return cfg.get("chat_template") is None
                except Exception:
                    continue

        # Not found / unreadable
        return True
    except Exception:
        return True
