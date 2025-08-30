#!/usr/bin/env python3
"""
Plot NIAH performance from RULER outputs.

Generates:
- Accuracy vs context length (per task)
- Accuracy vs relative needle depth (for a selected length)

Inputs (created by scripts/launch.py or the legacy pipeline):
- Predictions: benchmark_root/<model>/<benchmark>/<seq>/pred/<task>.jsonl
- Data:        benchmark_root/<model>/<benchmark>/<seq>/data/<task>/validation.jsonl

Usage example:
  python scripts/plot/plot_niah.py \
    --root benchmark_root/<model>/synthetic \
    --niah_tasks niah_single_2,niah_multikey_1 \
    --seqs 4096,8192,16384,32768 \
    --out_dir plots
"""

import argparse
import json
import os
import unicodedata
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# Keep in sync with scripts/data/synthetic/constants.py for 'niah'
TOKENS_TO_GENERATE = 128


def norm(s: str) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", s).casefold().strip()


def load_jsonl(p: Path):
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def correctness_all(pred: str, refs):
    """All-match: every gold string must appear in prediction."""
    p = norm(pred)
    rlist = [norm(r) for r in refs]
    return 1.0 if all(r in p for r in rlist) else 0.0


def correctness_part(pred: str, refs):
    """Partial-match: any gold string appearing is success."""
    p = norm(pred)
    rlist = [norm(r) for r in refs]
    return 1.0 if any(r in p for r in rlist) else 0.0


def get_correctness_fn(task_name: str):
    # QA tasks use partial match; others (niah, vt, cwe, fwe) use all-match
    if task_name.startswith("qa") or task_name == "qa":
        return correctness_part
    return correctness_all


def plot_acc_vs_length(root: Path, tasks, seqs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        score_fn = get_correctness_fn(task)
        xs, ys = [], []
        for L in seqs:
            pred_file = root / str(L) / "pred" / f"{task}.jsonl"
            data_file = root / str(L) / "data" / task / "validation.jsonl"
            if not pred_file.exists() or not data_file.exists():
                print(f"[warn] Missing files for {task} @ {L}: {pred_file if pred_file.exists() else 'pred missing'}, {data_file if data_file.exists() else 'data missing'}")
                continue
            preds = load_jsonl(pred_file)
            total, correct = 0, 0
            for row in preds:
                pred_text = row.get("pred", "")
                refs = row.get("outputs", [])
                if not isinstance(refs, list) or not refs:
                    continue
                total += 1
                if score_fn(pred_text, refs):
                    correct += 1
            if total > 0:
                xs.append(L)
                ys.append(100.0 * correct / total)
        if xs:
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.title(f"{task} — Accuracy vs Context Length")
            plt.xlabel("Max sequence length (tokens)")
            plt.ylabel("Accuracy (%)")
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir / f"{task}_acc_vs_length.png", dpi=150, bbox_inches="tight")
            plt.close()
        else:
            print(f"[warn] No data to plot accuracy vs length for task {task} under {root}")


def plot_acc_vs_depth(root: Path, task: str, seq_len: int, out_dir: Path, bins: int = 10):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = root / str(seq_len) / "pred" / f"{task}.jsonl"
    data_file = root / str(seq_len) / "data" / task / "validation.jsonl"
    if not pred_file.exists() or not data_file.exists():
        print(f"[warn] Missing files for {task} @ {seq_len}: {pred_file if pred_file.exists() else 'pred missing'}, {data_file if data_file.exists() else 'data missing'}")
        return
    preds = {r["index"]: r for r in load_jsonl(pred_file)}
    data = load_jsonl(data_file)

    buckets = defaultdict(lambda: [0, 0])  # bin -> [correct, total]
    for row in data:
        idx = row.get("index")
        if idx not in preds:
            continue
        pred_text = preds[idx].get("pred", "")
        refs = preds[idx].get("outputs", [])
        length = row.get("length", 0)
        token_pos = row.get("token_position_answer", None)
        if token_pos is None or length <= TOKENS_TO_GENERATE:
            continue
        input_len = max(1, length - TOKENS_TO_GENERATE)
        rel = max(0.0, min(0.999, token_pos / input_len))  # 0..1
        bin_id = min(bins - 1, int(rel * bins))
        correct = correctness_niah(pred_text, refs)
        buckets[bin_id][0] += correct
        buckets[bin_id][1] += 1

    xs, ys = [], []
    for b in range(bins):
        c, t = buckets[b]
        xs.append((b + 0.5) * (100.0 / bins))  # mid of bin in %
        ys.append(100.0 * c / t if t else 0.0)

    if any(t for _, t in buckets.values()):
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.title(f"{task} — Accuracy vs Needle Depth @ {seq_len}")
        plt.xlabel("Relative depth in input (%)")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / f"{task}_acc_vs_depth_{seq_len}.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        print(f"[warn] No depth-binned data for {task} @ {seq_len} under {root}")


def plot_len_depth_heatmap(root: Path, task: str, seqs, out_dir: Path, bins: int = 20):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to seqs that actually have both data and preds for this task
    valid_seqs = []
    for L in seqs:
        pred_file = root / str(L) / "pred" / f"{task}.jsonl"
        data_file = root / str(L) / "data" / task / "validation.jsonl"
        if pred_file.exists() and data_file.exists():
            valid_seqs.append(L)
        else:
            print(f"[warn] Missing files for {task} @ {L}: {pred_file if pred_file.exists() else 'pred missing'}, {data_file if data_file.exists() else 'data missing'}")

    if not valid_seqs:
        print(f"[warn] Heatmap skipped for {task} under {root} — no lengths have both data and predictions")
        return

    acc = np.full((bins, len(valid_seqs)), np.nan, dtype=float)

    for ci, L in enumerate(valid_seqs):
        pred_file = root / str(L) / "pred" / f"{task}.jsonl"
        data_file = root / str(L) / "data" / task / "validation.jsonl"
        preds = {r["index"]: r for r in load_jsonl(pred_file)}
        data = load_jsonl(data_file)

        # bins: counts and totals
        cnt = np.zeros(bins, dtype=float)
        tot = np.zeros(bins, dtype=float)

        for row in data:
            idx = row.get("index")
            if idx not in preds:
                continue
            pred_text = preds[idx].get("pred", "")
            refs = preds[idx].get("outputs", [])
            length = row.get("length", 0)
            token_pos = row.get("token_position_answer", None)
            if token_pos is None or length <= TOKENS_TO_GENERATE:
                continue
            input_len = max(1, length - TOKENS_TO_GENERATE)
            rel = max(0.0, min(0.999, token_pos / input_len))
            bi = min(bins - 1, int(rel * bins))
            ok = correctness_all(pred_text, refs)
            cnt[bi] += ok
            tot[bi] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            acc[:, ci] = np.where(tot > 0, 100.0 * cnt / tot, np.nan)

    if np.all(np.isnan(acc)):
        print(f"[warn] Heatmap skipped for {task} under {root} — no valid bins across lengths {valid_seqs}")
        return

    plt.figure(figsize=(max(6, len(valid_seqs) * 0.6), 4.5))
    # Red (0%) to Green (100%) colormap (no deprecated get_cmap)
    cmap = mpl.colormaps['RdYlGn'].with_extremes(bad='#dddddd')
    # Grey-out missing bins via masked array
    acc_masked = np.ma.masked_invalid(acc)
    im = plt.imshow(acc_masked, origin='lower', aspect='auto', cmap=cmap,
                    vmin=0, vmax=100,
                    extent=[0, len(valid_seqs), 0, 100])
    plt.colorbar(im, label='Accuracy (%)')
    # X ticks at each seq index with labels of seqs
    plt.xticks(ticks=np.arange(len(valid_seqs)) + 0.5, labels=[str(s) for s in valid_seqs], rotation=45)
    plt.yticks(ticks=np.linspace(0, 100, 6), labels=[f"{int(v)}" for v in np.linspace(0, 100, 6)])
    plt.xlabel("Max sequence length (tokens)")
    plt.ylabel("Relative depth in input (%)")
    plt.title(f"{task} — Accuracy heatmap (depth × length)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{task}_heatmap_len_depth.png", dpi=150)
    plt.close()


def discover_base_dirs(root: Path) -> list[Path]:
    """Return list of base dirs that contain seq-length subdirs (e.g., <model>/<benchmark>)."""
    bases = []
    # Case 1: root is already a base (has numeric subdirs)
    numeric_kids = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
    if numeric_kids:
        bases.append(root)
    else:
        # Case 2: search one and two levels down
        for d1 in [d for d in root.iterdir() if d.is_dir()]:
            kids = [c for c in d1.iterdir() if c.is_dir() and c.name.isdigit()]
            if kids:
                bases.append(d1)
        if not bases:
            for d1 in [d for d in root.iterdir() if d.is_dir()]:
                for d2 in [c for c in d1.iterdir() if c.is_dir()]:
                    kids = [k for k in d2.iterdir() if k.is_dir() and k.name.isdigit()]
                    if kids:
                        bases.append(d2)
    return bases


def discover_seqs(base: Path) -> list[int]:
    seqs = []
    for d in base.iterdir():
        if d.is_dir() and d.name.isdigit() and (d / "pred").exists():
            try:
                seqs.append(int(d.name))
            except ValueError:
                pass
    return sorted(seqs)


def discover_tasks_for_seq(base: Path, seq: int) -> list[str]:
    data_dir = base / str(seq) / "data"
    pred_dir = base / str(seq) / "pred"
    tasks = set()
    if data_dir.exists():
        for d in data_dir.iterdir():
            if d.is_dir():
                tasks.add(d.name)
    if not tasks and pred_dir.exists():
        for f in pred_dir.glob("*.jsonl"):
            if f.name.startswith("summary"):
                continue
            tasks.add(f.stem)
    return sorted(tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="benchmark_root or benchmark_root/<model>/<benchmark>")
    ap.add_argument("--niah_tasks", default="", help="optional comma-list; auto-discovered if empty")
    ap.add_argument("--seqs", default="", help="optional comma-list; auto-discovered if empty")
    ap.add_argument("--out_dir", default="", help="optional; defaults to <base>/plots")
    ap.add_argument("--bins", type=int, default=20, help="depth bins for heatmap")
    args = ap.parse_args()

    root = Path(args.root)

    bases = discover_base_dirs(root)
    if not bases:
        print(f"No base directories with sequence subfolders found under {root}")
        return

    for base in bases:
        # Determine out_dir for this base
        if args.out_dir:
            model = base.parent.name
            benchmark = base.name
            out_dir = Path(args.out_dir) / f"{model}_{benchmark}"
        else:
            out_dir = base / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Seqs
        if args.seqs:
            seqs = [int(s) for s in args.seqs.split(",") if s.strip()]
        else:
            seqs = discover_seqs(base)
        if not seqs:
            print(f"No sequence lengths found under {base}")
            continue

        # Tasks
        if args.niah_tasks:
            tasks = [t.strip() for t in args.niah_tasks.split(",") if t.strip()]
        else:
            tasks = discover_tasks_for_seq(base, seqs[0])
        if not tasks:
            print(f"No tasks found under {base}/{seqs[0]}")
            continue

        # Plots
        plot_acc_vs_length(base, tasks, seqs, out_dir)
        for task in tasks:
            if task.startswith("niah_"):
                plot_len_depth_heatmap(base, task, seqs, out_dir, bins=args.bins)


if __name__ == "__main__":
    main()
