#!/usr/bin/env python3
"""
Summarize scores across context lengths.

Reads each <base>/<len>/pred/summary.csv, computes the macro-average across tasks at that length,
and then reports:
- Avg (unweighted mean over lengths)
- wAvg (inc) (linearly weighted toward longer lengths)

Writes <base>/summary-lengths.csv with:
- Row 1: Lengths, <L1>, <L2>, ...
- Row 2: AvgPerLength, <s1>, <s2>, ... (macro-average across tasks at each length)
- Row 3: OverallAvg, <avg>
- Row 4: wAvgInc, <wavg_inc>

Usage:
  python scripts/eval/summarize_lengths.py --base_dir benchmark_root/<model>/synthetic
"""

import argparse
import csv
import os
import re
from pathlib import Path


def read_length_avg(summary_csv: Path) -> float | None:
    if not summary_csv.exists():
        return None
    with open(summary_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None
    # Row[1] expected: ["Score", <task1>, <task2>, ...]
    try:
        scores = [float(x) for x in rows[1][1:]]
    except Exception:
        return None
    if not scores:
        return None
    return sum(scores) / len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="Path like benchmark_root/<model>/<benchmark>")
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        raise SystemExit(f"Base dir not found: {base}")

    # Discover numeric length dirs
    length_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda p: int(p.name)
    )
    if not length_dirs:
        raise SystemExit(f"No numeric length directories under {base}")

    per_len = []  # (length:int, avg:float)
    for d in length_dirs:
        summary = d / "pred" / "summary.csv"
        avg = read_length_avg(summary)
        if avg is None:
            print(f"[warn] Missing or invalid summary at {summary}")
            continue
        per_len.append((int(d.name), avg))

    if not per_len:
        raise SystemExit("No per-length averages found. Run evaluate first.")

    lengths = [L for L, _ in per_len]
    avgs = [s for _, s in per_len]

    # Overall averages over available lengths
    overall_avg = sum(avgs) / len(avgs)
    # wAvg (inc): weights 1..n
    n = len(avgs)
    weights = list(range(1, n + 1))
    wavg_inc = sum(a * w for a, w in zip(avgs, weights)) / sum(weights)

    out = base / "summary-lengths.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Lengths", *lengths])
        w.writerow(["AvgPerLength", *[f"{x:.2f}" for x in avgs]])
        w.writerow(["OverallAvg", f"{overall_avg:.2f}"])
        w.writerow(["wAvgInc", f"{wavg_inc:.2f}"])

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

