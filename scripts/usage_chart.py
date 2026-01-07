#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx",
#     "matplotlib",
# ]
# ///
"""
Fetch per-key usage for a date range and render a stacked bar chart of daily token
usage split by key, using the /monitor/key_usage endpoint.

Example:
    python scripts/usage_chart.py --start 2024-07-01 --end 2024-07-07 \
        --api-key $MONITOR_TOKEN --output data/usage.png
"""

import argparse
import datetime as dt
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import httpx
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


DATE_FMT = "%Y-%m-%d"
BUCKET_FMT = "%Y%m%d"
DEFAULT_BASE_URL = os.getenv("LLM_GATEWAY_BASE_URL", "http://localhost:8000")
DEFAULT_OUTPUT = "data/daily_token_usage.png"
DEFAULT_TOP_KEYS = 10


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, DATE_FMT).date()


def date_range(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    for offset in range((end - start).days + 1):
        yield start + dt.timedelta(days=offset)


def fetch_bucket(client: httpx.Client, bucket: str, limit: int = 500) -> List[Dict]:
    items: List[Dict] = []
    cursor = "0"
    while True:
        resp = client.get(
            "/monitor/key_usage", params={"bucket": bucket, "cursor": cursor, "limit": limit}
        )
        resp.raise_for_status()
        payload = resp.json() or {}
        items.extend(payload.get("items", []))
        cursor = str(payload.get("cursor", "0"))
        if cursor == "0":
            break
    return items


def collect_usage(
    client: httpx.Client, days: List[dt.date], top_keys: int
) -> Tuple[List[Tuple[dt.date, Dict[str, float]]], List[str], Counter]:
    per_day: List[Tuple[dt.date, Dict[str, float]]] = []
    totals = Counter()
    for day in days:
        bucket = day.strftime(BUCKET_FMT)
        entries = fetch_bucket(client, bucket)
        day_usage: Dict[str, float] = defaultdict(float)
        for entry in entries:
            key = entry.get("key_id") or "(unknown)"
            total_tokens = entry.get("total_tokens")
            if total_tokens is None:
                total_tokens = (entry.get("prompt_tokens") or 0.0) + (
                    entry.get("completion_tokens") or 0.0
                )
            day_usage[key] += float(total_tokens or 0.0)
        per_day.append((day, dict(day_usage)))
        totals.update(day_usage)
    key_order = [k for k, _ in totals.most_common(top_keys)]
    include_other = len(totals) > len(key_order)
    if include_other:
        key_order.append("Other")
    return per_day, key_order, totals


def format_tokens(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def plot_usage(
    per_day: List[Tuple[dt.date, Dict[str, float]]],
    key_order: List[str],
    totals: Counter,
    output_path: Path,
) -> None:
    labels = [d.strftime("%b %d") for d, _ in per_day]
    token_rows: List[List[float]] = []
    for day, usage in per_day:
        row: List[float] = []
        other_total = 0.0
        for idx, key in enumerate(key_order):
            if key == "Other":
                continue
            row.append(float(usage.get(key, 0.0)))
        if "Other" in key_order:
            listed = set(key_order) - {"Other"}
            other_total = sum(v for k, v in usage.items() if k not in listed)
            row.append(other_total)
        token_rows.append(row)

    palette = plt.get_cmap("tab20").colors
    base_color_count = len(palette)
    fig_width = max(8.0, 0.8 * len(per_day))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5), layout="constrained")
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#0f1724")

    bottoms = [0.0] * len(per_day)
    for idx, key in enumerate(key_order):
        values = [row[idx] for row in token_rows]
        color = palette[idx % base_color_count]
        ax.bar(
            labels,
            values,
            bottom=bottoms,
            label=key,
            linewidth=0.6,
            edgecolor="#0b0f14",
            color=color,
        )
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_title("Daily Token Usage by Key", color="#e8f0fb", fontsize=14, pad=16)
    ax.set_ylabel("Tokens", color="#c4cfdd")
    ax.tick_params(colors="#c4cfdd", labelsize=10)
    ax.yaxis.grid(True, color="#1c2a3b", alpha=0.6)
    ax.xaxis.grid(False)
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: format_tokens(x))
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1c2a3b")
    ax.spines["bottom"].set_color("#1c2a3b")

    legend_cols = 2 if len(key_order) > 8 else 1
    leg = ax.legend(
        ncol=legend_cols,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0, 1.12),
        labelcolor="#e8f0fb",
    )
    for text in leg.get_texts():
        text.set_color("#e8f0fb")

    for idx, total in enumerate(bottoms):
        if total <= 0:
            continue
        ax.text(
            idx,
            total + (total * 0.015),
            format_tokens(total),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#a6b7c9",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    print(f"Wrote chart to {output_path}")


def main() -> None:
    default_start = dt.date.today() - dt.timedelta(days=6)
    default_end = dt.date.today()
    parser = argparse.ArgumentParser(description="Render daily token usage per key.")
    parser.add_argument(
        "--start",
        type=parse_date,
        default=default_start,
        help=f"Start date (YYYY-MM-DD). Default: {default_start.strftime(DATE_FMT)}",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=default_end,
        help=f"End date inclusive (YYYY-MM-DD). Default: {default_end.strftime(DATE_FMT)}",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Gateway base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_GATEWAY_API_KEY") or os.getenv("LLM_GATEWAY_MONITOR_KEY"),
        help="Monitor-scope API key (env: LLM_GATEWAY_API_KEY or LLM_GATEWAY_MONITOR_KEY).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Where to write the PNG chart. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--top-keys",
        type=int,
        default=DEFAULT_TOP_KEYS,
        help=f"Show up to this many keys individually, group the rest as 'Other'. Default: {DEFAULT_TOP_KEYS}",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Drop days with zero total usage from the chart.",
    )
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("API key required (use --api-key or set LLM_GATEWAY_API_KEY).")
    if args.end < args.start:
        raise SystemExit("End date must be after start date.")

    days = list(date_range(args.start, args.end))
    headers = {"Authorization": f"Bearer {args.api_key}"}
    base_url = args.base_url.rstrip("/")
    with httpx.Client(base_url=base_url, headers=headers, timeout=20) as client:
        per_day, key_order, totals = collect_usage(client, days, args.top_keys)

    if args.skip_empty:
        per_day = [(d, usage) for d, usage in per_day if sum(usage.values()) > 0]
        if not per_day:
            raise SystemExit("No usage in the selected range.")

    if not totals:
        raise SystemExit("No usage found for the selected range.")

    output_path = Path(args.output)
    plot_usage(per_day, key_order, totals, output_path)

    print("Top keys by tokens:")
    for key, total in totals.most_common(10):
        print(f"  {key:24} {format_tokens(total)}")


if __name__ == "__main__":
    main()
