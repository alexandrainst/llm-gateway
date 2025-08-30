#!/usr/bin/env python3
"""
Build danish_words.json from a large Danish text (e.g., DanishLongText.json).

Reads:
  --longtext_json scripts/data/synthetic/json/DanishLongText.json (default)
Writes:
  scripts/data/synthetic/json/danish_words.json as a dict of index->word
  sorted by frequency (most frequent first), with basic filtering.

Example:
  python build_danish_words.py \
    --longtext_json scripts/data/synthetic/json/DanishLongText.json \
    --min_len 2 --min_freq 3
"""
import argparse
import collections
import json
import os
import re
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    toks = WORD_RE.findall(text)
    return [t.casefold() for t in toks]


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).parent
    parser.add_argument("--longtext_json", type=Path, default=base / "DanishLongText.json")
    parser.add_argument("--output", type=Path, default=base / "danish_words.json")
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--min_freq", type=int, default=1)
    args = parser.parse_args()

    with open(args.longtext_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = data.get("text", "")

    tokens = tokenize(text)
    cnt = collections.Counter(t for t in tokens if len(t) >= args.min_len)

    words = [w for w, c in cnt.most_common() if c >= args.min_freq]
    # Build dict of index->word for compatibility with existing loaders
    out = {str(i): w for i, w in enumerate(words)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Wrote {args.output} with {len(words)} words")


if __name__ == "__main__":
    main()

