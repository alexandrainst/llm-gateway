#!/usr/bin/env python3
"""
Build DanishLongText.json from local text sources.

Options:
  - Provide a directory of .txt files (recursively) via --sources_dir.
  - Provide explicit --input_files paths.
Output:
  - scripts/data/synthetic/json/DanishLongText.json (key: "text").

Example:
  python build_danish_longtext.py --sources_dir /path/to/da_wiki_extracted
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, List


def read_text_file(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def read_json_lines_file(path: Path) -> str:
    """Read WikiExtractor --json output file (JSON per line) and concatenate 'text' fields."""
    texts = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    t = obj.get("text", "")
                    if t:
                        texts.append(t)
                except Exception:
                    continue
    except Exception:
        return ""
    return "\n\n".join(texts)


def normalize_text(s: str) -> str:
    # collapse whitespace, keep sentence punctuation
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def collect_texts(sources_dir: Optional[Path], input_files: List[Path], max_words: Optional[int] = None) -> str:
    pieces = []
    files = []
    if sources_dir and sources_dir.exists():
        # Accept any regular file (WikiExtractor often uses files with no extension)
        files = [p for p in sources_dir.rglob("*") if p.is_file()]
    files += [p for p in input_files if p.is_file()]

    total_words = 0
    for p in files:
        t = ""
        # Prefer JSONL parsing when likely JSON
        if p.suffix == ".json" or (p.stat().st_size > 0 and 'wiki' in p.name and p.suffix == ""):
            t = read_json_lines_file(p)
        if not t:
            t = read_text_file(p)
        if not t:
            continue

        if max_words is not None and max_words > 0:
            words = t.split()
            if total_words + len(words) > max_words:
                need = max_words - total_words
                if need > 0:
                    pieces.append(" ".join(words[:need]))
                    total_words += need
                break
            else:
                pieces.append(t)
                total_words += len(words)
        else:
            pieces.append(t)

    combined = normalize_text("\n\n".join(pieces))
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources_dir", type=Path)
    parser.add_argument("--input_files", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "DanishLongText.json")
    parser.add_argument("--max_words", type=int, default=0, help="Cap output to first N words (0 = no cap)")
    args = parser.parse_args()

    text = collect_texts(args.sources_dir, args.input_files, max_words=(args.max_words if args.max_words else None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"text": text}, f, ensure_ascii=False)
    print(f"Wrote {args.output} ({len(text)} chars)")


if __name__ == "__main__":
    main()
