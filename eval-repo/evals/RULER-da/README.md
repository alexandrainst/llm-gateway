# RULER (Danish Adaptation) — Long‑Context Evaluation Made Simple

This repository provides a modernized, Danish‑first adaptation of the RULER benchmark for probing the effective context length of instruction‑tuned LLMs. It generates synthetic tasks on demand, calls your model via an OpenAI‑compatible Chat API (e.g., vLLM’s OpenAI server), and evaluates results with Unicode‑aware matching.

Important: The large model results table from the original RULER README does not apply here. The codebase, tasks, language, and evaluation flow have been changed. See “Citation & Acknowledgements” for the original project.


## Quick Start
- Serve a model via an OpenAI‑compatible endpoint (e.g., vLLM):
  - `python -m vllm.entrypoints.openai.api_server --model <hf-repo-or-path> --port 8000 --dtype bfloat16`
  - `export OPENAI_BASE_URL=http://127.0.0.1:8000/v1`
  - `export OPENAI_API_KEY=dummy`

- Prepare Danish long text (once):
  - Option A: Danish Wikipedia subset
    - `bash scripts/data/synthetic/json/download_danish_wikipedia.sh`  (requires `wget` and `pip install wikiextractor`)
    - Optional cap: `MAX_WORDS=250000 bash scripts/data/synthetic/json/download_danish_wikipedia.sh`
  - Option B: Local .txt sources
    - `python scripts/data/synthetic/json/build_danish_longtext.py --sources_dir /path/to/txts --max_words 250000`
  - Build Danish word pool from the long text:
    - `python scripts/data/synthetic/json/build_danish_words.py --longtext_json scripts/data/synthetic/json/DanishLongText.json --min_len 2 --min_freq 1`

- Run the unified launcher (discovers model, prepares on demand, predicts, evaluates):
  - `python scripts/launch.py --benchmark synthetic --seq-lengths 4096,8192,16384,32768 --num-samples 200 --exclude-qa`
  - Options:
    - `--model <id>`: override autodiscovery from `/v1/models`.
    - `--model_local_path </path/to/local/hf/model>`: use the local HF tokenizer for exact token budgeting.
    - `--tasks <comma-list>`: limit tasks; default is all synthetic tasks except QA when `--exclude-qa` is set.
    - `--save_root <dir>`: change output root (default: `benchmark_root`).

- Plot results (auto‑discovers models, seq lengths, and tasks):
  - `python scripts/plot/plot_niah.py --root benchmark_root`
  - Outputs per base (e.g., `benchmark_root/<model>/synthetic/plots/`):
    - All tasks: `<task>_acc_vs_length.png`
    - NIAH only: `<task>_heatmap_len_depth.png` (red=0%, green=100%, grey=missing)
  - After launch: the script also writes an aggregated `summary-lengths.csv` at the base (`benchmark_root/<model>/synthetic/`) with:
    - `Lengths`: the evaluated context lengths
    - `AvgPerLength`: macro‑average across tasks at each length
    - `OverallAvg`: unweighted average over lengths
    - `wAvgInc`: weighted average favoring longer contexts


## What’s Included
- Danish prompts and noise text; “essay” haystacks draw from `scripts/data/synthetic/json/DanishLongText.json`.
- Unicode‑aware evaluation (NFKC + casefold) for robust matching with æ/ø/å.
- HF and sentencepiece tokenizers; NeMo removed. Token budgets are computed using HF tokenizers and, when available, their chat templates (for counting only). Chat formatting is applied server‑side by your model’s tokenizer.
- OpenAI‑compatible client with async concurrency (default 100 via `OPENAI_CONCURRENCY`) — no custom `/generate` server.
- On‑demand data preparation: builders binary‑search haystack size to fit `max_seq_length`.
- Plotting utilities for NIAH (length curve + depth×length heatmap) and accuracy‑vs‑length for all tasks.


## Tasks
- Synthetic tasks designed to probe long‑context behavior:
  - Needle‑in‑a‑Haystack (NIAH): long‑range retrieval with depth control and distractors.
  - Variable Tracking (vt): multi‑hop alias tracing dispersed in the context.
  - Common Words Extraction (cwe): frequency aggregation across long lists.
  - Frequent Words Extraction (fwe): Zipf‑like synthetic token aggregation with noise.
  - Question Answering (qa): multi‑document, extractive QA (add Danish sets when ready).
- See `docs/tasks.md` for full details and all NIAH variants configured in `scripts/synthetic.yaml`.


## Output Layout
- Data: `benchmark_root/<model>/<benchmark>/<seq>/data/<task>/validation.jsonl`
- Predictions: `benchmark_root/<model>/<benchmark>/<seq>/pred/<task>.jsonl`
- Evaluation CSV: in the corresponding `pred/` folder
- Plots: `<base>/plots/` (or a global dir via `--out_dir`)


## Configuration & Environment
- Required env for the client:
  - `OPENAI_BASE_URL` (or `OPENAI_API_BASE`): e.g., `http://127.0.0.1:8000/v1`
  - `OPENAI_API_KEY`: non‑empty string (local servers typically ignore it)
  - `OPENAI_CONCURRENCY` (optional): async concurrency (default: 100)
- Decoding defaults (exposed as flags):
  - temperature: 0.0 (greedy), top_p: 1.0, tokens_to_generate per task


## Modernization Highlights
- Chat templates are applied server‑side (OpenAI Chat API). Manual prompt wrappers have been removed.
- Token budgets are computed with HF tokenizers; if a chat template is exposed, it is applied for counting only.
- No NeMo dependency: manifest utils are local; sentencepiece supports raw `.model` tokenizers.
- Legacy scripts (`run.sh`, `config_models.sh`, custom vLLM server code) are deprecated and not required for the modern flow.


## Building Danish Resources
- Long text: `scripts/data/synthetic/json/DanishLongText.json` (provided placeholder; replace with your corpus or use the Wikipedia script).
- Word pool: `scripts/data/synthetic/json/danish_words.json` built from the long text.
- Optional curated lists: place `da_nouns.txt`, `da_adjectives.txt`, `da_verbs.txt` under `scripts/data/synthetic/resources/` (UTF‑8, one per line). Generators will pick them up automatically; otherwise they use `danish_words.json` with safeguards.


## Known Limitations / TODO
- QA: English SQuAD/Hotpot loaders remain; add Danish equivalents in the same schemas and enable with `--tasks qa_1,qa_2`.
- Token counting: HF chat template counting is used for budgeting only; actual formatting is server‑side. This is accurate in practice but not byte‑for‑byte identical for all models.
- Packaging: The code runs as a repo; turning it into an installable package would ease imports and testing.
- Tests: No unit tests yet; adding tokenizer and end‑to‑end smoke tests would improve robustness.


## Citation & Acknowledgements
- Original paper: Hsieh et al., “RULER: What’s the Real Context Size of Your Long‑Context Language Models?” arXiv:2404.06654 — https://arxiv.org/abs/2404.06654
- This adaptation draws on the original RULER codebase by NVIDIA, but has been significantly changed (language, flow, clients, and utils). It is strictly for research purposes.
