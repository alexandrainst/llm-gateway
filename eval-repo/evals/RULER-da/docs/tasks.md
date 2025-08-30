# RULER Synthetic Tasks (Danish Adaptation)

This document describes the evaluation tasks and their variants used in the Danish RULER setup. It summarizes each task’s goal, input/output format, configuration knobs, and the specific variants enabled in `scripts/synthetic.yaml`.

## Conventions
- Language: All prompts are in Danish; “essay” haystacks draw from `DanishLongText.json`.
- Token budgeting: Dataset builders estimate input token counts with the selected tokenizer and fit the context to `max_seq_length` by binary search.
- Depth sampling (NIAH): Needles are inserted at 40 relative depths across the document, enabling depth–accuracy analysis.
- Evaluation metrics: Most tasks use string containment of normalized, case‑folded text.
  - Retrieval/aggregation tasks: all‑match (every gold string must appear).
  - QA: partial‑match (any one gold answer appearing is enough).

## Needle‑in‑a‑Haystack (NIAH)
- Purpose: Long‑range retrieval — recover value(s) associated with a key (or keys) hidden among distractors.
- Input structure:
  - Context: a haystack with one or more inserted “needle” sentences of the form:
    - `Et af de særlige magiske {type_needle_v} for {key} er: {value}.`
    - Visible labels are localized (e.g., `numbers → tal`, `words → ord`, `uuids → UUID’er`).
  - Query: the key (or comma‑separated keys) to retrieve values for.
- Output: All values associated with the queried key(s), as substrings.
- Difficulty controls:
  - Haystack type: `noise` (repeated neutral sentence), `essay` (DanishLongText excerpts), or `needle` (repeated needle lines as distractors).
  - Key/Value types: `words` (hyphenated DA word pairs), `numbers` (7‑digit), `uuids`.
  - Multiplicity: number of keys, values per key, and queried keys.
- Metric: all‑match (every gold value must appear in the model’s answer).

### Variants (from `scripts/synthetic.yaml`)

- niah_single_1
  - haystack: `noise`
  - key type: `words`
  - value type: `numbers`
  - num_needle_k: 1, num_needle_v: 1, num_needle_q: 1

- niah_single_2
  - haystack: `essay`
  - key type: `words`
  - value type: `numbers`
  - num_needle_k: 1, num_needle_v: 1, num_needle_q: 1

- niah_single_3
  - haystack: `essay`
  - key type: `words`
  - value type: `uuids`
  - num_needle_k: 1, num_needle_v: 1, num_needle_q: 1

- niah_multikey_1
  - haystack: `essay`
  - key type: `words`
  - value type: `numbers`
  - num_needle_k: 4, num_needle_v: 1, num_needle_q: 1

- niah_multikey_2
  - haystack: `needle` (distractor needles)
  - key type: `words`
  - value type: `numbers`
  - num_needle_k: 1, num_needle_v: 1, num_needle_q: 1

- niah_multikey_3
  - haystack: `needle` (distractor needles)
  - key type: `uuids`
  - value type: `uuids`
  - num_needle_k: 1, num_needle_v: 1, num_needle_q: 1

- niah_multivalue
  - haystack: `essay`
  - key type: `words`
  - value type: `numbers`
  - num_needle_k: 1, num_needle_v: 4 (multiple values per key), num_needle_q: 1

- niah_multiquery
  - haystack: `essay`
  - key type: `words`
  - value type: `numbers`
  - num_needle_k: 1, num_needle_v: 1, num_needle_q: 4 (query multiple keys)

Notes
- Keys of type `words` are generated as hyphenated Danish word pairs and checked for uniqueness.
- Values of type `numbers` and `uuids` are language‑neutral and minimize accidental collisions.

## Variable Tracking (vt)
- Purpose: Multi‑hop tracing — track aliasing chains and list all variables that end up with a target value.
- Input structure:
  - Context: noise or essay text with `num_chains` aliasing chains inserted at sampled depths. Example (one chain, 4 hops):
    - `VAR A = 12345. VAR B = VAR A. VAR C = VAR B. VAR D = VAR C.`
  - Query: the concrete value (e.g., `12345`).
- Output: All variable names in the chain (`A, B, C, D`).
- Variant:
  - vt
    - haystack: `noise`
    - num_chains: 1
    - num_hops: 4
- Difficulty controls: more chains, more hops, essay haystack, optional few‑shot ICL.
- Metric: all‑match.

## Common Words Extraction (cwe)
- Purpose: Aggregation — identify the most frequent words in a long, numbered list.
- Input structure:
  - Numbered list where a subset of “common” words repeats more frequently than the rest.
- Output: The top `num_cw` most frequent words.
- Variant:
  - cwe
    - freq_cw: 30 (repeats of common words)
    - freq_ucw: 3 (repeats of uncommon words)
    - num_cw: 10 (return top‑10)
- Metric: all‑match.

## Frequent Words Extraction (fwe)
- Purpose: Aggregation over synthetic tokens — find the most common items in a Zipf‑like distribution.
- Input structure:
  - Synthetic lowercase tokens from a fixed‑length vocabulary; counts drawn from a zeta distribution with parameter `alpha`.
  - Instruction to ignore `...` noise tokens.
- Output: The top 3 tokens by frequency.
- Variant:
  - fwe
    - alpha: 2.0 (skew; lower = harder)
- Metric: all‑match.

## Question Answering (qa)
- Purpose: Multi‑document retrieval + extractive answering.
- Input structure:
  - A set of documents (target + distractors) rendered as `Dokument i: ...`, followed by a question.
- Output: The short answer string (exact substring in one doc).
- Variants:
  - qa_1 — dataset: `squad`
  - qa_2 — dataset: `hotpotqa`
- Metric: partial‑match (any one gold answer appearing).
- Note: For a pure Danish setup, replace with Danish QA in the same schemas.

## JSONL Fields (datasets and predictions)
- Dataset JSONL (data/<task>/validation.jsonl):
  - `index`: integer id within the file.
  - `input`: the rendered prompt context (without `answer_prefix`).
  - `outputs`: list of gold strings (one or more).
  - `length`: tokenized input length (per selected tokenizer) + `tokens_to_generate`.
  - `answer_prefix`: the answer prefix stripped from `input` (present for most synthetic tasks).
  - `token_position_answer`: token index of the first answer occurrence (for NIAH; used in depth plots).
- Prediction JSONL (pred/<task>.jsonl):
  - `index`: mirrors dataset index.
  - `input`: original input text.
  - `outputs`: gold list.
  - `pred`: model output string.
  - `others`, `truncation`, `length`: optional metadata.

## Plots
- Accuracy vs length: One curve per task across configured sequence lengths.
- Heatmap (NIAH): Accuracy by relative needle depth (0–100%) vs length. Red = low, Green = high; missing bins shown in grey.

