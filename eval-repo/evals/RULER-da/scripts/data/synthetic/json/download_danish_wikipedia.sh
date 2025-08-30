#!/usr/bin/env bash
# Download and prepare Danish Wikipedia into DanishLongText.json
# Requirements: wget, python3, pip, wikiextractor

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_JSON="${THIS_DIR}/DanishLongText.json"
WORK_DIR="${THIS_DIR}/_dawiki_work"
mkdir -p "${WORK_DIR}"

URL="${DUMP_URL:-https://dumps.wikimedia.org/dawiki/latest/dawiki-latest-pages-articles.xml.bz2}"

if [ ! -f "${WORK_DIR}/dawiki-latest-pages-articles.xml.bz2" ]; then
  echo "Downloading Danish Wikipedia pages-articles dump..."
  wget -O "${WORK_DIR}/dawiki-latest-pages-articles.xml.bz2" "${URL}"
else
  echo "Using existing dump at ${WORK_DIR}/dawiki-latest-pages-articles.xml.bz2"
fi

echo "Extracting articles using wikiextractor..."
python3 -m wikiextractor.WikiExtractor \
  --json \
  --processes 4 \
  --no-templates \
  --output "${WORK_DIR}/extracted" \
  "${WORK_DIR}/dawiki-latest-pages-articles.xml.bz2"

echo "Concatenating extracted text into DanishLongText.json..."
if [ -n "${MAX_WORDS:-}" ]; then
  python3 "${THIS_DIR}/build_danish_longtext.py" \
    --sources_dir "${WORK_DIR}/extracted" \
    --output "${OUT_JSON}" \
    --max_words "${MAX_WORDS}"
else
  python3 "${THIS_DIR}/build_danish_longtext.py" \
    --sources_dir "${WORK_DIR}/extracted" \
    --output "${OUT_JSON}"
fi

if [ "${SKIP_CLEANUP:-0}" = "1" ]; then
  echo "Skipping cleanup per SKIP_CLEANUP=1; work dir left at ${WORK_DIR}"
else
  echo "Cleaning up work directory..."
  rm -rf "${WORK_DIR}"
fi

echo "Done: ${OUT_JSON}"
