#!/bin/bash
# Download WikiSQL dataset from GitHub
# Usage: bash scripts/download_wikisql.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

echo "==> Downloading WikiSQL dataset..."

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ -d "data" ] && [ -f "data/train.jsonl" ]; then
    echo "WikiSQL data already exists in $DATA_DIR/data/"
    echo "To re-download, remove the data/data/ directory first."
    exit 0
fi

echo "Downloading from GitHub..."
curl -L -o wikisql.tar.bz2 "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"

echo "Extracting..."
tar -xjf wikisql.tar.bz2

echo "Cleaning up..."
rm wikisql.tar.bz2

echo "==> WikiSQL dataset downloaded to $DATA_DIR/data/"
echo "Files:"
ls -lh data/

echo ""
echo "Dataset sizes:"
echo "  Train: $(wc -l < data/train.jsonl) examples"
echo "  Dev:   $(wc -l < data/dev.jsonl) examples"
echo "  Test:  $(wc -l < data/test.jsonl) examples"
