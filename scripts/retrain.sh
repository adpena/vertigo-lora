#!/usr/bin/env bash
# Vertigo LoRA incremental retrain
# Usage: ./scripts/retrain.sh
#
# Full pipeline: re-extract → validate → dedup → merge → train
# Each run produces a new timestamped adapter. Previous adapters are preserved.
# Run this after updating Vertigo's source code or adding new raw training data.

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="$(date +%Y%m%d_%H%M%S)"

echo "╔══════════════════════════════════════╗"
echo "║     Vertigo LoRA Retrain             ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Run: ${RUN_NAME}"
echo ""

# Step 1: Re-extract from current codebase
echo "--- Extracting from codebase (${PWD}/../src) ---"
python scripts/extract_codebase.py
echo ""

# Step 2: Rebuild MCP examples
echo "--- Building MCP tool-calling examples ---"
python scripts/build_mcp_examples.py
echo ""

# Step 3: Validate, dedup, merge, and split
echo "--- Validating, deduplicating, merging ---"
python scripts/merge_dataset.py
echo ""

# Step 4: Train new adapter
echo "--- Training new adapter ---"
exec ./scripts/train.sh "${RUN_NAME}"
