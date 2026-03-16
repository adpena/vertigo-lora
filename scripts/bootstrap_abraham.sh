#!/usr/bin/env bash
# Bootstrap abraham (128GB) for LoRA training
# Run this once after cloning the repo on the new machine
set -euo pipefail
cd "$(dirname "$0")/.."

echo "╔══════════════════════════════════════╗"
echo "║     Bootstrapping abraham (128GB)    ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Install uv if needed
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo ""
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install -e ".[full]"
echo ""

# Download models (parallel)
echo "Downloading models..."
uv run python -c "from mlx_lm import load; load('mlx-community/Qwen3.5-4B-4bit')" &
uv run python -c "from mlx_lm import load; load('mlx-community/Qwen3.5-9B-4bit')" &
uv run python -c "from mlx_lm import load; load('mlx-community/Qwen3.5-4B-8bit')" &
wait
echo "Models downloaded."
echo ""

# Verify training data exists
if [ ! -f "data/processed/train.jsonl" ]; then
    echo "Rebuilding processed data..."
    uv run python scripts/merge_dataset.py --mix-strategy three-dataset
    echo ""
fi

# Verify adapter registry
if [ ! -f "adapters/REGISTRY.json" ]; then
    echo "WARNING: adapters/REGISTRY.json not found — run auto_train.sh to initialize."
fi

echo ""
echo "╔══════════════════════════════════════╗"
echo "║              Ready!                  ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Recommended first run:"
echo "  ./scripts/auto_train.sh 4b v0.7-4b-full"
echo ""
echo "Available configs:"
ls configs/*.yaml | sed 's|configs/|  |'
