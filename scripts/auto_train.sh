#!/usr/bin/env bash
# Automated train -> benchmark -> promote pipeline
# Usage: ./scripts/auto_train.sh [config] [run_name]
# Example: ./scripts/auto_train.sh 4b v0.7-4b-curated
#          ./scripts/auto_train.sh 9b v0.1-9b
#          ./scripts/auto_train.sh  # defaults to 4b, auto-names

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-4b}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${2:-auto-${CONFIG}-${TIMESTAMP}}"
ADAPTER_DIR="adapters/${RUN_NAME}"
CONFIG_FILE="configs/${CONFIG}.yaml"

# M5 Max workaround: suppress NAX kernel dispatch and JIT compilation
export MLX_METAL_GPU_ARCH="${MLX_METAL_GPU_ARCH:-applegpu_g14p}"
export MLX_DISABLE_COMPILE="${MLX_DISABLE_COMPILE:-1}"

# Validate
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config not found: $CONFIG_FILE"
    echo "Available: $(ls configs/*.yaml | xargs -I{} basename {} .yaml | tr '\n' ' ')"
    exit 1
fi

# Extract model name from config
MODEL=$(grep '^model:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')

echo "╔══════════════════════════════════════╗"
echo "║    Auto Train -> Promote Pipeline    ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Config:  ${CONFIG} (${CONFIG_FILE})"
echo "Model:   ${MODEL}"
echo "Adapter: ${ADAPTER_DIR}"
echo "Run:     ${RUN_NAME}"
echo "Time:    $(date)"
echo ""

# ── Step 1: Train ─────────────────────────────────────────────────────
echo ">>> Step 1/4: Training..."
mkdir -p "${ADAPTER_DIR}"
cp "${CONFIG_FILE}" "${ADAPTER_DIR}/config.yaml"

uv run python -m mlx_lm lora \
    --config "$CONFIG_FILE" \
    --adapter-path "$ADAPTER_DIR"

# Check training succeeded
if [ ! -f "$ADAPTER_DIR/adapters.safetensors" ]; then
    echo "ERROR: Training failed — no weights produced"
    exit 1
fi
echo "Training complete. Adapter saved to: ${ADAPTER_DIR}"

# ── Step 2: Benchmark ────────────────────────────────────────────────
echo ""
echo ">>> Step 2/4: Benchmarking..."
# Storage convention: results go to data/eval/ locally.
# When APDataStore is mounted, scripts/storage.py resolves to external SSD.
# auto_train.sh keeps local paths since it's bash; Python scripts use storage.py.
RESULTS_FILE="data/eval/results_${RUN_NAME}.jsonl"
uv run python scripts/run_benchmark.py \
    --adapter "$ADAPTER_DIR" \
    --model "$MODEL" \
    --output "$RESULTS_FILE"

# ── Step 3: Compare to current production ────────────────────────────
echo ""
echo ">>> Step 3/4: Comparing to production..."
uv run python scripts/promote_adapter.py \
    --adapter "$RUN_NAME" \
    --results "$RESULTS_FILE" \
    --registry adapters/REGISTRY.json

# ── Step 4: Analyze ────────────────────────────────────────────────
echo ""
echo ">>> Step 4/6: Generating analysis report..."
uv run python scripts/analyze_run.py \
    --adapter "$RUN_NAME" \
    --config "$CONFIG_FILE" \
    --results "$RESULTS_FILE" \
    --registry adapters/REGISTRY.json

# ── Step 5: Log ────────────────────────────────────────────────────
echo ""
echo ">>> Step 5/6: Logging run..."
uv run python scripts/log_run.py \
    --adapter "$RUN_NAME" \
    --config "$CONFIG_FILE" \
    --results "$RESULTS_FILE" \
    --registry adapters/REGISTRY.json

# ── Step 6: Fuse for LM Studio (if promoted) ─────────────────────────
PROD_ADAPTER=$(python3 -c "import json; print(json.load(open('adapters/REGISTRY.json'))['production']['adapter'])")
if [ "$PROD_ADAPTER" = "$RUN_NAME" ]; then
    echo ""
    echo ">>> Step 6/7: Fusing promoted adapter for LM Studio..."
    # Extract base model name for Vertigo model naming
    # e.g. mlx-community/Qwen3.5-4B-4bit → Qwen3.5-4B
    BASE_NAME=$(echo "$MODEL" | sed 's|.*/||' | sed 's/-4bit$//' | sed 's/-8bit$//' | sed 's/-MLX//')
    FUSED_NAME="Vertigo-${BASE_NAME}-${RUN_NAME}"
    FUSED_PATH="${HOME}/.lmstudio/models/vertigo/${FUSED_NAME}"

    # Resolve base model local path (avoid HF network calls)
    MODEL_CACHE=$(python3 -c "
from pathlib import Path
slug = '${MODEL}'.replace('/', '--')
cache = Path.home() / '.cache/huggingface/hub' / f'models--{slug}'
snaps = list((cache / 'snapshots').iterdir()) if (cache / 'snapshots').exists() else []
print(snaps[0] if snaps else '${MODEL}')
")

    uv run python -m mlx_lm fuse \
        --model "$MODEL_CACHE" \
        --adapter-path "$ADAPTER_DIR" \
        --save-path "$FUSED_PATH"

    echo "Fused model saved to: $FUSED_PATH"
    echo "Load in LM Studio as: vertigo/${FUSED_NAME}"
else
    echo ""
    echo ">>> Step 6/7: Skipping fuse (not promoted)."
fi

# ── Step 7: Done ─────────────────────────────────────────────────────
echo ""
echo ">>> Step 7/7: Done."
echo "Check adapters/REGISTRY.json for results."
echo "Check docs/training-log.md for run history."
echo "Check docs/runs/${RUN_NAME}.md for detailed analysis."
echo ""
if [ "$PROD_ADAPTER" = "$RUN_NAME" ]; then
    echo "NEW PRODUCTION: Load 'vertigo/${FUSED_NAME}' in LM Studio."
else
    echo "To serve the current production adapter:"
    echo "  ./scripts/serve_adapter.sh adapters/${PROD_ADAPTER}"
fi
