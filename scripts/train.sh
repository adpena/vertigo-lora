#!/usr/bin/env bash
# Vertigo LoRA training script
# Usage: ./scripts/train.sh [config] [run_name]
#
# Configs:
#   2b        — Qwen3.5-2B for 8GB devices (rank 16, 2048 seq)
#   4b        — Qwen3.5-4B sweet spot for 36GB Apple Silicon (rank 16, 4096 seq)
#   4b-128gb  — Qwen3.5-4B for 128GB Apple Silicon (rank 32, 4096 seq)
#   4b-8bit-128gb — Qwen3.5-4B 8-bit for 128GB (highest quality, rank 32, 4096 seq)
#   9b        — Qwen3.5-9B for 36GB Apple Silicon
#   35b       — 35B-A3B full config for 128GB M-series (rank 64, 8192 seq)
#
# Archived (in configs/archive/):
#   builder, player, combined, default — legacy 27B/35B configs
#
# Examples:
#   ./scripts/train.sh 4b
#   ./scripts/train.sh 2b
#   ./scripts/train.sh 4b-8bit-128gb v0.1
#   ./scripts/train.sh                    # defaults to 4b

set -euo pipefail
cd "$(dirname "$0")/.."

# M5 Max workaround: suppress NAX kernel dispatch and JIT compilation
# to avoid Metal OOM during backward pass. Remove when MLX ships native M5 wheels.
export MLX_METAL_GPU_ARCH="${MLX_METAL_GPU_ARCH:-applegpu_g14p}"
export MLX_DISABLE_COMPILE="${MLX_DISABLE_COMPILE:-1}"

CONFIG_NAME="${1:-4b}"
RUN_NAME="${2:-${CONFIG_NAME}-$(date +%Y%m%d_%H%M%S)}"
CONFIG="configs/${CONFIG_NAME}.yaml"

if [ ! -f "${CONFIG}" ]; then
    echo "error: Config not found: ${CONFIG}"
    echo "Available: $(ls configs/*.yaml | xargs -I{} basename {} .yaml | tr '\n' ' ')"
    exit 1
fi

# Extract model name from config
MODEL=$(grep '^model:' "${CONFIG}" | awk '{print $2}' | tr -d '"')
DATA_DIR=$(grep '^data:' "${CONFIG}" | awk '{print $2}' | tr -d '"')

ADAPTER_DIR="adapters/${RUN_NAME}"

echo "╔══════════════════════════════════════╗"
echo "║       Vertigo LoRA Training          ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Config:  ${CONFIG_NAME} (${CONFIG})"
echo "Model:   ${MODEL}"
echo "Data:    ${DATA_DIR}"
echo "Adapter: ${ADAPTER_DIR}"
echo "Run:     ${RUN_NAME}"
echo ""

# Ensure data exists
if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "No training data found at ${DATA_DIR}. Running full pipeline..."
    echo ""

    echo "--- Step 1: Extract from codebase ---"
    .venv/bin/python scripts/extract_codebase_granular.py
    echo ""

    echo "--- Step 2: Build MCP examples ---"
    .venv/bin/python scripts/build_mcp_examples.py
    echo ""

    echo "--- Step 3: Run all generators ---"
    for gen in extract_api_docs.py extract_devforum.py generate_bugfix.py evolve_examples.py generate_synthetic.py; do
        if [ -f "scripts/${gen}" ]; then
            echo "  Running ${gen}..."
            .venv/bin/python "scripts/${gen}" 2>/dev/null || echo "  (skipped: ${gen})"
        fi
    done
    echo ""

    echo "--- Step 4: Validate, dedup, merge ---"
    .venv/bin/python scripts/merge_dataset.py
    echo ""
fi

TRAIN_COUNT=$(wc -l < "${DATA_DIR}/train.jsonl" | tr -d ' ')
VALID_COUNT=$(wc -l < "${DATA_DIR}/valid.jsonl" | tr -d ' ')
TEST_COUNT=$(wc -l < "${DATA_DIR}/test.jsonl" | tr -d ' ')
echo "Dataset: ${TRAIN_COUNT} train / ${VALID_COUNT} valid / ${TEST_COUNT} test"
echo ""

# Create adapter directory
mkdir -p "${ADAPTER_DIR}"

# Copy config for reproducibility
cp "${CONFIG}" "${ADAPTER_DIR}/config.yaml"

# Snapshot dataset stats
if [ -f scripts/analyze_dataset.py ]; then
    .venv/bin/python scripts/analyze_dataset.py --snapshot "${RUN_NAME}" 2>/dev/null || true
fi

# Train using MLX config file
echo "--- Training ---"
.venv/bin/python -m mlx_lm lora --config "${CONFIG}" --adapter-path "${ADAPTER_DIR}"

echo ""
echo "--- Training complete ---"
echo "Adapter saved to: ${ADAPTER_DIR}"

# Symlink latest (per-config and global)
ln -sfn "${RUN_NAME}" "adapters/latest-${CONFIG_NAME}"
ln -sfn "${RUN_NAME}" adapters/latest
echo "Symlinked adapters/latest-${CONFIG_NAME} -> ${RUN_NAME}"

# Run evaluation
echo ""
echo "--- Evaluation ---"
.venv/bin/python -m mlx_lm lora \
    --model "${MODEL}" \
    --adapter-path "${ADAPTER_DIR}" \
    --data "${DATA_DIR}" \
    --test

# Log results
echo ""
echo "Run ${RUN_NAME} complete."
echo ""
echo "To serve:  mlx_lm.server --model ${MODEL} --adapter-path ${ADAPTER_DIR}"
echo "To fuse:   mlx_lm.fuse --model ${MODEL} --adapter-path ${ADAPTER_DIR} --save-path adapters/fused-${RUN_NAME}"
echo "To compare: .venv/bin/python scripts/analyze_dataset.py --compare v0.3-pre-train ${RUN_NAME}"
