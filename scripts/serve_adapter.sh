#!/usr/bin/env bash
# Serve the latest LoRA adapter via mlx_lm.server
# Exposes an OpenAI-compatible endpoint that Obelisk or agents can hit directly.
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:-adapters/v0.5-4b-curated}"
MODEL="${2:-mlx-community/Qwen3.5-4B-4bit}"
PORT="${3:-8899}"

# Auto-detect model from adapter config if not specified
if [ -f "${ADAPTER}/adapter_config.json" ]; then
    DETECTED=$(python3 -c "import json; print(json.load(open('${ADAPTER}/adapter_config.json'))['model'])" 2>/dev/null || true)
    if [ -n "$DETECTED" ]; then
        MODEL="$DETECTED"
    fi
fi

echo "Serving ${MODEL} + ${ADAPTER} on :${PORT}"
exec uv run python -m mlx_lm.server \
    --model "$MODEL" \
    --adapter-path "$ADAPTER" \
    --port "$PORT"
