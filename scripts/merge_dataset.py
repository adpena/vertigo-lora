#!/usr/bin/env python3
from __future__ import annotations

"""
Merge all raw data sources into train/valid/test splits.

Pipeline:
1. Load all raw JSONL files
2. Run validation and dedup (via validate_and_dedup module)
3. Balance categories (oversample minority categories)
4. Split 80/10/10
5. Write MLX-compatible JSONL (only 'messages' and optionally 'tools' keys)

MLX expects data/processed/ to contain {train,valid,test}.jsonl
"""

import argparse
import fnmatch
import json
import random
from pathlib import Path

from validate_and_dedup import validate_and_filter

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
SEED = 42

# Include ~10% general capability preservation examples
# to prevent catastrophic forgetting of the base model's abilities
GENERAL_RATIO = 0.10

# ---------------------------------------------------------------------------
# 3-dataset family mixing strategy
# ---------------------------------------------------------------------------
DATASET_FAMILIES = {
    "sft": {
        "patterns": [
            "codebase*",
            "oss_roblox*",
            "api_docs*",
            "devforum*",
            "evolved*",
            "synthetic*",
            "roblox_creator*",
            "luau_*",
            "rojo_*",
        ],
        "target_ratio": 0.575,  # 57.5% of final mix
    },
    "trajectory": {
        "patterns": ["mcp_*", "gameplay_*", "embodiment_*", "studio_trajectory*"],
        "target_ratio": 0.275,  # 27.5% of final mix
    },
    "critic": {
        "patterns": ["bugfix_*", "critic_*", "preference_*"],
        "target_ratio": 0.15,  # 15% of final mix
    },
}


def balance_categories(examples: list[dict], max_ratio: float = 3.0) -> list[dict]:
    """
    Balance category distribution by oversampling minority categories.

    max_ratio: maximum ratio between largest and smallest category.
    Categories smaller than (largest / max_ratio) get oversampled.
    """
    by_category: dict[str, list[dict]] = {}
    for ex in examples:
        cat = ex.get("category", "unknown")
        by_category.setdefault(cat, []).append(ex)

    if not by_category:
        return examples

    largest = max(len(v) for v in by_category.values())
    target_min = int(largest / max_ratio)

    rng = random.Random(SEED)
    balanced = []
    for cat, cat_examples in by_category.items():
        balanced.extend(cat_examples)
        # Oversample if below target
        if len(cat_examples) < target_min:
            shortfall = target_min - len(cat_examples)
            oversampled = rng.choices(cat_examples, k=shortfall)
            balanced.extend(oversampled)

    return balanced


def _classify_family(source_file: str) -> str | None:
    """Return the family name for a source filename, or None if unmatched."""
    basename = Path(source_file).stem if "/" in source_file or "\\" in source_file else source_file
    for family, cfg in DATASET_FAMILIES.items():
        for pattern in cfg["patterns"]:
            if fnmatch.fnmatch(basename, pattern):
                return family
    return None


def mix_by_family(examples: list[dict]) -> list[dict]:
    """
    Mix examples according to DATASET_FAMILIES target ratios.

    Each example is classified into a family based on its ``source_file``
    metadata field (falls back to ``source`` or ``category``).  Examples that
    don't match any family pattern are placed in a special ``_unmatched`` bucket
    and distributed proportionally.

    Under-represented families are oversampled to hit target ratios.
    If any family has zero examples, we fall back to balanced mode.
    """
    # --- classify ---
    by_family: dict[str, list[dict]] = {f: [] for f in DATASET_FAMILIES}
    unmatched: list[dict] = []

    for ex in examples:
        # Try several metadata keys for the source filename
        src = ex.get("source_file") or ex.get("source") or ex.get("category") or ""
        family = _classify_family(src)
        if family:
            by_family[family].append(ex)
        else:
            unmatched.append(ex)

    # --- log raw counts ---
    print("\n--- Family classification ---")
    for fam, exs in by_family.items():
        print(f"  {fam}: {len(exs)} examples")
    if unmatched:
        print(f"  _unmatched: {len(unmatched)} examples (will be distributed proportionally)")

    # --- fall back if any family is empty ---
    empty_families = [f for f, exs in by_family.items() if len(exs) == 0]
    if empty_families:
        print(f"\n  WARNING: families with zero examples: {empty_families}")
        print("  Falling back to balanced category mixing.")
        return balance_categories(examples)

    # --- compute target counts ---
    # Distribute unmatched examples proportionally across families
    total_matched = sum(len(v) for v in by_family.values())
    total_available = total_matched + len(unmatched)

    # Target total size = largest possible mix (sum of all available,
    # expanded where needed to hit ratios).  We anchor on the family
    # whose (count / target_ratio) is largest — that sets the total.
    implied_totals = []
    for fam, cfg in DATASET_FAMILIES.items():
        if cfg["target_ratio"] > 0:
            implied_totals.append(len(by_family[fam]) / cfg["target_ratio"])
    target_total = int(max(implied_totals))

    # Don't let the mix shrink below what we already have
    target_total = max(target_total, total_available)

    rng = random.Random(SEED)

    mixed: list[dict] = []
    actual_counts: dict[str, int] = {}

    for fam, cfg in DATASET_FAMILIES.items():
        target_n = int(target_total * cfg["target_ratio"])
        pool = by_family[fam]

        if len(pool) >= target_n:
            # Downsample
            sampled = rng.sample(pool, target_n)
        else:
            # Oversample: include all originals + fill shortfall
            sampled = list(pool)
            shortfall = target_n - len(pool)
            sampled.extend(rng.choices(pool, k=shortfall))

        mixed.extend(sampled)
        actual_counts[fam] = len(sampled)

    # Append unmatched examples as-is (they sit outside the ratio math)
    mixed.extend(unmatched)

    # --- log achieved ratios ---
    mix_total = len(mixed)
    print("\n--- Achieved mix ratios ---")
    for fam in DATASET_FAMILIES:
        n = actual_counts[fam]
        pct = n / mix_total * 100 if mix_total else 0
        target_pct = DATASET_FAMILIES[fam]["target_ratio"] * 100
        print(f"  {fam}: {n} examples ({pct:.1f}%, target {target_pct:.1f}%)")
    if unmatched:
        print(f"  _unmatched: {len(unmatched)} examples ({len(unmatched) / mix_total * 100:.1f}%)")
    print(f"  total: {mix_total}")

    return mixed


def _clean_message(m: dict) -> dict:
    """Strip metadata from a message, keeping only role/content/tool_calls/tool_call_id.

    Qwen3.5 chat template expects tool_calls[].function.arguments to be a dict,
    not a JSON string — convert if needed.
    """
    msg: dict = {"role": m["role"]}
    if m.get("content") is not None:
        msg["content"] = m["content"]
    if m.get("tool_calls"):
        fixed_calls = []
        for tc in m["tool_calls"]:
            tc_copy = json.loads(json.dumps(tc))  # deep copy
            fn = tc_copy.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
            fixed_calls.append(tc_copy)
        msg["tool_calls"] = fixed_calls
    if m.get("tool_call_id"):
        msg["tool_call_id"] = m["tool_call_id"]
    return msg


def _trim_tools(tools: list[dict], messages: list[dict]) -> list[dict]:
    """Return only the tool definitions that are actually invoked in messages.

    Qwen3.5's chat template renders verbose format instructions for every tool,
    adding ~90 tokens per tool definition.  Trimming from 15 tools to the 1-2
    actually used saves 1000+ prompt tokens and prevents truncation-induced NaN.
    """
    used_names: set[str] = set()
    for m in messages:
        for tc in m.get("tool_calls", []):
            name = tc.get("function", {}).get("name")
            if name:
                used_names.add(name)
    if not used_names:
        return []
    return [t for t in tools if t.get("function", {}).get("name") in used_names]


def to_mlx_rows(example: dict) -> list[dict]:
    """Convert a training example to one or more MLX-compatible rows.

    For plain chat examples this returns a single row.

    For multi-turn tool-calling conversations (system, user, assistant+tool_call,
    tool_response, assistant, ...) this splits into multiple rows, one per
    assistant turn.  Each sub-example contains all preceding context so the model
    learns to generate BOTH the tool_call and the final answer.

    Without this split, MLX's mask_prompt flag only unmasks the *last* assistant
    message, so intermediate tool_call turns are never trained on — and when those
    masked prompts exceed max_seq_length the loss becomes 0/0 = NaN.
    """
    messages = [_clean_message(m) for m in example["messages"]]
    tools = example.get("tools")

    # ── Non-tool example: return as-is ──
    if not tools:
        return [{"messages": messages}]

    # ── Tool example: trim tools to only those actually used ──
    trimmed = _trim_tools(tools, messages)
    if not trimmed:
        # No tool calls found despite tools key — treat as plain chat
        return [{"messages": messages}]

    # ── Split into sub-examples at each assistant turn ──
    # This ensures mask_prompt unmasks the correct assistant message each time.
    #
    # Walk through messages and emit a row every time we reach an assistant
    # message.  Each row contains all messages from the start up to (and
    # including) that assistant message.
    rows: list[dict] = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            rows.append(
                {
                    "tools": trimmed,
                    "messages": messages[: i + 1],
                }
            )

    # Fallback: if no assistant message was found, emit the whole thing
    if not rows:
        rows.append({"tools": trimmed, "messages": messages})

    return rows


def split_data(examples: list[dict]) -> tuple[list, list, list]:
    """Shuffle and split into train/valid/test."""
    rng = random.Random(SEED)
    rng.shuffle(examples)

    n = len(examples)
    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    return examples[:train_end], examples[train_end:valid_end], examples[valid_end:]


def write_jsonl(path: Path, data: list[dict]):
    """Write list of dicts as JSONL."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge raw datasets into train/valid/test splits.")
    parser.add_argument(
        "--mix-strategy",
        choices=["balanced", "three-dataset"],
        default="balanced",
        help=(
            "Mixing strategy. 'balanced' (default) uses per-category oversampling. "
            "'three-dataset' groups sources into SFT/trajectory/critic families "
            "and samples to hit configured ratios."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing raw JSONL files (default: data/raw/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for output splits (default: data/processed/).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = args.raw_dir
    out_dir = args.out_dir
    print("=== Dataset Merge Pipeline ===")
    print(f"Mix strategy: {args.mix_strategy}\n")

    # Step 1: Validate and dedup
    examples = validate_and_filter(raw_dir)

    if not examples:
        print("No data found in data/raw/. Run extraction scripts first.")
        return

    # Step 2: Mix / balance
    if args.mix_strategy == "three-dataset":
        balanced = mix_by_family(examples)
    else:
        balanced = balance_categories(examples)
    print(f"\nAfter mixing: {len(balanced)} examples (was {len(examples)})")

    # Step 3: Convert to MLX format
    # to_mlx_rows may return multiple rows per example (multi-turn tool splits)
    mlx_rows = []
    for ex in balanced:
        mlx_rows.extend(to_mlx_rows(ex))

    # Step 4: Split
    train, valid, test = split_data(mlx_rows)

    # Step 5: Write
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "test.jsonl", test)

    print("\nSplit results:")
    print(f"  Train: {len(train)} ({len(train) / len(mlx_rows) * 100:.0f}%)")
    print(f"  Valid: {len(valid)} ({len(valid) / len(mlx_rows) * 100:.0f}%)")
    print(f"  Test:  {len(test)} ({len(test) / len(mlx_rows) * 100:.0f}%)")
    print(f"\nOutput: {out_dir}/")

    # Stats on final data
    tool_examples = sum(1 for r in mlx_rows if "tools" in r)
    chat_examples = len(mlx_rows) - tool_examples
    print(f"\nChat examples: {chat_examples}")
    print(f"Tool-calling examples: {tool_examples}")


if __name__ == "__main__":
    main()
