#!/usr/bin/env python3
from __future__ import annotations

"""
Pre-split training examples that exceed max_seq_length.

Many processed examples are >2048 tokens and get truncated during training,
losing valuable content. This script splits long examples into smaller chunks
that fit within the context window, preserving the system prompt in each chunk.

Split strategies (in priority order):
1. Multi-turn conversations: split at turn boundaries after each assistant turn
2. Long single assistant responses: split at code block boundaries (```...```)
3. No natural split point: truncate with a note
"""

import argparse
import json
import re
import tempfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_tokenizer = None
_use_tokenizer = False


def _init_tokenizer(model: str = "Qwen/Qwen2.5-3B") -> None:
    global _tokenizer, _use_tokenizer
    try:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        _use_tokenizer = True
    except Exception:
        _use_tokenizer = False


def estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count for a list of messages."""
    text = "".join(m.get("content") or "" for m in messages)
    if _use_tokenizer:
        return len(_tokenizer.encode(text))
    return int(len(text) / 3.5)


# ---------------------------------------------------------------------------
# Splitting logic
# ---------------------------------------------------------------------------


def _system_msg(messages: list[dict]) -> dict | None:
    if messages and messages[0]["role"] == "system":
        return messages[0]
    return None


def _split_at_code_blocks(text: str, max_chars: int) -> list[str]:
    """Split long text at code block boundaries."""
    blocks = re.split(r"(```[\s\S]*?```)", text)
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(current) + len(block) > max_chars and current.strip():
            chunks.append(current.strip())
            current = ""
        current += block
    if current.strip():
        chunks.append(current.strip())
    return chunks if len(chunks) > 1 else []


def split_example(example: dict, max_tokens: int) -> list[dict]:
    """Split a single example into chunks that fit within max_tokens.

    Returns a list of examples (1 = no split needed).
    """
    messages = example["messages"]
    if estimate_tokens(messages) <= max_tokens:
        return [example]

    sys_msg = _system_msg(messages)
    non_sys = messages[1:] if sys_msg else messages

    # All current data is single user+assistant pairs.
    # Strategy 1: multi-turn — split at turn boundaries (future-proofing)
    if len(non_sys) > 2:
        chunks = []
        current_turns: list[dict] = []
        for msg in non_sys:
            current_turns.append(msg)
            if msg["role"] == "assistant":
                candidate = ([sys_msg] if sys_msg else []) + current_turns
                if estimate_tokens(candidate) > max_tokens and len(current_turns) > 2:
                    # Flush previous turns as a chunk
                    prev = current_turns[:-2]
                    if prev and prev[-1]["role"] == "assistant":
                        chunks.append(([sys_msg] if sys_msg else []) + prev)
                    current_turns = current_turns[-2:]
        if current_turns:
            candidate = ([sys_msg] if sys_msg else []) + current_turns
            chunks.append(candidate)
        if len(chunks) > 1:
            results = []
            for ch in chunks:
                ex = {"messages": ch}
                if "tools" in example:
                    ex["tools"] = example["tools"]
                results.append(ex)
            return results

    # Strategy 2: split long assistant response at code block boundaries
    assistant_msgs = [m for m in non_sys if m["role"] == "assistant"]
    if assistant_msgs:
        asst = assistant_msgs[-1]
        content = asst.get("content") or ""
        # Estimate how many chars we can fit for the assistant part
        overhead = ([sys_msg] if sys_msg else []) + [m for m in non_sys if m["role"] != "assistant"]
        overhead_tokens = estimate_tokens(overhead)
        remaining = max_tokens - overhead_tokens
        max_chars = int(remaining * 3.5) if not _use_tokenizer else int(remaining * 3.0)

        if max_chars > 200:
            text_chunks = _split_at_code_blocks(content, max_chars)
            if text_chunks:
                results = []
                user_msgs = [m for m in non_sys if m["role"] != "assistant"]
                for i, chunk_text in enumerate(text_chunks):
                    if i > 0:
                        prefix = f"(continued, part {i + 1}/{len(text_chunks)})\n\n"
                    else:
                        prefix = ""
                    msgs = []
                    if sys_msg:
                        msgs.append(sys_msg)
                    if i == 0:
                        msgs.extend(user_msgs)
                    else:
                        context = user_msgs[-1]["content"] if user_msgs else "Continue."
                        msgs.append(
                            {
                                "role": "user",
                                "content": f"Continue from where you left off. Original question: {context[:200]}",
                            }
                        )
                    msgs.append({"role": "assistant", "content": prefix + chunk_text})
                    ex = {"messages": msgs}
                    if "tools" in example:
                        ex["tools"] = example["tools"]
                    results.append(ex)
                # Verify all chunks fit
                if all(estimate_tokens(r["messages"]) <= max_tokens for r in results):
                    return results

    # Strategy 3: truncate with note
    # Keep system + user, truncate assistant
    if assistant_msgs:
        asst = assistant_msgs[-1]
        content = asst.get("content") or ""
        overhead = ([sys_msg] if sys_msg else []) + [m for m in non_sys if m["role"] != "assistant"]
        overhead_tokens = estimate_tokens(overhead)
        remaining = max_tokens - overhead_tokens - 20  # room for truncation note
        max_chars = int(remaining * 3.5) if not _use_tokenizer else int(remaining * 3.0)
        if max_chars > 200:
            truncated = content[:max_chars] + "\n\n[truncated]"
            msgs = []
            if sys_msg:
                msgs.append(sys_msg)
            msgs.extend(m for m in non_sys if m["role"] != "assistant")
            msgs.append({"role": "assistant", "content": truncated})
            return [{"messages": msgs}]

    # Can't meaningfully split — return as-is (will be truncated by trainer)
    return [example]


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------


def process_file(path: Path, max_tokens: int, dry_run: bool) -> tuple[int, int, int]:
    """Process a single JSONL file. Returns (split_count, new_total, truncated)."""
    if not path.exists():
        return 0, 0, 0

    examples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    len(examples)
    split_count = 0
    truncated = 0
    results: list[dict] = []

    for ex in examples:
        chunks = split_example(ex, max_tokens)
        if len(chunks) > 1:
            split_count += 1
        elif len(chunks) == 1 and estimate_tokens(chunks[0]["messages"]) > max_tokens:
            truncated += 1
        results.extend(chunks)

    if not dry_run and (split_count > 0 or truncated > 0):
        # Atomic write
        tmp = tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".jsonl", delete=False)
        try:
            for ex in results:
                tmp.write(json.dumps(ex, ensure_ascii=False) + "\n")
            tmp.close()
            Path(tmp.name).replace(path)
        except Exception:
            Path(tmp.name).unlink(missing_ok=True)
            raise

    return split_count, len(results), truncated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-split long training sequences")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per example (default: 2048)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Data directory")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write")
    args = parser.parse_args()

    _init_tokenizer()
    if _use_tokenizer:
        print("Using transformers tokenizer")
    else:
        print("Using char estimation (chars / 3.5)")

    total_split = 0
    total_truncated = 0
    for split_name in ["train", "valid", "test"]:
        path = args.data_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"  {split_name}: not found, skipping")
            continue

        original = sum(1 for _ in path.open())
        split_count, new_total, truncated = process_file(path, args.max_tokens, args.dry_run)
        total_split += split_count
        total_truncated += truncated

        gained = new_total - original
        print(f"  {split_name}: {original} examples -> {new_total} ({'+' if gained >= 0 else ''}{gained})")
        if split_count:
            print(f"    Split {split_count} examples into multiple chunks")
        if truncated:
            print(f"    Truncated {truncated} examples (no natural split point)")

    action = "Would write" if args.dry_run else "Wrote"
    print(f"\nSplit {total_split} examples, truncated {total_truncated}. {action} results.")
    if args.dry_run:
        print("(dry run — no files modified)")


if __name__ == "__main__":
    main()
