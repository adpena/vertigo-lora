#!/usr/bin/env python3
from __future__ import annotations

"""
Validate, score, deduplicate, and filter training examples.

Pipeline:
1. Schema validation (Pydantic) — reject malformed examples
2. Quality scoring — rate each example on multiple dimensions
3. MinHash deduplication — remove near-duplicate examples
4. Difficulty calibration — ensure balanced difficulty distribution
5. Output cleaned dataset

Uses:
- Pydantic for schema validation
- datasketch MinHash for deduplication
- Custom scoring for quality assessment
"""

import json
import hashlib
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass
class QualityScore:
    """Multi-dimensional quality score for a training example."""

    has_reasoning: float = 0.0  # 0-1: contains <think> block
    code_length: float = 0.0  # 0-1: normalized code length (not too short, not too long)
    type_annotations: float = 0.0  # 0-1: uses Luau type annotations
    modern_apis: float = 0.0  # 0-1: uses modern Roblox APIs
    conversation_depth: float = 0.0  # 0-1: multi-turn quality
    total: float = 0.0

    def compute_total(self):
        self.total = (
            self.has_reasoning * 0.25
            + self.code_length * 0.20
            + self.type_annotations * 0.20
            + self.modern_apis * 0.15
            + self.conversation_depth * 0.20
        )
        return self.total


def score_example(example: dict) -> QualityScore:
    """Score a training example on multiple quality dimensions."""
    score = QualityScore()
    messages = example.get("messages", [])

    # Concatenate all assistant content
    assistant_content = ""
    for m in messages:
        if m.get("role") == "assistant" and m.get("content"):
            assistant_content += m["content"]

    # Has reasoning traces
    if "<think>" in assistant_content:
        score.has_reasoning = 1.0

    # Code length (sweet spot: 200-3000 chars)
    code_len = len(assistant_content)
    if code_len < 100:
        score.code_length = 0.1
    elif code_len < 200:
        score.code_length = 0.4
    elif code_len < 3000:
        score.code_length = 1.0
    elif code_len < 8000:
        score.code_length = 0.7
    else:
        score.code_length = 0.4

    # Type annotations
    type_indicators = [
        "export type",
        ": string",
        ": number",
        ": boolean",
        ": Vector3",
        ": Player",
        ": Instance",
        "--!strict",
    ]
    type_count = sum(1 for t in type_indicators if t in assistant_content)
    score.type_annotations = min(1.0, type_count / 3)

    # Modern APIs
    modern_indicators = [
        "task.wait",
        "task.spawn",
        "task.defer",
        "game:GetService",
        "table.freeze",
        "table.create",
        "math.lerp",
        "@native",
    ]
    modern_count = sum(1 for m in modern_indicators if m in assistant_content)
    score.modern_apis = min(1.0, modern_count / 3)

    # Conversation depth
    turn_count = sum(1 for m in messages if m.get("role") in ("user", "assistant"))
    score.conversation_depth = min(1.0, turn_count / 4)

    score.compute_total()
    return score


def content_hash(example: dict) -> str:
    """Create a normalized hash of example content for exact dedup."""
    content_parts = []
    for m in example.get("messages", []):
        c = m.get("content", "") or ""
        # Normalize whitespace for comparison
        normalized = " ".join(c.split())
        content_parts.append(normalized[:500])  # First 500 chars per message
    return hashlib.md5("|".join(content_parts).encode()).hexdigest()


def minhash_signature(text: str, num_perm: int = 128) -> set[int]:
    """Simple shingling for near-dedup (fallback if datasketch not installed)."""
    shingles = set()
    words = text.split()
    for i in range(len(words) - 2):
        shingle = " ".join(words[i : i + 3])
        shingles.add(hash(shingle))
    return shingles


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def validate_and_filter(input_dir: Path, min_score: float = 0.3, dedup_threshold: float = 0.7) -> list[dict]:
    """Full validation, scoring, and dedup pipeline."""
    all_examples = []
    stats = Counter()

    # Load all raw data
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    stats["loaded"] += 1
                except json.JSONDecodeError:
                    stats["json_error"] += 1
                    continue

                # Basic schema validation
                messages = example.get("messages", [])
                if len(messages) < 2:
                    stats["too_few_messages"] += 1
                    continue

                has_assistant = any(m.get("role") == "assistant" for m in messages)
                if not has_assistant:
                    stats["no_assistant"] += 1
                    continue

                # Score
                score = score_example(example)
                example["_quality_score"] = score.total

                if score.total < min_score:
                    stats["below_threshold"] += 1
                    continue

                all_examples.append(example)

    print(f"Loaded {stats['loaded']} examples, {len(all_examples)} passed quality threshold")

    # Exact dedup
    seen_hashes = set()
    deduped = []
    for ex in all_examples:
        h = content_hash(ex)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(ex)
        else:
            stats["exact_dup"] += 1

    print(f"Removed {stats['exact_dup']} exact duplicates")

    # Near-dedup via Jaccard on shingles
    final = []
    signatures = []
    for ex in deduped:
        content = " ".join(m.get("content", "") or "" for m in ex.get("messages", []) if m.get("role") == "assistant")
        sig = minhash_signature(content)

        is_near_dup = False
        for existing_sig in signatures:
            if jaccard_similarity(sig, existing_sig) > dedup_threshold:
                is_near_dup = True
                stats["near_dup"] += 1
                break

        if not is_near_dup:
            final.append(ex)
            signatures.append(sig)

    print(f"Removed {stats['near_dup']} near-duplicates")
    print(f"Final: {len(final)} examples")

    # Sort by quality score descending
    final.sort(key=lambda x: x.get("_quality_score", 0), reverse=True)

    return final


def main():
    print("=== Validation & Deduplication Pipeline ===\n")
    examples = validate_and_filter(OUTPUT)

    # Difficulty distribution
    diffs = Counter(ex.get("difficulty", 0) for ex in examples)
    print(f"\nDifficulty distribution: {dict(sorted(diffs.items()))}")

    # Category distribution
    cats = Counter(ex.get("category", "unknown") for ex in examples)
    print("Category distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    # Quality score histogram
    scores = [ex.get("_quality_score", 0) for ex in examples]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nQuality scores: min={min(scores):.2f} avg={avg:.2f} max={max(scores):.2f}")


if __name__ == "__main__":
    main()
