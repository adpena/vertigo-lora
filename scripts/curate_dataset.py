#!/usr/bin/env python3
"""Curate raw JSONL datasets into a filtered training set.

Reproduces the v0.5-4b-curated pipeline (82.9%) from a YAML config.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
from pathlib import Path

import yaml

CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")


def count_code_lines(text: str) -> int:
    """Count lines of code inside fenced ``` blocks across all text fields."""
    total = 0
    for block in CODE_BLOCK_RE.findall(text):
        lines = block.strip("`").strip().splitlines()
        # First line may be language tag — skip if it looks like one
        if lines and not lines[0].strip().startswith(("--", "local", "{", "[", "return", "function")):
            lines = lines[1:]
        total += len([l for l in lines if l.strip()])
    return total


def extract_text(example: dict) -> str:
    """Pull all text from a JSONL example (handles messages list or text field)."""
    if "messages" in example:
        return "\n".join(m.get("content", "") for m in example["messages"])
    if "text" in example:
        return example["text"]
    return json.dumps(example)


def resolve_files(raw_dir: Path, patterns: list[str]) -> list[Path]:
    """Glob-expand patterns against raw_dir, return sorted unique paths."""
    paths: set[Path] = set()
    for pat in patterns:
        paths.update(Path(p) for p in glob.glob(str(raw_dir / pat)))
    return sorted(paths)


def copy_file(src: Path, dst: Path, dry_run: bool) -> int:
    """Copy a JSONL file, return example count."""
    count = sum(1 for _ in open(src))
    if not dry_run:
        shutil.copy2(src, dst)
    return count


def filter_file(src: Path, dst: Path, predicate, dry_run: bool) -> tuple[int, int]:
    """Filter JSONL by predicate, return (raw_count, kept_count)."""
    raw, kept_lines = 0, []
    with open(src) as f:
        for line in f:
            raw += 1
            ex = json.loads(line)
            if predicate(ex):
                kept_lines.append(line)
    if not dry_run:
        with open(dst, "w") as f:
            f.writelines(kept_lines)
    return raw, len(kept_lines)


def main():
    ap = argparse.ArgumentParser(description="Curate raw datasets into a training set")
    ap.add_argument("--config", default="configs/curation.yaml", help="YAML config path")
    ap.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    ap.add_argument("--output-dir", default="data/curated", help="Output directory")
    ap.add_argument("--dry-run", action="store_true", help="Report counts without writing")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    excluded = set(cfg.get("excluded", []))
    tiers = cfg["tiers"]
    total_raw, total_kept = 0, 0

    def process(label: str, files: list[Path], handler):
        nonlocal total_raw, total_kept
        for src in files:
            if src.name in excluded:
                print(f"  SKIP {src.name} (excluded)")
                continue
            if not src.exists():
                print(f"  MISS {src.name} (not found)")
                continue
            dst = out_dir / src.name
            raw, kept, note = handler(src, dst)
            total_raw += raw
            total_kept += kept
            if raw == kept:
                print(f"  COPY {src.name}: {kept} examples")
            else:
                print(f"  COPY {src.name}: {raw} -> {kept} ({note})")

    # Tier 1 + 2: copy as-is
    for tier_name in ("copy", "verified"):
        patterns = tiers.get(tier_name, [])
        files = resolve_files(raw_dir, patterns)
        print(f"\n[{tier_name.upper()}] {len(files)} files")

        def copy_handler(src, dst):
            n = copy_file(src, dst, args.dry_run)
            return n, n, "copied"

        process(tier_name, files, copy_handler)

    # Tier 3: filter by code density
    fc = tiers.get("filter_code", {})
    fc_files = resolve_files(raw_dir, fc.get("files", []))
    min_lines = fc.get("min_code_lines", 5)
    print(f"\n[FILTER_CODE] {len(fc_files)} files (min {min_lines} code lines)")

    def code_handler(src, dst):
        def pred(ex):
            return count_code_lines(extract_text(ex)) >= min_lines

        raw, kept = filter_file(src, dst, pred, args.dry_run)
        return raw, kept, f"filtered, min {min_lines} code lines"

    process("filter_code", fc_files, code_handler)

    # Tier 4: filter by score
    fs = tiers.get("filter_score", {})
    fs_files = resolve_files(raw_dir, fs.get("files", []))
    min_score = fs.get("min_score", 0.6)
    score_field = fs.get("score_field", "eval_score")
    print(f"\n[FILTER_SCORE] {len(fs_files)} files (min {score_field} >= {min_score})")

    def score_handler(src, dst):
        def pred(ex):
            return ex.get(score_field, 0) >= min_score

        raw, kept = filter_file(src, dst, pred, args.dry_run)
        return raw, kept, f"filtered, {score_field} >= {min_score}"

    process("filter_score", fs_files, score_handler)

    # Summary
    pct = (total_kept / total_raw * 100) if total_raw else 0
    mode = " (dry run)" if args.dry_run else ""
    print(f"\nCurated: {total_kept} examples from {total_raw} raw ({pct:.1f}% retained){mode}")


if __name__ == "__main__":
    main()
