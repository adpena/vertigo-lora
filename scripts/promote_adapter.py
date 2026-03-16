#!/usr/bin/env python3
from __future__ import annotations

"""
Adapter promotion logic — compare benchmark results against production and
optionally promote the new adapter in REGISTRY.json.

Usage:
    python3 scripts/promote_adapter.py --adapter v0.7-4b-curated --results data/eval/results_v07.jsonl
    python3 scripts/promote_adapter.py --adapter v0.7-4b-curated --results data/eval/results_v07.jsonl --dry-run
    python3 scripts/promote_adapter.py --adapter v0.7-4b-curated --results data/eval/results_v07.jsonl --force
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# 4B base model scores — the absolute floor for regression checks
# Re-derived 2026-03-16 on fixed MCP benchmark (real Studio tools)
BASE_SCORES: dict[str, float] = {
    "coding": 63.7,
    "bugfix": 83.3,
    "architecture": 67.5,
    "mcp_tool_calling": 97.5,
    "embodiment": 75.0,
    "overall": 75.1,
}

# Categories exempt from regression checks (high variance on small sample)
REGRESSION_EXEMPT: set[str] = set()


def load_results(path: Path) -> dict[str, float]:
    """Load benchmark results JSONL, return per-category mean scores and overall."""
    cat_scores: dict[str, list[float]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cat = r.get("category", "unknown")
            score = r["scores"]["overall"]
            cat_scores[cat].append(score)

    scores: dict[str, float] = {}
    all_scores: list[float] = []
    for cat, vals in sorted(cat_scores.items()):
        mean = sum(vals) / len(vals) * 100  # convert to percentage
        scores[cat] = round(mean, 1)
        all_scores.extend(vals)

    scores["overall"] = round(sum(all_scores) / len(all_scores) * 100, 1) if all_scores else 0.0
    return scores


def check_promotion(
    new_scores: dict[str, float],
    registry: dict,
) -> tuple[bool, list[str]]:
    """Check if new adapter should be promoted. Returns (pass, reasons)."""
    reasons: list[str] = []
    minimum = registry.get("minimum_score", BASE_SCORES["overall"])
    threshold = registry.get("regression_threshold", 2.0)
    prod = registry.get("production", {})
    prod_overall = prod.get("overall_score", 0.0)

    new_overall = new_scores.get("overall", 0.0)

    # Check 1: above minimum score (base model floor)
    if new_overall < minimum:
        reasons.append(f"overall {new_overall:.1f}% below minimum {minimum:.1f}%")

    # Check 2: must beat current production overall
    if new_overall <= prod_overall:
        reasons.append(f"overall {new_overall:.1f}% does not beat production {prod_overall:.1f}%")

    # Check 3: no category regression below base - threshold
    for cat, base_val in BASE_SCORES.items():
        if cat == "overall" or cat in REGRESSION_EXEMPT:
            continue
        new_val = new_scores.get(cat)
        if new_val is None:
            continue
        floor = base_val - threshold
        if new_val < floor:
            reasons.append(
                f"{cat} regressed to {new_val:.1f}% "
                f"(base: {base_val:.1f}%, threshold: {threshold:.1f}pp -> min {floor:.1f}%)"
            )

    return len(reasons) == 0, reasons


def promote(
    adapter_name: str,
    new_scores: dict[str, float],
    registry: dict,
    registry_path: Path,
) -> None:
    """Update REGISTRY.json: move current to history, set new as production."""
    old_prod = registry.get("production")
    now = datetime.now().isoformat(timespec="seconds")

    # Move current production to history
    if old_prod and old_prod.get("adapter"):
        history_entry = {
            "adapter": old_prod["adapter"],
            "overall_score": old_prod.get("overall_score", 0.0),
            "promoted": old_prod.get("timestamp", "unknown"),
            "demoted": now,
            "reason": (
                f"{adapter_name} scored {new_scores['overall']:.1f}% "
                f"({new_scores['overall'] - old_prod.get('overall_score', 0):.1f}pp)"
            ),
        }
        registry.setdefault("history", []).insert(0, history_entry)

    # Build category scores (exclude 'overall')
    cat_scores = {k: v for k, v in new_scores.items() if k != "overall"}

    # Set new production
    registry["production"] = {
        "adapter": adapter_name,
        "model": registry.get("production", {}).get("model", "unknown"),
        "overall_score": new_scores["overall"],
        "scores": cat_scores,
        "timestamp": now,
    }

    # Write atomically
    tmp = registry_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, registry_path)

    # Update symlink
    adapters_dir = registry_path.parent
    latest = adapters_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(adapter_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertigo LoRA adapter promotion")
    parser.add_argument("--adapter", required=True, help="Adapter directory name")
    parser.add_argument("--results", required=True, type=Path, help="Benchmark results JSONL")
    parser.add_argument("--registry", type=Path, default=Path("adapters/REGISTRY.json"))
    parser.add_argument("--force", action="store_true", help="Promote even if checks fail")
    parser.add_argument("--dry-run", action="store_true", help="Check but don't modify registry")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: results file not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    # Load
    new_scores = load_results(args.results)
    registry = json.loads(args.registry.read_text()) if args.registry.exists() else {}

    prod = registry.get("production", {})
    prod_name = prod.get("adapter", "(none)")
    prod_overall = prod.get("overall_score", 0.0)

    # Print scores
    print(f"  New adapter:  {args.adapter}")
    print(f"  Production:   {prod_name} ({prod_overall:.1f}%)")
    print(f"  New overall:  {new_scores['overall']:.1f}%")
    print()
    for cat in sorted(k for k in new_scores if k != "overall"):
        prod_cat = prod.get("scores", {}).get(cat, 0.0)
        delta = new_scores[cat] - prod_cat
        sign = "+" if delta >= 0 else ""
        print(f"    {cat:<20} {new_scores[cat]:>6.1f}%  (prod: {prod_cat:>6.1f}%, {sign}{delta:.1f}pp)")
    print()

    # Check
    passed, reasons = check_promotion(new_scores, registry)

    if passed:
        print(f"PASSED: {args.adapter} beats production ({new_scores['overall']:.1f}% > {prod_overall:.1f}%)")
        if args.dry_run:
            print("DRY RUN: registry not modified.")
        else:
            promote(args.adapter, new_scores, registry, args.registry)
            print(f"PROMOTED: {args.adapter} is now production.")
            print(f"Symlinked adapters/latest -> {args.adapter}")
    else:
        for r in reasons:
            print(f"REJECTED: {r}")
        if args.force and not args.dry_run:
            print()
            print("WARNING: --force specified, promoting despite failures.")
            promote(args.adapter, new_scores, registry, args.registry)
            print(f"FORCE PROMOTED: {args.adapter} is now production.")
        elif args.force:
            print("DRY RUN: would force-promote but --dry-run is set.")


if __name__ == "__main__":
    main()
