#!/usr/bin/env python3
from __future__ import annotations

"""Generate a comprehensive analysis report after a training run.

Called by auto_train.sh. Produces a markdown report in docs/runs/ with:
- Training summary (config, model, hyperparams, val loss)
- Per-category benchmark scores with deltas vs base and production
- Per-task breakdown (best/worst performing tasks)
- Data stats (dataset size, mix ratios, curated vs raw)
- Promotion decision with reasoning
- Recommendations for next run
"""

import argparse
import json
import socket
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def load_results(path: Path) -> list[dict]:
    """Load benchmark results as list of task dicts."""
    results = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def load_config(path: Path) -> dict:
    """Parse YAML config into a dict (simple key: value parsing)."""
    cfg = {}
    if not path.exists():
        return cfg
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        cfg[key.strip()] = val.strip().strip('"')
    return cfg


def load_adapter_config(adapter_dir: Path) -> dict:
    """Load adapter_config.json if it exists."""
    p = adapter_dir / "adapter_config.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def category_scores(results: list[dict]) -> dict[str, dict]:
    """Compute per-category stats."""
    cats: dict[str, list[float]] = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r["scores"]["overall"])

    stats = {}
    for cat, vals in sorted(cats.items()):
        stats[cat] = {
            "mean": round(sum(vals) / len(vals) * 100, 1),
            "min": round(min(vals) * 100, 1),
            "max": round(max(vals) * 100, 1),
            "count": len(vals),
        }
    all_vals = [v for vs in cats.values() for v in vs]
    stats["overall"] = {
        "mean": round(sum(all_vals) / len(all_vals) * 100, 1),
        "count": len(all_vals),
    }
    return stats


def task_rankings(results: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return top 5 and bottom 5 tasks by score."""
    sorted_tasks = sorted(results, key=lambda r: r["scores"]["overall"], reverse=True)
    top = sorted_tasks[:5]
    bottom = sorted_tasks[-5:]
    return top, bottom


def data_stats() -> dict:
    """Compute dataset statistics."""
    raw_dir = PROJECT_DIR / "data" / "raw"
    proc_dir = PROJECT_DIR / "data" / "processed"

    stats = {"raw_files": 0, "raw_examples": 0, "train": 0, "valid": 0, "test": 0}

    if raw_dir.exists():
        for f in raw_dir.glob("*.jsonl"):
            count = sum(1 for line in f.read_text().splitlines() if line.strip())
            stats["raw_files"] += 1
            stats["raw_examples"] += count

    for split in ["train", "valid", "test"]:
        p = proc_dir / f"{split}.jsonl"
        if p.exists():
            stats[split] = sum(1 for line in p.read_text().splitlines() if line.strip())

    return stats


def _load_from_index(index_path: Path) -> dict[str, dict]:
    """Load run data from RESULTS_INDEX.json (fast, canonical)."""
    index = json.loads(index_path.read_text())
    runs: dict[str, dict] = {}

    # Base scores
    for label, entry in index.get("base_scores", {}).items():
        scores = {k: v for k, v in entry.items() if k not in ("date", "benchmark")}
        runs[label] = {
            "file": f"base_{label}.jsonl",
            "model": label,
            "adapter": "none",
            "scores": scores,
            "overall": entry.get("overall", 0),
        }

    # Adapters
    for label, entry in index.get("adapters", {}).items():
        scores = {k: v for k, v in entry.items() if k not in ("date", "benchmark", "status")}
        runs[label] = {
            "file": f"results_{label}.jsonl",
            "model": "adapter",
            "adapter": label,
            "scores": scores,
            "overall": entry.get("overall", 0),
        }

    return runs


def _scan_jsonl_results() -> dict[str, dict]:
    """Fallback: scan data/eval/ for result JSONL files and compute scores."""
    eval_dir = PROJECT_DIR / "data" / "eval"
    if not eval_dir.exists():
        return {}

    runs: dict[str, dict] = {}
    for f in sorted(eval_dir.glob("results_*.jsonl")):
        results = load_results(f)
        if not results:
            continue

        cats = category_scores(results)
        label = f.stem.replace("results_", "")
        model = results[0].get("model", "?") if results else "?"
        adapter = results[0].get("adapter", "none") if results else "none"

        runs[label] = {
            "file": f.name,
            "model": model,
            "adapter": adapter,
            "scores": {c: v.get("mean", 0) for c, v in cats.items()},
            "overall": cats.get("overall", {}).get("mean", 0),
        }

    # Also include base rescore if it exists
    base_file = eval_dir / "base_4b_rescore.jsonl"
    if base_file.exists() and "base_4b_rescore" not in runs:
        results = load_results(base_file)
        if results:
            cats = category_scores(results)
            runs["4b_base (current)"] = {
                "file": base_file.name,
                "model": results[0].get("model", "?"),
                "adapter": "none",
                "scores": {c: v.get("mean", 0) for c, v in cats.items()},
                "overall": cats.get("overall", {}).get("mean", 0),
            }

    return runs


def update_results_index(label: str, scores: dict, adapter: str | None = None) -> None:
    """Append a new run to RESULTS_INDEX.json."""
    index_path = PROJECT_DIR / "data" / "eval" / "RESULTS_INDEX.json"
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text())
    target = "adapters" if adapter else "base_scores"
    entry = {k: v for k, v in scores.items()}
    entry["date"] = datetime.now().strftime("%Y-%m-%d")
    entry["benchmark"] = index.get("benchmark_version", "unknown")
    if adapter:
        entry["status"] = "candidate"
    index.setdefault(target, {})[label] = entry
    index["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    index_path.write_text(json.dumps(index, indent=4) + "\n")


def load_all_results() -> dict[str, dict]:
    """Load all benchmark results — from index if available, JSONL fallback."""
    index_path = PROJECT_DIR / "data" / "eval" / "RESULTS_INDEX.json"
    if index_path.exists():
        return _load_from_index(index_path)
    return _scan_jsonl_results()


def format_comparison_matrix(all_runs: dict[str, dict], current_label: str) -> list[str]:
    """Format a markdown table comparing all runs side by side."""
    if not all_runs:
        return []

    # Sort by overall score descending
    sorted_runs = sorted(all_runs.items(), key=lambda x: x[1]["overall"], reverse=True)
    categories = ["coding", "bugfix", "architecture", "mcp_tool_calling", "embodiment", "overall"]

    lines = [
        "## Comparison Matrix (all runs)",
        "",
        "| Run | Model | " + " | ".join(c[:7] for c in categories) + " |",
        "|---|---|" + "|".join(["---|"] * len(categories)),
    ]

    for label, run in sorted_runs:
        marker = " **←**" if label == current_label else ""
        model_short = run["model"].split("/")[-1][:20] if "/" in str(run["model"]) else str(run["model"])[:20]
        scores_str = " | ".join(
            f"**{run['scores'].get(c, 0):.1f}%**" if c == "overall" else f"{run['scores'].get(c, 0):.1f}%"
            for c in categories
        )
        lines.append(f"| {label}{marker} | {model_short} | {scores_str} |")

    lines.append("")
    return lines


def generate_report(
    adapter: str,
    config_path: Path,
    results_path: Path,
    registry_path: Path,
) -> str:
    """Generate a comprehensive markdown analysis report."""
    now = datetime.now()
    host = socket.gethostname().split(".")[0]
    results = load_results(results_path)
    cfg = load_config(config_path)
    adapter_dir = PROJECT_DIR / "adapters" / adapter
    adapter_cfg = load_adapter_config(adapter_dir)
    cats = category_scores(results)
    top, bottom = task_rankings(results)
    data = data_stats()

    # Load registry for comparison
    reg = {}
    base_scores = {}
    prod_scores = {}
    if registry_path.exists():
        reg = json.loads(registry_path.read_text())
        prod_scores = reg.get("production", {}).get("scores", {})

    # Import BASE_SCORES
    try:
        import sys
        sys.path.insert(0, str(SCRIPT_DIR))
        from promote_adapter import BASE_SCORES
        base_scores = BASE_SCORES
    except ImportError:
        base_scores = {}

    overall = cats.get("overall", {}).get("mean", 0)
    base_overall = base_scores.get("overall", 0)
    prod_overall = reg.get("production", {}).get("overall_score", 0)
    prod_adapter = reg.get("production", {}).get("adapter", "none")

    lines = [
        f"# Run Analysis: {adapter}",
        "",
        f"**Date:** {now.strftime('%Y-%m-%d %H:%M')}",
        f"**Host:** {host}",
        f"**Config:** {config_path.name}",
        f"**Model:** {cfg.get('model', adapter_cfg.get('model', '?'))}",
        "",
        "---",
        "",
        "## Training Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Model | {cfg.get('model', adapter_cfg.get('model', '?'))} |",
        f"| Rank | {adapter_cfg.get('lora_parameters', {}).get('rank', cfg.get('rank', '?'))} |",
        f"| Layers | {adapter_cfg.get('num_layers', cfg.get('num_layers', '?'))} |",
        f"| Seq Length | {adapter_cfg.get('max_seq_length', cfg.get('max_seq_length', '?'))} |",
        f"| Learning Rate | {adapter_cfg.get('learning_rate', cfg.get('learning_rate', '?'))} |",
        f"| Iterations | {adapter_cfg.get('iters', cfg.get('iters', '?'))} |",
        f"| Batch Size | {adapter_cfg.get('batch_size', cfg.get('batch_size', '?'))} |",
        "",
        "## Benchmark Results",
        "",
        f"| Category | Score | vs Base | vs Production ({prod_adapter}) |",
        "|---|---|---|---|",
    ]

    for cat in ["coding", "bugfix", "architecture", "mcp_tool_calling", "embodiment"]:
        score = cats.get(cat, {}).get("mean", 0)
        base = base_scores.get(cat, 0)
        prod = prod_scores.get(cat, 0)
        delta_base = score - base
        delta_prod = score - prod
        sign_b = "+" if delta_base >= 0 else ""
        sign_p = "+" if delta_prod >= 0 else ""
        lines.append(
            f"| {cat} | **{score:.1f}%** | {sign_b}{delta_base:.1f}pp | {sign_p}{delta_prod:.1f}pp |"
        )

    delta_base_o = overall - base_overall
    delta_prod_o = overall - prod_overall
    sign_bo = "+" if delta_base_o >= 0 else ""
    sign_po = "+" if delta_prod_o >= 0 else ""
    lines.extend([
        f"| **Overall** | **{overall:.1f}%** | **{sign_bo}{delta_base_o:.1f}pp** | **{sign_po}{delta_prod_o:.1f}pp** |",
        "",
        "## Task Breakdown",
        "",
        "### Top 5 Tasks",
        "",
        "| Task | Category | Difficulty | Score |",
        "|---|---|---|---|",
    ])
    for r in top:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['difficulty']} | {r['scores']['overall']*100:.1f}% |"
        )

    lines.extend([
        "",
        "### Bottom 5 Tasks",
        "",
        "| Task | Category | Difficulty | Score |",
        "|---|---|---|---|",
    ])
    for r in bottom:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['difficulty']} | {r['scores']['overall']*100:.1f}% |"
        )

    # Luau compile rate
    compile_count = sum(1 for r in results if r.get("scores", {}).get("luau_compiles"))
    lines.extend([
        "",
        "## Quality Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total tasks | {len(results)} |",
        f"| Luau compile rate | {compile_count}/{len(results)} ({compile_count/max(len(results),1)*100:.0f}%) |",
        f"| Mean correctness | {sum(r['scores']['correctness'] for r in results)/max(len(results),1)*100:.1f}% |",
        f"| Mean convention | {sum(r['scores'].get('convention_score', 0) or 0 for r in results)/max(len(results),1)*100:.1f}% |",
    ])

    # Data stats
    lines.extend([
        "",
        "## Dataset",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Raw files | {data['raw_files']} |",
        f"| Raw examples | {data['raw_examples']} |",
        f"| Train split | {data['train']} |",
        f"| Valid split | {data['valid']} |",
        f"| Test split | {data['test']} |",
    ])

    # Promotion
    promoted = overall > prod_overall and overall >= reg.get("minimum_score", 0)
    lines.extend([
        "",
        "## Promotion Decision",
        "",
        f"- Overall: {overall:.1f}% {'>' if overall > prod_overall else '<='} production {prod_overall:.1f}%",
        f"- Above minimum ({reg.get('minimum_score', 0):.1f}%): {'Yes' if overall >= reg.get('minimum_score', 0) else 'No'}",
        f"- **{'PROMOTED' if promoted else 'REJECTED'}**",
    ])

    # Comparison matrix
    all_runs = load_all_results()
    # Find label for current run
    current_label = None
    for label, run in all_runs.items():
        if run.get("file") == results_path.name:
            current_label = label
            break
    matrix_lines = format_comparison_matrix(all_runs, current_label)
    if matrix_lines:
        lines.extend(matrix_lines)

    # Recommendations
    lines.extend([
        "",
        "## Recommendations",
        "",
    ])

    if overall > prod_overall:
        lines.append("- New production adapter. Update serving endpoint.")
    if any(cats.get(c, {}).get("mean", 100) < base_scores.get(c, 0) for c in base_scores if c != "overall"):
        regressed = [c for c in base_scores if c != "overall" and cats.get(c, {}).get("mean", 100) < base_scores.get(c, 0)]
        lines.append(f"- Category regressions vs base: {', '.join(regressed)}. Consider adding more training data in these areas.")
    if compile_count == 0:
        lines.append("- Zero Luau compile rate. Investigate why generated code doesn't compile.")
    if overall < prod_overall:
        lines.append("- Did not beat production. Consider: different learning rate, more/less data, different rank.")

    bottom_cats = sorted(cats.items(), key=lambda x: x[1].get("mean", 0) if x[0] != "overall" else 999)
    if bottom_cats and bottom_cats[0][0] != "overall":
        weakest = bottom_cats[0]
        lines.append(f"- Weakest category: {weakest[0]} ({weakest[1]['mean']:.1f}%). Prioritize data generation for this category.")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate comprehensive run analysis")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--registry", default="adapters/REGISTRY.json")
    ap.add_argument("--output", default=None, help="Output path (default: docs/runs/<adapter>.md)")
    args = ap.parse_args()

    output = Path(args.output) if args.output else PROJECT_DIR / "docs" / "runs" / f"{args.adapter}.md"
    output.parent.mkdir(parents=True, exist_ok=True)

    report = generate_report(
        adapter=args.adapter,
        config_path=Path(args.config),
        results_path=Path(args.results),
        registry_path=Path(args.registry),
    )

    output.write_text(report)
    print(f"  Analysis: {output}")

    # Also print summary to stdout
    results = load_results(Path(args.results))
    cats = category_scores(results)
    overall = cats.get("overall", {}).get("mean", 0)
    print(f"  Overall: {overall:.1f}%")
    for cat in ["coding", "bugfix", "architecture", "mcp_tool_calling", "embodiment"]:
        score = cats.get(cat, {}).get("mean", 0)
        print(f"    {cat:<20} {score:.1f}%")


if __name__ == "__main__":
    main()
