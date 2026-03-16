#!/usr/bin/env python3
from __future__ import annotations

"""
Dataset Quality Analyzer — comprehensive analytics for the Vertigo LoRA dataset.

Reports on:
- Size metrics (examples, tokens, files)
- Quality scores across dimensions
- Category and difficulty distributions
- Data source coverage
- Token length distribution
- Feature coverage (which Roblox APIs, patterns, conventions appear)
- Richness indicators (reasoning traces, multi-turn, tool calls)
- Growth tracking over time

Usage:
    python3 scripts/analyze_dataset.py                  # Analyze raw data
    python3 scripts/analyze_dataset.py --processed      # Analyze processed splits
    python3 scripts/analyze_dataset.py --compare A B    # Compare two snapshots
    python3 scripts/analyze_dataset.py --snapshot NAME  # Save snapshot for tracking
"""

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "snapshots"


@dataclass
class DatasetStats:
    """Comprehensive dataset statistics."""

    # Size
    total_examples: int = 0
    total_tokens_est: int = 0  # rough estimate: chars / 4
    total_chars: int = 0
    file_count: int = 0

    # Sources
    sources: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)
    difficulties: dict = field(default_factory=dict)

    # Quality
    with_reasoning: int = 0
    with_tool_calls: int = 0
    with_tool_responses: int = 0
    multi_turn: int = 0  # >2 messages
    verified: int = 0

    # Richness
    avg_assistant_length: float = 0.0
    median_assistant_length: float = 0.0
    min_assistant_length: int = 0
    max_assistant_length: int = 0

    # Feature coverage
    features_found: dict = field(default_factory=dict)
    roblox_services_covered: set = field(default_factory=set)
    vertigo_patterns_covered: set = field(default_factory=set)

    # Quality scores
    quality_scores: list = field(default_factory=list)


# Roblox services to check for coverage
ROBLOX_SERVICES = [
    "DataStoreService",
    "Players",
    "RunService",
    "ReplicatedStorage",
    "CollectionService",
    "UserInputService",
    "Workspace",
    "TweenService",
    "SoundService",
    "TextService",
    "Lighting",
    "HttpService",
    "MarketplaceService",
    "MemoryStoreService",
    "MessagingService",
    "PathfindingService",
    "PhysicsService",
    "TextChatService",
    "RemoteEvent",
    "RemoteFunction",
    "BindableEvent",
]

# Vertigo-specific patterns
VERTIGO_PATTERNS = [
    "Init/Start lifecycle",
    "Server-authoritative",
    "Rate limiting",
    "Schema migration",
    "DataStore retry",
    "KDTree spatial",
    "Deterministic RNG",
    "CollectionService tags",
    "WeldConstraint assembly",
    "@native hot path",
    "SIMD vector math",
    "Exponential damping",
    "Verlet integration",
    "Spring solver",
    "FlowField",
    "Grapple mechanics",
    "Glide mechanics",
    "Vehicle system",
    "Zone builder",
    "Telemetry funnel",
    "Actor memory",
    "Beacon placement",
    "Checkpoint system",
    "Ability chaining",
]

# Pattern detection regexes
PATTERN_DETECTORS = {
    "Init/Start lifecycle": [":Init()", ":Start()"],
    "Server-authoritative": ["IsServer", "OnServerEvent", "FireClient"],
    "Rate limiting": ["rate_limit", "lastRequestTime", "MAX_RPS", "debounce"],
    "Schema migration": ["schemaVersion", "migrateSchema", "CURRENT_SCHEMA"],
    "DataStore retry": ["MAX_RETRIES", "RETRY_DELAY", "pcall"],
    "KDTree spatial": ["KDTree", "findNearest", "spatialQuery"],
    "Deterministic RNG": ["Random.new(", "seed"],
    "CollectionService tags": ["CollectionService", "GetTagged", "AddTag"],
    "WeldConstraint assembly": ["WeldConstraint", "Part0", "Part1"],
    "@native hot path": ["@native", "--[[@native]]"],
    "SIMD vector math": [":Dot(", "vector.", "math.lerp"],
    "Exponential damping": ["math.exp("],
    "Verlet integration": ["Verlet", "previousPosition", "integratePoint"],
    "Spring solver": ["Spring", "stiffness", "damping", "zeta"],
    "FlowField": ["FlowField", "trilinear", "cellSize"],
    "Grapple mechanics": ["grapple", "Grapple", "hookSpeed", "reelSpeed"],
    "Glide mechanics": ["glide", "Glide", "liftForce", "airControl"],
    "Vehicle system": ["Vehicle", "DirtBike", "Kite", "NetworkOwner"],
    "Zone builder": ["Builder", "build(", "BuildOptions"],
    "Telemetry funnel": ["Telemetry", "funnel", "aggregate"],
    "Actor memory": ["actorMemory", "interactions", "LivingMemory"],
    "Beacon placement": ["Beacon", "beacon"],
    "Checkpoint system": ["Checkpoint", "respawn"],
    "Ability chaining": ["graceWindow", "chainWindow", "grappleWallChain"],
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4 for English/code)."""
    return len(text) // 4


def analyze_example(example: dict, stats: DatasetStats):
    """Analyze a single training example."""
    messages = example.get("messages", [])
    stats.total_examples += 1

    # Source and category
    source = example.get("source", "unknown")
    stats.sources[source] = stats.sources.get(source, 0) + 1

    category = example.get("category", "unknown")
    stats.categories[category] = stats.categories.get(category, 0) + 1

    difficulty = example.get("difficulty", 0)
    stats.difficulties[difficulty] = stats.difficulties.get(difficulty, 0) + 1

    # Multi-turn
    if len(messages) > 3:
        stats.multi_turn += 1

    # Verified
    if example.get("verified"):
        stats.verified += 1

    # Tool calls
    has_tools = "tools" in example
    if has_tools:
        stats.with_tool_calls += 1

    has_tool_response = any(m.get("role") == "tool" for m in messages)
    if has_tool_response:
        stats.with_tool_responses += 1

    # Analyze assistant content
    assistant_content = ""
    for m in messages:
        if m.get("role") == "assistant" and m.get("content"):
            assistant_content += m["content"]

    char_count = len(assistant_content)
    stats.total_chars += char_count
    stats.total_tokens_est += estimate_tokens(assistant_content)
    stats.quality_scores.append(char_count)

    # Reasoning
    if "<think>" in assistant_content:
        stats.with_reasoning += 1

    # Feature coverage
    full_text = assistant_content
    for m in messages:
        if m.get("content"):
            full_text += m["content"]

    for service in ROBLOX_SERVICES:
        if service in full_text:
            stats.roblox_services_covered.add(service)

    for pattern, detectors in PATTERN_DETECTORS.items():
        if any(d in full_text for d in detectors):
            stats.vertigo_patterns_covered.add(pattern)


def analyze_directory(data_dir: Path) -> DatasetStats:
    """Analyze all JSONL files in a directory."""
    stats = DatasetStats()

    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        stats.file_count += 1
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    analyze_example(example, stats)
                except json.JSONDecodeError:
                    continue

    # Compute length stats
    if stats.quality_scores:
        scores = sorted(stats.quality_scores)
        stats.avg_assistant_length = sum(scores) / len(scores)
        stats.median_assistant_length = scores[len(scores) // 2]
        stats.min_assistant_length = scores[0]
        stats.max_assistant_length = scores[-1]

    return stats


def print_report(stats: DatasetStats, title: str = "Dataset Analysis"):
    """Print a comprehensive report."""
    w = 60

    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")

    # Size
    print(f"\n{'SIZE':─^{w}}")
    print(f"  Total examples:      {stats.total_examples:>8,}")
    print(f"  Estimated tokens:    {stats.total_tokens_est:>8,}")
    print(f"  Total characters:    {stats.total_chars:>8,}")
    print(f"  Source files:         {stats.file_count:>8}")

    # Quality
    print(f"\n{'QUALITY':─^{w}}")
    pct = lambda n: f"{n / max(stats.total_examples, 1) * 100:.1f}%"
    print(f"  With reasoning (<think>):  {stats.with_reasoning:>5} ({pct(stats.with_reasoning)})")
    print(f"  With tool calls:           {stats.with_tool_calls:>5} ({pct(stats.with_tool_calls)})")
    print(f"  With tool responses:       {stats.with_tool_responses:>5} ({pct(stats.with_tool_responses)})")
    print(f"  Multi-turn (>3 msgs):      {stats.multi_turn:>5} ({pct(stats.multi_turn)})")
    print(f"  Verified (Luau CLI):       {stats.verified:>5} ({pct(stats.verified)})")

    # Length distribution
    print(f"\n{'ASSISTANT RESPONSE LENGTH (chars)':─^{w}}")
    print(f"  Min:     {stats.min_assistant_length:>8,}")
    print(f"  Median:  {stats.median_assistant_length:>8,.0f}")
    print(f"  Mean:    {stats.avg_assistant_length:>8,.0f}")
    print(f"  Max:     {stats.max_assistant_length:>8,}")

    # Sources
    print(f"\n{'DATA SOURCES':─^{w}}")
    for source, count in sorted(stats.sources.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, count // max(1, stats.total_examples // 40))
        print(f"  {source:<25} {count:>5} {bar}")

    # Categories
    print(f"\n{'CATEGORIES':─^{w}}")
    for cat, count in sorted(stats.categories.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, count // max(1, stats.total_examples // 40))
        print(f"  {cat:<25} {count:>5} {bar}")

    # Difficulty
    print(f"\n{'DIFFICULTY DISTRIBUTION':─^{w}}")
    for d in sorted(stats.difficulties.keys()):
        count = stats.difficulties[d]
        bar = "█" * min(40, count // max(1, stats.total_examples // 40))
        print(f"  Level {d}: {count:>5} {bar}")

    # Roblox service coverage
    print(f"\n{'ROBLOX SERVICE COVERAGE':─^{w}}")
    covered = len(stats.roblox_services_covered)
    total = len(ROBLOX_SERVICES)
    print(f"  {covered}/{total} services covered ({covered / total * 100:.0f}%)")
    missing = set(ROBLOX_SERVICES) - stats.roblox_services_covered
    if missing:
        print(f"  Missing: {', '.join(sorted(missing))}")

    # Vertigo pattern coverage
    print(f"\n{'VERTIGO PATTERN COVERAGE':─^{w}}")
    covered = len(stats.vertigo_patterns_covered)
    total = len(VERTIGO_PATTERNS)
    print(f"  {covered}/{total} patterns covered ({covered / total * 100:.0f}%)")
    missing = set(VERTIGO_PATTERNS) - stats.vertigo_patterns_covered
    if missing:
        print(f"  Missing: {', '.join(sorted(missing))}")

    # Overall score
    print(f"\n{'OVERALL DATASET SCORE':─^{w}}")
    scores = {
        "Size (>500 examples)": min(1.0, stats.total_examples / 500),
        "Reasoning coverage": stats.with_reasoning / max(1, stats.total_examples),
        "Service coverage": len(stats.roblox_services_covered) / len(ROBLOX_SERVICES),
        "Pattern coverage": len(stats.vertigo_patterns_covered) / len(VERTIGO_PATTERNS),
        "Multi-turn ratio": min(1.0, stats.multi_turn / max(1, stats.total_examples) * 5),
        "Tool call coverage": min(1.0, stats.with_tool_calls / max(1, stats.total_examples) * 3),
    }
    for name, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {name:<30} {bar} {score * 100:>5.1f}%")

    total_score = sum(scores.values()) / len(scores) * 100
    print(f"\n  {'COMPOSITE SCORE:':<30} {'':>20} {total_score:>5.1f}%")
    print(f"{'=' * w}\n")

    return total_score


def save_snapshot(stats: DatasetStats, name: str):
    """Save a snapshot for growth tracking."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "total_examples": stats.total_examples,
        "total_tokens_est": stats.total_tokens_est,
        "sources": stats.sources,
        "categories": stats.categories,
        "difficulties": stats.difficulties,
        "with_reasoning": stats.with_reasoning,
        "with_tool_calls": stats.with_tool_calls,
        "multi_turn": stats.multi_turn,
        "roblox_services_covered": sorted(stats.roblox_services_covered),
        "vertigo_patterns_covered": sorted(stats.vertigo_patterns_covered),
    }
    path = SNAPSHOTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Snapshot saved: {path}")


def compare_snapshots(a_path: str, b_path: str):
    """Compare two snapshots."""
    with open(SNAPSHOTS_DIR / f"{a_path}.json") as f:
        a = json.load(f)
    with open(SNAPSHOTS_DIR / f"{b_path}.json") as f:
        b = json.load(f)

    print(f"\n{'Comparing Snapshots':=^60}")
    print(f"  A: {a['name']} ({a['timestamp'][:10]})")
    print(f"  B: {b['name']} ({b['timestamp'][:10]})")
    print()

    metrics = [
        ("Total examples", a["total_examples"], b["total_examples"]),
        ("Est. tokens", a["total_tokens_est"], b["total_tokens_est"]),
        ("With reasoning", a["with_reasoning"], b["with_reasoning"]),
        ("With tool calls", a["with_tool_calls"], b["with_tool_calls"]),
        ("Multi-turn", a["multi_turn"], b["multi_turn"]),
        ("Services covered", len(a["roblox_services_covered"]), len(b["roblox_services_covered"])),
        ("Patterns covered", len(a["vertigo_patterns_covered"]), len(b["vertigo_patterns_covered"])),
    ]

    print(f"  {'Metric':<25} {'A':>8} {'B':>8} {'Delta':>8}")
    print(f"  {'-' * 49}")
    for name, va, vb in metrics:
        delta = vb - va
        sign = "+" if delta > 0 else ""
        print(f"  {name:<25} {va:>8,} {vb:>8,} {sign}{delta:>7,}")


def main():
    parser = argparse.ArgumentParser(description="Vertigo LoRA Dataset Analyzer")
    parser.add_argument("--processed", action="store_true", help="Analyze processed splits")
    parser.add_argument("--snapshot", type=str, help="Save snapshot with given name")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two snapshots")
    args = parser.parse_args()

    if args.compare:
        compare_snapshots(args.compare[0], args.compare[1])
        return

    data_dir = PROC_DIR if args.processed else RAW_DIR
    title = "Processed Dataset" if args.processed else "Raw Dataset"

    if not data_dir.exists() or not list(data_dir.glob("*.jsonl")):
        print(f"No JSONL files found in {data_dir}")
        return

    stats = analyze_directory(data_dir)
    print_report(stats, title)

    if args.snapshot:
        save_snapshot(stats, args.snapshot)


if __name__ == "__main__":
    main()
