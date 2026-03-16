#!/usr/bin/env python3
from __future__ import annotations

"""
Teacher Distillation — generate training data by capturing a strong teacher
model's responses to benchmark tasks and their variants.

Sends benchmark tasks to a teacher model (e.g. 27B via LM Studio), scores
responses with the same pattern-matching scorer as run_benchmark.py, keeps
the best sample per task, and outputs SFT-ready training JSONL.

Usage:
    uv run python scripts/distill_from_teacher.py --teacher-model qwen3.5-27b --samples 2
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scoring import (
    VERTIGO_SYSTEM_PROMPT,
    score_code_presence,
    score_convention,
    score_correctness,
    score_failure_penalty,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_BENCHMARK = PROJECT_DIR / "data" / "eval" / "benchmark.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "raw" / "teacher_distillation.jsonl"
DEFAULT_API = "http://127.0.0.1:1234/v1"
REQUEST_TIMEOUT = 180


def score_task_simple(response: str, task: dict) -> float:
    """Lightweight scorer returning a single float (used for distillation ranking)."""
    category = task.get("category", "coding")
    correctness = score_correctness(response, task.get("expected_patterns", []))
    convention = score_convention(response, category)
    code_pres = score_code_presence(response, category)
    penalty = score_failure_penalty(response)

    weights: dict[str, tuple[float, float]] = {"correctness": (correctness, 0.4)}
    if convention is not None:
        weights["convention"] = (convention, 0.2)
    if code_pres is not None:
        weights["code_presence"] = (code_pres, 0.2)

    total_w = sum(w for _, w in weights.values())
    overall = sum(s * w / total_w for s, w in weights.values())
    return round(overall * penalty, 3)


# --- Variant generation ---

VARIANT_MAP: dict[str, list[dict]] = {
    "code_01": [
        {
            "prompt": "Write a Vertigo service module called AchievementService that follows the Init/Start lifecycle pattern. It should initialize an empty achievement tracker cache in Init and connect to Players.PlayerAdded in Start to load each player's unlocked achievements.",
            "expected_patterns": ["--!strict", ":Init\\(\\)", ":Start\\(\\)", "return\\s+\\w+Service"],
        },
    ],
    "code_02": [
        {
            "prompt": "Write a zone builder module that creates glowing mushroom geometry for a swamp zone. It should use CollectionService to tag mushrooms as 'GlowShroom', create BaseParts with ForceField material and random scales, and be optimized with @native on the hot geometry generation function.",
            "expected_patterns": ["CollectionService", "Instance\\.new", "@native", "GlowShroom"],
        },
    ],
    "code_03": [
        {
            "prompt": "Write an air dash ability controller for the client side. It should listen for RequestUseAbility, apply an instant velocity burst in the player's look direction, consume a charge, and use @native on the per-frame cooldown tick function.",
            "expected_patterns": ["RequestUseAbility", "velocity|Velocity|burst", "charge|cooldown", "@native"],
        },
    ],
    "code_05": [
        {
            "prompt": "Write a leaderboard service that reads and writes player scores to an OrderedDataStore. Use pcall for error handling, implement a cache layer to avoid excessive reads, and follow the Init/Start lifecycle.",
            "expected_patterns": ["OrderedDataStore|DataStoreService", "pcall", "cache|Cache|_cache", ":Init\\(\\)"],
        },
    ],
    "code_09": [
        {
            "prompt": "Write a loot spawner system that uses CollectionService tags to find spawn points. Points tagged 'LootSpawn' should periodically create collectible items. Track active loot count and use Debris:AddItem for cleanup after a timeout.",
            "expected_patterns": ["CollectionService", "GetTagged", "LootSpawn", "Debris|AddItem"],
        },
    ],
    "fix_01": [
        {
            "prompt": 'Fix this buggy Vertigo service module. It has a type annotation issue that will cause problems:\n\n```luau\n--!strict\nlocal ReplicatedStorage = game:GetService("ReplicatedStorage")\n\nlocal HealthService = {}\n\nfunction HealthService:Init()\n    self._health = {}\nend\n\nfunction HealthService:GetHealth(player)\n    return self._health[player] or 100\nend\n\nfunction HealthService:Start()\n    print("HealthService started")\nend\n\nreturn HealthService\n```\n\nWhat is wrong and how do you fix it?',
            "expected_patterns": ["type.*annot|strict.*type|parameter.*type", "player.*Player|: Player"],
        },
    ],
    "fix_02": [
        {
            "prompt": 'Fix this performance bug. The code creates a new connection every time a tagged instance is added, without ever disconnecting:\n\n```luau\n--!strict\nlocal CollectionService = game:GetService("CollectionService")\nlocal RunService = game:GetService("RunService")\n\nlocal function setupGlow(part: BasePart)\n    RunService.Heartbeat:Connect(function(dt)\n        part.Transparency = 0.5 + math.sin(os.clock() * 3) * 0.3\n    end)\nend\n\nCollectionService:GetInstanceAddedSignal("GlowPart"):Connect(setupGlow)\nfor _, part in CollectionService:GetTagged("GlowPart") do\n    setupGlow(part)\nend\n```\n\nRefactor to properly clean up connections when instances are removed.',
            "expected_patterns": [
                "disconnect|Disconnect|:Destroy|Trove|Maid|cleanup",
                "InstanceRemovedSignal|GetInstanceRemovedSignal",
                "connection|Connection|_conn",
            ],
        },
    ],
    "fix_04": [
        {
            "prompt": 'Fix this RemoteEvent handler that is vulnerable to client exploitation — it trusts client-sent values without validation:\n\n```luau\n--!strict\nlocal ReplicatedStorage = game:GetService("ReplicatedStorage")\nlocal remote = ReplicatedStorage:WaitForChild("GiveCoins")\n\nremote.OnServerEvent:Connect(function(player, amount)\n    local profile = getProfile(player)\n    profile.coins += amount\n    saveProfile(player, profile)\nend)\n```\n\nAdd proper server-authoritative validation.',
            "expected_patterns": ["valid|sanity|check|clamp", "type.*number|typeof", "max|MAX|limit|cap"],
        },
    ],
    "arch_01": [
        {
            "prompt": "Design the Init/Start boot ordering for these 4 interdependent Vertigo services:\n- ConfigService: loads shared config tables\n- AudioService: needs config for volume defaults from ConfigService\n- UIService: needs audio references from AudioService for button sounds\n- InputService: needs UI bindings from UIService before accepting input\n\nProvide the boot order, explain why, and show the init.client.luau boot sequence code.",
            "expected_patterns": [
                "Init.*Start|boot.*order",
                "ConfigService.*AudioService|sequential",
                "require",
                "InputService.*last|final",
            ],
        },
    ],
    "arch_02": [
        {
            "prompt": "Design server-authoritative validation for a Vertigo vehicle spawn flow. A client requests to spawn a DirtBike. Show the full request/validation/response flow including:\n- Client sends RequestSpawnVehicle with vehicle ID and desired position\n- Server validates: ownership, cooldown, spawn zone, max active vehicles\n- Server spawns or rejects with reason\n- Client receives result and enters vehicle\n\nProvide both client and server code.",
            "expected_patterns": [
                "RequestSpawnVehicle|OnServerEvent",
                "valid|sanity|check",
                "FireClient|StateSync",
                "cooldown|ownership|zone",
            ],
        },
    ],
    "play_01": [
        {
            "prompt": "You are an agent embodied in the Vertigo world at the Crystal Caverns (Y=25). Navigate down to the Abyss zone (Y=-80) using ability chains. Describe your traversal plan: which anchor points to target, how to manage fall speed with glide, and what hazards to avoid.",
            "expected_patterns": [
                "grapple|Grapple|glide|Glide",
                "anchor|Anchor",
                "Y.*-80|Abyss|abyss|downward",
                "chain|sequence|hazard",
            ],
        },
    ],
    "play_02": [
        {
            "prompt": "You are an agent in the Vertigo world. Chain abilities for maximum vertical height gain: wall run to build upward momentum, release into a grapple to an overhead anchor, then air dash upward at the peak. Describe the exact input sequence, timing windows, and expected altitude gain.",
            "expected_patterns": [
                "wall.?run.*grapple.*air.?dash|chain",
                "timing|window|grace",
                "velocity|speed|momentum",
                "height|vertical|altitude",
            ],
        },
    ],
}


def generate_variants(task: dict) -> list[dict]:
    """Generate variant tasks for a given benchmark task."""
    task_id = task.get("id", "")
    templates = VARIANT_MAP.get(task_id, [])
    variants = []
    for i, tmpl in enumerate(templates, 1):
        variant = dict(task)
        variant["id"] = f"{task_id}_var{i}"
        variant["prompt"] = tmpl["prompt"]
        variant["expected_patterns"] = tmpl["expected_patterns"]
        variant["is_variant"] = True
        variant["source_task"] = task_id
        variants.append(variant)
    return variants


# --- API calls ---


def check_api(api_base: str) -> str | None:
    """Check API health. Returns model ID or None on failure."""
    from api import detect_model as _detect_model

    try:
        return _detect_model(api_base=api_base)
    except Exception:
        return None


def call_teacher(
    api_base: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion request to the teacher model."""
    from api import call_llm

    if tools:
        tool_desc = json.dumps(tools, indent=2)
        messages = [dict(m) for m in messages]
        messages[0] = dict(messages[0])
        messages[0]["content"] = (
            messages[0]["content"]
            + "\n\nYou have access to the following tools:\n```json\n"
            + tool_desc
            + "\n```\nCall tools by responding with a JSON tool_call block."
        )

    result = call_llm(
        messages,
        model=model,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=REQUEST_TIMEOUT,
    )
    choice = result["choices"][0]
    content = choice.get("message", {}).get("content", "")
    tool_calls = choice.get("message", {}).get("tool_calls")
    if tool_calls:
        content += "\n" + json.dumps(tool_calls)
    return content


def build_messages(task: dict) -> list[dict]:
    return [
        {"role": "system", "content": VERTIGO_SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]


def format_training_example(
    task: dict,
    response: str,
    eval_score: float,
    teacher_model: str,
) -> dict:
    """Format a scored response as an SFT training example."""
    has_reasoning = bool(re.search(r"<think>|because|reason|let me|first,|step \d", response, re.IGNORECASE))
    return {
        "messages": [
            {"role": "system", "content": VERTIGO_SYSTEM_PROMPT},
            {"role": "user", "content": task["prompt"]},
            {"role": "assistant", "content": response},
        ],
        "source": "teacher_distillation",
        "category": task.get("category", "coding"),
        "difficulty": task.get("difficulty", 1),
        "has_reasoning": has_reasoning,
        "teacher_model": teacher_model,
        "rights_basis": "generated",
        "task_family": "sft_scripter",
        "eval_score": eval_score,
        "task_id": task.get("id", "unknown"),
        "is_variant": task.get("is_variant", False),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher Distillation Data Generator")
    parser.add_argument("--api", type=str, default=DEFAULT_API, help="OpenAI-compatible API URL")
    parser.add_argument("--teacher-model", type=str, default=None, help="Teacher model name (auto-detect if omitted)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK, help="Benchmark JSONL path")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--samples", type=int, default=2, help="Samples per task (keep best)")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum score to include")
    args = parser.parse_args()

    # Check API
    print(f"Checking API at {args.api} ...")
    detected = check_api(args.api)
    if detected is None:
        print("ERROR: Cannot reach API. Is LM Studio running with the teacher model loaded?", file=sys.stderr)
        sys.exit(1)

    teacher_model = args.teacher_model or detected
    print(f"Teacher model: {teacher_model} (detected: {detected})")

    # Load benchmark
    if not args.benchmark.exists():
        print(f"ERROR: benchmark not found: {args.benchmark}", file=sys.stderr)
        sys.exit(1)

    tasks: list[dict] = []
    with open(args.benchmark) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    print(f"Loaded {len(tasks)} benchmark tasks")

    # Generate variants
    all_tasks: list[dict] = []
    for task in tasks:
        all_tasks.append(task)
        variants = generate_variants(task)
        all_tasks.extend(variants)

    variant_count = len(all_tasks) - len(tasks)
    print(f"Generated {variant_count} variants -> {len(all_tasks)} total tasks")

    # Process each task
    examples: list[dict] = []
    skipped = 0
    errors = 0

    for i, task in enumerate(all_tasks, 1):
        task_id = task.get("id", f"task_{i}")
        category = task.get("category", "unknown")
        is_var = " (variant)" if task.get("is_variant") else ""
        print(f"  [{i}/{len(all_tasks)}] {task_id}{is_var} ({category}) ...", end=" ", flush=True)

        messages = build_messages(task)
        tools = task.get("tools")
        best_response = ""
        best_score = -1.0

        for s in range(args.samples):
            try:
                response = call_teacher(args.api, teacher_model, messages, tools, args.temperature)
                sc = score_task_simple(response, task)
                if sc > best_score:
                    best_score = sc
                    best_response = response
            except Exception as e:
                print(f"\n    sample {s + 1} error: {e}", file=sys.stderr)
                errors += 1

            # Small delay between samples to avoid overwhelming the API
            if s < args.samples - 1:
                time.sleep(0.5)

        if best_score < args.min_score:
            print(f"SKIP (score={best_score:.1%} < {args.min_score:.0%})")
            skipped += 1
            continue

        example = format_training_example(task, best_response, best_score, teacher_model)
        examples.append(example)
        print(f"score={best_score:.1%} ({len(best_response)} chars)")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'=' * 60}")
    print("  Teacher Distillation Complete")
    print(f"{'=' * 60}")
    print(f"  Teacher model:    {teacher_model}")
    print(f"  Temperature:      {args.temperature}")
    print(f"  Samples per task: {args.samples}")
    print(f"  Total tasks:      {len(all_tasks)}")
    print(f"  Examples saved:   {len(examples)}")
    print(f"  Skipped (low):    {skipped}")
    print(f"  API errors:       {errors}")

    if examples:
        scores = [e["eval_score"] for e in examples]
        print(f"  Mean score:       {sum(scores) / len(scores):.1%}")
        print(f"  Min score:        {min(scores):.1%}")
        print(f"  Max score:        {max(scores):.1%}")

        # Per-category breakdown
        cats: dict[str, list[float]] = {}
        for e in examples:
            cats.setdefault(e["category"], []).append(e["eval_score"])
        print("\n  Per-category:")
        for cat in sorted(cats):
            sc = cats[cat]
            print(f"    {cat:<20} {len(sc):>3} examples, mean={sum(sc) / len(sc):.1%}")

    print(f"\n  Output: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
