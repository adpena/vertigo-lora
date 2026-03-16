#!/usr/bin/env python3
"""
Rejection Sampling Fine-Tuning (RFT) Data Generator

Generate N completions per prompt, score each with programmatic verification
(pattern matching, convention checks, Luau compilation), keep only the best
response per task if it meets a quality threshold.

Two prompt sources:
  1. Benchmark tasks from data/eval/benchmark.jsonl
  2. Programmatically generated variant tasks

Usage:
    uv run scripts/generate_rejection_sampling.py --samples 4 --threshold 0.5
    uv run scripts/generate_rejection_sampling.py --api-model qwen3.5-27b --samples 8
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scoring import (
    CODE_EXPECTED_CATEGORIES,
    VERTIGO_SYSTEM_PROMPT,
    score_code_presence,
    score_convention,
    score_correctness,
    score_luau_compiles,
    score_tool_selection,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_BENCHMARK = PROJECT_DIR / "data" / "eval" / "benchmark.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "raw" / "rejection_sampled.jsonl"
DEFAULT_API = "http://127.0.0.1:1234/v1"

# ---------------------------------------------------------------------------
# Variant task generation
# ---------------------------------------------------------------------------

VARIANT_TASKS: list[dict] = [
    # code_01 variants
    {
        "id": "code_01_v1",
        "category": "coding",
        "difficulty": 2,
        "prompt": "Write a Vertigo service module called AchievementService that follows the Init/Start lifecycle pattern. It should initialize an empty achievement tracking table in Init and connect to Players.PlayerAdded in Start to load player achievements.",
        "expected_patterns": ["--!strict", r":Init\(\)", r":Start\(\)", r"return\s+\w+Service"],
    },
    {
        "id": "code_01_v2",
        "category": "coding",
        "difficulty": 2,
        "prompt": "Write a Vertigo service module called LeaderboardService that follows the Init/Start lifecycle pattern. It should initialize an empty leaderboard cache in Init and connect to Players.PlayerAdded in Start to create leaderboard entries.",
        "expected_patterns": ["--!strict", r":Init\(\)", r":Start\(\)", r"return\s+\w+Service"],
    },
    # code_02 variants
    {
        "id": "code_02_v1",
        "category": "coding",
        "difficulty": 3,
        "prompt": "Write a zone builder module that creates mushroom geometry for a swamp zone. It should use CollectionService to tag mushrooms as 'GlowMushroom', create BaseParts with ForceField material and random scales, and be optimized with @native on the hot geometry generation function.",
        "expected_patterns": ["CollectionService", r"Instance\.new", "@native", "GlowMushroom"],
    },
    {
        "id": "code_02_v2",
        "category": "coding",
        "difficulty": 3,
        "prompt": "Write a zone builder module that creates pillar geometry for a ruins zone. It should use CollectionService to tag pillars as 'AncientPillar', create BaseParts with slate material and varying heights, and be optimized with @native on the hot geometry generation function.",
        "expected_patterns": ["CollectionService", r"Instance\.new", "@native", "AncientPillar"],
    },
    # code_04 variants
    {
        "id": "code_04_v1",
        "category": "coding",
        "difficulty": 2,
        "prompt": "Write a config module for glide ability tuning values following Vertigo conventions. It should export multiple named tables including GlideTuning with fields like maxSpeed, liftFactor, dragCoefficient, and turnRate. Use table.freeze on all config tables.",
        "expected_patterns": [r"return\s*\{", "GlideTuning", "export type", r"table\.freeze"],
    },
    {
        "id": "code_04_v2",
        "category": "coding",
        "difficulty": 2,
        "prompt": "Write a config module for air dash ability tuning values following Vertigo conventions. It should export multiple named tables including AirDashTuning with fields like dashSpeed, dashDuration, cooldown, and maxCharges. Use table.freeze on all config tables.",
        "expected_patterns": [r"return\s*\{", "AirDashTuning", "export type", r"table\.freeze"],
    },
    # code_05 variants
    {
        "id": "code_05_v1",
        "category": "coding",
        "difficulty": 3,
        "prompt": "Write a DataStore-backed player settings service that loads and saves settings (keybinds, volume, graphics quality). Use pcall for error handling, implement retry logic for failed saves, and follow the Init/Start lifecycle.",
        "expected_patterns": ["DataStoreService", "pcall", r"UpdateAsync|GetAsync|SetAsync", r":Init\(\)"],
    },
    {
        "id": "code_05_v2",
        "category": "coding",
        "difficulty": 3,
        "prompt": "Write a DataStore-backed analytics service that records and persists player session stats (playtime, zones visited, abilities used). Use pcall for error handling, implement retry logic for failed writes, and follow the Init/Start lifecycle.",
        "expected_patterns": ["DataStoreService", "pcall", r"UpdateAsync|GetAsync|SetAsync", r":Init\(\)"],
    },
    # code_06 variants
    {
        "id": "code_06_v1",
        "category": "coding",
        "difficulty": 3,
        "prompt": "Write a RemoteEvent handler for a server-authoritative vehicle mount system. When a client fires RequestSpawnVehicle, the server should validate the request (check ownership, proximity to spawn point, cooldown), reject invalid requests, and only then spawn the vehicle and FireClient the result.",
        "expected_patterns": [
            "OnServerEvent",
            r"valid|check|ownership|proximity",
            "FireClient",
            r"reject|deny|invalid",
        ],
    },
    # code_09 variants
    {
        "id": "code_09_v1",
        "category": "coding",
        "difficulty": 3,
        "prompt": "Write a collectible pickup system that uses CollectionService tags to find collectible items. Items tagged with 'Collectible' should detect player proximity via Touched, award points, play a collection effect, and respawn after a delay.",
        "expected_patterns": ["CollectionService", "GetTagged", "Touched", r"respawn|delay|wait"],
    },
    # fix_01 variants
    {
        "id": "fix_01_v1",
        "category": "bugfix",
        "difficulty": 2,
        "prompt": 'Fix this buggy Vertigo service module. It\'s missing type annotations that --!strict mode requires:\n\n```luau\n--!strict\nlocal QuestService = {}\n\nfunction QuestService:Init()\n    self._quests = {}\n    self._active = {}\nend\n\nfunction QuestService:AddQuest(playerId, questId, data)\n    self._quests[playerId] = self._quests[playerId] or {}\n    table.insert(self._quests[playerId], {id = questId, progress = 0, data = data})\nend\n\nfunction QuestService:Start()\n    print("QuestService started")\nend\n\nreturn QuestService\n```\n\nWhat is wrong and how do you fix it?',
        "expected_patterns": [r"type\s+annotation|strict|typed", r":\s*number|:\s*string|:\s*any"],
    },
    {
        "id": "fix_01_v2",
        "category": "bugfix",
        "difficulty": 2,
        "prompt": 'Fix this buggy Vertigo service that will crash because pcall is missing around the DataStore call:\n\n```luau\n--!strict\nlocal DataStoreService = game:GetService("DataStoreService")\nlocal store = DataStoreService:GetDataStore("Rewards")\n\nlocal RewardService = {}\n\nfunction RewardService:Init()\n    self._pending = {}\nend\n\nfunction RewardService:ClaimReward(playerId: number, rewardId: string)\n    local data = store:GetAsync("reward_" .. tostring(playerId))\n    if data and data[rewardId] then\n        store:RemoveAsync("reward_" .. tostring(playerId))\n        return true\n    end\n    return false\nend\n\nfunction RewardService:Start() end\n\nreturn RewardService\n```\n\nWhat is wrong and how do you fix it?',
        "expected_patterns": ["pcall", r"error|fail|crash|wrap"],
    },
    # fix_03 variants
    {
        "id": "fix_03_v1",
        "category": "bugfix",
        "difficulty": 3,
        "prompt": 'Fix this code that uses Vector3.new math in a hot path instead of vector.* SIMD operations, and is missing @native:\n\n```luau\n--!strict\nlocal RunService = game:GetService("RunService")\n\nlocal function updateProjectiles(dt: number)\n    for _, proj in projectiles do\n        local dir = (proj.target - proj.position)\n        local dist = dir.Magnitude\n        if dist > 0.1 then\n            proj.position = proj.position + dir.Unit * proj.speed * dt\n        end\n    end\nend\n\nRunService.Heartbeat:Connect(updateProjectiles)\n```\n\nConvert to use vector.create, vector.normalize, vector.magnitude and add @native.',
        "expected_patterns": [r"vector\.create", r"vector\.normalize|vector\.magnitude", "@native"],
    },
    # fix_05 variants
    {
        "id": "fix_05_v1",
        "category": "bugfix",
        "difficulty": 3,
        "prompt": "Fix this code to use math.lerp instead of manual interpolation, and vector.* SIMD instead of Vector3 math:\n\n```luau\n--!strict\nlocal function smoothDamp(current: Vector3, target: Vector3, velocity: Vector3, smoothTime: number, dt: number): (Vector3, Vector3)\n    local omega = 2.0 / smoothTime\n    local x = omega * dt\n    local exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)\n    local change = current - target\n    local temp = (velocity + change * omega) * dt\n    velocity = (velocity - temp * omega) * exp\n    local output = target + (change + temp) * exp\n    return output, velocity\nend\n```\n\nConvert to use vector.create and @native for NCG optimization.",
        "expected_patterns": [r"vector\.create", "@native", r"math\.lerp|vector"],
    },
    # architecture variants
    {
        "id": "arch_01_v1",
        "category": "architecture",
        "difficulty": 4,
        "prompt": "Design the Init/Start boot ordering for these 4 interdependent Vertigo services:\n- ConfigService: loads config tables from ModuleScripts\n- AudioService: needs config for volume/SFX settings from ConfigService\n- UIService: needs audio references from AudioService for button sounds\n- InputService: needs UI references from UIService for input mapping\n\nProvide the boot order, explain why, and show the init.client.luau boot sequence code.",
        "expected_patterns": [
            r"Init.*Start|boot.*order",
            r"ConfigService.*AudioService|sequential",
            "require",
            r"InputService.*last|final",
        ],
    },
    {
        "id": "arch_02_v1",
        "category": "architecture",
        "difficulty": 4,
        "prompt": "Design server-authoritative validation for the Vertigo vehicle spawn flow. A client wants to spawn a DirtBike. Show the full request/validation/response flow including:\n- Client sends RequestSpawnVehicle with vehicle ID and desired spawn position\n- Server validates: ownership, spawn zone proximity, existing vehicle limit, player state\n- Server spawns or rejects with reason\n- Client receives result and updates UI\n\nProvide both client and server code.",
        "expected_patterns": [
            r"RequestSpawnVehicle|OnServerEvent",
            r"valid|check|ownership",
            "FireClient|StateSync",
            r"spawn|vehicle|limit",
        ],
    },
    # embodiment variants
    {
        "id": "play_01_v1",
        "category": "embodiment",
        "difficulty": 3,
        "prompt": "You are an agent embodied in the Vertigo world at the Crystal Caverns zone (Y=-30). Navigate to the Hub zone (Y=8) using wall runs and grapple chains. Describe your traversal plan: which surfaces to wall-run, which GrappleAnchor points to target, and how to chain abilities for efficient vertical ascent.",
        "expected_patterns": [
            r"grapple|wall.?run",
            r"anchor|GrappleAnchor",
            r"Y.*8|Hub|hub|upward",
            r"chain|sequence|vertical",
        ],
    },
    {
        "id": "play_02_v1",
        "category": "embodiment",
        "difficulty": 4,
        "prompt": "You are an agent in the Vertigo world. Execute a speed run sequence: slide down a slope to build velocity, launch into a grapple swing at the bottom, release into an air dash, then glide to cover maximum distance. Describe the exact input sequence, ability cooldown management, and expected speed values.",
        "expected_patterns": [
            r"slide.*grapple.*dash|chain",
            r"cooldown|timing|window",
            r"velocity|speed",
            r"distance|horizontal",
        ],
    },
    # mcp variants
    {
        "id": "mcp_01_v1",
        "category": "mcp_tool_calling",
        "difficulty": 2,
        "prompt": "Search for all parts tagged with 'BloomCrystal' in the current Roblox Studio place and report how many exist, their materials, and their average size.",
        "expected_patterns": ["studio_find_tagged", "BloomCrystal", r"count|#|total"],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "studio_find_tagged",
                    "description": "Find all instances with a given CollectionService tag.",
                    "parameters": {
                        "type": "object",
                        "properties": {"tag": {"type": "string"}, "limit": {"type": "integer", "default": 100}},
                        "required": ["tag"],
                    },
                },
            }
        ],
    },
    {
        "id": "mcp_03_v1",
        "category": "mcp_tool_calling",
        "difficulty": 3,
        "prompt": "Run a builder audit on the current Studio place to check how many descendants each builder has created, what CollectionService tags are present, and flag any builders with suspiciously high part counts.",
        "expected_patterns": ["studio_builder_audit", r"builder|descendant|tag"],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "studio_builder_audit",
                    "description": "Per-builder descendant class census + CollectionService tags.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ],
    },
]


def score_response(response: str, task: dict) -> dict:
    """Score a response. Returns dict with individual scores and composite."""
    category = task.get("category", "coding")
    correctness = score_correctness(response, task.get("expected_patterns", []))
    convention = score_convention(response, category)
    code_pres = score_code_presence(response, category)
    luau_ok = score_luau_compiles(response) if category in CODE_EXPECTED_CATEGORIES else False

    # Build weighted composite
    weights: dict[str, tuple[float, float]] = {"correctness": (correctness, 0.4)}
    if convention is not None:
        weights["convention"] = (convention, 0.2)
    if code_pres is not None:
        weights["code_presence"] = (code_pres, 0.15)
    # Luau compilation bonus
    if category in CODE_EXPECTED_CATEGORIES:
        weights["luau_compiles"] = (1.0 if luau_ok else 0.0, 0.15)

    # Check for tool selection (mcp tasks)
    tool_sel = score_tool_selection(response, task)
    if tool_sel is not None:
        weights["tool_selection"] = (tool_sel, 0.2)

    total_w = sum(w for _, w in weights.values())
    composite = sum(s * w / total_w for s, w in weights.values())

    # Failure penalty
    stripped = response.strip()
    if len(stripped) < 10 or stripped.startswith("[TIMEOUT]") or stripped.startswith("[ERROR"):
        composite = 0.0
    elif len(stripped) < 50:
        composite *= 0.3

    return {
        "correctness": round(correctness, 3),
        "convention": round(convention, 3) if convention is not None else None,
        "code_presence": round(code_pres, 3) if code_pres is not None else None,
        "luau_compiles": luau_ok,
        "composite": round(composite, 3),
    }


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------


def strip_thinking(text: str) -> str:
    """Strip thinking/reasoning preamble from model output.

    Many reasoning models (Qwen3.5, etc.) emit a 'Thinking Process:' block
    that can consume most of the output. We extract the actual answer.

    Strategy: find the LAST fenced code block and surrounding prose as the
    actual answer, since thinking models put their final code at the end.
    """
    # Pattern: <think>...</think> wrapper
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if cleaned != text:
        return cleaned.strip()

    # For thinking models: extract the last code block and any text after it
    # as the "answer". The thinking preamble usually starts with
    # "Thinking Process:" or numbered steps.
    has_thinking = bool(re.search(r"^(Thinking Process|##?\s*Thinking|1\.\s+\*\*Analyze)", text[:200]))

    if has_thinking:
        # Find all fenced code blocks
        blocks = list(re.finditer(r"```(?:luau|lua)?\s*\n(.*?)```", text, re.DOTALL))
        if blocks:
            # Take from the start of the last code block to end
            last_block = blocks[-1]
            # Include some context before the block (the intro line)
            start = last_block.start()
            # Walk back to find the intro line (e.g., "Here is the code:")
            pre_start = max(0, start - 200)
            pre_text = text[pre_start:start]
            # Find last newline in pre_text as intro boundary
            nl = pre_text.rfind("\n")
            if nl >= 0:
                start = pre_start + nl + 1

            result = text[start:].strip()
            # Remove trailing thinking after the code block
            trailing_think = re.search(r"\n\s+(Wait,|Okay,|Let me|One more|Adding|Final)", result)
            if trailing_think and trailing_think.start() > 50:
                result = result[: trailing_think.start()].strip()

            return result if len(result) > 30 else text.strip()

    return text.strip()


def detect_model(api_base: str) -> str:
    from api import detect_model as _detect_model

    try:
        return _detect_model(api_base=api_base)
    except Exception:
        return "default"


def generate_one(
    api_base: str,
    model: str,
    task: dict,
    temperature: float,
    max_tokens: int = 1024,
    max_retries: int = 2,
) -> str:
    """Generate a single completion via OpenAI-compatible API with retry."""
    messages = [
        {"role": "system", "content": VERTIGO_SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]

    tools = task.get("tools")
    if tools:
        tool_desc = json.dumps(tools, indent=2)
        messages[0] = dict(messages[0])
        messages[0]["content"] += (
            "\n\nYou have access to the following tools:\n```json\n"
            + tool_desc
            + "\n```\nCall tools by responding with a JSON tool_call block."
        )

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    data = json.dumps(body).encode()

    import time as _time

    for attempt in range(max_retries + 1):
        try:
            # Use subprocess+curl for reliable timeout (urllib hangs on model crashes)
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "--max-time",
                    "120",
                    "-X",
                    "POST",
                    f"{api_base}/chat/completions",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    data.decode(),
                ],
                capture_output=True,
                text=True,
                timeout=130,
            )
            if result.returncode != 0 or not result.stdout.strip():
                raise RuntimeError(f"curl failed (rc={result.returncode})")
            raw_text = result.stdout.strip()
            if raw_text.startswith("<!DOCTYPE") or raw_text.startswith("<html"):
                raise RuntimeError("server returned HTML error")
            parsed = json.loads(raw_text)
            if "error" in parsed:
                raise RuntimeError(str(parsed["error"])[:100])
            choice = parsed["choices"][0]
            content = choice.get("message", {}).get("content", "")
            tc = choice.get("message", {}).get("tool_calls")
            if tc:
                content += "\n" + json.dumps(tc)
            return content
        except Exception as e:
            if attempt < max_retries:
                wait = 8 * (attempt + 1)
                print(f"    (retry {attempt + 1}/{max_retries}, wait {wait}s: {str(e)[:60]})", flush=True)
                _time.sleep(wait)
                continue
            return f"[ERROR: {e}]"
    return "[ERROR: max retries exceeded]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Rejection Sampling FT Data Generator")
    parser.add_argument("--api", default=DEFAULT_API, help="API base URL")
    parser.add_argument("--api-model", default=None, help="Model name (auto-detect)")
    parser.add_argument("--samples", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--threshold", type=float, default=0.5, help="Min score to keep")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--no-variants", action="store_true", help="Skip generated variant tasks")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per generation")
    args = parser.parse_args()

    # Detect model
    model_name = args.api_model or detect_model(args.api)
    print(f"API: {args.api}  model: {model_name}")
    print(f"Samples per task: {args.samples}  threshold: {args.threshold}  temp: {args.temperature}")

    # Load benchmark tasks
    tasks: list[dict] = []
    if DEFAULT_BENCHMARK.exists():
        with open(DEFAULT_BENCHMARK) as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        print(f"Loaded {len(tasks)} benchmark tasks")
    else:
        print(f"WARNING: benchmark file not found: {DEFAULT_BENCHMARK}", file=sys.stderr)

    # Add variant tasks
    if not args.no_variants:
        tasks.extend(VARIANT_TASKS)
        print(f"Added {len(VARIANT_TASKS)} variant tasks ({len(tasks)} total)")

    # Rejection sampling loop
    kept: list[dict] = []
    total_generated = 0
    scores_all: list[float] = []

    print(f"\n{'=' * 70}")
    print(f"  Rejection Sampling: {len(tasks)} tasks x {args.samples} samples")
    print(f"{'=' * 70}\n")

    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", f"task_{i}")
        category = task.get("category", "coding")

        # Generate N samples
        samples: list[tuple[str, str, dict]] = []  # (raw, cleaned, scores)
        for s in range(args.samples):
            raw = generate_one(args.api, model_name, task, args.temperature, max_tokens=args.max_tokens)
            cleaned = strip_thinking(raw)
            scored = score_response(cleaned, task)
            samples.append((raw, cleaned, scored))
            total_generated += 1

        # Rank by composite score, pick best
        samples.sort(key=lambda x: x[2]["composite"], reverse=True)
        best_raw, best_cleaned, best_scores = samples[0]
        best_score = best_scores["composite"]
        scores_all.append(best_score)

        is_kept = best_score >= args.threshold
        status = "KEPT" if is_kept else "skip"
        luau_tag = " luau-ok" if best_scores.get("luau_compiles") else ""

        print(
            f"  [{i:>3}/{len(tasks)}] {task_id:<20} {args.samples} samples, best={best_score:.2f}{luau_tag}, {status}"
        )

        if is_kept:
            # Determine task family
            if category == "bugfix":
                task_family = "critic"
            elif category == "embodiment":
                task_family = "embodiment"
            elif category == "mcp_tool_calling":
                task_family = "sft_tool_calling"
            else:
                task_family = "sft_scripter"

            example = {
                "messages": [
                    {"role": "system", "content": VERTIGO_SYSTEM_PROMPT},
                    {"role": "user", "content": task["prompt"]},
                    {"role": "assistant", "content": best_cleaned},
                ],
                "source": "synthetic_evolved",
                "rights_basis": "generated",
                "teacher_model": model_name,
                "task_family": task_family,
                "has_reasoning": True,
                "eval_score": best_score,
                "verified": best_scores.get("luau_compiles", False),
                "task_id": task_id,
                "category": category,
                "difficulty": task.get("difficulty", 0),
                "scores_detail": best_scores,
                "samples_generated": args.samples,
                "timestamp": datetime.now().isoformat(),
            }
            kept.append(example)

    # Summary
    mean_score = sum(scores_all) / len(scores_all) if scores_all else 0
    kept_count = len(kept)

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {len(tasks)} tasks x {args.samples} samples = {total_generated} total generations")
    print(f"  {kept_count} kept (score >= {args.threshold}), {len(tasks) - kept_count} rejected")
    print(f"  Mean best score: {mean_score:.3f}")
    verified_count = sum(1 for e in kept if e.get("verified"))
    print(f"  Luau-verified: {verified_count}/{kept_count}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n  Saved {kept_count} examples to: {args.output}")


if __name__ == "__main__":
    main()
