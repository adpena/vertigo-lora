#!/usr/bin/env python3
from __future__ import annotations

"""
STaR (Self-Taught Reasoner) loop for Luau code generation.

Generate code with reasoning -> validate with Luau compiler -> keep correct
traces -> format as training data. The model's own correct reasoning traces
become training signal.

Usage:
  uv run scripts/generate_star.py
  uv run scripts/generate_star.py --samples 5 --temperature 0.8
  uv run scripts/generate_star.py --api-model qwen3.5-27b
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Roblox global stubs (from verify_luau.py) so luau-compile doesn't choke
# ---------------------------------------------------------------------------
ROBLOX_GLOBAL_STUBS = """\
-- Stubs for Roblox globals (syntax check only)
local game = {} :: any
local workspace = {} :: any
local script = {} :: any
local plugin = {} :: any
local shared = {} :: any
local Enum = {} :: any
local Instance = {} :: any
local Vector3 = {} :: any
local Vector2 = {} :: any
local CFrame = {} :: any
local Color3 = {} :: any
local BrickColor = {} :: any
local UDim2 = {} :: any
local UDim = {} :: any
local Rect = {} :: any
local Region3 = {} :: any
local Ray = {} :: any
local TweenInfo = {} :: any
local NumberRange = {} :: any
local NumberSequence = {} :: any
local NumberSequenceKeypoint = {} :: any
local ColorSequence = {} :: any
local ColorSequenceKeypoint = {} :: any
local PhysicalProperties = {} :: any
local Random = {} :: any
local DateTime = {} :: any
local task = {} :: any
local debug = {} :: any
local buffer = {} :: any
local bit32 = {} :: any
local utf8 = {} :: any
local tick = (nil :: any) :: () -> number
local time = (nil :: any) :: () -> number
local wait = (nil :: any) :: (n: number?) -> number
local delay = (nil :: any) :: (t: number, f: () -> ()) -> ()
local spawn = (nil :: any) :: (f: () -> ()) -> ()
local warn = (nil :: any) :: (...any) -> ()
local typeof = (nil :: any) :: (any) -> string
"""

ROBLOX_PATTERNS = re.compile(
    r"\b(?:game|workspace|script|plugin|shared|Enum|Instance|Vector3|Vector2|"
    r"CFrame|Color3|BrickColor|UDim2|UDim|Rect|Region3|Ray|TweenInfo|"
    r"NumberRange|NumberSequence|ColorSequence|PhysicalProperties|Random|"
    r"DateTime|task|tick|time|wait|delay|spawn|warn|typeof)\b"
)

# ---------------------------------------------------------------------------
# Task templates (30 coding tasks with verifiable outputs)
# ---------------------------------------------------------------------------
TASKS = [
    # Data structures
    {
        "prompt": "Write a --!strict Luau module that implements a Stack data structure with push, pop, peek, and isEmpty methods. Include type annotations.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module for a simple EventBus with subscribe, unsubscribe, and fire methods.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau function that performs binary search on a sorted array of numbers.",
        "verify": "compile",
    },
    {"prompt": "Write a --!strict Luau module that implements an object pool for reusing Parts.", "verify": "compile"},
    {
        "prompt": "Write a --!strict Luau module for a simple state machine with addState, transition, and getCurrentState.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module that implements a Queue with enqueue, dequeue, peek, and size methods. Include type annotations.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module that implements a doubly linked list with insert, remove, find, and iterate methods.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module that implements a min-heap priority queue with insert, extractMin, and peek.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module that implements a Trie data structure for string prefix lookups with insert, search, and startsWith.",
        "verify": "compile",
    },
    # Roblox patterns
    {
        "prompt": "Write a --!strict Luau service module for managing player data with Init and Start lifecycle methods, following the Roblox service pattern.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau controller module for handling client-side input with keybind registration and action mapping.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau config module that exports tuning tables for a grapple hook ability (speed, range, cooldown, etc.).",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau builder pattern module for constructing UI elements with method chaining.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module that wraps RemoteEvents with type-safe request/response patterns.",
        "verify": "compile",
    },
    # Algorithms
    {
        "prompt": "Write a --!strict Luau function that implements merge sort on an array of numbers. Include type annotations.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module that implements A* pathfinding on a 2D grid. Include type annotations for the grid and path.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module that implements a spatial hash grid for efficient nearby-object queries.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module that implements a quadtree for 2D spatial partitioning with insert and queryRange.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau function that implements quicksort with a custom comparator function parameter.",
        "verify": "compile",
    },
    # Utilities
    {
        "prompt": "Write a --!strict Luau function that implements debounce: given a function and delay, returns a debounced version.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau function that implements throttle: limits how often a function can be called.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau function that implements retry logic with exponential backoff and max attempts.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau function that implements memoize: caches results of a pure function based on arguments.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau function that performs deep clone of a nested table, handling metatables and cyclic references.",
        "verify": "compile",
    },
    # Game systems
    {
        "prompt": "Write a --!strict Luau module for an inventory system with addItem, removeItem, hasItem, and getCount methods. Use type annotations.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module for a cooldown tracker that manages multiple named cooldowns with start, isReady, and remaining methods.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module for a damage calculator that applies armor, resistance, critical hits, and damage types.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module for a weighted loot table with addItem, roll, and rollMultiple methods. Include type annotations.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module for an achievement system that tracks progress, unlocks achievements, and fires events on completion.",
        "verify": "compile",
    },
    {
        "prompt": "Write a --!strict Luau module for a quest system with quest definitions, progress tracking, prerequisites, and reward granting.",
        "verify": "compile",
    },
]

from prompts import STAR_SYSTEM_PROMPT as SYSTEM_PROMPT  # noqa: E402


# ---------------------------------------------------------------------------
# Luau verification
# ---------------------------------------------------------------------------


def verify_luau_code(code: str) -> bool:
    """Run luau-compile on the code via temp file, return True if it compiles."""
    import tempfile

    # Prepend Roblox stubs if the code references Roblox globals
    full_code = code
    if ROBLOX_PATTERNS.search(code):
        full_code = ROBLOX_GLOBAL_STUBS + "\n" + code

    # Write to temp file (luau-compile doesn't support --stdin)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".luau", delete=False, encoding="utf-8") as f:
            f.write(full_code)
            tmp_path = f.name

        for cmd in ["luau-compile", "luau"]:
            try:
                result = subprocess.run([cmd, tmp_path], capture_output=True, text=True, timeout=10)
                return result.returncode == 0
            except FileNotFoundError:
                continue
    except Exception:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Fallback: syntax heuristics if no Luau binary
    return "--!strict" in code or "local " in code


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_code_blocks(text: str) -> list[str]:
    """Extract Luau/Lua code blocks from markdown-formatted text."""
    blocks: list[str] = []
    fenced = re.findall(r"```(?:lua(?:u)?|)\s*\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        for block in fenced:
            code = block.strip()
            if code:
                blocks.append(code)
        return blocks

    # Heuristic: raw code without fences
    stripped = text.strip()
    luau_start = re.compile(r"^(?:--[!\[]|local\s|function\s|return\s|type\s|export\s|if\s|for\s|while\s|repeat\b)")
    if luau_start.match(stripped):
        blocks.append(stripped)
    return blocks


def strip_think_blocks(text: str) -> str:
    """Remove reasoning traces from response, handling multiple formats.

    Handles:
    - <think>...</think> tags
    - Qwen 3.5 "Thinking Process:" prefix sections (text before first code block)
    """
    if not text:
        return ""
    # Remove <think> tags
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Remove Qwen-style "Thinking Process:" prefix (everything before first code fence)
    if cleaned.startswith("Thinking Process:") or cleaned.startswith("Thinking process:"):
        # Find first code block and keep from there
        fence_match = re.search(r"```(?:lua(?:u)?|)\s*\n", cleaned)
        if fence_match:
            cleaned = cleaned[fence_match.start() :]
    return cleaned.strip()


def extract_thinking(text: str) -> str:
    """Extract reasoning content from response."""
    if not text:
        return ""
    # Try <think> tags first
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if matches:
        return "\n".join(m.strip() for m in matches)
    # Try Qwen-style "Thinking Process:" prefix
    if text.startswith("Thinking Process:") or text.startswith("Thinking process:"):
        fence_match = re.search(r"```(?:lua(?:u)?|)\s*\n", text)
        if fence_match:
            return text[: fence_match.start()].strip()
    return ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_response(code_blocks: list[str], reasoning: str) -> float:
    """Score a response by quality signals. Higher is better (0-1 range)."""
    if not code_blocks:
        return 0.0

    all_code = "\n".join(code_blocks)
    score = 0.0

    # Code length (reward substance, cap at 500 chars)
    code_len = len(all_code)
    score += min(code_len / 500, 1.0) * 0.2

    # Type annotations
    type_count = len(re.findall(r":\s*\w+", all_code))
    score += min(type_count / 10, 1.0) * 0.25

    # --!strict present
    if "--!strict" in all_code:
        score += 0.2

    # export type usage
    if "export type" in all_code:
        score += 0.1

    # Reasoning quality (reward longer, structured thinking)
    if reasoning:
        reason_len = len(reasoning)
        score += min(reason_len / 300, 1.0) * 0.15

    # Function/method count (reward completeness)
    func_count = len(re.findall(r"\bfunction\b", all_code))
    score += min(func_count / 5, 1.0) * 0.1

    return round(score, 3)


# ---------------------------------------------------------------------------
# LLM API
# ---------------------------------------------------------------------------


def chat_completion(api_url: str, model: str, prompt: str, temperature: float) -> str:
    """Call OpenAI-compatible chat completion API."""
    from api import call_llm, extract_content

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(3):
        try:
            resp = call_llm(
                messages,
                model=model,
                api_base=api_url,
                temperature=temperature,
                max_tokens=4096,
                timeout=300,
            )
            return extract_content(resp)
        except Exception as e:
            if attempt < 2:
                wait_secs = 5 * (attempt + 1)
                print(f"  API error (attempt {attempt + 1}/3): {e} — retrying in {wait_secs}s", file=sys.stderr)
                time.sleep(wait_secs)
            else:
                print(f"  API error (giving up): {e}", file=sys.stderr)
    return ""


def detect_model(api_url: str) -> str:
    """Auto-detect the first available model."""
    from api import detect_model as _detect_model

    try:
        return _detect_model(api_base=api_url)
    except RuntimeError:
        print("No models found at API.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# STaR loop
# ---------------------------------------------------------------------------


def run_star(
    api_url: str,
    model: str,
    samples: int,
    temperature: float,
    output_path: Path,
) -> None:
    """Run the full STaR loop over all tasks."""
    total_generated = 0
    total_compiled = 0
    kept: list[dict] = []

    print(f"STaR loop: {len(TASKS)} tasks x {samples} samples = {len(TASKS) * samples} total", flush=True)
    print(f"Model: {model} | Temp: {temperature} | API: {api_url}", flush=True)
    print(flush=True)

    for task_idx, task in enumerate(TASKS):
        prompt = task["prompt"]
        task_slug = prompt.split("that implements")[-1].split("for")[-1].split("with")[0].strip()
        task_slug = re.sub(r"[^a-z0-9]+", "_", task_slug.lower())[:30].strip("_")
        if not task_slug:
            task_slug = f"task_{task_idx}"

        best_score = 0.0
        best_example = None
        compiled_count = 0

        for s in range(samples):
            total_generated += 1
            response = chat_completion(api_url, model, prompt, temperature)
            if not response:
                continue

            # Extract reasoning and code
            reasoning = extract_thinking(response)
            clean_text = strip_think_blocks(response)
            code_blocks = extract_code_blocks(clean_text)

            if not code_blocks:
                continue

            # Verify ALL code blocks compile
            all_ok = all(verify_luau_code(block) for block in code_blocks)
            if not all_ok:
                continue

            compiled_count += 1
            total_compiled += 1

            # Score and keep best
            sc = score_response(code_blocks, reasoning)
            if sc > best_score:
                best_score = sc
                # Build the training example
                assistant_content = response  # keep full response with thinking
                best_example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "source": "synthetic_evolved",
                    "rights_basis": "generated",
                    "teacher_model": model,
                    "task_family": "sft_scripter",
                    "verified": True,
                    "has_reasoning": bool(reasoning),
                    "star_score": sc,
                    "task_slug": task_slug,
                }

        status = f"[{task_idx + 1}/{len(TASKS)}] {task_slug}: {compiled_count}/{samples} compiled"
        if best_example:
            status += f", best score={best_score}"
            kept.append(best_example)
        else:
            status += ", NO valid response"
        print(status, flush=True)

    # Write output (append mode — preserves previous runs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = 0
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())
        print(f"Appending to {output_path} ({existing} existing examples)")
    with open(output_path, "a", encoding="utf-8") as f:
        for example in kept:
            f.write(json.dumps(example) + "\n")

    print()
    print("=" * 55)
    print(f"Generated:  {len(TASKS)} tasks x {samples} samples = {total_generated} total")
    print(f"Compiled:   {total_compiled}")
    print(f"Best kept:  {len(kept)}")
    print(f"Output:     {output_path}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="STaR loop for Luau code generation")
    parser.add_argument("--api", default="http://127.0.0.1:1234/v1", help="API base URL")
    parser.add_argument("--api-model", default=None, help="Model name (auto-detect if omitted)")
    parser.add_argument("--samples", type=int, default=3, help="Responses per task")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    # Resolve model
    model = args.api_model or detect_model(args.api)
    print(f"Using model: {model}")

    # Resolve output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "star_verified.jsonl"

    run_star(
        api_url=args.api,
        model=model,
        samples=args.samples,
        temperature=args.temperature,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
