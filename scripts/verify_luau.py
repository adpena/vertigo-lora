#!/usr/bin/env python3
from __future__ import annotations

"""
Verify that training examples contain valid Luau code.

Uses the Luau CLI to syntax-check extracted code blocks
from assistant messages in JSONL training data. Roblox globals are stubbed
so the parser doesn't fail on game/workspace/script references.

Output:
  data/raw/verified_results.json  — summary report
  Updates original JSONL files in-place with "verified" field

Usage:
  python3 scripts/verify_luau.py
  python3 scripts/verify_luau.py --strict       # remove failing examples
  python3 scripts/verify_luau.py --install      # brew install luau first
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_OUTPUT = DATA_RAW / "verified_results.json"

# Roblox globals that need stubs so luau --compile doesn't error on undeclared identifiers
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
local Os = {} :: any
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

# Patterns that indicate Roblox API usage (triggers stub prepending)
ROBLOX_PATTERNS = re.compile(
    r"\b(?:game|workspace|script|plugin|shared|Enum|Instance|Vector3|Vector2|"
    r"CFrame|Color3|BrickColor|UDim2|UDim|Rect|Region3|Ray|TweenInfo|"
    r"NumberRange|NumberSequence|ColorSequence|PhysicalProperties|Random|"
    r"DateTime|task|tick|time|wait|delay|spawn|warn|typeof)\b"
)


def check_luau_installed() -> bool:
    """Return True if luau-compile is available on PATH (preferred for syntax checking)."""
    return shutil.which("luau-compile") is not None


def install_luau() -> bool:
    """Attempt to install luau via Homebrew."""
    if shutil.which("brew") is None:
        print("Error: Homebrew not found. Install luau manually:")
        print("  brew install luau")
        print("  # or build from source: https://github.com/luau-lang/luau")
        return False

    print("Installing luau via Homebrew...")
    result = subprocess.run(["brew", "install", "luau"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"brew install luau failed:\n{result.stderr}")
        return False

    print("luau installed successfully.")
    return True


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning traces."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_code_blocks(text: str) -> list[str]:
    """Extract Luau code from assistant message text.

    Handles:
    - Fenced code blocks (```lua, ```luau, or bare ```)
    - Contiguous Luau code without fences (heuristic: lines starting with
      common Luau keywords or containing Luau-specific syntax)
    """
    blocks: list[str] = []

    # First, try fenced code blocks
    fenced = re.findall(
        r"```(?:lua(?:u)?|)\s*\n(.*?)```",
        text,
        flags=re.DOTALL,
    )
    if fenced:
        for block in fenced:
            code = block.strip()
            if code:
                blocks.append(code)
        return blocks

    # Heuristic: if the text looks like raw Luau code (starts with --!strict,
    # local, function, return, or a comment), treat the whole thing as code
    stripped = text.strip()
    luau_start = re.compile(r"^(?:--[!\[]|local\s|function\s|return\s|type\s|export\s|if\s|for\s|while\s|repeat\b)")
    if luau_start.match(stripped):
        blocks.append(stripped)
        return blocks

    return blocks


def needs_roblox_stubs(code: str) -> bool:
    """Check if code references Roblox globals that need stubbing."""
    return bool(ROBLOX_PATTERNS.search(code))


def syntax_check(code: str) -> tuple[bool, str]:
    """Run luau-compile on code to check for syntax errors. Returns (ok, error_message).

    Uses luau-compile (compile-only, no execution) so Roblox globals don't
    need runtime stubs — the compiler only checks syntax and bytecode generation.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".luau", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["luau-compile", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # luau-compile exits 0 on success, non-zero on syntax error
        if result.returncode == 0:
            return True, ""

        # Check for syntax errors in stderr
        err = result.stderr.strip()
        if not err:
            err = result.stdout.strip()
        err = err.replace(tmp_path, "<input>")
        return False, err
    except subprocess.TimeoutExpired:
        return False, "luau-compile timed out (10s)"
    except Exception as e:
        return False, f"Failed to run luau-compile: {e}"
    finally:
        os.unlink(tmp_path)


def is_intentional_bugfix_example(messages: list[dict]) -> bool:
    """Detect bugfix pair examples where the 'bug' code is intentionally broken.

    Heuristic: user message mentions 'bug', 'fix', 'wrong', 'broken', 'error',
    or assistant message contains both a bad/wrong example and a corrected one.
    """
    for msg in messages:
        if msg.get("role") == "user":
            text = msg.get("content", "").lower()
            if any(kw in text for kw in ("bug", "fix", "wrong", "broken", "error", "issue", "problem")):
                return True
    return False


def verify_example(example: dict) -> dict:
    """Verify a single training example. Returns augmented example with verification info."""
    messages = example.get("messages", [])
    is_bugfix = is_intentional_bugfix_example(messages)

    all_codes: list[tuple[int, str]] = []  # (msg_index, code)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        text = strip_think_blocks(msg.get("content", ""))
        blocks = extract_code_blocks(text)
        for block in blocks:
            all_codes.append((i, block))

    # No extractable code — pure text explanation, passes by default
    if not all_codes:
        example["verified"] = True
        return example

    errors: list[dict] = []
    for msg_idx, code in all_codes:
        ok, err = syntax_check(code)
        if not ok:
            errors.append(
                {
                    "message_index": msg_idx,
                    "error": err,
                    "code_snippet": code[:200] + ("..." if len(code) > 200 else ""),
                }
            )

    if errors and is_bugfix:
        # Bugfix examples may have intentionally broken code; don't fail them
        # unless ALL code blocks fail (the fix should parse)
        if len(errors) < len(all_codes):
            # At least one block passed — the fix is valid
            example["verified"] = True
        else:
            # All blocks failed — flag but note it's a bugfix example
            example["verified"] = False
            example["verify_errors"] = errors
            example["verify_note"] = "bugfix_example_all_blocks_failed"
    elif errors:
        example["verified"] = False
        example["verify_errors"] = errors
    else:
        example["verified"] = True

    return example


def process_jsonl_file(filepath: Path, strict: bool) -> dict:
    """Process a single JSONL file. Returns stats dict."""
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    stats = {
        "file": str(filepath.name),
        "total": len(examples),
        "passed": 0,
        "failed": 0,
        "skipped_no_code": 0,
        "errors": [],
    }

    verified_examples = []
    for i, example in enumerate(examples):
        result = verify_example(example)

        if result.get("verified"):
            stats["passed"] += 1
            verified_examples.append(result)
        elif "verify_errors" not in result and result.get("verified") is True:
            # No code found, auto-pass
            stats["skipped_no_code"] += 1
            verified_examples.append(result)
        else:
            stats["failed"] += 1
            error_info = {
                "example_index": i,
                "source": result.get("source", "unknown"),
                "category": result.get("category", "unknown"),
                "errors": result.get("verify_errors", []),
                "note": result.get("verify_note", ""),
            }
            stats["errors"].append(error_info)
            if not strict:
                verified_examples.append(result)

    # Count examples with no code as "skipped" (they still passed)
    for ex in verified_examples:
        msgs = ex.get("messages", [])
        has_code = False
        for msg in msgs:
            if msg.get("role") == "assistant":
                text = strip_think_blocks(msg.get("content", ""))
                if extract_code_blocks(text):
                    has_code = True
                    break
        if not has_code and ex.get("verified"):
            stats["skipped_no_code"] += 1
            stats["passed"] -= 1

    # Write back in-place
    with open(filepath, "w", encoding="utf-8") as f:
        for ex in verified_examples:
            f.write(json.dumps(ex) + "\n")

    if strict and stats["failed"] > 0:
        removed = stats["total"] - len(verified_examples)
        print(f"  [strict] Removed {removed} failing example(s) from {filepath.name}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Verify Luau syntax in training examples")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Remove examples with syntax errors from the dataset",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install luau via Homebrew before running",
    )
    args = parser.parse_args()

    # Handle --install
    if args.install:
        if check_luau_installed():
            print("luau is already installed.")
        else:
            if not install_luau():
                sys.exit(1)

    # Check luau-compile availability
    if not check_luau_installed():
        print("Error: luau-compile not found on PATH.")
        print()
        print("Install via Homebrew:")
        print("  brew install luau")
        print()
        print("Or build from source:")
        print("  git clone https://github.com/luau-lang/luau.git")
        print("  cd luau && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release")
        print("  cmake --build build --target Luau.CLI")
        print()
        print("Or pass --install to have this script run brew install for you.")
        sys.exit(1)

    # Find JSONL files
    if not DATA_RAW.exists():
        print(f"Error: data directory not found at {DATA_RAW}")
        sys.exit(1)

    jsonl_files = sorted(DATA_RAW.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {DATA_RAW}")
        sys.exit(0)

    print(f"Luau syntax verification {'(strict mode)' if args.strict else ''}")
    print(f"Scanning {len(jsonl_files)} file(s) in {DATA_RAW}\n")

    all_stats: list[dict] = []
    totals = {"total": 0, "passed": 0, "failed": 0, "skipped_no_code": 0}

    for filepath in jsonl_files:
        print(f"Checking {filepath.name}...")
        stats = process_jsonl_file(filepath, strict=args.strict)
        all_stats.append(stats)

        totals["total"] += stats["total"]
        totals["passed"] += stats["passed"]
        totals["failed"] += stats["failed"]
        totals["skipped_no_code"] += stats["skipped_no_code"]

        print(f"  {stats['passed']} passed, {stats['failed']} failed, {stats['skipped_no_code']} skipped (no code)")

        if stats["errors"]:
            for err in stats["errors"]:
                idx = err["example_index"]
                cat = err.get("category", "?")
                note = f" ({err['note']})" if err.get("note") else ""
                print(f"    FAIL example #{idx} [{cat}]{note}:")
                for detail in err.get("errors", []):
                    print(f"      {detail['error']}")

    # Write summary report
    report = {
        "totals": totals,
        "strict_mode": args.strict,
        "files": all_stats,
    }

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Total examples:    {totals['total']}")
    print(f"Passed:            {totals['passed']}")
    print(f"Failed:            {totals['failed']}")
    print(f"Skipped (no code): {totals['skipped_no_code']}")
    print(f"\nReport written to {RESULTS_OUTPUT}")

    if totals["failed"] > 0:
        if args.strict:
            print(f"\n[strict] Removed {totals['failed']} failing example(s) from dataset.")
        else:
            print("\nRe-run with --strict to remove failing examples.")
        sys.exit(1 if not args.strict else 0)


if __name__ == "__main__":
    main()
