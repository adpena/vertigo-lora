#!/usr/bin/env python3
from __future__ import annotations

"""Extract OSS Roblox/Luau repos -> data/raw/oss_roblox.jsonl training pairs."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT = PROJECT_DIR / "data" / "raw" / "oss_roblox.jsonl"

CACHE_DIR = Path(os.environ.get("VERTIGO_LORA_CACHE_DIR", "/Volumes/APDataStore/vertigo/cache/oss-repos"))

REPOS: dict[str, dict] = {
    "NevermoreEngine": {
        "url": "https://github.com/Quenty/NevermoreEngine",
        "focus": ["src/"],
        "desc": "Quenty's Nevermore framework — modular Roblox packages",
    },
    "jecs": {
        "url": "https://github.com/Ukendio/jecs",
        "focus": ["src/"],
        "desc": "jecs ECS library — archetype-based entity component system for Luau",
    },
    "matter": {
        "url": "https://github.com/matter-ecs/matter",
        "focus": ["lib/"],
        "desc": "Matter ECS — pure ECS library for Roblox",
    },
    "reflex": {
        "url": "https://github.com/littensy/reflex",
        "focus": ["src/"],
        "desc": "Reflex — reactive state management for Roblox",
    },
    "ByteNet": {
        "url": "https://github.com/ffrostfall/ByteNet",
        "focus": ["src/"],
        "desc": "ByteNet — efficient binary networking library for Roblox",
    },
    "Knit": {
        "url": "https://github.com/Sleitnick/Knit",
        "focus": ["src/"],
        "desc": "Knit — lightweight game framework for Roblox",
    },
    "luau": {
        "url": "https://github.com/luau-lang/luau",
        "focus": ["tests/", "bench/"],
        "desc": "Luau language — official tests and benchmarks",
    },
}

# Category inference from file content and path
CATEGORY_RULES: list[tuple[str, str]] = [
    ("network", "networking"),
    ("remote", "networking"),
    ("net", "networking"),
    ("server", "service"),
    ("service", "service"),
    ("client", "controller"),
    ("controller", "controller"),
    ("config", "config"),
    ("type", "types"),
    ("physics", "physics"),
    ("ecs", "general_luau"),
    ("state", "general_luau"),
    ("signal", "general_luau"),
]


def clone_repo(name: str, info: dict, base_dir: Path) -> Path | None:
    dest = base_dir / name
    if dest.exists() and (dest / ".git").exists():
        print(f"  [cached] {name}")
        return dest

    print(f"  [clone] {info['url']} -> {dest}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--single-branch", info["url"], str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [error] Failed to clone {name}: {result.stderr.strip()}")
        return None
    return dest


def collect_lua_files(repo_dir: Path, focus_dirs: list[str]) -> list[Path]:
    files: list[Path] = []
    for focus in focus_dirs:
        focus_path = repo_dir / focus
        if not focus_path.exists():
            continue
        for ext in ("*.luau", "*.lua"):
            files.extend(focus_path.rglob(ext))
    return sorted(files)


def infer_category(filepath: Path, content: str) -> str:
    lower_path = filepath.as_posix().lower()
    lower_content = content[:2000].lower()
    for keyword, category in CATEGORY_RULES:
        if keyword in lower_path or keyword in lower_content:
            return category
    return "general_luau"


def difficulty_from_length(content: str) -> int:
    lines = content.count("\n")
    if lines < 30:
        return 1
    if lines < 80:
        return 2
    if lines < 200:
        return 3
    if lines < 500:
        return 4
    return 5


def generate_reasoning(repo_name: str, filepath: Path, content: str) -> str:
    parts: list[str] = []
    rel = filepath.name

    parts.append(f"This is {rel} from the {repo_name} open-source project.")

    if "--!strict" in content:
        parts.append("It uses --!strict mode for type safety.")
    if "--!native" in content or "@native" in content:
        parts.append("It uses @native annotations for NCG compilation on hot paths.")
    if "export type" in content:
        parts.append("It exports typed interfaces for consumers.")
    if "table.freeze" in content:
        parts.append("It freezes tables for immutability.")
    if "Signal" in content or "signal" in content.lower():
        parts.append("It uses a signal/event pattern for decoupled communication.")
    if "Promise" in content:
        parts.append("It uses Promises for async control flow.")
    if "metatab" in content.lower() or "__index" in content:
        parts.append("It uses metatables for OOP-style class construction.")
    if "RunService" in content:
        parts.append("It hooks into RunService for frame-based updates.")

    lines = content.count("\n")
    if lines > 200:
        parts.append(f"At ~{lines} lines, this is a substantial module demonstrating production-level patterns.")

    return " ".join(parts)


def make_instruction(repo_name: str, repo_desc: str, filepath: Path) -> str:
    stem = filepath.stem
    # PascalCase to spaced
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem).strip()
    ext = filepath.suffix

    return (
        f"Write a Luau module called `{stem}{ext}` in the style of {repo_desc}. "
        f"The module should implement {spaced.lower()} functionality. "
        f"Use idiomatic Luau with proper type annotations and follow "
        f"Roblox open-source conventions."
    )


def build_example(
    repo_name: str,
    repo_info: dict,
    repo_dir: Path,
    filepath: Path,
    content: str,
) -> dict:
    category = infer_category(filepath, content)
    difficulty = difficulty_from_length(content)
    reasoning = generate_reasoning(repo_name, filepath, content)
    instruction = make_instruction(repo_name, repo_info["desc"], filepath)

    completion = f"<think>\n{reasoning}\n</think>\n\n{content}"

    rel_path = filepath.relative_to(repo_dir)

    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": completion},
    ]

    return {
        "messages": messages,
        "source": "oss_roblox",
        "category": category,
        "file_path": f"{repo_name}/{rel_path}",
        "difficulty": difficulty,
        "has_reasoning": True,
        "verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract training data from open-source Roblox/Luau repos.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without writing output.",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        choices=list(REPOS.keys()),
        default=None,
        help="Select specific repos to extract from.",
    )
    args = parser.parse_args()

    selected = {k: REPOS[k] for k in (args.repos or REPOS.keys())}

    # Resolve clone directory
    if CACHE_DIR.parent.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        base_dir = CACHE_DIR
        use_temp = False
        print(f"Using cache dir: {base_dir}")
    else:
        base_dir = Path(tempfile.mkdtemp(prefix="oss-roblox-"))
        use_temp = True
        print(f"Using temp dir: {base_dir}")

    examples: list[dict] = []
    stats: dict[str, int] = {}

    try:
        for name, info in selected.items():
            print(f"\n--- {name} ---")
            repo_dir = clone_repo(name, info, base_dir)
            if repo_dir is None:
                continue

            files = collect_lua_files(repo_dir, info["focus"])
            print(f"  Found {len(files)} Lua/Luau files")
            repo_count = 0

            for fp in files:
                try:
                    content = fp.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                # Skip trivially small files
                stripped = content.strip()
                if len(stripped) < 40 or stripped.count("\n") < 3:
                    continue

                example = build_example(name, info, repo_dir, fp, content)
                examples.append(example)
                repo_count += 1

            stats[name] = repo_count
            print(f"  Generated {repo_count} examples")

        # Summary
        print(f"\n{'=' * 50}")
        print(f"Total examples: {len(examples)}")
        for name, count in stats.items():
            print(f"  {name}: {count}")

        if args.dry_run:
            print("\n[dry-run] No output written.")
            if examples:
                print("\nSample example:")
                sample = examples[0]
                print(f"  instruction: {sample['messages'][0]['content'][:120]}...")
                print(f"  category: {sample['category']}")
                print(f"  difficulty: {sample['difficulty']}")
                print(f"  file_path: {sample['file_path']}")
            return

        # Write output
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"\nWrote {len(examples)} examples to {OUTPUT}")

    finally:
        if use_temp:
            print(f"\nCleaning up temp dir: {base_dir}")
            shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
