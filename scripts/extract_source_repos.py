#!/usr/bin/env python3
from __future__ import annotations

"""
Extract training data from 7 cloned rights-clean repos in data/sources/.

Each repo is processed differently to generate high-quality training pairs.
Output: JSONL files in data/raw/ matching TrainingExample schema.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

SYSTEM_PROMPT = (
    "You are an expert Roblox game developer specializing in Luau. "
    "You write production-grade Luau code with --!strict mode, full type annotations, "
    "and follow modern Roblox best practices. When explaining code, include reasoning "
    "about WHY each design choice was made."
)

EXECUTE_LUAU_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_luau",
        "description": "Execute Luau code in Roblox Studio. The code runs in the command bar context with access to game services, workspace, and all APIs.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Luau code to execute in Studio"}},
            "required": ["code"],
        },
    },
}


def make_example(
    messages: list[dict],
    *,
    source: str,
    task_family: str,
    rights_basis: str = "open_source",
    license_id: str,
    file_path: str | None = None,
    difficulty: int = 2,
    category: str = "general_luau",
    tools: list[dict] | None = None,
    project_id: str | None = None,
) -> dict:
    """Build a TrainingExample-compatible dict."""
    ex: dict = {
        "messages": messages,
        "source": source,
        "category": category,
        "difficulty": difficulty,
        "has_reasoning": any("<think>" in (m.get("content") or "") for m in messages),
        "verified": False,
        "provenance": {
            "source_id": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16],
            "rights_basis": rights_basis,
            "license": license_id,
            "task_family": task_family,
            "modality": "tool_trajectory" if tools else "code",
            "project_id": project_id or source,
        },
    }
    if file_path:
        ex["file_path"] = file_path
    if tools:
        ex["tools"] = tools
    return ex


def write_jsonl(examples: list[dict], path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  [{label}] Wrote {len(examples)} examples to {path}")


# --- 1. creator-docs: Markdown with code samples -> QA pairs ---


def extract_heading(content: str) -> str:
    m = re.search(r"^#{1,2}\s+(.+)", content, re.MULTILINE)
    return m.group(1).strip() if m else "Untitled"


def extract_code_blocks(content: str) -> list[tuple[str, str]]:
    return re.findall(r"```(lua[u]?)\n(.*?)```", content, re.DOTALL | re.IGNORECASE)


def heading_to_question(heading: str, rel_path: str) -> str:
    parts = Path(rel_path).parts
    topic = parts[0] if parts else "Roblox"
    if any(w in heading.lower() for w in ("how", "what", "when", "why")):
        return heading + " in Roblox Luau?"
    return f"How do I implement {heading} in Roblox ({topic})?"


def process_creator_docs(sources_dir: Path) -> list[dict]:
    docs_root = sources_dir / "creator-docs" / "content" / "en-us"
    if not docs_root.exists():
        print("  [creator-docs] SKIP: directory not found")
        return []
    examples = []
    for md_file in sorted(docs_root.rglob("*.md")):
        content = md_file.read_text(errors="replace")
        code_blocks = extract_code_blocks(content)
        if not code_blocks:
            continue
        heading = extract_heading(content)
        rel = md_file.relative_to(docs_root)
        question = heading_to_question(heading, str(rel))
        code_parts = [c.strip() for _, c in code_blocks[:3] if 20 <= len(c.strip()) <= 3000]
        if not code_parts:
            continue
        code_combined = "\n\n".join(f"```luau\n{c}\n```" for c in code_parts)
        explanation = re.sub(r"---.*?---", "", content, count=1, flags=re.DOTALL)
        explanation = re.sub(r"```.*?```", "", explanation, flags=re.DOTALL).strip()
        prose = " ".join(explanation.split()[:80])
        area = rel.parts[0] if rel.parts else "general"
        assistant_content = (
            f"<think>\nThe user is asking about {heading}. "
            f"This involves Roblox API usage for {area}. "
            f"Let me provide the relevant code with explanation.\n</think>\n\n"
            f"{prose}\n\nHere's how to do it:\n\n{code_combined}"
        )
        examples.append(
            make_example(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": assistant_content},
                ],
                source="roblox_creator_docs",
                task_family="sft_scripter",
                license_id="CC-BY-4.0",
                file_path=str(md_file),
                difficulty=2 if len(code_parts) == 1 else 3,
                category="api_usage",
            )
        )
    return examples


# --- 2. luau repo: Test cases -> understanding tasks ---

INTERESTING_PATTERNS = {
    "type": r"type\s+\w+",
    "generic": r"<\w+>",
    "coroutine": r"coroutine\.",
    "export type": r"export\s+type",
    "if expression": r"if\s+.+\s+then\s+.+\s+else",
    "buffer": r"buffer\.",
    "vector": r"vector\.",
    "typeof": r"typeof\(",
}

FEATURE_EXPLANATIONS = {
    "type": "- **Type annotations**: Luau's gradual type system for better tooling and runtime safety.",
    "export type": "- **Type annotations**: Luau's gradual type system for better tooling and runtime safety.",
    "generic": "- **Generics**: Parameterized types enable reusable, type-safe data structures.",
    "coroutine": "- **Coroutines**: Cooperative multitasking for yielding operations.",
    "buffer": "- **Buffer API**: Efficient binary data manipulation for networking.",
    "vector": "- **Vector API**: SIMD-accelerated vector math for performance-critical code.",
}


def process_luau_tests(sources_dir: Path) -> list[dict]:
    tests_dir = sources_dir / "luau" / "tests"
    if not tests_dir.exists():
        print("  [luau] SKIP: directory not found")
        return []
    examples = []
    for luau_file in sorted(tests_dir.rglob("*.luau")):
        content = luau_file.read_text(errors="replace")
        if len(content) < 100 or len(content) > 5000:
            continue
        features = [n for n, p in INTERESTING_PATTERNS.items() if re.search(p, content)]
        if not features:
            continue
        feat_str = ", ".join(features)
        explanations = "\n".join(dict.fromkeys(FEATURE_EXPLANATIONS[f] for f in features if f in FEATURE_EXPLANATIONS))
        assistant_content = (
            f"<think>\nThis Luau test code exercises: {feat_str}. "
            f"Let me analyze the code and identify key patterns.\n</think>\n\n"
            f"**Features used:** {feat_str}\n\n```luau\n{content[:2000]}\n```\n\n{explanations}"
        )
        examples.append(
            make_example(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Explain what this Luau code does and identify any type system features it uses:\n\n```luau\n{content[:2000]}\n```",
                    },
                    {"role": "assistant", "content": assistant_content},
                ],
                source="luau_repo",
                task_family="sft_scripter",
                license_id="MIT",
                file_path=str(luau_file),
                difficulty=min(5, 1 + len(features)),
            )
        )
    return examples


# --- 3. rojo: Plugin + Rust src -> architecture tasks ---

ROJO_TOPICS = {
    "sync": "How does Rojo handle file syncing between the filesystem and Roblox Studio?",
    "serve": "How does Rojo's live-sync server work?",
    "project": "How does Rojo handle project file configuration and path mapping?",
    "snapshot": "How does Rojo's snapshot system represent the Roblox instance tree?",
    "patch": "How does Rojo apply patches to reconcile filesystem changes with Studio?",
    "instance": "How does Rojo map filesystem paths to Roblox instances?",
}


def process_rojo(sources_dir: Path) -> list[dict]:
    rojo_dir = sources_dir / "rojo"
    if not rojo_dir.exists():
        print("  [rojo] SKIP: directory not found")
        return []
    examples, seen = [], set()

    for search_dir, ext, lang, diff in [
        (rojo_dir / "plugin", "*.lua", "lua", 3),
        (rojo_dir / "src", "*.rs", "rust", 4),
    ]:
        if not search_dir.exists():
            continue
        for f in sorted(search_dir.rglob(ext)):
            content = f.read_text(errors="replace")
            if len(content) < 100 or len(content) > 5000:
                continue
            name_lower, head = f.stem.lower(), content[:500].lower()
            topic = next((t for t in ROJO_TOPICS if t in name_lower or t in head), None)
            if not topic or topic in seen:
                continue
            seen.add(topic)
            assistant_content = (
                f"<think>\nExamining Rojo's {lang.title()} implementation for {topic} in {f.name}.\n</think>\n\n"
                f"Rojo's {topic} implementation in `{f.name}`:\n\n```{lang}\n{content[:2500]}\n```\n\n"
                f"This is part of Rojo's architecture for bridging the filesystem and Studio's DataModel."
            )
            examples.append(
                make_example(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": ROJO_TOPICS[topic]},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    source="rojo_ecosystem",
                    task_family="sft_scripter",
                    license_id="MPL-2.0",
                    file_path=str(f),
                    difficulty=diff,
                    project_id="rojo",
                )
            )
    return examples


# --- 4. selene: Lint rule docs -> code quality tasks ---
# Each entry: (rule_name, bad_code, good_code, explanation)
LINT_RULES = [
    (
        "divide_by_zero",
        "local inf = 1 / 0\nprint(inf)",
        "local inf = math.huge\nprint(inf)",
        "Division by zero is confusing. Use `math.huge` directly for infinity.",
    ),
    (
        "almost_swapped",
        "local a, b = 1, 2\na = b\nb = a",
        "local a, b = 1, 2\na, b = b, a",
        "This looks like an attempted swap but loses the original value of `a`. Use simultaneous assignment.",
    ),
    (
        "compare_nan",
        'if x == 0/0 then\n\tprint("is nan")\nend',
        'if x ~= x then\n\tprint("is nan")\nend',
        "NaN is never equal to anything, including itself. Use `x ~= x` to check for NaN.",
    ),
    (
        "empty_if",
        "if condition then\nend",
        "if condition then\n\t-- handle condition\n\tprocess()\nend",
        "Empty if blocks are likely incomplete code. Either add the body or remove the block.",
    ),
    (
        "duplicate_keys",
        'local t = {\n\tname = "foo",\n\tvalue = 1,\n\tname = "bar",\n}',
        'local t = {\n\tname = "bar",\n\tvalue = 1,\n}',
        "Duplicate table keys silently override earlier values. This is almost always a mistake.",
    ),
    (
        "global_usage",
        "function doStuff()\n\tresult = compute()\n\treturn result\nend",
        "function doStuff()\n\tlocal result = compute()\n\treturn result\nend",
        "Implicit globals are slow and error-prone. Always declare variables with `local`.",
    ),
    (
        "shadowing",
        "local x = 1\nfor i = 1, 10 do\n\tlocal x = i * 2\n\tprint(x)\nend\nprint(x)",
        "local x = 1\nfor i = 1, 10 do\n\tlocal doubled = i * 2\n\tprint(doubled)\nend\nprint(x)",
        "Variable shadowing can cause confusion about which variable is being referenced.",
    ),
    (
        "suspicious_reverse_loop",
        "for i = #list, 1 do\n\tprint(list[i])\nend",
        "for i = #list, 1, -1 do\n\tprint(list[i])\nend",
        "Reverse loops need a negative step (-1). Without it, the loop body never executes.",
    ),
    (
        "unbalanced_assignments",
        "local a, b, c = 1, 2",
        "local a, b = 1, 2\nlocal c = nil  -- explicitly nil if intended",
        "More variables than values means some are silently nil. Make this explicit.",
    ),
    (
        "manual_table_clone",
        "local copy = {}\nfor k, v in pairs(original) do\n\tcopy[k] = v\nend",
        "local copy = table.clone(original)",
        "Luau has a built-in `table.clone()` that is faster and clearer than manual iteration.",
    ),
    (
        "mixed_table",
        'local t = { 1, 2, name = "foo", 3 }',
        'local array = { 1, 2, 3 }\nlocal map = { name = "foo" }',
        "Mixing array and dictionary entries in a single table is confusing and error-prone.",
    ),
    (
        "constant_table_comparison",
        'if myTable == {} then\n\tprint("empty")\nend',
        'if next(myTable) == nil then\n\tprint("empty")\nend',
        "Comparing a table to `{}` always returns false (new table each time). Use `next()`.",
    ),
    (
        "roblox_incorrect_color3_new_bounds",
        "local color = Color3.new(255, 0, 0)",
        "local color = Color3.fromRGB(255, 0, 0)\n-- or: Color3.new(1, 0, 0)",
        "Color3.new() takes values 0-1, not 0-255. Use Color3.fromRGB() for 0-255 values.",
    ),
]


def process_selene(sources_dir: Path) -> list[dict]:
    selene_dir = sources_dir / "selene"
    if not selene_dir.exists():
        print("  [selene] SKIP: directory not found")
        return []
    docs_dir = selene_dir / "docs" / "src" / "lints"
    examples = []
    for rule_name, bad_code, good_code, explanation in LINT_RULES:
        doc_file = docs_dir / f"{rule_name}.md"
        extra = ""
        if doc_file.exists():
            doc_text = doc_file.read_text(errors="replace")
            why_m = re.search(r"## Why this is bad\n(.*?)(?=\n##|\Z)", doc_text, re.DOTALL)
            if why_m:
                extra = f"\n\n**Why:** {why_m.group(1).strip()}"
        assistant_content = (
            f"<think>\nThis code triggers the selene `{rule_name}` lint. "
            f"Let me identify the issue and provide the correct fix.\n</think>\n\n"
            f"**Issue: `{rule_name}`**\n\n{explanation}\n\n"
            f"The problematic code:\n```luau\n{bad_code}\n```\n\n"
            f"Should be written as:\n```luau\n{good_code}\n```{extra}"
        )
        examples.append(
            make_example(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"What's wrong with this Luau code?\n\n```luau\n{bad_code}\n```"},
                    {"role": "assistant", "content": assistant_content},
                ],
                source="rojo_ecosystem",
                task_family="sft_scripter",
                license_id="MPL-2.0",
                file_path=str(doc_file) if doc_file.exists() else None,
                difficulty=2,
                category="debugging",
                project_id="selene",
            )
        )
    return examples


# --- 5. open-game-eval: Eval files -> trajectory training data ---

DIFFICULTY_MAP = {"easy": 2, "medium": 3, "hard": 4, "very_hard": 5}


def parse_eval_prompt(content: str) -> tuple[str, str, str, list[str]]:
    """Extract scenario_name, prompt content, difficulty, and tags from eval file."""
    name_m = re.search(r'scenario_name\s*=\s*"([^"]+)"', content)
    scenario = name_m.group(1) if name_m else "unknown"
    # Try nested table format, then plain string, then DebugEval string-array format
    prompt_m = re.search(r"content\s*=\s*\[\[(.*?)\]\]", content, re.DOTALL)
    if not prompt_m:
        prompt_m = re.search(r'content\s*=\s*"([^"]+)"', content)
    if not prompt_m:
        prompt_m = re.search(r'prompt\s*=\s*\{\s*"((?:[^"\\]|\\.)*)"', content)
    prompt = prompt_m.group(1).strip() if prompt_m else ""
    diff_m = re.search(r'difficulty\s*=\s*"(\w+)"', content)
    diff = diff_m.group(1) if diff_m else "medium"
    tags_m = re.search(r"tags\s*=\s*\{([^}]+)\}", content)
    tags = [t.strip().strip('"') for t in tags_m.group(1).split(",")] if tags_m else []
    return scenario, prompt, diff, tags


def process_open_game_eval(sources_dir: Path) -> list[dict]:
    eval_dir = sources_dir / "open-game-eval"
    if not eval_dir.exists():
        print("  [open-game-eval] SKIP: directory not found")
        return []
    examples = []
    for d in [eval_dir / "Evals", eval_dir / "DebugEvals"]:
        if not d.exists():
            continue
        for lua_file in sorted(d.glob("*.lua")):
            content = lua_file.read_text(errors="replace")
            scenario, prompt, diff_str, tags = parse_eval_prompt(content)
            if not prompt:
                continue
            is_debug = "DebugEvals" in str(lua_file)
            kind = "debug/bug-fix" if is_debug else "game modification"
            action = "fix the bug" if is_debug else "implement the change"
            verb = "debug and fix" if is_debug else "implement"
            detail = "identify the issue and apply a fix" if is_debug else "make the requested changes"
            assistant_content = (
                f"<think>\nThe user wants me to: {prompt}\n\n"
                f"This is a {kind} task ({scenario}). I need to:\n"
                f"1. Understand the current game state\n"
                f"2. Write Luau code to {action}\n"
                f"3. Execute it using the execute_luau tool\n"
                f"4. Verify the result\n</think>\n\n"
                f"I'll {verb} this by executing Luau code in Studio. Let me {detail}."
            )
            examples.append(
                make_example(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    tools=[EXECUTE_LUAU_TOOL],
                    source="open_game_eval",
                    task_family="trajectory",
                    license_id="MIT",
                    file_path=str(lua_file),
                    difficulty=DIFFICULTY_MAP.get(diff_str, 3),
                    category="trajectory",
                    project_id="open-game-eval",
                )
            )
    return examples


# --- 6. roblox-ts: TS -> Luau conversion patterns ---

_ROBLOX_RE = [
    re.compile(p)
    for p in [
        r"game\.GetService",
        r"Workspace",
        r"Players",
        r"ReplicatedStorage",
        r"ServerScriptService",
        r"RunService",
        r"UserInputService",
        r"Instance\.new",
        r"\.Connect\(",
        r"RemoteEvent",
        r"RemoteFunction",
    ]
]


def process_roblox_ts(sources_dir: Path) -> list[dict]:
    ts_dir = sources_dir / "roblox-ts"
    if not ts_dir.exists():
        print("  [roblox-ts] SKIP: directory not found")
        return []
    examples = []
    for scan_dir in [ts_dir / "src", ts_dir / "tests"]:
        if not scan_dir.exists():
            continue
        for ts_file in sorted(scan_dir.rglob("*.ts")):
            content = ts_file.read_text(errors="replace")
            if len(content) < 100 or len(content) > 4000:
                continue
            is_spec = ts_file.name.endswith(".spec.ts")
            if not is_spec and not any(p.search(content) for p in _ROBLOX_RE):
                continue
            assistant_content = (
                f"<think>\nThis TypeScript code from roblox-ts shows TS-to-Luau patterns. "
                f"Let me convert the key patterns to idiomatic Luau.\n</think>\n\n"
                f"Here's the TypeScript source from `{ts_file.name}`:\n\n"
                f"```typescript\n{content[:2000]}\n```\n\n"
                f"Key conversion notes for Luau:\n"
                f"- TS imports become `require()` calls\n"
                f"- TS interfaces become `export type` definitions\n"
                f"- `const`/`let` become `local`\n"
                f"- Arrow functions `=>` become `function() end`\n"
                f"- Optional chaining `?.` is not available in Luau; use explicit nil checks\n"
                f"- Array methods like `.map()`, `.filter()` are not built-in; use loops or table library"
            )
            examples.append(
                make_example(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Convert this TypeScript (roblox-ts) code to idiomatic Luau:\n\n```typescript\n{content[:2000]}\n```",
                        },
                        {"role": "assistant", "content": assistant_content},
                    ],
                    source="rojo_ecosystem",
                    task_family="sft_scripter",
                    license_id="MIT",
                    file_path=str(ts_file),
                    difficulty=3,
                    project_id="roblox-ts",
                )
            )
    return examples


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract training data from cloned source repos")
    parser.add_argument("--sources-dir", type=Path, default=PROJECT_DIR / "data" / "sources")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "data" / "raw")
    args = parser.parse_args()
    sources_dir, output_dir = args.sources_dir, args.output_dir
    if not sources_dir.exists():
        print(f"ERROR: Sources directory not found: {sources_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    processors = [
        ("creator-docs", "roblox_creator_docs.jsonl", process_creator_docs),
        ("luau", "luau_tests.jsonl", process_luau_tests),
        ("rojo", "rojo_architecture.jsonl", process_rojo),
        ("selene", "selene_lint.jsonl", process_selene),
        ("open-game-eval", "opengameeval_tasks.jsonl", process_open_game_eval),
        ("roblox-ts", "roblox_ts_patterns.jsonl", process_roblox_ts),
    ]
    total = 0
    print(f"Extracting from {sources_dir} -> {output_dir}\n")
    for label, filename, proc_fn in processors:
        print(f"Processing {label}...")
        examples = proc_fn(sources_dir)
        if examples:
            write_jsonl(examples, output_dir / filename, label)
        else:
            print(f"  [{label}] No examples generated")
        total += len(examples)
    print(f"\n{'=' * 50}")
    print(f"Total examples extracted: {total}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
