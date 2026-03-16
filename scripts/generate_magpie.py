#!/usr/bin/env python3
from __future__ import annotations

"""
Magpie-style synthetic data generator for Roblox/Luau training data.

Magpie: model generates its own user queries, then we send those back to get
full responses — natural instruction-following pairs without seed prompts.
We rotate Roblox-specific system prompts for domain diversity.

Usage:
    uv run python scripts/generate_magpie.py --count 50 --api-model qwen3.5-27b
"""

import argparse, hashlib, json, re, subprocess, sys, time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "raw" / "magpie_synthetic.jsonl"
DEFAULT_API = "http://127.0.0.1:1234/v1"
REQUEST_TIMEOUT = 180

# -- Rotating system prompts (10) for domain diversity -------------------------
SYSTEM_PROMPTS = [
    "You are a Roblox/Luau coding expert. You write --!strict Luau with full type annotations, @native on hot-path functions, and follow Init/Start service lifecycle conventions.",
    "You are a Roblox game architect specializing in services and controllers. You design modular systems with two-phase boot (Init then Start), clean separation of concerns, and data-driven config modules.",
    "You are a Roblox physics programmer who optimizes with @native and vector.* SIMD math. You use math.lerp for interpolation, avoid closures in loops, and pool instances instead of creating them in Heartbeat callbacks.",
    "You are a Roblox UI/UX developer building responsive game interfaces with Luau. You create dynamic HUDs, inventory systems, and settings menus using modern Roblox UI patterns.",
    "You are a Roblox networking expert handling RemoteEvents and server-authoritative validation. You design secure client-server protocols that prevent exploits while keeping gameplay responsive.",
    "You are a Roblox DataStore specialist designing player data persistence. You handle session locking, data migration, retry logic, and schema versioning with ProfileService-style patterns.",
    "You are a Roblox NPC behavior programmer using CollectionService and pathfinding. You build state machines for AI agents, use spatial queries efficiently, and manage NPC pools.",
    "You are a Roblox Studio MCP tool user who automates game development tasks. You write scripts that interact with Studio APIs, automate builds, and integrate with external tooling.",
    "You are a Roblox performance engineer who profiles and optimizes Luau code. You identify bottlenecks with MicroProfiler, reduce memory allocations, and apply NCG compilation best practices.",
    "You are a Roblox world builder creating procedural environments and zones. You generate terrain, scatter objects with noise functions, and build dynamic worlds using code-driven builders.",
]

CATEGORY_KW: dict[str, list[str]] = {
    "service": ["service", "server", "init", "start", "lifecycle"],
    "controller": ["controller", "client", "input", "camera"],
    "physics": ["physics", "velocity", "force", "cframe", "vector", "raycast"],
    "networking": ["remote", "fireserver", "fireclient", "network", "replication"],
    "builder": ["builder", "procedural", "terrain", "generate", "spawn", "zone"],
    "config": ["config", "tuning", "settings", "constants"],
    "debugging": ["debug", "profile", "benchmark", "optimize", "performance"],
    "api_usage": ["datastore", "httpservice", "teleport", "marketplace"],
    "mcp_tool_call": ["mcp", "studio tool", "run_code", "plugin"],
    "general_luau": ["luau", "type", "generic", "module", "require"],
}

REFUSAL_RE = re.compile(r"I cannot|I'm sorry|I apologize|As an AI|I'm not able|I don't have access", re.I)
CODE_BLOCK_RE = re.compile(r"```(?:lua|luau)\b", re.I)
CODE_EXTRACT_RE = re.compile(r"```(?:lua|luau)\n(.*?)```", re.DOTALL | re.I)

QUERY_PROMPT = (
    "Generate a specific, practical Roblox/Luau coding question that a developer "
    "might ask. The question should require a code solution. Output ONLY the question, nothing else."
)

# -- API helpers ---------------------------------------------------------------


def api_call(
    url: str,
    messages: list[dict],
    model: str,
    temperature: float = 0.8,
    max_tokens: int = 2048,
    stop: list[str] | None = None,
) -> str | None:
    from api import call_llm, extract_content

    try:
        resp = call_llm(
            messages,
            model=model,
            api_base=url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=REQUEST_TIMEOUT,
        )
        return extract_content(resp)
    except Exception as exc:
        print(f"  [warn] API error: {exc}", file=sys.stderr)
        return None


def detect_model(api_url: str) -> str | None:
    from api import detect_model as _detect_model

    try:
        return _detect_model(api_base=api_url)
    except Exception:
        return None


# -- Quality filters -----------------------------------------------------------


def categorize(text: str) -> str:
    lower = text.lower()
    best, best_n = "general_luau", 0
    for cat, kws in CATEGORY_KW.items():
        n = sum(1 for kw in kws if kw in lower)
        if n > best_n:
            best, best_n = cat, n
    return best


def estimate_difficulty(text: str) -> int:
    score = 1
    if len(text) > 500:
        score += 1
    if len(text) > 1200:
        score += 1
    if len(CODE_BLOCK_RE.findall(text)) > 1:
        score += 1
    if any(kw in text.lower() for kw in ["@native", "vector.", "metatype", "generic"]):
        score += 1
    return min(score, 5)


def ensure_reasoning(text: str) -> str:
    if text.strip().startswith("<think>"):
        return text
    return "<think>\nLet me work through this step by step.\n</think>\n\n" + text


def verify_luau(code: str) -> bool:
    try:
        return (
            subprocess.run(
                ["luau-compile", "--check", "-"], input=code, capture_output=True, text=True, timeout=5
            ).returncode
            == 0
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True  # skip if unavailable


# -- Core generation -----------------------------------------------------------


def generate_query(api_url: str, model: str, sys_prompt: str, temp: float) -> str | None:
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": QUERY_PROMPT}]
    return api_call(api_url, msgs, model, temperature=temp, max_tokens=256)


def generate_response(api_url: str, model: str, sys_prompt: str, query: str, temp: float) -> str | None:
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}]
    return api_call(api_url, msgs, model, temperature=temp, max_tokens=4096)


def build_example(sys_prompt: str, query: str, response: str, model_name: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ],
        "source": "synthetic_evolved",
        "category": categorize(response),
        "has_reasoning": "<think>" in response,
        "difficulty": estimate_difficulty(response),
        "verified": False,
        "provenance": {
            "source_id": hashlib.sha256(query.encode()).hexdigest()[:16],
            "rights_basis": "generated",
            "task_family": "sft_builder",
            "modality": "code",
            "teacher_model": model_name,
        },
    }


# -- Main loop -----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Magpie-style synthetic data generator for Roblox/Luau")
    ap.add_argument("--api", default=DEFAULT_API, help="OpenAI-compatible API URL")
    ap.add_argument("--api-model", default=None, help="Model name (auto-detect if omitted)")
    ap.add_argument("--count", type=int, default=100, help="Total examples to generate")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path")
    ap.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    ap.add_argument("--min-length", type=int, default=200, help="Minimum response length")
    ap.add_argument("--verify-luau", action="store_true", help="Verify Luau syntax via luau-compile")
    args = ap.parse_args()

    model = args.api_model or detect_model(args.api)
    if not model:
        print("ERROR: Could not auto-detect model. Use --api-model.", file=sys.stderr)
        sys.exit(1)
    print(f"Model: {model}")
    print(f"Target: {args.count} examples -> {args.output}")
    print(f"Temperature: {args.temperature}, min-length: {args.min_length}")

    # Pre-flight check
    print("Pre-flight check...", end=" ", flush=True)
    if api_call(args.api, [{"role": "user", "content": "Say OK"}], model, temperature=0.0, max_tokens=10) is None:
        print("FAILED")
        print("ERROR: Model not responding. Is it loaded in LM Studio?", file=sys.stderr)
        sys.exit(1)
    print("OK")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    examples: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = args.count * 4
    stats = {"queries": 0, "too_short": 0, "no_code": 0, "refusal": 0, "duplicate": 0, "luau_fail": 0}

    while len(examples) < args.count and attempts < max_attempts:
        sys_prompt = SYSTEM_PROMPTS[attempts % len(SYSTEM_PROMPTS)]
        attempts += 1

        # Step 1: model generates a user query
        query = generate_query(args.api, model, sys_prompt, args.temperature)
        if not query or len(query.strip()) < 10:
            continue
        query = query.strip().strip('"').strip("'")
        stats["queries"] += 1

        # Dedup by first 100 chars
        prefix = query[:100].lower()
        if prefix in seen:
            stats["duplicate"] += 1
            continue
        seen.add(prefix)

        # Step 2: generate response to the query
        response = generate_response(args.api, model, sys_prompt, query, args.temperature)
        if not response:
            continue

        # Step 3: quality filters
        if len(response) < args.min_length:
            stats["too_short"] += 1
            continue
        if not CODE_BLOCK_RE.search(response):
            stats["no_code"] += 1
            continue
        if REFUSAL_RE.search(response[:300]):
            stats["refusal"] += 1
            continue

        # Step 4: optional Luau syntax verification
        if args.verify_luau:
            blocks = CODE_EXTRACT_RE.findall(response)
            if blocks and not all(verify_luau(b) for b in blocks):
                stats["luau_fail"] += 1
                continue

        # Step 5: ensure reasoning, build and save
        response = ensure_reasoning(response)
        examples.append(build_example(sys_prompt, query, response, model))
        n = len(examples)
        if n % 10 == 0 or n == args.count:
            print(f"  [{n}/{args.count}] {query[:80]}...")
        time.sleep(0.2)

    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(examples)} examples in {attempts} attempts.")
    print(f"  Queries: {stats['queries']}  Too-short: {stats['too_short']}  No-code: {stats['no_code']}")
    print(f"  Refusals: {stats['refusal']}  Duplicates: {stats['duplicate']}  Luau-fail: {stats['luau_fail']}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
