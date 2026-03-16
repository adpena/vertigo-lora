#!/usr/bin/env python3
"""OpenGameEval local harness — run Roblox eval benchmarks against local models.

Usage:
    python run_opengameeval.py --dry-run --api http://127.0.0.1:1234/v1
    python run_opengameeval.py --api http://127.0.0.1:1234/v1 --mcp-endpoint http://127.0.0.1:3000
    python run_opengameeval.py --model mlx-community/Qwen2.5-Coder-32B-Instruct-4bit --adapter /path --dry-run
"""

from __future__ import annotations

import argparse, fnmatch, json, math, os, re, sys, urllib.error, urllib.request
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

PUBLISHED_LEADERBOARD = {
    "Gemini 3.1 Pro": {"pass_1": 55.32, "pass_5": 72.34, "cons_5": 54.67, "all_5": 44.66},
    "Claude Opus 4.6": {"pass_1": 51.91, "pass_5": 64.98, "cons_5": 52.25, "all_5": 39.29},
    "Claude Opus 4.5": {"pass_1": 44.47, "pass_5": 56.60, "cons_5": 43.82, "all_5": 35.44},
    "Claude Sonnet 4.5": {"pass_1": 38.51, "pass_5": 49.76, "cons_5": 39.87, "all_5": 25.81},
    "GPT-5.4": {"pass_1": 35.11, "pass_5": 55.43, "cons_5": 35.30, "all_5": 16.74},
    "Claude Haiku 4.5": {"pass_1": 35.74, "pass_5": 45.63, "cons_5": 36.20, "all_5": 25.46},
    "GPT-5.2": {"pass_1": 30.64, "pass_5": 46.08, "cons_5": 29.52, "all_5": 19.69},
    "GPT-OSS-120B": {"pass_1": 29.79, "pass_5": 46.81, "cons_5": 28.48, "all_5": 19.39},
}

EXECUTE_LUAU_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_luau",
        "description": "Execute Luau code in Roblox Studio with full access to game services.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Luau code to execute in Studio."}},
            "required": ["code"],
        },
    },
}

# Suit SDK tools — high-level agent toolkit for Roblox game development
SUIT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "suit_body",
            "description": "Convert natural language to executable Luau code via the Suit DSL compiler. The highest-level interface — describe what to do in plain English and get compiled, optimized Luau.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Natural language instruction (e.g., 'make all cars 2x faster', 'add a health regen script to the player')",
                    },
                    "compile": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true, compile to Luau. If false, return expression tree only.",
                    },
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suit_compile",
            "description": "Compile a Suit expression tree or Logo turtle source to optimized Luau code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "program": {"description": "Suit program dict with 'root' key, or Logo turtle source string."},
                    "source_type": {"type": "string", "enum": ["json", "turtle"], "default": "json"},
                    "format": {"type": "string", "enum": ["luau", "compact", "both"], "default": "luau"},
                },
                "required": ["program"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suit_studio",
            "description": "Run Luau code in Studio or insert assets. Use for direct Studio manipulation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Studio action (run_code, insert_asset)."},
                    "code": {"type": "string", "description": "Luau code to execute."},
                    "asset_id": {"type": "string", "description": "Asset ID to insert."},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suit_sense",
            "description": "Perception: read agent position, scan nearby objects, perform raycasts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Perception channels: position, nearby, raycast.",
                    },
                    "radius": {"type": "number", "default": 50, "description": "Search radius for nearby queries."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suit_move",
            "description": "Locomotion: move agent to position with easing, follow paths, PID tracking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"description": "Target position [x, y, z]."},
                    "speed": {"type": "number", "default": 16},
                    "ease": {"type": "string", "enum": ["linear", "cubic", "pid", "exponential"], "default": "linear"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suit_act",
            "description": "Agent actions: emotes, speech bubbles, equip items, interact with objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "emote": {"type": "string", "description": "Play emote animation."},
                    "say": {"type": "string", "description": "Display speech bubble."},
                    "equip": {"type": "string", "description": "Equip named item."},
                    "interact": {"type": "string", "description": "Interact with named object."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suit_camera",
            "description": "Camera control: look-at, follow agent, orbit, FOV, shake.",
            "parameters": {
                "type": "object",
                "properties": {
                    "look_at": {"description": "Camera look-at target."},
                    "follow": {"type": "string", "description": "Follow named agent."},
                    "fov": {"type": "number", "description": "Field of view."},
                    "orbit": {"type": "object", "description": "Orbit parameters."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suit_atmosphere",
            "description": "Environment control: time of day, fog, sky, ambient lighting, bloom, blur.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "number", "description": "Clock time 0-24."},
                    "fog": {"type": "object", "description": "Fog parameters."},
                    "bloom": {"type": "object", "description": "Bloom parameters."},
                    "ambient": {"description": "Ambient color."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suit_choreography",
            "description": "Multi-agent coordination: synchronized actions, formations, group emotes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Choreography action."},
                    "agents": {"type": "array", "items": {"type": "string"}, "description": "Agent names."},
                    "emote": {"type": "string", "description": "Group emote."},
                },
                "required": ["action"],
            },
        },
    },
]

# -- Lua eval parser ----------------------------------------------------------


def parse_eval_file(path: str) -> dict:
    text = Path(path).read_text()
    r: dict = {"file": os.path.basename(path), "path": path}

    def _s(pat: str, default: str = "") -> str:
        m = re.search(pat, text)
        return m.group(1) if m else default

    r["scenario_name"] = _s(r'scenario_name\s*=\s*"([^"]+)"', Path(path).stem)
    r["place"] = _s(r'place\s*=\s*"([^"]+)"', "unknown.rbxl")
    r["difficulty"] = _s(r'difficulty\s*=\s*"([^"]+)"', "unknown")

    def _list(pat: str) -> list[str]:
        m = re.search(pat, text)
        return [t.strip().strip('"').strip("'") for t in m.group(1).split(",") if t.strip()] if m else []

    r["tags"] = _list(r"tags\s*=\s*\{([^}]*)\}")
    r["expected_tool_calls"] = _list(r"expected_tool_calls\s*=\s*\{([^}]*)\}")
    prompts = []
    for cm in re.finditer(r'role\s*=\s*"(\w+)"[^}]*content\s*=\s*(?:\[\[(.+?)\]\]|"([^"]*)")', text, re.DOTALL):
        prompts.append({"role": cm.group(1), "content": (cm.group(2) or cm.group(3) or "").strip()})
    r["prompts"] = prompts

    def _fn(name: str) -> str:
        m = re.search(rf"eval\.{name}\s*=\s*function\(\)(.*?)^end", text, re.DOTALL | re.MULTILINE)
        return m.group(1).strip() if m else ""

    r["check_scene_src"] = _fn("check_scene")
    r["check_game_src"] = _fn("check_game")
    r["setup_src"] = _fn("setup")
    return r


# -- Code extraction helper ---------------------------------------------------


def _extract_code_from_text(text: str) -> str | None:
    blocks = re.findall(r"```(?:lua(?:u)?)\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\n(.*?)```", text, re.DOTALL)
    return blocks[0].strip() if blocks else None


# -- Model interaction ---------------------------------------------------------


def generate_code_api(api_base: str, model: str, prompts: list[dict], tools: list[dict] | None = None) -> dict:
    from api import call_llm

    messages = list(prompts)
    # Inline tools into the system/user message for API servers that don't support
    # the tools parameter natively (LM Studio, vLLM without tool support, etc.)
    if tools:
        [t["function"]["name"] for t in tools]
        tool_desc = json.dumps([t["function"] for t in tools], indent=2)
        tool_instruction = (
            "\n\nYou have access to the following tools:\n```json\n"
            + tool_desc
            + "\n```\n\nTo use a tool, respond with a JSON code block containing "
            "a tool_call with the function name and arguments. For execute_luau, "
            "include the Luau code to run in Studio. You may also write code directly "
            "in a ```luau code block."
        )
        # Append to the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i] = dict(messages[i])
                messages[i]["content"] = messages[i]["content"] + tool_instruction
                break

    try:
        data = call_llm(
            messages,
            model=model,
            api_base=api_base,
            temperature=0.7,
            max_tokens=4096,
            timeout=120,
        )
    except Exception as e:
        return {"error": str(e), "code": None, "raw": None, "used_tool_call": False}
    msg = data.get("choices", [{}])[0].get("message", {})
    code, used_tc = None, False
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        if fn.get("name") == "execute_luau":
            try:
                code = json.loads(fn.get("arguments", "{}")).get("code")
                used_tc = True
            except json.JSONDecodeError:
                pass
    if not code and msg.get("content"):
        code = _extract_code_from_text(msg["content"])
    return {"code": code, "raw": data, "used_tool_call": used_tc, "error": None}


def generate_code_mlx(model: str, adapter: str | None, prompts: list[dict]) -> dict:
    try:
        from mlx_lm import generate, load  # type: ignore
    except ImportError:
        return {"error": "mlx_lm not installed", "code": None, "raw": None, "used_tool_call": False}
    m, tok = load(model, adapter_path=adapter, tokenizer_config={})
    prompt_text = tok.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    output = generate(m, tok, prompt=prompt_text, max_tokens=4096, temp=0.7)
    return {"code": _extract_code_from_text(output), "raw": output, "used_tool_call": False, "error": None}


# -- Studio MCP ----------------------------------------------------------------


def mcp_call(endpoint: str, method: str, payload: dict, timeout: int = 30) -> dict:
    url = f"{endpoint.rstrip('/')}/{method}"
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"ok": True, "data": json.loads(resp.read())}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def execute_in_studio(ep: str, code: str, place: str) -> dict:
    return mcp_call(ep, "run_code", {"code": code})


def run_check_functions(ep: str, cs_src: str, cg_src: str) -> dict:
    r: dict = {"check_scene": None, "check_game": None}
    if cs_src:
        r["check_scene"] = mcp_call(ep, "run_code", {"code": cs_src})
    if cg_src:
        r["check_game"] = mcp_call(ep, "run_script_in_play_mode", {"code": cg_src})
    return r


# -- Scoring -------------------------------------------------------------------


def dry_run_score(ev: dict, gen: dict) -> dict:
    code = gen.get("code")
    if not code:
        return {"code_generated": False, "used_tool_call": False, "references_relevant": False, "score": 0.0}
    kws = ["game:GetService", "workspace", "Instance.new", "FindFirstChild", "Script"]
    refs = sum(1 for kw in kws if kw.lower() in code.lower()) >= 2
    tc = gen.get("used_tool_call", False)
    return {
        "code_generated": True,
        "used_tool_call": tc,
        "references_relevant": refs,
        "score": 0.4 + (0.3 if tc else 0) + (0.3 if refs else 0),
    }


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# -- Output --------------------------------------------------------------------


def safe_output_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p.with_name(f"{p.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{p.suffix}"))
    return path


def write_results_atomic(path: str, results: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, path)


def print_results_table(results: list[dict], model_name: str, dry_run: bool, with_suit: bool = False) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    mode = "dry-run (no Studio execution)" if dry_run else "full (Studio execution)"
    suit_label = " + Suit SDK" if with_suit else " (execute_luau only)"
    print(f"\n{'=' * 78}")
    print("OpenGameEval Local Harness -- Results")
    print(f"Model: {model_name}{suit_label}\nMode:  {mode}\nTasks: {passed}/{total}")
    print(f"{'=' * 78}")
    print(f"  {'Scenario':<42} {'Diff':<9} {'CodeGen':<9} {'Studio':<9} {'Result':<9}")
    print(f"  {'-' * 42} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}")
    for r in results:
        cg = "PASS" if r.get("code_generated") else "FAIL"
        if dry_run:
            st, ov = "SKIP", ("PARTIAL" if r.get("code_generated") else "FAIL")
        else:
            st = "PASS" if r.get("studio_passed") else "FAIL"
            ov = "PASS" if r.get("passed") else "FAIL"
        print(f"  {r.get('scenario_name', '?')[:42]:<42} {r.get('difficulty', '?')[:9]:<9} {cg:<9} {st:<9} {ov:<9}")
    p1 = (passed / total * 100) if total else 0
    near = min(PUBLISHED_LEADERBOARD.items(), key=lambda x: abs(x[1]["pass_1"] - p1))
    print(f"\n  Pass@1: {p1:.1f}%  (nearest: {near[0]}: {near[1]['pass_1']}%)")
    print("\n  Published Leaderboard:")
    for name, s in sorted(PUBLISHED_LEADERBOARD.items(), key=lambda x: -x[1]["pass_1"]):
        tag = " <--" if abs(s["pass_1"] - p1) < 3 else ""
        print(f"    {name:<22} pass@1={s['pass_1']:5.1f}%  pass@5={s['pass_5']:5.1f}%{tag}")
    print()


# -- Main ----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenGameEval local benchmark harness")
    ap.add_argument(
        "--evals-dir",
        default=os.environ.get("OPENGAMEEVAL_DIR", str(PROJECT_DIR.parent.parent / "open-game-eval" / "Evals")),
    )
    ap.add_argument("--api", default="http://127.0.0.1:1234/v1", help="OpenAI-compatible API base URL")
    ap.add_argument("--api-model", default=None, help="Model name for API (auto-detect if omitted)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path for local mlx_lm")
    ap.add_argument("--model", default=None, help="Model path for local mlx_lm")
    ap.add_argument("--mcp-endpoint", default="http://127.0.0.1:3000", help="Studio MCP endpoint")
    ap.add_argument("--samples", type=int, default=1, help="Samples per task for pass@k")
    ap.add_argument("--output", default="data/eval/opengameeval_results.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="Parse and generate but skip Studio execution")
    ap.add_argument("--tasks", nargs="*", default=None, help="Filter eval files (glob patterns)")
    ap.add_argument(
        "--with-suit",
        action="store_true",
        help="Include Suit SDK tools alongside execute_luau (tests SDK as force multiplier)",
    )
    args = ap.parse_args()
    use_mlx = args.model is not None

    # Auto-detect model name
    model_name = args.api_model
    if not model_name and not use_mlx:
        from api import detect_model as _detect_model

        try:
            model_name = _detect_model(api_base=args.api)
        except Exception:
            pass
        if not model_name:
            model_name = "unknown"
    elif use_mlx:
        model_name = Path(args.model).name

    # Discover eval files
    evals_dir = Path(args.evals_dir)
    if not evals_dir.is_dir():
        print(f"Error: evals directory not found: {evals_dir}", file=sys.stderr)
        sys.exit(1)
    eval_files = sorted(evals_dir.glob("*.lua"))
    if args.tasks:
        eval_files = [ef for ef in eval_files if any(fnmatch.fnmatch(ef.name, p) for p in args.tasks)]
    if not eval_files:
        print("No eval files matched.", file=sys.stderr)
        sys.exit(1)

    suit_label = " + Suit SDK" if args.with_suit else ""
    print(f"OpenGameEval Harness -- {len(eval_files)} tasks, model={model_name}{suit_label}")
    print(f"Mode: {'dry-run' if args.dry_run else 'full'}, samples={args.samples}")
    if not args.dry_run:
        print(f"MCP endpoint: {args.mcp_endpoint}")
    print()

    evals = []
    for ef in eval_files:
        try:
            evals.append(parse_eval_file(str(ef)))
        except Exception as e:
            print(f"  WARN: failed to parse {ef.name}: {e}")

    results = []
    places_needed: set[str] = set()

    for i, ev in enumerate(evals):
        name = ev["scenario_name"]
        places_needed.add(ev["place"])
        print(f"  [{i + 1}/{len(evals)}] {name} (difficulty={ev['difficulty']}, place={ev['place']})")
        if not ev["prompts"]:
            print("    SKIP: no prompt extracted")
            results.append({**ev, "passed": False, "code_generated": False, "error": "no prompt"})
            continue

        tools = [EXECUTE_LUAU_TOOL] + (SUIT_TOOLS if args.with_suit else [])
        sample_results = []
        for s in range(args.samples):
            gen = (
                generate_code_mlx(args.model, args.adapter, ev["prompts"])
                if use_mlx
                else generate_code_api(args.api, model_name, ev["prompts"], tools=tools)
            )
            if gen.get("error"):
                print(f"    Sample {s + 1}: ERROR -- {gen['error']}")
                sample_results.append({"passed": False, "error": gen["error"]})
                continue

            code = gen.get("code")
            cg = code is not None and len(code) > 10
            print(
                f"    Sample {s + 1}: code={'yes' if cg else 'no'} tool_call={gen.get('used_tool_call', False)} len={len(code) if code else 0}"
            )
            preview = (code[:200] + "...") if code and len(code) > 200 else code

            if args.dry_run:
                ds = dry_run_score(ev, gen)
                sample_results.append(
                    {"passed": ds["score"] >= 0.7, "code_generated": cg, "dry_run_score": ds, "code_preview": preview}
                )
            else:
                passed = studio_passed = False
                if cg:
                    ex = execute_in_studio(args.mcp_endpoint, code, ev["place"])
                    if ex.get("ok"):
                        chk = run_check_functions(args.mcp_endpoint, ev["check_scene_src"], ev["check_game_src"])
                        scene_ok = (not ev["check_scene_src"]) or (chk["check_scene"] and chk["check_scene"].get("ok"))
                        game_ok = (not ev["check_game_src"]) or (chk["check_game"] and chk["check_game"].get("ok"))
                        studio_passed = passed = bool(scene_ok and game_ok)
                        if not passed:
                            errs = []
                            for k in ("check_scene", "check_game"):
                                if chk[k] and not chk[k].get("ok"):
                                    errs.append(f"{k}: {chk[k].get('error', '?')}")
                            print(f"      Check failures: {'; '.join(errs)}")
                    else:
                        print(f"      Studio exec error: {ex.get('error', '?')}")
                sample_results.append(
                    {"passed": passed, "code_generated": cg, "studio_passed": studio_passed, "code_preview": preview}
                )

        any_passed = any(sr.get("passed") for sr in sample_results)
        n_passed = sum(1 for sr in sample_results if sr.get("passed"))
        results.append(
            {
                "scenario_name": name,
                "difficulty": ev["difficulty"],
                "place": ev["place"],
                "tags": ev["tags"],
                "expected_tool_calls": ev["expected_tool_calls"],
                "passed": any_passed,
                "code_generated": any(sr.get("code_generated") for sr in sample_results),
                "studio_passed": any(sr.get("studio_passed") for sr in sample_results) if not args.dry_run else None,
                "n_passed": n_passed,
                "n_samples": args.samples,
                "pass_at_1": compute_pass_at_k(args.samples, n_passed, min(args.samples, 1)),
                "with_suit": args.with_suit,
                "model": model_name,
                "samples": sample_results,
            }
        )

    out_path = safe_output_path(args.output)
    write_results_atomic(out_path, results)
    print(f"\nResults written to: {out_path}")
    if not args.dry_run:
        print(f"\nPlaces needed (must be open in Studio): {', '.join(sorted(places_needed))}")
    print_results_table(results, model_name, args.dry_run, args.with_suit)


if __name__ == "__main__":
    main()
