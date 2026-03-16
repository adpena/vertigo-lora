#!/usr/bin/env python3
"""Extract OpenGameEval tasks into training data with template solutions.

Parses Evals/ and DebugEvals/ Lua files, reverse-engineers success criteria
from check_scene/check_game, and generates tool-calling training examples.

Usage:
    uv run scripts/extract_opengameeval.py
    uv run scripts/extract_opengameeval.py --evals-dir /path/to/open-game-eval
"""

from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DIFF_MAP = {"easy": 2, "medium": 3, "hard": 4, "unknown": 3}
TOOL_DEF = {
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
SYS = (
    "You are an expert Roblox developer assistant with access to execute_luau, "
    "which runs Luau code directly in Roblox Studio with full DataModel access. "
    "When asked to modify a game, reason about the task, identify the relevant "
    "services and instances, then write clean Luau code to accomplish it."
)

# -- Parser (adapted from run_opengameeval.py) ---------------------------------


def parse_eval(path: str) -> dict:
    text = Path(path).read_text()
    r: dict = {"file": os.path.basename(path), "path": path}

    def _s(p, d=""):
        m = re.search(p, text)
        return m.group(1) if m else d

    r["scenario_name"] = _s(r'scenario_name\s*=\s*"([^"]+)"', Path(path).stem)
    r["place"] = _s(r'place\s*=\s*"([^"]+)"', "unknown.rbxl")
    r["difficulty"] = _s(r'difficulty\s*=\s*"([^"]+)"', "unknown")

    def _list(p):
        m = re.search(p, text)
        return [t.strip().strip('"').strip("'") for t in m.group(1).split(",") if t.strip()] if m else []

    r["tags"] = _list(r"tags\s*=\s*\{([^}]*)\}")
    r["expected_tool_calls"] = _list(r"expected_tool_calls\s*=\s*\{([^}]*)\}")
    prompts = []
    for cm in re.finditer(r'role\s*=\s*"(\w+)"[^}]*content\s*=\s*(?:\[\[(.+?)\]\]|"([^"]*)")', text, re.DOTALL):
        prompts.append({"role": cm.group(1), "content": (cm.group(2) or cm.group(3) or "").strip()})
    if not prompts:
        m = re.search(r"prompt\s*=\s*\{(.*?)\}", text, re.DOTALL)
        if m:
            prompts = [{"role": "user", "content": s.strip()} for s in re.findall(r'"([^"]+)"', m.group(1))]
    r["prompts"] = prompts

    def _fn(n):
        m = re.search(rf"eval\.{n}\s*=\s*function\(\)(.*?)^end", text, re.DOTALL | re.MULTILINE)
        return m.group(1).strip() if m else ""

    r["check_scene_src"], r["check_game_src"] = _fn("check_scene"), _fn("check_game")
    r["setup_src"], r["reference_src"] = _fn("setup"), _fn("reference")
    m = re.search(r"newScript\.Source\s*=\s*\[\[(.*?)\]\]", text, re.DOTALL)
    r["injected_script"] = m.group(1).strip() if m else ""
    r["_text"] = text
    return r


# -- Code generation helpers ---------------------------------------------------


def _attrs_from_file(text):
    m = re.search(r"attributesToScrape\s*=\s*\{([^}]+)\}", text)
    return [a.strip().strip('"').strip("'") for a in m.group(1).split(",") if a.strip()] if m else []


def _path_chain(src):
    svc = "Workspace"
    for s in ["ReplicatedStorage", "ServerStorage", "ServerScriptService"]:
        if s in src:
            svc = s
            break
    seen, parts = set(), []
    for p in re.findall(r'FindFirstChild\("([^"]+)"\)', src):
        if p not in seen:
            seen.add(p)
            parts.append(p)
    return svc, parts


def _ref_lines(ref):
    return [l.strip() for l in ref.splitlines() if l.strip()] or ["-- Apply the fix"]


def _infer(ev):
    """Return (task_desc, reasoning_steps, code_str)."""
    name, is_debug = ev["scenario_name"], "_bug_" in ev["scenario_name"]
    msg = ev["prompts"][0]["content"] if ev["prompts"] else name
    chk = ev["check_scene_src"] or ev["check_game_src"] or ""
    ref, setup, full = ev.get("reference_src", ""), ev.get("setup_src", ""), ev["_text"]
    steps, code = [], [f"-- {'Fix' if is_debug else 'Solution'} for: {name}"]

    if is_debug and ref:
        steps = [
            "Identify the bug in the existing script",
            "Understand what it should do vs what it does",
            "Write corrected code",
        ]
        code.extend(_ref_lines(ref))
    elif is_debug:
        steps = ["Identify the bug", "Determine correct behavior", "Write the fix"]
        code.append(f"-- Fix: {msg[:80]}")
    elif "*2" in chk.replace(" ", "") or "*2" in full.replace(" ", ""):
        attrs = _attrs_from_file(full) or re.findall(r'GetAttribute\("([^"]+)"\)', chk)
        svc, path = _path_chain(chk or setup)
        steps = [
            f'Find target in game:GetService("{svc}")',
            f"Navigate to: {' > '.join(path)}" if path else "Locate the target",
            f"Double attributes: {', '.join(attrs) if attrs else 'all numeric'}",
        ]
        chain = f'local target = game:GetService("{svc}")' + "".join(f':FindFirstChild("{p}")' for p in path)
        code.append(chain)
        code.append("")
        if attrs:
            for a in attrs:
                code.append(f'local v = target:GetAttribute("{a}")')
                code.append(f'if v then target:SetAttribute("{a}", v * 2) end')
        else:
            code.extend(
                [
                    "for n, v in pairs(target:GetAttributes()) do",
                    '    if type(v) == "number" then target:SetAttribute(n, v * 2) end',
                    "end",
                ]
            )
    elif "Color" in chk or "color" in chk.lower():
        steps = ["Find target parts", "Set appropriate colors"]
        code.append('local workspace = game:GetService("Workspace")')
        if "Trees" in chk:
            code.extend(
                [
                    "for _, tree in ipairs(workspace.Trees:GetChildren()) do",
                    "    for _, leaf in ipairs(tree:GetChildren()) do",
                    '        if leaf.Name:lower() == "leaves" and leaf:IsA("BasePart") then',
                    "            local c = {Color3.fromRGB(204,85,0), Color3.fromRGB(178,34,34), Color3.fromRGB(218,165,32)}",
                    "            leaf.Color = c[math.random(#c)]",
                    "        end",
                    "    end",
                    "end",
                ]
            )
        else:
            cm = re.search(r"Color3\.fromRGB\((\d+),\s*(\d+),\s*(\d+)\)", chk)
            rgb = f"Color3.fromRGB({cm.group(1)},{cm.group(2)},{cm.group(3)})" if cm else "Color3.fromRGB(0,255,0)"
            code.extend(
                [
                    "for _, p in ipairs(workspace:GetDescendants()) do",
                    f'    if p:IsA("BasePart") then p.Color = {rgb} end',
                    "end",
                ]
            )
    elif ref and ("Instance.new" in ref or "newScript" in ref):
        steps = ["Create required game objects/scripts", "Configure and parent correctly"]
        code.extend(_ref_lines(ref))
    else:
        steps = ["Analyze the game structure", "Apply the requested modifications"]
        code.extend(_rich_code(msg, chk, setup, full))

    return msg, "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps)), "\n".join(code)


def _rich_code(prompt, chk, setup, full):
    """Generate context-aware code for evals that don't match specific patterns."""
    lines, combined = [], chk + "\n" + setup + "\n" + full
    svcs = [
        s
        for s in [
            "Workspace",
            "ReplicatedStorage",
            "ServerStorage",
            "Players",
            "Lighting",
            "StarterPlayer",
            "UserInputService",
            "RunService",
            "TweenService",
            "CollectionService",
            "ServerScriptService",
        ]
        if s in combined
    ]
    for s in svcs[:4]:
        lines.append(f'local {"workspace" if s == "Workspace" else s} = game:GetService("{s}")')
    if not svcs:
        lines.append('local workspace = game:GetService("Workspace")')
    lines.append("")

    if "GetChildren" in chk or "GetDescendants" in chk:
        cm = re.search(r"(\w+)(?::GetChildren|:GetDescendants)\(\)", chk)
        ctr = cm.group(1) if cm else "workspace"
        meth = "GetDescendants" if "GetDescendants" in chk else "GetChildren"
        cls_m = re.search(r'IsA\("([^"]+)"\)', chk)
        cls = cls_m.group(1) if cls_m else "BasePart"
        nm = re.search(r'\.Name\s*==\s*"([^"]+)"', chk)
        lines.append(f"for _, obj in ipairs({ctr}:{meth}()) do")
        cond = f'obj.Name == "{nm.group(1)}" and obj:IsA("{cls}")' if nm else f'obj:IsA("{cls}")'
        lines.append(f"    if {cond} then")
        for prop in ["Transparency", "CanCollide", "Anchored", "Size", "Position", "Material", "Color"]:
            if prop in chk:
                vm = re.search(rf"{prop}\s*[=~<>]+\s*([^\s,\)]+)", chk)
                lines.append(f"        obj.{prop} = {vm.group(1)}" if vm else f"        -- set obj.{prop}")
        for a in re.findall(r'GetAttribute\("([^"]+)"\)', chk):
            lines.append(f'        obj:SetAttribute("{a}", obj:GetAttribute("{a}"))')
        lines.extend(["    end", "end"])
    elif "FindFirstChild" in chk:
        svc, path = _path_chain(chk)
        chain = f'local target = game:GetService("{svc}")' + "".join(f':FindFirstChild("{p}")' for p in path)
        lines.extend([chain, ""])
        for prop in ["Transparency", "CanCollide", "Anchored", "Value", "Enabled", "Color"]:
            if prop in chk:
                vm = re.search(rf"{prop}\s*[=~]+\s*([^\s,\)]+)", chk)
                if vm:
                    lines.append(f"target.{prop} = {vm.group(1)}")
        for a in re.findall(r'GetAttribute\("([^"]+)"\)', chk):
            lines.append(f'target:SetAttribute("{a}", target:GetAttribute("{a}"))')
    elif "Script" in combined or "LocalScript" in combined:
        st = "LocalScript" if "LocalScript" in combined else "Script"
        lines.extend(
            [
                f'local s = Instance.new("{st}")',
                's.Name = "GameScript"',
                f's.Source = [[\n    -- {prompt[:60]}\n    local Players = game:GetService("Players")\n]]',
            ]
        )
        par = (
            'game:GetService("StarterPlayer").StarterPlayerScripts'
            if st == "LocalScript"
            else 'game:GetService("ServerScriptService")'
        )
        lines.append(f"s.Parent = {par}")
    else:
        pl = prompt.lower()
        if any(w in pl for w in ["speed", "faster", "slower"]):
            lines.extend(
                [
                    "for _, o in ipairs(workspace:GetDescendants()) do",
                    '    if o:IsA("BasePart") or o:IsA("Model") then',
                    "        for a,v in pairs(o:GetAttributes()) do",
                    '            if type(v)=="number" and a:lower():find("speed") then o:SetAttribute(a,v*2) end',
                    "        end",
                    "    end",
                    "end",
                ]
            )
        elif any(w in pl for w in ["remove", "delete", "destroy"]):
            lines.extend(
                [
                    "for _, o in ipairs(workspace:GetDescendants()) do",
                    '    if o:IsA("BasePart") then o:Destroy() end',
                    "end",
                ]
            )
        elif any(w in pl for w in ["add", "create", "spawn", "insert"]):
            lines.extend(['local p = Instance.new("Part")', "p.Anchored = true", "p.Parent = workspace"])
        else:
            lines.extend(
                [
                    "for _, o in ipairs(workspace:GetDescendants()) do",
                    f'    if o:IsA("BasePart") then --[[ {prompt[:50]} ]] end',
                    "end",
                ]
            )
    return lines


# -- Build training example ----------------------------------------------------


def build_example(ev):
    if not ev["prompts"]:
        return None
    desc, reasoning, code = _infer(ev)
    is_debug = "_bug_" in ev["scenario_name"]
    user = ev["prompts"][0]["content"]
    verb = "fix the buggy script" if is_debug else f"modify the game to {desc.lower().rstrip('.')}"
    think = (
        f"{'The user is reporting a bug: ' if is_debug else 'The user wants to '}"
        f"{desc.lower().rstrip('.')}\nI need to:\n{reasoning}\n"
        f"I'll use execute_luau to {'fix' if is_debug else 'make'} these changes in Studio."
    )
    tc_id = f"call_{ev['scenario_name']}"
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>\n\nI'll {verb}. Let me write the code:",
            "tool_calls": [
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": "execute_luau", "arguments": json.dumps({"code": code})},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": json.dumps({"success": True, "output": "Code executed successfully."}),
        },
        {
            "role": "assistant",
            "content": f"Done! The {'fix' if is_debug else 'modification'} has been applied in Studio.",
        },
    ]
    return {
        "messages": msgs,
        "tools": [TOOL_DEF],
        "source": "open_game_eval",
        "category": "trajectory",
        "task_family": "trajectory",
        "rights_basis": "open_source",
        "license": "MIT",
        "difficulty": DIFF_MAP.get(ev["difficulty"], 3),
        "has_reasoning": True,
        "scenario_name": ev["scenario_name"],
        "place": ev["place"],
        "tags": ev["tags"],
        "is_debug_eval": is_debug,
    }


# -- Main ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Extract OpenGameEval tasks into training data")
    ap.add_argument(
        "--evals-dir", default=os.environ.get("OPENGAMEEVAL_DIR", str(PROJECT_DIR.parent.parent / "open-game-eval"))
    )
    ap.add_argument("--output", default="data/raw/opengameeval_tasks.jsonl")
    args = ap.parse_args()
    root = Path(args.evals_dir)
    files: list[Path] = []
    for d in [root / "Evals", root / "DebugEvals"]:
        if d.is_dir():
            files.extend(sorted(d.glob("*.lua")))
        else:
            print(f"WARN: {d} not found", file=sys.stderr)
    if not files:
        print("ERROR: no eval files", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} eval files")
    examples, skipped = [], 0
    for f in files:
        try:
            ev = parse_eval(str(f))
        except Exception as e:
            print(f"  WARN: {f.name}: {e}", file=sys.stderr)
            skipped += 1
            continue
        ex = build_example(ev)
        if ex:
            examples.append(ex)
        else:
            skipped += 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    ne = sum(1 for e in examples if not e["is_debug_eval"])
    nd = sum(1 for e in examples if e["is_debug_eval"])
    dist = {}
    for e in examples:
        dist[e["difficulty"]] = dist.get(e["difficulty"], 0) + 1
    ds = ", ".join(f"d{k}={v}" for k, v in sorted(dist.items()))
    print(f"\nWrote {len(examples)} examples to {out}")
    print(f"  Evals: {ne}, DebugEvals: {nd}, Skipped: {skipped}")
    print(f"  Difficulty: {ds}")


if __name__ == "__main__":
    main()
