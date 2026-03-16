#!/usr/bin/env python3
from __future__ import annotations

"""
Generate Dataset A: Tool-calling SFT examples for tool selection training.

Produces single-turn and short multi-turn examples that teach WHEN and HOW
to select Studio MCP tools (or not). Three types:
  Type 1: "Which tool?" — 30 examples, 6 per tool
  Type 2: "Don't use a tool" — 10 examples
  Type 3: Multi-tool selection — 10 examples

Output: data/raw/tool_calling_sft.jsonl
"""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "tool_calling_sft.jsonl"

STUDIO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Luau code in Roblox Studio with full access to game services and the DataModel.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Luau code to execute"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_console_output",
            "description": "Get recent console output from Roblox Studio including prints, warnings, and errors.",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 50}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_script_in_play_mode",
            "description": "Execute a Luau script during play mode for testing runtime behavior.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Luau code to run in play mode"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_studio_mode",
            "description": "Get the current Studio mode (Edit or Play).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_stop_play",
            "description": "Toggle between Edit and Play mode in Roblox Studio.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = (
    "You are a Roblox Studio assistant for the Vertigo experience with access to MCP tools. "
    "Vertigo is a physics-driven vertical exploration game with:\n"
    "- 56 procedural zone builders spanning Y=-120 (Abyss) to Y=210 (Quiet Ring)\n"
    "- Traversal abilities: grapple (180 stud range), glide (0.55 lift), wallrun, swim, slide, airdash\n"
    "- Vehicles: dirt bike (95 max speed), glider kite (0.78 lift)\n"
    "- Service/controller architecture with Init/Start two-phase boot\n"
    "- CollectionService tags: GrappleAnchor, WindVolume, CurrentVolume, Checkpoint, BloomCrystal\n"
    "- DataStore schema v7 with auto-save and migration\n"
    "- --!strict on all files, @native on hot paths, type annotations required\n\n"
    "Think step by step. Use tools only when you need to interact with Studio."
)

_call_counter = 0


def _next_id() -> str:
    global _call_counter
    _call_counter += 1
    return f"call_{_call_counter}"


def _reset_ids():
    global _call_counter
    _call_counter = 0


def tc(name: str, args: dict) -> dict:
    return {"id": _next_id(), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def tool_msg(call_id: str, content: str) -> dict:
    return {"role": "tool", "content": content, "tool_call_id": call_id}


def asst(content: str | None, tool_calls: list[dict] | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def sys() -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT}


def wrap(messages: list[dict], *, category: str) -> dict:
    return {
        "messages": messages,
        "tools": STUDIO_TOOLS,
        "source": "mcp_tool_calling",
        "task_family": "sft_scripter",
        "rights_basis": "generated",
        "teacher_model": "hand-authored",
        "has_reasoning": True,
        "category": category,
    }


# ── Type 1: "Which tool?" (30 examples, 6 per tool) ──────────────────────


def _type1(tool_name: str, specs: list[tuple], cat: str) -> list[dict]:
    """Build Type 1 single-tool examples from (prompt, reasoning, args, resp, reply) tuples."""
    examples = []
    for spec in specs:
        _reset_ids()
        prompt, reasoning, args, resp, reply = spec
        c = tc(tool_name, args)
        examples.append(
            wrap(
                [sys(), user(prompt), asst(f"<think>{reasoning}</think>", [c]), tool_msg(c["id"], resp), asst(reply)],
                category=cat,
            )
        )
    return examples


def type1_run_code() -> list[dict]:
    S = '{"success": true}'
    return _type1(
        "run_code",
        [
            (
                "Add a script that prints all GrappleAnchor positions.",
                "Need run_code to iterate CollectionService tags. Not a play-mode test or console read.",
                {
                    "code": 'local CS = game:GetService("CollectionService")\nfor _, a in CS:GetTagged("GrappleAnchor") do print(a.Name, a.Position) end'
                },
                S,
                "Done.",
            ),
            (
                "Set the Transparency of all BloomCrystals to 0.3.",
                "Property mutation on tagged instances. run_code is right — changing DataModel, not testing runtime.",
                {
                    "code": 'local CS = game:GetService("CollectionService")\nfor _, c in CS:GetTagged("BloomCrystal") do c.Transparency = 0.3 end\nprint("[OK] set")'
                },
                S,
                "Done.",
            ),
            (
                "Create a new Part named LandingPad at position (50, 0, 50).",
                "Creating an Instance in the DataModel. run_code, not play-mode or console.",
                {
                    "code": 'local p = Instance.new("Part")\np.Name = "LandingPad"\np.Size = Vector3.new(20,1,20)\np.Position = Vector3.new(50,0,50)\np.Anchored = true\np.Parent = workspace\nprint("[OK] created")'
                },
                S,
                "Done.",
            ),
            (
                "Rename the Workspace child 'OldZone' to 'DeprecatedZone'.",
                "Simple property write. run_code finds and renames it. No other tool relevant.",
                {
                    "code": 'local o = workspace:FindFirstChild("OldZone")\nif o then o.Name = "DeprecatedZone" print("[OK] renamed") else print("[WARN] not found") end'
                },
                S,
                "Done.",
            ),
            (
                "Count how many Checkpoints exist in the game.",
                "Count tagged instances. run_code to query CollectionService. No play-mode needed.",
                {
                    "code": 'local CS = game:GetService("CollectionService")\nprint("[OK] Checkpoints:", #CS:GetTagged("Checkpoint"))'
                },
                S,
                "Done.",
            ),
            (
                "Add a PointLight to every part tagged WindVolume.",
                "Inserting children into tagged instances. run_code — edit-mode mutation, not runtime test.",
                {
                    "code": 'local CS = game:GetService("CollectionService")\nfor _, v in CS:GetTagged("WindVolume") do\n\tlocal l = Instance.new("PointLight") l.Brightness = 2 l.Range = 16 l.Parent = v\nend\nprint("[OK] added")'
                },
                S,
                "Done.",
            ),
        ],
        "tool_selection_run_code",
    )


def type1_get_console() -> list[dict]:
    R = "Here's what I found in the console."
    return _type1(
        "get_console_output",
        [
            (
                "Check for any errors in the output log.",
                "get_console_output retrieves log lines. No code execution needed.",
                {"limit": 20},
                '{"output": [{"level":"error","message":"Asset failed to load"}]}',
                R,
            ),
            (
                "What's in the output log right now?",
                "Direct console read. get_console_output reads the Studio output window.",
                {"limit": 50},
                '{"output": ["[OK] NeonPillar created"]}',
                R,
            ),
            (
                "Any warnings from the last script I ran?",
                "Check for warnings. get_console_output retrieves recent console lines.",
                {"limit": 30},
                '{"output": ["[WARN] OldZone not found"]}',
                R,
            ),
            (
                "Show me the last 5 print statements.",
                "Console read with specific limit. get_console_output with limit=5.",
                {"limit": 5},
                '{"output": ["p1","p2","p3","p4","p5"]}',
                R,
            ),
            (
                "Did my script produce any output?",
                "Verify output from previous run_code. get_console_output reads prints.",
                {"limit": 10},
                '{"output": ["[OK] Checkpoint count: 42"]}',
                R,
            ),
            (
                "Are there any error messages about DataStore?",
                "Looking for error patterns in console. get_console_output returns log lines.",
                {"limit": 50},
                '{"output": ["[error] DataStore request was throttled"]}',
                R,
            ),
        ],
        "tool_selection_get_console",
    )


def type1_play_mode() -> list[dict]:
    S, R = '{"success": true}', "Test executed in play mode."
    return _type1(
        "run_script_in_play_mode",
        [
            (
                "Test if the player can grapple to the nearest anchor.",
                "Runtime gameplay test. run_script_in_play_mode, not run_code — need active play session.",
                {
                    "code": 'local CS = game:GetService("CollectionService")\nprint("[TEST] Anchors:", #CS:GetTagged("GrappleAnchor"))'
                },
                S,
                R,
            ),
            (
                "Verify the NPC walks to the checkpoint after spawning.",
                "Runtime behavior test — observe NPC during play. run_script_in_play_mode, not run_code.",
                {
                    "code": 'local n = workspace:FindFirstChild("TestNPC")\nprint("[TEST] NPC:", n and n:FindFirstChildOfClass("Humanoid").MoveDirection or "not found")'
                },
                S,
                R,
            ),
            (
                "Check if the dirt bike spawns correctly at runtime.",
                "Vehicle spawning is runtime. run_script_in_play_mode to test in play mode.",
                {
                    "code": 'local v = workspace:FindFirstChild("Vehicles")\nprint("[TEST] Vehicles:", v and #v:GetChildren() or 0)'
                },
                S,
                R,
            ),
            (
                "Verify that the player's health regenerates during gameplay.",
                "Health regen is runtime-only. Need run_script_in_play_mode to observe it live.",
                {
                    "code": 'local p = game:GetService("Players").LocalPlayer\nlocal h = p.Character and p.Character:FindFirstChildOfClass("Humanoid")\nprint("[TEST] Health:", h and h.Health or "N/A")'
                },
                S,
                R,
            ),
            (
                "Test the glide ability when the player jumps off a ledge.",
                "Glide requires physics simulation. run_script_in_play_mode for active gameplay testing.",
                {
                    "code": 'local p = game:GetService("Players").LocalPlayer\nlocal r = p.Character and p.Character:FindFirstChild("HumanoidRootPart")\nprint("[TEST] Y:", r and r.Position.Y or "N/A")'
                },
                S,
                R,
            ),
            (
                "Check if the airdash cooldown resets properly between uses.",
                "Cooldown is runtime state. run_script_in_play_mode to inspect it.",
                {
                    "code": 'local p = game:GetService("Players").LocalPlayer\nlocal c = p:FindFirstChild("AbilityController")\nprint("[TEST] Airdash:", c and c:GetAttribute("AirdashReady") or "N/A")'
                },
                S,
                R,
            ),
        ],
        "tool_selection_play_mode",
    )


def type1_get_mode() -> list[dict]:
    M, R = '{"mode": "Edit"}', "Studio is currently in Edit mode."
    return _type1(
        "get_studio_mode",
        [
            ("What mode am I in?", "Mode query. get_studio_mode returns Edit or Play. No code needed.", {}, M, R),
            ("Check if we're in Edit or Play mode.", "Direct mode inquiry. get_studio_mode is precise.", {}, M, R),
            ("Am I in play mode right now?", "Mode check. get_studio_mode answers directly.", {}, M, R),
            ("Before I run code, what mode is Studio in?", "Pre-flight mode check. get_studio_mode.", {}, M, R),
            ("Is the game running or stopped?", "Play state query. get_studio_mode distinguishes Edit/Play.", {}, M, R),
            ("Tell me the current Studio state.", "Mode query. No mutations or console reads needed.", {}, M, R),
        ],
        "tool_selection_get_mode",
    )


def type1_start_stop() -> list[dict]:
    S = '{"success": true}'
    return _type1(
        "start_stop_play",
        [
            (
                "Start a playtest.",
                "User wants play mode. start_stop_play toggles Edit→Play.",
                {},
                S,
                "Play mode started.",
            ),
            (
                "Switch to play mode so I can test something.",
                "Mode transition. start_stop_play, not run_code.",
                {},
                S,
                "Switched to Play mode.",
            ),
            ("Stop the current test.", "Exit play mode. start_stop_play toggles back.", {}, S, "Play mode stopped."),
            (
                "End the playtest and go back to editing.",
                "Return to edit. start_stop_play.",
                {},
                S,
                "Back in Edit mode.",
            ),
            ("Toggle play mode.", "Direct toggle. start_stop_play flips Edit↔Play.", {}, S, "Mode toggled."),
            (
                "I need to enter play mode to debug an issue.",
                "Mode switch for debugging. start_stop_play, not run_code.",
                {},
                S,
                "Play mode activated.",
            ),
        ],
        "tool_selection_start_stop",
    )


# ── Type 2: "Don't use a tool" (10 examples) ─────────────────────────────


def type2_no_tool() -> list[dict]:
    examples = []
    specs = [
        (
            "Explain the Init/Start two-phase boot pattern.",
            "Architecture concept question. No Studio interaction needed — just explain.",
            "The Init/Start pattern is a two-phase boot. Init sets up state with no side effects. Start connects listeners and begins loops. This guarantees all modules are initialized before any Start runs.",
        ),
        (
            "What's the difference between server and client scripts in Roblox?",
            "Conceptual question. No tool needed — just explain.",
            "Server scripts run in ServerScriptService (game logic, data, security). Client scripts (LocalScripts) run on each player's device (input, UI, camera). The server is authoritative — clients send requests via RemoteEvents.",
        ),
        (
            "Write me a module for managing player inventory.",
            "The user wants code written, not executed. No tool call needed.",
            "--!strict\nlocal InventoryService = {}\nlocal inventories: {[Player]: {string}} = {}\nfunction InventoryService:Init() inventories = {} end\nfunction InventoryService:AddItem(player: Player, itemId: string)\n\tlocal inv = inventories[player] or {}\n\ttable.insert(inv, itemId)\n\tinventories[player] = inv\nend\nfunction InventoryService:GetItems(player: Player): {string}\n\treturn inventories[player] or {}\nend\nreturn InventoryService",
        ),
        (
            "How does CollectionService tagging work?",
            "Conceptual question. No tool needed.",
            "CollectionService assigns string tags to Instances. Use GetTagged('Tag') to find all tagged objects, and GetInstanceAddedSignal('Tag') to react to new ones. Vertigo uses tags like GrappleAnchor, WindVolume, Checkpoint, and BloomCrystal.",
        ),
        (
            "What are the best practices for --!strict mode?",
            "Luau conventions question. No tool needed.",
            "In --!strict: annotate all function signatures, use :: Type for WaitForChild casts, avoid `any`, use export type in shared modules, and let inference handle obvious local assignments.",
        ),
        (
            "Can you describe how DataStore schema migration works in Vertigo?",
            "Architecture question. No tool needed.",
            "Vertigo uses DataStore schema v7 with auto-migration. Each profile has a schemaVersion field. On load, migrations run sequentially (v1 through v7). Auto-save runs on a timer and on PlayerRemoving with session locking.",
        ),
        (
            "What Luau types should I use for a position tracker?",
            "Type design question. No tool needed.",
            "Use Vector3 for positions, CFrame for orientation, number for timestamps. Define an export type PositionRecord = { position: Vector3, timestamp: number, playerId: number }. Store in a {PositionRecord} array.",
        ),
        (
            "Explain how RemoteEvents work for client-server communication.",
            "Conceptual question. No tool needed.",
            "RemoteEvents enable one-way client-server communication. Client calls FireServer, server handles OnServerEvent. Server calls FireClient, client handles OnClientEvent. Always validate on server. Vertigo uses RequestUseAbility, StateSync, etc.",
        ),
        (
            "How should I structure a new zone builder?",
            "Design pattern question. No tool needed.",
            "Zone builders live in src/Server/World/Builders/. Each returns a table with :Build() that creates parts, sets materials, applies CollectionService tags, and parents under a zone folder. All parts must be Anchored. WorldService calls builders in Y-order.",
        ),
        (
            "What's the purpose of @native in Luau?",
            "Language feature question. No tool needed.",
            "The @native attribute enables Luau NCG — compiling to machine code for hot paths. Use on Heartbeat/RenderStepped callbacks. Requires full type annotations. Vertigo marks all per-frame physics and traversal functions @native.",
        ),
    ]
    for prompt, reasoning, reply in specs:
        _reset_ids()
        examples.append(
            wrap(
                [
                    sys(),
                    user(prompt),
                    asst(f"<think>{reasoning}</think>"),
                    asst(reply),
                ],
                category="no_tool_needed",
            )
        )
    return examples


# ── Type 3: Multi-tool (10 examples) ─────────────────────────────────────


def _multi2(prompt, t1_name, t1_args, t1_resp, think1, t2_name, t2_args, t2_resp, think2, reply):
    """Build a 2-tool multi-step example."""
    _reset_ids()
    c1, c2 = tc(t1_name, t1_args), tc(t2_name, t2_args)
    return wrap(
        [
            sys(),
            user(prompt),
            asst(f"<think>{think1}</think>", [c1]),
            tool_msg(c1["id"], t1_resp),
            asst(f"<think>{think2}</think>", [c2]),
            tool_msg(c2["id"], t2_resp),
            asst(reply),
        ],
        category="multi_tool",
    )


def type3_multi_tool() -> list[dict]:
    examples = [
        _multi2(
            "Check what mode we're in, then count all grapple anchors.",
            "get_studio_mode",
            {},
            '{"mode": "Edit"}',
            "Two steps: check mode first, then run_code to count tagged instances.",
            "run_code",
            {
                "code": 'local CS = game:GetService("CollectionService")\nprint("[OK] Anchors:", #CS:GetTagged("GrappleAnchor"))'
            },
            '{"success": true}',
            "Edit mode confirmed. Now I can count anchors.",
            "We're in Edit mode. I ran a count of GrappleAnchor-tagged instances.",
        ),
        _multi2(
            "Scan for unanchored parts and show me the results.",
            "run_code",
            {
                "code": 'for _, c in workspace:GetChildren() do\n\tif c:IsA("BasePart") and not c.Anchored then warn("[FLOAT]", c.Name, c.Position) end\nend\nprint("[OK] Float scan complete")'
            },
            '{"success": true}',
            "Run code to find unanchored parts, then read console for warnings.",
            "get_console_output",
            {"limit": 20},
            '{"output": ["[FLOAT] LoosePart 10, 50, 30", "[OK] Float scan complete"]}',
            "Code executed. Now read console to see flagged parts.",
            "Found 1 unanchored part: LoosePart at (10, 50, 30).",
        ),
        _multi2(
            "Start play mode and verify a player spawns.",
            "start_stop_play",
            {},
            '{"success": true}',
            "Enter play mode first, then run a test script to check player loading.",
            "run_script_in_play_mode",
            {"code": 'local p = game:GetService("Players").LocalPlayer\nprint("[TEST] Player loaded:", p ~= nil)'},
            '{"success": true}',
            "Play mode started. Now test player spawning.",
            "Play mode started and the player spawned successfully.",
        ),
        _multi2(
            "Make sure we're in play mode — switch if needed.",
            "get_studio_mode",
            {},
            '{"mode": "Edit"}',
            "Check current mode first. If Edit, toggle to Play.",
            "start_stop_play",
            {},
            '{"success": true}',
            "Edit mode — toggling to Play.",
            "Switched from Edit to Play mode.",
        ),
        _multi2(
            "Fix the BrokenPart by anchoring it, then verify it worked.",
            "run_code",
            {
                "code": 'local p = workspace:FindFirstChild("BrokenPart")\nif p then p.Anchored = true print("[OK] Fixed") else print("[WARN] Not found") end'
            },
            '{"success": true}',
            "Run code to fix, then console to verify.",
            "get_console_output",
            {"limit": 5},
            '{"output": ["[OK] Fixed"]}',
            "Code ran. Checking console for confirmation.",
            "BrokenPart anchored. Console confirms the fix.",
        ),
        _multi2(
            "Test if RequestUseAbility remote exists at runtime, then show results.",
            "run_script_in_play_mode",
            {
                "code": 'local r = game:GetService("ReplicatedStorage"):FindFirstChild("RequestUseAbility")\nprint("[TEST] exists:", r ~= nil)'
            },
            '{"success": true}',
            "Use run_script_in_play_mode for runtime check, then console for output.",
            "get_console_output",
            {"limit": 5},
            '{"output": ["[TEST] exists: true"]}',
            "Script ran. Reading console for test result.",
            "RequestUseAbility exists at runtime. Test passed.",
        ),
        _multi2(
            "Stop the playtest and clean up the TestDummy part.",
            "start_stop_play",
            {},
            '{"success": true}',
            "Stop play mode first, then remove the part in edit mode.",
            "run_code",
            {
                "code": 'local p = workspace:FindFirstChild("TestDummy")\nif p then p:Destroy() print("[OK] Removed") end'
            },
            '{"success": true}',
            "Back in Edit mode. Cleaning up.",
            "Playtest stopped. TestDummy removed.",
        ),
        _multi2(
            "Check console for errors, then fix undersized grapple anchors.",
            "get_console_output",
            {"limit": 20},
            '{"output": ["[WARN] GrappleAnchor_12 too small for reliable grapple"]}',
            "Diagnose first with console, then fix with run_code.",
            "run_code",
            {
                "code": 'local CS = game:GetService("CollectionService")\nfor _, a in CS:GetTagged("GrappleAnchor") do\n\tif a.Size.X < 2 then a.Size = Vector3.new(2, a.Size.Y, a.Size.Z) print("[FIX]", a.Name) end\nend'
            },
            '{"success": true}',
            "Warning found. Running fix for undersized anchors.",
            "Fixed undersized GrappleAnchors (X < 2 studs).",
        ),
    ]
    # 3-tool examples
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": 'local SSS = game:GetService("ServerScriptService")\nfor _, s in SSS:GetChildren() do print("[AUDIT]", s.Name, s.ClassName) end\nprint("[OK] done")'
        },
    )
    c3 = tc("get_console_output", {"limit": 30})
    examples.append(
        wrap(
            [
                sys(),
                user("Audit all scripts in ServerScriptService."),
                asst("<think>Three steps: check mode, run audit, read console.</think>", [c1]),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst("<think>Edit mode. Running audit.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst("<think>Reading console for audit results.</think>", [c3]),
                tool_msg(
                    c3["id"], '{"output": ["[AUDIT] init Script", "[AUDIT] HealthRegenScript Script", "[OK] done"]}'
                ),
                asst("ServerScriptService contains: init, HealthRegenScript."),
            ],
            category="multi_tool",
        )
    )

    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": 'local t = Instance.new("Part")\nt.Name = "TestTrigger"\nt.Size = Vector3.new(10,10,10)\nt.Position = Vector3.new(0,5,0)\nt.Anchored = true\nt.Transparency = 1\nt.CanCollide = false\nt.Parent = workspace\nprint("[OK] created")'
        },
    )
    c2 = tc("start_stop_play", {})
    c3 = tc(
        "run_script_in_play_mode", {"code": 'print("[TEST] Trigger:", workspace:FindFirstChild("TestTrigger") ~= nil)'}
    )
    examples.append(
        wrap(
            [
                sys(),
                user("Create a trigger zone, then playtest to verify it exists at runtime."),
                asst("<think>Create in edit, enter play, verify with play-mode script.</think>", [c1]),
                tool_msg(c1["id"], '{"success": true}'),
                asst("<think>Created. Entering play mode.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst("<think>Play mode active. Verifying trigger.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true}'),
                asst("TestTrigger created and verified at runtime."),
            ],
            category="multi_tool",
        )
    )

    return examples


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    all_examples: list[dict] = []
    all_examples.extend(type1_run_code())
    all_examples.extend(type1_get_console())
    all_examples.extend(type1_play_mode())
    all_examples.extend(type1_get_mode())
    all_examples.extend(type1_start_stop())
    all_examples.extend(type2_no_tool())
    all_examples.extend(type3_multi_tool())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Summary
    from collections import Counter

    cats = Counter(ex["category"] for ex in all_examples)
    print(f"Wrote {len(all_examples)} examples to {OUTPUT}")
    for cat, n in sorted(cats.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
