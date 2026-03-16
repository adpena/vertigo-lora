#!/usr/bin/env python3
from __future__ import annotations

"""
Generate Dataset B: Studio MCP action traces with tool calling.

Produces realistic multi-turn trajectories showing observation -> plan -> action -> verify
patterns using Roblox Studio MCP tools. Includes failure-and-recovery traces.

Output: data/raw/studio_trajectories.jsonl
"""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "studio_trajectories.jsonl"

# ---------------------------------------------------------------------------
# Tool definitions (actual Roblox Studio MCP tools)
# ---------------------------------------------------------------------------

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
    "Think step by step. Check mode before running code. Verify results after mutations."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_call_counter = 0


def _next_id() -> str:
    global _call_counter
    _call_counter += 1
    return f"call_{_call_counter}"


def _reset_ids():
    global _call_counter
    _call_counter = 0


def tc(name: str, args: dict) -> dict:
    """Create a tool_call dict."""
    return {
        "id": _next_id(),
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


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


def wrap(messages: list[dict], *, category: str, difficulty: int) -> dict:
    return {
        "messages": messages,
        "tools": STUDIO_TOOLS,
        "source": "studio_trajectory",
        "category": category,
        "task_family": "trajectory",
        "rights_basis": "generated",
        "teacher_model": "hand-authored",
        "has_reasoning": True,
        "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# Type 1: Build tasks (15 examples)
# ---------------------------------------------------------------------------


def build_tasks() -> list[dict]:
    traces = []

    # 1. Add a neon part
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local part = Instance.new("Part")\n'
                'part.Name = "NeonPillar"\n'
                "part.Size = Vector3.new(4, 20, 4)\n"
                "part.Position = Vector3.new(0, 10, 0)\n"
                "part.Material = Enum.Material.Neon\n"
                'part.BrickColor = BrickColor.new("Cyan")\n'
                "part.Anchored = true\n"
                "part.Parent = workspace\n"
                'print("[OK] NeonPillar created")'
            )
        },
    )
    c3 = tc("get_console_output", {"limit": 5})
    traces.append(
        wrap(
            [
                sys(),
                user("Add a tall cyan neon pillar to the workspace."),
                asst("<think>I should check the Studio mode first, then create the part.</think>", [c1]),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst("<think>Edit mode confirmed. I'll create an anchored neon part with cyan color.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst("<think>Let me verify it was created by checking console output.</think>", [c3]),
                tool_msg(c3["id"], '{"output": ["[OK] NeonPillar created"]}'),
                asst("Done. I created a 4x20x4 anchored neon pillar at the origin with cyan color."),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 2. Create a health regen script
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                "--!strict\n"
                'local script = Instance.new("Script")\n'
                'script.Name = "HealthRegenScript"\n'
                "script.Source = [[\n"
                "--!strict\n"
                'local Players = game:GetService("Players")\n'
                "\n"
                "local REGEN_RATE: number = 5 -- HP per second\n"
                "local REGEN_INTERVAL: number = 1\n"
                "\n"
                "local function onCharacterAdded(character: Model)\n"
                '\tlocal humanoid = character:WaitForChild("Humanoid") :: Humanoid\n'
                "\twhile humanoid and humanoid.Health > 0 do\n"
                "\t\tif humanoid.Health < humanoid.MaxHealth then\n"
                "\t\t\thumanoid.Health = math.min(humanoid.Health + REGEN_RATE, humanoid.MaxHealth)\n"
                "\t\tend\n"
                "\t\ttask.wait(REGEN_INTERVAL)\n"
                "\tend\n"
                "end\n"
                "\n"
                "Players.PlayerAdded:Connect(function(player: Player)\n"
                "\tplayer.CharacterAdded:Connect(onCharacterAdded)\n"
                "end)\n"
                "]]\n"
                'script.Parent = game:GetService("ServerScriptService")\n'
                'print("[OK] HealthRegenScript created")'
            )
        },
    )
    c3 = tc("get_console_output", {"limit": 5})
    traces.append(
        wrap(
            [
                sys(),
                user("Create a health regeneration script for players. 5 HP per second."),
                asst(
                    "<think>I need to create a Script in ServerScriptService that regenerates player health. Let me check mode first.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst(
                    "<think>Good, Edit mode. I'll create a server script with a per-character regen loop using task.wait.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"success": true}'),
                asst("<think>Verify creation via console.</think>", [c3]),
                tool_msg(c3["id"], '{"output": ["[OK] HealthRegenScript created"]}'),
                asst(
                    "Created a HealthRegenScript in ServerScriptService. It regenerates 5 HP/second for all players using a per-character loop with proper type annotations and --!strict."
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    # 3. Set up a RemoteEvent handler
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local remote = Instance.new("RemoteEvent")\n'
                'remote.Name = "RequestAbilityActivation"\n'
                'remote.Parent = game:GetService("ReplicatedStorage")\n'
                'print("[OK] RemoteEvent created:", remote:GetFullName())'
            )
        },
    )
    c3 = tc(
        "run_code",
        {
            "code": (
                "--!strict\n"
                'local script = Instance.new("Script")\n'
                'script.Name = "AbilityActivationHandler"\n'
                "script.Source = [[\n"
                "--!strict\n"
                'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
                'local remote = ReplicatedStorage:WaitForChild("RequestAbilityActivation") :: RemoteEvent\n'
                "\n"
                "local COOLDOWNS: {[Player]: number} = {}\n"
                "local COOLDOWN_TIME: number = 2.0\n"
                "\n"
                "remote.OnServerEvent:Connect(function(player: Player, abilityId: string)\n"
                "\tlocal now: number = tick()\n"
                "\tif COOLDOWNS[player] and now - COOLDOWNS[player] < COOLDOWN_TIME then\n"
                '\t\twarn("[AbilityHandler] Cooldown active for", player.Name)\n'
                "\t\treturn\n"
                "\tend\n"
                "\tCOOLDOWNS[player] = now\n"
                '\tprint("[AbilityHandler] Activating", abilityId, "for", player.Name)\n'
                "end)\n"
                "]]\n"
                'script.Parent = game:GetService("ServerScriptService")\n'
                'print("[OK] AbilityActivationHandler created")'
            )
        },
    )
    c4 = tc("get_console_output", {"limit": 5})
    traces.append(
        wrap(
            [
                sys(),
                user("Set up a RemoteEvent for ability activation with server-side cooldown validation."),
                asst(
                    "<think>I need to: 1) Check mode, 2) Create the RemoteEvent, 3) Create the handler script with cooldown logic.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst("<think>Edit mode. First create the RemoteEvent in ReplicatedStorage.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst("<think>Now create the server handler script with cooldown tracking per player.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true}'),
                asst("<think>Verify both were created.</think>", [c4]),
                tool_msg(
                    c4["id"],
                    '{"output": ["[OK] RemoteEvent created: ReplicatedStorage.RequestAbilityActivation", "[OK] AbilityActivationHandler created"]}',
                ),
                asst(
                    "Set up the ability activation system:\n1. Created `RequestAbilityActivation` RemoteEvent in ReplicatedStorage\n2. Created server handler in ServerScriptService with a 2-second per-player cooldown"
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    # 4. Configure grapple tuning values
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local config = game:GetService("ReplicatedStorage"):FindFirstChild("Shared")\n'
                '\tand game:GetService("ReplicatedStorage").Shared:FindFirstChild("Config")\n'
                '\tand game:GetService("ReplicatedStorage").Shared.Config:FindFirstChild("Abilities")\n'
                "if config then\n"
                '\tprint("[OK] Abilities config found:", config:GetFullName())\n'
                "else\n"
                '\twarn("[WARN] Abilities config not found")\n'
                "end"
            )
        },
    )
    c3 = tc(
        "run_code",
        {
            "code": (
                "local attrs = {\n"
                "\tGrappleMaxRange = 180,\n"
                "\tGrappleReelSpeed = 65,\n"
                "\tGrappleLaunchForce = 45,\n"
                "\tGrappleCooldown = 0.8,\n"
                "\tGrappleMaxAngle = 75,\n"
                "}\n"
                "local ws = workspace\n"
                "for k, v in pairs(attrs) do\n"
                "\tws:SetAttribute(k, v)\n"
                "end\n"
                'print("[OK] Grapple tuning attributes set on workspace")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user(
                    "Set the grapple tuning values: 180 range, 65 reel speed, 45 launch force, 0.8s cooldown, 75 degree max angle."
                ),
                asst(
                    "<think>Let me check mode and then find the config module to understand current structure.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst("<think>I'll check if the Abilities config exists.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "<think>Config exists. I'll set the tuning values as workspace attributes for runtime access.</think>",
                    [c3],
                ),
                tool_msg(c3["id"], '{"success": true}'),
                asst(
                    "Grapple tuning values are set as workspace attributes: MaxRange=180, ReelSpeed=65, LaunchForce=45, Cooldown=0.8s, MaxAngle=75 degrees."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 5. Build a simple NPC
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local model = Instance.new("Model")\n'
                'model.Name = "GuideNPC"\n'
                "\n"
                'local humanoid = Instance.new("Humanoid")\n'
                'humanoid.DisplayName = "Aether Guide"\n'
                "humanoid.MaxHealth = 100\n"
                "humanoid.Health = 100\n"
                "humanoid.Parent = model\n"
                "\n"
                'local rootPart = Instance.new("Part")\n'
                'rootPart.Name = "HumanoidRootPart"\n'
                "rootPart.Size = Vector3.new(2, 2, 1)\n"
                "rootPart.Transparency = 1\n"
                "rootPart.Anchored = true\n"
                "rootPart.Position = Vector3.new(0, 50, 0)\n"
                "rootPart.Parent = model\n"
                "\n"
                'local head = Instance.new("Part")\n'
                'head.Name = "Head"\n'
                "head.Shape = Enum.PartType.Ball\n"
                "head.Size = Vector3.new(2, 2, 2)\n"
                "head.Position = Vector3.new(0, 53, 0)\n"
                "head.Anchored = true\n"
                "head.Material = Enum.Material.Neon\n"
                'head.BrickColor = BrickColor.new("Bright blue")\n'
                "head.Parent = model\n"
                "\n"
                'local torso = Instance.new("Part")\n'
                'torso.Name = "Torso"\n'
                "torso.Size = Vector3.new(2, 3, 1)\n"
                "torso.Position = Vector3.new(0, 51.5, 0)\n"
                "torso.Anchored = true\n"
                "torso.Material = Enum.Material.SmoothPlastic\n"
                'torso.BrickColor = BrickColor.new("Institutional white")\n'
                "torso.Parent = model\n"
                "\n"
                "model.PrimaryPart = rootPart\n"
                "model.Parent = workspace\n"
                "\n"
                'game:GetService("CollectionService"):AddTag(model, "NPC")\n'
                'game:GetService("CollectionService"):AddTag(model, "GuideNPC")\n'
                'print("[OK] GuideNPC created at Y=50 with tags NPC, GuideNPC")'
            )
        },
    )
    c3 = tc("get_console_output", {"limit": 5})
    traces.append(
        wrap(
            [
                sys(),
                user(
                    "Build a simple guide NPC at Y=50 in the Hub zone. It should have a neon blue head and white torso."
                ),
                asst(
                    "<think>I need to create a Model with Humanoid, HumanoidRootPart, Head, and Torso. Should tag it with CollectionService for runtime discovery. Check mode first.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst(
                    "<think>Edit mode. I'll build the NPC model with proper parts, set PrimaryPart, and add CS tags.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"success": true}'),
                asst("<think>Verify creation.</think>", [c3]),
                tool_msg(c3["id"], '{"output": ["[OK] GuideNPC created at Y=50 with tags NPC, GuideNPC"]}'),
                asst(
                    'Created "Aether Guide" NPC at Y=50 with:\n- Neon blue ball head\n- White smooth plastic torso\n- Invisible HumanoidRootPart as PrimaryPart\n- CollectionService tags: `NPC`, `GuideNPC`'
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    # 6. Add a checkpoint
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local checkpoint = Instance.new("Part")\n'
                'checkpoint.Name = "Checkpoint_Hub_01"\n'
                "checkpoint.Size = Vector3.new(8, 1, 8)\n"
                "checkpoint.Position = Vector3.new(0, 5, 0)\n"
                "checkpoint.Anchored = true\n"
                "checkpoint.Material = Enum.Material.ForceField\n"
                "checkpoint.Transparency = 0.5\n"
                "checkpoint.CanCollide = false\n"
                "checkpoint.Parent = workspace\n"
                "\n"
                'game:GetService("CollectionService"):AddTag(checkpoint, "Checkpoint")\n'
                'checkpoint:SetAttribute("ZoneId", "hub")\n'
                'checkpoint:SetAttribute("Order", 1)\n'
                'print("[OK] Checkpoint_Hub_01 created")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Add a checkpoint platform at the hub zone origin."),
                asst(
                    "<think>I'll create a semi-transparent force-field checkpoint part, tag it with CollectionService, and set zone attributes.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst("<think>Edit mode. Creating checkpoint with proper tags and attributes.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "Created `Checkpoint_Hub_01` at the hub origin: 8x1x8 force-field platform, tagged `Checkpoint`, with ZoneId=hub and Order=1."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 7. Create a proximity prompt interaction
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local part = workspace:FindFirstChild("GuideNPC")\n'
                "if not part then\n"
                '\twarn("[WARN] GuideNPC not found in workspace")\n'
                "\treturn\n"
                "end\n"
                "\n"
                'local prompt = Instance.new("ProximityPrompt")\n'
                'prompt.ActionText = "Talk"\n'
                'prompt.ObjectText = "Aether Guide"\n'
                "prompt.HoldDuration = 0.5\n"
                "prompt.MaxActivationDistance = 10\n"
                "prompt.RequiresLineOfSight = false\n"
                'prompt.Parent = part.PrimaryPart or part:FindFirstChild("HumanoidRootPart")\n'
                'print("[OK] ProximityPrompt added to GuideNPC")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Add a proximity prompt to the guide NPC so players can talk to it."),
                asst(
                    "<think>I need to find the GuideNPC and attach a ProximityPrompt to its root part. Check mode first.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst("<think>Edit mode. I'll find the NPC and add the prompt with appropriate settings.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    'Added a ProximityPrompt to the Guide NPC with "Talk" action, 0.5s hold duration, and 10 stud activation range.'
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 8. Create a wind volume
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local volume = Instance.new("Part")\n'
                'volume.Name = "WindVolume_Ascent_01"\n'
                "volume.Size = Vector3.new(40, 80, 40)\n"
                "volume.Position = Vector3.new(0, 100, 0)\n"
                "volume.Anchored = true\n"
                "volume.Transparency = 1\n"
                "volume.CanCollide = false\n"
                "volume.Parent = workspace\n"
                "\n"
                'game:GetService("CollectionService"):AddTag(volume, "WindVolume")\n'
                'volume:SetAttribute("WindDirection", Vector3.new(0, 1, 0))\n'
                'volume:SetAttribute("WindForce", 30)\n'
                'volume:SetAttribute("ZoneId", "ascent")\n'
                'print("[OK] WindVolume_Ascent_01 created")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Create an upward wind volume in the Ascent zone at Y=100, 40x80x40 studs."),
                asst(
                    "<think>Wind volumes are invisible anchored parts with the WindVolume CollectionService tag plus direction/force attributes.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst(
                    "Created `WindVolume_Ascent_01` at Y=100: 40x80x40 invisible volume with upward wind force of 30."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 9. Create a kill brick
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local killBrick = Instance.new("Part")\n'
                'killBrick.Name = "KillBrick_Abyss"\n'
                "killBrick.Size = Vector3.new(200, 2, 200)\n"
                "killBrick.Position = Vector3.new(0, -120, 0)\n"
                "killBrick.Anchored = true\n"
                "killBrick.Material = Enum.Material.Neon\n"
                'killBrick.BrickColor = BrickColor.new("Really red")\n'
                "killBrick.Transparency = 0.3\n"
                "killBrick.Parent = workspace\n"
                "\n"
                'game:GetService("CollectionService"):AddTag(killBrick, "KillBrick")\n'
                'killBrick:SetAttribute("DamageType", "void")\n'
                'print("[OK] KillBrick_Abyss created at Y=-120")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Add a kill brick at the bottom of the Abyss zone, Y=-120."),
                asst(
                    "<think>Kill bricks should be large, anchored, and tagged with CollectionService so the server can handle the death logic. Let me check mode.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst("<think>Edit mode. Creating a large neon red kill brick at the abyss floor.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst("Created a 200x2x200 kill brick at Y=-120 tagged `KillBrick` with DamageType=void."),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 10. Create a leaderboard script
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local script = Instance.new("Script")\n'
                'script.Name = "LeaderboardService"\n'
                "script.Source = [[\n"
                "--!strict\n"
                'local Players = game:GetService("Players")\n'
                "\n"
                "local function setupLeaderstats(player: Player)\n"
                '\tlocal leaderstats = Instance.new("Folder")\n'
                '\tleaderstats.Name = "leaderstats"\n'
                "\tleaderstats.Parent = player\n"
                "\n"
                '\tlocal crystals = Instance.new("IntValue")\n'
                '\tcrystals.Name = "Crystals"\n'
                "\tcrystals.Value = 0\n"
                "\tcrystals.Parent = leaderstats\n"
                "\n"
                '\tlocal height = Instance.new("IntValue")\n'
                '\theight.Name = "MaxHeight"\n'
                "\theight.Value = 0\n"
                "\theight.Parent = leaderstats\n"
                "end\n"
                "\n"
                "Players.PlayerAdded:Connect(setupLeaderstats)\n"
                "]]\n"
                'script.Parent = game:GetService("ServerScriptService")\n'
                'print("[OK] LeaderboardService created")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Create a leaderboard showing Crystals collected and MaxHeight reached."),
                asst(
                    "<think>Standard Roblox leaderboard uses leaderstats folder in Player. Check mode first.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst(
                    "<think>I'll create a server script that sets up leaderstats with Crystals and MaxHeight IntValues.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "Created LeaderboardService in ServerScriptService. Each player gets a `leaderstats` folder with Crystals (IntValue) and MaxHeight (IntValue)."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 11. Add grapple anchors to a zone
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local CS = game:GetService("CollectionService")\n'
                "local positions = {\n"
                "\tVector3.new(10, 60, 5),\n"
                "\tVector3.new(-15, 75, 10),\n"
                "\tVector3.new(5, 90, -8),\n"
                "\tVector3.new(20, 105, 15),\n"
                "\tVector3.new(-10, 120, -5),\n"
                "}\n"
                "\n"
                "for i, pos in ipairs(positions) do\n"
                '\tlocal anchor = Instance.new("Part")\n'
                '\tanchor.Name = string.format("GrappleAnchor_Ascent_%02d", i)\n'
                "\tanchor.Size = Vector3.new(3, 3, 3)\n"
                "\tanchor.Shape = Enum.PartType.Ball\n"
                "\tanchor.Position = pos\n"
                "\tanchor.Anchored = true\n"
                "\tanchor.Material = Enum.Material.Neon\n"
                '\tanchor.BrickColor = BrickColor.new("Lime green")\n'
                "\tanchor.Parent = workspace\n"
                '\tCS:AddTag(anchor, "GrappleAnchor")\n'
                "end\n"
                'print("[OK] Created 5 GrappleAnchors in Ascent zone")'
            )
        },
    )
    c2 = tc("get_console_output", {"limit": 5})
    traces.append(
        wrap(
            [
                sys(),
                user("Place 5 grapple anchors along a vertical path in the Ascent zone, from Y=60 to Y=120."),
                asst(
                    "<think>I'll create 5 neon ball parts along a vertical path, tagging each with GrappleAnchor via CollectionService.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst("<think>Verify they were created.</think>", [c2]),
                tool_msg(c2["id"], '{"output": ["[OK] Created 5 GrappleAnchors in Ascent zone"]}'),
                asst(
                    "Placed 5 grapple anchors along a vertical path from Y=60 to Y=120 in the Ascent zone, each tagged `GrappleAnchor`."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 12. Create a DataStore wrapper
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local module = Instance.new("ModuleScript")\n'
                'module.Name = "DataService"\n'
                "module.Source = [[\n"
                "--!strict\n"
                'local DataStoreService = game:GetService("DataStoreService")\n'
                'local Players = game:GetService("Players")\n'
                "\n"
                "local DataService = {}\n"
                "\n"
                'local STORE_NAME: string = "VertigoPlayerData_v7"\n'
                "local store = DataStoreService:GetDataStore(STORE_NAME)\n"
                "local cache: {[number]: {[string]: any}} = {}\n"
                "\n"
                "function DataService:LoadProfile(player: Player): {[string]: any}?\n"
                '\tlocal key: string = "player_" .. tostring(player.UserId)\n'
                "\tlocal ok, data = pcall(function()\n"
                "\t\treturn store:GetAsync(key)\n"
                "\tend)\n"
                "\tif ok and data then\n"
                "\t\tcache[player.UserId] = data\n"
                "\t\treturn data\n"
                "\tend\n"
                "\tlocal default = {crystals = 0, maxHeight = 0, checkpoints = {}, schemaVersion = 7}\n"
                "\tcache[player.UserId] = default\n"
                "\treturn default\n"
                "end\n"
                "\n"
                "function DataService:SaveProfile(player: Player): boolean\n"
                "\tlocal data = cache[player.UserId]\n"
                "\tif not data then return false end\n"
                '\tlocal key: string = "player_" .. tostring(player.UserId)\n'
                "\tlocal ok, err = pcall(function()\n"
                "\t\tstore:SetAsync(key, data)\n"
                "\tend)\n"
                '\tif not ok then warn("[DataService] Save failed:", err) end\n'
                "\treturn ok\n"
                "end\n"
                "\n"
                "return DataService\n"
                "]]\n"
                'module.Parent = game:GetService("ServerScriptService")\n'
                'print("[OK] DataService module created")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Create a DataStore wrapper module for player profiles with load, save, and caching."),
                asst(
                    "<think>I'll create a ModuleScript with DataStore operations wrapped in pcall for safety. Check mode first.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst(
                    "<think>Edit mode. Creating the DataService module with schema v7 defaults, in-memory cache, and pcall-wrapped operations.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "Created DataService ModuleScript in ServerScriptService with:\n- Schema v7 default profiles\n- In-memory cache per player\n- pcall-wrapped GetAsync/SetAsync\n- LoadProfile and SaveProfile methods"
                ),
            ],
            category="trajectory",
            difficulty=4,
        )
    )

    # 13. Add a SurfaceGui info sign
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local part = Instance.new("Part")\n'
                'part.Name = "InfoSign_Hub"\n'
                "part.Size = Vector3.new(10, 6, 0.5)\n"
                "part.Position = Vector3.new(15, 8, 0)\n"
                "part.Anchored = true\n"
                "part.Material = Enum.Material.SmoothPlastic\n"
                'part.BrickColor = BrickColor.new("Dark stone grey")\n'
                "part.Parent = workspace\n"
                "\n"
                'local gui = Instance.new("SurfaceGui")\n'
                "gui.Face = Enum.NormalId.Front\n"
                "gui.Parent = part\n"
                "\n"
                'local title = Instance.new("TextLabel")\n'
                "title.Size = UDim2.new(1, 0, 0.3, 0)\n"
                'title.Text = "Welcome to Vertigo"\n'
                "title.TextScaled = true\n"
                "title.BackgroundTransparency = 1\n"
                "title.TextColor3 = Color3.fromRGB(0, 200, 255)\n"
                "title.Font = Enum.Font.GothamBold\n"
                "title.Parent = gui\n"
                "\n"
                'local body = Instance.new("TextLabel")\n'
                "body.Size = UDim2.new(1, 0, 0.7, 0)\n"
                "body.Position = UDim2.new(0, 0, 0.3, 0)\n"
                'body.Text = "Explore 330 vertical studs. Grapple, glide, and climb."\n'
                "body.TextScaled = true\n"
                "body.BackgroundTransparency = 1\n"
                "body.TextColor3 = Color3.fromRGB(220, 220, 220)\n"
                "body.Font = Enum.Font.Gotham\n"
                "body.Parent = gui\n"
                'print("[OK] InfoSign_Hub created")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Add a welcome sign near the hub spawn with a SurfaceGui."),
                asst(
                    "<think>I'll create a flat part with a SurfaceGui containing a title and body TextLabel.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst(
                    'Created an info sign near the hub with a SurfaceGui displaying "Welcome to Vertigo" and a brief description.'
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 14. Create a sound region
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local region = Instance.new("Part")\n'
                'region.Name = "SoundRegion_Ocean"\n'
                "region.Size = Vector3.new(100, 40, 100)\n"
                "region.Position = Vector3.new(0, -20, 0)\n"
                "region.Anchored = true\n"
                "region.Transparency = 1\n"
                "region.CanCollide = false\n"
                "region.Parent = workspace\n"
                "\n"
                'local sound = Instance.new("Sound")\n'
                'sound.Name = "OceanAmbience"\n'
                'sound.SoundId = "rbxassetid://9125402735"\n'
                "sound.Looped = true\n"
                "sound.Volume = 0.6\n"
                "sound.RollOffMode = Enum.RollOffMode.Linear\n"
                "sound.RollOffMaxDistance = 60\n"
                "sound.Parent = region\n"
                "\n"
                'game:GetService("CollectionService"):AddTag(region, "SoundRegion")\n'
                'region:SetAttribute("ZoneId", "ocean")\n'
                'print("[OK] SoundRegion_Ocean created")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Create an ambient sound region for the Ocean zone."),
                asst(
                    "<think>I'll create an invisible part with a Sound child for ambient ocean audio, tagged for runtime discovery.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst("<think>Edit mode. Creating the sound region with linear rolloff.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "Created `SoundRegion_Ocean` at Y=-20: 100x40x100 invisible volume with looping ocean ambience (linear rolloff, 60 stud max distance)."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 15. Create a bloom crystal cluster
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local CS = game:GetService("CollectionService")\n'
                "local center = Vector3.new(5, 140, -10)\n"
                "local offsets = {\n"
                "\tVector3.new(0, 0, 0),\n"
                "\tVector3.new(2, 3, 1),\n"
                "\tVector3.new(-1, 5, 2),\n"
                "\tVector3.new(3, 1, -1),\n"
                "}\n"
                "\n"
                'local cluster = Instance.new("Model")\n'
                'cluster.Name = "BloomCrystalCluster_01"\n'
                "cluster.Parent = workspace\n"
                "\n"
                "for i, offset in ipairs(offsets) do\n"
                '\tlocal crystal = Instance.new("Part")\n'
                '\tcrystal.Name = string.format("BloomCrystal_%02d", i)\n'
                "\tcrystal.Size = Vector3.new(1, math.random(3, 8), 1)\n"
                "\tcrystal.Position = center + offset\n"
                "\tcrystal.Anchored = true\n"
                "\tcrystal.Material = Enum.Material.Neon\n"
                '\tcrystal.BrickColor = BrickColor.new("Magenta")\n'
                "\tcrystal.Transparency = 0.15\n"
                "\tcrystal.Parent = cluster\n"
                '\tCS:AddTag(crystal, "BloomCrystal")\n'
                '\tcrystal:SetAttribute("Collected", false)\n'
                "end\n"
                'print("[OK] BloomCrystalCluster_01 created with 4 crystals")'
            )
        },
    )
    c2 = tc("get_console_output", {"limit": 5})
    traces.append(
        wrap(
            [
                sys(),
                user("Create a cluster of 4 bloom crystals at Y=140 in the AetherSea zone."),
                asst(
                    "<think>Bloom crystals are neon magenta parts tagged with BloomCrystal. I'll create a model containing 4 crystal parts at slightly offset positions with a Collected attribute for tracking.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst("<think>Verify.</think>", [c2]),
                tool_msg(c2["id"], '{"output": ["[OK] BloomCrystalCluster_01 created with 4 crystals"]}'),
                asst(
                    "Created a cluster of 4 bloom crystals at Y=140, each tagged `BloomCrystal` with a `Collected` attribute."
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    return traces


# ---------------------------------------------------------------------------
# Type 2: Debug tasks (10 examples)
# ---------------------------------------------------------------------------


def debug_tasks() -> list[dict]:
    traces = []

    # 1. Players disconnecting — RemoteEvent flood
    _reset_ids()
    c1 = tc("get_console_output", {"limit": 50})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local RS = game:GetService("ReplicatedStorage")\n'
                "for _, remote in RS:GetDescendants() do\n"
                '\tif remote:IsA("RemoteEvent") then\n'
                '\t\tprint("[Audit]", remote.Name, remote:GetFullName())\n'
                "\tend\n"
                "end"
            )
        },
    )
    c3 = tc(
        "run_code",
        {
            "code": (
                "--!strict\n"
                'local RS = game:GetService("ReplicatedStorage")\n'
                'local remote = RS:FindFirstChild("SyncState", true) :: RemoteEvent?\n'
                "if not remote then\n"
                '\twarn("[Fix] SyncState remote not found")\n'
                "\treturn\n"
                "end\n"
                "\n"
                "-- Add rate limiting wrapper\n"
                'local rateLimiter = Instance.new("Script")\n'
                'rateLimiter.Name = "SyncStateRateLimiter"\n'
                "rateLimiter.Source = [[\n"
                "--!strict\n"
                'local RS = game:GetService("ReplicatedStorage")\n'
                'local remote = RS:WaitForChild("SyncState") :: RemoteEvent\n'
                "local lastFire: {[Player]: number} = {}\n"
                "local MIN_INTERVAL: number = 0.1\n"
                "\n"
                "remote.OnServerEvent:Connect(function(player: Player, ...)\n"
                "\tlocal now = tick()\n"
                "\tif lastFire[player] and now - lastFire[player] < MIN_INTERVAL then\n"
                "\t\treturn -- rate limited\n"
                "\tend\n"
                "\tlastFire[player] = now\n"
                "\t-- forward to actual handler\n"
                "end)\n"
                "]]\n"
                'rateLimiter.Parent = game:GetService("ServerScriptService")\n'
                'print("[OK] Rate limiter installed for SyncState")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Players are getting disconnected randomly. Can you investigate?"),
                asst(
                    "<think>Random disconnects often indicate RemoteEvent flooding or network saturation. Let me check the console for errors first.</think>",
                    [c1],
                ),
                tool_msg(
                    c1["id"],
                    '{"output": ["[Warn] Excessive remote invocations from Player1 (SyncState: 847 calls/sec)", "[Warn] Excessive remote invocations from Player2 (SyncState: 923 calls/sec)", "[Error] Player1 was kicked for sending too many events"]}',
                ),
                asst(
                    "<think>Found it. SyncState is being fired hundreds of times per second. This is a classic RemoteEvent flood. Let me audit all remotes and then add rate limiting to SyncState.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "<think>I see the SyncState remote. I need to add server-side rate limiting to prevent the flood.</think>",
                    [c3],
                ),
                tool_msg(c3["id"], '{"success": true}'),
                asst(
                    "Found the issue: the `SyncState` RemoteEvent was being fired 800-900 times/sec per player, causing disconnects.\n\nFix: Installed a server-side rate limiter (0.1s minimum interval) that drops excessive calls silently. The client-side code should also be throttled to avoid unnecessary network traffic."
                ),
            ],
            category="debugging",
            difficulty=4,
        )
    )

    # 2. Grapple doesn't work
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc("get_console_output", {"limit": 30})
    c3 = tc(
        "run_code",
        {
            "code": (
                'local CS = game:GetService("CollectionService")\n'
                'local anchors = CS:GetTagged("GrappleAnchor")\n'
                'print("[Audit] GrappleAnchor count:", #anchors)\n'
                "for i, a in ipairs(anchors) do\n"
                "\tif i <= 5 then\n"
                '\t\tprint("  ", a:GetFullName(), "Anchored:", a.Anchored, "Pos:", a.Position)\n'
                "\tend\n"
                "end"
            )
        },
    )
    c4 = tc(
        "run_code",
        {
            "code": (
                "-- Check grapple config\n"
                'local range = workspace:GetAttribute("GrappleMaxRange")\n'
                'local cooldown = workspace:GetAttribute("GrappleCooldown")\n'
                'print("[Config] GrappleMaxRange:", range or "NOT SET")\n'
                'print("[Config] GrappleCooldown:", cooldown or "NOT SET")\n'
                "\n"
                "-- Check if ability is enabled\n"
                'local RS = game:GetService("ReplicatedStorage")\n'
                'local shared = RS:FindFirstChild("Shared")\n'
                "if shared then\n"
                '\tlocal config = shared:FindFirstChild("Config")\n'
                "\tif config then\n"
                '\t\tprint("[Config] Config folder exists:", config:GetFullName())\n'
                "\tend\n"
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Grapple hook isn't working for players. They press the button but nothing happens."),
                asst(
                    "<think>Could be: missing anchors, wrong mode, disabled ability, or config issue. Let me check mode and console.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst(
                    "<think>Edit mode — I can't test in play mode directly. Let me check console for errors and audit grapple anchor state.</think>",
                    [c2],
                ),
                tool_msg(
                    c2["id"],
                    '{"output": ["[GrappleService] Init complete", "[GrappleService] No valid targets in range for Player1"]}',
                ),
                asst(
                    "<think>The service is loaded but says no valid targets. Let me check if anchors exist and are properly tagged.</think>",
                    [c3],
                ),
                tool_msg(c3["id"], '{"success": true}'),
                asst("<think>Let me also check the config values.</think>", [c4]),
                tool_msg(c4["id"], '{"success": true}'),
                asst(
                    "Investigation complete. The grapple service is running but reports no valid targets in range. This could mean:\n1. GrappleAnchors exist but are too far from player spawn\n2. GrappleMaxRange attribute may be unset (defaulting to 0)\n\nCheck the anchor positions relative to player spawn and ensure GrappleMaxRange is set to 180."
                ),
            ],
            category="debugging",
            difficulty=4,
        )
    )

    # 3. Parts falling through floor
    _reset_ids()
    c1 = tc("get_console_output", {"limit": 20})
    c2 = tc(
        "run_code",
        {
            "code": (
                "local unanchored = {}\n"
                "for _, part in workspace:GetDescendants() do\n"
                '\tif part:IsA("BasePart") and not part.Anchored and part.Position.Y > -50 then\n'
                "\t\ttable.insert(unanchored, part)\n"
                "\tend\n"
                "end\n"
                'print("[Audit] Unanchored parts above Y=-50:", #unanchored)\n'
                "for i, p in ipairs(unanchored) do\n"
                "\tif i <= 10 then\n"
                '\t\tprint("  ", p:GetFullName(), "Y:", string.format("%.1f", p.Position.Y))\n'
                "\tend\n"
                "end"
            )
        },
    )
    c3 = tc(
        "run_code",
        {
            "code": (
                "local count = 0\n"
                "for _, part in workspace:GetDescendants() do\n"
                '\tif part:IsA("BasePart") and not part.Anchored and not part.Parent:IsA("Model") then\n'
                "\t\tpart.Anchored = true\n"
                "\t\tcount += 1\n"
                "\tend\n"
                "end\n"
                'print("[Fix] Anchored", count, "loose parts")'
            )
        },
    )
    c4 = tc("get_console_output", {"limit": 5})
    traces.append(
        wrap(
            [
                sys(),
                user("Some world parts are falling through the floor and disappearing."),
                asst(
                    "<think>Parts falling = unanchored BaseParts being affected by gravity. Let me check console and audit unanchored parts.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"output": ["[Workspace] Part fell below threshold: Platform_03 at Y=-892"]}'),
                asst(
                    "<think>Confirmed parts are falling. Let me find all unanchored parts above the abyss floor.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "<think>Found unanchored parts. I'll anchor all loose parts that aren't inside character models.</think>",
                    [c3],
                ),
                tool_msg(c3["id"], '{"success": true}'),
                asst("<think>Verify the fix.</think>", [c4]),
                tool_msg(c4["id"], '{"output": ["[Fix] Anchored 12 loose parts"]}'),
                asst(
                    "Found and fixed 12 unanchored world parts that were falling due to gravity. Anchored all loose BaseParts that aren't inside character Models. The builders should set `Anchored = true` when creating these parts."
                ),
            ],
            category="debugging",
            difficulty=3,
        )
    )

    # 4. Script error on player join
    _reset_ids()
    c1 = tc("get_console_output", {"limit": 30})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local SSS = game:GetService("ServerScriptService")\n'
                'local ds = SSS:FindFirstChild("DataService", true)\n'
                'if ds and ds:IsA("ModuleScript") then\n'
                '\tprint("[Source] DataService found:", ds:GetFullName())\n'
                "\tprint(ds.Source:sub(1, 500))\n"
                "end"
            )
        },
    )
    c3 = tc(
        "run_code",
        {
            "code": (
                'local SSS = game:GetService("ServerScriptService")\n'
                'local ds = SSS:FindFirstChild("DataService", true)\n'
                'if ds and ds:IsA("ModuleScript") then\n'
                "\tlocal src = ds.Source\n"
                '\tsrc = src:gsub("store:GetAsync%(key%)", "store:GetAsync(key) :: {[string]: any}?")\n'
                "\tds.Source = src\n"
                '\tprint("[Fix] Added type cast to GetAsync call")\n'
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Getting a type error when players join. Something about DataService."),
                asst("<think>Let me check the console for the specific error message.</think>", [c1]),
                tool_msg(
                    c1["id"],
                    '{"output": ["[Error] ServerScriptService.DataService:14: TypeError: Expected {[string]: any} got any", "[Error] DataService failed to load profile for Player1"]}',
                ),
                asst(
                    "<think>Type error on line 14 of DataService. GetAsync returns `any` but the code expects a typed table. Let me read the source.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "<think>The issue is a missing type cast on the GetAsync return value. In --!strict mode, the `any` return needs explicit casting.</think>",
                    [c3],
                ),
                tool_msg(c3["id"], '{"success": true}'),
                asst(
                    "Fixed. The `DataService:GetAsync(key)` call returns `any` which conflicted with --!strict type checking. Added an explicit `:: {[string]: any}?` type cast to satisfy the type checker."
                ),
            ],
            category="debugging",
            difficulty=3,
        )
    )

    # 5. Glide not activating when airborne
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc("start_stop_play", {})
    c3 = tc(
        "run_script_in_play_mode",
        {
            "code": (
                'local Players = game:GetService("Players")\n'
                "local player = Players.LocalPlayer\n"
                "if not player then\n"
                '\tprint("[Test] No local player")\n'
                "\treturn\n"
                "end\n"
                "local char = player.Character\n"
                "if char then\n"
                '\tlocal hrp = char:FindFirstChild("HumanoidRootPart")\n'
                '\tlocal hum = char:FindFirstChild("Humanoid") :: Humanoid?\n'
                "\tif hrp and hum then\n"
                '\t\tprint("[Test] Player Y:", string.format("%.1f", hrp.Position.Y))\n'
                '\t\tprint("[Test] FloorMaterial:", tostring(hum.FloorMaterial))\n'
                '\t\tprint("[Test] Airborne:", hum.FloorMaterial == Enum.Material.Air)\n'
                "\tend\n"
                "end"
            )
        },
    )
    c4 = tc("get_console_output", {"limit": 10})
    traces.append(
        wrap(
            [
                sys(),
                user("Glide ability won't activate. Players press the button in the air but nothing happens."),
                asst(
                    "<think>Glide requires airborne check (FloorMaterial == Air). Let me test in play mode to see the state.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst("<think>I need play mode to test runtime behavior. Switching.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true, "mode": "Play"}'),
                asst("<think>Now in play mode. Let me check the player's airborne state.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true}'),
                asst("<think>Check the output.</think>", [c4]),
                tool_msg(
                    c4["id"],
                    '{"output": ["[Test] Player Y: 5.0", "[Test] FloorMaterial: Enum.Material.Plastic", "[Test] Airborne: false"]}',
                ),
                asst(
                    "The player is on the ground (FloorMaterial=Plastic, not Air). Glide requires the player to be airborne first. The glide validation correctly rejects grounded activation. Players need to jump or be launched before pressing glide."
                ),
            ],
            category="debugging",
            difficulty=3,
        )
    )

    # 6. Memory leak from connections
    _reset_ids()
    c1 = tc("get_console_output", {"limit": 40})
    c2 = tc(
        "run_code",
        {
            "code": (
                "local count = 0\n"
                "for _, obj in game:GetDescendants() do\n"
                '\tif obj:IsA("Script") or obj:IsA("LocalScript") then\n'
                "\t\tcount += 1\n"
                "\tend\n"
                "end\n"
                'print("[Audit] Total scripts:", count)\n'
                'print("[Audit] Descendant count:", #game:GetDescendants())'
            )
        },
    )
    c3 = tc(
        "run_code",
        {
            "code": (
                'local stats = game:GetService("Stats")\n'
                'print("[Memory] InstanceCount:", stats.InstanceCount)\n'
                'print("[Memory] DataModel descendants:", #game:GetDescendants())'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Server memory keeps growing over time. Suspect a memory leak."),
                asst(
                    "<think>Memory leaks in Roblox typically come from uncleared connections, growing tables, or instance accumulation. Let me check.</think>",
                    [c1],
                ),
                tool_msg(
                    c1["id"],
                    '{"output": ["[Warn] High memory usage: 1.2GB", "[GrappleService] Creating new BodyVelocity for Player3", "[GrappleService] Creating new BodyVelocity for Player3", "[GrappleService] Creating new BodyVelocity for Player3"]}',
                ),
                asst(
                    "<think>The GrappleService is creating BodyVelocity instances repeatedly without cleaning up. Classic instance leak. Let me audit total instances.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"success": true}'),
                asst("<think>Let me check the instance count specifically.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true}'),
                asst(
                    "Found the leak: `GrappleService` is creating new BodyVelocity instances on every grapple without destroying previous ones. Fix: use a Trove/maid pattern or pool BodyVelocity instances, calling `:Destroy()` on release. The service should track active movers per player and clean up on disconnect."
                ),
            ],
            category="debugging",
            difficulty=4,
        )
    )

    # 7. Vehicles not spawning
    _reset_ids()
    c1 = tc("get_console_output", {"limit": 20})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local RS = game:GetService("ReplicatedStorage")\n'
                'local vehicles = RS:FindFirstChild("Vehicles")\n'
                "if vehicles then\n"
                '\tprint("[Audit] Vehicles folder found")\n'
                "\tfor _, v in vehicles:GetChildren() do\n"
                '\t\tprint("  ", v.Name, v.ClassName)\n'
                "\tend\n"
                "else\n"
                '\twarn("[Audit] Vehicles folder NOT FOUND in ReplicatedStorage")\n'
                "end"
            )
        },
    )
    c3 = tc(
        "run_code",
        {
            "code": (
                'local folder = Instance.new("Folder")\n'
                'folder.Name = "Vehicles"\n'
                'folder.Parent = game:GetService("ReplicatedStorage")\n'
                'print("[Fix] Created Vehicles folder in ReplicatedStorage")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Players can't spawn vehicles. The spawn button does nothing."),
                asst(
                    "<think>Vehicle spawning likely requires a template in ReplicatedStorage. Let me check console and the vehicles folder.</think>",
                    [c1],
                ),
                tool_msg(
                    c1["id"],
                    '{"output": ["[VehicleService] Init complete", "[VehicleService] ERROR: Vehicles folder not found in ReplicatedStorage"]}',
                ),
                asst("<think>The Vehicles folder is missing. Let me verify and create it.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst("<think>Confirmed missing. Creating the folder.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true}'),
                asst(
                    "The `Vehicles` folder was missing from ReplicatedStorage. VehicleService requires it to find vehicle templates. Created the folder — you'll also need to add vehicle Model templates inside it (e.g., DirtBike, GliderKite)."
                ),
            ],
            category="debugging",
            difficulty=3,
        )
    )

    # 8. CollectionService tags not working
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local CS = game:GetService("CollectionService")\n'
                'local tags = {"GrappleAnchor", "Checkpoint", "BloomCrystal", "WindVolume", "KillBrick"}\n'
                "for _, tag in ipairs(tags) do\n"
                "\tlocal count = #CS:GetTagged(tag)\n"
                '\tprint(string.format("[Tags] %s: %d instances", tag, count))\n'
                "end"
            )
        },
    )
    c2 = tc(
        "run_code",
        {
            "code": (
                "-- Check for typo variants\n"
                'local CS = game:GetService("CollectionService")\n'
                "local allTags: {string} = {}\n"
                "for _, obj in workspace:GetDescendants() do\n"
                "\tfor _, tag in CS:GetTags(obj) do\n"
                "\t\tif not table.find(allTags, tag) then\n"
                "\t\t\ttable.insert(allTags, tag)\n"
                "\t\tend\n"
                "\tend\n"
                "end\n"
                "table.sort(allTags)\n"
                'print("[Tags] All unique tags in workspace:")\n'
                "for _, tag in ipairs(allTags) do\n"
                '\tprint("  ", tag, "#" .. #CS:GetTagged(tag))\n'
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Some tagged objects aren't being detected by services. GrappleAnchors seem to be missing."),
                asst(
                    "<think>Could be a tag typo or the anchors weren't tagged properly. Let me audit all tags.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst("<think>Let me also check for tag typos by listing all unique tags in workspace.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "Audit complete. Check the tag counts and look for typo variants (e.g., `GrappleAnchor` vs `Grapple_Anchor` or `grappleanchor`). The most common issue is inconsistent casing or underscore usage. All builder scripts should use the exact tag strings defined in the config."
                ),
            ],
            category="debugging",
            difficulty=3,
        )
    )

    # 9. DataStore not saving
    _reset_ids()
    c1 = tc("get_console_output", {"limit": 30})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local DSS = game:GetService("DataStoreService")\n'
                "local ok, err = pcall(function()\n"
                '\tlocal store = DSS:GetDataStore("VertigoPlayerData_v7")\n'
                '\tlocal testData = store:GetAsync("test_probe_key")\n'
                '\tprint("[DataStore] Read probe OK, value:", testData)\n'
                "end)\n"
                "if not ok then\n"
                '\twarn("[DataStore] Probe FAILED:", err)\n'
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Player data isn't saving between sessions. They lose all progress."),
                asst(
                    "<think>DataStore save failures. Could be Studio API access, pcall missing, or the save trigger isn't firing. Let me check console for DataStore errors.</think>",
                    [c1],
                ),
                tool_msg(
                    c1["id"],
                    '{"output": ["[DataService] Save failed: 403: Studio access to API services is not allowed", "[DataService] Save failed: 403: Studio access to API services is not allowed"]}',
                ),
                asst(
                    "<think>403 error — DataStore API access is disabled in Studio. Need to enable it in Game Settings > Security > Enable Studio Access to API Services.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "The DataStore is failing with a 403 error because **Studio Access to API Services** is disabled. Fix: Game Settings > Security > Enable Studio Access to API Services. In production (live servers), this setting isn't needed — it only affects Studio testing."
                ),
            ],
            category="debugging",
            difficulty=3,
        )
    )

    # 10. Audio not playing
    _reset_ids()
    c1 = tc("get_console_output", {"limit": 20})
    c2 = tc(
        "run_code",
        {
            "code": (
                "local blocked = {}\n"
                "for _, sound in workspace:GetDescendants() do\n"
                '\tif sound:IsA("Sound") and sound.SoundId ~= "" then\n'
                '\t\tif sound.SoundId:find("1839841655") then\n'
                "\t\t\ttable.insert(blocked, sound:GetFullName())\n"
                "\t\tend\n"
                "\tend\n"
                "end\n"
                'print("[Audit] Sounds using placeholder ID:", #blocked)\n'
                "for _, path in ipairs(blocked) do\n"
                '\tprint("  ", path)\n'
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("No audio is playing in any zone. Everything is silent."),
                asst(
                    "<think>Known issue: placeholder audio IDs (rbxassetid://1839841655) are runtime-blocked. Let me check console and audit Sound instances.</think>",
                    [c1],
                ),
                tool_msg(
                    c1["id"], '{"output": ["[SoundService] Failed to load sound rbxassetid://1839841655: Blocked"]}'
                ),
                asst(
                    "<think>Confirmed — the placeholder asset ID is blocked at runtime. Let me find all sounds using it.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "All zone sounds are using the placeholder asset ID `rbxassetid://1839841655` which is blocked at runtime. Each Sound instance needs a real, uploaded audio asset ID. Until then, affected zones will remain silent."
                ),
            ],
            category="debugging",
            difficulty=2,
        )
    )

    return traces


# ---------------------------------------------------------------------------
# Type 3: Inspect/audit tasks (10 examples)
# ---------------------------------------------------------------------------


def inspect_tasks() -> list[dict]:
    traces = []

    # 1. Count tagged instances
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local CS = game:GetService("CollectionService")\n'
                'local tags = {"GrappleAnchor", "Checkpoint", "BloomCrystal", "WindVolume",\n'
                '\t"KillBrick", "NPC", "SoundRegion", "PressurePlate", "CurrentVolume"}\n'
                'print("[Census] Tagged instance counts:")\n'
                "local total = 0\n"
                "for _, tag in ipairs(tags) do\n"
                "\tlocal count = #CS:GetTagged(tag)\n"
                "\ttotal += count\n"
                '\tprint(string.format("  %-20s %d", tag, count))\n'
                "end\n"
                'print(string.format("  %-20s %d", "TOTAL", total))'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("How many tagged instances are in the game? Give me a census."),
                asst(
                    "<think>I'll query CollectionService for all known gameplay tags and count instances of each.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst(
                    "Ran a full tag census across 9 gameplay tags (GrappleAnchor, Checkpoint, BloomCrystal, WindVolume, KillBrick, NPC, SoundRegion, PressurePlate, CurrentVolume). Check the console output for exact counts per tag."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 2. Check player DataStore
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_code",
        {
            "code": (
                'local DSS = game:GetService("DataStoreService")\n'
                'local store = DSS:GetDataStore("VertigoPlayerData_v7")\n'
                "local userId = 12345678\n"
                "local ok, data = pcall(function()\n"
                '\treturn store:GetAsync("player_" .. tostring(userId))\n'
                "end)\n"
                "if ok and data then\n"
                '\tprint("[DataStore] Profile for user", userId, ":")\n'
                "\tfor k, v in pairs(data) do\n"
                '\t\tprint(string.format("  %s = %s", k, tostring(v)))\n'
                "\tend\n"
                "elseif ok then\n"
                '\tprint("[DataStore] No profile found for user", userId)\n'
                "else\n"
                '\twarn("[DataStore] Read failed:", data)\n'
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Check the DataStore profile for user ID 12345678."),
                asst(
                    "<think>I'll read from the v7 DataStore using the standard key format. Check mode first — DataStore requires API access.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst("<think>Edit mode, good. Reading the player profile.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "Queried the `VertigoPlayerData_v7` DataStore for user 12345678. The profile contents (or lack thereof) are printed to the console. If you see a 403 error, Studio API access needs to be enabled in Game Settings."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 3. Audit physics state
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                "local floating = {}\n"
                "local massless = {}\n"
                "local unanchored = {}\n"
                "\n"
                "for _, part in workspace:GetDescendants() do\n"
                '\tif part:IsA("BasePart") then\n'
                "\t\tif not part.Anchored and part.Position.Y > 10 then\n"
                "\t\t\ttable.insert(floating, {name = part:GetFullName(), y = part.Position.Y})\n"
                "\t\tend\n"
                "\t\tif part.Massless and part.Anchored then\n"
                "\t\t\ttable.insert(massless, part:GetFullName())\n"
                "\t\tend\n"
                '\t\tif not part.Anchored and not part.Parent:IsA("Model") then\n'
                "\t\t\ttable.insert(unanchored, part:GetFullName())\n"
                "\t\tend\n"
                "\tend\n"
                "end\n"
                "\n"
                'print("[Physics] Floating unanchored parts:", #floating)\n'
                'print("[Physics] Massless+anchored (suspicious):", #massless)\n'
                'print("[Physics] Unanchored loose parts:", #unanchored)\n'
                "for i, f in ipairs(floating) do\n"
                '\tif i <= 5 then print("  Float:", f.name, "Y:", f.y) end\n'
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Run a physics audit. Any parts that shouldn't be unanchored?"),
                asst(
                    "<think>I'll scan all BaseParts for physics red flags: unanchored parts floating above ground, massless+anchored combos, and loose parts outside models.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst(
                    "Physics audit complete. Checked for:\n1. **Floating unanchored parts** (above Y=10, will fall)\n2. **Massless + anchored** (contradictory, likely a mistake)\n3. **Loose unanchored parts** (outside Models, will drift)\n\nResults are in the console. Fix any floating unanchored parts by setting `Anchored = true`."
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    # 4. Verify Rojo sync status
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local RS = game:GetService("ReplicatedStorage")\n'
                'local shared = RS:FindFirstChild("Shared")\n'
                'print("[Sync] ReplicatedStorage.Shared exists:", shared ~= nil)\n'
                "if shared then\n"
                '\tprint("[Sync] Shared children:")\n'
                "\tfor _, child in shared:GetChildren() do\n"
                '\t\tprint("  ", child.Name, child.ClassName)\n'
                "\tend\n"
                "end\n"
                "\n"
                'local SSS = game:GetService("ServerScriptService")\n'
                'print("[Sync] ServerScriptService children:", #SSS:GetChildren())\n'
                "for _, child in SSS:GetChildren() do\n"
                '\tprint("  ", child.Name, child.ClassName)\n'
                "end"
            )
        },
    )
    c2 = tc(
        "run_code",
        {
            "code": (
                "-- Check for Rojo marker\n"
                'local marker = game:GetService("ServerStorage"):FindFirstChild("__Rojo")\n'
                '\tor game:GetService("ServerStorage"):FindFirstChild("__VertigoSync")\n'
                "if marker then\n"
                '\tprint("[Sync] Sync marker found:", marker:GetFullName())\n'
                "else\n"
                '\twarn("[Sync] No sync marker found — source may be stale")\n'
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Is the source code synced properly? Verify the sync status."),
                asst(
                    "<think>I'll check if the expected folder structure exists and look for sync markers.</think>", [c1]
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst("<think>Let me also check for the sync tool marker in ServerStorage.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true}'),
                asst(
                    "Sync verification complete. Checked:\n1. ReplicatedStorage.Shared folder structure and children\n2. ServerScriptService contents\n3. Sync tool marker (__Rojo or __VertigoSync)\n\nIf the marker is missing, source may be stale. Reconnect the sync tool in Studio."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 5. List all RemoteEvents
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                "local remotes = {events = {}, functions = {}}\n"
                'for _, obj in game:GetService("ReplicatedStorage"):GetDescendants() do\n'
                '\tif obj:IsA("RemoteEvent") then\n'
                "\t\ttable.insert(remotes.events, obj.Name)\n"
                '\telseif obj:IsA("RemoteFunction") then\n'
                "\t\ttable.insert(remotes.functions, obj.Name)\n"
                "\tend\n"
                "end\n"
                'print("[Remotes] RemoteEvents:", #remotes.events)\n'
                'for _, name in ipairs(remotes.events) do print("  Event:", name) end\n'
                'print("[Remotes] RemoteFunctions:", #remotes.functions)\n'
                'for _, name in ipairs(remotes.functions) do print("  Func:", name) end'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("List all RemoteEvents and RemoteFunctions in the game."),
                asst("<think>I'll scan ReplicatedStorage for all Remote instances and categorize them.</think>", [c1]),
                tool_msg(c1["id"], '{"success": true}'),
                asst(
                    "Listed all remotes in ReplicatedStorage. Check the console for the full inventory of RemoteEvents and RemoteFunctions."
                ),
            ],
            category="trajectory",
            difficulty=1,
        )
    )

    # 6. Workspace size audit
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                "local classCounts: {[string]: number} = {}\n"
                "local totalParts = 0\n"
                "for _, obj in workspace:GetDescendants() do\n"
                "\tlocal cls = obj.ClassName\n"
                "\tclassCounts[cls] = (classCounts[cls] or 0) + 1\n"
                '\tif obj:IsA("BasePart") then totalParts += 1 end\n'
                "end\n"
                'print("[Audit] Total descendants:", #workspace:GetDescendants())\n'
                'print("[Audit] Total BaseParts:", totalParts)\n'
                'print("[Audit] Top classes:")\n'
                "local sorted = {}\n"
                "for cls, count in pairs(classCounts) do\n"
                "\ttable.insert(sorted, {cls = cls, count = count})\n"
                "end\n"
                "table.sort(sorted, function(a, b) return a.count > b.count end)\n"
                "for i, entry in ipairs(sorted) do\n"
                "\tif i <= 15 then\n"
                '\t\tprint(string.format("  %-25s %d", entry.cls, entry.count))\n'
                "\tend\n"
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("How big is the workspace? Give me a class breakdown."),
                asst(
                    "<think>I'll count all descendants in workspace and group by ClassName for a size audit.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst(
                    "Workspace audit complete. The console shows total descendants, total BaseParts, and the top 15 instance classes by count. Use this to identify if any class is over-represented (potential optimization target)."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 7. Check ability tuning values
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                "local tuning = {\n"
                '\t"GrappleMaxRange", "GrappleReelSpeed", "GrappleLaunchForce",\n'
                '\t"GrappleCooldown", "GrappleMaxAngle",\n'
                '\t"GlideLift", "GlideMaxSpeed", "GlideDrag",\n'
                '\t"SlideSpeed", "SlideFriction",\n'
                '\t"AirDashForce", "AirDashCooldown",\n'
                '\t"WallRunDuration", "WallRunSpeed",\n'
                "}\n"
                'print("[Tuning] Workspace ability attributes:")\n'
                "for _, key in ipairs(tuning) do\n"
                "\tlocal val = workspace:GetAttribute(key)\n"
                '\tprint(string.format("  %-25s %s", key, val ~= nil and tostring(val) or "NOT SET"))\n'
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("What are the current ability tuning values?"),
                asst(
                    "<think>Ability tuning is stored as workspace attributes. I'll read all known tuning keys.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst(
                    "Printed all ability tuning attributes from workspace. Keys marked `NOT SET` need values configured. Expected values: GrappleMaxRange=180, GlideLift=0.55, etc."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 8. Check zone builder coverage
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                "local builders = {}\n"
                "for _, obj in workspace:GetDescendants() do\n"
                '\tif obj:IsA("Model") or obj:IsA("Folder") then\n'
                '\t\tif obj.Name:find("Builder") or obj.Name:find("Zone") then\n'
                "\t\t\tlocal partCount = 0\n"
                "\t\t\tfor _, child in obj:GetDescendants() do\n"
                '\t\t\t\tif child:IsA("BasePart") then partCount += 1 end\n'
                "\t\t\tend\n"
                "\t\t\ttable.insert(builders, {name = obj.Name, parts = partCount})\n"
                "\t\tend\n"
                "\tend\n"
                "end\n"
                "table.sort(builders, function(a, b) return a.parts > b.parts end)\n"
                'print("[Builders] Zone builder census:")\n'
                "for _, b in ipairs(builders) do\n"
                '\tprint(string.format("  %-30s %d parts", b.name, b.parts))\n'
                "end\n"
                'print("[Builders] Total builders:", #builders)'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Which zone builders have been built and how many parts does each have?"),
                asst(
                    "<think>I'll scan workspace for Models/Folders with Builder or Zone in the name and count their BasePart descendants.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst(
                    "Zone builder census complete. The console shows each builder sorted by part count (heaviest first). This helps identify which zones need optimization or which are empty."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 9. Audit lighting settings
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local Lighting = game:GetService("Lighting")\n'
                'print("[Lighting] Current settings:")\n'
                'print("  Ambient:", tostring(Lighting.Ambient))\n'
                'print("  OutdoorAmbient:", tostring(Lighting.OutdoorAmbient))\n'
                'print("  Brightness:", Lighting.Brightness)\n'
                'print("  ClockTime:", Lighting.ClockTime)\n'
                'print("  FogEnd:", Lighting.FogEnd)\n'
                'print("  FogStart:", Lighting.FogStart)\n'
                'print("  FogColor:", tostring(Lighting.FogColor))\n'
                'print("  Technology:", tostring(Lighting.Technology))\n'
                'print("  EnvironmentDiffuseScale:", Lighting.EnvironmentDiffuseScale)\n'
                'print("  EnvironmentSpecularScale:", Lighting.EnvironmentSpecularScale)\n'
                "\n"
                'print("[Lighting] Post-processing:")\n'
                "for _, child in Lighting:GetChildren() do\n"
                '\tprint("  ", child.ClassName, child.Name)\n'
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("What are the current lighting settings?"),
                asst(
                    "<think>I'll read all relevant Lighting service properties and list post-processing effects.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst(
                    "Lighting audit printed to console: ambient colors, brightness, clock time, fog settings, technology mode, environment scales, and all post-processing effects (Bloom, ColorCorrection, etc.)."
                ),
            ],
            category="trajectory",
            difficulty=1,
        )
    )

    # 10. Check for streaming-unsafe instances
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                "local issues = {}\n"
                "for _, obj in workspace:GetDescendants() do\n"
                "\t-- Scripts referencing workspace children by path might break with streaming\n"
                '\tif obj:IsA("Script") or obj:IsA("LocalScript") then\n'
                "\t\tlocal src = obj.Source\n"
                '\t\tif src:find("workspace%.") and src:find("WaitForChild") == nil then\n'
                "\t\t\ttable.insert(issues, {\n"
                "\t\t\t\tscript = obj:GetFullName(),\n"
                '\t\t\t\tissue = "Direct workspace path without WaitForChild"\n'
                "\t\t\t})\n"
                "\t\tend\n"
                "\tend\n"
                "end\n"
                'print("[Streaming] Potentially unsafe scripts:", #issues)\n'
                "for _, iss in ipairs(issues) do\n"
                '\tprint("  ", iss.script, "->", iss.issue)\n'
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Audit for instance streaming safety. Any scripts that might break with streaming enabled?"),
                asst(
                    "<think>Scripts that reference workspace children directly without WaitForChild will fail when streaming is enabled because children may not exist yet. Let me scan for this pattern.</think>",
                    [c1],
                ),
                tool_msg(c1["id"], '{"success": true}'),
                asst(
                    "Streaming safety audit complete. Any scripts accessing `workspace.SomePart` without `:WaitForChild()` will break when StreamingEnabled is turned on, because the instance may not have streamed in yet. Fix: use `:WaitForChild()` or listen for `CollectionService:GetInstanceAddedSignal()`."
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    return traces


# ---------------------------------------------------------------------------
# Type 4: Failure-and-recovery traces (10 examples)
# ---------------------------------------------------------------------------


def failure_recovery_tasks() -> list[dict]:
    traces = []

    # 1. Run code in wrong mode → switch → retry
    _reset_ids()
    c1 = tc(
        "run_script_in_play_mode",
        {
            "code": (
                'print("Testing player state...")\n'
                'local player = game:GetService("Players").LocalPlayer\n'
                'print("Player:", player.Name)'
            )
        },
    )
    c2 = tc("get_studio_mode", {})
    c3 = tc("start_stop_play", {})
    c4 = tc(
        "run_script_in_play_mode",
        {
            "code": (
                'print("Testing player state...")\n'
                'local player = game:GetService("Players").LocalPlayer\n'
                "if player then\n"
                '\tprint("[OK] Player:", player.Name)\n'
                "\tlocal char = player.Character\n"
                "\tif char then\n"
                '\t\tprint("[OK] Character exists at Y:", char.PrimaryPart.Position.Y)\n'
                "\tend\n"
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Check the player's current position in the game."),
                asst("<think>I need to run code in play mode to access the local player. Let me try.</think>", [c1]),
                tool_msg(c1["id"], '{"error": "Cannot run in play mode: Studio is in Edit mode"}'),
                asst(
                    "<think>Error — Studio is in Edit mode. I need to switch to Play mode first. Let me confirm the mode and toggle.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"mode": "Edit"}'),
                asst("<think>Confirmed Edit mode. Switching to Play mode.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true, "mode": "Play"}'),
                asst("<think>Now in Play mode. Retrying the player state check.</think>", [c4]),
                tool_msg(c4["id"], '{"success": true}'),
                asst(
                    "Initially failed because Studio was in Edit mode. Switched to Play mode and successfully checked the player position."
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    # 2. Syntax error → read console → fix
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local part = Instance.new("Part")\n'
                "part.Size = Vector3.new(5, 5, 5\n"  # Missing closing paren
                "part.Parent = workspace"
            )
        },
    )
    c2 = tc("get_console_output", {"limit": 5})
    c3 = tc(
        "run_code",
        {
            "code": (
                'local part = Instance.new("Part")\n'
                "part.Size = Vector3.new(5, 5, 5)\n"
                "part.Anchored = true\n"
                "part.Parent = workspace\n"
                'print("[OK] Part created")'
            )
        },
    )
    c4 = tc("get_console_output", {"limit": 5})
    traces.append(
        wrap(
            [
                sys(),
                user("Create a 5x5x5 part in workspace."),
                asst("<think>Simple part creation. I'll run the code directly.</think>", [c1]),
                tool_msg(c1["id"], "{\"error\": \"Syntax error: Expected ')' to close '(' at line 2\"}"),
                asst(
                    "<think>Syntax error — I'm missing a closing parenthesis on line 2. Let me fix and retry.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], "{\"output\": [\"Syntax error: Expected ')' to close '(' at line 2\"]}"),
                asst(
                    "<think>Confirmed the syntax error. Fixing the missing parenthesis and also adding Anchored=true so it doesn't fall.</think>",
                    [c3],
                ),
                tool_msg(c3["id"], '{"success": true}'),
                asst("<think>Verify success.</think>", [c4]),
                tool_msg(c4["id"], '{"output": ["[OK] Part created"]}'),
                asst(
                    "Fixed a syntax error (missing closing parenthesis) and created the 5x5x5 anchored part in workspace."
                ),
            ],
            category="trajectory",
            difficulty=2,
        )
    )

    # 3. Wrong API usage → error → correct approach
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local CS = game:GetService("CollectionService")\n'
                'local anchors = CS:GetTagged("GrappleAnchor")\n'
                "for _, anchor in anchors do\n"
                '\tanchor:SetAttribute("Enabled", true)\n'
                "\tanchor.Color = Color3.new(0, 1, 0)\n"  # Color3 not BrickColor
                "end\n"
                'print("[OK] Updated", #anchors, "anchors")'
            )
        },
    )
    c2 = tc("get_console_output", {"limit": 10})
    c3 = tc(
        "run_code",
        {
            "code": (
                'local CS = game:GetService("CollectionService")\n'
                'local anchors = CS:GetTagged("GrappleAnchor")\n'
                "for _, anchor in ipairs(anchors) do\n"
                '\tanchor:SetAttribute("Enabled", true)\n'
                "\tanchor.Color = Color3.fromRGB(0, 255, 0)\n"
                "end\n"
                'print("[OK] Updated", #anchors, "anchors")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Enable all grapple anchors and make them green."),
                asst(
                    "<think>I'll iterate over tagged anchors and set their Enabled attribute and color.</think>", [c1]
                ),
                tool_msg(
                    c1["id"], '{"error": "invalid argument #1 to \'Color3.new\' (expected number 0-1 range hint)"}'
                ),
                asst(
                    "<think>The color values are correct (0-1 range) but I should use Color3.fromRGB for clarity. Also, I used a generalized `for` without `ipairs`. Let me fix both issues.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"output": ["invalid argument #1..."]}'),
                asst("<think>Let me use Color3.fromRGB(0, 255, 0) and ipairs for the iteration.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true}'),
                asst(
                    "Fixed the iteration (added `ipairs`) and color assignment (switched to `Color3.fromRGB`). All grapple anchors are now enabled and green."
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    # 4. Script creates instance in wrong location → move it
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local script = Instance.new("Script")\n'
                'script.Name = "ZoneManager"\n'
                'script.Source = "--!strict\\nreturn {}"\n'
                'script.Parent = game:GetService("ReplicatedStorage")\n'
                'print("[OK] ZoneManager created")'
            )
        },
    )
    c2 = tc("get_console_output", {"limit": 5})
    c3 = tc(
        "run_code",
        {
            "code": (
                "-- Scripts in ReplicatedStorage don't run on the server!\n"
                'local RS = game:GetService("ReplicatedStorage")\n'
                'local script = RS:FindFirstChild("ZoneManager")\n'
                "if script then\n"
                '\tscript.Parent = game:GetService("ServerScriptService")\n'
                '\tprint("[Fix] Moved ZoneManager to ServerScriptService")\n'
                "end"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Create a ZoneManager server script."),
                asst("<think>I'll create the script. Let me put it in the right service.</think>", [c1]),
                tool_msg(c1["id"], '{"success": true}'),
                asst(
                    "<think>Wait — I accidentally put a Script in ReplicatedStorage. Scripts there won't run on the server. I need to move it to ServerScriptService.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"output": ["[OK] ZoneManager created"]}'),
                asst("<think>I need to fix the parent. Moving to ServerScriptService.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true}'),
                asst(
                    "Caught a mistake: initially placed the Script in ReplicatedStorage where server scripts don't execute. Moved it to ServerScriptService where it will run properly."
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    # 5. Nil reference during play test → add guard
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc(
        "run_script_in_play_mode",
        {
            "code": (
                'local player = game:GetService("Players").LocalPlayer\n'
                "local char = player.Character\n"
                "local hrp = char.HumanoidRootPart\n"
                'print("Position:", hrp.Position)'
            )
        },
    )
    c3 = tc("get_console_output", {"limit": 5})
    c4 = tc(
        "run_script_in_play_mode",
        {
            "code": (
                'local player = game:GetService("Players").LocalPlayer\n'
                "if not player then\n"
                '\twarn("[Test] No LocalPlayer")\n'
                "\treturn\n"
                "end\n"
                "local char = player.Character or player.CharacterAdded:Wait()\n"
                'local hrp = char:WaitForChild("HumanoidRootPart") :: BasePart\n'
                'print("[OK] Position:", hrp.Position)'
            )
        },
    )
    c5 = tc("get_console_output", {"limit": 5})
    traces.append(
        wrap(
            [
                sys(),
                user("Get the player's position during play mode."),
                asst("<think>Need play mode for this. Let me check mode.</think>", [c1]),
                tool_msg(c1["id"], '{"mode": "Play"}'),
                asst("<think>Already in Play mode. Reading player position.</think>", [c2]),
                tool_msg(c2["id"], '{"error": "attempt to index nil with \'HumanoidRootPart\'"}'),
                asst(
                    "<think>Nil error — the character hasn't loaded yet. I need to use CharacterAdded:Wait() and WaitForChild for safety.</think>",
                    [c3],
                ),
                tool_msg(c3["id"], '{"output": ["attempt to index nil with \'HumanoidRootPart\'"]}'),
                asst("<think>Adding proper nil guards with CharacterAdded:Wait() and WaitForChild.</think>", [c4]),
                tool_msg(c4["id"], '{"success": true}'),
                asst("<think>Verify it worked.</think>", [c5]),
                tool_msg(c5["id"], '{"output": ["[OK] Position: 0, 5, 0"]}'),
                asst(
                    'First attempt hit a nil reference because the character hadn\'t loaded yet. Fixed by using `CharacterAdded:Wait()` and `:WaitForChild("HumanoidRootPart")` for safe access.'
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    # 6. Code runs but produces wrong result → investigate → fix
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local CS = game:GetService("CollectionService")\n'
                'local anchors = CS:GetTagged("GrappleAnchor")\n'
                "local avgY = 0\n"
                "for _, a in ipairs(anchors) do\n"
                "\tavgY += a.Position.Y\n"
                "end\n"
                "avgY = avgY / #anchors\n"
                'print("[Result] Average GrappleAnchor Y:", avgY)'
            )
        },
    )
    c2 = tc("get_console_output", {"limit": 5})
    c3 = tc(
        "run_code",
        {
            "code": (
                'local CS = game:GetService("CollectionService")\n'
                'local anchors = CS:GetTagged("GrappleAnchor")\n'
                "if #anchors == 0 then\n"
                '\twarn("[Result] No GrappleAnchors found")\n'
                "\treturn\n"
                "end\n"
                "local avgY: number = 0\n"
                "local minY: number = math.huge\n"
                "local maxY: number = -math.huge\n"
                "for _, a in ipairs(anchors) do\n"
                "\tlocal y: number = a.Position.Y\n"
                "\tavgY += y\n"
                "\tif y < minY then minY = y end\n"
                "\tif y > maxY then maxY = y end\n"
                "end\n"
                "avgY = avgY / #anchors\n"
                'print(string.format("[Result] Anchors: %d, AvgY: %.1f, MinY: %.1f, MaxY: %.1f",\n'
                "\t#anchors, avgY, minY, maxY))"
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("What's the average height of all grapple anchors?"),
                asst("<think>I'll calculate the average Y position of all tagged GrappleAnchors.</think>", [c1]),
                tool_msg(c1["id"], '{"error": "attempt to perform arithmetic (div) on number and number (nan)"}'),
                asst(
                    "<think>NaN error — likely dividing by zero because there are no anchors. I need to add a zero-count guard. Let me also add min/max for better context.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"output": ["attempt to perform arithmetic (div) on number and number (nan)"]}'),
                asst("<think>Adding empty-check guard and computing min/max as well.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true}'),
                asst(
                    "First attempt crashed with NaN because the anchor count was zero (division by zero). Fixed by adding an empty-array guard and also computing min/max Y for fuller context."
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    # 7. Tries to modify read-only property → catches error → uses correct API
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local Lighting = game:GetService("Lighting")\n'
                "Lighting.Technology = Enum.Technology.Future\n"
                'print("[OK] Set lighting technology")'
            )
        },
    )
    c2 = tc("get_console_output", {"limit": 5})
    c3 = tc(
        "run_code",
        {
            "code": (
                "-- Technology is read-only at runtime, must be set in Studio properties\n"
                "-- Instead, adjust what we can control:\n"
                'local Lighting = game:GetService("Lighting")\n'
                "Lighting.Brightness = 2\n"
                "Lighting.EnvironmentDiffuseScale = 0.8\n"
                "Lighting.EnvironmentSpecularScale = 0.5\n"
                "Lighting.Ambient = Color3.fromRGB(40, 40, 60)\n"
                "Lighting.OutdoorAmbient = Color3.fromRGB(80, 80, 100)\n"
                'print("[OK] Lighting properties updated (Technology must be set via Studio UI)")'
            )
        },
    )
    traces.append(
        wrap(
            [
                sys(),
                user("Switch the lighting to Future technology and make it look more atmospheric."),
                asst("<think>I'll set the Lighting.Technology to Future and adjust atmosphere settings.</think>", [c1]),
                tool_msg(c1["id"], '{"error": "Unable to assign property Technology. It is read-only."}'),
                asst(
                    "<think>Technology is a read-only property that can only be changed through the Studio properties panel, not via code. I can still adjust brightness, ambient, and environment scales.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"output": ["Unable to assign property Technology. It is read-only."]}'),
                asst("<think>I'll adjust the properties I can control via code.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true}'),
                asst(
                    "The `Technology` property is read-only and must be changed through Studio's Properties panel. I adjusted the controllable lighting properties instead: Brightness=2, EnvironmentDiffuseScale=0.8, EnvironmentSpecularScale=0.5, and darker ambient colors for atmosphere."
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    # 8. Play mode test fails → discover state issue → fix
    _reset_ids()
    c1 = tc("get_studio_mode", {})
    c2 = tc("start_stop_play", {})
    c3 = tc(
        "run_script_in_play_mode",
        {
            "code": (
                'local RS = game:GetService("ReplicatedStorage")\n'
                'local remote = RS:FindFirstChild("RequestUseAbility") :: RemoteEvent\n'
                'remote:FireServer("grapple", {targetPosition = Vector3.new(10, 50, 0)})\n'
                'print("[Test] Fired grapple request")'
            )
        },
    )
    c4 = tc("get_console_output", {"limit": 10})
    c5 = tc(
        "run_script_in_play_mode",
        {
            "code": (
                "-- Check if the remote actually exists\n"
                'local RS = game:GetService("ReplicatedStorage")\n'
                'print("[Debug] ReplicatedStorage children:")\n'
                "for _, child in RS:GetChildren() do\n"
                '\tprint("  ", child.Name, child.ClassName)\n'
                "end\n"
                'local remote = RS:FindFirstChild("RequestUseAbility")\n'
                'print("[Debug] RequestUseAbility exists:", remote ~= nil)'
            )
        },
    )
    c6 = tc("get_console_output", {"limit": 10})
    traces.append(
        wrap(
            [
                sys(),
                user("Test the grapple ability in play mode — fire the remote and check if it works."),
                asst(
                    "<think>I need to be in play mode to fire remotes. Let me check and switch if needed.</think>", [c1]
                ),
                tool_msg(c1["id"], '{"mode": "Edit"}'),
                asst("<think>Switching to play mode.</think>", [c2]),
                tool_msg(c2["id"], '{"success": true, "mode": "Play"}'),
                asst("<think>Now in play mode. Firing the grapple remote.</think>", [c3]),
                tool_msg(c3["id"], '{"error": "attempt to index nil with \'FireServer\'"}'),
                asst(
                    "<think>The remote is nil — it doesn't exist. Let me check what's actually in ReplicatedStorage.</think>",
                    [c4],
                ),
                tool_msg(c4["id"], '{"output": ["attempt to index nil with \'FireServer\'"]}'),
                asst("<think>I need to debug what remotes exist.</think>", [c5]),
                tool_msg(c5["id"], '{"success": true}'),
                asst("<think>Check the output to see what's there.</think>", [c6]),
                tool_msg(
                    c6["id"],
                    '{"output": ["[Debug] ReplicatedStorage children:", "  Shared Folder", "  Packages Folder", "  Net Folder", "[Debug] RequestUseAbility exists: false"]}',
                ),
                asst(
                    'The `RequestUseAbility` RemoteEvent doesn\'t exist in ReplicatedStorage. The remotes are inside the `Net` folder (ReplicatedStorage.Net). The correct path would be `RS.Net:FindFirstChild("RequestUseAbility")` or the remote might have a different name like `RequestUseAbility` inside Remotes.luau.'
                ),
            ],
            category="trajectory",
            difficulty=4,
        )
    )

    # 9. Incorrect instance path → not found → search and correct
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local config = game:GetService("ReplicatedStorage").Config.Abilities\n'
                'print("[OK] Found config:", config:GetFullName())'
            )
        },
    )
    c2 = tc("get_console_output", {"limit": 5})
    c3 = tc(
        "run_code",
        {
            "code": (
                "-- Search for the correct path\n"
                'local RS = game:GetService("ReplicatedStorage")\n'
                "local function findModule(parent, name, depth)\n"
                "\tif depth > 5 then return nil end\n"
                "\tfor _, child in parent:GetChildren() do\n"
                "\t\tif child.Name == name then return child end\n"
                "\t\tlocal found = findModule(child, name, depth + 1)\n"
                "\t\tif found then return found end\n"
                "\tend\n"
                "\treturn nil\n"
                "end\n"
                'local abilities = findModule(RS, "Abilities", 0)\n'
                "if abilities then\n"
                '\tprint("[Found] Abilities at:", abilities:GetFullName())\n'
                "else\n"
                '\twarn("[NotFound] Abilities module not in ReplicatedStorage")\n'
                "end"
            )
        },
    )
    c4 = tc("get_console_output", {"limit": 5})
    traces.append(
        wrap(
            [
                sys(),
                user("Read the Abilities config module."),
                asst("<think>The config should be at ReplicatedStorage.Config.Abilities.</think>", [c1]),
                tool_msg(c1["id"], '{"error": "Config is not a valid member of ReplicatedStorage"}'),
                asst(
                    "<think>Wrong path. The Config folder might be nested under Shared. Let me search for it.</think>",
                    [c2],
                ),
                tool_msg(c2["id"], '{"output": ["Config is not a valid member of ReplicatedStorage"]}'),
                asst("<think>I'll do a recursive search for the Abilities module.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true}'),
                asst("<think>Check where it was found.</think>", [c4]),
                tool_msg(c4["id"], '{"output": ["[Found] Abilities at: ReplicatedStorage.Shared.Config.Abilities"]}'),
                asst(
                    "The correct path is `ReplicatedStorage.Shared.Config.Abilities` (not `ReplicatedStorage.Config.Abilities`). The Config folder is nested under the Shared folder, following the Vertigo project structure."
                ),
            ],
            category="trajectory",
            difficulty=3,
        )
    )

    # 10. Bulk operation partially fails → detect → complete remaining
    _reset_ids()
    c1 = tc(
        "run_code",
        {
            "code": (
                'local CS = game:GetService("CollectionService")\n'
                'local anchors = CS:GetTagged("GrappleAnchor")\n'
                "local updated = 0\n"
                "for _, anchor in ipairs(anchors) do\n"
                "\tanchor.Material = Enum.Material.Neon\n"
                '\tanchor.BrickColor = BrickColor.new("Lime green")\n'
                "\tanchor.Size = Vector3.new(3, 3, 3)\n"
                "\tupdated += 1\n"
                "end\n"
                'print("[OK] Updated", updated, "of", #anchors, "anchors")'
            )
        },
    )
    c2 = tc("get_console_output", {"limit": 10})
    c3 = tc(
        "run_code",
        {
            "code": (
                "-- Some anchors were in locked Models. Retry with pcall per-instance.\n"
                'local CS = game:GetService("CollectionService")\n'
                'local anchors = CS:GetTagged("GrappleAnchor")\n'
                "local success, failed = 0, 0\n"
                "for _, anchor in ipairs(anchors) do\n"
                "\tlocal ok, err = pcall(function()\n"
                "\t\tanchor.Material = Enum.Material.Neon\n"
                '\t\tanchor.BrickColor = BrickColor.new("Lime green")\n'
                "\t\tanchor.Size = Vector3.new(3, 3, 3)\n"
                "\tend)\n"
                "\tif ok then\n"
                "\t\tsuccess += 1\n"
                "\telse\n"
                "\t\tfailed += 1\n"
                '\t\twarn("[Skip]", anchor:GetFullName(), err)\n'
                "\tend\n"
                "end\n"
                'print(string.format("[OK] Updated %d anchors, %d failed (locked)", success, failed))'
            )
        },
    )
    c4 = tc("get_console_output", {"limit": 10})
    traces.append(
        wrap(
            [
                sys(),
                user("Update all grapple anchors to neon green 3x3x3 balls."),
                asst(
                    "<think>I'll iterate through all tagged GrappleAnchors and update their visual properties.</think>",
                    [c1],
                ),
                tool_msg(
                    c1["id"],
                    '{"error": "The current identity (2) cannot set property of locked instance Workspace.LockedZone.GrappleAnchor_05"}',
                ),
                asst(
                    "<think>Some anchors are inside locked Models. I need to wrap each update in pcall to handle locked instances gracefully.</think>",
                    [c2],
                ),
                tool_msg(
                    c2["id"], '{"output": ["The current identity (2) cannot set property of locked instance..."]}'
                ),
                asst("<think>Retrying with per-instance pcall to skip locked ones.</think>", [c3]),
                tool_msg(c3["id"], '{"success": true}'),
                asst("<think>Check results.</think>", [c4]),
                tool_msg(c4["id"], '{"output": ["[OK] Updated 18 anchors, 3 failed (locked)"]}'),
                asst(
                    "Updated 18 of 21 grapple anchors to neon green 3x3x3 balls. 3 anchors are inside locked Models and couldn't be modified — you'll need to unlock those Models in Studio to update them."
                ),
            ],
            category="trajectory",
            difficulty=4,
        )
    )

    return traces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    all_traces = []
    all_traces.extend(build_tasks())
    all_traces.extend(debug_tasks())
    all_traces.extend(inspect_tasks())
    all_traces.extend(failure_recovery_tasks())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for trace in all_traces:
            f.write(json.dumps(trace) + "\n")

    print(f"Wrote {len(all_traces)} trajectories to {OUTPUT}")
    cats = {}
    for t in all_traces:
        c = t["category"]
        cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
