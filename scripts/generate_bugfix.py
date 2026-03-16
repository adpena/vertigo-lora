#!/usr/bin/env python3
from __future__ import annotations

"""
Generate bug injection/fix training pairs for Roblox Luau.

Each pair shows BUGGY code and FIXED code with reasoning explanation.
Covers 15 bug categories common in Roblox development.

Output: data/raw/bugfix_pairs.jsonl
"""

import json
import textwrap
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "bugfix_pairs.jsonl"

SYSTEM_PROMPT = (
    "You are an expert Roblox game developer specializing in Luau for the Vertigo experience. "
    "Vertigo is a physics-driven exploration game with traversal mechanics (grapple, glide, wallrun, swim), "
    "procedurally generated zones, and a service/controller architecture with Init/Start lifecycle. "
    "You write production-grade Luau code following these conventions:\n"
    "- --!strict mode with full type annotations\n"
    "- @native on hot-path functions (Heartbeat, RenderStepped callbacks)\n"
    "- SIMD vector math (vector.* API), math.lerp instead of manual interpolation\n"
    "- table.create(n) for pre-sized arrays, table.freeze() for constants\n"
    "- No closures in loops, no Instance.new() in Heartbeat\n"
    "- Server-authoritative validation on all client inputs\n"
    "- Data-driven config (tuning via config modules, not hardcoded values)\n"
    "- Modern APIs: task.wait(), game:GetService(), task.spawn()\n"
    "When explaining code, include reasoning about WHY each design choice was made."
)


# ---------------------------------------------------------------------------
# Bug/fix pair definitions
# ---------------------------------------------------------------------------

BUG_FIX_PAIRS: list[dict] = [
    # -----------------------------------------------------------------------
    # 1. Nil reference — accessing .Character before load
    # -----------------------------------------------------------------------
    {
        "category_tag": "nil_reference",
        "difficulty": 2,
        "user": (
            "I have this code but it errors with 'attempt to index nil with FindFirstChild'. "
            "Can you fix it?\n\n"
            "```lua\n"
            'local Players = game:GetService("Players")\n\n'
            "Players.PlayerAdded:Connect(function(player)\n"
            '\tlocal humanoid = player.Character:FindFirstChild("Humanoid")\n'
            "\thumanoid.Health = 100\n"
            "end)\n"
            "```"
        ),
        "thinking": (
            "The bug is that `player.Character` can be nil when PlayerAdded fires — "
            "the character hasn't loaded yet. We need to wait for it with "
            "CharacterAdded or player.CharacterAdded:Wait(). Also, FindFirstChild "
            "can return nil so we should guard that too."
        ),
        "explanation": (
            "When `PlayerAdded` fires, the player's `Character` hasn't spawned yet — "
            "it's `nil`. You need to wait for the character using `CharacterAdded`. "
            "Additionally, you should nil-check the result of `FindFirstChild` since "
            "the Humanoid could theoretically be missing."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Players = game:GetService("Players")

            local function onCharacterAdded(character: Model)
            \tlocal humanoid = character:FindFirstChildOfClass("Humanoid")
            \tif humanoid then
            \t\thumanoid.Health = 100
            \tend
            end

            Players.PlayerAdded:Connect(function(player)
            \tplayer.CharacterAdded:Connect(onCharacterAdded)
            \tif player.Character then
            \t\tonCharacterAdded(player.Character)
            \tend
            end)
        """),
    },
    {
        "category_tag": "nil_reference",
        "difficulty": 2,
        "user": (
            "I have this code but it throws 'attempt to index nil' randomly. "
            "Can you fix it?\n\n"
            "```lua\n"
            "local function getPlayerSpeed(player)\n"
            "\tlocal humanoid = player.Character.Humanoid\n"
            "\treturn humanoid.WalkSpeed\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "player.Character can be nil if the player hasn't spawned or is respawning. "
            "Chaining .Character.Humanoid without nil checks causes the error. Need to "
            "guard both Character and Humanoid."
        ),
        "explanation": (
            "`player.Character` is `nil` during respawn or before the first spawn. "
            "You must nil-check the character and the humanoid before accessing properties."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local function getPlayerSpeed(player: Player): number?
            \tlocal character = player.Character
            \tif not character then
            \t\treturn nil
            \tend
            \tlocal humanoid = character:FindFirstChildOfClass("Humanoid")
            \tif not humanoid then
            \t\treturn nil
            \tend
            \treturn humanoid.WalkSpeed
            end
        """),
    },
    {
        "category_tag": "nil_reference",
        "difficulty": 3,
        "user": (
            "My damage script errors intermittently. Can you fix it?\n\n"
            "```lua\n"
            "local function onTouched(hit)\n"
            "\tlocal character = hit.Parent\n"
            '\tlocal humanoid = character:FindFirstChild("Humanoid")\n'
            "\thumanoid:TakeDamage(25)\n"
            "end\n\n"
            "damagePart.Touched:Connect(onTouched)\n"
            "```"
        ),
        "thinking": (
            "Two problems: hit.Parent might not be a character (could be any part), "
            "and FindFirstChild can return nil. The code assumes every touching part "
            "belongs to a character with a Humanoid."
        ),
        "explanation": (
            "Not everything that touches the part is a character. `hit.Parent` could be "
            "any model or even nil. You must verify the parent has a Humanoid before "
            "calling TakeDamage, and add a debounce to prevent damage spam."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local debounce: { [Model]: boolean } = {}

            local function onTouched(hit: BasePart)
            \tlocal character = hit.Parent :: Model?
            \tif not character then
            \t\treturn
            \tend
            \tlocal humanoid = character:FindFirstChildOfClass("Humanoid")
            \tif not humanoid then
            \t\treturn
            \tend
            \tif debounce[character] then
            \t\treturn
            \tend
            \tdebounce[character] = true
            \thumanoid:TakeDamage(25)
            \ttask.delay(1, function()
            \t\tdebounce[character] = nil
            \tend)
            end

            damagePart.Touched:Connect(onTouched)
        """),
    },
    # -----------------------------------------------------------------------
    # 2. Off-by-one errors
    # -----------------------------------------------------------------------
    {
        "category_tag": "off_by_one",
        "difficulty": 2,
        "user": (
            "I'm trying to iterate over all items but the last one is always skipped. "
            "Can you fix it?\n\n"
            "```lua\n"
            'local items = {"Sword", "Shield", "Potion", "Bow"}\n'
            "for i = 1, #items - 1 do\n"
            "\tprint(items[i])\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "Classic off-by-one: `#items - 1` excludes the last element. Lua/Luau "
            "arrays are 1-indexed and # gives the full length, so the loop should "
            "go to #items. Or better yet, use ipairs."
        ),
        "explanation": (
            "Luau arrays are 1-indexed and `#items` gives the total count. Using "
            "`#items - 1` skips the last element. Either loop to `#items` or use `ipairs`."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local items = {"Sword", "Shield", "Potion", "Bow"}
            for _, item in ipairs(items) do
            \tprint(item)
            end
        """),
    },
    {
        "category_tag": "off_by_one",
        "difficulty": 3,
        "user": (
            "My pagination system shows one fewer page than expected. Can you fix it?\n\n"
            "```lua\n"
            "local ITEMS_PER_PAGE = 10\n"
            "local totalItems = 25\n"
            "local totalPages = totalItems / ITEMS_PER_PAGE\n\n"
            "for page = 1, totalPages do\n"
            '\tprint("Page", page)\n'
            "end\n"
            "```"
        ),
        "thinking": (
            "25 / 10 = 2.5 which truncates to 2 in the loop range, missing page 3. "
            "Need math.ceil to round up for partial pages."
        ),
        "explanation": (
            "Integer division truncates: `25 / 10` gives `2.5`, and the for loop "
            "stops at 2, missing the final partial page. Use `math.ceil` to round up."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local ITEMS_PER_PAGE = 10
            local totalItems = 25
            local totalPages = math.ceil(totalItems / ITEMS_PER_PAGE)

            for page = 1, totalPages do
            \tprint("Page", page)
            end
        """),
    },
    {
        "category_tag": "off_by_one",
        "difficulty": 3,
        "user": (
            "My inventory swap function sometimes overwrites the wrong slot. "
            "Can you fix it?\n\n"
            "```lua\n"
            "local function swapSlots(inventory, fromIndex, toIndex)\n"
            "\tif fromIndex > #inventory or toIndex > #inventory then\n"
            "\t\treturn false\n"
            "\tend\n"
            "\tlocal temp = inventory[fromIndex]\n"
            "\tinventory[fromIndex] = inventory[toIndex]\n"
            "\tinventory[toIndex] = temp\n"
            "\treturn true\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "Missing lower bound check — indices of 0 or negative would pass the "
            "guard but access invalid slots. Luau arrays are 1-indexed so we need "
            "to check >= 1 as well."
        ),
        "explanation": (
            "The bounds check only validates the upper limit. Indices of 0 or negative "
            "numbers slip through and corrupt memory. Always validate both lower and "
            "upper bounds for 1-indexed Luau arrays."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local function swapSlots(inventory: { any }, fromIndex: number, toIndex: number): boolean
            \tif fromIndex < 1 or fromIndex > #inventory then
            \t\treturn false
            \tend
            \tif toIndex < 1 or toIndex > #inventory then
            \t\treturn false
            \tend
            \tif fromIndex == toIndex then
            \t\treturn true
            \tend
            \tlocal temp = inventory[fromIndex]
            \tinventory[fromIndex] = inventory[toIndex]
            \tinventory[toIndex] = temp
            \treturn true
            end
        """),
    },
    # -----------------------------------------------------------------------
    # 3. Client/server boundary violations
    # -----------------------------------------------------------------------
    {
        "category_tag": "client_server_boundary",
        "difficulty": 3,
        "user": (
            "My LocalScript can't access ServerStorage. It errors with "
            "'ServerStorage is not a valid member of DataModel'. Can you fix it?\n\n"
            "```lua\n"
            "-- LocalScript in StarterPlayerScripts\n"
            'local ServerStorage = game:GetService("ServerStorage")\n'
            "local weapons = ServerStorage.Weapons:GetChildren()\n\n"
            "for _, weapon in weapons do\n"
            "\tprint(weapon.Name)\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "ServerStorage is inaccessible from the client. LocalScripts cannot see "
            "ServerStorage at all. The weapons need to be in ReplicatedStorage if the "
            "client needs to read them, or fetched via a RemoteFunction."
        ),
        "explanation": (
            "`ServerStorage` is server-only — clients cannot access it at all. Move "
            "assets the client needs to read into `ReplicatedStorage`, or use a "
            "`RemoteFunction` to request data from the server."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            -- LocalScript in StarterPlayerScripts
            -- Weapons folder must be in ReplicatedStorage for client access
            local ReplicatedStorage = game:GetService("ReplicatedStorage")
            local weapons = ReplicatedStorage:WaitForChild("Weapons"):GetChildren()

            for _, weapon in weapons do
            \tprint(weapon.Name)
            end
        """),
    },
    {
        "category_tag": "client_server_boundary",
        "difficulty": 3,
        "user": (
            "My LocalScript tries to give coins directly but nothing happens. "
            "Can you fix it?\n\n"
            "```lua\n"
            "-- LocalScript\n"
            'local Players = game:GetService("Players")\n'
            "local player = Players.LocalPlayer\n\n"
            "local function onCoinTouched(coin)\n"
            "\tplayer.leaderstats.Coins.Value += 1\n"
            "\tcoin:Destroy()\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "The client is directly modifying leaderstats which is a server-owned "
            "value. Changes made on the client don't replicate to the server. Also, "
            "destroying instances from the client doesn't replicate. This needs to "
            "fire a RemoteEvent to the server."
        ),
        "explanation": (
            "Leaderstats values are server-authoritative — client changes don't replicate. "
            "The client must fire a `RemoteEvent` to the server, and the server validates "
            "and applies the change. Never trust the client to modify game state directly."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            -- LocalScript: detect coin touch and request pickup from server
            local ReplicatedStorage = game:GetService("ReplicatedStorage")
            local requestPickup = ReplicatedStorage:WaitForChild("Remotes"):WaitForChild("RequestCoinPickup")

            local function onCoinTouched(coin: BasePart)
            \trequestPickup:FireServer(coin)
            end

            -- Server Script: validate and award coin
            --[[
            local function onPickupRequested(player: Player, coin: BasePart)
            \tif not coin or not coin:IsDescendantOf(workspace) then
            \t\treturn
            \tend
            \tlocal character = player.Character
            \tif not character then
            \t\treturn
            \tend
            \tlocal distance = (coin.Position - character:GetPivot().Position).Magnitude
            \tif distance > 15 then
            \t\treturn -- too far, possible exploit
            \tend
            \tplayer.leaderstats.Coins.Value += 1
            \tcoin:Destroy()
            end
            requestPickup.OnServerEvent:Connect(onPickupRequested)
            ]]
        """),
    },
    {
        "category_tag": "client_server_boundary",
        "difficulty": 4,
        "user": (
            "My admin command script on the client can kick players but exploiters "
            "could abuse it. Can you fix it?\n\n"
            "```lua\n"
            "-- LocalScript\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            "local kickRemote = ReplicatedStorage.Remotes.KickPlayer\n\n"
            "local function kickPlayer(targetName: string)\n"
            "\tkickRemote:FireServer(targetName)\n"
            "end\n\n"
            "-- Server Script\n"
            "kickRemote.OnServerEvent:Connect(function(player, targetName)\n"
            "\tlocal target = game.Players:FindFirstChild(targetName)\n"
            "\tif target then\n"
            '\t\ttarget:Kick("Kicked by admin")\n'
            "\tend\n"
            "end)\n"
            "```"
        ),
        "thinking": (
            "The server blindly trusts whoever fires the RemoteEvent. Any exploiter "
            "can fire KickPlayer and kick anyone. The server must verify that the "
            "requesting player is actually an admin before processing the command."
        ),
        "explanation": (
            "The server never checks if the requesting player is an admin. Any exploiter "
            "can fire this RemoteEvent and kick anyone. Always validate permissions "
            "server-side — never trust the client."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            -- Server Script
            local ADMIN_IDS: { number } = table.freeze({ 123456789 }) -- your UserId(s)

            local function isAdmin(player: Player): boolean
            \treturn table.find(ADMIN_IDS, player.UserId) ~= nil
            end

            kickRemote.OnServerEvent:Connect(function(player: Player, targetName: unknown)
            \tif not isAdmin(player) then
            \t\twarn("[Security] Non-admin", player.Name, "attempted kick command")
            \t\treturn
            \tend
            \tif typeof(targetName) ~= "string" then
            \t\treturn
            \tend
            \tlocal target = game.Players:FindFirstChild(targetName)
            \tif target and not isAdmin(target) then
            \t\ttarget:Kick("Kicked by admin " .. player.Name)
            \tend
            end)
        """),
    },
    # -----------------------------------------------------------------------
    # 4. Race conditions — not using WaitForChild
    # -----------------------------------------------------------------------
    {
        "category_tag": "race_condition",
        "difficulty": 2,
        "user": (
            "My LocalScript sometimes can't find 'GameConfig' in ReplicatedStorage. "
            "It works when I rejoin. Can you fix it?\n\n"
            "```lua\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            "local config = ReplicatedStorage.Shared.Config.GameConfig\n"
            "print(config.MaxPlayers.Value)\n"
            "```"
        ),
        "thinking": (
            "The client script runs before the server has replicated the instances. "
            "Direct indexing fails if the instance hasn't arrived yet. Need WaitForChild "
            "to wait for replication."
        ),
        "explanation": (
            "On the client, instances from the server replicate asynchronously. If your "
            "script runs before `GameConfig` has replicated, direct indexing errors. "
            "Use `WaitForChild` to safely wait for replication."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local ReplicatedStorage = game:GetService("ReplicatedStorage")
            local Shared = ReplicatedStorage:WaitForChild("Shared")
            local Config = Shared:WaitForChild("Config")
            local gameConfig = Config:WaitForChild("GameConfig")
            local maxPlayers = (gameConfig :: any):WaitForChild("MaxPlayers")
            print(maxPlayers.Value)
        """),
    },
    {
        "category_tag": "race_condition",
        "difficulty": 3,
        "user": (
            "My character setup script sometimes fails because the HumanoidRootPart "
            "isn't there yet. Can you fix it?\n\n"
            "```lua\n"
            'local Players = game:GetService("Players")\n'
            "local player = Players.LocalPlayer\n\n"
            "player.CharacterAdded:Connect(function(character)\n"
            "\tlocal rootPart = character.HumanoidRootPart\n"
            "\trootPart.Anchored = true\n"
            "\ttask.wait(3)\n"
            "\trootPart.Anchored = false\n"
            "end)\n"
            "```"
        ),
        "thinking": (
            "CharacterAdded fires when the model is added but its children (like "
            "HumanoidRootPart) may not be loaded yet. Need WaitForChild."
        ),
        "explanation": (
            "`CharacterAdded` fires as soon as the Character model is parented, but its "
            "children (HumanoidRootPart, Humanoid, etc.) may not exist yet. Always use "
            "`WaitForChild` to wait for specific descendants."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Players = game:GetService("Players")
            local player = Players.LocalPlayer

            player.CharacterAdded:Connect(function(character: Model)
            \tlocal rootPart = character:WaitForChild("HumanoidRootPart", 10)
            \tif not rootPart then
            \t\twarn("HumanoidRootPart did not load in time")
            \t\treturn
            \tend
            \t(rootPart :: BasePart).Anchored = true
            \ttask.wait(3)
            \tif rootPart.Parent then -- character might have been removed
            \t\t(rootPart :: BasePart).Anchored = false
            \tend
            end)
        """),
    },
    # -----------------------------------------------------------------------
    # 5. Memory leaks — not disconnecting connections
    # -----------------------------------------------------------------------
    {
        "category_tag": "memory_leak",
        "difficulty": 3,
        "user": (
            "My game gets slower over time and I think it's a memory leak. "
            "Can you fix it?\n\n"
            "```lua\n"
            'local RunService = game:GetService("RunService")\n'
            'local Players = game:GetService("Players")\n\n'
            "Players.PlayerAdded:Connect(function(player)\n"
            "\tplayer.CharacterAdded:Connect(function(character)\n"
            "\t\tRunService.Heartbeat:Connect(function(dt)\n"
            "\t\t\t-- Update overhead UI for this character\n"
            '\t\t\tlocal head = character:FindFirstChild("Head")\n'
            "\t\t\tif head then\n"
            "\t\t\t\t-- update billboard position\n"
            "\t\t\tend\n"
            "\t\tend)\n"
            "\tend)\n"
            "end)\n"
            "```"
        ),
        "thinking": (
            "Every respawn creates a new Heartbeat connection that is never disconnected. "
            "After N respawns, there are N heartbeat listeners running simultaneously. "
            "Need to store the connection and disconnect it when the character is removed "
            "or a new one spawns. A Trove or manual cleanup pattern is needed."
        ),
        "explanation": (
            "Every time the character respawns, a new `Heartbeat` connection is created "
            "but the old one is never disconnected. After multiple respawns, dozens of "
            "listeners stack up. Store connections and disconnect them on character removal."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local RunService = game:GetService("RunService")
            local Players = game:GetService("Players")

            Players.PlayerAdded:Connect(function(player: Player)
            \tlocal heartbeatConn: RBXScriptConnection? = nil

            \tplayer.CharacterAdded:Connect(function(character: Model)
            \t\t-- Clean up previous connection
            \t\tif heartbeatConn then
            \t\t\theartbeatConn:Disconnect()
            \t\t\theartbeatConn = nil
            \t\tend

            \t\theartbeatConn = RunService.Heartbeat:Connect(function(_dt: number)
            \t\t\tlocal head = character:FindFirstChild("Head")
            \t\t\tif head then
            \t\t\t\t-- update billboard position
            \t\t\tend
            \t\tend)

            \t\t-- Also disconnect when character is removed
            \t\tcharacter.AncestryChanged:Connect(function(_, newParent)
            \t\t\tif newParent == nil and heartbeatConn then
            \t\t\t\theartbeatConn:Disconnect()
            \t\t\t\theartbeatConn = nil
            \t\t\tend
            \t\tend)
            \tend)
            end)
        """),
    },
    {
        "category_tag": "memory_leak",
        "difficulty": 3,
        "user": (
            "I add Touched connections to parts in a zone but never clean them up "
            "when the zone unloads. Can you fix it?\n\n"
            "```lua\n"
            "local function setupZone(zoneParts)\n"
            "\tfor _, part in zoneParts do\n"
            "\t\tpart.Touched:Connect(function(hit)\n"
            '\t\t\tprint(hit.Name, "touched", part.Name)\n'
            "\t\tend)\n"
            "\tend\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "The connections are created but never stored, so there's no way to "
            "disconnect them when the zone unloads. Need to collect connections and "
            "return them or use a Trove-style cleanup."
        ),
        "explanation": (
            "Without storing references to the connections, you can never disconnect them. "
            "When the zone unloads, the listeners remain active, leaking memory. "
            "Collect connections and return a cleanup function."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            type CleanupFn = () -> ()

            local function setupZone(zoneParts: { BasePart }): CleanupFn
            \tlocal connections: { RBXScriptConnection } = {}

            \tfor _, part in zoneParts do
            \t\tlocal conn = part.Touched:Connect(function(hit: BasePart)
            \t\t\tprint(hit.Name, "touched", part.Name)
            \t\tend)
            \t\ttable.insert(connections, conn)
            \tend

            \treturn function()
            \t\tfor _, conn in connections do
            \t\t\tconn:Disconnect()
            \t\tend
            \t\ttable.clear(connections)
            \tend
            end
        """),
    },
    # -----------------------------------------------------------------------
    # 6. Deprecated API usage
    # -----------------------------------------------------------------------
    {
        "category_tag": "deprecated_api",
        "difficulty": 2,
        "user": (
            "My code uses some old APIs and I want to modernize it. Can you fix it?\n\n"
            "```lua\n"
            "wait(2)\n"
            "spawn(function()\n"
            "\twhile wait(0.5) do\n"
            '\t\tprint("tick")\n'
            "\tend\n"
            "end)\n"
            "delay(5, function()\n"
            '\tprint("delayed")\n'
            "end)\n"
            "```"
        ),
        "thinking": (
            "wait(), spawn(), and delay() are deprecated legacy globals. They have "
            "inconsistent throttling behavior. Modern Luau uses task.wait(), "
            "task.spawn(), and task.delay() from the task library."
        ),
        "explanation": (
            "`wait()`, `spawn()`, and `delay()` are deprecated legacy globals with "
            "throttling issues. Use the modern `task` library equivalents: `task.wait()`, "
            "`task.spawn()`, and `task.delay()`. They have more predictable scheduling."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            task.wait(2)
            task.spawn(function()
            \twhile true do
            \t\ttask.wait(0.5)
            \t\tprint("tick")
            \tend
            end)
            task.delay(5, function()
            \tprint("delayed")
            end)
        """),
    },
    {
        "category_tag": "deprecated_api",
        "difficulty": 2,
        "user": (
            "I'm told my requires are using deprecated paths. Can you fix it?\n\n"
            "```lua\n"
            "local module = require(game.ServerScriptService.Modules.DataManager)\n"
            "local config = require(game.ReplicatedStorage.Config)\n"
            "```"
        ),
        "thinking": (
            "Using dot notation for service access (game.ServerScriptService) is "
            "fragile and deprecated style. Should use game:GetService() which is "
            "the canonical way to access services."
        ),
        "explanation": (
            "Accessing services via `game.ServiceName` instead of `game:GetService()` "
            "is fragile — it depends on the service already existing in the DataModel. "
            "`GetService` is guaranteed to return the service and is the canonical approach."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local ServerScriptService = game:GetService("ServerScriptService")
            local ReplicatedStorage = game:GetService("ReplicatedStorage")

            local module = require(ServerScriptService:WaitForChild("Modules"):WaitForChild("DataManager"))
            local config = require(ReplicatedStorage:WaitForChild("Config"))
        """),
    },
    {
        "category_tag": "deprecated_api",
        "difficulty": 2,
        "user": (
            "I'm using the old Vector3 math pattern. Can you modernize it?\n\n"
            "```lua\n"
            "local function interpolatePosition(a, b, t)\n"
            "\treturn a + (b - a) * t\n"
            "end\n\n"
            "local function clampMagnitude(v, maxMag)\n"
            "\tif v.Magnitude > maxMag then\n"
            "\t\treturn v.Unit * maxMag\n"
            "\tend\n"
            "\treturn v\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "Manual lerp should use math.lerp for NCG optimization (compiles to FMA). "
            "Also should add type annotations for --!strict and @native for hot paths."
        ),
        "explanation": (
            "Manual `a + (b - a) * t` should be `math.lerp` — it compiles to a single "
            "FMA instruction under NCG. Additionally, `v.Unit` on a zero-length vector "
            "produces NaN. Add a zero-check guard."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            @native
            local function interpolatePosition(a: Vector3, b: Vector3, t: number): Vector3
            \treturn a:Lerp(b, t)
            end

            @native
            local function clampMagnitude(v: Vector3, maxMag: number): Vector3
            \tlocal mag = v.Magnitude
            \tif mag <= maxMag or mag < 1e-6 then
            \t\treturn v
            \tend
            \treturn v * (maxMag / mag)
            end
        """),
    },
    # -----------------------------------------------------------------------
    # 7. Type errors — wrong types to Roblox APIs
    # -----------------------------------------------------------------------
    {
        "category_tag": "type_error",
        "difficulty": 2,
        "user": (
            "I get 'Unable to cast string to Vector3' when I run this. Can you fix it?\n\n"
            "```lua\n"
            'local part = Instance.new("Part")\n'
            'part.Position = "10, 20, 30"\n'
            'part.Size = "4, 1, 4"\n'
            "part.Parent = workspace\n"
            "```"
        ),
        "thinking": (
            "Position and Size expect Vector3 values, not strings. The string "
            "representation of a vector won't auto-convert."
        ),
        "explanation": (
            "`Position` and `Size` are `Vector3` properties — they don't accept strings. "
            "Use `Vector3.new()` to construct the values."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local part = Instance.new("Part")
            part.Position = Vector3.new(10, 20, 30)
            part.Size = Vector3.new(4, 1, 4)
            part.Parent = workspace
        """),
    },
    {
        "category_tag": "type_error",
        "difficulty": 3,
        "user": (
            "My TweenService call errors with 'invalid argument #3'. Can you fix it?\n\n"
            "```lua\n"
            'local TweenService = game:GetService("TweenService")\n'
            "local part = workspace.MyPart\n\n"
            "local tween = TweenService:Create(part, {\n"
            "\tTime = 2,\n"
            '\tEasingStyle = "Quad",\n'
            '\tEasingDirection = "Out",\n'
            "}, {\n"
            "\tPosition = Vector3.new(0, 10, 0),\n"
            "})\n"
            "tween:Play()\n"
            "```"
        ),
        "thinking": (
            "TweenService:Create takes a TweenInfo object as second argument, not a "
            "table. Also, EasingStyle and EasingDirection are Enum values not strings."
        ),
        "explanation": (
            "The second argument to `TweenService:Create` must be a `TweenInfo` object, "
            "not a plain table. Easing values must be `Enum.EasingStyle` and "
            "`Enum.EasingDirection`, not strings."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local TweenService = game:GetService("TweenService")
            local part = workspace:WaitForChild("MyPart")

            local tweenInfo = TweenInfo.new(
            \t2,                            -- Time
            \tEnum.EasingStyle.Quad,        -- EasingStyle
            \tEnum.EasingDirection.Out       -- EasingDirection
            )
            local tween = TweenService:Create(part, tweenInfo, {
            \tPosition = Vector3.new(0, 10, 0),
            })
            tween:Play()
        """),
    },
    # -----------------------------------------------------------------------
    # 8. Network ownership issues
    # -----------------------------------------------------------------------
    {
        "category_tag": "network_ownership",
        "difficulty": 4,
        "user": (
            "My server-controlled NPC jitters when players get near it. "
            "Can you fix it?\n\n"
            "```lua\n"
            "-- Server Script\n"
            "local npc = workspace.NPC\n"
            "local humanoid = npc.Humanoid\n\n"
            "while true do\n"
            "\thumanoid:MoveTo(workspace.Waypoint1.Position)\n"
            "\thumanoid.MoveToFinished:Wait()\n"
            "\thumanoid:MoveTo(workspace.Waypoint2.Position)\n"
            "\thumanoid.MoveToFinished:Wait()\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "When a player gets close, Roblox auto-assigns network ownership of the NPC "
            "to that player's client. The server is trying to control the NPC but the "
            "client now owns its physics, causing jitter. Need to set network ownership "
            "to the server explicitly."
        ),
        "explanation": (
            "Roblox automatically transfers network ownership of unanchored parts to "
            "nearby players. When the server tries to move the NPC but a client owns "
            "its physics, it causes jitter. Use `SetNetworkOwner(nil)` to keep server "
            "ownership permanently."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local npc = workspace:WaitForChild("NPC")
            local humanoid = npc:WaitForChild("Humanoid") :: Humanoid
            local rootPart = npc:WaitForChild("HumanoidRootPart") :: BasePart

            -- Force server ownership so clients don't steal physics control
            rootPart:SetNetworkOwner(nil)

            while true do
            \thumanoid:MoveTo(workspace:WaitForChild("Waypoint1").Position)
            \thumanoid.MoveToFinished:Wait()
            \thumanoid:MoveTo(workspace:WaitForChild("Waypoint2").Position)
            \thumanoid.MoveToFinished:Wait()
            end
        """),
    },
    {
        "category_tag": "network_ownership",
        "difficulty": 4,
        "user": (
            "My physics projectile goes crazy when it gets near other players. "
            "Can you fix it?\n\n"
            "```lua\n"
            "local function fireProjectile(origin, direction)\n"
            '\tlocal projectile = Instance.new("Part")\n'
            "\tprojectile.Size = Vector3.new(1, 1, 1)\n"
            "\tprojectile.CFrame = CFrame.new(origin)\n"
            "\tprojectile.Parent = workspace\n\n"
            '\tlocal velocity = Instance.new("LinearVelocity")\n'
            "\tvelocity.VectorVelocity = direction * 100\n"
            "\tvelocity.Parent = projectile\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "The projectile is unanchored and has no explicit network owner. When it "
            "flies near a player, ownership transfers mid-flight, causing the projectile "
            "to stutter or change direction. Need server ownership and also an attachment "
            "for the LinearVelocity constraint."
        ),
        "explanation": (
            "Unanchored parts auto-transfer network ownership to nearby clients, causing "
            "projectiles to stutter mid-flight. Lock ownership to the server. Also, "
            "`LinearVelocity` requires an `Attachment` — it won't work parented directly "
            "to the part."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Debris = game:GetService("Debris")

            local function fireProjectile(origin: Vector3, direction: Vector3)
            \tlocal projectile = Instance.new("Part")
            \tprojectile.Size = Vector3.new(1, 1, 1)
            \tprojectile.CFrame = CFrame.new(origin)
            \tprojectile.CanCollide = true
            \tprojectile.Anchored = false

            \tlocal attachment = Instance.new("Attachment")
            \tattachment.Parent = projectile

            \tlocal velocity = Instance.new("LinearVelocity")
            \tvelocity.Attachment0 = attachment
            \tvelocity.VectorVelocity = direction.Unit * 100
            \tvelocity.MaxForce = math.huge
            \tvelocity.RelativeTo = Enum.ActuatorRelativeTo.World
            \tvelocity.Parent = projectile

            \tprojectile.Parent = workspace

            \t-- Lock network ownership to server
            \tprojectile:SetNetworkOwner(nil)

            \t-- Auto-cleanup after 10 seconds
            \tDebris:AddItem(projectile, 10)
            end
        """),
    },
    # -----------------------------------------------------------------------
    # 9. DataStore throttling
    # -----------------------------------------------------------------------
    {
        "category_tag": "datastore_throttling",
        "difficulty": 3,
        "user": (
            "My DataStore save sometimes silently fails and players lose progress. "
            "Can you fix it?\n\n"
            "```lua\n"
            'local DataStoreService = game:GetService("DataStoreService")\n'
            'local store = DataStoreService:GetDataStore("PlayerData")\n\n'
            "local function saveData(player, data)\n"
            '\tstore:SetAsync("player_" .. player.UserId, data)\n'
            "end\n"
            "```"
        ),
        "thinking": (
            "SetAsync can fail due to throttling, network errors, or service outages. "
            "Without pcall, the error is unhandled and the data is silently lost. "
            "Need pcall wrapping, retry logic, and UpdateAsync instead of SetAsync "
            "to avoid race conditions."
        ),
        "explanation": (
            "`SetAsync` can fail from throttling, network issues, or service outages. "
            "Without `pcall`, the error crashes the thread and data is lost. Use `pcall` "
            "with retries, and prefer `UpdateAsync` over `SetAsync` to avoid race conditions "
            "with concurrent writes."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local DataStoreService = game:GetService("DataStoreService")
            local store = DataStoreService:GetDataStore("PlayerData")

            local MAX_RETRIES = 3
            local RETRY_DELAY = 2

            local function saveData(player: Player, data: { [string]: any }): boolean
            \tlocal key = "player_" .. tostring(player.UserId)

            \tfor attempt = 1, MAX_RETRIES do
            \t\tlocal ok, err = pcall(function()
            \t\t\tstore:UpdateAsync(key, function(_oldData)
            \t\t\t\treturn data
            \t\t\tend)
            \t\tend)

            \t\tif ok then
            \t\t\treturn true
            \t\tend

            \t\twarn(
            \t\t\tstring.format("[DataStore] Save failed for %s (attempt %d/%d): %s",
            \t\t\t\tplayer.Name, attempt, MAX_RETRIES, tostring(err))
            \t\t)

            \t\tif attempt < MAX_RETRIES then
            \t\t\ttask.wait(RETRY_DELAY * attempt) -- exponential-ish backoff
            \t\tend
            \tend

            \twarn("[DataStore] All retries exhausted for", player.Name)
            \treturn false
            end
        """),
    },
    {
        "category_tag": "datastore_throttling",
        "difficulty": 4,
        "user": (
            "My DataStore load doesn't handle the case where it's a new player "
            "with no data. Can you fix it?\n\n"
            "```lua\n"
            "local function loadData(player)\n"
            '\tlocal data = store:GetAsync("player_" .. player.UserId)\n'
            "\tplayer.leaderstats.Coins.Value = data.Coins\n"
            "\tplayer.leaderstats.Level.Value = data.Level\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "GetAsync returns nil for new players. Indexing nil.Coins errors. Also, "
            "GetAsync can fail and needs pcall. Need default data for new players."
        ),
        "explanation": (
            "`GetAsync` returns `nil` for new players who have no saved data, and it can "
            "also fail from throttling. You need `pcall` for error handling and a default "
            "data template for new players."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local DEFAULT_DATA = table.freeze({
            \tCoins = 0,
            \tLevel = 1,
            })

            local function loadData(player: Player): { [string]: any }
            \tlocal key = "player_" .. tostring(player.UserId)

            \tlocal ok, result = pcall(function()
            \t\treturn store:GetAsync(key)
            \tend)

            \tif not ok then
            \t\twarn("[DataStore] Load failed for", player.Name, ":", result)
            \t\treturn table.clone(DEFAULT_DATA)
            \tend

            \tif result == nil then
            \t\t-- New player, no saved data
            \t\treturn table.clone(DEFAULT_DATA)
            \tend

            \t-- Merge with defaults to handle schema additions
            \tlocal data = table.clone(DEFAULT_DATA)
            \tfor k, v in result :: { [string]: any } do
            \t\tdata[k] = v
            \tend
            \treturn data
            end
        """),
    },
    # -----------------------------------------------------------------------
    # 10. Infinite yield warnings — WaitForChild without timeout
    # -----------------------------------------------------------------------
    {
        "category_tag": "infinite_yield",
        "difficulty": 2,
        "user": (
            "I get 'Infinite yield possible' warnings in the output. Can you fix it?\n\n"
            "```lua\n"
            'local tool = player.Backpack:WaitForChild("Sword")\n'
            'local gui = player.PlayerGui:WaitForChild("ShopGui")\n'
            'local data = ReplicatedStorage:WaitForChild("PlayerData"):WaitForChild(player.Name)\n'
            "```"
        ),
        "thinking": (
            "WaitForChild without a timeout waits indefinitely. If the instance never "
            "exists (removed, renamed, never created), it yields forever and prints "
            "the warning after 5 seconds. Add timeouts and nil checks."
        ),
        "explanation": (
            "`WaitForChild` without a timeout yields forever if the instance never appears. "
            "Always pass a timeout (second argument) and handle the `nil` return when "
            "the timeout expires."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local WAIT_TIMEOUT = 10

            local tool = player.Backpack:WaitForChild("Sword", WAIT_TIMEOUT)
            if not tool then
            \twarn("Sword not found in backpack after", WAIT_TIMEOUT, "seconds")
            \treturn
            end

            local gui = player.PlayerGui:WaitForChild("ShopGui", WAIT_TIMEOUT)
            if not gui then
            \twarn("ShopGui not found after", WAIT_TIMEOUT, "seconds")
            \treturn
            end

            local playerDataFolder = ReplicatedStorage:WaitForChild("PlayerData", WAIT_TIMEOUT)
            if not playerDataFolder then
            \twarn("PlayerData folder not found")
            \treturn
            end
            local dataEntry = playerDataFolder:WaitForChild(player.Name, WAIT_TIMEOUT)
            if not dataEntry then
            \twarn("Data entry for", player.Name, "not found")
            \treturn
            end
        """),
    },
    # -----------------------------------------------------------------------
    # 11. Closure-in-loop bugs
    # -----------------------------------------------------------------------
    {
        "category_tag": "closure_in_loop",
        "difficulty": 3,
        "user": (
            "All my buttons print the same number when clicked. Can you fix it?\n\n"
            "```lua\n"
            "local buttons = script.Parent:GetChildren()\n"
            "for i = 1, #buttons do\n"
            "\tbuttons[i].Activated:Connect(function()\n"
            '\t\tprint("Button", i, "clicked")\n'
            "\tend)\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "Actually in Luau, numeric for loops create a new scope per iteration, "
            "so this shouldn't have the classic JS closure bug. But the real issue "
            "here might be a misunderstanding. Let me reconsider — in Luau, for loops "
            "DO create a new local per iteration, so i is correctly captured. "
            "The more likely bug pattern in Luau closures is with while loops or "
            "generic iteration where a variable is reused. Let me adjust this to "
            "a pattern that actually has the closure bug in Luau."
        ),
        "explanation": (
            "While Luau numeric `for` loops do capture `i` correctly per iteration, "
            "the broader issue here is that closures in loops create garbage pressure. "
            "For hot-path code, extract the handler to a named function and pass context "
            "via the instance itself to avoid per-iteration allocations."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local function onButtonActivated(button: GuiButton, index: number)
            \treturn function()
            \t\tprint("Button", index, "clicked:", button.Name)
            \tend
            end

            local buttons = script.Parent:GetChildren()
            for i, button in buttons do
            \tif button:IsA("GuiButton") then
            \t\tbutton.Activated:Connect(onButtonActivated(button, i))
            \tend
            end
        """),
    },
    {
        "category_tag": "closure_in_loop",
        "difficulty": 3,
        "user": (
            "My spawner creates enemies but they all reference the last spawn point. "
            "Can you fix it?\n\n"
            "```lua\n"
            "local spawnPoints = workspace.SpawnPoints:GetChildren()\n"
            "local currentPoint = nil\n\n"
            "for _, point in spawnPoints do\n"
            "\tcurrentPoint = point\n"
            "\ttask.delay(2, function()\n"
            "\t\tlocal enemy = createEnemy()\n"
            "\t\tenemy:PivotTo(currentPoint.CFrame)\n"
            "\tend)\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "currentPoint is an upvalue shared by all closures. By the time the "
            "task.delay callbacks fire, the loop has finished and currentPoint is "
            "the last spawn point. Each closure should capture point directly."
        ),
        "explanation": (
            "`currentPoint` is a single variable shared by all delayed closures. By the "
            "time they execute, the loop has ended and it references the last point. "
            "Use the loop variable `point` directly — it's scoped per iteration in Luau."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local spawnPoints = workspace:WaitForChild("SpawnPoints"):GetChildren()

            for _, point in spawnPoints do
            \ttask.delay(2, function()
            \t\tlocal enemy = createEnemy()
            \t\tenemy:PivotTo(point.CFrame)
            \tend)
            end
        """),
    },
    # -----------------------------------------------------------------------
    # 12. Missing pcall on fallible operations
    # -----------------------------------------------------------------------
    {
        "category_tag": "missing_pcall",
        "difficulty": 2,
        "user": (
            "My HTTP request crashes the whole script when the API is down. "
            "Can you fix it?\n\n"
            "```lua\n"
            'local HttpService = game:GetService("HttpService")\n\n'
            "local function fetchLeaderboard()\n"
            '\tlocal response = HttpService:GetAsync("https://api.example.com/leaderboard")\n'
            "\tlocal data = HttpService:JSONDecode(response)\n"
            "\treturn data\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "HttpService:GetAsync can fail from network errors, timeouts, rate limits. "
            "JSONDecode can fail if response isn't valid JSON. Both need pcall wrapping."
        ),
        "explanation": (
            "`GetAsync` can fail from network errors, timeouts, or rate limits. "
            "`JSONDecode` can fail on malformed responses. Both are fallible operations "
            "that need `pcall` wrapping with proper error handling."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local HttpService = game:GetService("HttpService")

            type LeaderboardResult = {
            \tok: boolean,
            \tdata: { any }?,
            \terror: string?,
            }

            local function fetchLeaderboard(): LeaderboardResult
            \tlocal ok, response = pcall(function()
            \t\treturn HttpService:GetAsync("https://api.example.com/leaderboard")
            \tend)

            \tif not ok then
            \t\twarn("[HTTP] Request failed:", response)
            \t\treturn { ok = false, data = nil, error = tostring(response) }
            \tend

            \tlocal decodeOk, data = pcall(function()
            \t\treturn HttpService:JSONDecode(response)
            \tend)

            \tif not decodeOk then
            \t\twarn("[HTTP] JSON decode failed:", data)
            \t\treturn { ok = false, data = nil, error = "Invalid JSON response" }
            \tend

            \treturn { ok = true, data = data, error = nil }
            end
        """),
    },
    {
        "category_tag": "missing_pcall",
        "difficulty": 3,
        "user": (
            "My MarketplaceService purchase prompt sometimes crashes. Can you fix it?\n\n"
            "```lua\n"
            'local MarketplaceService = game:GetService("MarketplaceService")\n\n'
            "local function getProductInfo(assetId)\n"
            "\tlocal info = MarketplaceService:GetProductInfo(assetId)\n"
            "\treturn info.Name, info.PriceInRobux\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "GetProductInfo is a yielding function that contacts Roblox servers. "
            "It can fail from invalid asset IDs, rate limiting, or network issues. "
            "Needs pcall and proper nil handling."
        ),
        "explanation": (
            "`GetProductInfo` contacts Roblox servers and can fail from invalid IDs, "
            "rate limiting, or network issues. Always wrap external API calls in `pcall`."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local MarketplaceService = game:GetService("MarketplaceService")

            local function getProductInfo(assetId: number): (string?, number?)
            \tlocal ok, info = pcall(function()
            \t\treturn MarketplaceService:GetProductInfo(assetId, Enum.InfoType.Asset)
            \tend)

            \tif not ok then
            \t\twarn("[Marketplace] Failed to get product info for", assetId, ":", info)
            \t\treturn nil, nil
            \tend

            \treturn info.Name, info.PriceInRobux
            end
        """),
    },
    # -----------------------------------------------------------------------
    # 13. Instance.new with parent in constructor (deprecated)
    # -----------------------------------------------------------------------
    {
        "category_tag": "instance_new_parent",
        "difficulty": 2,
        "user": (
            "I heard setting parent in Instance.new is bad practice. Can you fix it?\n\n"
            "```lua\n"
            'local part = Instance.new("Part", workspace)\n'
            "part.Size = Vector3.new(4, 1, 4)\n"
            "part.Position = Vector3.new(0, 10, 0)\n"
            "part.Anchored = true\n"
            'part.BrickColor = BrickColor.new("Bright red")\n'
            "```"
        ),
        "thinking": (
            "Setting parent in the Instance.new constructor is deprecated because it "
            "causes the instance to replicate/render before all properties are set. "
            "This triggers unnecessary replication events for each subsequent property "
            "change. Set parent last after all properties are configured."
        ),
        "explanation": (
            "Passing the parent as the second argument to `Instance.new()` is deprecated. "
            "It causes the instance to replicate immediately, then each property change "
            "triggers another replication event. Set all properties first, then parent last."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local part = Instance.new("Part")
            part.Size = Vector3.new(4, 1, 4)
            part.Position = Vector3.new(0, 10, 0)
            part.Anchored = true
            part.BrickColor = BrickColor.new("Bright red")
            part.Parent = workspace -- parent LAST after all properties set
        """),
    },
    {
        "category_tag": "instance_new_parent",
        "difficulty": 3,
        "user": (
            "My UI builder creates frames with the parent in the constructor. "
            "Can you fix it?\n\n"
            "```lua\n"
            "local function createShopItem(parent, itemName, price)\n"
            '\tlocal frame = Instance.new("Frame", parent)\n'
            '\tlocal nameLabel = Instance.new("TextLabel", frame)\n'
            "\tnameLabel.Text = itemName\n"
            "\tnameLabel.Size = UDim2.new(1, 0, 0.5, 0)\n"
            '\tlocal priceLabel = Instance.new("TextLabel", frame)\n'
            '\tpriceLabel.Text = "$" .. price\n'
            "\tpriceLabel.Size = UDim2.new(1, 0, 0.5, 0)\n"
            "\tpriceLabel.Position = UDim2.new(0, 0, 0.5, 0)\n"
            "\treturn frame\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "Multiple Instance.new calls with parent in constructor. Each child is "
            "parented immediately, causing replication churn. Build the entire tree "
            "off-screen and parent the root last."
        ),
        "explanation": (
            "Every `Instance.new(class, parent)` triggers immediate replication. When "
            "building UI trees, construct everything off-screen first, then parent the "
            "root frame last. This batches all changes into a single replication event."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local function createShopItem(parent: GuiObject, itemName: string, price: number): Frame
            \tlocal frame = Instance.new("Frame")
            \tframe.Size = UDim2.new(1, 0, 0, 60)

            \tlocal nameLabel = Instance.new("TextLabel")
            \tnameLabel.Text = itemName
            \tnameLabel.Size = UDim2.new(1, 0, 0.5, 0)
            \tnameLabel.Parent = frame -- parent to frame (not yet in tree)

            \tlocal priceLabel = Instance.new("TextLabel")
            \tpriceLabel.Text = "$" .. tostring(price)
            \tpriceLabel.Size = UDim2.new(1, 0, 0.5, 0)
            \tpriceLabel.Position = UDim2.new(0, 0, 0.5, 0)
            \tpriceLabel.Parent = frame

            \tframe.Parent = parent -- parent root LAST — single replication
            \treturn frame
            end
        """),
    },
    # -----------------------------------------------------------------------
    # 14. while wait() do → while true do task.wait()
    # -----------------------------------------------------------------------
    {
        "category_tag": "while_wait",
        "difficulty": 2,
        "user": (
            "I've been told `while wait() do` is an antipattern. Can you fix it?\n\n"
            "```lua\n"
            "while wait(1) do\n"
            "\tfor _, player in game.Players:GetPlayers() do\n"
            "\t\tupdatePlayerUI(player)\n"
            "\tend\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "while wait() do is a legacy pattern. wait() is deprecated and using it "
            "as the loop condition is confusing — it always returns a truthy value. "
            "Modern pattern is while true do with task.wait() inside."
        ),
        "explanation": (
            "`while wait() do` is a legacy antipattern. `wait()` is deprecated and using "
            "it as the loop condition obscures intent. The modern pattern is "
            "`while true do ... task.wait(n) end` — clearer and uses the modern scheduler."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Players = game:GetService("Players")

            while true do
            \tfor _, player in Players:GetPlayers() do
            \t\tupdatePlayerUI(player)
            \tend
            \ttask.wait(1)
            end
        """),
    },
    {
        "category_tag": "while_wait",
        "difficulty": 3,
        "user": (
            "My game loop uses old patterns. Can you modernize it?\n\n"
            "```lua\n"
            "spawn(function()\n"
            "\twhile wait() do\n"
            "\t\tcheckRoundStatus()\n"
            "\tend\n"
            "end)\n\n"
            "spawn(function()\n"
            "\twhile wait(5) do\n"
            "\t\tsaveAllData()\n"
            "\tend\n"
            "end)\n"
            "```"
        ),
        "thinking": (
            "Double deprecated: spawn() and wait(). Replace spawn with task.spawn "
            "and while wait() do with while true do task.wait()."
        ),
        "explanation": (
            "Both `spawn()` and `wait()` are deprecated. Use `task.spawn()` and "
            "`task.wait()`. Also, `while wait()` with no argument waits ~0.03s "
            "(one frame) which may be too fast — consider if you really need per-frame "
            "checks or if a longer interval is appropriate."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            task.spawn(function()
            \twhile true do
            \t\tcheckRoundStatus()
            \t\ttask.wait(1) -- check once per second, not every frame
            \tend
            end)

            task.spawn(function()
            \twhile true do
            \t\tsaveAllData()
            \t\ttask.wait(5)
            \tend
            end)
        """),
    },
    # -----------------------------------------------------------------------
    # 15. Not cleaning up on PlayerRemoving
    # -----------------------------------------------------------------------
    {
        "category_tag": "player_removing_cleanup",
        "difficulty": 3,
        "user": (
            "Players complain about data loss when they leave. Can you fix it?\n\n"
            "```lua\n"
            "local playerData = {}\n\n"
            "game.Players.PlayerAdded:Connect(function(player)\n"
            "\tplayerData[player] = loadData(player)\n"
            "end)\n\n"
            "-- No PlayerRemoving handler!\n"
            "```"
        ),
        "thinking": (
            "No PlayerRemoving handler means data is never saved when players leave. "
            "Also the playerData table grows forever since entries are never removed, "
            "causing a memory leak. Need to save data and clean up the table entry."
        ),
        "explanation": (
            "Without a `PlayerRemoving` handler, data is never saved when players leave, "
            "and the `playerData` table grows forever (memory leak). You also need a "
            "`game:BindToClose` handler for server shutdowns."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Players = game:GetService("Players")
            local playerData: { [Player]: { [string]: any } } = {}

            Players.PlayerAdded:Connect(function(player: Player)
            \tplayerData[player] = loadData(player)
            end)

            Players.PlayerRemoving:Connect(function(player: Player)
            \tlocal data = playerData[player]
            \tif data then
            \t\tsaveData(player, data)
            \t\tplayerData[player] = nil -- prevent memory leak
            \tend
            end)

            -- Save all data on server shutdown
            game:BindToClose(function()
            \tlocal threads: { thread } = {}
            \tfor player, data in playerData do
            \t\tlocal t = task.spawn(function()
            \t\t\tsaveData(player, data)
            \t\tend)
            \t\ttable.insert(threads, t)
            \tend
            \t-- BindToClose gives ~30s before forced shutdown
            \ttask.wait(5)
            end)
        """),
    },
    {
        "category_tag": "player_removing_cleanup",
        "difficulty": 3,
        "user": (
            "My pet system spawns pets but they stay in workspace after the player "
            "leaves. Can you fix it?\n\n"
            "```lua\n"
            "local activePets = {}\n\n"
            "local function spawnPet(player)\n"
            '\tlocal pet = Instance.new("Model")\n'
            '\tpet.Name = player.Name .. "_Pet"\n'
            "\tpet.Parent = workspace\n"
            "\tactivePets[player] = pet\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "When the player leaves, the pet model stays in workspace and the "
            "activePets entry is never cleaned up. Need PlayerRemoving to destroy "
            "the pet and remove the table entry."
        ),
        "explanation": (
            "Pet models persist in workspace after the player leaves, wasting server "
            "resources and creating visual artifacts. Clean up pets in `PlayerRemoving` "
            "and nil out the table entry to prevent memory leaks."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Players = game:GetService("Players")
            local activePets: { [Player]: Model } = {}

            local function spawnPet(player: Player)
            \tlocal pet = Instance.new("Model")
            \tpet.Name = player.Name .. "_Pet"
            \t-- Set up pet properties before parenting
            \tpet.Parent = workspace
            \tactivePets[player] = pet
            end

            local function cleanupPet(player: Player)
            \tlocal pet = activePets[player]
            \tif pet then
            \t\tpet:Destroy()
            \t\tactivePets[player] = nil
            \tend
            end

            Players.PlayerRemoving:Connect(cleanupPet)
        """),
    },
    {
        "category_tag": "player_removing_cleanup",
        "difficulty": 4,
        "user": (
            "My game tracks per-player connections and timers but leaks them on leave. "
            "Can you fix it?\n\n"
            "```lua\n"
            "local playerConnections = {}\n\n"
            "game.Players.PlayerAdded:Connect(function(player)\n"
            "\tlocal conns = {}\n"
            "\ttable.insert(conns, RunService.Heartbeat:Connect(function(dt)\n"
            "\t\tupdatePlayerEffects(player, dt)\n"
            "\tend))\n"
            "\ttable.insert(conns, player.Chatted:Connect(function(msg)\n"
            "\t\tprocessCommand(player, msg)\n"
            "\tend))\n"
            "\tplayerConnections[player] = conns\n"
            "end)\n"
            "```"
        ),
        "thinking": (
            "Connections are stored but never disconnected. PlayerRemoving needs to "
            "disconnect all connections and clean up the table. This is a common "
            "pattern that Trove solves elegantly."
        ),
        "explanation": (
            "Connections accumulate but are never cleaned up when players leave. "
            "Each leaks a Heartbeat listener that runs forever. Disconnect all "
            "connections in `PlayerRemoving` and clear the table entry."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Players = game:GetService("Players")
            local RunService = game:GetService("RunService")

            local playerConnections: { [Player]: { RBXScriptConnection } } = {}

            Players.PlayerAdded:Connect(function(player: Player)
            \tlocal conns: { RBXScriptConnection } = {}

            \ttable.insert(conns, RunService.Heartbeat:Connect(function(dt: number)
            \t\tupdatePlayerEffects(player, dt)
            \tend))

            \ttable.insert(conns, player.Chatted:Connect(function(msg: string)
            \t\tprocessCommand(player, msg)
            \tend))

            \tplayerConnections[player] = conns
            end)

            Players.PlayerRemoving:Connect(function(player: Player)
            \tlocal conns = playerConnections[player]
            \tif conns then
            \t\tfor _, conn in conns do
            \t\t\tconn:Disconnect()
            \t\tend
            \t\tplayerConnections[player] = nil
            \tend
            end)
        """),
    },
    # -----------------------------------------------------------------------
    # Additional pairs to reach 50+
    # -----------------------------------------------------------------------
    # More nil reference
    {
        "category_tag": "nil_reference",
        "difficulty": 3,
        "user": (
            "My raycast hit detection crashes randomly. Can you fix it?\n\n"
            "```lua\n"
            "local function shootRay(origin, direction)\n"
            "\tlocal result = workspace:Raycast(origin, direction)\n"
            "\tlocal hitPart = result.Instance\n"
            "\tlocal hitPos = result.Position\n"
            "\treturn hitPart, hitPos\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "workspace:Raycast returns nil if it hits nothing. Indexing nil.Instance "
            "errors. Need to nil-check the result."
        ),
        "explanation": (
            "`workspace:Raycast()` returns `nil` if the ray doesn't hit anything. "
            "Indexing `.Instance` on `nil` crashes. Always check the result before "
            "accessing its properties."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local function shootRay(origin: Vector3, direction: Vector3): (BasePart?, Vector3?)
            \tlocal result = workspace:Raycast(origin, direction)
            \tif not result then
            \t\treturn nil, nil
            \tend
            \treturn result.Instance, result.Position
            end
        """),
    },
    # More off-by-one
    {
        "category_tag": "off_by_one",
        "difficulty": 3,
        "user": (
            "My ring buffer wraps incorrectly. Can you fix it?\n\n"
            "```lua\n"
            "local BUFFER_SIZE = 10\n"
            "local buffer = table.create(BUFFER_SIZE)\n"
            "local writeIndex = 0\n\n"
            "local function push(value)\n"
            "\twriteIndex = writeIndex + 1\n"
            "\tif writeIndex > BUFFER_SIZE then\n"
            "\t\twriteIndex = 0\n"
            "\tend\n"
            "\tbuffer[writeIndex] = value\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "Two issues: writeIndex = 0 is invalid for 1-indexed Luau arrays, and "
            "the wrap logic should reset to 1, not 0. Also the modulo approach is "
            "cleaner and less error-prone."
        ),
        "explanation": (
            "Luau arrays are 1-indexed. Resetting `writeIndex` to 0 writes to an "
            "invalid index. The wrap should reset to 1. Using modulo arithmetic "
            "is cleaner and avoids this class of error entirely."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local BUFFER_SIZE = 10
            local buffer = table.create(BUFFER_SIZE)
            local writeIndex = 0

            local function push(value: any)
            \twriteIndex = (writeIndex % BUFFER_SIZE) + 1
            \tbuffer[writeIndex] = value
            end
        """),
    },
    # More race condition
    {
        "category_tag": "race_condition",
        "difficulty": 4,
        "user": (
            "My matchmaking system has a race condition where two players can "
            "be matched with the same opponent. Can you fix it?\n\n"
            "```lua\n"
            "local queue = {}\n\n"
            "local function joinQueue(player)\n"
            "\ttable.insert(queue, player)\n"
            "\tif #queue >= 2 then\n"
            "\t\tlocal p1 = queue[1]\n"
            "\t\tlocal p2 = queue[2]\n"
            "\t\ttable.remove(queue, 1)\n"
            "\t\ttable.remove(queue, 1)\n"
            "\t\tstartMatch(p1, p2)\n"
            "\tend\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "If two players join nearly simultaneously, both trigger the >= 2 check "
            "and could start duplicate matches. In Roblox's single-threaded model "
            "this is less likely, but the real bug is the double table.remove — after "
            "the first remove(queue, 1), the indices shift, so the second remove "
            "gets the right element but if queue had exactly 2 elements, the second "
            "remove operates on a now-1-element array. Actually the logic works but "
            "is fragile. The bigger issue is that players who disconnect between "
            "being queued and matched aren't handled. Let me focus on robustness."
        ),
        "explanation": (
            "The queue manipulation is fragile — removing by index while iterating "
            "causes subtle shifts. Also, players who leave between queuing and matching "
            "aren't validated. Use a lock flag to prevent re-entrance and validate "
            "players are still in the game before starting a match."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Players = game:GetService("Players")
            local queue: { Player } = {}
            local matching = false

            local function isValidPlayer(player: Player): boolean
            \treturn player.Parent ~= nil -- still in game
            end

            local function tryMatch()
            \tif matching then
            \t\treturn
            \tend
            \tmatching = true

            \t-- Remove invalid players
            \tlocal i = #queue
            \twhile i >= 1 do
            \t\tif not isValidPlayer(queue[i]) then
            \t\t\ttable.remove(queue, i)
            \t\tend
            \t\ti -= 1
            \tend

            \twhile #queue >= 2 do
            \t\tlocal p1 = table.remove(queue, 1) :: Player
            \t\tlocal p2 = table.remove(queue, 1) :: Player

            \t\tif isValidPlayer(p1) and isValidPlayer(p2) then
            \t\t\tstartMatch(p1, p2)
            \t\telse
            \t\t\t-- Re-queue valid player
            \t\t\tif isValidPlayer(p1) then
            \t\t\t\ttable.insert(queue, 1, p1)
            \t\t\tend
            \t\t\tif isValidPlayer(p2) then
            \t\t\t\ttable.insert(queue, 1, p2)
            \t\t\tend
            \t\tend
            \tend

            \tmatching = false
            end

            local function joinQueue(player: Player)
            \ttable.insert(queue, player)
            \ttryMatch()
            end

            Players.PlayerRemoving:Connect(function(player: Player)
            \tlocal idx = table.find(queue, player)
            \tif idx then
            \t\ttable.remove(queue, idx)
            \tend
            end)
        """),
    },
    # More memory leak
    {
        "category_tag": "memory_leak",
        "difficulty": 4,
        "user": (
            "My CollectionService tag handler leaks connections when tagged "
            "instances are removed. Can you fix it?\n\n"
            "```lua\n"
            'local CollectionService = game:GetService("CollectionService")\n'
            'local RunService = game:GetService("RunService")\n\n'
            'for _, part in CollectionService:GetTagged("Spinner") do\n'
            "\tRunService.Heartbeat:Connect(function(dt)\n"
            "\t\tpart.CFrame *= CFrame.Angles(0, math.rad(90) * dt, 0)\n"
            "\tend)\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "Three issues: 1) Only handles existing tagged instances, not new ones. "
            "2) Heartbeat connections are never disconnected when parts are removed. "
            "3) No handling for instances added/removed dynamically."
        ),
        "explanation": (
            "This only handles instances tagged at script start, not dynamically added "
            "ones. Heartbeat connections are never cleaned up when parts are destroyed. "
            "Use `GetInstanceAddedSignal`/`GetInstanceRemovedSignal` and track connections."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local CollectionService = game:GetService("CollectionService")
            local RunService = game:GetService("RunService")

            local TAG = "Spinner"
            local connections: { [Instance]: RBXScriptConnection } = {}

            local function onSpinnerAdded(part: Instance)
            \tif connections[part] then
            \t\treturn -- already tracked
            \tend
            \tconnections[part] = RunService.Heartbeat:Connect(function(dt: number)
            \t\tif not part.Parent then
            \t\t\t-- Part was destroyed, clean up
            \t\t\tlocal conn = connections[part]
            \t\t\tif conn then
            \t\t\t\tconn:Disconnect()
            \t\t\t\tconnections[part] = nil
            \t\t\tend
            \t\t\treturn
            \t\tend
            \t\t(part :: BasePart).CFrame *= CFrame.Angles(0, math.rad(90) * dt, 0)
            \tend)
            end

            local function onSpinnerRemoved(part: Instance)
            \tlocal conn = connections[part]
            \tif conn then
            \t\tconn:Disconnect()
            \t\tconnections[part] = nil
            \tend
            end

            -- Handle existing + future tagged instances
            for _, part in CollectionService:GetTagged(TAG) do
            \tonSpinnerAdded(part)
            end
            CollectionService:GetInstanceAddedSignal(TAG):Connect(onSpinnerAdded)
            CollectionService:GetInstanceRemovedSignal(TAG):Connect(onSpinnerRemoved)
        """),
    },
    # More type error
    {
        "category_tag": "type_error",
        "difficulty": 3,
        "user": (
            "My color lerp doesn't work — I get 'invalid argument'. Can you fix it?\n\n"
            "```lua\n"
            "local function lerpColor(c1, c2, t)\n"
            "\treturn Color3.new(\n"
            "\t\tc1.R + (c2.R - c1.R) * t,\n"
            "\t\tc1.G + (c2.G - c1.G) * t,\n"
            "\t\tc1.B + (c2.B - c1.B) * t\n"
            "\t)\n"
            "end\n\n"
            'local result = lerpColor("#FF0000", "#00FF00", 0.5)\n'
            "```"
        ),
        "thinking": (
            "Passing hex strings to a function that expects Color3 objects. Color3 "
            "doesn't auto-convert from strings. Also, Color3 has a built-in :Lerp method."
        ),
        "explanation": (
            "You're passing hex strings where `Color3` values are expected. Luau doesn't "
            "auto-convert strings to colors. Use `Color3.fromHex()` to parse hex strings, "
            "and use the built-in `Color3:Lerp()` method instead of manual interpolation."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local function lerpColor(c1: Color3, c2: Color3, t: number): Color3
            \treturn c1:Lerp(c2, t)
            end

            local color1 = Color3.fromHex("#FF0000")
            local color2 = Color3.fromHex("#00FF00")
            local result = lerpColor(color1, color2, 0.5)
        """),
    },
    # More datastore
    {
        "category_tag": "datastore_throttling",
        "difficulty": 4,
        "user": (
            "My OrderedDataStore leaderboard query sometimes fails. Can you fix it?\n\n"
            "```lua\n"
            'local DataStoreService = game:GetService("DataStoreService")\n'
            'local leaderboard = DataStoreService:GetOrderedDataStore("Wins")\n\n'
            "local function getTopPlayers(count)\n"
            "\tlocal pages = leaderboard:GetSortedAsync(false, count)\n"
            "\tlocal data = pages:GetCurrentPage()\n"
            "\tlocal results = {}\n"
            "\tfor _, entry in data do\n"
            "\t\ttable.insert(results, { key = entry.key, value = entry.value })\n"
            "\tend\n"
            "\treturn results\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "GetSortedAsync is a yielding call that can fail. No pcall wrapping. "
            "Also no caching — hitting the DataStore every call risks throttling."
        ),
        "explanation": (
            "`GetSortedAsync` is a yielding call that can fail from throttling or "
            "service outages. Wrap in `pcall` and add caching to reduce DataStore "
            "requests."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local DataStoreService = game:GetService("DataStoreService")
            local leaderboard = DataStoreService:GetOrderedDataStore("Wins")

            local CACHE_DURATION = 60
            local cachedResults: { { key: string, value: number } }? = nil
            local lastFetchTime = 0

            local function getTopPlayers(count: number): { { key: string, value: number } }
            \t-- Return cache if fresh
            \tif cachedResults and (os.clock() - lastFetchTime) < CACHE_DURATION then
            \t\treturn cachedResults
            \tend

            \tlocal ok, pages = pcall(function()
            \t\treturn leaderboard:GetSortedAsync(false, count)
            \tend)

            \tif not ok then
            \t\twarn("[Leaderboard] GetSortedAsync failed:", pages)
            \t\treturn cachedResults or {} -- return stale cache or empty
            \tend

            \tlocal data = pages:GetCurrentPage()
            \tlocal results: { { key: string, value: number } } = {}
            \tfor _, entry in data do
            \t\ttable.insert(results, { key = entry.key, value = entry.value })
            \tend

            \tcachedResults = results
            \tlastFetchTime = os.clock()
            \treturn results
            end
        """),
    },
    # More deprecated API
    {
        "category_tag": "deprecated_api",
        "difficulty": 3,
        "user": (
            "My code uses the deprecated BodyVelocity for movement. Can you modernize it?\n\n"
            "```lua\n"
            "local function launchCharacter(rootPart, direction)\n"
            '\tlocal bv = Instance.new("BodyVelocity")\n'
            "\tbv.MaxForce = Vector3.new(math.huge, math.huge, math.huge)\n"
            "\tbv.Velocity = direction * 100\n"
            "\tbv.Parent = rootPart\n"
            "\tgame.Debris:AddItem(bv, 0.5)\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "BodyVelocity is deprecated. Modern approach uses LinearVelocity constraint "
            'with an Attachment. Also game.Debris should use game:GetService("Debris").'
        ),
        "explanation": (
            "`BodyVelocity` is deprecated in favor of `LinearVelocity` constraints. "
            "The new constraints-based approach uses an `Attachment` and provides better "
            'physics behavior. Also use `game:GetService("Debris")` instead of '
            "`game.Debris`."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Debris = game:GetService("Debris")

            local function launchCharacter(rootPart: BasePart, direction: Vector3)
            \tlocal attachment = Instance.new("Attachment")
            \tattachment.Parent = rootPart

            \tlocal linearVelocity = Instance.new("LinearVelocity")
            \tlinearVelocity.Attachment0 = attachment
            \tlinearVelocity.VectorVelocity = direction.Unit * 100
            \tlinearVelocity.MaxForce = math.huge
            \tlinearVelocity.RelativeTo = Enum.ActuatorRelativeTo.World
            \tlinearVelocity.Parent = rootPart

            \tDebris:AddItem(attachment, 0.5)
            \tDebris:AddItem(linearVelocity, 0.5)
            end
        """),
    },
    # More infinite yield
    {
        "category_tag": "infinite_yield",
        "difficulty": 3,
        "user": (
            "My module loader yields forever when a module doesn't exist. "
            "Can you fix it?\n\n"
            "```lua\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            "local function loadModule(moduleName)\n"
            '\tlocal module = ReplicatedStorage:WaitForChild("Modules"):WaitForChild(moduleName)\n'
            "\treturn require(module)\n"
            "end\n\n"
            'local CombatModule = loadModule("CombatSystem")\n'
            'local PetModule = loadModule("PetSystem")\n'
            'local nonExistent = loadModule("DoesNotExist") -- yields forever\n'
            "```"
        ),
        "thinking": (
            "Chained WaitForChild with no timeouts. If Modules or the named module "
            "doesn't exist, it yields forever. Need timeouts and error handling."
        ),
        "explanation": (
            "Chained `WaitForChild` without timeouts yields forever if any link in the "
            "chain doesn't exist. Always add timeouts and handle the `nil` case. Consider "
            "returning a sentinel or erroring explicitly rather than yielding indefinitely."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local ReplicatedStorage = game:GetService("ReplicatedStorage")

            local TIMEOUT = 10

            local function loadModule(moduleName: string): any?
            \tlocal modulesFolder = ReplicatedStorage:WaitForChild("Modules", TIMEOUT)
            \tif not modulesFolder then
            \t\twarn("[ModuleLoader] 'Modules' folder not found after", TIMEOUT, "seconds")
            \t\treturn nil
            \tend

            \tlocal moduleInstance = modulesFolder:WaitForChild(moduleName, TIMEOUT)
            \tif not moduleInstance then
            \t\twarn("[ModuleLoader] Module '" .. moduleName .. "' not found after", TIMEOUT, "seconds")
            \t\treturn nil
            \tend

            \tlocal ok, result = pcall(require, moduleInstance)
            \tif not ok then
            \t\twarn("[ModuleLoader] Failed to require '" .. moduleName .. "':", result)
            \t\treturn nil
            \tend

            \treturn result
            end

            local CombatModule = loadModule("CombatSystem")
            local PetModule = loadModule("PetSystem")
            local nonExistent = loadModule("DoesNotExist") -- returns nil after timeout
        """),
    },
    # More client/server
    {
        "category_tag": "client_server_boundary",
        "difficulty": 4,
        "user": (
            "My RemoteEvent handler doesn't validate types from the client. "
            "Exploiters can crash the server. Can you fix it?\n\n"
            "```lua\n"
            "-- Server Script\n"
            "local remote = ReplicatedStorage.Remotes.DamageEntity\n\n"
            "remote.OnServerEvent:Connect(function(player, targetId, damage)\n"
            "\tlocal target = entityRegistry[targetId]\n"
            "\ttarget.Health -= damage\n"
            "\tif target.Health <= 0 then\n"
            "\t\ttarget:Destroy()\n"
            "\tend\n"
            "end)\n"
            "```"
        ),
        "thinking": (
            "Exploiters can send any types via RemoteEvents. targetId could be a table, "
            "damage could be negative (healing) or huge. No type validation, no range "
            "validation, no cooldown/rate limiting."
        ),
        "explanation": (
            "Exploiters can send any data type through RemoteEvents. `targetId` could be "
            "a table, `damage` could be negative (healing) or enormous. Always validate "
            "types, ranges, and add rate limiting on the server."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local ReplicatedStorage = game:GetService("ReplicatedStorage")
            local remote = ReplicatedStorage:WaitForChild("Remotes"):WaitForChild("DamageEntity")

            local MAX_DAMAGE = 100
            local COOLDOWN = 0.5
            local lastAttack: { [Player]: number } = {}

            remote.OnServerEvent:Connect(function(player: Player, targetId: unknown, damage: unknown)
            \t-- Type validation
            \tif typeof(targetId) ~= "string" then
            \t\twarn("[Security] Invalid targetId type from", player.Name)
            \t\treturn
            \tend
            \tif typeof(damage) ~= "number" then
            \t\twarn("[Security] Invalid damage type from", player.Name)
            \t\treturn
            \tend

            \t-- Range validation
            \tif damage <= 0 or damage > MAX_DAMAGE or damage ~= damage then -- NaN check
            \t\twarn("[Security] Invalid damage value from", player.Name, ":", damage)
            \t\treturn
            \tend

            \t-- Rate limiting
            \tlocal now = os.clock()
            \tif lastAttack[player] and (now - lastAttack[player]) < COOLDOWN then
            \t\treturn
            \tend
            \tlastAttack[player] = now

            \t-- Entity validation
            \tlocal target = entityRegistry[targetId :: string]
            \tif not target then
            \t\treturn
            \tend

            \t-- Apply server-calculated damage (don't trust client damage value)
            \tlocal serverDamage = math.min(damage :: number, MAX_DAMAGE)
            \ttarget.Health -= serverDamage
            \tif target.Health <= 0 then
            \t\ttarget:Destroy()
            \tend
            end)

            -- Clean up rate limit entries
            game:GetService("Players").PlayerRemoving:Connect(function(player: Player)
            \tlastAttack[player] = nil
            end)
        """),
    },
    # More missing pcall
    {
        "category_tag": "missing_pcall",
        "difficulty": 3,
        "user": (
            "My TeleportService call crashes when teleporting fails. Can you fix it?\n\n"
            "```lua\n"
            'local TeleportService = game:GetService("TeleportService")\n\n'
            "local function teleportPlayer(player, placeId)\n"
            "\tTeleportService:Teleport(placeId, player)\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "Teleport can fail for many reasons: invalid place ID, player disconnected "
            "mid-teleport, Roblox service issues. Needs pcall and retry."
        ),
        "explanation": (
            "`TeleportService:Teleport` can fail from invalid place IDs, floodchecks, "
            "player disconnects, or Roblox service issues. Wrap in `pcall` and implement "
            "retry logic with `TeleportService.TeleportInitFailed` for automatic retries."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local TeleportService = game:GetService("TeleportService")

            local MAX_RETRIES = 3

            local function teleportPlayer(player: Player, placeId: number): boolean
            \tfor attempt = 1, MAX_RETRIES do
            \t\tlocal ok, err = pcall(function()
            \t\t\tTeleportService:Teleport(placeId, player)
            \t\tend)

            \t\tif ok then
            \t\t\treturn true
            \t\tend

            \t\twarn(string.format(
            \t\t\t"[Teleport] Failed for %s to %d (attempt %d/%d): %s",
            \t\t\tplayer.Name, placeId, attempt, MAX_RETRIES, tostring(err)
            \t\t))

            \t\tif attempt < MAX_RETRIES then
            \t\t\ttask.wait(2 * attempt)
            \t\tend
            \tend

            \twarn("[Teleport] All retries exhausted for", player.Name)
            \treturn false
            end
        """),
    },
    # More while wait
    {
        "category_tag": "while_wait",
        "difficulty": 2,
        "user": (
            "My regen loop uses deprecated patterns. Can you fix it?\n\n"
            "```lua\n"
            "while wait(1) do\n"
            "\tlocal character = player.Character\n"
            "\tif character then\n"
            '\t\tlocal humanoid = character:FindFirstChild("Humanoid")\n'
            "\t\tif humanoid and humanoid.Health < humanoid.MaxHealth then\n"
            "\t\t\thumanoid.Health = humanoid.Health + 5\n"
            "\t\tend\n"
            "\tend\n"
            "end\n"
            "```"
        ),
        "thinking": ("while wait() do pattern. Replace with while true do + task.wait()."),
        "explanation": (
            "Replace deprecated `while wait(1) do` with `while true do` and `task.wait(1)`. "
            "Also use `math.min` to clamp health to `MaxHealth` instead of risking overheal."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local REGEN_RATE = 5
            local REGEN_INTERVAL = 1

            while true do
            \tlocal character = player.Character
            \tif character then
            \t\tlocal humanoid = character:FindFirstChildOfClass("Humanoid")
            \t\tif humanoid and humanoid.Health < humanoid.MaxHealth then
            \t\t\thumanoid.Health = math.min(humanoid.Health + REGEN_RATE, humanoid.MaxHealth)
            \t\tend
            \tend
            \ttask.wait(REGEN_INTERVAL)
            end
        """),
    },
    # Instance.new parent — bulk creation
    {
        "category_tag": "instance_new_parent",
        "difficulty": 3,
        "user": (
            "My particle spawner creates effects in a loop with parent in constructor. "
            "Can you fix it?\n\n"
            "```lua\n"
            "local function createExplosion(position, count)\n"
            "\tfor i = 1, count do\n"
            '\t\tlocal p = Instance.new("Part", workspace)\n'
            "\t\tp.Size = Vector3.new(0.5, 0.5, 0.5)\n"
            "\t\tp.Position = position\n"
            "\t\tp.BrickColor = BrickColor.random()\n"
            "\t\tp.Anchored = false\n"
            "\t\tgame.Debris:AddItem(p, 3)\n"
            "\tend\n"
            "end\n"
            "```"
        ),
        "thinking": (
            "Instance.new with parent in a loop means each part replicates immediately "
            "then gets property updates. For many parts this is very wasteful. "
            "Also, Instance.new in a hot loop is a performance problem. Should pool "
            "parts or at least set parent last."
        ),
        "explanation": (
            "Creating instances with parent in a loop causes N replication events "
            "followed by N*properties update events. Set parent last. For effects like "
            "this, consider pooling parts in `:Init()` instead of creating new ones."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Debris = game:GetService("Debris")

            local function createExplosion(position: Vector3, count: number)
            \tlocal folder = Instance.new("Folder")
            \tfolder.Name = "Explosion"

            \tfor _ = 1, count do
            \t\tlocal p = Instance.new("Part")
            \t\tp.Size = Vector3.new(0.5, 0.5, 0.5)
            \t\tp.Position = position
            \t\tp.BrickColor = BrickColor.random()
            \t\tp.Anchored = false
            \t\tp.CanCollide = false
            \t\t-- Apply random impulse
            \t\tlocal rng = Random.new()
            \t\tlocal dir = Vector3.new(
            \t\t\trng:NextNumber(-1, 1),
            \t\t\trng:NextNumber(0.5, 1),
            \t\t\trng:NextNumber(-1, 1)
            \t\t).Unit
            \t\tp:ApplyImpulse(dir * rng:NextNumber(20, 50))
            \t\tp.Parent = folder -- parent to folder (not yet in workspace)
            \tend

            \tfolder.Parent = workspace -- single replication event
            \tDebris:AddItem(folder, 3)
            end
        """),
    },
    # Network ownership — vehicles
    {
        "category_tag": "network_ownership",
        "difficulty": 4,
        "user": (
            "My vehicle seat driver script jitters because of ownership fights. "
            "Can you fix it?\n\n"
            "```lua\n"
            "local seat = script.Parent\n\n"
            'seat:GetPropertyChangedSignal("Occupant"):Connect(function()\n'
            "\tlocal humanoid = seat.Occupant\n"
            "\tif humanoid then\n"
            "\t\t-- Player sat down, start driving\n"
            "\t\tlocal character = humanoid.Parent\n"
            '\t\tprint(character.Name, "is driving")\n'
            "\tend\n"
            "end)\n"
            "```"
        ),
        "thinking": (
            "When a player sits in the seat, network ownership should transfer to them "
            "for smooth driving. The vehicle parts need SetNetworkOwner set to the "
            "driver. When they leave, it should revert to server."
        ),
        "explanation": (
            "Vehicles jitter when the server and client fight over physics ownership. "
            "Transfer network ownership of all vehicle parts to the seated player for "
            "smooth client-side physics, and revert to server ownership when they exit."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Players = game:GetService("Players")
            local seat = script.Parent :: VehicleSeat
            local vehicleModel = seat.Parent :: Model

            local function setVehicleOwner(owner: Player?)
            \tfor _, part in vehicleModel:GetDescendants() do
            \t\tif part:IsA("BasePart") then
            \t\t\tpart:SetNetworkOwner(owner)
            \t\tend
            \tend
            end

            seat:GetPropertyChangedSignal("Occupant"):Connect(function()
            \tlocal humanoid = seat.Occupant
            \tif humanoid then
            \t\tlocal character = humanoid.Parent :: Model
            \t\tlocal player = Players:GetPlayerFromCharacter(character)
            \t\tif player then
            \t\t\tsetVehicleOwner(player)
            \t\t\tprint(player.Name, "is driving")
            \t\tend
            \telse
            \t\t-- No driver, revert to server ownership
            \t\tsetVehicleOwner(nil)
            \t\tprint("Vehicle unoccupied")
            \tend
            end)
        """),
    },
    # Closure-in-loop — object pooling
    {
        "category_tag": "closure_in_loop",
        "difficulty": 4,
        "user": (
            "My Heartbeat loop creates closures every frame, hurting performance. "
            "Can you fix it?\n\n"
            "```lua\n"
            "RunService.Heartbeat:Connect(function(dt)\n"
            "\tfor _, entity in entities do\n"
            "\t\ttask.spawn(function()\n"
            "\t\t\tlocal pos = entity.Part.Position\n"
            "\t\t\tlocal target = entity.Target\n"
            "\t\t\tlocal dir = (target - pos).Unit\n"
            "\t\t\tentity.Part.CFrame = CFrame.new(pos + dir * entity.Speed * dt)\n"
            "\t\tend)\n"
            "\tend\n"
            "end)\n"
            "```"
        ),
        "thinking": (
            "Creating a closure for each entity every frame in Heartbeat is terrible "
            "for performance. The task.spawn is unnecessary — this isn't yielding. "
            "Extract the update to a @native function and call it directly."
        ),
        "explanation": (
            "Creating a closure per entity per frame in Heartbeat causes massive GC "
            "pressure. The `task.spawn` is also unnecessary since nothing here yields. "
            "Extract the update logic to a `@native` module-level function and call it "
            "directly for NCG optimization."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local RunService = game:GetService("RunService")

            @native
            local function updateEntity(entity: Entity, dt: number)
            \tlocal pos = entity.Part.Position
            \tlocal target = entity.Target
            \tlocal delta = target - pos
            \tlocal mag = delta.Magnitude
            \tif mag < 0.1 then
            \t\treturn
            \tend
            \tlocal dir = delta * (1 / mag) -- avoid .Unit allocation
            \tentity.Part.CFrame = CFrame.new(pos + dir * entity.Speed * dt)
            end

            RunService.Heartbeat:Connect(function(dt: number)
            \tfor _, entity in entities do
            \t\tupdateEntity(entity, dt)
            \tend
            end)
        """),
    },
    # Player removing — full lifecycle
    {
        "category_tag": "player_removing_cleanup",
        "difficulty": 4,
        "user": (
            "My game has multiple systems tracking per-player state but no centralized "
            "cleanup. Can you fix this pattern?\n\n"
            "```lua\n"
            "-- Different scripts each track their own player state\n"
            "local playerInventory = {}\n"
            "local playerQuests = {}\n"
            "local playerCooldowns = {}\n"
            "local playerVehicles = {}\n\n"
            "game.Players.PlayerAdded:Connect(function(player)\n"
            "\tplayerInventory[player] = {}\n"
            "\tplayerQuests[player] = {}\n"
            "\tplayerCooldowns[player] = {}\n"
            "end)\n"
            "-- No cleanup anywhere!\n"
            "```"
        ),
        "thinking": (
            "Multiple systems with no centralized cleanup. Need a cleanup registry "
            "pattern where each system registers its cleanup function."
        ),
        "explanation": (
            "Scattered per-player state with no cleanup leaks memory for every player "
            "who joins and leaves. Use a centralized cleanup registry where each system "
            "registers its cleanup callback, then run all of them in `PlayerRemoving`."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local Players = game:GetService("Players")

            -- Centralized cleanup registry
            type CleanupFn = (player: Player) -> ()
            local cleanupRegistry: { CleanupFn } = {}

            local function registerCleanup(fn: CleanupFn)
            \ttable.insert(cleanupRegistry, fn)
            end

            -- Per-system state
            local playerInventory: { [Player]: { any } } = {}
            local playerQuests: { [Player]: { any } } = {}
            local playerCooldowns: { [Player]: { [string]: number } } = {}
            local playerVehicles: { [Player]: Model? } = {}

            -- Register cleanup for each system
            registerCleanup(function(player: Player)
            \tplayerInventory[player] = nil
            end)

            registerCleanup(function(player: Player)
            \tplayerQuests[player] = nil
            end)

            registerCleanup(function(player: Player)
            \tplayerCooldowns[player] = nil
            end)

            registerCleanup(function(player: Player)
            \tlocal vehicle = playerVehicles[player]
            \tif vehicle then
            \t\tvehicle:Destroy()
            \tend
            \tplayerVehicles[player] = nil
            end)

            Players.PlayerAdded:Connect(function(player: Player)
            \tplayerInventory[player] = {}
            \tplayerQuests[player] = {}
            \tplayerCooldowns[player] = {}
            end)

            Players.PlayerRemoving:Connect(function(player: Player)
            \tfor _, cleanupFn in cleanupRegistry do
            \t\tlocal ok, err = pcall(cleanupFn, player)
            \t\tif not ok then
            \t\t\twarn("[Cleanup] Error cleaning up for", player.Name, ":", err)
            \t\tend
            \tend
            end)
        """),
    },
    # Missing pcall — badge service
    {
        "category_tag": "missing_pcall",
        "difficulty": 2,
        "user": (
            "My badge award crashes when the service is unavailable. Can you fix it?\n\n"
            "```lua\n"
            'local BadgeService = game:GetService("BadgeService")\n\n'
            "local function awardBadge(player, badgeId)\n"
            "\tif not BadgeService:UserHasBadgeAsync(player.UserId, badgeId) then\n"
            "\t\tBadgeService:AwardBadge(player.UserId, badgeId)\n"
            "\tend\n"
            "end\n"
            "```"
        ),
        "thinking": ("Both UserHasBadgeAsync and AwardBadge can fail. Neither is wrapped in pcall."),
        "explanation": (
            "Both `UserHasBadgeAsync` and `AwardBadge` contact Roblox servers and can "
            "fail. Without `pcall`, a service outage crashes your script thread. Wrap "
            "both calls."
        ),
        "fixed_code": textwrap.dedent("""\
            --!strict
            local BadgeService = game:GetService("BadgeService")

            local function awardBadge(player: Player, badgeId: number)
            \tlocal hasOk, hasBadge = pcall(function()
            \t\treturn BadgeService:UserHasBadgeAsync(player.UserId, badgeId)
            \tend)

            \tif not hasOk then
            \t\twarn("[Badge] Failed to check badge", badgeId, "for", player.Name, ":", hasBadge)
            \t\treturn
            \tend

            \tif hasBadge then
            \t\treturn -- already has it
            \tend

            \tlocal awardOk, awardErr = pcall(function()
            \t\tBadgeService:AwardBadge(player.UserId, badgeId)
            \tend)

            \tif not awardOk then
            \t\twarn("[Badge] Failed to award badge", badgeId, "to", player.Name, ":", awardErr)
            \tend
            end
        """),
    },
]


# ---------------------------------------------------------------------------
# Build JSONL
# ---------------------------------------------------------------------------


def build_example(pair: dict) -> dict:
    """Convert a bug/fix pair into a training example."""
    assistant_content = (
        f"<think>\n{pair['thinking']}\n</think>\n\n"
        f"{pair['explanation']}\n\n"
        f"Here's the fixed code:\n\n"
        f"```lua\n{pair['fixed_code'].rstrip()}\n```"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["user"]},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": "bug_fix_pairs",
        "category": "debugging",
        "difficulty": pair["difficulty"],
        "has_reasoning": True,
        "verified": False,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    examples = [build_example(pair) for pair in BUG_FIX_PAIRS]

    with OUTPUT.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Summary
    categories: dict[str, int] = {}
    for pair in BUG_FIX_PAIRS:
        tag = pair["category_tag"]
        categories[tag] = categories.get(tag, 0) + 1

    print(f"Wrote {len(examples)} bug/fix pairs to {OUTPUT}")
    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
