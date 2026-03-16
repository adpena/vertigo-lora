#!/usr/bin/env python3
from __future__ import annotations

"""
Generate Dataset C: critic/repair/preference training data.

Three types:
  1. Critique -> Rewrite pairs (category: critic)
  2. Coach data (category: critic)
  3. Pairwise preference (category: preference)

Output: data/raw/critic_repair.jsonl
"""

import json
import textwrap
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "critic_repair.jsonl"

CRITIC_SYSTEM = (
    "You are a senior Roblox developer reviewing Luau code for the Vertigo experience. "
    "Vertigo uses --!strict mode, @native on hot paths, Init/Start service lifecycle, "
    "CollectionService tags for object discovery, server-authoritative networking, and "
    "NCG-friendly patterns (vector.* SIMD, math.lerp, table.create, no closures in loops). "
    "When reviewing code, explain WHY each issue matters and provide both a minimal fix "
    "and a production-quality fix."
)

PREFERENCE_SYSTEM = (
    "You are a senior Roblox developer comparing two Luau solutions. "
    "Evaluate on: idiomatic Luau (--!strict, type annotations, @native), "
    "server-authoritative patterns, performance (NCG-friendly), error handling, "
    "and readability/maintainability. Provide reasoning before your verdict."
)

COACH_SYSTEM = (
    "You are a senior Roblox developer and mentor. When asked about a bug or pattern, "
    "explain at multiple levels: one-sentence summary, minimal fix, production fix, "
    "and the test that catches it. Use Vertigo conventions (--!strict, @native, Init/Start)."
)


# ---------------------------------------------------------------------------
# Provenance helper
# ---------------------------------------------------------------------------


def make_provenance(task_family: str) -> dict:
    return {
        "rights_basis": "generated",
        "task_family": task_family,
        "teacher_model": "hand-authored",
        "modality": "critique_rewrite" if task_family == "critic" else "pairwise",
    }


# ---------------------------------------------------------------------------
# Type 1: Critique -> Rewrite pairs
# ---------------------------------------------------------------------------

CRITIQUE_REWRITE: list[dict] = [
    # --- Non-idiomatic Luau ---
    {
        "tag": "non_idiomatic_luau",
        "difficulty": 2,
        "user_code": textwrap.dedent("""\
            local items = {1, 2, 3, 4, 5}
            for k, v in pairs(items) do
                print(string.format("Item %d: %d", k, v))
            end"""),
        "thinking": (
            "This uses Lua 5.1 patterns: pairs() on an array and string.format. "
            "Modern Luau has generalized iteration (just `for k, v in items`) and "
            "string interpolation with backticks. These aren't just style — the "
            "generalized for compiles to faster bytecode and interpolation avoids "
            "format string parsing overhead."
        ),
        "issue": "Uses legacy Lua 5.1 patterns instead of modern Luau idioms.",
        "minimal_fix": textwrap.dedent("""\
            local items = {1, 2, 3, 4, 5}
            for k, v in items do
                print(`Item {k}: {v}`)
            end"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local items: {number} = table.freeze({1, 2, 3, 4, 5})
            for k, v in items do
                print(`Item {k}: {v}`)
            end"""),
    },
    {
        "tag": "non_idiomatic_luau",
        "difficulty": 2,
        "user_code": textwrap.dedent("""\
            function calculateDamage(base, multiplier)
                local result = base * multiplier
                return result
            end"""),
        "thinking": (
            "No type annotations, no --!strict, uses global function declaration. "
            "Without types, NCG can't optimize this and the strict analyzer won't "
            "catch type mismatches at call sites."
        ),
        "issue": "Missing type annotations and strict mode, global function declaration.",
        "minimal_fix": textwrap.dedent("""\
            --!strict
            local function calculateDamage(base: number, multiplier: number): number
                return base * multiplier
            end"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local DamageUtil = {}

            @native
            function DamageUtil.calculate(base: number, multiplier: number): number
                return base * multiplier
            end

            return DamageUtil"""),
    },
    {
        "tag": "non_idiomatic_luau",
        "difficulty": 3,
        "user_code": textwrap.dedent("""\
            local name = "Player"
            local score = 100
            local msg = name .. " scored " .. tostring(score) .. " points!"
            local tbl = {}
            for i = 1, 50 do
                table.insert(tbl, i)
            end"""),
        "thinking": (
            "String concatenation with .. and tostring is the old Lua way. "
            "Luau has string interpolation. Also, table.insert in a loop with "
            "known size — should use table.create to pre-allocate."
        ),
        "issue": "String concatenation instead of interpolation, no table pre-allocation.",
        "minimal_fix": textwrap.dedent("""\
            local name = "Player"
            local score = 100
            local msg = `{name} scored {score} points!`
            local tbl = table.create(50)
            for i = 1, 50 do
                tbl[i] = i
            end"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local name: string = "Player"
            local score: number = 100
            local msg: string = `{name} scored {score} points!`
            local tbl: {number} = table.create(50)
            for i = 1, 50 do
                tbl[i] = i
            end"""),
    },
    {
        "tag": "non_idiomatic_luau",
        "difficulty": 3,
        "user_code": textwrap.dedent("""\
            local function findItem(list, target)
                for i = 1, #list do
                    if list[i] == target then
                        return i
                    end
                end
                return nil
            end"""),
        "thinking": (
            "This reimplements table.find which is a built-in Luau function. "
            "table.find is implemented in C and is faster than a Luau loop. "
            "Also missing type annotations."
        ),
        "issue": "Reimplements table.find, missing type annotations.",
        "minimal_fix": textwrap.dedent("""\
            local function findItem(list: {string}, target: string): number?
                return table.find(list, target)
            end"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            -- Just use table.find directly at call sites:
            local index: number? = table.find(inventory, "Sword")
            if index then
                -- found at position `index`
            end"""),
    },
    {
        "tag": "non_idiomatic_luau",
        "difficulty": 2,
        "user_code": textwrap.dedent("""\
            wait(2)
            local part = Instance.new("Part")
            part.Parent = workspace
            wait(1)
            part:Destroy()"""),
        "thinking": (
            "wait() is deprecated in favor of task.wait(). Also, Instance.new "
            "with deferred Parent setting is fine here but the deprecated wait() "
            "is the main issue. Modern Luau uses task.wait, task.spawn, task.defer."
        ),
        "issue": "Uses deprecated wait() instead of task.wait().",
        "minimal_fix": textwrap.dedent("""\
            task.wait(2)
            local part = Instance.new("Part")
            part.Parent = workspace
            task.wait(1)
            part:Destroy()"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            task.wait(2)
            local part = Instance.new("Part")
            part.Size = Vector3.new(4, 1, 4)
            part.Anchored = true
            part.Parent = workspace
            task.delay(1, function()
                part:Destroy()
            end)"""),
    },
    # --- Replication bugs ---
    {
        "tag": "replication_bug",
        "difficulty": 3,
        "user_code": textwrap.dedent("""\
            -- Client script
            local Players = game:GetService("Players")
            local player = Players.LocalPlayer

            local function takeDamage(amount)
                local character = player.Character
                if not character then return end
                local humanoid = character:FindFirstChildOfClass("Humanoid")
                if humanoid then
                    humanoid.Health -= amount
                end
            end"""),
        "thinking": (
            "The client is directly modifying Health. In a server-authoritative "
            "architecture, health changes MUST go through the server via a "
            "RemoteEvent. A client modifying Health locally will be overwritten "
            "by the server and is trivially exploitable."
        ),
        "issue": "Client directly modifies Health — must be server-authoritative.",
        "minimal_fix": textwrap.dedent("""\
            -- Client script
            local ReplicatedStorage = game:GetService("ReplicatedStorage")
            local Remotes = ReplicatedStorage:WaitForChild("Remotes")
            local RequestDamage = Remotes:WaitForChild("RequestDamage")

            local function takeDamage(amount: number)
                RequestDamage:FireServer(amount)
            end"""),
        "production_fix": textwrap.dedent("""\
            -- Server script (DamageService)
            --!strict
            local ReplicatedStorage = game:GetService("ReplicatedStorage")
            local Players = game:GetService("Players")

            local DamageService = {}

            function DamageService:Init()
                self._remotes = ReplicatedStorage:WaitForChild("Remotes")
            end

            function DamageService:Start()
                self._remotes.RequestDamage.OnServerEvent:Connect(function(player: Player, amount: unknown)
                    if typeof(amount) ~= "number" then return end
                    if amount < 0 or amount > 100 then return end

                    local character = player.Character
                    if not character then return end
                    local humanoid = character:FindFirstChildOfClass("Humanoid")
                    if not humanoid then return end

                    humanoid:TakeDamage(amount)
                end)
            end

            return DamageService"""),
    },
    {
        "tag": "replication_bug",
        "difficulty": 4,
        "user_code": textwrap.dedent("""\
            -- Server script
            local ReplicatedStorage = game:GetService("ReplicatedStorage")
            local SpawnRemote = ReplicatedStorage.Remotes.RequestSpawn

            SpawnRemote.OnServerEvent:Connect(function(player, position)
                local part = Instance.new("Part")
                part.Position = position
                part.Parent = workspace
            end)"""),
        "thinking": (
            "No input validation on the RemoteEvent. A malicious client can send "
            "any position (or non-Vector3 value) and the server blindly trusts it. "
            "Need to validate typeof, range-check the position, and rate-limit."
        ),
        "issue": "Missing RemoteEvent input validation — exploitable.",
        "minimal_fix": textwrap.dedent("""\
            SpawnRemote.OnServerEvent:Connect(function(player, position)
                if typeof(position) ~= "Vector3" then return end
                if position.Magnitude > 2000 then return end
                local part = Instance.new("Part")
                part.Position = position
                part.Parent = workspace
            end)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local SPAWN_RANGE = 50
            local COOLDOWN = 1
            local lastSpawn: {[Player]: number} = {}

            SpawnRemote.OnServerEvent:Connect(function(player: Player, position: unknown)
                -- Type validation
                if typeof(position) ~= "Vector3" then
                    warn(`[SpawnService] Invalid position type from {player.Name}`)
                    return
                end
                local pos = position :: Vector3

                -- Range validation (must be near the player)
                local character = player.Character
                if not character then return end
                local root = character:FindFirstChild("HumanoidRootPart") :: BasePart?
                if not root then return end
                if (pos - root.Position).Magnitude > SPAWN_RANGE then
                    warn(`[SpawnService] Position too far from {player.Name}`)
                    return
                end

                -- Rate limiting
                local now = os.clock()
                if lastSpawn[player] and now - lastSpawn[player] < COOLDOWN then
                    return
                end
                lastSpawn[player] = now

                local part = Instance.new("Part")
                part.Position = pos
                part.Anchored = true
                part.Parent = workspace
            end)"""),
    },
    {
        "tag": "replication_bug",
        "difficulty": 3,
        "user_code": textwrap.dedent("""\
            -- Client script
            local Players = game:GetService("Players")
            local player = Players.LocalPlayer

            -- Give player coins directly
            local function collectCoin(coinPart)
                player:SetAttribute("Coins", (player:GetAttribute("Coins") or 0) + 1)
                coinPart:Destroy()
            end"""),
        "thinking": (
            "Client is modifying a player attribute that likely represents currency. "
            "Attributes set on the client replicate to the server only if the client "
            "owns the instance, but for currency this is trivially exploitable. "
            "Coin collection must be server-authoritative."
        ),
        "issue": "Client directly modifies currency attribute — exploitable.",
        "minimal_fix": textwrap.dedent("""\
            local ReplicatedStorage = game:GetService("ReplicatedStorage")
            local CollectCoinRemote = ReplicatedStorage.Remotes.RequestCollectCoin

            local function collectCoin(coinPart: BasePart)
                CollectCoinRemote:FireServer(coinPart)
            end"""),
        "production_fix": textwrap.dedent("""\
            -- Server: CoinService.luau
            --!strict
            local CollectionService = game:GetService("CollectionService")
            local Players = game:GetService("Players")

            local CoinService = {}

            local collected: {[string]: {[Player]: boolean}} = {}

            function CoinService:Init()
                -- Pre-index all coins
                for _, coin in CollectionService:GetTagged("Coin") do
                    collected[coin:GetAttribute("CoinId") or coin.Name] = {}
                end
            end

            function CoinService:Start()
                local remotes = game:GetService("ReplicatedStorage"):WaitForChild("Remotes")
                remotes.RequestCollectCoin.OnServerEvent:Connect(function(player: Player, coinRef: unknown)
                    if typeof(coinRef) ~= "Instance" then return end
                    local coin = coinRef :: BasePart
                    if not coin:HasTag("Coin") then return end

                    local coinId = coin:GetAttribute("CoinId") or coin.Name
                    if collected[coinId] and collected[coinId][player] then return end

                    -- Proximity check
                    local character = player.Character
                    if not character then return end
                    local root = character:FindFirstChild("HumanoidRootPart") :: BasePart?
                    if not root then return end
                    if (coin.Position - root.Position).Magnitude > 15 then return end

                    if not collected[coinId] then collected[coinId] = {} end
                    collected[coinId][player] = true
                    player:SetAttribute("Coins", (player:GetAttribute("Coins") or 0) + 1)
                    coin:Destroy()
                end)
            end

            return CoinService"""),
    },
    # --- Memory leaks ---
    {
        "tag": "memory_leak",
        "difficulty": 3,
        "user_code": textwrap.dedent("""\
            local RunService = game:GetService("RunService")
            local Players = game:GetService("Players")

            Players.PlayerAdded:Connect(function(player)
                player.CharacterAdded:Connect(function(character)
                    RunService.Heartbeat:Connect(function(dt)
                        local root = character:FindFirstChild("HumanoidRootPart")
                        if root then
                            -- Update something based on position
                            print(root.Position)
                        end
                    end)
                end)
            end)"""),
        "thinking": (
            "Every time a character spawns, a NEW Heartbeat connection is created "
            "and never disconnected. After a few respawns, there are N connections "
            "all running every frame. The character reference is also captured in "
            "the closure even after the character is destroyed. Need to disconnect "
            "on character removing or use a Trove/maid pattern."
        ),
        "issue": "Heartbeat connections accumulate on respawn — never disconnected.",
        "minimal_fix": textwrap.dedent("""\
            Players.PlayerAdded:Connect(function(player)
                local heartbeatConn: RBXScriptConnection? = nil

                player.CharacterAdded:Connect(function(character)
                    if heartbeatConn then heartbeatConn:Disconnect() end
                    heartbeatConn = RunService.Heartbeat:Connect(function(dt)
                        local root = character:FindFirstChild("HumanoidRootPart")
                        if root then
                            print(root.Position)
                        end
                    end)
                end)
            end)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local RunService = game:GetService("RunService")
            local Players = game:GetService("Players")
            local Trove = require(game:GetService("ReplicatedStorage").Packages.Trove)

            local PlayerTracker = {}

            function PlayerTracker:Start()
                Players.PlayerAdded:Connect(function(player: Player)
                    local playerTrove = Trove.new()

                    playerTrove:Connect(player.CharacterAdded, function(character: Model)
                        local charTrove = playerTrove:Extend()

                        charTrove:Connect(RunService.Heartbeat, function(_dt: number)
                            local root = character:FindFirstChild("HumanoidRootPart") :: BasePart?
                            if root then
                                -- Track position
                            end
                        end)

                        charTrove:Connect(character.AncestryChanged, function()
                            if not character:IsDescendantOf(workspace) then
                                charTrove:Clean()
                            end
                        end)
                    end)

                    playerTrove:Connect(Players.PlayerRemoving, function(leaving: Player)
                        if leaving == player then
                            playerTrove:Clean()
                        end
                    end)
                end)
            end

            return PlayerTracker"""),
    },
    {
        "tag": "memory_leak",
        "difficulty": 3,
        "user_code": textwrap.dedent("""\
            local cache = {}

            local function onPartAdded(part)
                cache[part] = {
                    position = part.Position,
                    created = os.clock(),
                }
            end

            workspace.DescendantAdded:Connect(onPartAdded)"""),
        "thinking": (
            "Parts are added to the cache but never removed. When parts are "
            "destroyed, the cache still holds references to them (strong keys), "
            "preventing garbage collection. Need to listen for DescendantRemoving "
            "or use Instance.Destroying."
        ),
        "issue": "Instance references in cache never cleaned up on Destroy.",
        "minimal_fix": textwrap.dedent("""\
            local cache = {}

            workspace.DescendantAdded:Connect(function(part)
                cache[part] = { position = part.Position, created = os.clock() }
            end)
            workspace.DescendantRemoving:Connect(function(part)
                cache[part] = nil
            end)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            type PartData = { position: Vector3, created: number }
            local cache: {[Instance]: PartData} = {}

            local function onAdded(desc: Instance)
                if not desc:IsA("BasePart") then return end
                cache[desc] = { position = (desc :: BasePart).Position, created = os.clock() }
            end

            local function onRemoving(desc: Instance)
                cache[desc] = nil
            end

            workspace.DescendantAdded:Connect(onAdded)
            workspace.DescendantRemoving:Connect(onRemoving)"""),
    },
    {
        "tag": "memory_leak",
        "difficulty": 4,
        "user_code": textwrap.dedent("""\
            local CollectionService = game:GetService("CollectionService")

            local function setupGrappleAnchor(anchor)
                local highlight = Instance.new("Highlight")
                highlight.FillTransparency = 0.8
                highlight.Parent = anchor

                anchor.Touched:Connect(function(hit)
                    -- Grapple logic
                end)
            end

            for _, anchor in CollectionService:GetTagged("GrappleAnchor") do
                setupGrappleAnchor(anchor)
            end
            CollectionService:GetInstanceAddedSignal("GrappleAnchor"):Connect(setupGrappleAnchor)"""),
        "thinking": (
            "When a GrappleAnchor is removed (e.g. zone unloads), the Touched "
            "connection is never disconnected and the Highlight instance is never "
            "cleaned up. Need to handle GetInstanceRemovedSignal to clean up. "
            "Also, Instance.new('Highlight') in the setup is fine but the Touched "
            "connection leaks."
        ),
        "issue": "No cleanup when tagged instances are removed — connections and children leak.",
        "minimal_fix": textwrap.dedent("""\
            local connections = {}

            local function setupGrappleAnchor(anchor)
                local highlight = Instance.new("Highlight")
                highlight.FillTransparency = 0.8
                highlight.Parent = anchor
                connections[anchor] = anchor.Touched:Connect(function(hit) end)
            end

            local function cleanupGrappleAnchor(anchor)
                if connections[anchor] then
                    connections[anchor]:Disconnect()
                    connections[anchor] = nil
                end
            end

            CollectionService:GetInstanceAddedSignal("GrappleAnchor"):Connect(setupGrappleAnchor)
            CollectionService:GetInstanceRemovedSignal("GrappleAnchor"):Connect(cleanupGrappleAnchor)
            for _, anchor in CollectionService:GetTagged("GrappleAnchor") do
                setupGrappleAnchor(anchor)
            end"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local CollectionService = game:GetService("CollectionService")
            local Trove = require(game:GetService("ReplicatedStorage").Packages.Trove)

            local TAG = "GrappleAnchor"
            local troves: {[Instance]: typeof(Trove.new())} = {}

            local function setupAnchor(anchor: BasePart)
                local trove = Trove.new()
                troves[anchor] = trove

                local highlight = trove:Add(Instance.new("Highlight"))
                highlight.FillTransparency = 0.8
                highlight.Parent = anchor

                trove:Connect(anchor.Touched, function(_hit: BasePart)
                    -- Grapple logic
                end)
            end

            local function cleanupAnchor(anchor: Instance)
                local trove = troves[anchor]
                if trove then
                    trove:Clean()
                    troves[anchor] = nil
                end
            end

            CollectionService:GetInstanceAddedSignal(TAG):Connect(setupAnchor)
            CollectionService:GetInstanceRemovedSignal(TAG):Connect(cleanupAnchor)
            for _, anchor in CollectionService:GetTagged(TAG) do
                setupAnchor(anchor)
            end"""),
    },
    # --- Performance antipatterns ---
    {
        "tag": "performance_antipattern",
        "difficulty": 4,
        "user_code": textwrap.dedent("""\
            local RunService = game:GetService("RunService")

            RunService.Heartbeat:Connect(function(dt)
                for _, player in game.Players:GetPlayers() do
                    local character = player.Character
                    if character then
                        local indicator = Instance.new("BillboardGui")
                        indicator.Size = UDim2.new(0, 100, 0, 40)
                        indicator.Parent = character.Head
                        -- Update indicator...
                        task.delay(0.1, function()
                            indicator:Destroy()
                        end)
                    end
                end
            end)"""),
        "thinking": (
            "Instance.new inside Heartbeat is catastrophic for performance. "
            "This creates and destroys BillboardGuis 60 times per second per "
            "player. Must pool or create once and update."
        ),
        "issue": "Instance.new in Heartbeat — creates/destroys 60x/sec per player.",
        "minimal_fix": textwrap.dedent("""\
            local indicators: {[Player]: BillboardGui} = {}

            local function getIndicator(character: Model): BillboardGui?
                local head = character:FindFirstChild("Head")
                if not head then return nil end
                local existing = head:FindFirstChild("Indicator")
                if existing then return existing :: BillboardGui end
                local gui = Instance.new("BillboardGui")
                gui.Name = "Indicator"
                gui.Size = UDim2.new(0, 100, 0, 40)
                gui.Parent = head
                return gui
            end

            RunService.Heartbeat:Connect(function(dt)
                for _, player in game.Players:GetPlayers() do
                    local character = player.Character
                    if character then
                        local indicator = getIndicator(character)
                        if indicator then
                            -- Update indicator
                        end
                    end
                end
            end)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local RunService = game:GetService("RunService")
            local Players = game:GetService("Players")

            local indicatorPool: {BillboardGui} = table.create(20)
            local activeIndicators: {[Player]: BillboardGui} = {}

            local function acquireIndicator(): BillboardGui
                if #indicatorPool > 0 then
                    return table.remove(indicatorPool) :: BillboardGui
                end
                local gui = Instance.new("BillboardGui")
                gui.Name = "PlayerIndicator"
                gui.Size = UDim2.new(0, 100, 0, 40)
                gui.ResetOnSpawn = false
                return gui
            end

            local function releaseIndicator(gui: BillboardGui)
                gui.Parent = nil
                table.insert(indicatorPool, gui)
            end

            @native
            local function updateIndicators(_dt: number)
                for _, player in Players:GetPlayers() do
                    local character = player.Character
                    local head = if character then character:FindFirstChild("Head") else nil
                    if head then
                        if not activeIndicators[player] then
                            local ind = acquireIndicator()
                            ind.Parent = head
                            activeIndicators[player] = ind
                        end
                        -- Update indicator content
                    elseif activeIndicators[player] then
                        releaseIndicator(activeIndicators[player])
                        activeIndicators[player] = nil
                    end
                end
            end

            RunService.Heartbeat:Connect(updateIndicators)

            Players.PlayerRemoving:Connect(function(player: Player)
                if activeIndicators[player] then
                    releaseIndicator(activeIndicators[player])
                    activeIndicators[player] = nil
                end
            end)"""),
    },
    {
        "tag": "performance_antipattern",
        "difficulty": 3,
        "user_code": textwrap.dedent("""\
            RunService.Heartbeat:Connect(function(dt)
                local data = httpService:JSONEncode({time = os.clock()})
                for match in string.gmatch(data, '"(%w+)"') do
                    print(match)
                end
            end)"""),
        "thinking": (
            "string.gmatch in a Heartbeat callback is terrible for performance. "
            "Pattern matching functions are not NCG-friendly and run in the "
            "interpreter. Also JSONEncode every frame is wasteful. This should "
            "use structured data access, not string parsing."
        ),
        "issue": "gmatch + JSONEncode in Heartbeat — interpreter-bound, 60x/sec.",
        "minimal_fix": textwrap.dedent("""\
            local lastLog = 0
            RunService.Heartbeat:Connect(function(dt)
                local now = os.clock()
                if now - lastLog < 1 then return end
                lastLog = now
                print(`time: {now}`)
            end)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local TELEMETRY_INTERVAL = 1.0
            local elapsed = 0.0

            @native
            local function onHeartbeat(dt: number)
                elapsed += dt
                if elapsed < TELEMETRY_INTERVAL then return end
                elapsed -= TELEMETRY_INTERVAL
                -- Use structured data, not string parsing
                local now = os.clock()
                -- Queue for batch send, don't process inline
            end

            RunService.Heartbeat:Connect(onHeartbeat)"""),
    },
    {
        "tag": "performance_antipattern",
        "difficulty": 4,
        "user_code": textwrap.dedent("""\
            local RunService = game:GetService("RunService")
            local positions = {}

            RunService.Heartbeat:Connect(function(dt)
                for _, part in workspace:GetDescendants() do
                    if part:IsA("BasePart") then
                        local function updatePos()
                            positions[part] = part.Position
                        end
                        updatePos()
                    end
                end
            end)"""),
        "thinking": (
            "Three problems: (1) workspace:GetDescendants() every frame is O(n) "
            "allocation, (2) closure `updatePos` is created inside the loop body "
            "every frame — that's a new closure per part per frame, and (3) "
            "capturing `part` as an upvalue in the closure. NCG cannot optimize "
            "closures created in loops."
        ),
        "issue": "Closure in hot loop, GetDescendants every frame, upvalue capture.",
        "minimal_fix": textwrap.dedent("""\
            local positions: {[BasePart]: Vector3} = {}
            local parts: {BasePart} = {}

            -- Cache parts list, refresh periodically
            local function refreshParts()
                table.clear(parts)
                for _, desc in workspace:GetDescendants() do
                    if desc:IsA("BasePart") then
                        table.insert(parts, desc)
                    end
                end
            end
            refreshParts()
            task.spawn(function() while true do task.wait(5); refreshParts() end end)

            @native
            local function updatePositions(_dt: number)
                for _, part in parts do
                    positions[part] = part.Position
                end
            end
            RunService.Heartbeat:Connect(updatePositions)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local RunService = game:GetService("RunService")
            local CollectionService = game:GetService("CollectionService")

            local TAG = "Tracked"
            local positions: {[BasePart]: Vector3} = {}
            local tracked: {BasePart} = {}

            local function onAdded(inst: Instance)
                if inst:IsA("BasePart") then
                    table.insert(tracked, inst)
                end
            end
            local function onRemoved(inst: Instance)
                local idx = table.find(tracked, inst)
                if idx then
                    local n = #tracked
                    tracked[idx] = tracked[n]
                    tracked[n] = nil
                    positions[inst :: BasePart] = nil
                end
            end

            CollectionService:GetInstanceAddedSignal(TAG):Connect(onAdded)
            CollectionService:GetInstanceRemovedSignal(TAG):Connect(onRemoved)
            for _, inst in CollectionService:GetTagged(TAG) do onAdded(inst) end

            @native
            local function updatePositions(_dt: number)
                for i = 1, #tracked do
                    local part = tracked[i]
                    positions[part] = part.Position
                end
            end
            RunService.Heartbeat:Connect(updatePositions)"""),
    },
    # --- Architecture smells ---
    {
        "tag": "architecture_smell",
        "difficulty": 4,
        "user_code": textwrap.dedent("""\
            -- Everything in one giant script
            local Players = game:GetService("Players")
            local RunService = game:GetService("RunService")
            local DataStoreService = game:GetService("DataStoreService")

            local MAX_HEALTH = 100
            local WALK_SPEED = 16
            local JUMP_POWER = 50
            local COIN_VALUE = 10

            local dataStore = DataStoreService:GetDataStore("PlayerData")

            Players.PlayerAdded:Connect(function(player)
                local data = dataStore:GetAsync(player.UserId)
                player.CharacterAdded:Connect(function(character)
                    local humanoid = character:WaitForChild("Humanoid")
                    humanoid.MaxHealth = MAX_HEALTH
                    humanoid.WalkSpeed = WALK_SPEED
                    humanoid.JumpPower = JUMP_POWER
                end)
                -- ...200 more lines of unrelated logic...
            end)"""),
        "thinking": (
            "This is a God module — data persistence, character setup, and config "
            "values all in one script. Vertigo convention is separate services with "
            "Init/Start lifecycle. Config should be in a config module. DataStore "
            "calls need pcall. The inline constants should be in a config module."
        ),
        "issue": "God module with mixed concerns, inline config, no Init/Start lifecycle.",
        "minimal_fix": textwrap.dedent("""\
            -- Split into: Config/Player.luau, DataService.luau, CharacterService.luau
            -- At minimum, extract config:
            local Config = require(Shared.Config.Player)

            Players.PlayerAdded:Connect(function(player)
                local ok, data = pcall(dataStore.GetAsync, dataStore, player.UserId)
                if not ok then warn(`[Data] Failed to load {player.Name}: {data}`) end
                -- ...
            end)"""),
        "production_fix": textwrap.dedent("""\
            -- src/Shared/Config/Player.luau
            --!strict
            local PlayerConfig = table.freeze({
                MAX_HEALTH = 100,
                WALK_SPEED = 16,
                JUMP_POWER = 50,
                COIN_VALUE = 10,
            })
            return { PlayerConfig = PlayerConfig }

            -- src/Server/Services/DataService.luau
            --!strict
            local DataStoreService = game:GetService("DataStoreService")
            local DataService = {}
            local store: DataStore

            function DataService:Init()
                store = DataStoreService:GetDataStore("PlayerData")
            end

            function DataService:LoadProfile(player: Player): {[string]: any}?
                local ok, data = pcall(store.GetAsync, store, tostring(player.UserId))
                if not ok then
                    warn(`[DataService] Load failed for {player.Name}: {data}`)
                    return nil
                end
                return data
            end

            return DataService

            -- src/Server/Services/CharacterService.luau
            --!strict
            local Config = require(Shared.Config.Player).PlayerConfig
            local CharacterService = {}

            function CharacterService:Start()
                game.Players.PlayerAdded:Connect(function(player: Player)
                    player.CharacterAdded:Connect(function(character: Model)
                        local humanoid = character:WaitForChild("Humanoid") :: Humanoid
                        humanoid.MaxHealth = Config.MAX_HEALTH
                        humanoid.WalkSpeed = Config.WALK_SPEED
                        humanoid.JumpPower = Config.JUMP_POWER
                    end)
                end)
            end

            return CharacterService"""),
    },
    {
        "tag": "architecture_smell",
        "difficulty": 3,
        "user_code": textwrap.dedent("""\
            local module = {}

            -- No Init or Start, just runs on require
            local connection = RunService.Heartbeat:Connect(function(dt)
                -- process every frame
            end)

            local remote = Instance.new("RemoteEvent")
            remote.Name = "MyRemote"
            remote.Parent = ReplicatedStorage

            return module"""),
        "thinking": (
            "Side effects on require: the module connects to Heartbeat and "
            "creates a RemoteEvent at require-time. This breaks the Init/Start "
            "lifecycle — boot order becomes implicit and fragile. Remotes should "
            "be declared in the Remotes module, and connections belong in Start."
        ),
        "issue": "Side effects at require-time — breaks Init/Start lifecycle.",
        "minimal_fix": textwrap.dedent("""\
            local module = {}
            local connection: RBXScriptConnection?

            function module:Start()
                connection = RunService.Heartbeat:Connect(function(dt)
                    -- process every frame
                end)
            end

            return module"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local RunService = game:GetService("RunService")

            local MyService = {}
            local _connection: RBXScriptConnection?

            function MyService:Init()
                -- Setup state, no side effects
            end

            @native
            local function onHeartbeat(dt: number)
                -- Process every frame
            end

            function MyService:Start()
                _connection = RunService.Heartbeat:Connect(onHeartbeat)
            end

            return MyService"""),
    },
    # --- Missing error handling ---
    {
        "tag": "missing_error_handling",
        "difficulty": 3,
        "user_code": textwrap.dedent("""\
            local DataStoreService = game:GetService("DataStoreService")
            local store = DataStoreService:GetDataStore("PlayerData")

            local function saveData(player)
                store:SetAsync(player.UserId, {
                    coins = player:GetAttribute("Coins"),
                    level = player:GetAttribute("Level"),
                })
            end

            game.Players.PlayerRemoving:Connect(saveData)"""),
        "thinking": (
            "DataStore:SetAsync can fail (rate limits, network errors, budget "
            "exhaustion). Without pcall, a failure crashes the handler and the "
            "player's data is lost. Also using UserId directly as key (should "
            "be string), and no retry logic."
        ),
        "issue": "Bare DataStore:SetAsync without pcall — data loss on failure.",
        "minimal_fix": textwrap.dedent("""\
            local function saveData(player)
                local ok, err = pcall(store.SetAsync, store, tostring(player.UserId), {
                    coins = player:GetAttribute("Coins"),
                    level = player:GetAttribute("Level"),
                })
                if not ok then
                    warn(`[Data] Failed to save {player.Name}: {err}`)
                end
            end"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local DataStoreService = game:GetService("DataStoreService")

            local MAX_RETRIES = 3
            local RETRY_DELAY = 1.0

            local DataService = {}
            local store: DataStore

            function DataService:Init()
                store = DataStoreService:GetDataStore("PlayerData")
            end

            function DataService:SaveProfile(player: Player): boolean
                local data = {
                    coins = player:GetAttribute("Coins") or 0,
                    level = player:GetAttribute("Level") or 1,
                    savedAt = os.time(),
                }

                for attempt = 1, MAX_RETRIES do
                    local ok, err = pcall(store.SetAsync, store, tostring(player.UserId), data)
                    if ok then
                        return true
                    end
                    warn(`[DataService] Save attempt {attempt}/{MAX_RETRIES} failed for {player.Name}: {err}`)
                    if attempt < MAX_RETRIES then
                        task.wait(RETRY_DELAY * attempt)
                    end
                end

                warn(`[DataService] All save attempts failed for {player.Name}`)
                return false
            end

            function DataService:Start()
                game.Players.PlayerRemoving:Connect(function(player: Player)
                    self:SaveProfile(player)
                end)

                game:BindToClose(function()
                    local threads: {thread} = {}
                    for _, player in game.Players:GetPlayers() do
                        table.insert(threads, task.spawn(function()
                            self:SaveProfile(player)
                        end))
                    end
                    task.wait(5) -- Grace period
                end)
            end

            return DataService"""),
    },
    {
        "tag": "missing_error_handling",
        "difficulty": 3,
        "user_code": textwrap.dedent("""\
            local HttpService = game:GetService("HttpService")

            local function fetchLeaderboard()
                local response = HttpService:GetAsync("https://api.example.com/leaderboard")
                local data = HttpService:JSONDecode(response)
                return data
            end"""),
        "thinking": (
            "GetAsync can fail (network timeout, 4xx/5xx, HttpEnabled=false). "
            "JSONDecode can fail if the response isn't valid JSON (e.g. error "
            "page HTML). Neither is wrapped in pcall."
        ),
        "issue": "No pcall on HttpService:GetAsync or JSONDecode.",
        "minimal_fix": textwrap.dedent("""\
            local function fetchLeaderboard()
                local ok, response = pcall(HttpService.GetAsync, HttpService,
                    "https://api.example.com/leaderboard")
                if not ok then
                    warn(`[HTTP] Request failed: {response}`)
                    return nil
                end
                local parseOk, data = pcall(HttpService.JSONDecode, HttpService, response)
                if not parseOk then
                    warn(`[HTTP] JSON parse failed: {data}`)
                    return nil
                end
                return data
            end"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local HttpService = game:GetService("HttpService")

            type LeaderboardEntry = { name: string, score: number }

            local MAX_RETRIES = 2
            local TIMEOUT = 10

            local function fetchLeaderboard(): {LeaderboardEntry}?
                for attempt = 1, MAX_RETRIES do
                    local ok, response = pcall(HttpService.RequestAsync, HttpService, {
                        Url = "https://api.example.com/leaderboard",
                        Method = "GET",
                        Headers = { ["Accept"] = "application/json" },
                    })
                    if not ok then
                        warn(`[Leaderboard] Request attempt {attempt} failed: {response}`)
                        if attempt < MAX_RETRIES then task.wait(1) end
                        continue
                    end

                    local resp = response :: {StatusCode: number, Body: string}
                    if resp.StatusCode ~= 200 then
                        warn(`[Leaderboard] HTTP {resp.StatusCode}`)
                        continue
                    end

                    local parseOk, data = pcall(HttpService.JSONDecode, HttpService, resp.Body)
                    if not parseOk then
                        warn(`[Leaderboard] Parse failed: {data}`)
                        return nil
                    end
                    return data :: {LeaderboardEntry}
                end
                return nil
            end"""),
    },
]


# ---------------------------------------------------------------------------
# Type 2: Coach data
# ---------------------------------------------------------------------------

COACH_EXAMPLES: list[dict] = [
    {
        "difficulty": 2,
        "question": "My script uses `wait()` and someone said it's bad. Why?",
        "thinking": (
            "wait() is deprecated and has a minimum yield of ~0.03s (one frame), "
            "not the exact time requested. task.wait() is the modern replacement "
            "with better precision and scheduler integration."
        ),
        "one_sentence": "`wait()` is deprecated — use `task.wait()` which has better precision and is actively maintained.",
        "minimal_fix": "Replace `wait(n)` with `task.wait(n)` and `spawn(fn)` with `task.spawn(fn)`.",
        "production_fix": textwrap.dedent("""\
            --!strict
            -- Before (deprecated):
            -- wait(1); spawn(function() ... end); delay(2, fn)

            -- After (modern task library):
            task.wait(1)
            task.spawn(function()
                -- runs next resumption cycle
            end)
            task.delay(2, function()
                -- runs after 2 seconds
            end)
            task.defer(function()
                -- runs at end of current resumption cycle
            end)"""),
        "test_code": textwrap.dedent("""\
            -- Selene lint catches deprecated globals:
            -- selene.toml: std = "roblox"
            -- selene will warn: `wait` is deprecated, use `task.wait` instead
            -- No runtime test needed — this is a static analysis catch."""),
    },
    {
        "difficulty": 3,
        "question": "Why does my Touched event fire multiple times per contact?",
        "thinking": (
            "Touched fires for every contact point between parts. A character "
            "has multiple body parts, so walking over a trigger fires once per "
            "limb per physics step. Need a debounce pattern."
        ),
        "one_sentence": "`Touched` fires per-contact-point per-physics-step — you need debounce to get one logical trigger.",
        "minimal_fix": textwrap.dedent("""\
            local debounce = false
            part.Touched:Connect(function(hit)
                if debounce then return end
                debounce = true
                -- handle touch
                task.delay(0.5, function() debounce = false end)
            end)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local COOLDOWN = 0.5
            local lastTouch: {[Player]: number} = {}

            local function onTouched(hit: BasePart)
                local character = hit.Parent :: Model?
                if not character then return end
                local humanoid = character:FindFirstChildOfClass("Humanoid")
                if not humanoid then return end

                local player = game.Players:GetPlayerFromCharacter(character)
                if not player then return end

                local now = os.clock()
                if lastTouch[player] and now - lastTouch[player] < COOLDOWN then return end
                lastTouch[player] = now

                -- Handle touch logic here
            end

            triggerPart.Touched:Connect(onTouched)"""),
        "test_code": textwrap.dedent("""\
            -- Test: rapid touches should only trigger once within cooldown
            local touchCount = 0
            local mockPart = { Parent = mockCharacter }
            for i = 1, 10 do
                onTouched(mockPart)
            end
            assert(touchCount == 1, `Expected 1 touch, got {touchCount}`)"""),
    },
    {
        "difficulty": 3,
        "question": "What's wrong with `player.Character.Humanoid.Health`?",
        "thinking": (
            "Three potential nil dereferences chained together. Character can be "
            "nil during respawn, Humanoid might not exist yet (WaitForChild race), "
            "and Health access on nil Humanoid errors. Each access needs a guard."
        ),
        "one_sentence": "Three chained nil-unsafe accesses — Character, Humanoid, and Health can each be nil during respawn.",
        "minimal_fix": textwrap.dedent("""\
            local character = player.Character
            if not character then return end
            local humanoid = character:FindFirstChildOfClass("Humanoid")
            if not humanoid then return end
            local health = humanoid.Health"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local function getHealth(player: Player): number?
                local character = player.Character
                if not character then return nil end
                local humanoid = character:FindFirstChildOfClass("Humanoid")
                if not humanoid then return nil end
                return humanoid.Health
            end"""),
        "test_code": textwrap.dedent("""\
            -- Test with nil character
            local mockPlayer = { Character = nil }
            assert(getHealth(mockPlayer) == nil, "Should return nil for nil character")

            -- Test with valid character
            local mockHumanoid = { Health = 75 }
            local mockChar = { FindFirstChildOfClass = function(_, _) return mockHumanoid end }
            mockPlayer.Character = mockChar
            assert(getHealth(mockPlayer) == 75, "Should return health value")"""),
    },
    {
        "difficulty": 4,
        "question": "How should I handle DataStore errors properly?",
        "thinking": (
            "DataStore calls can fail due to rate limits, network issues, or "
            "budget exhaustion. Every call needs pcall, retry with backoff, "
            "and BindToClose for save-on-shutdown. Profile pattern (ProfileService) "
            "is the gold standard but manual pcall+retry is the minimum."
        ),
        "one_sentence": "Every DataStore call must be wrapped in pcall with retry logic and BindToClose for save-on-shutdown.",
        "minimal_fix": textwrap.dedent("""\
            local ok, result = pcall(store.GetAsync, store, key)
            if not ok then warn("DataStore error:", result) end"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local RETRIES = 3

            local function safeGet(store: DataStore, key: string): (boolean, any)
                for attempt = 1, RETRIES do
                    local ok, result = pcall(store.GetAsync, store, key)
                    if ok then return true, result end
                    warn(`[DataStore] GetAsync attempt {attempt} failed: {result}`)
                    if attempt < RETRIES then task.wait(attempt) end
                end
                return false, nil
            end

            game:BindToClose(function()
                for _, player in game.Players:GetPlayers() do
                    task.spawn(function()
                        local ok, err = pcall(store.SetAsync, store,
                            tostring(player.UserId), gatherSaveData(player))
                        if not ok then warn(`[DataStore] Shutdown save failed: {err}`) end
                    end)
                end
                task.wait(5)
            end)"""),
        "test_code": textwrap.dedent("""\
            -- Mock DataStore that fails first N times
            local failCount = 0
            local mockStore = {
                GetAsync = function(_, key)
                    failCount += 1
                    if failCount <= 2 then error("Rate limited") end
                    return { coins = 100 }
                end
            }
            local ok, data = safeGet(mockStore, "test_key")
            assert(ok, "Should succeed after retries")
            assert(data.coins == 100, "Should return correct data")"""),
    },
    {
        "difficulty": 3,
        "question": "Why does my CollectionService tagged object leak when the zone unloads?",
        "thinking": (
            "GetInstanceAddedSignal fires when tagged objects appear, but if you "
            "only listen for added and never for removed, you accumulate state "
            "(connections, cache entries, UI elements) for objects that no longer "
            "exist. Must pair every AddedSignal with a RemovedSignal."
        ),
        "one_sentence": "You listen for tag-added but not tag-removed — state accumulates for destroyed instances.",
        "minimal_fix": textwrap.dedent("""\
            CollectionService:GetInstanceRemovedSignal(TAG):Connect(function(inst)
                cleanup(inst)
            end)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local Trove = require(game:GetService("ReplicatedStorage").Packages.Trove)

            local TAG = "Crystal"
            local troves: {[Instance]: typeof(Trove.new())} = {}

            local function onAdded(inst: Instance)
                local trove = Trove.new()
                troves[inst] = trove
                -- All connections/instances via trove
                trove:Connect(inst:GetPropertyChangedSignal("Position"), function() end)
            end

            local function onRemoved(inst: Instance)
                if troves[inst] then
                    troves[inst]:Clean()
                    troves[inst] = nil
                end
            end

            CollectionService:GetInstanceAddedSignal(TAG):Connect(onAdded)
            CollectionService:GetInstanceRemovedSignal(TAG):Connect(onRemoved)
            for _, inst in CollectionService:GetTagged(TAG) do onAdded(inst) end"""),
        "test_code": textwrap.dedent("""\
            -- Test: adding then removing should clean up
            local mockInst = Instance.new("Part")
            mockInst:AddTag("Crystal")
            assert(troves[mockInst] ~= nil, "Trove should exist after add")
            mockInst:RemoveTag("Crystal")
            -- After removed signal fires:
            assert(troves[mockInst] == nil, "Trove should be cleaned after remove")"""),
    },
    {
        "difficulty": 4,
        "question": "When should I use @native and what does it actually do?",
        "thinking": (
            "@native tells the Luau compiler to generate native code (NCG) for "
            "that function. It works best on tight numeric loops with full type "
            "annotations. It does NOT help with functions that call Roblox APIs "
            "heavily or do string operations. The function must have type info "
            "for NCG to be effective."
        ),
        "one_sentence": "`@native` enables native code generation — use it on typed, numeric hot-path functions (Heartbeat callbacks, math utilities), not on API-heavy code.",
        "minimal_fix": textwrap.dedent("""\
            @native
            local function lerp(a: number, b: number, t: number): number
                return math.lerp(a, b, t)
            end"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            -- Good: tight numeric loop with full types
            @native
            local function updatePositions(parts: {BasePart}, targets: {Vector3}, dt: number)
                local alpha = math.clamp(dt * 10, 0, 1)
                for i = 1, #parts do
                    local current = parts[i].Position
                    parts[i].Position = current:Lerp(targets[i], alpha)
                end
            end

            -- Bad: @native on API-heavy code (no benefit, may hurt)
            -- DON'T do this:
            -- @native
            -- local function setupPlayer(player: Player)
            --     local gui = Instance.new("ScreenGui")  -- Roblox API call
            --     gui.Parent = player.PlayerGui           -- Roblox API call
            -- end"""),
        "test_code": textwrap.dedent("""\
            -- Benchmark to verify @native benefit:
            local N = 100000
            local start = os.clock()
            for i = 1, N do
                updatePositions(testParts, testTargets, 0.016)
            end
            local elapsed = os.clock() - start
            print(`@native loop: {elapsed * 1000:.2f}ms for {N} iterations`)
            -- Expect 2-5x faster than non-@native version"""),
    },
    {
        "difficulty": 2,
        "question": "What's the difference between task.spawn and task.defer?",
        "thinking": (
            "task.spawn runs the function immediately (synchronously until first "
            "yield), while task.defer queues it to run at the end of the current "
            "resumption cycle. Use spawn when you need the function to start "
            "executing now, defer when you want to avoid blocking the current path."
        ),
        "one_sentence": "`task.spawn` runs immediately until first yield; `task.defer` queues to end of current cycle.",
        "minimal_fix": textwrap.dedent("""\
            -- Use task.spawn for immediate execution:
            task.spawn(function() print("runs now") end)
            -- Use task.defer to avoid blocking:
            task.defer(function() print("runs after current cycle") end)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            -- task.spawn: good for fire-and-forget that needs to start NOW
            task.spawn(function()
                local ok, err = pcall(store.SetAsync, store, key, data)
                if not ok then warn(`Save failed: {err}`) end
            end)

            -- task.defer: good for avoiding deep call stacks
            -- Example: event handler that triggers another event
            signal:Connect(function(data: any)
                -- Process synchronously
                processData(data)
                -- Defer the notification to avoid re-entrancy
                task.defer(function()
                    otherSignal:Fire(data)
                end)
            end)"""),
        "test_code": textwrap.dedent("""\
            -- Demonstrate ordering difference
            local order = {}
            task.defer(function() table.insert(order, "defer") end)
            task.spawn(function() table.insert(order, "spawn") end)
            table.insert(order, "inline")
            task.wait() -- yield to let deferred run
            -- order = {"spawn", "inline", "defer"}
            assert(order[1] == "spawn")
            assert(order[2] == "inline")
            assert(order[3] == "defer")"""),
    },
    {
        "difficulty": 3,
        "question": "My RemoteEvent handler doesn't validate input types. What can go wrong?",
        "thinking": (
            "A malicious client can send ANY value type through a RemoteEvent. "
            "If your server handler expects a number but receives a table, string, "
            "or nil, it can crash, corrupt state, or be exploited. Every argument "
            "must be typeof-checked before use."
        ),
        "one_sentence": "Clients can send any type — without typeof checks, exploiters can crash your server or corrupt state.",
        "minimal_fix": textwrap.dedent("""\
            remote.OnServerEvent:Connect(function(player, amount)
                if typeof(amount) ~= "number" then return end
                -- safe to use amount as number
            end)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            type ValidatedInput = {
                action: string,
                target: Vector3,
                value: number,
            }

            local function validateInput(raw: unknown): ValidatedInput?
                if typeof(raw) ~= "table" then return nil end
                local t = raw :: {[string]: unknown}
                if typeof(t.action) ~= "string" then return nil end
                if typeof(t.target) ~= "Vector3" then return nil end
                if typeof(t.value) ~= "number" then return nil end
                if t.value < 0 or t.value > 1000 then return nil end
                return {
                    action = t.action :: string,
                    target = t.target :: Vector3,
                    value = t.value :: number,
                }
            end

            remote.OnServerEvent:Connect(function(player: Player, raw: unknown)
                local input = validateInput(raw)
                if not input then
                    warn(`[Net] Invalid input from {player.Name}`)
                    return
                end
                -- Safe to use input.action, input.target, input.value
            end)"""),
        "test_code": textwrap.dedent("""\
            -- Test type validation
            assert(validateInput(nil) == nil, "nil should fail")
            assert(validateInput("string") == nil, "string should fail")
            assert(validateInput({action = 5}) == nil, "wrong action type should fail")
            assert(validateInput({
                action = "attack",
                target = Vector3.new(0,0,0),
                value = 50
            }) ~= nil, "valid input should pass")
            assert(validateInput({
                action = "attack",
                target = Vector3.new(0,0,0),
                value = -1
            }) == nil, "negative value should fail")"""),
    },
    {
        "difficulty": 3,
        "question": "How do I properly clean up when a player leaves?",
        "thinking": (
            "PlayerRemoving fires when a player disconnects. If you have per-player "
            "state (connections, cache entries, GUI, character tracking), it all "
            "needs cleanup. Common pattern is a per-player Trove or maid that "
            "gets cleaned on PlayerRemoving."
        ),
        "one_sentence": "Use PlayerRemoving + a per-player Trove to clean connections, caches, and instances — otherwise they leak until server shutdown.",
        "minimal_fix": textwrap.dedent("""\
            local playerData = {}
            Players.PlayerRemoving:Connect(function(player)
                playerData[player] = nil
            end)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            local Players = game:GetService("Players")
            local Trove = require(game:GetService("ReplicatedStorage").Packages.Trove)

            local playerTroves: {[Player]: typeof(Trove.new())} = {}

            local function onPlayerAdded(player: Player)
                local trove = Trove.new()
                playerTroves[player] = trove

                -- All per-player connections go through the trove
                trove:Connect(player.CharacterAdded, function(character: Model)
                    -- Character setup
                end)

                -- Per-player state
                trove:Add(function()
                    -- Custom cleanup logic
                end)
            end

            local function onPlayerRemoving(player: Player)
                local trove = playerTroves[player]
                if trove then
                    trove:Clean()
                    playerTroves[player] = nil
                end
            end

            Players.PlayerAdded:Connect(onPlayerAdded)
            Players.PlayerRemoving:Connect(onPlayerRemoving)
            for _, player in Players:GetPlayers() do onPlayerAdded(player) end"""),
        "test_code": textwrap.dedent("""\
            -- Verify cleanup on remove
            local mockPlayer = createMockPlayer()
            onPlayerAdded(mockPlayer)
            assert(playerTroves[mockPlayer] ~= nil, "Trove should exist")
            onPlayerRemoving(mockPlayer)
            assert(playerTroves[mockPlayer] == nil, "Trove should be cleaned up")"""),
    },
    {
        "difficulty": 4,
        "question": "Why is vector.create faster than Vector3.new in tight loops?",
        "thinking": (
            "vector.create returns a native Luau vector value type that lives on "
            "the stack (no heap allocation). Vector3.new creates a userdata object "
            "on the heap with metatable overhead. In tight loops, the allocation "
            "pressure from Vector3.new causes GC pauses. With @native + vector.*, "
            "the compiler can use SIMD instructions."
        ),
        "one_sentence": "`vector.create` produces a stack-allocated value type with SIMD support; `Vector3.new` allocates heap userdata with GC pressure.",
        "minimal_fix": textwrap.dedent("""\
            -- In hot paths, prefer:
            local v = vector.create(x, y, z)
            local sum = v + vector.create(1, 0, 0)"""),
        "production_fix": textwrap.dedent("""\
            --!strict
            -- Hot path: particle system update
            @native
            local function updateParticles(
                positions: {vector},
                velocities: {vector},
                gravity: vector,
                dt: number
            )
                for i = 1, #positions do
                    velocities[i] += gravity * dt
                    positions[i] += velocities[i] * dt
                end
            end

            local GRAVITY = vector.create(0, -196.2, 0)
            -- Note: when interfacing with Roblox APIs, convert back:
            -- part.Position = Vector3.new(pos.x, pos.y, pos.z)"""),
        "test_code": textwrap.dedent("""\
            -- Benchmark comparison
            local N = 1000000
            local t0 = os.clock()
            for i = 1, N do
                local v = vector.create(i, i, i)
            end
            local vectorTime = os.clock() - t0

            t0 = os.clock()
            for i = 1, N do
                local v = Vector3.new(i, i, i)
            end
            local v3Time = os.clock() - t0

            print(`vector.create: {vectorTime*1000:.2f}ms`)
            print(`Vector3.new: {v3Time*1000:.2f}ms`)
            print(`Speedup: {v3Time/vectorTime:.1f}x`)"""),
    },
]


# ---------------------------------------------------------------------------
# Type 3: Pairwise preference
# ---------------------------------------------------------------------------

PREFERENCE_PAIRS: list[dict] = [
    {
        "difficulty": 3,
        "task": "Iterate over all players and print their names",
        "solution_a": textwrap.dedent("""\
            for i, v in pairs(game.Players:GetPlayers()) do
                print(v.Name)
            end"""),
        "solution_b": textwrap.dedent("""\
            --!strict
            local Players = game:GetService("Players")
            for _, player in Players:GetPlayers() do
                print(player.Name)
            end"""),
        "better": "B",
        "thinking": (
            "Solution A uses pairs() (deprecated pattern), direct game.Players "
            "access (should use GetService), and generic variable names. "
            "Solution B uses modern generalized for, GetService, descriptive "
            "variable names, and --!strict."
        ),
        "reasoning": (
            "Solution B is better because: uses modern Luau iteration (no pairs()), "
            "game:GetService for reliable access, descriptive variable name `player` "
            "instead of `v`, and --!strict mode."
        ),
    },
    {
        "difficulty": 3,
        "task": "Create a function that lerps between two numbers",
        "solution_a": textwrap.dedent("""\
            local function lerp(a, b, t)
                return a + (b - a) * t
            end"""),
        "solution_b": textwrap.dedent("""\
            --!strict
            @native
            local function lerp(a: number, b: number, t: number): number
                return math.lerp(a, b, t)
            end"""),
        "better": "B",
        "thinking": (
            "Solution A manually computes lerp without types. Solution B uses "
            "math.lerp (compiles to FMA on NCG), has full type annotations for "
            "NCG optimization, @native for hot-path use, and --!strict."
        ),
        "reasoning": (
            "Solution B is better because: math.lerp compiles to a single FMA "
            "instruction under NCG, @native enables native code gen, full type "
            "annotations enable strict checking and optimization."
        ),
    },
    {
        "difficulty": 4,
        "task": "Handle a remote event for player ability activation",
        "solution_a": textwrap.dedent("""\
            remote.OnServerEvent:Connect(function(player, abilityName, targetPos)
                local ability = abilities[abilityName]
                ability:Activate(player, targetPos)
            end)"""),
        "solution_b": textwrap.dedent("""\
            --!strict
            remote.OnServerEvent:Connect(function(player: Player, abilityName: unknown, targetPos: unknown)
                if typeof(abilityName) ~= "string" then return end
                if typeof(targetPos) ~= "Vector3" then return end

                local ability = abilities[abilityName]
                if not ability then
                    warn(`[AbilityService] Unknown ability: {abilityName} from {player.Name}`)
                    return
                end

                local character = player.Character
                if not character then return end
                local root = character:FindFirstChild("HumanoidRootPart") :: BasePart?
                if not root then return end

                if (targetPos :: Vector3 - root.Position).Magnitude > ability.maxRange then
                    return
                end

                ability:Activate(player, targetPos :: Vector3)
            end)"""),
        "better": "B",
        "thinking": (
            "Solution A trusts client input completely — no type checks, no nil "
            "checks, no range validation. An exploiter can send any abilityName "
            "or targetPos. Solution B validates every input, checks ability exists, "
            "verifies range, and handles missing character."
        ),
        "reasoning": (
            "Solution B is better because: validates all input types (exploiter "
            "protection), checks ability existence, enforces range limits, handles "
            "nil character, and uses --!strict with type annotations."
        ),
    },
    {
        "difficulty": 4,
        "task": "Track per-player cooldowns for an ability system",
        "solution_a": textwrap.dedent("""\
            local cooldowns = {}

            remote.OnServerEvent:Connect(function(player, ability)
                if cooldowns[player.UserId .. ability] then return end
                cooldowns[player.UserId .. ability] = true
                -- Use ability
                wait(5)
                cooldowns[player.UserId .. ability] = nil
            end)"""),
        "solution_b": textwrap.dedent("""\
            --!strict
            local cooldowns: {[number]: {[string]: number}} = {}

            remote.OnServerEvent:Connect(function(player: Player, ability: unknown)
                if typeof(ability) ~= "string" then return end
                local abilityName = ability :: string

                local playerCooldowns = cooldowns[player.UserId]
                if not playerCooldowns then
                    playerCooldowns = {}
                    cooldowns[player.UserId] = playerCooldowns
                end

                local now = os.clock()
                local lastUse = playerCooldowns[abilityName]
                if lastUse and now - lastUse < 5 then return end

                playerCooldowns[abilityName] = now
                -- Use ability
            end)

            game.Players.PlayerRemoving:Connect(function(player: Player)
                cooldowns[player.UserId] = nil
            end)"""),
        "better": "B",
        "thinking": (
            "Solution A uses string concatenation for keys (fragile, no type "
            "safety), wait() (deprecated, yields the handler), and never cleans "
            "up on player leave. Solution B uses nested tables with proper types, "
            "os.clock timestamps instead of yielding, input validation, and "
            "PlayerRemoving cleanup."
        ),
        "reasoning": (
            "Solution B is better because: uses structured nested tables instead "
            "of string-concat keys, os.clock timestamps instead of yielding with "
            "wait(), validates input types, cleans up on PlayerRemoving, and "
            "uses --!strict with full annotations."
        ),
    },
    {
        "difficulty": 3,
        "task": "Pre-allocate an array of 100 zeros",
        "solution_a": textwrap.dedent("""\
            local arr = {}
            for i = 1, 100 do
                table.insert(arr, 0)
            end"""),
        "solution_b": textwrap.dedent("""\
            --!strict
            local arr: {number} = table.create(100, 0)"""),
        "better": "B",
        "thinking": (
            "Solution A uses a loop with table.insert which causes repeated "
            "array resizing (realloc). Solution B uses table.create which "
            "pre-allocates the exact size in one call — zero resizing overhead."
        ),
        "reasoning": (
            "Solution B is better because: table.create(100, 0) pre-allocates "
            "in a single C call with zero resizing. It's both faster and more "
            "readable than a loop with table.insert."
        ),
    },
    {
        "difficulty": 4,
        "task": "Build a spatial query to find parts near a position",
        "solution_a": textwrap.dedent("""\
            local function findNearby(pos, radius)
                local result = {}
                for _, part in workspace:GetDescendants() do
                    if part:IsA("BasePart") then
                        if (part.Position - pos).Magnitude < radius then
                            table.insert(result, part)
                        end
                    end
                end
                return result
            end"""),
        "solution_b": textwrap.dedent("""\
            --!strict
            local function findNearby(position: Vector3, radius: number): {BasePart}
                local params = OverlapParams.new()
                params.FilterType = Enum.RaycastFilterType.Exclude
                params.FilterDescendantsInstances = {}

                return workspace:GetPartBoundsInRadius(position, radius, params)
            end"""),
        "better": "B",
        "thinking": (
            "Solution A iterates ALL descendants of workspace (potentially tens "
            "of thousands), checks IsA, and computes distance. O(n) every call. "
            "Solution B uses the engine's spatial query (GetPartBoundsInRadius) "
            "which uses the internal BVH/octree — O(log n) and runs in C++."
        ),
        "reasoning": (
            "Solution B is better because: GetPartBoundsInRadius uses the engine's "
            "native spatial index (O(log n) vs O(n)), runs in C++ not Luau, has "
            "proper type annotations, and is the idiomatic Roblox approach."
        ),
    },
    {
        "difficulty": 4,
        "task": "Create constants for ability tuning values",
        "solution_a": textwrap.dedent("""\
            -- In the ability script directly
            local GRAPPLE_RANGE = 80
            local GRAPPLE_SPEED = 120
            local GRAPPLE_COOLDOWN = 0.5
            local GLIDE_SPEED = 40
            local GLIDE_DRAG = 0.02"""),
        "solution_b": textwrap.dedent("""\
            -- src/Shared/Config/Abilities.luau
            --!strict
            local GrappleTuning = table.freeze({
                RANGE = 80,
                SPEED = 120,
                COOLDOWN = 0.5,
            })

            local GlideTuning = table.freeze({
                SPEED = 40,
                DRAG = 0.02,
            })

            return { GrappleTuning = GrappleTuning, GlideTuning = GlideTuning }"""),
        "better": "B",
        "thinking": (
            "Solution A scatters tuning values as local constants in the script. "
            "Changing them requires finding and editing the right script. Solution B "
            "centralizes config in a shared config module with table.freeze for "
            "immutability, matching Vertigo's Config module pattern."
        ),
        "reasoning": (
            "Solution B is better because: centralizes tuning in a config module "
            "(single source of truth), uses table.freeze for immutability, follows "
            "Vertigo's Config/ pattern with named exports, and enables hot-tuning."
        ),
    },
    {
        "difficulty": 3,
        "task": "Format a player greeting message",
        "solution_a": textwrap.dedent("""\
            local msg = "Welcome, " .. player.Name .. "! You have " .. tostring(coins) .. " coins and are level " .. tostring(level) .. "."
            print(msg)"""),
        "solution_b": textwrap.dedent("""\
            --!strict
            local msg: string = `Welcome, {player.Name}! You have {coins} coins and are level {level}.`
            print(msg)"""),
        "better": "B",
        "thinking": (
            "Solution A uses .. concatenation with tostring calls — verbose, "
            "hard to read, creates intermediate strings. Solution B uses Luau "
            "string interpolation which is cleaner, handles tostring implicitly, "
            "and is slightly faster (single allocation)."
        ),
        "reasoning": (
            "Solution B is better because: string interpolation is more readable, "
            "handles type coercion automatically, creates fewer intermediate "
            "strings, and is idiomatic modern Luau."
        ),
    },
    {
        "difficulty": 5,
        "task": "Implement a particle system update loop",
        "solution_a": textwrap.dedent("""\
            RunService.Heartbeat:Connect(function(dt)
                for i, particle in pairs(particles) do
                    local vel = particle.velocity
                    local pos = particle.position
                    particle.velocity = Vector3.new(vel.X, vel.Y - 9.8 * dt, vel.Z)
                    particle.position = Vector3.new(
                        pos.X + vel.X * dt,
                        pos.Y + vel.Y * dt,
                        pos.Z + vel.Z * dt
                    )
                    if particle.position.Y < 0 then
                        table.remove(particles, i)
                    end
                end
            end)"""),
        "solution_b": textwrap.dedent("""\
            --!strict
            type Particle = { position: vector, velocity: vector, alive: boolean }

            local GRAVITY: vector = vector.create(0, -9.8, 0)
            local particles: {Particle} = table.create(256)
            local count: number = 0

            @native
            local function updateParticles(dt: number)
                local writeIdx = 0
                for i = 1, count do
                    local p = particles[i]
                    p.velocity += GRAVITY * dt
                    p.position += p.velocity * dt

                    if p.position.y >= 0 then
                        writeIdx += 1
                        particles[writeIdx] = p
                    end
                end
                -- Clear dead slots
                for i = writeIdx + 1, count do
                    particles[i] = nil :: any
                end
                count = writeIdx
            end

            RunService.Heartbeat:Connect(updateParticles)"""),
        "better": "B",
        "thinking": (
            "Solution A: uses pairs() on array, Vector3.new in hot loop (heap "
            "alloc), manual component extraction, table.remove inside iteration "
            "(shifts array, O(n^2)), no @native, no types. Solution B: native "
            "vector type (stack-allocated SIMD), @native for NCG, swap-remove "
            "compaction (O(n)), pre-allocated array, full type annotations."
        ),
        "reasoning": (
            "Solution B is better because: vector.create uses stack-allocated "
            "SIMD values (zero GC pressure), @native enables NCG compilation, "
            "swap-remove compaction is O(n) vs O(n^2) table.remove, table.create "
            "pre-allocates, and full types enable strict checking + NCG optimization."
        ),
    },
    {
        "difficulty": 4,
        "task": "Set up a service module for managing grapple anchors",
        "solution_a": textwrap.dedent("""\
            local anchors = {}

            workspace.DescendantAdded:Connect(function(d)
                if d.Name == "GrappleAnchor" then
                    table.insert(anchors, d)
                end
            end)

            return {
                getAnchors = function() return anchors end
            }"""),
        "solution_b": textwrap.dedent("""\
            --!strict
            local CollectionService = game:GetService("CollectionService")

            local TAG = "GrappleAnchor"

            local GrappleAnchorService = {}
            local anchors: {BasePart} = {}

            function GrappleAnchorService:Init()
                -- No side effects, just setup state
            end

            function GrappleAnchorService:Start()
                local function onAdded(inst: Instance)
                    if inst:IsA("BasePart") then
                        table.insert(anchors, inst)
                    end
                end
                local function onRemoved(inst: Instance)
                    local idx = table.find(anchors, inst)
                    if idx then
                        local n = #anchors
                        anchors[idx] = anchors[n]
                        anchors[n] = nil
                    end
                end

                CollectionService:GetInstanceAddedSignal(TAG):Connect(onAdded)
                CollectionService:GetInstanceRemovedSignal(TAG):Connect(onRemoved)
                for _, inst in CollectionService:GetTagged(TAG) do
                    onAdded(inst)
                end
            end

            function GrappleAnchorService:GetAnchors(): {BasePart}
                return anchors
            end

            return GrappleAnchorService"""),
        "better": "B",
        "thinking": (
            "Solution A: side effects on require, name-based discovery (fragile), "
            "no removal handling, no Init/Start, no types. Solution B: proper "
            "Init/Start lifecycle, CollectionService tags (robust discovery), "
            "handles removal with O(1) swap-remove, full types, --!strict."
        ),
        "reasoning": (
            "Solution B is better because: follows Init/Start lifecycle (no "
            "require-time side effects), uses CollectionService tags instead of "
            "name matching, handles instance removal (prevents leaks), uses O(1) "
            "swap-remove, and has full type annotations with --!strict."
        ),
    },
    {
        "difficulty": 3,
        "task": "Check if a value exists in a list",
        "solution_a": textwrap.dedent("""\
            local function contains(list, value)
                for _, item in pairs(list) do
                    if item == value then return true end
                end
                return false
            end"""),
        "solution_b": textwrap.dedent("""\
            --!strict
            local function contains(list: {string}, value: string): boolean
                return table.find(list, value) ~= nil
            end"""),
        "better": "B",
        "thinking": (
            "Solution A reimplements table.find with a manual loop and pairs(). "
            "Solution B uses the built-in table.find which is implemented in C "
            "and is faster. Also has type annotations and --!strict."
        ),
        "reasoning": (
            "Solution B is better because: table.find is a C-implemented builtin "
            "(faster than Luau loops), the code is more concise, and it has proper "
            "type annotations with --!strict."
        ),
    },
]


# ---------------------------------------------------------------------------
# Build JSONL
# ---------------------------------------------------------------------------


def build_critique(entry: dict) -> dict:
    assistant_content = (
        f"<think>\n{entry['thinking']}\n</think>\n\n"
        f"**Issue:** {entry['issue']}\n\n"
        f"**Minimal fix:**\n```luau\n{entry['minimal_fix'].rstrip()}\n```\n\n"
        f"**Production fix:**\n```luau\n{entry['production_fix'].rstrip()}\n```"
    )
    return {
        "messages": [
            {"role": "system", "content": CRITIC_SYSTEM},
            {
                "role": "user",
                "content": f"Review this code and identify issues:\n```luau\n{entry['user_code'].rstrip()}\n```",
            },
            {"role": "assistant", "content": assistant_content},
        ],
        "source": "critic_repair",
        "category": "critic",
        "difficulty": entry["difficulty"],
        "has_reasoning": True,
        "verified": False,
        "provenance": make_provenance("critic"),
    }


def build_coach(entry: dict) -> dict:
    assistant_content = (
        f"<think>\n{entry['thinking']}\n</think>\n\n"
        f"**One sentence:** {entry['one_sentence']}\n\n"
        f"**Minimal fix:**\n```luau\n{entry['minimal_fix'].rstrip()}\n```\n\n"
        f"**Production fix:**\n```luau\n{entry['production_fix'].rstrip()}\n```\n\n"
        f"**Test that catches it:**\n```luau\n{entry['test_code'].rstrip()}\n```"
    )
    return {
        "messages": [
            {"role": "system", "content": COACH_SYSTEM},
            {"role": "user", "content": entry["question"]},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": "critic_repair",
        "category": "critic",
        "difficulty": entry["difficulty"],
        "has_reasoning": True,
        "verified": False,
        "provenance": make_provenance("critic"),
    }


def build_preference(entry: dict) -> dict:
    assistant_content = f"<think>\n{entry['thinking']}\n</think>\n\n{entry['reasoning']}"
    return {
        "messages": [
            {"role": "system", "content": PREFERENCE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Task: {entry['task']}\n\n"
                    f"Solution A:\n```luau\n{entry['solution_a'].rstrip()}\n```\n\n"
                    f"Solution B:\n```luau\n{entry['solution_b'].rstrip()}\n```\n\n"
                    f"Which is better and why?"
                ),
            },
            {"role": "assistant", "content": assistant_content},
        ],
        "source": "critic_repair",
        "category": "preference",
        "difficulty": entry["difficulty"],
        "has_reasoning": True,
        "verified": False,
        "provenance": make_provenance("preference"),
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []

    for entry in CRITIQUE_REWRITE:
        examples.append(build_critique(entry))

    for entry in COACH_EXAMPLES:
        examples.append(build_coach(entry))

    for entry in PREFERENCE_PAIRS:
        examples.append(build_preference(entry))

    with OUTPUT.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Summary
    critique_count = len(CRITIQUE_REWRITE)
    coach_count = len(COACH_EXAMPLES)
    preference_count = len(PREFERENCE_PAIRS)

    print(f"Wrote {len(examples)} examples to {OUTPUT}")
    print(f"  Critique/rewrite: {critique_count}")
    print(f"  Coach:            {coach_count}")
    print(f"  Preference:       {preference_count}")

    # Tag breakdown
    tags: dict[str, int] = {}
    for entry in CRITIQUE_REWRITE:
        tag = entry["tag"]
        tags[tag] = tags.get(tag, 0) + 1
    print("Critique tags:")
    for tag, count in sorted(tags.items()):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
