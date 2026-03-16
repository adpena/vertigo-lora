#!/usr/bin/env python3
from __future__ import annotations

"""
Extract training data from Roblox API documentation patterns.

Generates instruction/completion pairs for all major Roblox services
that Vertigo uses, covering:
- Basic usage patterns
- Error handling (pcall wrapping)
- Common mistakes and corrections
- Integration with Vertigo's architecture (Init/Start, server-authoritative)
- Performance considerations
- Deprecated → modern API migrations

Output: data/raw/api_docs.jsonl
"""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "api_docs.jsonl"

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


def _ex(
    instruction: str,
    completion: str,
    category: str = "api_usage",
    difficulty: int = 2,
) -> dict:
    """Build a single training example dict."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": completion},
        ],
        "source": "roblox_api_docs",
        "category": category,
        "has_reasoning": True,
        "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# DataStoreService examples
# ---------------------------------------------------------------------------


def _datastore_examples() -> list[dict]:
    return [
        _ex(
            "How do I use DataStoreService:GetAsync to load player data in Vertigo?",
            "<think>\n"
            "DataStoreService:GetAsync retrieves data by key. In Vertigo, player data "
            "is keyed by UserId. We must wrap in pcall because DataStore calls can fail "
            "due to throttling or network issues. The server is authoritative for all "
            "player data — never trust client-sent profiles.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n'
            'local Players = game:GetService("Players")\n\n'
            'local playerStore = DataStoreService:GetDataStore("PlayerData_v2")\n\n'
            "local function loadPlayerData(player: Player): { [string]: any }?\n"
            '\tlocal key = "player_" .. tostring(player.UserId)\n'
            "\tlocal success, result = pcall(function()\n"
            "\t\treturn playerStore:GetAsync(key)\n"
            "\tend)\n\n"
            "\tif success then\n"
            "\t\treturn result or {}\n"
            "\telse\n"
            '\t\twarn("[DataService] GetAsync failed for", player.Name, ":", result)\n'
            "\t\treturn nil\n"
            "\tend\n"
            "end\n",
            difficulty=2,
        ),
        _ex(
            "How do I safely save player data with DataStoreService:SetAsync?",
            "<think>\n"
            "SetAsync overwrites the entire key. We must pcall it because it can fail. "
            "In production Vertigo code, we prefer UpdateAsync for atomic read-modify-write, "
            "but SetAsync is appropriate for full overwrites on PlayerRemoving or periodic saves. "
            "Always save on the server — never let clients trigger saves directly.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            'local playerStore = DataStoreService:GetDataStore("PlayerData_v2")\n\n'
            "local function savePlayerData(player: Player, data: { [string]: any }): boolean\n"
            '\tlocal key = "player_" .. tostring(player.UserId)\n'
            "\tlocal success, err = pcall(function()\n"
            "\t\tplayerStore:SetAsync(key, data)\n"
            "\tend)\n\n"
            "\tif not success then\n"
            '\t\twarn("[DataService] SetAsync failed for", player.Name, ":", err)\n'
            "\tend\n"
            "\treturn success\n"
            "end\n",
            difficulty=2,
        ),
        _ex(
            "Show me how to use DataStoreService:UpdateAsync for atomic updates in Vertigo.",
            "<think>\n"
            "UpdateAsync does an atomic read-modify-write. The transform function receives "
            "the current value and returns the new value. If it returns nil, the write is "
            "cancelled. This is critical for currency, inventory, or any state where two "
            "servers might write simultaneously. Always pcall — DataStore calls can throw.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            'local currencyStore = DataStoreService:GetDataStore("Currency_v1")\n\n'
            "local function addCoins(userId: number, amount: number): boolean\n"
            '\tlocal key = "coins_" .. tostring(userId)\n'
            "\tlocal success, err = pcall(function()\n"
            "\t\tcurrencyStore:UpdateAsync(key, function(currentValue: number?)\n"
            "\t\t\tlocal current = currentValue or 0\n"
            "\t\t\treturn current + amount\n"
            "\t\tend)\n"
            "\tend)\n\n"
            "\tif not success then\n"
            '\t\twarn("[DataService] UpdateAsync failed:", err)\n'
            "\tend\n"
            "\treturn success\n"
            "end\n",
            difficulty=3,
        ),
        _ex(
            "How do I use DataStoreService:RemoveAsync?",
            "<think>\n"
            "RemoveAsync deletes a key from the DataStore. Use cases include GDPR data "
            "deletion requests or clearing stale entries. Like all DataStore methods, it "
            "must be pcall-wrapped. It returns the last value before deletion, which is "
            "useful for logging.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            'local playerStore = DataStoreService:GetDataStore("PlayerData_v2")\n\n'
            "local function clearPlayerData(userId: number): boolean\n"
            '\tlocal key = "player_" .. tostring(userId)\n'
            "\tlocal success, lastValue = pcall(function()\n"
            "\t\treturn playerStore:RemoveAsync(key)\n"
            "\tend)\n\n"
            "\tif success then\n"
            '\t\tprint("[DataService] Cleared data for", userId)\n'
            "\telse\n"
            '\t\twarn("[DataService] RemoveAsync failed:", lastValue)\n'
            "\tend\n"
            "\treturn success\n"
            "end\n",
            difficulty=2,
        ),
        _ex(
            "How do I use OrderedDataStore for a leaderboard in Vertigo?",
            "<think>\n"
            "OrderedDataStore stores integer values and supports sorted queries via "
            "GetSortedAsync. This is ideal for leaderboards. GetSortedAsync returns a "
            "DataStorePages object that we iterate with GetCurrentPage/AdvanceToNextPageAsync. "
            "We sort descending for top scores and limit page size for performance.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            'local leaderboard = DataStoreService:GetOrderedDataStore("Leaderboard_v1")\n\n'
            "type LeaderboardEntry = {\n"
            "\tkey: string,\n"
            "\tvalue: number,\n"
            "}\n\n"
            "local function getTopPlayers(count: number): { LeaderboardEntry }\n"
            "\tlocal success, pages = pcall(function()\n"
            "\t\treturn leaderboard:GetSortedAsync(false, count)\n"
            "\tend)\n\n"
            "\tif not success then\n"
            '\t\twarn("[Leaderboard] GetSortedAsync failed:", pages)\n'
            "\t\treturn {}\n"
            "\tend\n\n"
            "\tlocal entries: { LeaderboardEntry } = {}\n"
            "\tlocal page = pages:GetCurrentPage()\n"
            "\tfor _, entry in page do\n"
            "\t\ttable.insert(entries, {\n"
            "\t\t\tkey = entry.key,\n"
            "\t\t\tvalue = entry.value,\n"
            "\t\t})\n"
            "\tend\n"
            "\treturn entries\n"
            "end\n\n"
            "local function setPlayerScore(userId: number, score: number): boolean\n"
            "\tlocal key = tostring(userId)\n"
            "\tlocal success, err = pcall(function()\n"
            "\t\tleaderboard:SetAsync(key, score)\n"
            "\tend)\n"
            "\tif not success then\n"
            '\t\twarn("[Leaderboard] SetAsync failed:", err)\n'
            "\tend\n"
            "\treturn success\n"
            "end\n",
            difficulty=3,
        ),
        _ex(
            "How do I implement DataStore retry logic with exponential backoff in Vertigo?",
            "<think>\n"
            "DataStore calls can fail due to throttling (error code 304). We should retry "
            "with exponential backoff. Use task.wait() (not wait()) for the delay. "
            "Cap retries to avoid infinite loops. This is critical for PlayerRemoving saves "
            "where data loss is unacceptable.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            "local MAX_RETRIES = 5\n"
            "local BASE_DELAY = 1\n\n"
            'local playerStore = DataStoreService:GetDataStore("PlayerData_v2")\n\n'
            "local function saveWithRetry(key: string, data: { [string]: any }): boolean\n"
            "\tfor attempt = 1, MAX_RETRIES do\n"
            "\t\tlocal success, err = pcall(function()\n"
            "\t\t\tplayerStore:SetAsync(key, data)\n"
            "\t\tend)\n\n"
            "\t\tif success then\n"
            "\t\t\treturn true\n"
            "\t\tend\n\n"
            "\t\twarn(string.format(\n"
            '\t\t\t"[DataService] Attempt %d/%d failed for %s: %s",\n'
            "\t\t\tattempt, MAX_RETRIES, key, tostring(err)\n"
            "\t\t))\n\n"
            "\t\tif attempt < MAX_RETRIES then\n"
            "\t\t\tlocal delay = BASE_DELAY * (2 ^ (attempt - 1))\n"
            "\t\t\ttask.wait(delay)\n"
            "\t\tend\n"
            "\tend\n"
            "\treturn false\n"
            "end\n",
            difficulty=3,
            category="service",
        ),
        _ex(
            "How do I integrate DataStoreService into a Vertigo service with Init/Start lifecycle?",
            "<think>\n"
            "In Vertigo, DataStore access is centralized in a DataService following the "
            "two-phase boot pattern. Init sets up the store references and cache table. "
            "Start connects PlayerAdded/PlayerRemoving to load/save. This keeps DataStore "
            "logic server-authoritative and avoids scattered GetDataStore calls.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n'
            'local Players = game:GetService("Players")\n\n'
            "local DataService = {}\n\n"
            "type PlayerProfile = {\n"
            "\tcoins: number,\n"
            "\tunlockedAbilities: { string },\n"
            "\tschemaVersion: number,\n"
            "}\n\n"
            "local DEFAULT_PROFILE: PlayerProfile = table.freeze({\n"
            "\tcoins = 0,\n"
            "\tunlockedAbilities = {},\n"
            "\tschemaVersion = 2,\n"
            "})\n\n"
            "function DataService:Init()\n"
            '\tself._store = DataStoreService:GetDataStore("PlayerData_v2")\n'
            "\tself._profiles = {} :: { [Player]: PlayerProfile }\n"
            "end\n\n"
            "function DataService:Start()\n"
            "\tPlayers.PlayerAdded:Connect(function(player: Player)\n"
            "\t\tself:_loadProfile(player)\n"
            "\tend)\n\n"
            "\tPlayers.PlayerRemoving:Connect(function(player: Player)\n"
            "\t\tself:_saveProfile(player)\n"
            "\t\tself._profiles[player] = nil\n"
            "\tend)\n\n"
            "\t-- Handle players already in game (late server start)\n"
            "\tfor _, player in Players:GetPlayers() do\n"
            "\t\ttask.spawn(self._loadProfile, self, player)\n"
            "\tend\n"
            "end\n\n"
            "function DataService:GetProfile(player: Player): PlayerProfile?\n"
            "\treturn self._profiles[player]\n"
            "end\n\n"
            "function DataService:_loadProfile(player: Player)\n"
            '\tlocal key = "player_" .. tostring(player.UserId)\n'
            "\tlocal success, data = pcall(function()\n"
            "\t\treturn self._store:GetAsync(key)\n"
            "\tend)\n\n"
            "\tif success then\n"
            "\t\tself._profiles[player] = data or table.clone(DEFAULT_PROFILE)\n"
            "\telse\n"
            '\t\twarn("[DataService] Failed to load", player.Name, ":", data)\n'
            "\t\tself._profiles[player] = table.clone(DEFAULT_PROFILE)\n"
            "\tend\n"
            "end\n\n"
            "function DataService:_saveProfile(player: Player)\n"
            "\tlocal profile = self._profiles[player]\n"
            "\tif not profile then return end\n\n"
            '\tlocal key = "player_" .. tostring(player.UserId)\n'
            "\tlocal success, err = pcall(function()\n"
            "\t\tself._store:SetAsync(key, profile)\n"
            "\tend)\n\n"
            "\tif not success then\n"
            '\t\twarn("[DataService] Failed to save", player.Name, ":", err)\n'
            "\tend\n"
            "end\n\n"
            "return DataService\n",
            difficulty=4,
            category="service",
        ),
    ]


# ---------------------------------------------------------------------------
# Players examples
# ---------------------------------------------------------------------------


def _players_examples() -> list[dict]:
    return [
        _ex(
            "How do I use Players.PlayerAdded and PlayerRemoving in Vertigo?",
            "<think>\n"
            "Players.PlayerAdded fires when a new player joins. PlayerRemoving fires just "
            "before they leave — this is where you save data and clean up. In Vertigo, "
            "these connections live in :Start() since they produce side effects. We also "
            "handle players who joined before the service started with GetPlayers().\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "local SessionService = {}\n\n"
            "function SessionService:Init()\n"
            "\tself._sessions = {} :: { [Player]: { joinTime: number } }\n"
            "end\n\n"
            "function SessionService:Start()\n"
            "\tPlayers.PlayerAdded:Connect(function(player: Player)\n"
            "\t\tself._sessions[player] = {\n"
            "\t\t\tjoinTime = os.clock(),\n"
            "\t\t}\n"
            '\t\tprint("[Session] Player joined:", player.Name)\n'
            "\tend)\n\n"
            "\tPlayers.PlayerRemoving:Connect(function(player: Player)\n"
            "\t\tlocal session = self._sessions[player]\n"
            "\t\tif session then\n"
            "\t\t\tlocal duration = os.clock() - session.joinTime\n"
            '\t\t\tprint(string.format("[Session] %s left after %.1fs", player.Name, duration))\n'
            "\t\tend\n"
            "\t\tself._sessions[player] = nil\n"
            "\tend)\n\n"
            "\tfor _, player in Players:GetPlayers() do\n"
            "\t\tself._sessions[player] = { joinTime = os.clock() }\n"
            "\tend\n"
            "end\n\n"
            "return SessionService\n",
            difficulty=2,
            category="service",
        ),
        _ex(
            "How do I get all currently connected players?",
            "<think>\n"
            "Players:GetPlayers() returns an array of all Player objects currently in the "
            "server. This is useful for broadcasting state, periodic saves, or iterating "
            "over all connected users. Use ipairs or the generalized for loop.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "local function broadcastMessage(message: string)\n"
            "\tfor _, player in Players:GetPlayers() do\n"
            "\t\t-- Send to each player's UI\n"
            '\t\tprint("[Broadcast] ->", player.Name, ":", message)\n'
            "\tend\n"
            "end\n\n"
            "local function getPlayerCount(): number\n"
            "\treturn #Players:GetPlayers()\n"
            "end\n",
            difficulty=1,
        ),
        _ex(
            "How do I safely get a player's character and HumanoidRootPart?",
            "<think>\n"
            "Player.Character can be nil if the player hasn't spawned yet. "
            "HumanoidRootPart can also be nil during respawn. Always nil-check both. "
            "Use CharacterAdded to wait for spawning. For Vertigo server validation, "
            "always read position from the server's view of the character, never trust "
            "client-reported positions.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "local function getPlayerPosition(player: Player): Vector3?\n"
            "\tlocal character = player.Character\n"
            "\tif not character then\n"
            "\t\treturn nil\n"
            "\tend\n\n"
            '\tlocal rootPart = character:FindFirstChild("HumanoidRootPart") :: BasePart?\n'
            "\tif not rootPart then\n"
            "\t\treturn nil\n"
            "\tend\n\n"
            "\treturn rootPart.Position\n"
            "end\n\n"
            "local function onCharacterAdded(player: Player, character: Model)\n"
            '\tlocal humanoid = character:WaitForChild("Humanoid") :: Humanoid\n'
            "\thumanoid.Died:Connect(function()\n"
            '\t\tprint("[Game]", player.Name, "died")\n'
            "\tend)\n"
            "end\n\n"
            "Players.PlayerAdded:Connect(function(player: Player)\n"
            "\tplayer.CharacterAdded:Connect(function(character: Model)\n"
            "\t\tonCharacterAdded(player, character)\n"
            "\tend)\n"
            "end)\n",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# RunService examples
# ---------------------------------------------------------------------------


def _runservice_examples() -> list[dict]:
    return [
        _ex(
            "How do I use RunService.Heartbeat for a per-frame server update in Vertigo?",
            "<think>\n"
            "RunService.Heartbeat fires every frame after physics simulation. On the server "
            "this runs at ~60Hz. For Vertigo, we use @native on the callback function and "
            "extract it to module scope to avoid closures in the connection. The dt parameter "
            "gives delta time for frame-rate independent logic.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n\n'
            "local UpdateService = {}\n\n"
            "local accumulated = 0\n"
            "local TICK_RATE = 1 / 20 -- 20Hz fixed step\n\n"
            "--[[@native]]\n"
            "local function fixedUpdate(dt: number)\n"
            "\t-- Fixed-step physics or game logic\n"
            "\taccumulated += dt\n"
            "\twhile accumulated >= TICK_RATE do\n"
            "\t\taccumulated -= TICK_RATE\n"
            "\t\t-- Run fixed-step logic here\n"
            "\tend\n"
            "end\n\n"
            "function UpdateService:Init()\n"
            "\tself._connection = nil :: RBXScriptConnection?\n"
            "end\n\n"
            "function UpdateService:Start()\n"
            "\tself._connection = RunService.Heartbeat:Connect(fixedUpdate)\n"
            "end\n\n"
            "return UpdateService\n",
            difficulty=3,
            category="service",
        ),
        _ex(
            "How do I use RunService.RenderStepped for client-side rendering in Vertigo?",
            "<think>\n"
            "RenderStepped fires every frame BEFORE rendering on the client only. It's ideal "
            "for camera updates, UI animations, and visual-only effects. Never do heavy "
            "computation here — it blocks rendering. In Vertigo, camera controllers and VFX "
            "use RenderStepped with @native for NCG optimization.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n'
            'local Players = game:GetService("Players")\n\n'
            "local CameraController = {}\n\n"
            "local localPlayer = Players.LocalPlayer\n"
            "local camera = workspace.CurrentCamera\n\n"
            "--[[@native]]\n"
            "local function updateCamera(dt: number)\n"
            "\tlocal character = localPlayer.Character\n"
            "\tif not character then return end\n\n"
            '\tlocal rootPart = character:FindFirstChild("HumanoidRootPart") :: BasePart?\n'
            "\tif not rootPart then return end\n\n"
            "\tlocal targetCF = rootPart.CFrame * CFrame.new(0, 5, 12)\n"
            "\tcamera.CFrame = camera.CFrame:Lerp(targetCF, 1 - math.exp(-10 * dt))\n"
            "end\n\n"
            "function CameraController:Init()\n"
            "\tself._connection = nil :: RBXScriptConnection?\n"
            "end\n\n"
            "function CameraController:Start()\n"
            "\tself._connection = RunService.RenderStepped:Connect(updateCamera)\n"
            "end\n\n"
            "return CameraController\n",
            difficulty=3,
            category="controller",
        ),
        _ex(
            "How do I check if code is running on the server or client?",
            "<think>\n"
            "RunService:IsServer() and RunService:IsClient() determine execution context. "
            "In Vertigo's architecture, server code lives in src/Server/ and client code "
            "in src/Client/, so context is usually clear. But shared modules in "
            "ReplicatedStorage may need runtime checks to branch behavior.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n\n'
            "local function getContext(): string\n"
            "\tif RunService:IsServer() then\n"
            '\t\treturn "server"\n'
            "\telseif RunService:IsClient() then\n"
            '\t\treturn "client"\n'
            "\telse\n"
            '\t\treturn "unknown"\n'
            "\tend\n"
            "end\n\n"
            "-- Shared module with context-dependent behavior\n"
            "local SharedModule = {}\n\n"
            "function SharedModule:Init()\n"
            "\tif RunService:IsServer() then\n"
            "\t\tself._authority = true\n"
            "\telse\n"
            "\t\tself._authority = false\n"
            "\tend\n"
            "end\n\n"
            "return SharedModule\n",
            difficulty=1,
        ),
        _ex(
            "How do I use RunService.Stepped for pre-physics updates?",
            "<think>\n"
            "RunService.Stepped fires every frame BEFORE physics simulation. The callback "
            "receives (time, dt). This is useful for applying forces or setting velocities "
            "before the physics engine processes them. In Vertigo, traversal forces "
            "(grapple pull, glide lift) are applied in Stepped.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n\n'
            "--[[@native]]\n"
            "local function prePhysicsUpdate(_time: number, dt: number)\n"
            "\t-- Apply forces before physics simulation\n"
            "\t-- This ensures forces affect the current frame's simulation\n"
            "end\n\n"
            "RunService.Stepped:Connect(prePhysicsUpdate)\n",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# ReplicatedStorage examples
# ---------------------------------------------------------------------------


def _replicated_storage_examples() -> list[dict]:
    return [
        _ex(
            "How do I use ReplicatedStorage:WaitForChild for safe module requires in Vertigo?",
            "<think>\n"
            "WaitForChild yields until the child exists, which is critical during loading. "
            'In Vertigo, all shared requires anchor on ReplicatedStorage:WaitForChild("Shared"). '
            "This ensures modules are available before requiring them. Always use WaitForChild "
            "for cross-boundary references, never FindFirstChild which can return nil.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            "-- Standard Vertigo require pattern\n"
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
            "local Config = require(Shared.Config.Abilities)\n"
            "local Types = require(Shared.Util.Types)\n"
            "local Remotes = require(Shared.Net.Remotes)\n\n"
            "-- WaitForChild with timeout (5 second default)\n"
            'local OptionalModule = Shared:WaitForChild("OptionalFeature", 5)\n'
            "if OptionalModule then\n"
            "\tlocal module = require(OptionalModule)\n"
            "end\n",
            difficulty=1,
        ),
        _ex(
            "What is the difference between WaitForChild and FindFirstChild?",
            "<think>\n"
            "FindFirstChild returns the child immediately or nil if it doesn't exist — it "
            "never yields. WaitForChild yields the current thread until the child appears. "
            "In Vertigo, we use WaitForChild during Init/boot to guarantee dependencies "
            "exist. FindFirstChild is used for optional lookups where nil is acceptable "
            "(e.g., checking if a character has a tool equipped).\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            "-- WaitForChild: YIELDS until child exists (use during boot)\n"
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n\n'
            "-- FindFirstChild: returns immediately, may be nil\n"
            "local function hasWeapon(character: Model): boolean\n"
            '\tlocal tool = character:FindFirstChild("Weapon")\n'
            "\treturn tool ~= nil\n"
            "end\n\n"
            "-- WaitForChild with timeout: yields up to N seconds, returns nil on timeout\n"
            'local maybeChild = Shared:WaitForChild("LateLoadedModule", 10)\n'
            "if not maybeChild then\n"
            '\twarn("[Boot] LateLoadedModule did not appear within 10s")\n'
            "end\n",
            difficulty=1,
        ),
    ]


# ---------------------------------------------------------------------------
# CollectionService examples
# ---------------------------------------------------------------------------


def _collection_service_examples() -> list[dict]:
    return [
        _ex(
            "How do I use CollectionService:GetTagged to find all grapple anchors in Vertigo?",
            "<think>\n"
            "CollectionService tags are Vertigo's primary mechanism for runtime object "
            "discovery. Instead of path-based lookups, we tag parts in builders and query "
            "them at runtime. GetTagged returns all instances with that tag. We use this "
            "for GrappleAnchor, PressurePlate, BloomCrystal, etc.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n\n'
            "local function getAllAnchors(): { BasePart }\n"
            '\treturn CollectionService:GetTagged("GrappleAnchor") :: { BasePart }\n'
            "end\n\n"
            "-- Listen for new anchors added at runtime (e.g., from zone streaming)\n"
            "local function onAnchorAdded(instance: Instance)\n"
            "\tlocal part = instance :: BasePart\n"
            '\tprint("[Grapple] New anchor at", part.Position)\n'
            "end\n\n"
            'CollectionService:GetInstanceAddedSignal("GrappleAnchor"):Connect(onAnchorAdded)\n'
            'CollectionService:GetInstanceRemovedSignal("GrappleAnchor"):Connect(function(instance: Instance)\n'
            '\tprint("[Grapple] Anchor removed")\n'
            "end)\n",
            difficulty=2,
        ),
        _ex(
            "How do I add and remove CollectionService tags?",
            "<think>\n"
            "AddTag/RemoveTag/HasTag manage tags on instances. In Vertigo builders, we "
            "tag parts during procedural generation so services can discover them. Tags "
            "persist across the session but not across server restarts unless set in Studio. "
            "HasTag is O(1) and safe to call in hot paths.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n\n'
            "-- In a builder: tag generated parts\n"
            "local function tagPart(part: BasePart, tag: string)\n"
            "\tCollectionService:AddTag(part, tag)\n"
            "end\n\n"
            "-- Check if a part has a specific tag\n"
            "local function isGrappleAnchor(part: BasePart): boolean\n"
            '\treturn CollectionService:HasTag(part, "GrappleAnchor")\n'
            "end\n\n"
            "-- Remove a tag (e.g., when a destructible object is destroyed)\n"
            "local function destroyAnchor(part: BasePart)\n"
            '\tCollectionService:RemoveTag(part, "GrappleAnchor")\n'
            "\tpart:Destroy()\n"
            "end\n\n"
            "-- Get all tags on an instance\n"
            "local function debugTags(instance: Instance)\n"
            "\tlocal tags = CollectionService:GetTags(instance)\n"
            '\tprint("Tags:", table.concat(tags, ", "))\n'
            "end\n",
            difficulty=2,
        ),
        _ex(
            "How do I build a tag-based system with CollectionService in Vertigo's Init/Start pattern?",
            "<think>\n"
            "A common Vertigo pattern: a service discovers tagged objects during Start, "
            "then listens for future additions/removals. This is used for interactive "
            "objects like pressure plates, crystals, and NPCs. The Init phase sets up "
            "internal data structures, Start queries existing tagged objects and connects "
            "signals.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n\n'
            'local TAG = "PressurePlate"\n\n'
            "local PressurePlateService = {}\n\n"
            "type PlateState = {\n"
            "\tpart: BasePart,\n"
            "\tactivated: boolean,\n"
            "\tconnection: RBXScriptConnection?,\n"
            "}\n\n"
            "function PressurePlateService:Init()\n"
            "\tself._plates = {} :: { [BasePart]: PlateState }\n"
            "end\n\n"
            "function PressurePlateService:Start()\n"
            "\t-- Register existing plates\n"
            "\tfor _, instance in CollectionService:GetTagged(TAG) do\n"
            "\t\tself:_registerPlate(instance :: BasePart)\n"
            "\tend\n\n"
            "\t-- Listen for new plates (zone streaming)\n"
            "\tCollectionService:GetInstanceAddedSignal(TAG):Connect(function(instance: Instance)\n"
            "\t\tself:_registerPlate(instance :: BasePart)\n"
            "\tend)\n\n"
            "\tCollectionService:GetInstanceRemovedSignal(TAG):Connect(function(instance: Instance)\n"
            "\t\tself:_unregisterPlate(instance :: BasePart)\n"
            "\tend)\n"
            "end\n\n"
            "function PressurePlateService:_registerPlate(part: BasePart)\n"
            "\tlocal connection = part.Touched:Connect(function(hit: BasePart)\n"
            "\t\tlocal state = self._plates[part]\n"
            "\t\tif state and not state.activated then\n"
            "\t\t\tstate.activated = true\n"
            '\t\t\tprint("[Plate] Activated:", part.Name)\n'
            "\t\tend\n"
            "\tend)\n\n"
            "\tself._plates[part] = {\n"
            "\t\tpart = part,\n"
            "\t\tactivated = false,\n"
            "\t\tconnection = connection,\n"
            "\t}\n"
            "end\n\n"
            "function PressurePlateService:_unregisterPlate(part: BasePart)\n"
            "\tlocal state = self._plates[part]\n"
            "\tif state and state.connection then\n"
            "\t\tstate.connection:Disconnect()\n"
            "\tend\n"
            "\tself._plates[part] = nil\n"
            "end\n\n"
            "return PressurePlateService\n",
            difficulty=4,
            category="service",
        ),
    ]


# ---------------------------------------------------------------------------
# UserInputService examples
# ---------------------------------------------------------------------------


def _userinput_examples() -> list[dict]:
    return [
        _ex(
            "How do I handle keyboard input with UserInputService in Vertigo?",
            "<think>\n"
            "UserInputService.InputBegan fires when a key is pressed or mouse button is "
            "clicked. The gameProcessedEvent parameter indicates if Roblox already handled "
            "it (e.g., typing in chat). In Vertigo, input handling lives in client "
            "controllers during Start(). We check gameProcessedEvent to avoid conflicts.\n"
            "</think>\n\n"
            "--!strict\n"
            'local UserInputService = game:GetService("UserInputService")\n\n'
            "local InputController = {}\n\n"
            "function InputController:Init()\n"
            "\tself._connections = {} :: { RBXScriptConnection }\n"
            "end\n\n"
            "function InputController:Start()\n"
            "\ttable.insert(self._connections, UserInputService.InputBegan:Connect(\n"
            "\t\tfunction(input: InputObject, gameProcessed: boolean)\n"
            "\t\t\tif gameProcessed then return end\n\n"
            "\t\t\tif input.KeyCode == Enum.KeyCode.E then\n"
            "\t\t\t\tself:_onInteract()\n"
            "\t\t\telseif input.KeyCode == Enum.KeyCode.Q then\n"
            "\t\t\t\tself:_onGlideToggle()\n"
            "\t\t\telseif input.UserInputType == Enum.UserInputType.MouseButton1 then\n"
            "\t\t\t\tself:_onGrapple()\n"
            "\t\t\tend\n"
            "\t\tend\n"
            "\t))\n\n"
            "\ttable.insert(self._connections, UserInputService.InputEnded:Connect(\n"
            "\t\tfunction(input: InputObject, gameProcessed: boolean)\n"
            "\t\t\tif gameProcessed then return end\n\n"
            "\t\t\tif input.UserInputType == Enum.UserInputType.MouseButton1 then\n"
            "\t\t\t\tself:_onGrappleRelease()\n"
            "\t\t\tend\n"
            "\t\tend\n"
            "\t))\n"
            "end\n\n"
            "return InputController\n",
            difficulty=2,
            category="controller",
        ),
        _ex(
            "How do I get the mouse position with UserInputService?",
            "<think>\n"
            "GetMouseLocation returns a Vector2 of the mouse position in screen space "
            "(pixels from top-left). This is useful for UI interactions and raycasting "
            "from the camera. Note: this accounts for the GUI inset (topbar), unlike "
            "Mouse.X/Mouse.Y which don't.\n"
            "</think>\n\n"
            "--!strict\n"
            'local UserInputService = game:GetService("UserInputService")\n\n'
            "local function getMouseScreenPosition(): Vector2\n"
            "\treturn UserInputService:GetMouseLocation()\n"
            "end\n\n"
            "-- Raycast from mouse position into the 3D world\n"
            "local function getMouseWorldRay(): Ray\n"
            "\tlocal mousePos = UserInputService:GetMouseLocation()\n"
            "\tlocal camera = workspace.CurrentCamera\n"
            "\tlocal ray = camera:ViewportPointToRay(mousePos.X, mousePos.Y)\n"
            "\treturn ray\n"
            "end\n",
            difficulty=2,
            category="controller",
        ),
    ]


# ---------------------------------------------------------------------------
# Workspace (raycasting) examples
# ---------------------------------------------------------------------------


def _workspace_examples() -> list[dict]:
    return [
        _ex(
            "How do I use workspace:Raycast in Vertigo?",
            "<think>\n"
            "workspace:Raycast is the modern raycasting API (replaces FindPartOnRay). "
            "It takes an origin, direction, and optional RaycastParams. Returns a "
            "RaycastResult or nil. In Vertigo, we use this for grapple anchor detection, "
            "ground checks, and line-of-sight validation.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local function raycast(\n"
            "\torigin: Vector3,\n"
            "\tdirection: Vector3,\n"
            "\tignoreList: { Instance }?\n"
            "): RaycastResult?\n"
            "\tlocal params = RaycastParams.new()\n"
            "\tparams.FilterType = Enum.RaycastFilterType.Exclude\n"
            "\tparams.FilterDescendantsInstances = ignoreList or {}\n"
            "\tparams.IgnoreWater = true\n\n"
            "\treturn workspace:Raycast(origin, direction, params)\n"
            "end\n\n"
            "-- Ground check example\n"
            "local function isGrounded(rootPart: BasePart): boolean\n"
            "\tlocal origin = rootPart.Position\n"
            "\tlocal direction = Vector3.new(0, -4, 0)\n"
            "\tlocal result = raycast(origin, direction, { rootPart.Parent :: Instance })\n"
            "\treturn result ~= nil\n"
            "end\n\n"
            "-- Grapple line-of-sight check\n"
            "local function hasLineOfSight(from: Vector3, to: Vector3, ignore: { Instance }): boolean\n"
            "\tlocal direction = to - from\n"
            "\tlocal result = raycast(from, direction, ignore)\n"
            "\tif not result then\n"
            "\t\treturn true -- nothing blocking\n"
            "\tend\n"
            "\t-- Check if we hit something before the target\n"
            "\treturn (result.Position - from).Magnitude >= (to - from).Magnitude - 0.5\n"
            "end\n",
            difficulty=3,
        ),
        _ex(
            "Why should I use workspace:Raycast instead of FindPartOnRay?",
            "<think>\n"
            "FindPartOnRay is deprecated. workspace:Raycast is the modern replacement. "
            "Key differences: Raycast uses RaycastParams (cleaner API), returns a typed "
            "RaycastResult, and has better performance. Always use Raycast in new code.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- DEPRECATED: Do not use FindPartOnRay\n"
            "--[[\n"
            "local ray = Ray.new(origin, direction)\n"
            "local part, position, normal = workspace:FindPartOnRay(ray, ignoreModel)\n"
            "]]\n\n"
            "-- CORRECT: Use workspace:Raycast\n"
            "local params = RaycastParams.new()\n"
            "params.FilterType = Enum.RaycastFilterType.Exclude\n"
            "params.FilterDescendantsInstances = { ignoreModel }\n\n"
            "local result = workspace:Raycast(origin, direction, params)\n"
            "if result then\n"
            "\tlocal part = result.Instance\n"
            "\tlocal position = result.Position\n"
            "\tlocal normal = result.Normal\n"
            "\tlocal material = result.Material\n"
            "end\n",
            difficulty=2,
            category="debugging",
        ),
    ]


# ---------------------------------------------------------------------------
# TweenService examples
# ---------------------------------------------------------------------------


def _tween_examples() -> list[dict]:
    return [
        _ex(
            "How do I use TweenService to animate properties in Vertigo?",
            "<think>\n"
            "TweenService:Create takes an instance, TweenInfo, and a property goal table. "
            "Call :Play() to start. Tweens are great for UI animations, door movements, "
            "and non-physics visual effects. For per-frame physics movement, use "
            "RunService.Heartbeat with math.lerp instead — tweens aren't frame-locked.\n"
            "</think>\n\n"
            "--!strict\n"
            'local TweenService = game:GetService("TweenService")\n\n'
            "local function tweenPartTo(part: BasePart, targetCF: CFrame, duration: number)\n"
            "\tlocal info = TweenInfo.new(\n"
            "\t\tduration,\n"
            "\t\tEnum.EasingStyle.Quad,\n"
            "\t\tEnum.EasingDirection.Out\n"
            "\t)\n\n"
            "\tlocal tween = TweenService:Create(part, info, {\n"
            "\t\tCFrame = targetCF,\n"
            "\t})\n\n"
            "\ttween:Play()\n"
            "\treturn tween\n"
            "end\n\n"
            "-- Tween transparency for fade-in effect\n"
            "local function fadeIn(part: BasePart, duration: number)\n"
            "\tpart.Transparency = 1\n"
            "\tlocal info = TweenInfo.new(duration, Enum.EasingStyle.Linear)\n"
            "\tlocal tween = TweenService:Create(part, info, { Transparency = 0 })\n"
            "\ttween:Play()\n"
            "\ttween.Completed:Wait()\n"
            "end\n",
            difficulty=2,
        ),
        _ex(
            "How do I chain tweens or wait for a tween to complete?",
            "<think>\n"
            "Tween.Completed is a signal that fires when the tween finishes. You can "
            ":Wait() on it to yield, or :Connect() for non-blocking. For chains, yield "
            "with Completed:Wait() between tweens. Use task.spawn to run chains without "
            "blocking the caller.\n"
            "</think>\n\n"
            "--!strict\n"
            'local TweenService = game:GetService("TweenService")\n\n'
            "local function animateDoor(door: BasePart)\n"
            "\tlocal openInfo = TweenInfo.new(0.5, Enum.EasingStyle.Back, Enum.EasingDirection.Out)\n"
            "\tlocal closeInfo = TweenInfo.new(0.3, Enum.EasingStyle.Quad, Enum.EasingDirection.In)\n\n"
            "\tlocal openCF = door.CFrame * CFrame.new(0, 5, 0)\n"
            "\tlocal closedCF = door.CFrame\n\n"
            "\t-- Open\n"
            "\tlocal openTween = TweenService:Create(door, openInfo, { CFrame = openCF })\n"
            "\topenTween:Play()\n"
            "\topenTween.Completed:Wait()\n\n"
            "\t-- Hold open\n"
            "\ttask.wait(3)\n\n"
            "\t-- Close\n"
            "\tlocal closeTween = TweenService:Create(door, closeInfo, { CFrame = closedCF })\n"
            "\tcloseTween:Play()\n"
            "\tcloseTween.Completed:Wait()\n"
            "end\n\n"
            "-- Non-blocking: run in a separate thread\n"
            "task.spawn(animateDoor, someDoor)\n",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# RemoteEvent / RemoteFunction examples
# ---------------------------------------------------------------------------


def _remote_examples() -> list[dict]:
    return [
        _ex(
            "How do I use RemoteEvent:FireServer from a Vertigo client controller?",
            "<think>\n"
            "In Vertigo, clients send requests via RemoteEvent:FireServer(). The server "
            "never trusts client data — it validates everything. Remotes are defined in "
            "Shared/Net/Remotes.luau and accessed via require. The naming convention is "
            "Request* for client→server and StateSync/EventNotify for server→client.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
            "local Remotes = require(Shared.Net.Remotes)\n\n"
            "local AbilityController = {}\n\n"
            "function AbilityController:Init()\n"
            "\tself._remote = nil :: RemoteEvent?\n"
            "end\n\n"
            "function AbilityController:Start()\n"
            '\tself._remote = Remotes.get("RequestUseAbility")\n'
            "end\n\n"
            "function AbilityController:RequestAbility(abilityId: string, targetPosition: Vector3?)\n"
            "\tlocal remote = self._remote\n"
            "\tif not remote then return end\n"
            "\tremote:FireServer(abilityId, targetPosition)\n"
            "end\n\n"
            "return AbilityController\n",
            difficulty=2,
            category="controller",
        ),
        _ex(
            "How do I handle RemoteEvent:OnServerEvent with validation in Vertigo?",
            "<think>\n"
            "OnServerEvent fires on the server when a client calls FireServer. The first "
            "argument is always the Player who sent it. In Vertigo, we validate every "
            "argument — never trust client input. Check types, ranges, cooldowns, and "
            "spatial constraints. Rate-limit to prevent spam.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
            "local Remotes = require(Shared.Net.Remotes)\n\n"
            "local AbilityService = {}\n\n"
            "function AbilityService:Init()\n"
            "\tself._cooldowns = {} :: { [Player]: { [string]: number } }\n"
            "end\n\n"
            "function AbilityService:Start()\n"
            '\tlocal remote = Remotes.get("RequestUseAbility")\n'
            "\tremote.OnServerEvent:Connect(function(player: Player, ...)\n"
            "\t\tself:_handleAbilityRequest(player, ...)\n"
            "\tend)\n"
            "end\n\n"
            "function AbilityService:_handleAbilityRequest(\n"
            "\tplayer: Player,\n"
            "\tabilityId: unknown,\n"
            "\ttargetPos: unknown\n"
            ")\n"
            "\t-- Type validation\n"
            '\tif typeof(abilityId) ~= "string" then return end\n'
            '\tif targetPos ~= nil and typeof(targetPos) ~= "Vector3" then return end\n\n'
            "\t-- Cooldown check\n"
            "\tlocal now = os.clock()\n"
            "\tlocal playerCDs = self._cooldowns[player]\n"
            "\tif playerCDs and playerCDs[abilityId :: string] then\n"
            "\t\tif (now - playerCDs[abilityId :: string]) < 0.3 then\n"
            "\t\t\treturn -- Still on cooldown\n"
            "\t\tend\n"
            "\tend\n\n"
            "\t-- Record cooldown\n"
            "\tif not self._cooldowns[player] then\n"
            "\t\tself._cooldowns[player] = {}\n"
            "\tend\n"
            "\tself._cooldowns[player][abilityId :: string] = now\n\n"
            "\t-- Execute ability\n"
            '\tprint("[Ability]", player.Name, "used", abilityId)\n'
            "end\n\n"
            "return AbilityService\n",
            difficulty=3,
            category="service",
        ),
        _ex(
            "How do I use RemoteEvent:FireClient and FireAllClients?",
            "<think>\n"
            "FireClient sends data to a specific client. FireAllClients broadcasts to "
            "all connected clients. In Vertigo, the server uses StateSync for targeted "
            "state updates and EventNotify for broadcasts. Only the server can call "
            "FireClient/FireAllClients.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local Players = game:GetService("Players")\n\n'
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
            "local Remotes = require(Shared.Net.Remotes)\n\n"
            "local NotifyService = {}\n\n"
            "function NotifyService:Init()\n"
            "\tself._stateSync = nil :: RemoteEvent?\n"
            "\tself._eventNotify = nil :: RemoteEvent?\n"
            "end\n\n"
            "function NotifyService:Start()\n"
            '\tself._stateSync = Remotes.get("StateSync")\n'
            '\tself._eventNotify = Remotes.get("EventNotify")\n'
            "end\n\n"
            "-- Send state update to one player\n"
            "function NotifyService:SyncState(player: Player, stateKey: string, value: any)\n"
            "\tlocal remote = self._stateSync\n"
            "\tif remote then\n"
            "\t\tremote:FireClient(player, stateKey, value)\n"
            "\tend\n"
            "end\n\n"
            "-- Broadcast event to all players\n"
            "function NotifyService:BroadcastEvent(eventType: string, data: any)\n"
            "\tlocal remote = self._eventNotify\n"
            "\tif remote then\n"
            "\t\tremote:FireAllClients(eventType, data)\n"
            "\tend\n"
            "end\n\n"
            "return NotifyService\n",
            difficulty=2,
            category="networking",
        ),
        _ex(
            "How do I listen for server events on the client with OnClientEvent?",
            "<think>\n"
            "OnClientEvent fires on the client when the server calls FireClient or "
            "FireAllClients. In Vertigo, client controllers connect to StateSync and "
            "EventNotify during Start() to receive state updates and global events.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
            "local Remotes = require(Shared.Net.Remotes)\n\n"
            "local StateController = {}\n\n"
            "function StateController:Init()\n"
            "\tself._state = {} :: { [string]: any }\n"
            "end\n\n"
            "function StateController:Start()\n"
            '\tlocal stateSync = Remotes.get("StateSync")\n'
            "\tstateSync.OnClientEvent:Connect(function(key: string, value: any)\n"
            "\t\tself._state[key] = value\n"
            "\t\tself:_onStateChanged(key, value)\n"
            "\tend)\n\n"
            '\tlocal eventNotify = Remotes.get("EventNotify")\n'
            "\teventNotify.OnClientEvent:Connect(function(eventType: string, data: any)\n"
            "\t\tself:_onEvent(eventType, data)\n"
            "\tend)\n"
            "end\n\n"
            "function StateController:_onStateChanged(key: string, value: any)\n"
            '\tprint("[State]", key, "=", value)\n'
            "end\n\n"
            "function StateController:_onEvent(eventType: string, data: any)\n"
            '\tprint("[Event]", eventType, data)\n'
            "end\n\n"
            "return StateController\n",
            difficulty=2,
            category="controller",
        ),
        _ex(
            "How do I use RemoteFunction for request/response patterns?",
            "<think>\n"
            "RemoteFunction allows clients to invoke the server and get a return value. "
            "InvokeServer yields until the server responds via OnServerInvoke. In Vertigo, "
            "GetPlayerProfile uses this pattern. Warning: if the server never returns, the "
            "client hangs — always set timeouts or use Promises. Never use OnClientInvoke "
            "as a malicious client can hang the server.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
            "local Remotes = require(Shared.Net.Remotes)\n\n"
            "-- SERVER: Handle the invoke\n"
            "local function setupProfileEndpoint()\n"
            '\tlocal remote = Remotes.get("GetPlayerProfile") :: RemoteFunction\n'
            "\tremote.OnServerInvoke = function(player: Player): { [string]: any }?\n"
            "\t\t-- Return the player's profile data\n"
            "\t\tlocal profile = DataService:GetProfile(player)\n"
            "\t\tif not profile then\n"
            "\t\t\treturn nil\n"
            "\t\tend\n"
            "\t\t-- Only return safe-to-share fields\n"
            "\t\treturn {\n"
            "\t\t\tcoins = profile.coins,\n"
            "\t\t\tunlockedAbilities = profile.unlockedAbilities,\n"
            "\t\t}\n"
            "\tend\n"
            "end\n\n"
            "-- CLIENT: Request profile\n"
            "local function requestProfile(): { [string]: any }?\n"
            '\tlocal remote = Remotes.get("GetPlayerProfile") :: RemoteFunction\n'
            "\tlocal success, result = pcall(function()\n"
            "\t\treturn remote:InvokeServer()\n"
            "\tend)\n"
            "\tif success then\n"
            "\t\treturn result\n"
            "\telse\n"
            '\t\twarn("[Profile] InvokeServer failed:", result)\n'
            "\t\treturn nil\n"
            "\tend\n"
            "end\n",
            difficulty=3,
            category="networking",
        ),
    ]


# ---------------------------------------------------------------------------
# HttpService examples
# ---------------------------------------------------------------------------


def _httpservice_examples() -> list[dict]:
    return [
        _ex(
            "How do I use HttpService:JSONEncode and JSONDecode?",
            "<think>\n"
            "HttpService:JSONEncode converts a Lua table to a JSON string. JSONDecode "
            "parses a JSON string into a Lua table. These are useful for DataStore "
            "serialization, logging, and analytics payloads. Always pcall JSONDecode "
            "since invalid JSON throws.\n"
            "</think>\n\n"
            "--!strict\n"
            'local HttpService = game:GetService("HttpService")\n\n'
            "local function encode(data: { [string]: any }): string\n"
            "\treturn HttpService:JSONEncode(data)\n"
            "end\n\n"
            "local function decode(jsonStr: string): { [string]: any }?\n"
            "\tlocal success, result = pcall(function()\n"
            "\t\treturn HttpService:JSONDecode(jsonStr)\n"
            "\tend)\n"
            "\tif success then\n"
            "\t\treturn result\n"
            "\telse\n"
            '\t\twarn("[HTTP] JSONDecode failed:", result)\n'
            "\t\treturn nil\n"
            "\tend\n"
            "end\n\n"
            "-- GenerateGUID for unique identifiers\n"
            "local function newId(): string\n"
            "\treturn HttpService:GenerateGUID(false) -- false = no curly braces\n"
            "end\n",
            difficulty=1,
        ),
    ]


# ---------------------------------------------------------------------------
# MarketplaceService examples
# ---------------------------------------------------------------------------


def _marketplace_examples() -> list[dict]:
    return [
        _ex(
            "How do I implement in-game purchases with MarketplaceService in Vertigo?",
            "<think>\n"
            "MarketplaceService handles Roblox developer products and game passes. "
            "ProcessReceipt is the critical callback — it must return "
            "Enum.ProductPurchaseDecision.PurchaseGranted after granting the item, or "
            "NotProcessedYet if something fails. This callback must be idempotent because "
            "Roblox may call it multiple times for the same purchase. Server-authoritative.\n"
            "</think>\n\n"
            "--!strict\n"
            'local MarketplaceService = game:GetService("MarketplaceService")\n'
            'local Players = game:GetService("Players")\n\n'
            "local COIN_PRODUCT_ID = 123456789 -- Developer product ID\n"
            "local COIN_AMOUNT = 100\n\n"
            "local PurchaseService = {}\n\n"
            "function PurchaseService:Init()\n"
            "\tself._processedReceipts = {} :: { [string]: boolean }\n"
            "end\n\n"
            "function PurchaseService:Start()\n"
            "\tMarketplaceService.ProcessReceipt = function(receiptInfo)\n"
            "\t\treturn self:_handleReceipt(receiptInfo)\n"
            "\tend\n"
            "end\n\n"
            "function PurchaseService:_handleReceipt(\n"
            "\treceiptInfo: { PlayerId: number, ProductId: number, PurchaseId: string }\n"
            "): Enum.ProductPurchaseDecision\n"
            "\t-- Idempotency check\n"
            "\tif self._processedReceipts[receiptInfo.PurchaseId] then\n"
            "\t\treturn Enum.ProductPurchaseDecision.PurchaseGranted\n"
            "\tend\n\n"
            "\tlocal player = Players:GetPlayerByUserId(receiptInfo.PlayerId)\n"
            "\tif not player then\n"
            "\t\treturn Enum.ProductPurchaseDecision.NotProcessedYet\n"
            "\tend\n\n"
            "\tif receiptInfo.ProductId == COIN_PRODUCT_ID then\n"
            "\t\tlocal success = DataService:AddCoins(player, COIN_AMOUNT)\n"
            "\t\tif not success then\n"
            "\t\t\treturn Enum.ProductPurchaseDecision.NotProcessedYet\n"
            "\t\tend\n"
            "\tend\n\n"
            "\tself._processedReceipts[receiptInfo.PurchaseId] = true\n"
            "\treturn Enum.ProductPurchaseDecision.PurchaseGranted\n"
            "end\n\n"
            "-- Prompt a purchase from the client\n"
            "function PurchaseService:PromptCoinPurchase(player: Player)\n"
            "\tMarketplaceService:PromptProductPurchase(player, COIN_PRODUCT_ID)\n"
            "end\n\n"
            "return PurchaseService\n",
            difficulty=4,
            category="service",
        ),
    ]


# ---------------------------------------------------------------------------
# MemoryStoreService examples
# ---------------------------------------------------------------------------


def _memorystore_examples() -> list[dict]:
    return [
        _ex(
            "How do I use MemoryStoreService for cross-server data in Vertigo?",
            "<think>\n"
            "MemoryStoreService provides fast, temporary cross-server data storage. "
            "GetSortedMap creates a sorted map with automatic expiration. GetQueue "
            "creates a FIFO queue. Unlike DataStoreService, data expires and is not "
            "persisted permanently. Good for matchmaking, real-time leaderboards, and "
            "cross-server events. All calls must be pcall-wrapped.\n"
            "</think>\n\n"
            "--!strict\n"
            'local MemoryStoreService = game:GetService("MemoryStoreService")\n\n'
            "local EXPIRATION = 600 -- 10 minutes\n\n"
            "-- Sorted Map for cross-server leaderboard\n"
            'local leaderboardMap = MemoryStoreService:GetSortedMap("ActiveLeaderboard")\n\n'
            "local function setScore(userId: number, score: number): boolean\n"
            "\tlocal success, err = pcall(function()\n"
            "\t\tleaderboardMap:SetAsync(tostring(userId), score, EXPIRATION)\n"
            "\tend)\n"
            "\tif not success then\n"
            '\t\twarn("[MemoryStore] SetAsync failed:", err)\n'
            "\tend\n"
            "\treturn success\n"
            "end\n\n"
            "local function getTopScores(count: number): { { key: string, value: number } }\n"
            "\tlocal success, items = pcall(function()\n"
            "\t\treturn leaderboardMap:GetRangeAsync(SortDirection.Descending, count)\n"
            "\tend)\n"
            "\tif success then\n"
            "\t\treturn items\n"
            "\telse\n"
            '\t\twarn("[MemoryStore] GetRangeAsync failed:", items)\n'
            "\t\treturn {}\n"
            "\tend\n"
            "end\n\n"
            "-- Queue for cross-server matchmaking\n"
            'local matchQueue = MemoryStoreService:GetQueue("MatchmakingQueue")\n\n'
            "local function enqueuePlayer(userId: number): boolean\n"
            "\tlocal success, err = pcall(function()\n"
            "\t\tmatchQueue:AddAsync(tostring(userId), EXPIRATION)\n"
            "\tend)\n"
            "\tif not success then\n"
            '\t\twarn("[MemoryStore] AddAsync failed:", err)\n'
            "\tend\n"
            "\treturn success\n"
            "end\n",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# MessagingService examples
# ---------------------------------------------------------------------------


def _messaging_examples() -> list[dict]:
    return [
        _ex(
            "How do I use MessagingService for cross-server communication?",
            "<think>\n"
            "MessagingService enables pub/sub messaging between servers in the same game. "
            "SubscribeAsync registers a listener for a topic. PublishAsync sends a message "
            "to all subscribers. Messages have a 1KB limit. Use for global announcements, "
            "server shutdown coordination, or cross-server events.\n"
            "</think>\n\n"
            "--!strict\n"
            'local MessagingService = game:GetService("MessagingService")\n\n'
            "local CrossServerService = {}\n\n"
            "function CrossServerService:Init()\n"
            "\tself._subscriptions = {} :: { RBXScriptConnection }\n"
            "end\n\n"
            "function CrossServerService:Start()\n"
            "\tlocal success, err = pcall(function()\n"
            '\t\tlocal sub = MessagingService:SubscribeAsync("GlobalAnnounce",\n'
            "\t\t\tfunction(message: { Data: any, Sent: number })\n"
            "\t\t\t\tself:_onAnnouncement(message.Data)\n"
            "\t\t\tend\n"
            "\t\t)\n"
            "\t\ttable.insert(self._subscriptions, sub)\n"
            "\tend)\n\n"
            "\tif not success then\n"
            '\t\twarn("[CrossServer] SubscribeAsync failed:", err)\n'
            "\tend\n"
            "end\n\n"
            "function CrossServerService:Announce(message: string)\n"
            "\tlocal success, err = pcall(function()\n"
            '\t\tMessagingService:PublishAsync("GlobalAnnounce", message)\n'
            "\tend)\n"
            "\tif not success then\n"
            '\t\twarn("[CrossServer] PublishAsync failed:", err)\n'
            "\tend\n"
            "end\n\n"
            "function CrossServerService:_onAnnouncement(data: any)\n"
            '\tif typeof(data) == "string" then\n'
            '\t\tprint("[CrossServer] Announcement:", data)\n'
            "\tend\n"
            "end\n\n"
            "return CrossServerService\n",
            difficulty=3,
            category="service",
        ),
    ]


# ---------------------------------------------------------------------------
# SoundService / TextService / Lighting examples
# ---------------------------------------------------------------------------


def _misc_service_examples() -> list[dict]:
    return [
        _ex(
            "How do I use SoundService and play sounds in Vertigo?",
            "<think>\n"
            "Sounds in Roblox can be played by parenting a Sound instance to a part "
            "(3D positional audio) or to SoundService (2D ambient). In Vertigo, we "
            "pool Sound objects during Init to avoid Instance.new in hot paths. "
            "SoundService also controls global properties like DistanceFactor.\n"
            "</think>\n\n"
            "--!strict\n"
            'local SoundService = game:GetService("SoundService")\n\n'
            "local AudioService = {}\n\n"
            "function AudioService:Init()\n"
            "\tself._soundPool = {} :: { [string]: Sound }\n"
            "end\n\n"
            "function AudioService:Start()\n"
            "\t-- Configure global audio\n"
            "\tSoundService.DistanceFactor = 10\n"
            "\tSoundService.RespectFilteringEnabled = true\n"
            "end\n\n"
            "function AudioService:PlaySound(soundId: string, parent: Instance?)\n"
            "\tlocal sound = self._soundPool[soundId]\n"
            "\tif not sound then\n"
            '\t\tsound = Instance.new("Sound")\n'
            "\t\tsound.SoundId = soundId\n"
            "\t\tself._soundPool[soundId] = sound\n"
            "\tend\n\n"
            "\tsound.Parent = parent or SoundService\n"
            "\tsound:Play()\n"
            "end\n\n"
            "return AudioService\n",
            difficulty=2,
        ),
        _ex(
            "How do I use TextService to measure text size for UI?",
            "<think>\n"
            "TextService:GetTextSize returns the pixel dimensions a string would occupy "
            "given a font size, font, and frame size. This is useful for dynamic UI "
            "layouts. The newer GetTextBoundsAsync is preferred for rich text.\n"
            "</think>\n\n"
            "--!strict\n"
            'local TextService = game:GetService("TextService")\n\n'
            "local function measureText(\n"
            "\ttext: string,\n"
            "\tfontSize: number,\n"
            "\tfont: Enum.Font,\n"
            "\tmaxWidth: number\n"
            "): Vector2\n"
            "\tlocal frameSize = Vector2.new(maxWidth, math.huge)\n"
            "\treturn TextService:GetTextSize(text, fontSize, font, frameSize)\n"
            "end\n\n"
            "-- Example: size a frame to fit text\n"
            "local function autoSizeLabel(label: TextLabel)\n"
            "\tlocal size = measureText(\n"
            "\t\tlabel.Text,\n"
            "\t\tlabel.TextSize,\n"
            "\t\tlabel.Font,\n"
            "\t\tlabel.AbsoluteSize.X\n"
            "\t)\n"
            "\tlabel.Size = UDim2.new(1, 0, 0, size.Y + 8)\n"
            "end\n",
            difficulty=2,
        ),
        _ex(
            "How do I configure Lighting for different zones in Vertigo?",
            "<think>\n"
            "Lighting controls the global atmosphere. In Vertigo, different zones have "
            "different lighting — the Abyss is dark with fog, the Canopy is bright. "
            "We tween Lighting properties during zone transitions. ColorCorrection, "
            "Bloom, and Atmosphere post-processing effects are children of Lighting.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Lighting = game:GetService("Lighting")\n'
            'local TweenService = game:GetService("TweenService")\n\n'
            "type ZoneLighting = {\n"
            "\tambient: Color3,\n"
            "\tbrightness: number,\n"
            "\tfogEnd: number,\n"
            "\tfogColor: Color3,\n"
            "\tclockTime: number,\n"
            "}\n\n"
            "local ZONE_PRESETS: { [string]: ZoneLighting } = table.freeze({\n"
            "\tabyss = {\n"
            "\t\tambient = Color3.fromRGB(20, 15, 30),\n"
            "\t\tbrightness = 0.5,\n"
            "\t\tfogEnd = 200,\n"
            "\t\tfogColor = Color3.fromRGB(10, 8, 20),\n"
            "\t\tclockTime = 0,\n"
            "\t},\n"
            "\tcanopy = {\n"
            "\t\tambient = Color3.fromRGB(120, 140, 100),\n"
            "\t\tbrightness = 2,\n"
            "\t\tfogEnd = 2000,\n"
            "\t\tfogColor = Color3.fromRGB(180, 200, 160),\n"
            "\t\tclockTime = 14,\n"
            "\t},\n"
            "})\n\n"
            "local function transitionToZone(zoneName: string, duration: number)\n"
            "\tlocal preset = ZONE_PRESETS[zoneName]\n"
            "\tif not preset then return end\n\n"
            "\tlocal info = TweenInfo.new(duration, Enum.EasingStyle.Sine, Enum.EasingDirection.InOut)\n"
            "\tlocal tween = TweenService:Create(Lighting, info, {\n"
            "\t\tAmbient = preset.ambient,\n"
            "\t\tBrightness = preset.brightness,\n"
            "\t\tFogEnd = preset.fogEnd,\n"
            "\t\tFogColor = preset.fogColor,\n"
            "\t\tClockTime = preset.clockTime,\n"
            "\t})\n"
            "\ttween:Play()\n"
            "end\n",
            difficulty=3,
            category="controller",
        ),
    ]


# ---------------------------------------------------------------------------
# PhysicsService / PathfindingService examples
# ---------------------------------------------------------------------------


def _physics_path_examples() -> list[dict]:
    return [
        _ex(
            "How do I use PhysicsService for collision groups in Vertigo?",
            "<think>\n"
            "PhysicsService manages collision groups. In Vertigo, we use collision groups "
            "to prevent players from colliding with certain zone geometry, or to let "
            "projectiles pass through specific objects. RegisterCollisionGroup creates a "
            "group, CollisionGroupSetCollidable controls whether two groups collide.\n"
            "</think>\n\n"
            "--!strict\n"
            'local PhysicsService = game:GetService("PhysicsService")\n\n'
            "local CollisionService = {}\n\n"
            "function CollisionService:Init()\n"
            "\t-- Register collision groups during Init\n"
            '\tPhysicsService:RegisterCollisionGroup("Players")\n'
            '\tPhysicsService:RegisterCollisionGroup("Projectiles")\n'
            '\tPhysicsService:RegisterCollisionGroup("GhostParts")\n\n'
            "\t-- Configure collision rules\n"
            '\tPhysicsService:CollisionGroupSetCollidable("Players", "GhostParts", false)\n'
            '\tPhysicsService:CollisionGroupSetCollidable("Projectiles", "Projectiles", false)\n'
            "end\n\n"
            "function CollisionService:SetCollisionGroup(part: BasePart, group: string)\n"
            "\tpart.CollisionGroup = group\n"
            "end\n\n"
            "return CollisionService\n",
            difficulty=2,
            category="service",
        ),
        _ex(
            "How do I use PathfindingService for NPC navigation in Vertigo?",
            "<think>\n"
            "PathfindingService:CreatePath creates a pathfinding agent. ComputeAsync "
            "calculates a path between two points. GetWaypoints returns the waypoints "
            "to follow. For Vertigo's procedurally generated world, NPCs must recompute "
            "paths when zones change. PathfindingModifiers let us mark surfaces as "
            "preferred or blocked.\n"
            "</think>\n\n"
            "--!strict\n"
            'local PathfindingService = game:GetService("PathfindingService")\n\n'
            "local function navigateTo(humanoid: Humanoid, targetPosition: Vector3): boolean\n"
            "\tlocal path = PathfindingService:CreatePath({\n"
            "\t\tAgentRadius = 2,\n"
            "\t\tAgentHeight = 5,\n"
            "\t\tAgentCanJump = true,\n"
            "\t\tAgentCanClimb = true,\n"
            "\t\tWaypointSpacing = 4,\n"
            "\t})\n\n"
            "\tlocal rootPart = humanoid.RootPart\n"
            "\tif not rootPart then return false end\n\n"
            "\tlocal success, err = pcall(function()\n"
            "\t\tpath:ComputeAsync(rootPart.Position, targetPosition)\n"
            "\tend)\n\n"
            "\tif not success or path.Status ~= Enum.PathStatus.Success then\n"
            '\t\twarn("[Pathfinding] Failed:", err or path.Status)\n'
            "\t\treturn false\n"
            "\tend\n\n"
            "\tlocal waypoints = path:GetWaypoints()\n"
            "\tfor _, waypoint in waypoints do\n"
            "\t\tif waypoint.Action == Enum.PathWaypointAction.Jump then\n"
            "\t\t\thumanoid.Jump = true\n"
            "\t\tend\n"
            "\t\thumanoid:MoveTo(waypoint.Position)\n"
            "\t\thumanoid.MoveToFinished:Wait()\n"
            "\tend\n\n"
            "\treturn true\n"
            "end\n",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Deprecated API migration examples
# ---------------------------------------------------------------------------


def _deprecated_examples() -> list[dict]:
    return [
        _ex(
            "What is the correct replacement for wait() in modern Luau?",
            "<think>\n"
            "wait() is deprecated and has unpredictable minimum yield times (~1/30s). "
            "task.wait() is the modern replacement — it has more precise timing and "
            "better scheduling behavior. In Vertigo, we never use wait(); all code "
            "must use task.wait(), task.spawn(), task.delay(), and task.defer().\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- DEPRECATED: Do not use\n"
            "--[[\n"
            "wait(1)       -- Imprecise, min ~1/30s\n"
            "wait()        -- Yields at least 1/30s\n"
            "]]\n\n"
            "-- CORRECT: Use task.wait()\n"
            "task.wait(1)    -- Waits approximately 1 second\n"
            "task.wait()     -- Yields for one frame (Heartbeat)\n"
            "task.wait(0.1)  -- Waits approximately 0.1 seconds\n\n"
            "-- task.wait returns the actual elapsed time\n"
            "local elapsed = task.wait(0.5)\n"
            'print("Actually waited:", elapsed, "seconds")\n',
            difficulty=1,
            category="debugging",
        ),
        _ex(
            "What is the correct replacement for spawn() in modern Luau?",
            "<think>\n"
            "spawn() is deprecated. It defers execution to a later resumption cycle and "
            "has unpredictable timing. task.spawn() runs the function immediately in a "
            "new coroutine (synchronous until first yield). task.defer() is the closest "
            "behavioral match to old spawn() if you need deferred execution.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- DEPRECATED: Do not use\n"
            "--[[\n"
            "spawn(function()\n"
            '\tprint("This runs later, unpredictably")\n'
            "end)\n"
            "]]\n\n"
            "-- CORRECT: task.spawn runs immediately in a new thread\n"
            "task.spawn(function()\n"
            '\tprint("This runs immediately")\n'
            "\ttask.wait(1)\n"
            '\tprint("This runs after 1 second")\n'
            "end)\n\n"
            "-- task.defer runs after the current resumption cycle\n"
            "task.defer(function()\n"
            '\tprint("This runs next resumption cycle")\n'
            "end)\n\n"
            "-- task.spawn also accepts arguments\n"
            "local function greet(name: string)\n"
            '\tprint("Hello", name)\n'
            "end\n"
            'task.spawn(greet, "Player1")\n',
            difficulty=1,
            category="debugging",
        ),
        _ex(
            "What is the correct replacement for delay() in modern Luau?",
            "<think>\n"
            "delay() is deprecated for the same reasons as wait() — unpredictable timing "
            "and poor scheduling. task.delay() is the modern replacement with precise "
            "timing and proper thread scheduling.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- DEPRECATED: Do not use\n"
            "--[[\n"
            "delay(5, function()\n"
            '\tprint("After 5 seconds (imprecise)")\n'
            "end)\n"
            "]]\n\n"
            "-- CORRECT: Use task.delay()\n"
            "task.delay(5, function()\n"
            '\tprint("After 5 seconds (precise)")\n'
            "end)\n\n"
            "-- task.delay with arguments\n"
            "local function spawnEnemy(position: Vector3, level: number)\n"
            '\tprint("Spawning level", level, "enemy at", position)\n'
            "end\n\n"
            "task.delay(3, spawnEnemy, Vector3.new(0, 10, 0), 5)\n\n"
            "-- task.delay returns a thread that can be cancelled\n"
            "local thread = task.delay(10, function()\n"
            '\tprint("This might get cancelled")\n'
            "end)\n"
            "task.cancel(thread) -- Cancel before it fires\n",
            difficulty=1,
            category="debugging",
        ),
        _ex(
            "Why should I avoid Instance.new with a parent argument?",
            "<think>\n"
            'Instance.new("Part", parent) sets the parent during construction, which '
            "triggers all Changed/ChildAdded events before properties are set. This is "
            "inefficient and can cause bugs. Instead, create the instance, set all "
            "properties, then set Parent last. This fires events only once with the "
            "final state.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- DEPRECATED PATTERN: Parent as second argument\n"
            "--[[\n"
            'local part = Instance.new("Part", workspace)\n'
            "part.Size = Vector3.new(4, 1, 4)\n"
            "part.Color = Color3.fromRGB(255, 0, 0)\n"
            "part.Anchored = true\n"
            "-- Problem: ChildAdded fires BEFORE properties are set\n"
            "-- Listeners see a default-sized, grey, unanchored part\n"
            "]]\n\n"
            "-- CORRECT: Set Parent last\n"
            'local part = Instance.new("Part")\n'
            "part.Size = Vector3.new(4, 1, 4)\n"
            "part.Color = Color3.fromRGB(255, 0, 0)\n"
            "part.Anchored = true\n"
            "part.Parent = workspace\n"
            "-- ChildAdded fires with fully configured part\n",
            difficulty=1,
            category="debugging",
        ),
        _ex(
            "Show me all deprecated Roblox APIs and their modern replacements.",
            "<think>\n"
            "There are several deprecated APIs that Vertigo's codebase avoids. The main "
            "categories are: global functions replaced by the task library, Instance.new "
            "parent argument, old raycasting API, and various property/method renames. "
            "Vertigo's linter (selene + vertigo-sync validate) catches most of these.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- ============================================\n"
            "-- DEPRECATED → MODERN API REFERENCE\n"
            "-- ============================================\n\n"
            "-- wait() → task.wait()\n"
            "-- spawn(fn) → task.spawn(fn)\n"
            "-- delay(t, fn) → task.delay(t, fn)\n"
            "-- Instance.new(class, parent) → Instance.new(class); obj.Parent = parent\n"
            "-- workspace:FindPartOnRay() → workspace:Raycast()\n"
            "-- Ray.new() → use origin + direction with workspace:Raycast()\n\n"
            "-- Correct patterns:\n"
            "task.wait(1)\n"
            "task.spawn(function()\n"
            '\tprint("immediate new thread")\n'
            "end)\n"
            "task.delay(2, function()\n"
            '\tprint("after 2 seconds")\n'
            "end)\n"
            "task.defer(function()\n"
            '\tprint("next resumption cycle")\n'
            "end)\n\n"
            'local part = Instance.new("Part")\n'
            "part.Anchored = true\n"
            "part.Parent = workspace\n\n"
            "local params = RaycastParams.new()\n"
            "params.FilterType = Enum.RaycastFilterType.Exclude\n"
            "local result = workspace:Raycast(origin, direction, params)\n",
            difficulty=1,
            category="debugging",
        ),
        _ex(
            "How do I refactor legacy wait/spawn/delay code to use task library?",
            "<think>\n"
            "When migrating legacy Roblox code to modern Vertigo conventions, the task "
            "library is a direct replacement. The key behavioral difference is that "
            "task.spawn runs immediately (synchronous until first yield), while the old "
            "spawn() deferred. If you need deferred behavior, use task.defer(). For "
            "cancellation, task.delay returns a thread you can task.cancel().\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- BEFORE (legacy code):\n"
            "--[[\n"
            "function OldModule:DoThings()\n"
            "\twait(2)\n"
            "\tspawn(function()\n"
            "\t\twhile wait(0.5) do\n"
            "\t\t\tself:update()\n"
            "\t\tend\n"
            "\tend)\n"
            "\tdelay(10, function()\n"
            "\t\tself:cleanup()\n"
            "\tend)\n"
            "end\n"
            "]]\n\n"
            "-- AFTER (modern Vertigo code):\n"
            "function ModernModule:DoThings()\n"
            "\ttask.wait(2)\n"
            "\ttask.spawn(function()\n"
            "\t\twhile true do\n"
            "\t\t\ttask.wait(0.5)\n"
            "\t\t\tself:update()\n"
            "\t\tend\n"
            "\tend)\n"
            "\ttask.delay(10, function()\n"
            "\t\tself:cleanup()\n"
            "\tend)\n"
            "end\n\n"
            '-- Note: "while wait() do" pattern becomes "while true do task.wait(); ... end"\n'
            "-- The old pattern relied on wait() returning truthy — task.wait() also\n"
            "-- returns elapsed time (truthy), but explicit loops are clearer.\n",
            difficulty=2,
            category="refactoring",
        ),
    ]


# ---------------------------------------------------------------------------
# Performance / best-practice examples
# ---------------------------------------------------------------------------


def _performance_examples() -> list[dict]:
    return [
        _ex(
            "What are the common performance pitfalls with Roblox services?",
            "<think>\n"
            "Key performance rules for Vertigo:\n"
            "1. Never call Instance.new() in Heartbeat — pool objects in Init\n"
            "2. Never create closures in loops — extract to module-level @native functions\n"
            "3. Cache GetService results — don't call game:GetService every frame\n"
            "4. Use CollectionService tags instead of workspace:FindFirstChild chains\n"
            "5. Batch remote calls — don't fire a RemoteEvent per-frame\n"
            "6. Use RaycastParams.new() once and reuse — don't create per raycast\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- BAD: Service call and Instance.new every frame\n"
            "--[[\n"
            "RunService.Heartbeat:Connect(function(dt)\n"
            '\tlocal CS = game:GetService("CollectionService") -- Wasteful per-frame\n'
            '\tfor _, part in CS:GetTagged("Particle") do\n'
            '\t\tlocal p = Instance.new("Part") -- Allocation in hot path!\n'
            "\t\tp.Position = part.Position + Vector3.new(0, dt, 0)\n"
            "\t\tp.Parent = workspace\n"
            "\tend\n"
            "end)\n"
            "]]\n\n"
            "-- GOOD: Cache services, pool objects, extract callback\n"
            'local RunService = game:GetService("RunService")\n'
            'local CollectionService = game:GetService("CollectionService")\n\n'
            "local partPool: { BasePart } = {}\n\n"
            "local function initPool(size: number)\n"
            "\tfor i = 1, size do\n"
            '\t\tlocal part = Instance.new("Part")\n'
            "\t\tpart.Anchored = true\n"
            "\t\ttable.insert(partPool, part)\n"
            "\tend\n"
            "end\n\n"
            "-- Cache the tagged instances; refresh on add/remove signals\n"
            "local cachedParticles: { BasePart } = {}\n\n"
            "--[[@native]]\n"
            "local function updateParticles(dt: number)\n"
            "\tfor i = 1, #cachedParticles do\n"
            "\t\tlocal part = cachedParticles[i]\n"
            "\t\tpart.Position += Vector3.new(0, dt, 0)\n"
            "\tend\n"
            "end\n\n"
            "RunService.Heartbeat:Connect(updateParticles)\n",
            difficulty=4,
            category="physics",
        ),
        _ex(
            "How do I batch remote events to reduce network traffic?",
            "<think>\n"
            "Firing a RemoteEvent every frame creates excessive network traffic. Instead, "
            "accumulate state changes and send them at a fixed rate (e.g., 20Hz). In "
            "Vertigo, StateSync batches multiple state updates into a single remote call "
            "using a table payload. This dramatically reduces bandwidth usage.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n'
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
            "local Remotes = require(Shared.Net.Remotes)\n\n"
            "local SYNC_RATE = 1 / 20 -- 20Hz\n"
            "local accumulated = 0\n"
            "local pendingUpdates: { [string]: any } = {}\n"
            "local hasPending = false\n\n"
            "local function queueUpdate(key: string, value: any)\n"
            "\tpendingUpdates[key] = value\n"
            "\thasPending = true\n"
            "end\n\n"
            "local function flushUpdates(player: Player)\n"
            "\tif not hasPending then return end\n\n"
            '\tlocal remote = Remotes.get("StateSync")\n'
            "\tremote:FireClient(player, pendingUpdates)\n\n"
            "\ttable.clear(pendingUpdates)\n"
            "\thasPending = false\n"
            "end\n\n"
            "--[[@native]]\n"
            "local function onHeartbeat(dt: number)\n"
            "\taccumulated += dt\n"
            "\tif accumulated >= SYNC_RATE then\n"
            "\t\taccumulated -= SYNC_RATE\n"
            "\t\t-- Flush to all players who need updates\n"
            "\tend\n"
            "end\n\n"
            "RunService.Heartbeat:Connect(onHeartbeat)\n",
            difficulty=4,
            category="networking",
        ),
        _ex(
            "How do I properly clean up connections to avoid memory leaks?",
            "<think>\n"
            "Every :Connect() returns an RBXScriptConnection that must be :Disconnect() "
            "when no longer needed. In Vertigo, we use Trove (from Packages) for cleanup, "
            "or manually track connections in an array. Failure to disconnect causes memory "
            "leaks and ghost listeners. PlayerRemoving is a critical cleanup point.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n'
            'local RunService = game:GetService("RunService")\n\n'
            "local TrackingService = {}\n\n"
            "function TrackingService:Init()\n"
            "\tself._playerConnections = {} :: { [Player]: { RBXScriptConnection } }\n"
            "end\n\n"
            "function TrackingService:Start()\n"
            "\tPlayers.PlayerAdded:Connect(function(player: Player)\n"
            "\t\tself:_setupPlayer(player)\n"
            "\tend)\n\n"
            "\tPlayers.PlayerRemoving:Connect(function(player: Player)\n"
            "\t\tself:_cleanupPlayer(player)\n"
            "\tend)\n"
            "end\n\n"
            "function TrackingService:_setupPlayer(player: Player)\n"
            "\tlocal connections: { RBXScriptConnection } = {}\n\n"
            "\ttable.insert(connections, player.CharacterAdded:Connect(function(character: Model)\n"
            '\t\tprint("[Track]", player.Name, "spawned")\n'
            "\tend))\n\n"
            "\tself._playerConnections[player] = connections\n"
            "end\n\n"
            "function TrackingService:_cleanupPlayer(player: Player)\n"
            "\tlocal connections = self._playerConnections[player]\n"
            "\tif connections then\n"
            "\t\tfor _, conn in connections do\n"
            "\t\t\tconn:Disconnect()\n"
            "\t\tend\n"
            "\tend\n"
            "\tself._playerConnections[player] = nil\n"
            "end\n\n"
            "return TrackingService\n",
            difficulty=3,
            category="service",
        ),
        _ex(
            "How do I use table.create and table.freeze for performance in Vertigo?",
            "<think>\n"
            "table.create(n) pre-allocates an array with n slots, avoiding repeated "
            "resizing. table.freeze() makes a table read-only, which enables Luau VM "
            "optimizations and prevents accidental mutation of config/constant tables. "
            "Both are required conventions in Vertigo's codebase.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- table.create: pre-size arrays for known lengths\n"
            "local POOL_SIZE = 256\n"
            "local pool = table.create(POOL_SIZE) :: { BasePart? }\n\n"
            "-- table.freeze: immutable config tables\n"
            "local GrappleTuning = table.freeze({\n"
            "\tmaxRange = 120,\n"
            "\treelSpeed = 80,\n"
            "\tswingDamping = 0.95,\n"
            "\tcooldown = 0.3,\n"
            "\tlaunchForce = 50,\n"
            "})\n\n"
            "-- Nested freeze for deep immutability\n"
            "local AbilityConfig = table.freeze({\n"
            "\tgrapple = table.freeze({\n"
            '\t\tid = "ability_grapple_v1",\n'
            "\t\ttuning = GrappleTuning,\n"
            "\t}),\n"
            "\tglide = table.freeze({\n"
            '\t\tid = "ability_glide_v1",\n'
            "\t\tliftCoefficient = 0.5,\n"
            "\t\tdragCoefficient = 0.1,\n"
            "\t}),\n"
            "})\n\n"
            "-- table.clone: create a mutable copy of a frozen table\n"
            "local mutableCopy = table.clone(GrappleTuning)\n"
            "mutableCopy.maxRange = 150 -- OK: mutableCopy is not frozen\n\n"
            "-- GrappleTuning.maxRange = 150 -- ERROR: table is frozen\n",
            difficulty=2,
        ),
        _ex(
            "How do I use math.lerp and exponential decay for smooth movement?",
            "<think>\n"
            "math.lerp(a, b, t) compiles to a fused multiply-add instruction on NCG — "
            "faster than a + (b - a) * t. For frame-rate independent smoothing, use "
            "exponential decay: math.exp(-speed * dt) gives consistent results regardless "
            "of frame rate. This is used throughout Vertigo for camera, UI, and movement.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- BAD: Frame-rate dependent lerp\n"
            "--[[\n"
            "position = position + (target - position) * 0.1  -- Different speed at 30fps vs 60fps\n"
            "]]\n\n"
            "-- GOOD: Frame-rate independent exponential decay\n"
            "--[[@native]]\n"
            "local function expDecay(current: number, target: number, speed: number, dt: number): number\n"
            "\treturn math.lerp(current, target, 1 - math.exp(-speed * dt))\n"
            "end\n\n"
            "-- Vector3 version\n"
            "--[[@native]]\n"
            "local function expDecayVec3(\n"
            "\tcurrent: Vector3,\n"
            "\ttarget: Vector3,\n"
            "\tspeed: number,\n"
            "\tdt: number\n"
            "): Vector3\n"
            "\tlocal alpha = 1 - math.exp(-speed * dt)\n"
            "\treturn current:Lerp(target, alpha)\n"
            "end\n\n"
            "-- Usage in a Heartbeat callback:\n"
            "-- self._smoothPosition = expDecayVec3(self._smoothPosition, targetPos, 10, dt)\n",
            difficulty=3,
            category="physics",
        ),
        _ex(
            "How do I avoid common mistakes when using game:GetService?",
            "<think>\n"
            "game:GetService is the correct way to access Roblox services. Common mistakes: "
            "calling it every frame (wasteful), using game.ServiceName (can fail if not "
            "loaded), or using the wrong service name string. Cache the result at module "
            "scope. The string must exactly match the service class name.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- BAD: Per-frame service lookup\n"
            "--[[\n"
            "RunService.Heartbeat:Connect(function(dt)\n"
            '\tlocal WS = game:GetService("Workspace") -- Wasteful!\n'
            "\t-- use WS\n"
            "end)\n"
            "]]\n\n"
            "-- BAD: Direct property access (can fail)\n"
            "--[[\n"
            "local Players = game.Players -- May not be loaded yet\n"
            "]]\n\n"
            "-- CORRECT: Cache at module scope\n"
            'local Players = game:GetService("Players")\n'
            'local RunService = game:GetService("RunService")\n'
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local CollectionService = game:GetService("CollectionService")\n'
            'local DataStoreService = game:GetService("DataStoreService")\n'
            'local TweenService = game:GetService("TweenService")\n'
            'local UserInputService = game:GetService("UserInputService")\n'
            'local HttpService = game:GetService("HttpService")\n\n'
            "-- These are resolved once and reused throughout the module.\n",
            difficulty=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Integration / architecture examples
# ---------------------------------------------------------------------------


def _integration_examples() -> list[dict]:
    return [
        _ex(
            "How do I structure a complete Vertigo server service that uses multiple Roblox APIs?",
            "<think>\n"
            "A production Vertigo service combines multiple Roblox APIs within the "
            "Init/Start lifecycle. Init caches references and creates data structures. "
            "Start connects signals and begins operations. This example shows a zone "
            "management service using CollectionService, Players, RunService, and "
            "RemoteEvents together.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n'
            'local Players = game:GetService("Players")\n'
            'local RunService = game:GetService("RunService")\n'
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
            "local Remotes = require(Shared.Net.Remotes)\n\n"
            "local ZoneService = {}\n\n"
            "type ZoneData = {\n"
            "\tname: string,\n"
            "\tbounds: { min: Vector3, max: Vector3 },\n"
            "\tplayers: { Player },\n"
            "}\n\n"
            "function ZoneService:Init()\n"
            "\tself._zones = {} :: { [BasePart]: ZoneData }\n"
            "\tself._playerZone = {} :: { [Player]: BasePart? }\n"
            "\tself._stateSync = nil :: RemoteEvent?\n"
            "end\n\n"
            "function ZoneService:Start()\n"
            '\tself._stateSync = Remotes.get("StateSync")\n\n'
            "\t-- Discover zone triggers via CollectionService\n"
            '\tfor _, instance in CollectionService:GetTagged("ZoneTrigger") do\n'
            "\t\tself:_registerZone(instance :: BasePart)\n"
            "\tend\n\n"
            '\tCollectionService:GetInstanceAddedSignal("ZoneTrigger"):Connect(function(instance: Instance)\n'
            "\t\tself:_registerZone(instance :: BasePart)\n"
            "\tend)\n\n"
            "\t-- Track player positions at 10Hz\n"
            "\tlocal accumulated = 0\n"
            "\tRunService.Heartbeat:Connect(function(dt: number)\n"
            "\t\taccumulated += dt\n"
            "\t\tif accumulated >= 0.1 then\n"
            "\t\t\taccumulated -= 0.1\n"
            "\t\t\tself:_updatePlayerZones()\n"
            "\t\tend\n"
            "\tend)\n\n"
            "\tPlayers.PlayerRemoving:Connect(function(player: Player)\n"
            "\t\tself._playerZone[player] = nil\n"
            "\tend)\n"
            "end\n\n"
            "function ZoneService:_registerZone(part: BasePart)\n"
            '\tlocal name = part:GetAttribute("ZoneName") or part.Name\n'
            "\tlocal size = part.Size / 2\n"
            "\tlocal pos = part.Position\n"
            "\tself._zones[part] = {\n"
            "\t\tname = name :: string,\n"
            "\t\tbounds = {\n"
            "\t\t\tmin = pos - size,\n"
            "\t\t\tmax = pos + size,\n"
            "\t\t},\n"
            "\t\tplayers = {},\n"
            "\t}\n"
            "end\n\n"
            "function ZoneService:_updatePlayerZones()\n"
            "\tfor _, player in Players:GetPlayers() do\n"
            "\t\tlocal character = player.Character\n"
            "\t\tif not character then continue end\n"
            '\t\tlocal rootPart = character:FindFirstChild("HumanoidRootPart") :: BasePart?\n'
            "\t\tif not rootPart then continue end\n\n"
            "\t\tlocal pos = rootPart.Position\n"
            "\t\tlocal currentZone = self._playerZone[player]\n"
            "\t\tlocal newZone: BasePart? = nil\n\n"
            "\t\tfor part, data in self._zones do\n"
            "\t\t\tlocal b = data.bounds\n"
            "\t\t\tif pos.X >= b.min.X and pos.X <= b.max.X\n"
            "\t\t\t\tand pos.Y >= b.min.Y and pos.Y <= b.max.Y\n"
            "\t\t\t\tand pos.Z >= b.min.Z and pos.Z <= b.max.Z then\n"
            "\t\t\t\tnewZone = part\n"
            "\t\t\t\tbreak\n"
            "\t\t\tend\n"
            "\t\tend\n\n"
            "\t\tif newZone ~= currentZone then\n"
            "\t\t\tself._playerZone[player] = newZone\n"
            '\t\t\tlocal zoneName = if newZone then self._zones[newZone].name else "wilderness"\n'
            "\t\t\tlocal remote = self._stateSync\n"
            "\t\t\tif remote then\n"
            '\t\t\t\tremote:FireClient(player, "currentZone", zoneName)\n'
            "\t\t\tend\n"
            "\t\tend\n"
            "\tend\n"
            "end\n\n"
            "return ZoneService\n",
            difficulty=5,
            category="service",
        ),
        _ex(
            "How do I write a Vertigo client controller that uses UserInputService, RunService, and Remotes together?",
            "<think>\n"
            "A complete Vertigo client controller integrates input handling (UserInputService), "
            "per-frame updates (RunService.RenderStepped), remote communication, and proper "
            "cleanup. Init caches references, Start connects signals. This example shows a "
            "grapple controller that reads mouse input, validates locally, sends requests "
            "to the server, and renders the grapple line each frame.\n"
            "</think>\n\n"
            "--!strict\n"
            'local UserInputService = game:GetService("UserInputService")\n'
            'local RunService = game:GetService("RunService")\n'
            'local Players = game:GetService("Players")\n'
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
            "local Remotes = require(Shared.Net.Remotes)\n"
            "local Config = require(Shared.Config.Abilities)\n"
            "local GrappleTuning = Config.GrappleTuning\n\n"
            "local GrappleController = {}\n\n"
            "local localPlayer = Players.LocalPlayer\n\n"
            "function GrappleController:Init()\n"
            "\tself._active = false\n"
            "\tself._targetPosition = nil :: Vector3?\n"
            "\tself._connections = {} :: { RBXScriptConnection }\n"
            "\tself._remote = nil :: RemoteEvent?\n"
            "end\n\n"
            "function GrappleController:Start()\n"
            '\tself._remote = Remotes.get("RequestUseAbility")\n\n'
            "\ttable.insert(self._connections, UserInputService.InputBegan:Connect(\n"
            "\t\tfunction(input: InputObject, gameProcessed: boolean)\n"
            "\t\t\tif gameProcessed then return end\n"
            "\t\t\tif input.UserInputType == Enum.UserInputType.MouseButton1 then\n"
            "\t\t\t\tself:_tryGrapple()\n"
            "\t\t\tend\n"
            "\t\tend\n"
            "\t))\n\n"
            "\ttable.insert(self._connections, UserInputService.InputEnded:Connect(\n"
            "\t\tfunction(input: InputObject, gameProcessed: boolean)\n"
            "\t\t\tif gameProcessed then return end\n"
            "\t\t\tif input.UserInputType == Enum.UserInputType.MouseButton1 then\n"
            "\t\t\t\tself:_releaseGrapple()\n"
            "\t\t\tend\n"
            "\t\tend\n"
            "\t))\n\n"
            "\ttable.insert(self._connections, RunService.RenderStepped:Connect(\n"
            "\t\tfunction(dt: number)\n"
            "\t\t\tif self._active then\n"
            "\t\t\t\tself:_renderGrappleLine(dt)\n"
            "\t\t\tend\n"
            "\t\tend\n"
            "\t))\n"
            "end\n\n"
            "function GrappleController:_tryGrapple()\n"
            "\tlocal character = localPlayer.Character\n"
            "\tif not character then return end\n"
            '\tlocal rootPart = character:FindFirstChild("HumanoidRootPart") :: BasePart?\n'
            "\tif not rootPart then return end\n\n"
            "\t-- Raycast to find grapple target\n"
            "\tlocal mousePos = UserInputService:GetMouseLocation()\n"
            "\tlocal camera = workspace.CurrentCamera\n"
            "\tlocal ray = camera:ViewportPointToRay(mousePos.X, mousePos.Y)\n\n"
            "\tlocal params = RaycastParams.new()\n"
            "\tparams.FilterType = Enum.RaycastFilterType.Exclude\n"
            "\tparams.FilterDescendantsInstances = { character }\n\n"
            "\tlocal result = workspace:Raycast(ray.Origin, ray.Direction * GrappleTuning.maxRange, params)\n"
            "\tif not result then return end\n\n"
            "\tself._targetPosition = result.Position\n"
            "\tself._active = true\n\n"
            "\t-- Request server validation\n"
            "\tlocal remote = self._remote\n"
            "\tif remote then\n"
            '\t\tremote:FireServer("ability_grapple_v1", result.Position)\n'
            "\tend\n"
            "end\n\n"
            "function GrappleController:_releaseGrapple()\n"
            "\tself._active = false\n"
            "\tself._targetPosition = nil\n"
            "end\n\n"
            "function GrappleController:_renderGrappleLine(dt: number)\n"
            "\t-- Render grapple beam visual (implementation depends on VFX system)\n"
            "end\n\n"
            "return GrappleController\n",
            difficulty=5,
            category="controller",
        ),
        _ex(
            "How do I use Workspace.Terrain for terrain manipulation?",
            "<think>\n"
            "Workspace.Terrain provides methods for reading and writing voxel terrain. "
            "FillBlock, FillBall, FillRegion create terrain. ReadVoxels/WriteVoxels give "
            "low-level voxel access. In Vertigo's builders, terrain is procedurally "
            "generated during world construction. These calls should happen during "
            "building, not in Heartbeat.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local terrain = workspace.Terrain\n\n"
            "-- Fill a block of terrain\n"
            "local function createPlatform(center: CFrame, size: Vector3, material: Enum.Material)\n"
            "\tterrain:FillBlock(center, size, material)\n"
            "end\n\n"
            "-- Fill a sphere of terrain (useful for caves/craters)\n"
            "local function createCrater(center: Vector3, radius: number)\n"
            "\tterrain:FillBall(center, radius, Enum.Material.Air)\n"
            "end\n\n"
            "-- Create terrain during world building (not in Heartbeat)\n"
            "local function buildZoneTerrain()\n"
            "\t-- Flat ground plane\n"
            "\tcreatePlatform(\n"
            "\t\tCFrame.new(0, -2, 0),\n"
            "\t\tVector3.new(500, 4, 500),\n"
            "\t\tEnum.Material.Grass\n"
            "\t)\n\n"
            "\t-- Carve out a cave entrance\n"
            "\tcreateCrater(Vector3.new(50, 0, 50), 15)\n"
            "end\n",
            difficulty=2,
            category="builder",
        ),
    ]


# ---------------------------------------------------------------------------
# Additional DataStore / error-handling examples
# ---------------------------------------------------------------------------


def _datastore_advanced_examples() -> list[dict]:
    return [
        _ex(
            "How do I implement session locking with DataStoreService to prevent data duplication?",
            "<think>\n"
            "Session locking prevents two servers from writing to the same player's data "
            "simultaneously (e.g., during teleports). We store a session GUID with the data "
            "and use UpdateAsync to atomically claim/release the lock. If the lock belongs "
            "to another server, we retry or reject.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n'
            'local HttpService = game:GetService("HttpService")\n\n'
            'local playerStore = DataStoreService:GetDataStore("PlayerData_v2")\n'
            "local SESSION_ID = HttpService:GenerateGUID(false)\n\n"
            "type LockedData = {\n"
            "\tsessionId: string?,\n"
            "\tdata: { [string]: any },\n"
            "}\n\n"
            "local function acquireLock(key: string): LockedData?\n"
            "\tlocal acquired = false\n"
            "\tlocal result: LockedData? = nil\n\n"
            "\tlocal success, err = pcall(function()\n"
            "\t\tplayerStore:UpdateAsync(key, function(current: LockedData?)\n"
            "\t\t\tif current and current.sessionId and current.sessionId ~= SESSION_ID then\n"
            "\t\t\t\t-- Another server holds the lock\n"
            "\t\t\t\treturn nil -- Cancel write\n"
            "\t\t\tend\n\n"
            "\t\t\tlocal data = if current then current.data else {}\n"
            "\t\t\tresult = { sessionId = SESSION_ID, data = data }\n"
            "\t\t\tacquired = true\n"
            "\t\t\treturn result\n"
            "\t\tend)\n"
            "\tend)\n\n"
            "\tif success and acquired then\n"
            "\t\treturn result\n"
            "\tend\n"
            "\treturn nil\n"
            "end\n\n"
            "local function releaseLock(key: string, finalData: { [string]: any })\n"
            "\tpcall(function()\n"
            "\t\tplayerStore:UpdateAsync(key, function(current: LockedData?)\n"
            "\t\t\tif current and current.sessionId == SESSION_ID then\n"
            "\t\t\t\treturn { sessionId = nil :: string?, data = finalData }\n"
            "\t\t\tend\n"
            "\t\t\treturn nil\n"
            "\t\tend)\n"
            "\tend)\n"
            "end\n",
            difficulty=5,
            category="service",
        ),
        _ex(
            "How do I handle DataStore budget and throttling?",
            "<think>\n"
            "DataStoreService has per-server request budgets. GetRequestBudgetForRequestType "
            "returns the remaining budget for each operation type. In Vertigo, we check the "
            "budget before making calls and queue requests when the budget is low. This "
            "prevents 429 errors and data loss.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            "local function hasBudget(requestType: Enum.DataStoreRequestType): boolean\n"
            "\tlocal budget = DataStoreService:GetRequestBudgetForRequestType(requestType)\n"
            "\treturn budget > 0\n"
            "end\n\n"
            "local function safeSave(store: DataStore, key: string, data: any): boolean\n"
            "\tif not hasBudget(Enum.DataStoreRequestType.SetIncrementAsync) then\n"
            '\t\twarn("[DataService] No budget for SetAsync, queueing")\n'
            "\t\treturn false\n"
            "\tend\n\n"
            "\tlocal success, err = pcall(function()\n"
            "\t\tstore:SetAsync(key, data)\n"
            "\tend)\n\n"
            "\tif not success then\n"
            '\t\twarn("[DataService] SetAsync failed:", err)\n'
            "\tend\n"
            "\treturn success\n"
            "end\n",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Additional Players examples
# ---------------------------------------------------------------------------


def _players_advanced_examples() -> list[dict]:
    return [
        _ex(
            "How do I teleport players between places in Vertigo?",
            "<think>\n"
            "TeleportService handles cross-place teleportation. TeleportAsync is the modern "
            "API (replaces Teleport). It returns a TeleportAsyncResult. Always pcall because "
            "teleportation can fail due to moderation, network issues, or invalid placeIds. "
            "Save player data BEFORE teleporting.\n"
            "</think>\n\n"
            "--!strict\n"
            'local TeleportService = game:GetService("TeleportService")\n'
            'local Players = game:GetService("Players")\n\n'
            "local DESTINATION_PLACE_ID = 987654321\n\n"
            "local function teleportPlayer(player: Player): boolean\n"
            "\t-- Save data first!\n"
            "\t-- DataService:SaveProfile(player)\n\n"
            "\tlocal success, result = pcall(function()\n"
            "\t\treturn TeleportService:TeleportAsync(DESTINATION_PLACE_ID, { player })\n"
            "\tend)\n\n"
            "\tif not success then\n"
            '\t\twarn("[Teleport] Failed for", player.Name, ":", result)\n'
            "\tend\n"
            "\treturn success\n"
            "end\n\n"
            "-- Teleport with data passing\n"
            "local function teleportWithData(player: Player, data: { [string]: any }): boolean\n"
            '\tlocal options = Instance.new("TeleportOptions")\n'
            "\toptions:SetTeleportData(data)\n\n"
            "\tlocal success, result = pcall(function()\n"
            "\t\treturn TeleportService:TeleportAsync(DESTINATION_PLACE_ID, { player }, options)\n"
            "\tend)\n\n"
            "\tif not success then\n"
            '\t\twarn("[Teleport] Failed:", result)\n'
            "\tend\n"
            "\treturn success\n"
            "end\n",
            difficulty=3,
        ),
        _ex(
            "How do I handle player respawning and LoadCharacter?",
            "<think>\n"
            "Player:LoadCharacter() forces a respawn. In Vertigo, we control spawn points "
            "and respawn timing on the server. CharacterAutoLoads can be disabled in "
            "StarterPlayer for custom spawn control. Always wait for the character to "
            "actually load before interacting with it.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "local SpawnService = {}\n\n"
            "function SpawnService:Init()\n"
            "\tself._spawnPoints = {} :: { [string]: CFrame }\n"
            "end\n\n"
            "function SpawnService:Start()\n"
            "\tPlayers.PlayerAdded:Connect(function(player: Player)\n"
            "\t\t-- Custom spawn logic\n"
            "\t\ttask.wait(1) -- Brief delay for data loading\n"
            "\t\tplayer:LoadCharacter()\n\n"
            "\t\tplayer.CharacterAdded:Connect(function(character: Model)\n"
            '\t\t\tlocal rootPart = character:WaitForChild("HumanoidRootPart") :: BasePart\n'
            "\t\t\tlocal spawnCF = self:_getSpawnPoint(player)\n"
            "\t\t\trootPart.CFrame = spawnCF\n"
            "\t\tend)\n"
            "\tend)\n"
            "end\n\n"
            "function SpawnService:RespawnPlayer(player: Player)\n"
            "\ttask.spawn(function()\n"
            "\t\ttask.wait(3) -- Respawn delay\n"
            "\t\tplayer:LoadCharacter()\n"
            "\tend)\n"
            "end\n\n"
            "function SpawnService:_getSpawnPoint(player: Player): CFrame\n"
            "\t-- Return zone-appropriate spawn point\n"
            "\treturn CFrame.new(0, 50, 0)\n"
            "end\n\n"
            "return SpawnService\n",
            difficulty=3,
            category="service",
        ),
    ]


# ---------------------------------------------------------------------------
# Additional RunService examples
# ---------------------------------------------------------------------------


def _runservice_advanced_examples() -> list[dict]:
    return [
        _ex(
            "How do I implement a fixed timestep loop with RunService?",
            "<think>\n"
            "Fixed timestep loops ensure deterministic simulation regardless of frame rate. "
            "Accumulate dt and consume in fixed increments. This is critical for Vertigo's "
            "physics — vehicle simulation and ability mechanics must behave the same at "
            "30fps and 144fps. Leftover time is interpolated for rendering.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n\n'
            "local FIXED_DT = 1 / 60 -- 60Hz fixed step\n"
            "local MAX_STEPS = 4 -- Cap to prevent spiral of death\n"
            "local accumulator = 0\n\n"
            "--[[@native]]\n"
            "local function fixedStep(dt: number)\n"
            "\t-- Deterministic simulation step\n"
            "end\n\n"
            "RunService.Heartbeat:Connect(function(dt: number)\n"
            "\taccumulator += dt\n"
            "\tlocal steps = 0\n\n"
            "\twhile accumulator >= FIXED_DT and steps < MAX_STEPS do\n"
            "\t\tfixedStep(FIXED_DT)\n"
            "\t\taccumulator -= FIXED_DT\n"
            "\t\tsteps += 1\n"
            "\tend\n\n"
            "\t-- Consume remaining accumulator to prevent spiral of death\n"
            "\tif steps >= MAX_STEPS then\n"
            "\t\taccumulator = 0\n"
            "\tend\n\n"
            "\t-- Alpha for render interpolation\n"
            "\tlocal alpha = accumulator / FIXED_DT\n"
            "\t-- Use alpha to interpolate visual positions\n"
            "end)\n",
            difficulty=4,
            category="physics",
        ),
        _ex(
            "How do I use RunService:BindToRenderStep for ordered client updates?",
            "<think>\n"
            "BindToRenderStep lets you set an explicit priority for render-step callbacks. "
            "Lower priority numbers run first. Roblox defines standard priorities: "
            "Input (100), Camera (200), Character (300). In Vertigo, we bind camera "
            "updates just after the default camera (201) and VFX after that (250).\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n\n'
            "-- Priority constants for Vertigo's render pipeline\n"
            "local PRIORITY_CAMERA = Enum.RenderPriority.Camera.Value + 1 -- 201\n"
            "local PRIORITY_VFX = Enum.RenderPriority.Camera.Value + 50 -- 250\n"
            "local PRIORITY_UI = Enum.RenderPriority.Last.Value -- 2000\n\n"
            "local CameraController = {}\n\n"
            "function CameraController:Start()\n"
            '\tRunService:BindToRenderStep("VertigoCamera", PRIORITY_CAMERA,\n'
            "\t\tfunction(dt: number)\n"
            "\t\t\tself:_updateCamera(dt)\n"
            "\t\tend\n"
            "\t)\n\n"
            '\tRunService:BindToRenderStep("VertigoVFX", PRIORITY_VFX,\n'
            "\t\tfunction(dt: number)\n"
            "\t\t\tself:_updateVFX(dt)\n"
            "\t\tend\n"
            "\t)\n"
            "end\n\n"
            "-- Cleanup: always unbind when done\n"
            "function CameraController:Destroy()\n"
            '\tRunService:UnbindFromRenderStep("VertigoCamera")\n'
            '\tRunService:UnbindFromRenderStep("VertigoVFX")\n'
            "end\n\n"
            "return CameraController\n",
            difficulty=3,
            category="controller",
        ),
    ]


# ---------------------------------------------------------------------------
# Additional CollectionService examples
# ---------------------------------------------------------------------------


def _collection_advanced_examples() -> list[dict]:
    return [
        _ex(
            "How do I use CollectionService with zone streaming in Vertigo?",
            "<think>\n"
            "When Vertigo streams zones in/out, tagged instances appear and disappear. "
            "GetInstanceAddedSignal and GetInstanceRemovedSignal handle this dynamically. "
            "Services must not assume all tagged objects exist at Start time — they must "
            "handle late arrivals from zone loading.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n\n'
            "type ComponentHandler = {\n"
            "\tonAdded: (instance: Instance) -> (),\n"
            "\tonRemoved: (instance: Instance) -> (),\n"
            "}\n\n"
            "local function bindToTag(tag: string, handler: ComponentHandler)\n"
            "\t-- Handle existing instances\n"
            "\tfor _, instance in CollectionService:GetTagged(tag) do\n"
            "\t\ttask.spawn(handler.onAdded, instance)\n"
            "\tend\n\n"
            "\t-- Handle future instances\n"
            "\tCollectionService:GetInstanceAddedSignal(tag):Connect(handler.onAdded)\n"
            "\tCollectionService:GetInstanceRemovedSignal(tag):Connect(handler.onRemoved)\n"
            "end\n\n"
            "-- Usage:\n"
            'bindToTag("BloomCrystal", {\n'
            "\tonAdded = function(instance: Instance)\n"
            "\t\tlocal part = instance :: BasePart\n"
            "\t\t-- Add glow effect, register in spatial index\n"
            '\t\tprint("[Crystal] Registered at", part.Position)\n'
            "\tend,\n"
            "\tonRemoved = function(instance: Instance)\n"
            "\t\t-- Remove glow, unregister from spatial index\n"
            '\t\tprint("[Crystal] Unregistered")\n'
            "\tend,\n"
            "})\n",
            difficulty=3,
            category="service",
        ),
    ]


# ---------------------------------------------------------------------------
# Additional Remote examples
# ---------------------------------------------------------------------------


def _remote_advanced_examples() -> list[dict]:
    return [
        _ex(
            "How do I implement rate limiting for RemoteEvents on the server?",
            "<think>\n"
            "Rate limiting prevents clients from spamming the server with remote calls. "
            "Track timestamps per player and reject requests that exceed the rate limit. "
            "In Vertigo, we enforce 10 messages/second per player. This is critical for "
            "server stability and anti-exploit.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "local MAX_REQUESTS_PER_SEC = 10\n"
            "local WINDOW_SIZE = 1.0\n\n"
            "local RateLimiter = {}\n\n"
            "function RateLimiter.new(): RateLimiter\n"
            "\tlocal self = {\n"
            "\t\t_timestamps = {} :: { [Player]: { number } },\n"
            "\t}\n\n"
            "\tPlayers.PlayerRemoving:Connect(function(player: Player)\n"
            "\t\tself._timestamps[player] = nil\n"
            "\tend)\n\n"
            "\treturn self :: any\n"
            "end\n\n"
            "function RateLimiter:Check(player: Player): boolean\n"
            "\tlocal now = os.clock()\n"
            "\tlocal timestamps = self._timestamps[player]\n\n"
            "\tif not timestamps then\n"
            "\t\tself._timestamps[player] = { now }\n"
            "\t\treturn true\n"
            "\tend\n\n"
            "\t-- Prune old timestamps outside the window\n"
            "\tlocal cutoff = now - WINDOW_SIZE\n"
            "\tlocal pruned = table.create(MAX_REQUESTS_PER_SEC)\n"
            "\tfor _, ts in timestamps do\n"
            "\t\tif ts > cutoff then\n"
            "\t\t\ttable.insert(pruned, ts)\n"
            "\t\tend\n"
            "\tend\n\n"
            "\tif #pruned >= MAX_REQUESTS_PER_SEC then\n"
            "\t\tself._timestamps[player] = pruned\n"
            "\t\treturn false -- Rate limited\n"
            "\tend\n\n"
            "\ttable.insert(pruned, now)\n"
            "\tself._timestamps[player] = pruned\n"
            "\treturn true\n"
            "end\n\n"
            "return RateLimiter\n",
            difficulty=4,
            category="networking",
        ),
        _ex(
            "How do I type-validate remote arguments from untrusted clients?",
            "<think>\n"
            "Client remote arguments are untrusted. A malicious client can send any type. "
            "Always validate typeof() for each argument before using it. In Vertigo, we "
            "use a validation helper that checks types and ranges. Never cast without "
            "checking — typeof(arg) ~= 'string' means reject, not coerce.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- Validation helpers for remote arguments\n"
            "local function isString(value: unknown): boolean\n"
            '\treturn typeof(value) == "string"\n'
            "end\n\n"
            "local function isNumber(value: unknown): boolean\n"
            '\treturn typeof(value) == "number"\n'
            "end\n\n"
            "local function isVector3(value: unknown): boolean\n"
            '\treturn typeof(value) == "Vector3"\n'
            "end\n\n"
            "local function isInRange(value: number, min: number, max: number): boolean\n"
            "\treturn value >= min and value <= max and value == value -- NaN check\n"
            "end\n\n"
            "-- Usage in a remote handler:\n"
            "local function handlePurchaseRequest(\n"
            "\tplayer: Player,\n"
            "\titemId: unknown,\n"
            "\tquantity: unknown\n"
            ")\n"
            "\tif not isString(itemId) then return end\n"
            "\tif not isNumber(quantity) then return end\n\n"
            "\tlocal id = itemId :: string\n"
            "\tlocal qty = quantity :: number\n\n"
            "\tif not isInRange(qty, 1, 99) then return end\n"
            "\tif math.floor(qty) ~= qty then return end -- Must be integer\n\n"
            "\t-- Safe to use id and qty\n"
            '\tprint("[Purchase]", player.Name, "wants", qty, "x", id)\n'
            "end\n",
            difficulty=3,
            category="service",
        ),
        _ex(
            "How do I implement a request-response pattern with RemoteEvents instead of RemoteFunctions?",
            "<think>\n"
            "RemoteFunctions are dangerous because a malicious client can hang OnClientInvoke "
            "forever, blocking the server thread. A safer pattern uses two RemoteEvents: one "
            "for the request and one for the response, with a correlation ID and timeout. "
            "This is the pattern Vertigo uses for non-critical queries.\n"
            "</think>\n\n"
            "--!strict\n"
            'local HttpService = game:GetService("HttpService")\n'
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
            "local Remotes = require(Shared.Net.Remotes)\n\n"
            "-- CLIENT SIDE: Request with timeout\n"
            "local pendingRequests: { [string]: thread } = {}\n\n"
            "local function requestFromServer(requestType: string, data: any): any?\n"
            "\tlocal requestId = HttpService:GenerateGUID(false)\n"
            '\tlocal requestRemote = Remotes.get("RequestData")\n'
            '\tlocal responseRemote = Remotes.get("ResponseData")\n\n'
            "\t-- Store current thread for resumption\n"
            "\tlocal currentThread = coroutine.running()\n"
            "\tpendingRequests[requestId] = currentThread\n\n"
            "\trequestRemote:FireServer(requestId, requestType, data)\n\n"
            "\t-- Timeout after 5 seconds\n"
            "\tlocal timeoutThread = task.delay(5, function()\n"
            "\t\tlocal t = pendingRequests[requestId]\n"
            "\t\tif t then\n"
            "\t\t\tpendingRequests[requestId] = nil\n"
            "\t\t\ttask.spawn(t, nil) -- Resume with nil\n"
            "\t\tend\n"
            "\tend)\n\n"
            "\tlocal result = coroutine.yield()\n"
            "\ttask.cancel(timeoutThread)\n"
            "\treturn result\n"
            "end\n",
            difficulty=4,
            category="networking",
        ),
    ]


# ---------------------------------------------------------------------------
# Additional error handling / pcall examples
# ---------------------------------------------------------------------------


def _error_handling_examples() -> list[dict]:
    return [
        _ex(
            "How do I properly use pcall for error handling in Roblox Luau?",
            "<think>\n"
            "pcall (protected call) catches errors without crashing the script. It returns "
            "(success: boolean, result: any). If success is true, result is the return "
            "value. If false, result is the error message. In Vertigo, ALL external API "
            "calls (DataStore, HTTP, Teleport) must be pcall-wrapped.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- Basic pcall pattern\n"
            "local success, result = pcall(function()\n"
            "\treturn riskyOperation()\n"
            "end)\n\n"
            "if success then\n"
            '\tprint("Result:", result)\n'
            "else\n"
            '\twarn("Error:", result)\n'
            "end\n\n"
            "-- pcall with arguments (no closure needed)\n"
            "local success2, data = pcall(store.GetAsync, store, key)\n\n"
            "-- xpcall: pcall with error handler for stack traces\n"
            "local success3, result3 = xpcall(function()\n"
            "\treturn riskyOperation()\n"
            "end, function(err)\n"
            '\twarn("Error with traceback:", debug.traceback(tostring(err), 2))\n'
            "\treturn err\n"
            "end)\n\n"
            "-- Nested pcall for retry patterns\n"
            "local function withRetry(fn: () -> any, maxAttempts: number): (boolean, any)\n"
            "\tfor attempt = 1, maxAttempts do\n"
            "\t\tlocal ok, res = pcall(fn)\n"
            "\t\tif ok then return true, res end\n"
            "\t\tif attempt < maxAttempts then\n"
            "\t\t\ttask.wait(2 ^ (attempt - 1))\n"
            "\t\tend\n"
            "\tend\n"
            '\treturn false, "Max retries exceeded"\n'
            "end\n",
            difficulty=2,
        ),
        _ex(
            "How do I handle errors in async chains with Promises in Vertigo?",
            "<think>\n"
            "Vertigo uses the Promise library (from Packages) for async operations. Promises "
            "chain with :andThen(), handle errors with :catch(), and clean up with :finally(). "
            "This is cleaner than nested pcalls for multi-step async operations like "
            "load data → validate → apply.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local Promise = require(ReplicatedStorage:WaitForChild("Packages").Promise)\n\n'
            "local function loadAndApplyProfile(player: Player)\n"
            "\tPromise.new(function(resolve, reject)\n"
            "\t\tlocal success, data = pcall(function()\n"
            '\t\t\treturn store:GetAsync("player_" .. player.UserId)\n'
            "\t\tend)\n"
            "\t\tif success then\n"
            "\t\t\tresolve(data)\n"
            "\t\telse\n"
            "\t\t\treject(data)\n"
            "\t\tend\n"
            "\tend)\n"
            "\t:andThen(function(data)\n"
            "\t\t-- Validate and migrate schema\n"
            "\t\tif data and data.schemaVersion then\n"
            "\t\t\treturn migrateSchema(data)\n"
            "\t\tend\n"
            "\t\treturn getDefaultProfile()\n"
            "\tend)\n"
            "\t:andThen(function(profile)\n"
            "\t\t-- Apply to player\n"
            "\t\tapplyProfile(player, profile)\n"
            "\tend)\n"
            "\t:catch(function(err)\n"
            '\t\twarn("[DataService] Failed to load profile:", err)\n'
            "\tend)\n"
            "end\n",
            difficulty=3,
            category="service",
        ),
    ]


# ---------------------------------------------------------------------------
# Additional workspace/physics examples
# ---------------------------------------------------------------------------


def _workspace_advanced_examples() -> list[dict]:
    return [
        _ex(
            "How do I use spatial queries with OverlapParams in Vertigo?",
            "<think>\n"
            "workspace:GetPartBoundsInBox, GetPartBoundsInRadius, and GetPartsInPart "
            "are modern spatial queries. They use OverlapParams for filtering. These are "
            "useful for AoE detection, proximity checks, and explosion damage. More "
            "efficient than iterating all parts manually.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local function getPartsInRadius(\n"
            "\tcenter: Vector3,\n"
            "\tradius: number,\n"
            "\tignore: { Instance }?\n"
            "): { BasePart }\n"
            "\tlocal params = OverlapParams.new()\n"
            "\tparams.FilterType = Enum.RaycastFilterType.Exclude\n"
            "\tparams.FilterDescendantsInstances = ignore or {}\n\n"
            "\treturn workspace:GetPartBoundsInRadius(center, radius, params)\n"
            "end\n\n"
            "-- AoE damage example\n"
            "local function dealAoEDamage(center: Vector3, radius: number, damage: number)\n"
            "\tlocal parts = getPartsInRadius(center, radius)\n"
            "\tlocal hitPlayers: { [Player]: boolean } = {}\n\n"
            "\tfor _, part in parts do\n"
            "\t\tlocal character = part.Parent\n"
            "\t\tif not character then continue end\n"
            '\t\tlocal humanoid = character:FindFirstChild("Humanoid") :: Humanoid?\n'
            "\t\tif not humanoid then continue end\n"
            '\t\tlocal player = game:GetService("Players"):GetPlayerFromCharacter(character)\n'
            "\t\tif player and not hitPlayers[player] then\n"
            "\t\t\thitPlayers[player] = true\n"
            "\t\t\thumanoid:TakeDamage(damage)\n"
            "\t\tend\n"
            "\tend\n"
            "end\n",
            difficulty=3,
        ),
        _ex(
            "How do I implement a KD-tree or spatial hash for efficient proximity queries?",
            "<think>\n"
            "For large numbers of spatial objects (e.g., hundreds of grapple anchors), "
            "iterating all of them each frame is too slow. Vertigo uses a spatial index. "
            "A simple grid-based spatial hash is practical in Luau — divide space into "
            "cells and only check objects in nearby cells. For @native compatibility, "
            "keep the data structure simple with arrays and integer keys.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local CELL_SIZE = 50\n\n"
            "type SpatialHash = {\n"
            "\tcells: { [string]: { Vector3 } },\n"
            "}\n\n"
            "local function cellKey(x: number, z: number): string\n"
            "\tlocal cx = math.floor(x / CELL_SIZE)\n"
            "\tlocal cz = math.floor(z / CELL_SIZE)\n"
            '\treturn cx .. "_" .. cz\n'
            "end\n\n"
            "local function createSpatialHash(): SpatialHash\n"
            "\treturn { cells = {} }\n"
            "end\n\n"
            "local function insert(hash: SpatialHash, position: Vector3)\n"
            "\tlocal key = cellKey(position.X, position.Z)\n"
            "\tlocal cell = hash.cells[key]\n"
            "\tif not cell then\n"
            "\t\tcell = {}\n"
            "\t\thash.cells[key] = cell\n"
            "\tend\n"
            "\ttable.insert(cell, position)\n"
            "end\n\n"
            "--[[@native]]\n"
            "local function findNearest(hash: SpatialHash, pos: Vector3, maxDist: number): Vector3?\n"
            "\tlocal bestDist = maxDist * maxDist\n"
            "\tlocal bestPos: Vector3? = nil\n"
            "\tlocal radius = math.ceil(maxDist / CELL_SIZE)\n"
            "\tlocal cx = math.floor(pos.X / CELL_SIZE)\n"
            "\tlocal cz = math.floor(pos.Z / CELL_SIZE)\n\n"
            "\tfor dx = -radius, radius do\n"
            "\t\tfor dz = -radius, radius do\n"
            '\t\t\tlocal key = (cx + dx) .. "_" .. (cz + dz)\n'
            "\t\t\tlocal cell = hash.cells[key]\n"
            "\t\t\tif cell then\n"
            "\t\t\t\tfor i = 1, #cell do\n"
            "\t\t\t\t\tlocal delta = cell[i] - pos\n"
            "\t\t\t\t\tlocal distSq = delta:Dot(delta)\n"
            "\t\t\t\t\tif distSq < bestDist then\n"
            "\t\t\t\t\t\tbestDist = distSq\n"
            "\t\t\t\t\t\tbestPos = cell[i]\n"
            "\t\t\t\t\tend\n"
            "\t\t\t\tend\n"
            "\t\t\tend\n"
            "\t\tend\n"
            "\tend\n\n"
            "\treturn bestPos\n"
            "end\n",
            difficulty=5,
            category="physics",
        ),
    ]


# ---------------------------------------------------------------------------
# Additional TweenService examples
# ---------------------------------------------------------------------------


def _tween_advanced_examples() -> list[dict]:
    return [
        _ex(
            "How do I use TweenService for UI animations in Vertigo?",
            "<think>\n"
            "TweenService works on any Instance with numeric/Color3/UDim2 properties. "
            "For UI, we tween Position, Size, Transparency, and BackgroundColor3. "
            "Use EasingStyle.Back for bouncy pop-ins and EasingStyle.Sine for smooth "
            "transitions. In Vertigo, UI animations are controlled from client controllers.\n"
            "</think>\n\n"
            "--!strict\n"
            'local TweenService = game:GetService("TweenService")\n\n'
            "local UIAnimations = {}\n\n"
            "function UIAnimations.slideIn(frame: Frame, duration: number?)\n"
            "\tlocal d = duration or 0.3\n"
            "\tframe.Position = UDim2.new(1, 0, 0.5, 0) -- Start off-screen right\n"
            "\tframe.Visible = true\n\n"
            "\tlocal info = TweenInfo.new(d, Enum.EasingStyle.Back, Enum.EasingDirection.Out)\n"
            "\tlocal tween = TweenService:Create(frame, info, {\n"
            "\t\tPosition = UDim2.new(0.5, 0, 0.5, 0),\n"
            "\t})\n"
            "\ttween:Play()\n"
            "\treturn tween\n"
            "end\n\n"
            "function UIAnimations.fadeOut(frame: Frame, duration: number?)\n"
            "\tlocal d = duration or 0.2\n"
            "\tlocal info = TweenInfo.new(d, Enum.EasingStyle.Quad, Enum.EasingDirection.In)\n"
            "\tlocal tween = TweenService:Create(frame, info, {\n"
            "\t\tBackgroundTransparency = 1,\n"
            "\t})\n"
            "\ttween:Play()\n"
            "\ttween.Completed:Connect(function()\n"
            "\t\tframe.Visible = false\n"
            "\tend)\n"
            "\treturn tween\n"
            "end\n\n"
            "function UIAnimations.popIn(frame: Frame)\n"
            "\tframe.Size = UDim2.new(0, 0, 0, 0)\n"
            "\tframe.Visible = true\n"
            "\tlocal info = TweenInfo.new(0.4, Enum.EasingStyle.Back, Enum.EasingDirection.Out)\n"
            "\tlocal tween = TweenService:Create(frame, info, {\n"
            "\t\tSize = UDim2.new(0.5, 0, 0.5, 0),\n"
            "\t})\n"
            "\ttween:Play()\n"
            "\treturn tween\n"
            "end\n\n"
            "return UIAnimations\n",
            difficulty=2,
            category="controller",
        ),
    ]


# ---------------------------------------------------------------------------
# Additional HttpService examples
# ---------------------------------------------------------------------------


def _http_advanced_examples() -> list[dict]:
    return [
        _ex(
            "How do I make HTTP requests with HttpService in Vertigo?",
            "<think>\n"
            "HttpService:RequestAsync makes HTTP calls to external APIs. It requires "
            "HttpEnabled in Game Settings. Only works on the server. In Vertigo, this "
            "is used for analytics, webhook notifications, and external API integrations. "
            "Always pcall and respect rate limits. Never expose API keys to clients.\n"
            "</think>\n\n"
            "--!strict\n"
            'local HttpService = game:GetService("HttpService")\n\n'
            "type HttpResponse = {\n"
            "\tStatusCode: number,\n"
            "\tBody: string,\n"
            "\tHeaders: { [string]: string },\n"
            "\tSuccess: boolean,\n"
            "}\n\n"
            "local function postAnalytics(eventName: string, data: { [string]: any }): boolean\n"
            "\tlocal payload = HttpService:JSONEncode({\n"
            "\t\tevent = eventName,\n"
            "\t\tdata = data,\n"
            "\t\ttimestamp = os.time(),\n"
            "\t})\n\n"
            "\tlocal success, response = pcall(function(): HttpResponse\n"
            "\t\treturn HttpService:RequestAsync({\n"
            '\t\t\tUrl = "https://analytics.example.com/events",\n'
            '\t\t\tMethod = "POST",\n'
            "\t\t\tHeaders = {\n"
            '\t\t\t\t["Content-Type"] = "application/json",\n'
            "\t\t\t},\n"
            "\t\t\tBody = payload,\n"
            "\t\t})\n"
            "\tend)\n\n"
            "\tif not success then\n"
            '\t\twarn("[Analytics] Request failed:", response)\n'
            "\t\treturn false\n"
            "\tend\n\n"
            "\tlocal resp = response :: HttpResponse\n"
            "\tif not resp.Success then\n"
            '\t\twarn("[Analytics] HTTP", resp.StatusCode, resp.Body)\n'
            "\t\treturn false\n"
            "\tend\n\n"
            "\treturn true\n"
            "end\n",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Additional deprecated / migration examples
# ---------------------------------------------------------------------------


def _deprecated_advanced_examples() -> list[dict]:
    return [
        _ex(
            "How do I replace the deprecated 'while wait() do' loop pattern?",
            "<think>\n"
            "The 'while wait() do' idiom is extremely common in legacy Roblox code. "
            "It works because wait() returns elapsed time (truthy), but it's deprecated "
            "and imprecise. Replace with 'while true do task.wait(interval) end'. "
            "This is clearer and uses the modern task scheduler.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- DEPRECATED: while wait() do\n"
            "--[[\n"
            "while wait(1) do\n"
            '\tprint("Every ~1 second (imprecise)")\n'
            "end\n"
            "]]\n\n"
            "-- CORRECT: while true do task.wait()\n"
            "while true do\n"
            "\ttask.wait(1)\n"
            '\tprint("Every ~1 second (precise)")\n'
            "end\n\n"
            "-- For breakable loops, use a flag:\n"
            "local running = true\n"
            "task.spawn(function()\n"
            "\twhile running do\n"
            "\t\ttask.wait(0.5)\n"
            "\t\t-- periodic work\n"
            "\tend\n"
            "end)\n\n"
            "-- To stop:\n"
            "-- running = false\n",
            difficulty=1,
            category="refactoring",
        ),
        _ex(
            "How do I migrate code using game.Workspace to use game:GetService?",
            "<think>\n"
            "game.Workspace, game.Players, etc. use property-style access which can "
            "fail if the service isn't loaded yet. game:GetService always works and is "
            "the canonical pattern. Additionally, 'workspace' (lowercase global) is "
            "acceptable for Workspace specifically, as it's a Roblox global.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- LEGACY (avoid):\n"
            "--[[\n"
            "local Players = game.Players\n"
            "local RS = game.ReplicatedStorage\n"
            "local SS = game.ServerScriptService\n"
            "]]\n\n"
            "-- CORRECT:\n"
            'local Players = game:GetService("Players")\n'
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local ServerScriptService = game:GetService("ServerScriptService")\n\n'
            "-- 'workspace' global is acceptable (it's a Roblox built-in)\n"
            "local part = workspace.SomePart\n"
            '-- But game:GetService("Workspace") also works\n'
            'local ws = game:GetService("Workspace")\n',
            difficulty=1,
            category="refactoring",
        ),
    ]


# ---------------------------------------------------------------------------
# Additional performance examples
# ---------------------------------------------------------------------------


def _perf_advanced_examples() -> list[dict]:
    return [
        _ex(
            "How do I implement object pooling to avoid Instance.new in hot paths?",
            "<think>\n"
            "Creating instances is expensive. In Vertigo, we pre-create (pool) parts "
            "during Init and reuse them. When an object is 'destroyed', it's returned "
            "to the pool (moved to nil parent, reset properties). When needed, we take "
            "from the pool instead of calling Instance.new. This is mandatory for any "
            "system that creates/destroys objects frequently.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "type ObjectPool<T> = {\n"
            "\t_available: { T },\n"
            "\t_factory: () -> T,\n"
            "\t_reset: (obj: T) -> (),\n"
            "}\n\n"
            "local function createPool<T>(factory: () -> T, reset: (obj: T) -> (), initialSize: number): ObjectPool<T>\n"
            "\tlocal pool: ObjectPool<T> = {\n"
            "\t\t_available = table.create(initialSize),\n"
            "\t\t_factory = factory,\n"
            "\t\t_reset = reset,\n"
            "\t}\n\n"
            "\tfor _ = 1, initialSize do\n"
            "\t\ttable.insert(pool._available, factory())\n"
            "\tend\n\n"
            "\treturn pool\n"
            "end\n\n"
            "local function acquire<T>(pool: ObjectPool<T>): T\n"
            "\tlocal n = #pool._available\n"
            "\tif n > 0 then\n"
            "\t\tlocal obj = pool._available[n]\n"
            "\t\tpool._available[n] = nil\n"
            "\t\treturn obj\n"
            "\tend\n"
            "\treturn pool._factory()\n"
            "end\n\n"
            "local function release<T>(pool: ObjectPool<T>, obj: T)\n"
            "\tpool._reset(obj)\n"
            "\ttable.insert(pool._available, obj)\n"
            "end\n\n"
            "-- Usage: Part pool\n"
            "local partPool = createPool(\n"
            "\tfunction(): BasePart\n"
            '\t\tlocal part = Instance.new("Part")\n'
            "\t\tpart.Anchored = true\n"
            "\t\treturn part\n"
            "\tend,\n"
            "\tfunction(part: BasePart)\n"
            "\t\tpart.Parent = nil\n"
            "\t\tpart.CFrame = CFrame.identity\n"
            "\t\tpart.Transparency = 0\n"
            "\tend,\n"
            "\t64 -- Pre-allocate 64 parts\n"
            ")\n",
            difficulty=4,
            category="physics",
        ),
        _ex(
            "How do I avoid string concatenation performance issues in hot paths?",
            "<think>\n"
            "String concatenation with .. creates a new string each time. In hot paths, "
            "this causes excessive garbage collection. Use string.format or table.concat "
            "for building strings. In Vertigo's @native hot paths, avoid string operations "
            "entirely — use numeric IDs or enums instead.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- BAD: String concatenation in a loop\n"
            "--[[\n"
            'local log = ""\n'
            "for i = 1, 100 do\n"
            '\tlog = log .. "Item " .. tostring(i) .. "\\n" -- O(n^2) allocations!\n'
            "end\n"
            "]]\n\n"
            "-- GOOD: Table-based string building\n"
            "local parts = table.create(100)\n"
            "for i = 1, 100 do\n"
            '\tparts[i] = string.format("Item %d", i)\n'
            "end\n"
            'local log = table.concat(parts, "\\n")\n\n'
            "-- BEST (hot path): Avoid strings entirely, use numeric keys\n"
            "local ITEM_TYPE_COIN = 1\n"
            "local ITEM_TYPE_GEM = 2\n"
            "local counts = table.create(10, 0)\n"
            "counts[ITEM_TYPE_COIN] = 50\n"
            "counts[ITEM_TYPE_GEM] = 3\n",
            difficulty=3,
        ),
        _ex(
            "How do I profile performance bottlenecks in Vertigo?",
            "<think>\n"
            "debug.profilebegin/profileend mark regions in the MicroProfiler. os.clock() "
            "gives high-resolution timestamps for manual timing. In Vertigo, we profile "
            "Init/Start phases, Heartbeat callbacks, and DataStore operations. The Roblox "
            "MicroProfiler (Ctrl+F6) visualizes these regions.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- MicroProfiler regions (visible in Ctrl+F6)\n"
            "local function heavyComputation()\n"
            '\tdebug.profilebegin("ZoneService_UpdatePlayerZones")\n'
            "\t-- ... expensive work ...\n"
            "\tdebug.profileend()\n"
            "end\n\n"
            "-- Manual timing with os.clock()\n"
            "local function timedOperation(label: string, fn: () -> ())\n"
            "\tlocal start = os.clock()\n"
            "\tfn()\n"
            "\tlocal elapsed = os.clock() - start\n"
            "\tif elapsed > 0.001 then -- Log if >1ms\n"
            '\t\tprint(string.format("[Perf] %s: %.3fms", label, elapsed * 1000))\n'
            "\tend\n"
            "end\n\n"
            "-- Usage in Heartbeat\n"
            'local RunService = game:GetService("RunService")\n'
            "RunService.Heartbeat:Connect(function(dt: number)\n"
            '\tdebug.profilebegin("Vertigo_Heartbeat")\n'
            "\t-- game logic\n"
            "\tdebug.profileend()\n"
            "end)\n",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# Additional integration examples
# ---------------------------------------------------------------------------


def _integration_advanced_examples() -> list[dict]:
    return [
        _ex(
            "How do I implement a data migration system for DataStore schema changes?",
            "<think>\n"
            "When the data schema changes (e.g., adding new fields, renaming keys), "
            "existing player data needs migration. Vertigo stores a schemaVersion in "
            "each profile and applies sequential migrations. This runs on load, before "
            "the profile is used. Each migration is a pure function: old data in, new "
            "data out.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local CURRENT_VERSION = 3\n\n"
            "type MigrationFn = (data: { [string]: any }) -> { [string]: any }\n\n"
            "local migrations: { [number]: MigrationFn } = {\n"
            "\t-- v1 → v2: rename 'gold' to 'coins', add unlockedAbilities\n"
            "\t[1] = function(data)\n"
            "\t\tdata.coins = data.gold or 0\n"
            "\t\tdata.gold = nil\n"
            "\t\tdata.unlockedAbilities = data.unlockedAbilities or {}\n"
            "\t\tdata.schemaVersion = 2\n"
            "\t\treturn data\n"
            "\tend,\n\n"
            "\t-- v2 → v3: add actorMemory\n"
            "\t[2] = function(data)\n"
            "\t\tdata.actorMemory = data.actorMemory or {}\n"
            "\t\tdata.schemaVersion = 3\n"
            "\t\treturn data\n"
            "\tend,\n"
            "}\n\n"
            "local function migrateProfile(data: { [string]: any }): { [string]: any }\n"
            "\tlocal version = data.schemaVersion or 1\n\n"
            "\twhile version < CURRENT_VERSION do\n"
            "\t\tlocal migrateFn = migrations[version]\n"
            "\t\tif not migrateFn then\n"
            '\t\t\twarn("[DataService] No migration for version", version)\n'
            "\t\t\tbreak\n"
            "\t\tend\n\n"
            "\t\tdata = migrateFn(data)\n"
            "\t\tversion = data.schemaVersion or (version + 1)\n"
            '\t\tprint("[DataService] Migrated to schema v" .. tostring(version))\n'
            "\tend\n\n"
            "\treturn data\n"
            "end\n",
            difficulty=4,
            category="service",
        ),
        _ex(
            "How do I implement a Vertigo builder that uses CollectionService tags and WeldConstraints?",
            "<think>\n"
            "Vertigo builders procedurally create world geometry. They use deterministic "
            "Random.new(seed) for reproducibility, tag parts via CollectionService for "
            "runtime discovery, and weld parts together with WeldConstraints. Parts are "
            "created with Instance.new during the build phase (not Heartbeat), then frozen.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n\n'
            "local CrystalBuilder = {}\n\n"
            "type BuildOptions = {\n"
            "\tseed: number,\n"
            "\tcount: number,\n"
            "\tregion: { min: Vector3, max: Vector3 },\n"
            "}\n\n"
            "function CrystalBuilder:Build(options: BuildOptions): Model\n"
            "\tlocal rng = Random.new(options.seed)\n"
            '\tlocal model = Instance.new("Model")\n'
            '\tmodel.Name = "CrystalCluster_" .. tostring(options.seed)\n\n'
            "\tfor i = 1, options.count do\n"
            "\t\tlocal crystal = self:_createCrystal(rng)\n"
            "\t\tlocal pos = Vector3.new(\n"
            "\t\t\trng:NextNumber(options.region.min.X, options.region.max.X),\n"
            "\t\t\trng:NextNumber(options.region.min.Y, options.region.max.Y),\n"
            "\t\t\trng:NextNumber(options.region.min.Z, options.region.max.Z)\n"
            "\t\t)\n"
            "\t\tcrystal.CFrame = CFrame.new(pos) * CFrame.Angles(\n"
            "\t\t\trng:NextNumber(-0.3, 0.3),\n"
            "\t\t\trng:NextNumber(0, math.pi * 2),\n"
            "\t\t\trng:NextNumber(-0.3, 0.3)\n"
            "\t\t)\n"
            "\t\tcrystal.Parent = model\n\n"
            "\t\t-- Tag for runtime discovery\n"
            '\t\tCollectionService:AddTag(crystal, "BloomCrystal")\n'
            "\tend\n\n"
            "\t-- Weld all parts to the first for physics stability\n"
            '\tlocal primary = model:FindFirstChildWhichIsA("BasePart")\n'
            "\tif primary then\n"
            "\t\tmodel.PrimaryPart = primary :: BasePart\n"
            "\t\tfor _, child in model:GetChildren() do\n"
            '\t\t\tif child:IsA("BasePart") and child ~= primary then\n'
            '\t\t\t\tlocal weld = Instance.new("WeldConstraint")\n'
            "\t\t\t\tweld.Part0 = primary :: BasePart\n"
            "\t\t\t\tweld.Part1 = child :: BasePart\n"
            "\t\t\t\tweld.Parent = child\n"
            "\t\t\tend\n"
            "\t\tend\n"
            "\tend\n\n"
            "\treturn model\n"
            "end\n\n"
            "function CrystalBuilder:_createCrystal(rng: Random): BasePart\n"
            '\tlocal part = Instance.new("Part")\n'
            "\tpart.Shape = Enum.PartType.Block\n"
            "\tpart.Size = Vector3.new(\n"
            "\t\trng:NextNumber(1, 3),\n"
            "\t\trng:NextNumber(3, 8),\n"
            "\t\trng:NextNumber(1, 3)\n"
            "\t)\n"
            "\tpart.Material = Enum.Material.Neon\n"
            "\tpart.Color = Color3.fromHSV(rng:NextNumber(0.5, 0.7), 0.8, 1)\n"
            "\tpart.Anchored = true\n"
            "\tpart.CanCollide = false\n"
            "\treturn part\n"
            "end\n\n"
            "return CrystalBuilder\n",
            difficulty=4,
            category="builder",
        ),
        _ex(
            "How do I implement Attribute-based configuration on instances?",
            "<think>\n"
            "Instance:SetAttribute/GetAttribute stores key-value metadata on any instance. "
            "This is used in Vertigo builders to configure tagged objects — e.g., a "
            "GrappleAnchor part can have a 'maxSwingAngle' attribute. Attributes replicate "
            "automatically and are visible in Studio properties. Types are limited to "
            "primitives, Vector3, Color3, etc.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n\n'
            "-- Set attributes during building\n"
            "local function configureAnchor(part: BasePart, config: {\n"
            "\tmaxSwingAngle: number,\n"
            "\treelSpeedMultiplier: number,\n"
            "\tanchorType: string,\n"
            "})\n"
            '\tpart:SetAttribute("MaxSwingAngle", config.maxSwingAngle)\n'
            '\tpart:SetAttribute("ReelSpeedMultiplier", config.reelSpeedMultiplier)\n'
            '\tpart:SetAttribute("AnchorType", config.anchorType)\n'
            '\tCollectionService:AddTag(part, "GrappleAnchor")\n'
            "end\n\n"
            "-- Read attributes at runtime\n"
            "local function getAnchorConfig(part: BasePart): {\n"
            "\tmaxSwingAngle: number,\n"
            "\treelSpeedMultiplier: number,\n"
            "\tanchorType: string,\n"
            "}\n"
            "\treturn {\n"
            '\t\tmaxSwingAngle = part:GetAttribute("MaxSwingAngle") or 45,\n'
            '\t\treelSpeedMultiplier = part:GetAttribute("ReelSpeedMultiplier") or 1,\n'
            '\t\tanchorType = part:GetAttribute("AnchorType") or "standard",\n'
            "\t}\n"
            "end\n\n"
            "-- Listen for attribute changes\n"
            "local function watchAttribute(part: BasePart, name: string, callback: (value: any) -> ())\n"
            "\tpart:GetAttributeChangedSignal(name):Connect(function()\n"
            "\t\tcallback(part:GetAttribute(name))\n"
            "\tend)\n"
            "end\n",
            difficulty=2,
        ),
        _ex(
            "How do I implement auto-save with periodic DataStore writes and BindToClose?",
            "<think>\n"
            "game:BindToClose runs before the server shuts down, giving ~30 seconds to "
            "save all player data. Combined with periodic auto-saves (e.g., every 5 minutes), "
            "this minimizes data loss. In Vertigo, DataService handles both patterns. "
            "BindToClose must save synchronously — use task.wait in a loop, not Promises.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n'
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            "local AUTO_SAVE_INTERVAL = 300 -- 5 minutes\n\n"
            "local DataService = {}\n\n"
            "function DataService:Init()\n"
            '\tself._store = DataStoreService:GetDataStore("PlayerData_v2")\n'
            "\tself._profiles = {} :: { [Player]: { [string]: any } }\n"
            "end\n\n"
            "function DataService:Start()\n"
            "\t-- Auto-save loop\n"
            "\ttask.spawn(function()\n"
            "\t\twhile true do\n"
            "\t\t\ttask.wait(AUTO_SAVE_INTERVAL)\n"
            "\t\t\tself:_saveAllProfiles()\n"
            "\t\tend\n"
            "\tend)\n\n"
            "\t-- Save on player leave\n"
            "\tPlayers.PlayerRemoving:Connect(function(player: Player)\n"
            "\t\tself:_saveProfile(player)\n"
            "\t\tself._profiles[player] = nil\n"
            "\tend)\n\n"
            "\t-- Save all on server shutdown\n"
            "\tgame:BindToClose(function()\n"
            "\t\tself:_saveAllProfiles()\n"
            "\tend)\n"
            "end\n\n"
            "function DataService:_saveAllProfiles()\n"
            "\tlocal threads: { thread } = {}\n"
            "\tfor player, profile in self._profiles do\n"
            "\t\tlocal t = task.spawn(function()\n"
            "\t\t\tself:_saveProfile(player)\n"
            "\t\tend)\n"
            "\t\ttable.insert(threads, t)\n"
            "\tend\n"
            "\t-- Wait briefly for saves to complete\n"
            "\ttask.wait(3)\n"
            "end\n\n"
            "function DataService:_saveProfile(player: Player)\n"
            "\tlocal profile = self._profiles[player]\n"
            "\tif not profile then return end\n\n"
            '\tlocal key = "player_" .. tostring(player.UserId)\n'
            "\tlocal success, err = pcall(function()\n"
            "\t\tself._store:SetAsync(key, profile)\n"
            "\tend)\n"
            "\tif not success then\n"
            '\t\twarn("[DataService] Save failed for", player.Name, ":", err)\n'
            "\tend\n"
            "end\n\n"
            "return DataService\n",
            difficulty=4,
            category="service",
        ),
        _ex(
            "How do I use Attributes and CollectionService for a config-driven interactive object system?",
            "<think>\n"
            "Vertigo uses a data-driven approach: builders tag objects and set attributes, "
            "then services discover and operate on them generically. This avoids hardcoded "
            "references and supports zone streaming. The pattern is: Builder creates + tags "
            "+ configures → Service discovers + binds behavior.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n'
            'local TweenService = game:GetService("TweenService")\n\n'
            "-- Generic interactive object handler\n"
            "local InteractiveService = {}\n\n"
            "type InteractiveObject = {\n"
            "\tpart: BasePart,\n"
            "\tinteractionType: string,\n"
            "\tcooldown: number,\n"
            "\tlastUsed: number,\n"
            "\tconnection: RBXScriptConnection?,\n"
            "}\n\n"
            "function InteractiveService:Init()\n"
            "\tself._objects = {} :: { [BasePart]: InteractiveObject }\n"
            "end\n\n"
            "function InteractiveService:Start()\n"
            "\tlocal function onAdded(instance: Instance)\n"
            "\t\tlocal part = instance :: BasePart\n"
            "\t\tlocal obj: InteractiveObject = {\n"
            "\t\t\tpart = part,\n"
            '\t\t\tinteractionType = part:GetAttribute("InteractionType") or "touch",\n'
            '\t\t\tcooldown = part:GetAttribute("Cooldown") or 1,\n'
            "\t\t\tlastUsed = 0,\n"
            "\t\t\tconnection = nil,\n"
            "\t\t}\n\n"
            "\t\tobj.connection = part.Touched:Connect(function(hit: BasePart)\n"
            "\t\t\tlocal now = os.clock()\n"
            "\t\t\tif now - obj.lastUsed < obj.cooldown then return end\n"
            "\t\t\tobj.lastUsed = now\n"
            "\t\t\tself:_activate(obj, hit)\n"
            "\t\tend)\n\n"
            "\t\tself._objects[part] = obj\n"
            "\tend\n\n"
            "\tlocal function onRemoved(instance: Instance)\n"
            "\t\tlocal part = instance :: BasePart\n"
            "\t\tlocal obj = self._objects[part]\n"
            "\t\tif obj and obj.connection then\n"
            "\t\t\tobj.connection:Disconnect()\n"
            "\t\tend\n"
            "\t\tself._objects[part] = nil\n"
            "\tend\n\n"
            '\tfor _, instance in CollectionService:GetTagged("Interactive") do\n'
            "\t\tonAdded(instance)\n"
            "\tend\n"
            '\tCollectionService:GetInstanceAddedSignal("Interactive"):Connect(onAdded)\n'
            '\tCollectionService:GetInstanceRemovedSignal("Interactive"):Connect(onRemoved)\n'
            "end\n\n"
            "function InteractiveService:_activate(obj: InteractiveObject, hit: BasePart)\n"
            "\tlocal itype = obj.interactionType\n"
            '\tif itype == "bounce" then\n'
            '\t\tlocal bv = Instance.new("BodyVelocity")\n'
            "\t\tbv.Velocity = Vector3.new(0, 80, 0)\n"
            "\t\tbv.MaxForce = Vector3.new(0, math.huge, 0)\n"
            "\t\tbv.Parent = hit\n"
            "\t\ttask.delay(0.1, function() bv:Destroy() end)\n"
            '\telseif itype == "collect" then\n'
            "\t\tlocal info = TweenInfo.new(0.3, Enum.EasingStyle.Quad)\n"
            "\t\tlocal tween = TweenService:Create(obj.part, info, { Transparency = 1 })\n"
            "\t\ttween:Play()\n"
            "\t\ttween.Completed:Wait()\n"
            "\t\tobj.part:Destroy()\n"
            "\tend\n"
            "end\n\n"
            "return InteractiveService\n",
            difficulty=5,
            category="service",
        ),
    ]


# ---------------------------------------------------------------------------
# Additional service pattern examples
# ---------------------------------------------------------------------------


def _additional_service_examples() -> list[dict]:
    return [
        _ex(
            "How do I use GuiService and StarterGui for client notifications?",
            "<think>\n"
            "StarterGui:SetCore can control built-in Roblox UI elements like notifications, "
            "chat, and the topbar. SetCoreGuiEnabled toggles entire GUI layers. In Vertigo, "
            "we disable default UI elements and replace them with custom versions.\n"
            "</think>\n\n"
            "--!strict\n"
            'local StarterGui = game:GetService("StarterGui")\n\n'
            "local UIController = {}\n\n"
            "function UIController:Start()\n"
            "\t-- Disable default Roblox UI elements\n"
            "\tStarterGui:SetCoreGuiEnabled(Enum.CoreGuiType.Backpack, false)\n"
            "\tStarterGui:SetCoreGuiEnabled(Enum.CoreGuiType.Health, false)\n"
            "\tStarterGui:SetCoreGuiEnabled(Enum.CoreGuiType.PlayerList, false)\n\n"
            "\t-- Send a notification\n"
            "\tlocal success, err = pcall(function()\n"
            '\t\tStarterGui:SetCore("SendNotification", {\n'
            '\t\t\tTitle = "Vertigo",\n'
            '\t\t\tText = "Welcome to the world!",\n'
            "\t\t\tDuration = 5,\n"
            "\t\t})\n"
            "\tend)\n"
            "\tif not success then\n"
            '\t\twarn("[UI] Notification failed:", err)\n'
            "\tend\n"
            "end\n\n"
            "return UIController\n",
            difficulty=2,
            category="controller",
        ),
        _ex(
            "How do I use ContextActionService for mobile-compatible input in Vertigo?",
            "<think>\n"
            "ContextActionService binds actions to keys/buttons/touch and automatically "
            "creates mobile buttons. Unlike UserInputService, it handles action priority "
            "and context switching. In Vertigo, we use it for ability bindings that need "
            "mobile support. BindAction creates the button; UnbindAction removes it.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ContextActionService = game:GetService("ContextActionService")\n\n'
            "local AbilityController = {}\n\n"
            "function AbilityController:Init()\n"
            "\tself._boundActions = {} :: { string }\n"
            "end\n\n"
            "function AbilityController:Start()\n"
            "\t-- Bind grapple to mouse button 1 + touch\n"
            "\tContextActionService:BindAction(\n"
            '\t\t"Grapple",\n'
            "\t\tfunction(_actionName: string, state: Enum.UserInputState, _input: InputObject)\n"
            "\t\t\tif state == Enum.UserInputState.Begin then\n"
            "\t\t\t\tself:_onGrapple()\n"
            "\t\t\telseif state == Enum.UserInputState.End then\n"
            "\t\t\t\tself:_onGrappleRelease()\n"
            "\t\t\tend\n"
            "\t\tend,\n"
            "\t\tfalse, -- Don't sink input\n"
            "\t\tEnum.UserInputType.MouseButton1,\n"
            "\t\tEnum.KeyCode.ButtonR2 -- Gamepad\n"
            "\t)\n"
            '\ttable.insert(self._boundActions, "Grapple")\n\n'
            "\t-- Bind glide to Q key + touch button\n"
            "\tContextActionService:BindAction(\n"
            '\t\t"Glide",\n'
            "\t\tfunction(_actionName: string, state: Enum.UserInputState)\n"
            "\t\t\tif state == Enum.UserInputState.Begin then\n"
            "\t\t\t\tself:_onGlideToggle()\n"
            "\t\t\tend\n"
            "\t\tend,\n"
            "\t\ttrue, -- Create touch button\n"
            "\t\tEnum.KeyCode.Q,\n"
            "\t\tEnum.KeyCode.ButtonL1\n"
            "\t)\n"
            '\ttable.insert(self._boundActions, "Glide")\n'
            "end\n\n"
            "function AbilityController:Destroy()\n"
            "\tfor _, action in self._boundActions do\n"
            "\t\tContextActionService:UnbindAction(action)\n"
            "\tend\n"
            "\ttable.clear(self._boundActions)\n"
            "end\n\n"
            "return AbilityController\n",
            difficulty=3,
            category="controller",
        ),
        _ex(
            "How do I use BadgeService for achievements in Vertigo?",
            "<think>\n"
            "BadgeService awards badges (achievements) to players. AwardBadge is server-only "
            "and must be pcall-wrapped. UserHasBadgeAsync checks if a player already has a "
            "badge (to avoid re-awarding). Badge IDs come from the Roblox Creator Dashboard.\n"
            "</think>\n\n"
            "--!strict\n"
            'local BadgeService = game:GetService("BadgeService")\n\n'
            "local BADGES = table.freeze({\n"
            "\tfirstGrapple = 123456,\n"
            "\treachCanopy = 789012,\n"
            "\texploreAbyss = 345678,\n"
            "})\n\n"
            "local function awardBadge(player: Player, badgeId: number)\n"
            "\t-- Check if already owned\n"
            "\tlocal hasSuccess, hasBadge = pcall(function()\n"
            "\t\treturn BadgeService:UserHasBadgeAsync(player.UserId, badgeId)\n"
            "\tend)\n\n"
            "\tif not hasSuccess then\n"
            '\t\twarn("[Badges] Check failed:", hasBadge)\n'
            "\t\treturn\n"
            "\tend\n\n"
            "\tif hasBadge then return end\n\n"
            "\tlocal awardSuccess, err = pcall(function()\n"
            "\t\tBadgeService:AwardBadge(player.UserId, badgeId)\n"
            "\tend)\n\n"
            "\tif awardSuccess then\n"
            '\t\tprint("[Badges] Awarded badge", badgeId, "to", player.Name)\n'
            "\telse\n"
            '\t\twarn("[Badges] Award failed:", err)\n'
            "\tend\n"
            "end\n",
            difficulty=2,
            category="service",
        ),
        _ex(
            "How do I use GroupService to check player group membership?",
            "<think>\n"
            "GroupService and Player:IsInGroup/GetRankInGroup check group membership. "
            "In Vertigo, this can gate VIP zones, moderation tools, or group-exclusive "
            "cosmetics. IsInGroup/GetRankInGroup are cached per session but the initial "
            "call can fail, so pcall is needed.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local GROUP_ID = 12345678\n"
            "local ADMIN_RANK = 254 -- Group admin rank\n\n"
            "local function isGroupMember(player: Player): boolean\n"
            "\tlocal success, result = pcall(function()\n"
            "\t\treturn player:IsInGroup(GROUP_ID)\n"
            "\tend)\n"
            "\tif success then\n"
            "\t\treturn result\n"
            "\tend\n"
            '\twarn("[Group] Check failed:", result)\n'
            "\treturn false\n"
            "end\n\n"
            "local function isAdmin(player: Player): boolean\n"
            "\tlocal success, rank = pcall(function()\n"
            "\t\treturn player:GetRankInGroup(GROUP_ID)\n"
            "\tend)\n"
            "\tif success then\n"
            "\t\treturn rank >= ADMIN_RANK\n"
            "\tend\n"
            "\treturn false\n"
            "end\n",
            difficulty=2,
        ),
        _ex(
            "How do I use ProximityPrompt for interaction in Vertigo?",
            "<think>\n"
            "ProximityPrompt is an Instance that shows an interaction prompt when a player "
            "is near. It handles distance, hold duration, and key display automatically. "
            "In Vertigo, builders attach ProximityPrompts to interactive objects. The "
            "Triggered signal fires on the server when a player completes the interaction.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n\n'
            "local function setupPrompt(part: BasePart, config: {\n"
            "\tactionText: string,\n"
            "\tholdDuration: number?,\n"
            "\tmaxDistance: number?,\n"
            "}): ProximityPrompt\n"
            '\tlocal prompt = Instance.new("ProximityPrompt")\n'
            "\tprompt.ActionText = config.actionText\n"
            "\tprompt.HoldDuration = config.holdDuration or 0\n"
            "\tprompt.MaxActivationDistance = config.maxDistance or 10\n"
            "\tprompt.RequiresLineOfSight = true\n"
            "\tprompt.Parent = part\n"
            "\treturn prompt\n"
            "end\n\n"
            "-- Server: handle prompt triggers\n"
            "local InteractionService = {}\n\n"
            "function InteractionService:Start()\n"
            '\tfor _, instance in CollectionService:GetTagged("Interactable") do\n'
            "\t\tself:_bindPrompt(instance :: BasePart)\n"
            "\tend\n\n"
            '\tCollectionService:GetInstanceAddedSignal("Interactable"):Connect(function(instance)\n'
            "\t\tself:_bindPrompt(instance :: BasePart)\n"
            "\tend)\n"
            "end\n\n"
            "function InteractionService:_bindPrompt(part: BasePart)\n"
            '\tlocal prompt = part:FindFirstChildWhichIsA("ProximityPrompt")\n'
            "\tif not prompt then\n"
            "\t\tprompt = setupPrompt(part, {\n"
            '\t\t\tactionText = part:GetAttribute("ActionText") or "Interact",\n'
            '\t\t\tholdDuration = part:GetAttribute("HoldDuration"),\n'
            "\t\t})\n"
            "\tend\n\n"
            "\tprompt.Triggered:Connect(function(player: Player)\n"
            '\t\tprint("[Interaction]", player.Name, "interacted with", part.Name)\n'
            "\tend)\n"
            "end\n\n"
            "return InteractionService\n",
            difficulty=3,
            category="service",
        ),
        _ex(
            "How do I use Debris service for timed cleanup?",
            "<think>\n"
            "Debris:AddItem schedules an Instance for automatic destruction after a delay. "
            "This is simpler than task.delay + :Destroy() for temporary objects like "
            "projectiles, VFX parts, or decals. In Vertigo, we use it for short-lived "
            "visual effects spawned during abilities.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Debris = game:GetService("Debris")\n\n'
            "local function createImpactVFX(position: Vector3)\n"
            '\tlocal part = Instance.new("Part")\n'
            "\tpart.Size = Vector3.new(2, 2, 2)\n"
            "\tpart.Position = position\n"
            "\tpart.Anchored = true\n"
            "\tpart.CanCollide = false\n"
            "\tpart.Material = Enum.Material.Neon\n"
            "\tpart.Color = Color3.fromRGB(100, 200, 255)\n"
            "\tpart.Shape = Enum.PartType.Ball\n"
            "\tpart.Parent = workspace\n\n"
            "\t-- Auto-destroy after 2 seconds\n"
            "\tDebris:AddItem(part, 2)\n"
            "end\n",
            difficulty=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Type annotation examples
# ---------------------------------------------------------------------------


def _type_annotation_examples() -> list[dict]:
    return [
        _ex(
            "How do I write proper Luau type annotations in --!strict mode?",
            "<think>\n"
            "In --!strict mode, all function parameters and return types must be annotated. "
            "Luau supports primitive types (string, number, boolean), Roblox types (Vector3, "
            "CFrame, Player), union types (string | number), optional types (string?), "
            "and custom export types. Full annotations enable NCG optimization.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- Primitive types\n"
            "local function add(a: number, b: number): number\n"
            "\treturn a + b\n"
            "end\n\n"
            "-- Optional types (? suffix)\n"
            "local function greet(name: string?): string\n"
            '\treturn "Hello, " .. (name or "stranger")\n'
            "end\n\n"
            "-- Union types\n"
            "local function stringify(value: string | number | boolean): string\n"
            "\treturn tostring(value)\n"
            "end\n\n"
            "-- Table types\n"
            "local function sum(numbers: { number }): number\n"
            "\tlocal total = 0\n"
            "\tfor _, n in numbers do\n"
            "\t\ttotal += n\n"
            "\tend\n"
            "\treturn total\n"
            "end\n\n"
            "-- Dictionary types\n"
            "local function getConfig(key: string, defaults: { [string]: any }): any\n"
            "\treturn defaults[key]\n"
            "end\n\n"
            "-- Function types\n"
            "local function map(arr: { number }, fn: (n: number) -> number): { number }\n"
            "\tlocal result = table.create(#arr)\n"
            "\tfor i, v in arr do\n"
            "\t\tresult[i] = fn(v)\n"
            "\tend\n"
            "\treturn result\n"
            "end\n\n"
            "-- Multiple return types\n"
            "local function divide(a: number, b: number): (boolean, number?)\n"
            "\tif b == 0 then\n"
            "\t\treturn false, nil\n"
            "\tend\n"
            "\treturn true, a / b\n"
            "end\n",
            difficulty=2,
        ),
        _ex(
            "How do I define and use export types in Vertigo?",
            "<think>\n"
            "Vertigo defines shared types in src/Shared/Util/Types.luau using 'export type'. "
            "The module returns nil — it exists purely for type definitions. Other modules "
            "require it and use the types for annotations. This centralizes type definitions "
            "and prevents duplication.\n"
            "</think>\n\n"
            "--!strict\n"
            "-- Types.luau (returns nil, types-only module)\n\n"
            "export type AbilityId = string\n\n"
            "export type PlayerState = {\n"
            "\thealth: number,\n"
            "\tposition: Vector3,\n"
            "\tvelocity: Vector3,\n"
            "\tactiveAbility: AbilityId?,\n"
            "\tgrounded: boolean,\n"
            "}\n\n"
            "export type AbilityConfig = {\n"
            "\tid: AbilityId,\n"
            "\tname: string,\n"
            "\tcooldown: number,\n"
            "\tenergyCost: number,\n"
            "\tonActivate: (player: Player, state: PlayerState) -> boolean,\n"
            "}\n\n"
            "export type ZoneInfo = {\n"
            "\tname: string,\n"
            "\tminY: number,\n"
            "\tmaxY: number,\n"
            '\tbiome: "abyss" | "depths" | "canopy" | "summit",\n'
            "\tambientColor: Color3,\n"
            "}\n\n"
            "-- Usage in another module:\n"
            "--[[\n"
            "local Types = require(Shared.Util.Types)\n"
            "type PlayerState = Types.PlayerState\n"
            "type AbilityConfig = Types.AbilityConfig\n"
            "\n"
            "local function updateState(state: PlayerState, dt: number): PlayerState\n"
            "\t...\n"
            "end\n"
            "]]\n\n"
            "return nil\n",
            difficulty=2,
            category="types",
        ),
        _ex(
            "How do I type generic data structures in Luau?",
            "<think>\n"
            "Luau supports generic types with angle bracket syntax. This is useful for "
            "typed containers like pools, caches, and event dispatchers. Generic functions "
            "use the same syntax. In Vertigo, we use generics for object pools, spatial "
            "data structures, and signal wrappers.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- Generic stack\n"
            "type Stack<T> = {\n"
            "\titems: { T },\n"
            "\tsize: number,\n"
            "}\n\n"
            "local function createStack<T>(): Stack<T>\n"
            "\treturn { items = {}, size = 0 }\n"
            "end\n\n"
            "local function push<T>(stack: Stack<T>, item: T)\n"
            "\tstack.size += 1\n"
            "\tstack.items[stack.size] = item\n"
            "end\n\n"
            "local function pop<T>(stack: Stack<T>): T?\n"
            "\tif stack.size == 0 then return nil end\n"
            "\tlocal item = stack.items[stack.size]\n"
            "\tstack.items[stack.size] = nil :: any\n"
            "\tstack.size -= 1\n"
            "\treturn item\n"
            "end\n\n"
            "-- Generic result type (like Rust's Result)\n"
            "type Result<T, E> = {\n"
            "\tok: boolean,\n"
            "\tvalue: T?,\n"
            "\terror: E?,\n"
            "}\n\n"
            "local function ok<T>(value: T): Result<T, string>\n"
            "\treturn { ok = true, value = value, error = nil }\n"
            "end\n\n"
            "local function err<T>(message: string): Result<T, string>\n"
            "\treturn { ok = false, value = nil :: T?, error = message }\n"
            "end\n",
            difficulty=3,
            category="types",
        ),
    ]


# ---------------------------------------------------------------------------
# Signal pattern examples
# ---------------------------------------------------------------------------


def _signal_pattern_examples() -> list[dict]:
    return [
        _ex(
            "How do I use the Signal library for custom events in Vertigo?",
            "<think>\n"
            "Vertigo uses a Signal library (from Packages) for custom pub/sub events "
            "within services. Signals are typed, fire synchronously, and support "
            "Connect/Disconnect/Wait. They replace ad-hoc callback systems and "
            "BindableEvents. Declare signals in Init, connect in Start.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local Signal = require(ReplicatedStorage:WaitForChild("Packages").Signal)\n\n'
            "local ZoneService = {}\n\n"
            "function ZoneService:Init()\n"
            "\t-- Typed signal: fires with (player, oldZone, newZone)\n"
            "\tself.ZoneChanged = Signal.new() -- :: Signal<Player, string?, string>\n"
            "end\n\n"
            "function ZoneService:Start()\n"
            "\t-- Internal logic fires the signal\n"
            "\t-- self.ZoneChanged:Fire(player, oldZone, newZone)\n"
            "end\n\n"
            "-- Other services connect during their Start:\n"
            "--[[\n"
            "function LightingService:Start()\n"
            "\tZoneService.ZoneChanged:Connect(function(player, oldZone, newZone)\n"
            "\t\tself:_transitionLighting(newZone)\n"
            "\tend)\n"
            "end\n"
            "]]\n\n"
            "return ZoneService\n",
            difficulty=2,
            category="service",
        ),
        _ex(
            "How do I properly clean up with Trove in Vertigo controllers?",
            "<think>\n"
            "Trove (from Packages) is a cleanup utility that tracks objects, connections, "
            "and callbacks for batch destruction. In Vertigo controllers, each state or "
            "ability phase gets its own Trove. When the phase ends, trove:Clean() "
            "disconnects all signals, destroys instances, and calls cleanup functions.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local RunService = game:GetService("RunService")\n'
            'local Trove = require(ReplicatedStorage:WaitForChild("Packages").Trove)\n\n'
            "local GlideController = {}\n\n"
            "function GlideController:Init()\n"
            "\tself._activeTrove = nil :: typeof(Trove.new())?\n"
            "end\n\n"
            "function GlideController:StartGlide()\n"
            "\t-- Clean up any previous glide state\n"
            "\tif self._activeTrove then\n"
            "\t\tself._activeTrove:Clean()\n"
            "\tend\n\n"
            "\tlocal trove = Trove.new()\n"
            "\tself._activeTrove = trove\n\n"
            "\t-- Trove tracks connections — auto-disconnects on Clean()\n"
            "\ttrove:Connect(RunService.RenderStepped, function(dt: number)\n"
            "\t\tself:_updateGlide(dt)\n"
            "\tend)\n\n"
            "\t-- Trove tracks instances — auto-destroys on Clean()\n"
            '\tlocal trail = trove:Construct(Instance, "Trail")\n'
            "\ttrail.Parent = workspace\n\n"
            "\t-- Trove tracks callbacks — auto-calls on Clean()\n"
            "\ttrove:Add(function()\n"
            '\t\tprint("[Glide] Cleaned up")\n'
            "\tend)\n"
            "end\n\n"
            "function GlideController:StopGlide()\n"
            "\tif self._activeTrove then\n"
            "\t\tself._activeTrove:Clean() -- Disconnects, destroys, calls cleanup\n"
            "\t\tself._activeTrove = nil\n"
            "\tend\n"
            "end\n\n"
            "return GlideController\n",
            difficulty=3,
            category="controller",
        ),
        _ex(
            "How do I implement an event dispatcher pattern for cross-service communication?",
            "<think>\n"
            "In Vertigo, services sometimes need to communicate without direct references. "
            "An event dispatcher (pub/sub) lets services fire events by name and other "
            "services subscribe. This decouples systems. The dispatcher is a shared module "
            "required by all participating services.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "type Callback = (...any) -> ()\n\n"
            "local EventDispatcher = {}\n"
            "EventDispatcher._listeners = {} :: { [string]: { Callback } }\n\n"
            "function EventDispatcher:On(event: string, callback: Callback): () -> ()\n"
            "\tlocal listeners = self._listeners[event]\n"
            "\tif not listeners then\n"
            "\t\tlisteners = {}\n"
            "\t\tself._listeners[event] = listeners\n"
            "\tend\n"
            "\ttable.insert(listeners, callback)\n\n"
            "\t-- Return unsubscribe function\n"
            "\tlocal index = #listeners\n"
            "\treturn function()\n"
            "\t\tlocal current = self._listeners[event]\n"
            "\t\tif current then\n"
            "\t\t\tlocal pos = table.find(current, callback)\n"
            "\t\t\tif pos then\n"
            "\t\t\t\ttable.remove(current, pos)\n"
            "\t\t\tend\n"
            "\t\tend\n"
            "\tend\n"
            "end\n\n"
            "function EventDispatcher:Fire(event: string, ...: any)\n"
            "\tlocal listeners = self._listeners[event]\n"
            "\tif not listeners then return end\n"
            "\tfor _, callback in listeners do\n"
            "\t\ttask.spawn(callback, ...)\n"
            "\tend\n"
            "end\n\n"
            "return EventDispatcher\n",
            difficulty=3,
            category="networking",
        ),
        _ex(
            "How do I handle character death and respawn in a Vertigo service?",
            "<think>\n"
            "Character death triggers Humanoid.Died. The service must clean up any active "
            "state (abilities, effects, timers) on death, then reinitialize on respawn "
            "via CharacterAdded. In Vertigo, death cancels active traversal (grapple, "
            "glide), resets physics state, and triggers respawn after a delay.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "local CharacterService = {}\n\n"
            "type CharacterState = {\n"
            "\thumanoid: Humanoid,\n"
            "\trootPart: BasePart,\n"
            "\tconnections: { RBXScriptConnection },\n"
            "}\n\n"
            "function CharacterService:Init()\n"
            "\tself._states = {} :: { [Player]: CharacterState }\n"
            "end\n\n"
            "function CharacterService:Start()\n"
            "\tPlayers.PlayerAdded:Connect(function(player: Player)\n"
            "\t\tplayer.CharacterAdded:Connect(function(character: Model)\n"
            "\t\t\tself:_onCharacterAdded(player, character)\n"
            "\t\tend)\n"
            "\tend)\n\n"
            "\tPlayers.PlayerRemoving:Connect(function(player: Player)\n"
            "\t\tself:_cleanupCharacter(player)\n"
            "\tend)\n"
            "end\n\n"
            "function CharacterService:_onCharacterAdded(player: Player, character: Model)\n"
            "\t-- Clean up previous character state\n"
            "\tself:_cleanupCharacter(player)\n\n"
            '\tlocal humanoid = character:WaitForChild("Humanoid") :: Humanoid\n'
            '\tlocal rootPart = character:WaitForChild("HumanoidRootPart") :: BasePart\n'
            "\tlocal connections: { RBXScriptConnection } = {}\n\n"
            "\ttable.insert(connections, humanoid.Died:Connect(function()\n"
            "\t\tself:_onDied(player)\n"
            "\tend))\n\n"
            "\tself._states[player] = {\n"
            "\t\thumanoid = humanoid,\n"
            "\t\trootPart = rootPart,\n"
            "\t\tconnections = connections,\n"
            "\t}\n"
            "end\n\n"
            "function CharacterService:_onDied(player: Player)\n"
            '\tprint("[Character]", player.Name, "died")\n'
            "\t-- Cancel active abilities, reset state\n"
            "\t-- Respawn after delay\n"
            "\ttask.delay(3, function()\n"
            "\t\tif player.Parent then -- Still connected?\n"
            "\t\t\tplayer:LoadCharacter()\n"
            "\t\tend\n"
            "\tend)\n"
            "end\n\n"
            "function CharacterService:_cleanupCharacter(player: Player)\n"
            "\tlocal state = self._states[player]\n"
            "\tif state then\n"
            "\t\tfor _, conn in state.connections do\n"
            "\t\t\tconn:Disconnect()\n"
            "\t\tend\n"
            "\tend\n"
            "\tself._states[player] = nil\n"
            "end\n\n"
            "return CharacterService\n",
            difficulty=4,
            category="service",
        ),
        _ex(
            "How do I use task.cancel to stop scheduled operations?",
            "<think>\n"
            "task.delay and task.spawn return a thread handle that can be cancelled with "
            "task.cancel. This is important for cleanup — if a player leaves while a "
            "delayed operation is pending, cancel it. In Vertigo, we cancel respawn "
            "timers, cooldown resets, and delayed VFX on cleanup.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local pendingRespawns: { [Player]: thread } = {}\n\n"
            "local function scheduleRespawn(player: Player, delay: number)\n"
            "\t-- Cancel any existing respawn timer\n"
            "\tlocal existing = pendingRespawns[player]\n"
            "\tif existing then\n"
            "\t\ttask.cancel(existing)\n"
            "\tend\n\n"
            "\t-- Schedule new respawn\n"
            "\tpendingRespawns[player] = task.delay(delay, function()\n"
            "\t\tpendingRespawns[player] = nil\n"
            "\t\tif player.Parent then\n"
            "\t\t\tplayer:LoadCharacter()\n"
            "\t\tend\n"
            "\tend)\n"
            "end\n\n"
            "local function cancelRespawn(player: Player)\n"
            "\tlocal thread = pendingRespawns[player]\n"
            "\tif thread then\n"
            "\t\ttask.cancel(thread)\n"
            "\t\tpendingRespawns[player] = nil\n"
            "\tend\n"
            "end\n\n"
            "-- On player leaving, cancel their timer\n"
            'game:GetService("Players").PlayerRemoving:Connect(function(player: Player)\n'
            "\tcancelRespawn(player)\n"
            "end)\n",
            difficulty=2,
        ),
        _ex(
            "How do I implement exponential damping for camera smoothing in Vertigo?",
            "<think>\n"
            "Exponential damping with math.exp(-k*dt) gives frame-rate independent smoothing. "
            "The parameter k controls responsiveness: higher = snappier. This is used "
            "throughout Vertigo for camera follow, UI transitions, and value smoothing. "
            "It mathematically guarantees identical behavior at any frame rate.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n'
            'local Players = game:GetService("Players")\n\n'
            "local CAMERA_STIFFNESS = 12 -- Higher = more responsive\n"
            "local CAMERA_OFFSET = CFrame.new(0, 8, 16)\n\n"
            "local localPlayer = Players.LocalPlayer\n"
            "local camera = workspace.CurrentCamera\n"
            "local smoothCF = CFrame.identity\n\n"
            "--[[@native]]\n"
            "local function lerpCFrame(a: CFrame, b: CFrame, t: number): CFrame\n"
            "\treturn a:Lerp(b, t)\n"
            "end\n\n"
            "RunService.RenderStepped:Connect(function(dt: number)\n"
            "\tlocal character = localPlayer.Character\n"
            "\tif not character then return end\n"
            '\tlocal rootPart = character:FindFirstChild("HumanoidRootPart")\n'
            "\tif not rootPart then return end\n\n"
            "\tlocal targetCF = (rootPart :: BasePart).CFrame * CAMERA_OFFSET\n"
            "\tlocal alpha = 1 - math.exp(-CAMERA_STIFFNESS * dt)\n"
            "\tsmoothCF = lerpCFrame(smoothCF, targetCF, alpha)\n"
            "\tcamera.CFrame = smoothCF\n"
            "end)\n",
            difficulty=3,
            category="controller",
        ),
        _ex(
            "How do I create a cooldown system for abilities in Vertigo?",
            "<think>\n"
            "A cooldown system tracks per-player, per-ability timestamps. When an ability "
            "is used, record the current time. Before allowing another use, check if enough "
            "time has elapsed. In Vertigo, cooldowns are server-authoritative — the server "
            "rejects requests during cooldown even if the client shows it as ready.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "type CooldownMap = { [string]: number } -- ability_id -> expiry time\n\n"
            "local CooldownManager = {}\n"
            "CooldownManager._cooldowns = {} :: { [Player]: CooldownMap }\n\n"
            "function CooldownManager:IsReady(player: Player, abilityId: string): boolean\n"
            "\tlocal map = self._cooldowns[player]\n"
            "\tif not map then return true end\n"
            "\tlocal expiry = map[abilityId]\n"
            "\tif not expiry then return true end\n"
            "\treturn os.clock() >= expiry\n"
            "end\n\n"
            "function CooldownManager:Use(player: Player, abilityId: string, duration: number)\n"
            "\tlocal map = self._cooldowns[player]\n"
            "\tif not map then\n"
            "\t\tmap = {}\n"
            "\t\tself._cooldowns[player] = map\n"
            "\tend\n"
            "\tmap[abilityId] = os.clock() + duration\n"
            "end\n\n"
            "function CooldownManager:GetRemaining(player: Player, abilityId: string): number\n"
            "\tlocal map = self._cooldowns[player]\n"
            "\tif not map then return 0 end\n"
            "\tlocal expiry = map[abilityId]\n"
            "\tif not expiry then return 0 end\n"
            "\treturn math.max(0, expiry - os.clock())\n"
            "end\n\n"
            "function CooldownManager:ClearPlayer(player: Player)\n"
            "\tself._cooldowns[player] = nil\n"
            "end\n\n"
            "return CooldownManager\n",
            difficulty=3,
            category="service",
        ),
        _ex(
            "How do I safely parent instances using the correct order of operations?",
            "<think>\n"
            "In Roblox, the order you set properties matters. Always set all properties "
            "BEFORE setting Parent. This ensures ChildAdded listeners see the fully "
            "configured instance. Additionally, use Anchored = true before setting CFrame "
            "to prevent the part from falling while being positioned.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- CORRECT order of operations for part creation\n"
            "local function createConfiguredPart(config: {\n"
            "\tsize: Vector3,\n"
            "\tcframe: CFrame,\n"
            "\tcolor: Color3,\n"
            "\tmaterial: Enum.Material,\n"
            "\tanchored: boolean,\n"
            "\tparent: Instance,\n"
            "}): BasePart\n"
            '\tlocal part = Instance.new("Part")\n'
            "\t-- 1. Set Anchored first (prevents falling)\n"
            "\tpart.Anchored = config.anchored\n"
            "\t-- 2. Set geometric properties\n"
            "\tpart.Size = config.size\n"
            "\tpart.CFrame = config.cframe\n"
            "\t-- 3. Set visual properties\n"
            "\tpart.Color = config.color\n"
            "\tpart.Material = config.material\n"
            "\t-- 4. Set Parent LAST\n"
            "\tpart.Parent = config.parent\n"
            "\treturn part\n"
            "end\n\n"
            "-- WRONG: Parent first, then configure\n"
            "--[[\n"
            'local part = Instance.new("Part", workspace) -- Events fire with defaults!\n'
            "part.Size = Vector3.new(10, 1, 10) -- Listeners already saw 4x1.2x2\n"
            "]]\n",
            difficulty=1,
            category="debugging",
        ),
        _ex(
            "How do I use string.format and string.split effectively in Luau?",
            "<think>\n"
            "string.format is efficient for formatted output (uses C-style format specifiers). "
            "string.split is the modern way to tokenize strings (replaces gmatch for simple "
            "cases). In Vertigo, string.split + table.find replaces gmatch/gsub in hot "
            "paths because pattern matching is not NCG-friendly.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- string.format: efficient formatted output\n"
            "local function logEvent(player: string, event: string, value: number)\n"
            '\tprint(string.format("[%s] %s: %.2f", player, event, value))\n'
            "end\n\n"
            "-- string.split: tokenize strings\n"
            "local function parseCommand(input: string): (string, { string })\n"
            '\tlocal parts = string.split(input, " ")\n'
            '\tlocal command = parts[1] or ""\n'
            "\tlocal args = table.create(#parts - 1)\n"
            "\tfor i = 2, #parts do\n"
            "\t\targs[i - 1] = parts[i]\n"
            "\tend\n"
            "\treturn command, args\n"
            "end\n\n"
            "-- Hot-path safe tag checking (no gmatch)\n"
            "local function hasPrefix(str: string, prefix: string): boolean\n"
            "\treturn string.sub(str, 1, #prefix) == prefix\n"
            "end\n\n"
            '-- Stable ID parsing: "ability_grapple_v1" -> { "ability", "grapple", "v1" }\n'
            "local function parseStableId(id: string): { string }\n"
            '\treturn string.split(id, "_")\n'
            "end\n",
            difficulty=2,
        ),
        _ex(
            "How do I use coroutine.wrap for iterator patterns in Luau?",
            "<think>\n"
            "coroutine.wrap creates a resumable function that yields values, useful for "
            "lazy iteration. In Vertigo, this is used for paginated DataStore queries "
            "and iterating large spatial regions without loading everything into memory. "
            "Note: for simple iteration, generalized for loops are preferred.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- Iterator over chunks of a table\n"
            "local function chunks<T>(arr: { T }, size: number): () -> ({ T }?, number?)\n"
            "\tlocal i = 0\n"
            "\tlocal n = #arr\n"
            "\treturn function(): ({ T }?, number?)\n"
            "\t\ti += 1\n"
            "\t\tlocal startIdx = (i - 1) * size + 1\n"
            "\t\tif startIdx > n then\n"
            "\t\t\treturn nil, nil\n"
            "\t\tend\n"
            "\t\tlocal chunk = table.create(size)\n"
            "\t\tfor j = startIdx, math.min(startIdx + size - 1, n) do\n"
            "\t\t\ttable.insert(chunk, arr[j])\n"
            "\t\tend\n"
            "\t\treturn chunk, i\n"
            "\tend\n"
            "end\n\n"
            "-- Usage: process parts in batches to avoid frame spikes\n"
            "local function processBatch(allParts: { BasePart })\n"
            "\tfor chunk, batchNum in chunks(allParts, 50) do\n"
            "\t\tfor _, part in chunk :: { BasePart } do\n"
            "\t\t\t-- Process part\n"
            "\t\tend\n"
            "\t\ttask.wait() -- Yield between batches\n"
            "\tend\n"
            "end\n",
            difficulty=3,
        ),
        _ex(
            "How do I use Humanoid:MoveTo for NPC movement with arrival detection?",
            "<think>\n"
            "Humanoid:MoveTo makes an NPC walk toward a point. MoveToFinished fires when "
            "the humanoid reaches the target OR times out (8 seconds). In Vertigo, NPCs "
            "navigate between waypoints. We must handle timeout (re-issue MoveTo) and "
            "check if the NPC actually arrived.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local function moveToWithTimeout(\n"
            "\thumanoid: Humanoid,\n"
            "\ttarget: Vector3,\n"
            "\ttimeout: number?\n"
            "): boolean\n"
            "\tlocal maxTime = timeout or 10\n"
            "\tlocal startTime = os.clock()\n\n"
            "\trepeat\n"
            "\t\thumanoid:MoveTo(target)\n"
            "\t\tlocal reached = humanoid.MoveToFinished:Wait()\n"
            "\t\tif reached then\n"
            "\t\t\treturn true\n"
            "\t\tend\n"
            "\tuntil os.clock() - startTime > maxTime\n\n"
            "\treturn false\n"
            "end\n\n"
            "-- Patrol between waypoints\n"
            "local function patrol(humanoid: Humanoid, waypoints: { Vector3 })\n"
            "\twhile humanoid.Health > 0 do\n"
            "\t\tfor _, point in waypoints do\n"
            "\t\t\tlocal arrived = moveToWithTimeout(humanoid, point, 15)\n"
            "\t\t\tif not arrived then\n"
            '\t\t\t\twarn("[NPC] Failed to reach waypoint")\n'
            "\t\t\tend\n"
            "\t\t\ttask.wait(1) -- Pause at each waypoint\n"
            "\t\tend\n"
            "\tend\n"
            "end\n",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    all_examples: list[dict] = []

    generators = [
        ("DataStoreService", _datastore_examples),
        ("DataStoreService (advanced)", _datastore_advanced_examples),
        ("Players", _players_examples),
        ("Players (advanced)", _players_advanced_examples),
        ("RunService", _runservice_examples),
        ("RunService (advanced)", _runservice_advanced_examples),
        ("ReplicatedStorage", _replicated_storage_examples),
        ("CollectionService", _collection_service_examples),
        ("CollectionService (advanced)", _collection_advanced_examples),
        ("UserInputService", _userinput_examples),
        ("Workspace", _workspace_examples),
        ("Workspace (advanced)", _workspace_advanced_examples),
        ("TweenService", _tween_examples),
        ("TweenService (advanced)", _tween_advanced_examples),
        ("RemoteEvent/RemoteFunction", _remote_examples),
        ("RemoteEvent/RemoteFunction (advanced)", _remote_advanced_examples),
        ("HttpService", _httpservice_examples),
        ("HttpService (advanced)", _http_advanced_examples),
        ("MarketplaceService", _marketplace_examples),
        ("MemoryStoreService", _memorystore_examples),
        ("MessagingService", _messaging_examples),
        ("SoundService/TextService/Lighting", _misc_service_examples),
        ("PhysicsService/PathfindingService", _physics_path_examples),
        ("Deprecated API migrations", _deprecated_examples),
        ("Deprecated API migrations (advanced)", _deprecated_advanced_examples),
        ("Performance best practices", _performance_examples),
        ("Performance (advanced)", _perf_advanced_examples),
        ("Integration / architecture", _integration_examples),
        ("Integration (advanced)", _integration_advanced_examples),
        ("Error handling", _error_handling_examples),
        ("Additional service patterns", _additional_service_examples),
        ("Type annotations", _type_annotation_examples),
        ("Signal patterns", _signal_pattern_examples),
    ]

    for name, gen_fn in generators:
        examples = gen_fn()
        print(f"  {name}: {len(examples)} examples")
        all_examples.extend(examples)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nTotal: {len(all_examples)} training pairs -> {OUTPUT}")

    # Category breakdown
    from collections import Counter

    cats = Counter(ex.get("category", "unknown") for ex in all_examples)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
