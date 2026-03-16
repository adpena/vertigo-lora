#!/usr/bin/env python3
from __future__ import annotations

"""
Generate synthetic DevForum-style Q&A training pairs.

Covers 20 topic categories commonly asked on DevForum, with Vertigo-specific
conventions (--!strict, @native, server-authoritative, Init/Start lifecycle).

Output: data/raw/devforum_qa.jsonl
"""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "devforum_qa.jsonl"

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
    question: str,
    answer: str,
    category: str,
    difficulty: int = 2,
) -> dict:
    """Build a single training example dict."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "source": "devforum_qa",
        "category": category,
        "has_reasoning": True,
        "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# 1. DataStore patterns
# ---------------------------------------------------------------------------


def _datastore_examples() -> list[dict]:
    return [
        _ex(
            "How do I retry DataStore calls that fail due to throttling?",
            "<think>\n"
            "DataStore calls can fail from throttling or network errors. A simple retry "
            "with exponential backoff handles transient failures without hammering the "
            "service. We cap retries to avoid infinite loops and log each attempt.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            "local MAX_RETRIES: number = 5\n"
            "local BASE_DELAY: number = 1\n\n"
            "local function retryAsync<T>(callback: () -> T, label: string): T?\n"
            "\tfor attempt = 1, MAX_RETRIES do\n"
            "\t\tlocal success, result = pcall(callback)\n"
            "\t\tif success then\n"
            "\t\t\treturn result\n"
            "\t\tend\n"
            "\t\twarn(`[DataService] {label} attempt {attempt} failed: {result}`)\n"
            "\t\tif attempt < MAX_RETRIES then\n"
            "\t\t\ttask.wait(BASE_DELAY * (2 ^ (attempt - 1)))\n"
            "\t\tend\n"
            "\tend\n"
            "\twarn(`[DataService] {label} exhausted all {MAX_RETRIES} retries`)\n"
            "\treturn nil\n"
            "end\n",
            category="debugging",
            difficulty=3,
        ),
        _ex(
            "How do I migrate player data when I change the schema version?",
            "<think>\n"
            "Schema migration is critical for live games. We version the data, check the "
            "version on load, and apply migration functions sequentially. UpdateAsync is "
            "ideal because it's atomic — no race conditions during migration.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            "local CURRENT_VERSION: number = 3\n"
            'local playerStore = DataStoreService:GetDataStore("PlayerData_v2")\n\n'
            "type PlayerData = {\n"
            "\tversion: number,\n"
            "\tcoins: number,\n"
            "\tinventory: { string },\n"
            "\tsettings: { [string]: any },\n"
            "}\n\n"
            "local migrations: { [number]: (data: PlayerData) -> PlayerData } = {\n"
            "\t[1] = function(data: PlayerData): PlayerData\n"
            "\t\tdata.inventory = data.inventory or {}\n"
            "\t\tdata.version = 2\n"
            "\t\treturn data\n"
            "\tend,\n"
            "\t[2] = function(data: PlayerData): PlayerData\n"
            "\t\tdata.settings = data.settings or {}\n"
            "\t\tdata.version = 3\n"
            "\t\treturn data\n"
            "\tend,\n"
            "}\n\n"
            "local function migrateData(data: PlayerData): PlayerData\n"
            "\twhile data.version < CURRENT_VERSION do\n"
            "\t\tlocal migrator = migrations[data.version]\n"
            "\t\tif not migrator then\n"
            "\t\t\twarn(`[DataService] No migration for version {data.version}`)\n"
            "\t\t\tbreak\n"
            "\t\tend\n"
            "\t\tdata = migrator(data)\n"
            "\tend\n"
            "\treturn data\n"
            "end\n",
            category="service",
            difficulty=4,
        ),
        _ex(
            "What's the best way to save data on PlayerRemoving without data loss?",
            "<think>\n"
            "PlayerRemoving fires when a player leaves. We must save synchronously before "
            "the server shuts down. game:BindToClose gives extra time during server shutdown. "
            "Track active sessions to ensure every player's data is flushed.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n'
            'local Players = game:GetService("Players")\n\n'
            'local playerStore = DataStoreService:GetDataStore("PlayerData_v2")\n'
            "local sessionData: { [number]: { [string]: any } } = {}\n\n"
            "local function savePlayer(player: Player)\n"
            "\tlocal data = sessionData[player.UserId]\n"
            "\tif not data then return end\n"
            '\tlocal key = "player_" .. tostring(player.UserId)\n'
            "\tlocal ok, err = pcall(function()\n"
            "\t\tplayerStore:SetAsync(key, data)\n"
            "\tend)\n"
            "\tif not ok then\n"
            "\t\twarn(`[DataService] Final save failed for {player.Name}: {err}`)\n"
            "\tend\n"
            "\tsessionData[player.UserId] = nil\n"
            "end\n\n"
            "Players.PlayerRemoving:Connect(savePlayer)\n\n"
            "game:BindToClose(function()\n"
            "\tfor _, player in Players:GetPlayers() do\n"
            "\t\ttask.spawn(savePlayer, player)\n"
            "\tend\n"
            "end)\n",
            category="service",
            difficulty=3,
        ),
        _ex(
            "How do I use session locking to prevent data duplication across servers?",
            "<think>\n"
            "Session locking prevents two servers from writing the same player's data. We "
            "store a session GUID with the data. On load, if the GUID doesn't match, another "
            "server still owns the session — we wait or reject. This prevents item duplication "
            "from server-hopping exploits.\n"
            "</think>\n\n"
            "--!strict\n"
            'local HttpService = game:GetService("HttpService")\n'
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            'local playerStore = DataStoreService:GetDataStore("PlayerData_v2")\n'
            'local SESSION_KEY = "__sessionId"\n'
            "local mySessionId: string = HttpService:GenerateGUID(false)\n\n"
            "local function acquireSession(userId: number): boolean\n"
            '\tlocal key = "player_" .. tostring(userId)\n'
            "\tlocal acquired = false\n"
            "\tlocal ok, err = pcall(function()\n"
            "\t\tplayerStore:UpdateAsync(key, function(data)\n"
            "\t\t\tdata = data or {}\n"
            "\t\t\tlocal existingSession = data[SESSION_KEY]\n"
            "\t\t\tif existingSession and existingSession ~= mySessionId then\n"
            "\t\t\t\treturn nil -- another server owns it\n"
            "\t\t\tend\n"
            "\t\t\tdata[SESSION_KEY] = mySessionId\n"
            "\t\t\tacquired = true\n"
            "\t\t\treturn data\n"
            "\t\tend)\n"
            "\tend)\n"
            "\tif not ok then\n"
            "\t\twarn(`[DataService] Session acquire failed: {err}`)\n"
            "\tend\n"
            "\treturn acquired\n"
            "end\n",
            category="service",
            difficulty=5,
        ),
    ]


# ---------------------------------------------------------------------------
# 2. Remote debugging
# ---------------------------------------------------------------------------


def _remote_debugging_examples() -> list[dict]:
    return [
        _ex(
            "My RemoteEvent isn't working — client fires it but server never receives. What's wrong?",
            "<think>\n"
            "Common causes: 1) Remote isn't in ReplicatedStorage, 2) server script isn't "
            "running (not in ServerScriptService), 3) typo in remote name, 4) connecting "
            "before the remote exists. Check WaitForChild on client, and verify the server "
            "listener is set up in :Start() after Init.\n"
            "</think>\n\n"
            "Most common causes and fixes:\n\n"
            "**1. Remote not replicated** — must be in ReplicatedStorage:\n"
            "```luau\n"
            "-- Shared/Net/Remotes.luau\n"
            'local remotes = ReplicatedStorage:WaitForChild("Remotes")\n'
            "```\n\n"
            "**2. Server listener not connected** — ensure it's in :Start():\n"
            "```luau\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local remotes = ReplicatedStorage:WaitForChild("Remotes")\n\n'
            "local MyService = {}\n\n"
            "function MyService:Start()\n"
            "\tremotes.RequestUseAbility.OnServerEvent:Connect(function(player: Player, abilityId: string)\n"
            "\t\t-- validate and process\n"
            "\t\tprint(`[MyService] {player.Name} used {abilityId}`)\n"
            "\tend)\n"
            "end\n\n"
            "return MyService\n"
            "```\n\n"
            "**3. Client must WaitForChild**:\n"
            "```luau\n"
            'local remotes = ReplicatedStorage:WaitForChild("Remotes")\n'
            'remotes.RequestUseAbility:FireServer("ability_grapple_v1")\n'
            "```\n\n"
            "**4. Check spelling** — remote names are case-sensitive.",
            category="debugging",
            difficulty=2,
        ),
        _ex(
            "How do I debug RemoteEvent data not arriving correctly on the server?",
            "<think>\n"
            "Data serialization issues: Instance references, metatables, functions, and "
            "mixed tables don't cross the remote boundary cleanly. Use print/warn on both "
            "sides to inspect what's sent vs received. Keep payloads flat and serializable.\n"
            "</think>\n\n"
            "--!strict\n"
            "-- SERVER: Log raw incoming data\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local remotes = ReplicatedStorage:WaitForChild("Remotes")\n\n'
            "remotes.RequestUseAbility.OnServerEvent:Connect(function(player: Player, ...)\n"
            "\tlocal args = { ... }\n"
            "\tprint(`[DebugRemote] From {player.Name}, arg count: {#args}`)\n"
            "\tfor i, v in args do\n"
            "\t\tprint(`  [{i}] type={typeof(v)} value={tostring(v)}`)\n"
            "\tend\n"
            "end)\n\n"
            "-- Common pitfalls:\n"
            "-- 1. Sending a table with Instance keys — they become nil on the other side\n"
            "-- 2. Sending functions — they are silently dropped\n"
            "-- 3. Sending mixed tables (array + dict) — behavior is unpredictable\n"
            "-- 4. CFrame/Vector3 cross fine, but custom metatables do not\n",
            category="debugging",
            difficulty=2,
        ),
        _ex(
            "My RemoteFunction call hangs forever and the client freezes. Help!",
            "<think>\n"
            "RemoteFunction:InvokeServer yields until the server callback returns. If the "
            "server callback errors or never returns, the client hangs. Always pcall the "
            "invoke, add a timeout, and prefer RemoteEvent for fire-and-forget actions. "
            "RemoteFunction is only for request-response patterns like GetPlayerProfile.\n"
            "</think>\n\n"
            "--!strict\n"
            "-- SAFE client-side RemoteFunction call with timeout\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local remotes = ReplicatedStorage:WaitForChild("Remotes")\n\n'
            "local function safeInvoke(remote: RemoteFunction, timeout: number, ...: any): (boolean, any)\n"
            "\tlocal result: any = nil\n"
            "\tlocal done = false\n"
            "\tlocal args = { ... }\n\n"
            "\ttask.spawn(function()\n"
            "\t\tlocal ok, val = pcall(function()\n"
            "\t\t\treturn remote:InvokeServer(table.unpack(args))\n"
            "\t\tend)\n"
            "\t\tresult = if ok then val else nil\n"
            "\t\tdone = true\n"
            "\tend)\n\n"
            "\tlocal elapsed = 0\n"
            "\twhile not done and elapsed < timeout do\n"
            "\t\ttask.wait(0.1)\n"
            "\t\telapsed += 0.1\n"
            "\tend\n\n"
            "\tif not done then\n"
            '\t\twarn("[Net] RemoteFunction timed out after", timeout, "seconds")\n'
            "\t\treturn false, nil\n"
            "\tend\n"
            "\treturn true, result\n"
            "end\n",
            category="debugging",
            difficulty=3,
        ),
        _ex(
            "How do I rate-limit RemoteEvent calls from clients to prevent spam?",
            "<think>\n"
            "Without rate limiting, exploiters can spam RemoteEvents thousands of times per "
            "second, lagging the server. Track last fire time per player per remote. Reject "
            "calls that arrive too fast. This is server-authoritative validation.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "local RATE_LIMIT: number = 0.2 -- 5 calls per second max\n"
            "local lastCallTime: { [number]: number } = {}\n\n"
            "local function isRateLimited(player: Player): boolean\n"
            "\tlocal now = tick()\n"
            "\tlocal lastTime = lastCallTime[player.UserId] or 0\n"
            "\tif now - lastTime < RATE_LIMIT then\n"
            "\t\treturn true\n"
            "\tend\n"
            "\tlastCallTime[player.UserId] = now\n"
            "\treturn false\n"
            "end\n\n"
            "-- Clean up on leave\n"
            "Players.PlayerRemoving:Connect(function(player: Player)\n"
            "\tlastCallTime[player.UserId] = nil\n"
            "end)\n",
            category="networking",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# 3. Camera follow systems
# ---------------------------------------------------------------------------


def _camera_examples() -> list[dict]:
    return [
        _ex(
            "How do I make a smooth third-person camera that follows the player?",
            "<think>\n"
            "A smooth follow camera lerps toward the target position each frame. We use "
            "RenderStepped for camera work because it runs before the frame renders, giving "
            "the smoothest result. The @native annotation is essential for per-frame code. "
            "math.lerp gives us smooth interpolation.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n'
            'local Players = game:GetService("Players")\n\n'
            "local OFFSET = Vector3.new(0, 8, 12)\n"
            "local SMOOTHING: number = 0.1\n\n"
            "local camera = workspace.CurrentCamera\n"
            "local player = Players.LocalPlayer\n\n"
            "@native\n"
            "local function updateCamera(dt: number)\n"
            "\tlocal character = player.Character\n"
            "\tif not character then return end\n"
            '\tlocal root = character:FindFirstChild("HumanoidRootPart") :: BasePart?\n'
            "\tif not root then return end\n\n"
            "\tlocal targetPos = root.Position + OFFSET\n"
            "\tlocal currentPos = camera.CFrame.Position\n"
            "\tlocal smoothed = currentPos:Lerp(targetPos, SMOOTHING)\n"
            "\tcamera.CFrame = CFrame.lookAt(smoothed, root.Position)\n"
            "end\n\n"
            "RunService.RenderStepped:Connect(updateCamera)\n",
            category="controller",
            difficulty=2,
        ),
        _ex(
            "How do I make an orbit camera that rotates around the player with mouse input?",
            "<think>\n"
            "An orbit camera tracks mouse delta to adjust yaw and pitch, then positions "
            "the camera on a sphere around the player. We clamp pitch to prevent flipping. "
            "UserInputService gives mouse delta when the cursor is locked.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n'
            'local UserInputService = game:GetService("UserInputService")\n'
            'local Players = game:GetService("Players")\n\n'
            "local DISTANCE: number = 15\n"
            "local SENSITIVITY: number = 0.003\n"
            "local MIN_PITCH: number = math.rad(-80)\n"
            "local MAX_PITCH: number = math.rad(80)\n\n"
            "local yaw: number = 0\n"
            "local pitch: number = math.rad(20)\n"
            "local camera = workspace.CurrentCamera\n"
            "local player = Players.LocalPlayer\n\n"
            "UserInputService.InputChanged:Connect(function(input: InputObject)\n"
            "\tif input.UserInputType == Enum.UserInputType.MouseMovement then\n"
            "\t\tyaw -= input.Delta.X * SENSITIVITY\n"
            "\t\tpitch = math.clamp(pitch - input.Delta.Y * SENSITIVITY, MIN_PITCH, MAX_PITCH)\n"
            "\tend\n"
            "end)\n\n"
            "@native\n"
            "local function updateOrbitCamera(dt: number)\n"
            "\tlocal character = player.Character\n"
            "\tif not character then return end\n"
            '\tlocal root = character:FindFirstChild("HumanoidRootPart") :: BasePart?\n'
            "\tif not root then return end\n\n"
            "\tlocal cx = math.cos(yaw) * math.cos(pitch) * DISTANCE\n"
            "\tlocal cy = math.sin(pitch) * DISTANCE\n"
            "\tlocal cz = math.sin(yaw) * math.cos(pitch) * DISTANCE\n"
            "\tlocal offset = Vector3.new(cx, cy, cz)\n"
            "\tcamera.CFrame = CFrame.lookAt(root.Position + offset, root.Position)\n"
            "end\n\n"
            "RunService.RenderStepped:Connect(updateOrbitCamera)\n",
            category="controller",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# 4. Client/server boundary issues
# ---------------------------------------------------------------------------


def _client_server_examples() -> list[dict]:
    return [
        _ex(
            "I'm getting an error using game:GetService('Players').LocalPlayer on the server. Why?",
            "<think>\n"
            "LocalPlayer only exists on the client. Server scripts cannot access it — there "
            "is no single 'local' player on the server, it handles all of them. On the server, "
            "iterate Players:GetPlayers() or use the player argument from remote events.\n"
            "</think>\n\n"
            "`Players.LocalPlayer` is client-only. On the server, you get the player from:\n\n"
            "**1. Remote event callbacks** (most common):\n"
            "```luau\n"
            "--!strict\n"
            "remote.OnServerEvent:Connect(function(player: Player, data: string)\n"
            "\t-- 'player' is automatically passed by the engine\n"
            "\tprint(player.Name, data)\n"
            "end)\n"
            "```\n\n"
            "**2. Players:GetPlayers()** for iterating all:\n"
            "```luau\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n'
            "for _, player in Players:GetPlayers() do\n"
            "\tprint(player.Name)\n"
            "end\n"
            "```\n\n"
            "**3. Players.PlayerAdded** for per-player setup:\n"
            "```luau\n"
            "--!strict\n"
            "Players.PlayerAdded:Connect(function(player: Player)\n"
            "\t-- set up data, leaderstats, etc.\n"
            "end)\n"
            "```\n",
            category="debugging",
            difficulty=1,
        ),
        _ex(
            "How do I replicate a value from server to all clients efficiently?",
            "<think>\n"
            "Options: Attributes (automatic replication, good for simple values), RemoteEvent "
            "(explicit push, good for complex data), ValueObjects (legacy). Attributes are "
            "the modern approach for per-instance state. RemoteEvent is better for bulk or "
            "structured data like StateSync in Vertigo.\n"
            "</think>\n\n"
            "--!strict\n"
            "-- Option 1: Attributes (auto-replicate, best for simple per-instance values)\n"
            "-- SERVER:\n"
            "local part = workspace.SomePart\n"
            'part:SetAttribute("Health", 100)\n'
            "-- CLIENT (automatically available):\n"
            'local health = part:GetAttribute("Health") :: number\n'
            'part:GetAttributeChangedSignal("Health"):Connect(function()\n'
            '\tlocal newHealth = part:GetAttribute("Health") :: number\n'
            "\tprint(`Health changed to {newHealth}`)\n"
            "end)\n\n"
            "-- Option 2: RemoteEvent (explicit push, structured data)\n"
            "-- SERVER:\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local remotes = ReplicatedStorage:WaitForChild("Remotes")\n'
            "remotes.StateSync:FireAllClients({\n"
            '\ttype = "zone_entered",\n'
            '\tzoneId = "zone_canopy_v1",\n'
            "\tdata = { ambientColor = Color3.fromRGB(120, 200, 140) },\n"
            "})\n",
            category="networking",
            difficulty=2,
        ),
        _ex(
            "Why can't my client script see a part the server just created?",
            "<think>\n"
            "Parts created on the server replicate automatically to clients — but there's "
            "a brief delay. The client script may run before replication finishes. Use "
            "WaitForChild or CollectionService tags with GetInstanceAddedSignal to wait "
            "for the part reliably.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n\n'
            "-- Bad: direct access may be nil if replication hasn't finished\n"
            "-- local part = workspace.DynamicParts.MyPart -- might be nil!\n\n"
            "-- Good: WaitForChild (blocks until available)\n"
            'local container = workspace:WaitForChild("DynamicParts")\n'
            'local part = container:WaitForChild("MyPart", 10) -- 10s timeout\n'
            "if not part then\n"
            '\twarn("[Client] MyPart never replicated")\n'
            "\treturn\n"
            "end\n\n"
            "-- Better for tagged objects: CollectionService\n"
            "local function onGrappleAnchorAdded(instance: Instance)\n"
            "\tprint(`Anchor replicated: {instance.Name}`)\n"
            "end\n\n"
            'for _, anchor in CollectionService:GetTagged("GrappleAnchor") do\n'
            "\tonGrappleAnchorAdded(anchor)\n"
            "end\n"
            'CollectionService:GetInstanceAddedSignal("GrappleAnchor"):Connect(onGrappleAnchorAdded)\n',
            category="debugging",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# 5. Custom character controllers
# ---------------------------------------------------------------------------


def _character_controller_examples() -> list[dict]:
    return [
        _ex(
            "How do I make a custom character controller that replaces the default Humanoid movement?",
            "<think>\n"
            "Disable default Humanoid controls by setting WalkSpeed to 0 and using a custom "
            "state machine. Read raw input, apply forces via VectorForce or set Velocity "
            "directly. Vertigo uses this pattern for traversal states — the Humanoid is "
            "only used for health/death, not movement.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n'
            'local Players = game:GetService("Players")\n'
            'local UserInputService = game:GetService("UserInputService")\n\n'
            "local MOVE_SPEED: number = 50\n"
            "local player = Players.LocalPlayer\n\n"
            "local moveDirection = Vector3.zero\n\n"
            "local function getInputDirection(): Vector3\n"
            "\tlocal dir = Vector3.zero\n"
            "\tif UserInputService:IsKeyDown(Enum.KeyCode.W) then\n"
            "\t\tdir += Vector3.new(0, 0, -1)\n"
            "\tend\n"
            "\tif UserInputService:IsKeyDown(Enum.KeyCode.S) then\n"
            "\t\tdir += Vector3.new(0, 0, 1)\n"
            "\tend\n"
            "\tif UserInputService:IsKeyDown(Enum.KeyCode.A) then\n"
            "\t\tdir += Vector3.new(-1, 0, 0)\n"
            "\tend\n"
            "\tif UserInputService:IsKeyDown(Enum.KeyCode.D) then\n"
            "\t\tdir += Vector3.new(1, 0, 0)\n"
            "\tend\n"
            "\treturn if dir.Magnitude > 0 then dir.Unit else Vector3.zero\n"
            "end\n\n"
            "@native\n"
            "local function updateMovement(dt: number)\n"
            "\tlocal character = player.Character\n"
            "\tif not character then return end\n"
            '\tlocal root = character:FindFirstChild("HumanoidRootPart") :: BasePart?\n'
            "\tif not root then return end\n\n"
            "\tlocal camCF = workspace.CurrentCamera.CFrame\n"
            "\tlocal inputDir = getInputDirection()\n"
            "\tlocal worldDir = camCF:VectorToWorldSpace(inputDir)\n"
            "\tworldDir = Vector3.new(worldDir.X, 0, worldDir.Z)\n"
            "\tif worldDir.Magnitude > 0 then\n"
            "\t\tworldDir = worldDir.Unit\n"
            "\tend\n\n"
            "\troot.AssemblyLinearVelocity = Vector3.new(\n"
            "\t\tworldDir.X * MOVE_SPEED,\n"
            "\t\troot.AssemblyLinearVelocity.Y,\n"
            "\t\tworldDir.Z * MOVE_SPEED\n"
            "\t)\n"
            "end\n\n"
            "RunService.Heartbeat:Connect(updateMovement)\n",
            category="controller",
            difficulty=4,
        ),
        _ex(
            "How do I detect if the player is grounded for a custom jump?",
            "<think>\n"
            "Raycast downward from the root part to detect ground. The ray should be "
            "slightly longer than the distance from root to feet. Use RaycastParams to "
            "exclude the character. This is a hot-path function — mark @native.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local GROUND_RAY_LENGTH: number = 3.5 -- root part center to feet + margin\n"
            "local rayParams = RaycastParams.new()\n"
            "rayParams.FilterType = Enum.RaycastFilterType.Exclude\n\n"
            "@native\n"
            "local function isGrounded(root: BasePart): boolean\n"
            "\trayParams.FilterDescendantsInstances = { root.Parent :: Instance }\n"
            "\tlocal result = workspace:Raycast(\n"
            "\t\troot.Position,\n"
            "\t\tVector3.new(0, -GROUND_RAY_LENGTH, 0),\n"
            "\t\trayParams\n"
            "\t)\n"
            "\treturn result ~= nil\n"
            "end\n",
            category="physics",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# 6. Performance optimization
# ---------------------------------------------------------------------------


def _performance_examples() -> list[dict]:
    return [
        _ex(
            "My game lags with lots of parts. How do I optimize?",
            "<think>\n"
            "Common causes: too many unanchored parts (physics sim), too many unique meshes "
            "(draw calls), per-frame Instance.new() calls. Solutions: anchor static parts, "
            "use StreamingEnabled, pool objects, merge static geometry, and avoid creating "
            "instances in Heartbeat.\n"
            "</think>\n\n"
            "Key optimization strategies:\n\n"
            "**1. Enable StreamingEnabled** — only loads nearby parts:\n"
            "```luau\n"
            "-- In workspace properties:\n"
            "-- StreamingEnabled = true\n"
            "-- StreamingMinRadius = 128\n"
            "-- StreamingTargetRadius = 512\n"
            "```\n\n"
            "**2. Pool parts instead of creating/destroying**:\n"
            "```luau\n"
            "--!strict\n"
            "local partPool: { BasePart } = table.create(200)\n"
            "local poolSize: number = 0\n\n"
            "local function getPart(): BasePart\n"
            "\tif poolSize > 0 then\n"
            "\t\tlocal part = partPool[poolSize]\n"
            "\t\tpartPool[poolSize] = nil :: any\n"
            "\t\tpoolSize -= 1\n"
            "\t\tpart.Parent = workspace\n"
            "\t\treturn part\n"
            "\tend\n"
            '\tlocal part = Instance.new("Part")\n'
            "\tpart.Anchored = true\n"
            "\treturn part\n"
            "end\n\n"
            "local function returnPart(part: BasePart)\n"
            "\tpart.Parent = nil\n"
            "\tpoolSize += 1\n"
            "\tpartPool[poolSize] = part\n"
            "end\n"
            "```\n\n"
            "**3. Anchor static geometry** — unanchored parts cost physics simulation.\n\n"
            "**4. Avoid Instance.new() in Heartbeat** — pre-create in :Init().\n",
            category="general_luau",
            difficulty=2,
        ),
        _ex(
            "How do I profile which parts of my code are slow?",
            "<think>\n"
            "Use debug.profilebegin/end to tag sections in the MicroProfiler (Ctrl+F6 in "
            "Studio). This shows exactly which functions take the most time per frame. "
            "Also use os.clock() for simple timing measurements.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- Method 1: MicroProfiler tags (visible in Ctrl+F6)\n"
            "@native\n"
            "local function updatePhysics(dt: number)\n"
            '\tdebug.profilebegin("Vertigo_PhysicsUpdate")\n'
            "\t-- expensive work here\n"
            "\tdebug.profileend()\n"
            "end\n\n"
            "-- Method 2: os.clock() for logging\n"
            "local function benchmarkOperation(label: string, fn: () -> ())\n"
            "\tlocal start = os.clock()\n"
            "\tfn()\n"
            "\tlocal elapsed = os.clock() - start\n"
            "\tprint(`[Perf] {label}: {elapsed * 1000:.2f}ms`)\n"
            "end\n\n"
            "-- Method 3: Stats service\n"
            'local Stats = game:GetService("Stats")\n'
            "print(`Heartbeat: {Stats.HeartbeatTimeMs:.1f}ms`)\n",
            category="debugging",
            difficulty=2,
        ),
        _ex(
            "How do I reduce network bandwidth from RemoteEvents?",
            "<think>\n"
            "Send minimal data. Use numeric IDs instead of strings. Batch updates instead "
            "of firing per-tick. Compress repeated data with delta encoding. Vertigo uses "
            "StateSync with structured payloads rather than many individual remotes.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local RunService = game:GetService("RunService")\n\n'
            'local remotes = ReplicatedStorage:WaitForChild("Remotes")\n\n'
            "-- BAD: firing every frame\n"
            "-- RunService.Heartbeat:Connect(function()\n"
            "--     remote:FireAllClients(position, rotation, velocity)\n"
            "-- end)\n\n"
            "-- GOOD: batch at fixed rate\n"
            "local SYNC_RATE: number = 1 / 20 -- 20 Hz\n"
            "local accumulator: number = 0\n"
            "local pendingUpdates: { { id: number, pos: Vector3 } } = {}\n\n"
            "@native\n"
            "local function onHeartbeat(dt: number)\n"
            "\taccumulator += dt\n"
            "\tif accumulator < SYNC_RATE then return end\n"
            "\taccumulator = 0\n\n"
            "\tif #pendingUpdates == 0 then return end\n"
            "\tremotes.StateSync:FireAllClients(pendingUpdates)\n"
            "\ttable.clear(pendingUpdates)\n"
            "end\n\n"
            "RunService.Heartbeat:Connect(onHeartbeat)\n",
            category="networking",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# 7. Leaderboards (OrderedDataStore)
# ---------------------------------------------------------------------------


def _leaderboard_examples() -> list[dict]:
    return [
        _ex(
            "How do I create a persistent leaderboard that shows the top 10 players?",
            "<think>\n"
            "OrderedDataStore supports GetSortedAsync for ranked queries. We fetch the top "
            "entries, display them in a SurfaceGui or BillboardGui. Refresh periodically "
            "rather than every frame to avoid throttling.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            'local leaderboard = DataStoreService:GetOrderedDataStore("HighScores_v1")\n\n'
            "type LeaderboardEntry = {\n"
            "\trank: number,\n"
            "\tname: string,\n"
            "\tscore: number,\n"
            "}\n\n"
            "local function getTopPlayers(count: number): { LeaderboardEntry }\n"
            "\tlocal ok, pages = pcall(function()\n"
            "\t\treturn leaderboard:GetSortedAsync(false, count)\n"
            "\tend)\n"
            "\tif not ok then\n"
            "\t\twarn(`[Leaderboard] GetSortedAsync failed: {pages}`)\n"
            "\t\treturn {}\n"
            "\tend\n\n"
            "\tlocal entries: { LeaderboardEntry } = {}\n"
            "\tlocal page = pages:GetCurrentPage()\n"
            "\tfor rank, entry in page do\n"
            "\t\ttable.insert(entries, {\n"
            "\t\t\trank = rank,\n"
            "\t\t\tname = entry.key,\n"
            "\t\t\tscore = entry.value,\n"
            "\t\t})\n"
            "\tend\n"
            "\treturn entries\n"
            "end\n\n"
            "-- Refresh every 60 seconds\n"
            "task.spawn(function()\n"
            "\twhile true do\n"
            "\t\tlocal top = getTopPlayers(10)\n"
            "\t\tprint(`[Leaderboard] Refreshed, {#top} entries`)\n"
            "\t\ttask.wait(60)\n"
            "\tend\n"
            "end)\n",
            category="service",
            difficulty=3,
        ),
        _ex(
            "How do I show in-game leaderstats on the player list?",
            "<think>\n"
            "The built-in player list displays IntValue/StringValue children of a folder "
            "named 'leaderstats' under the Player instance. Create it on PlayerAdded, "
            "update values as the player earns points. This is server-side only.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "Players.PlayerAdded:Connect(function(player: Player)\n"
            '\tlocal leaderstats = Instance.new("Folder")\n'
            '\tleaderstats.Name = "leaderstats"\n'
            "\tleaderstats.Parent = player\n\n"
            '\tlocal score = Instance.new("IntValue")\n'
            '\tscore.Name = "Score"\n'
            "\tscore.Value = 0\n"
            "\tscore.Parent = leaderstats\n\n"
            '\tlocal level = Instance.new("IntValue")\n'
            '\tlevel.Name = "Level"\n'
            "\tlevel.Value = 1\n"
            "\tlevel.Parent = leaderstats\n"
            "end)\n\n"
            "-- Update from game logic:\n"
            "local function addScore(player: Player, amount: number)\n"
            '\tlocal leaderstats = player:FindFirstChild("leaderstats")\n'
            "\tif not leaderstats then return end\n"
            '\tlocal score = leaderstats:FindFirstChild("Score") :: IntValue?\n'
            "\tif score then\n"
            "\t\tscore.Value += amount\n"
            "\tend\n"
            "end\n",
            category="service",
            difficulty=1,
        ),
    ]


# ---------------------------------------------------------------------------
# 8. Part.Touched detection
# ---------------------------------------------------------------------------


def _touched_examples() -> list[dict]:
    return [
        _ex(
            "How do I make a kill brick that destroys the player when they touch it?",
            "<think>\n"
            "Part.Touched fires when any part touches it. We need to check if the touching "
            "part belongs to a character by finding a Humanoid in the parent. Then set "
            "Health to 0. Server-side is preferred for kill bricks to prevent exploits.\n"
            "</think>\n\n"
            "--!strict\n"
            "local killBrick = script.Parent :: BasePart\n\n"
            "local function onTouched(other: BasePart)\n"
            "\tlocal character = other.Parent\n"
            "\tif not character then return end\n"
            '\tlocal humanoid = character:FindFirstChildOfClass("Humanoid")\n'
            "\tif not humanoid then return end\n"
            "\tif humanoid.Health <= 0 then return end\n"
            "\thumanoid.Health = 0\n"
            "end\n\n"
            "killBrick.Touched:Connect(onTouched)\n",
            category="general_luau",
            difficulty=1,
        ),
        _ex(
            "Part.Touched fires multiple times rapidly. How do I debounce it?",
            "<think>\n"
            "Touched fires for every physics contact, which can be dozens per second. "
            "Use a debounce table keyed by player to ensure the effect only triggers once "
            "per player per cooldown period.\n"
            "</think>\n\n"
            "--!strict\n"
            "local COOLDOWN: number = 2\n"
            "local debounce: { [Player]: boolean } = {}\n\n"
            "local function onTouched(other: BasePart)\n"
            "\tlocal character = other.Parent\n"
            "\tif not character then return end\n"
            '\tlocal humanoid = character:FindFirstChildOfClass("Humanoid")\n'
            "\tif not humanoid then return end\n\n"
            '\tlocal player = game:GetService("Players"):GetPlayerFromCharacter(character)\n'
            "\tif not player then return end\n"
            "\tif debounce[player] then return end\n\n"
            "\tdebounce[player] = true\n"
            "\tprint(`{player.Name} triggered the pad!`)\n"
            "\t-- do the effect here\n\n"
            "\ttask.delay(COOLDOWN, function()\n"
            "\t\tdebounce[player] = nil\n"
            "\tend)\n"
            "end\n\n"
            "script.Parent.Touched:Connect(onTouched)\n",
            category="general_luau",
            difficulty=2,
        ),
        _ex(
            "How do I make a coin pickup that gives points and disappears?",
            "<think>\n"
            "Touched on the server for authoritative scoring. Give points through a service "
            "function, then destroy or pool the coin. Use debounce so multiple touches in "
            "the same frame don't grant duplicate rewards.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "local coin = script.Parent :: BasePart\n"
            "local collected = false\n\n"
            "local function onTouched(other: BasePart)\n"
            "\tif collected then return end\n"
            "\tlocal character = other.Parent\n"
            "\tif not character then return end\n"
            "\tlocal player = Players:GetPlayerFromCharacter(character)\n"
            "\tif not player then return end\n\n"
            "\tcollected = true\n"
            '\tlocal leaderstats = player:FindFirstChild("leaderstats")\n'
            "\tif leaderstats then\n"
            '\t\tlocal score = leaderstats:FindFirstChild("Score") :: IntValue?\n'
            "\t\tif score then\n"
            "\t\t\tscore.Value += 10\n"
            "\t\tend\n"
            "\tend\n"
            "\tcoin:Destroy()\n"
            "end\n\n"
            "coin.Touched:Connect(onTouched)\n",
            category="general_luau",
            difficulty=1,
        ),
    ]


# ---------------------------------------------------------------------------
# 9. Cooldown systems
# ---------------------------------------------------------------------------


def _cooldown_examples() -> list[dict]:
    return [
        _ex(
            "How do I implement a cooldown for an ability using tick()?",
            "<think>\n"
            "Store the last activation time per player. Compare with tick() on each use "
            "request. Server-side validation prevents clients from bypassing cooldowns. "
            "In Vertigo, cooldowns are config-driven via tuning modules.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "type CooldownTracker = {\n"
            "\tlastUsed: { [number]: { [string]: number } }, -- userId -> abilityId -> timestamp\n"
            "}\n\n"
            "local Cooldowns: CooldownTracker = {\n"
            "\tlastUsed = {},\n"
            "}\n\n"
            "local ABILITY_COOLDOWNS: { [string]: number } = table.freeze({\n"
            "\tability_grapple_v1 = 1.5,\n"
            "\tability_glide_v1 = 0.5,\n"
            "\tability_airdash_v1 = 3.0,\n"
            "})\n\n"
            "local function canUseAbility(userId: number, abilityId: string): boolean\n"
            "\tlocal cooldown = ABILITY_COOLDOWNS[abilityId]\n"
            "\tif not cooldown then return false end\n\n"
            "\tlocal playerCooldowns = Cooldowns.lastUsed[userId]\n"
            "\tif not playerCooldowns then return true end\n\n"
            "\tlocal lastTime = playerCooldowns[abilityId] or 0\n"
            "\treturn tick() - lastTime >= cooldown\n"
            "end\n\n"
            "local function markUsed(userId: number, abilityId: string)\n"
            "\tif not Cooldowns.lastUsed[userId] then\n"
            "\t\tCooldowns.lastUsed[userId] = {}\n"
            "\tend\n"
            "\tCooldowns.lastUsed[userId][abilityId] = tick()\n"
            "end\n",
            category="service",
            difficulty=2,
        ),
        _ex(
            "How do I show a cooldown indicator on the UI with a fill effect?",
            "<think>\n"
            "Use a Frame with ClipDescendants and a child frame whose Size.X.Scale goes "
            "from 0 to 1 as the cooldown progresses. Update each frame with RenderStepped. "
            "Store the end time and compute remaining fraction.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n\n'
            "local cooldownFrame = script.Parent :: Frame\n"
            'local fillBar = cooldownFrame:FindFirstChild("Fill") :: Frame\n'
            'local label = cooldownFrame:FindFirstChild("Label") :: TextLabel\n\n'
            "local endTime: number = 0\n"
            "local duration: number = 0\n\n"
            "local function startCooldown(seconds: number)\n"
            "\tduration = seconds\n"
            "\tendTime = tick() + seconds\n"
            "\tcooldownFrame.Visible = true\n"
            "end\n\n"
            "@native\n"
            "local function updateCooldownUI(dt: number)\n"
            "\tlocal remaining = endTime - tick()\n"
            "\tif remaining <= 0 then\n"
            "\t\tcooldownFrame.Visible = false\n"
            "\t\treturn\n"
            "\tend\n\n"
            "\tlocal fraction = remaining / duration\n"
            "\tfillBar.Size = UDim2.fromScale(fraction, 1)\n"
            '\tlabel.Text = string.format("%.1fs", remaining)\n'
            "end\n\n"
            "RunService.RenderStepped:Connect(updateCooldownUI)\n",
            category="controller",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# 10. Raycasting (guns, projectiles)
# ---------------------------------------------------------------------------


def _raycast_examples() -> list[dict]:
    return [
        _ex(
            "How do I make a gun that raycasts from the camera to detect hits?",
            "<think>\n"
            "The client performs the raycast for instant visual feedback, then sends the hit "
            "info to the server for validation. The server re-raycasts or validates the "
            "claim. Never trust client hit detection — server must verify.\n"
            "</think>\n\n"
            "--!strict\n"
            "-- CLIENT: visual raycast\n"
            'local Players = game:GetService("Players")\n'
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n\n'
            'local remotes = ReplicatedStorage:WaitForChild("Remotes")\n'
            "local camera = workspace.CurrentCamera\n"
            "local MAX_RANGE: number = 300\n\n"
            "local rayParams = RaycastParams.new()\n"
            "rayParams.FilterType = Enum.RaycastFilterType.Exclude\n\n"
            "local function shoot()\n"
            "\tlocal player = Players.LocalPlayer\n"
            "\tif player.Character then\n"
            "\t\trayParams.FilterDescendantsInstances = { player.Character }\n"
            "\tend\n\n"
            "\tlocal origin = camera.CFrame.Position\n"
            "\tlocal direction = camera.CFrame.LookVector * MAX_RANGE\n"
            "\tlocal result = workspace:Raycast(origin, direction, rayParams)\n\n"
            "\tif result then\n"
            "\t\t-- Send to server for validation\n"
            '\t\tremotes.RequestUseAbility:FireServer("shoot", {\n'
            "\t\t\torigin = origin,\n"
            "\t\t\thitPos = result.Position,\n"
            "\t\t\thitInstance = result.Instance,\n"
            "\t\t})\n"
            "\tend\n"
            "end\n",
            category="physics",
            difficulty=3,
        ),
        _ex(
            "How do I make a projectile that follows a ballistic arc using raycasting?",
            "<think>\n"
            "Step the projectile forward each frame using velocity + gravity. Raycast between "
            "the old and new position to detect hits. This is more accurate than moving a "
            "part and relying on Touched, and allows prediction on both client and server.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n\n'
            "local GRAVITY = Vector3.new(0, -workspace.Gravity, 0)\n\n"
            "type Projectile = {\n"
            "\tposition: Vector3,\n"
            "\tvelocity: Vector3,\n"
            "\talive: boolean,\n"
            "}\n\n"
            "local activeProjectiles: { Projectile } = {}\n\n"
            "local rayParams = RaycastParams.new()\n"
            "rayParams.FilterType = Enum.RaycastFilterType.Exclude\n\n"
            "local function spawnProjectile(origin: Vector3, velocity: Vector3)\n"
            "\ttable.insert(activeProjectiles, {\n"
            "\t\tposition = origin,\n"
            "\t\tvelocity = velocity,\n"
            "\t\talive = true,\n"
            "\t})\n"
            "end\n\n"
            "@native\n"
            "local function updateProjectiles(dt: number)\n"
            "\tfor i = #activeProjectiles, 1, -1 do\n"
            "\t\tlocal proj = activeProjectiles[i]\n"
            "\t\tif not proj.alive then\n"
            "\t\t\ttable.remove(activeProjectiles, i)\n"
            "\t\t\tcontinue\n"
            "\t\tend\n\n"
            "\t\tlocal newVelocity = proj.velocity + GRAVITY * dt\n"
            "\t\tlocal displacement = newVelocity * dt\n"
            "\t\tlocal result = workspace:Raycast(proj.position, displacement, rayParams)\n\n"
            "\t\tif result then\n"
            "\t\t\tproj.alive = false\n"
            "\t\t\tprint(`Projectile hit {result.Instance.Name} at {result.Position}`)\n"
            "\t\telse\n"
            "\t\t\tproj.position += displacement\n"
            "\t\t\tproj.velocity = newVelocity\n"
            "\t\tend\n"
            "\tend\n"
            "end\n\n"
            "RunService.Heartbeat:Connect(updateProjectiles)\n",
            category="physics",
            difficulty=4,
        ),
    ]


# ---------------------------------------------------------------------------
# 11. Grapple/glide/vehicle physics (Vertigo-specific)
# ---------------------------------------------------------------------------


def _vertigo_physics_examples() -> list[dict]:
    return [
        _ex(
            "How does a grapple hook system work with physics constraints?",
            "<think>\n"
            "Grapple uses a RopeConstraint or custom spring math. The client requests grapple "
            "with a target position, the server validates range and line-of-sight, then "
            "attaches the constraint. Vertigo uses spring/damping pull math rather than "
            "CFrame teleports for smooth swing feel.\n"
            "</think>\n\n"
            "--!strict\n"
            "-- Server-side grapple validation and attachment\n"
            "local MAX_GRAPPLE_RANGE: number = 120\n\n"
            "local rayParams = RaycastParams.new()\n"
            "rayParams.FilterType = Enum.RaycastFilterType.Exclude\n\n"
            "local function validateGrapple(\n"
            "\tplayer: Player,\n"
            "\ttargetPos: Vector3\n"
            "): boolean\n"
            "\tlocal character = player.Character\n"
            "\tif not character then return false end\n"
            '\tlocal root = character:FindFirstChild("HumanoidRootPart") :: BasePart?\n'
            "\tif not root then return false end\n\n"
            "\t-- Range check\n"
            "\tlocal distance = (targetPos - root.Position).Magnitude\n"
            "\tif distance > MAX_GRAPPLE_RANGE then\n"
            "\t\twarn(`[Grapple] {player.Name} target too far: {distance:.1f}`)\n"
            "\t\treturn false\n"
            "\tend\n\n"
            "\t-- Line of sight check\n"
            "\trayParams.FilterDescendantsInstances = { character }\n"
            "\tlocal dir = targetPos - root.Position\n"
            "\tlocal result = workspace:Raycast(root.Position, dir, rayParams)\n"
            "\tif not result then return false end\n\n"
            "\t-- Must hit near the target\n"
            "\tlocal hitDist = (result.Position - targetPos).Magnitude\n"
            "\treturn hitDist < 5\n"
            "end\n",
            category="physics",
            difficulty=4,
        ),
        _ex(
            "How do I implement a glide system that lets the player float and steer?",
            "<think>\n"
            "Glide reduces gravity and adds drag, letting the player descend slowly while "
            "steering. We modify AssemblyLinearVelocity each frame, applying a drag factor "
            "to vertical speed and allowing horizontal steering input. Vertigo uses smoothed "
            "planar velocity for steering feel.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n\n'
            "local GLIDE_GRAVITY_SCALE: number = 0.15\n"
            "local GLIDE_DRAG: number = 0.98\n"
            "local GLIDE_STEER_SPEED: number = 25\n"
            "local FULL_GRAVITY: number = workspace.Gravity\n\n"
            "local isGliding: boolean = false\n\n"
            "@native\n"
            "local function updateGlide(root: BasePart, dt: number, steerInput: Vector3)\n"
            "\tif not isGliding then return end\n\n"
            "\tlocal vel = root.AssemblyLinearVelocity\n"
            "\t-- Reduce downward velocity (glide float)\n"
            "\tlocal verticalVel = math.max(vel.Y, -FULL_GRAVITY * GLIDE_GRAVITY_SCALE * dt)\n"
            "\t-- Apply horizontal drag\n"
            "\tlocal horizontalVel = Vector3.new(vel.X, 0, vel.Z) * GLIDE_DRAG\n"
            "\t-- Add steering\n"
            "\thorizontalVel += steerInput * GLIDE_STEER_SPEED * dt\n\n"
            "\troot.AssemblyLinearVelocity = Vector3.new(\n"
            "\t\thorizontalVel.X,\n"
            "\t\tverticalVel,\n"
            "\t\thorizontalVel.Z\n"
            "\t)\n"
            "end\n",
            category="physics",
            difficulty=4,
        ),
        _ex(
            "How do I make a dirt bike with suspension using VehicleSeat and constraints?",
            "<think>\n"
            "A physics-based vehicle uses SpringConstraints for suspension attached between "
            "the chassis and wheel attachments. VehicleSeat handles throttle/steer input. "
            "HingeConstraints on wheels provide rotation. The server spawns the vehicle; "
            "physics runs on the network owner (seated player's client).\n"
            "</think>\n\n"
            "--!strict\n\n"
            "type SuspensionConfig = {\n"
            "\trestLength: number,\n"
            "\tstiffness: number,\n"
            "\tdamping: number,\n"
            "}\n\n"
            "local DEFAULT_SUSPENSION: SuspensionConfig = table.freeze({\n"
            "\trestLength = 2.5,\n"
            "\tstiffness = 800,\n"
            "\tdamping = 40,\n"
            "})\n\n"
            "local function setupWheel(\n"
            "\tchassis: BasePart,\n"
            "\twheel: BasePart,\n"
            "\toffset: Vector3,\n"
            "\tconfig: SuspensionConfig\n"
            ")\n"
            '\tlocal chassisAttach = Instance.new("Attachment")\n'
            "\tchassisAttach.Position = offset\n"
            "\tchassisAttach.Parent = chassis\n\n"
            '\tlocal wheelAttach = Instance.new("Attachment")\n'
            "\twheelAttach.Parent = wheel\n\n"
            '\tlocal spring = Instance.new("SpringConstraint")\n'
            "\tspring.Attachment0 = chassisAttach\n"
            "\tspring.Attachment1 = wheelAttach\n"
            "\tspring.FreeLength = config.restLength\n"
            "\tspring.Stiffness = config.stiffness\n"
            "\tspring.Damping = config.damping\n"
            "\tspring.Parent = chassis\n\n"
            '\tlocal hinge = Instance.new("HingeConstraint")\n'
            "\thinge.Attachment0 = wheelAttach\n"
            "\thinge.Attachment1 = chassisAttach\n"
            "\thinge.ActuatorType = Enum.ActuatorType.Motor\n"
            "\thinge.Parent = chassis\n\n"
            "\treturn hinge\n"
            "end\n",
            category="physics",
            difficulty=5,
        ),
        _ex(
            "How do I implement wall running in a custom character controller?",
            "<think>\n"
            "Wall run: raycast sideways to detect walls, then apply upward + forward velocity "
            "along the wall normal while disabling gravity. Exit when the player jumps off, "
            "runs out of wall, or a timer expires. Server validates the wall contact.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local WALL_RUN_SPEED: number = 35\n"
            "local WALL_RUN_UPWARD: number = 8\n"
            "local WALL_RUN_DURATION: number = 1.5\n"
            "local WALL_DETECT_DIST: number = 3\n\n"
            "local rayParams = RaycastParams.new()\n"
            "rayParams.FilterType = Enum.RaycastFilterType.Exclude\n\n"
            "@native\n"
            "local function tryWallRun(\n"
            "\troot: BasePart,\n"
            "\tmoveDir: Vector3\n"
            "): (boolean, Vector3?)\n"
            "\trayParams.FilterDescendantsInstances = { root.Parent :: Instance }\n\n"
            "\t-- Check left and right for walls\n"
            "\tlocal rightDir = root.CFrame.RightVector\n"
            "\tfor _, dir in { rightDir, -rightDir } do\n"
            "\t\tlocal result = workspace:Raycast(\n"
            "\t\t\troot.Position,\n"
            "\t\t\tdir * WALL_DETECT_DIST,\n"
            "\t\t\trayParams\n"
            "\t\t)\n"
            "\t\tif result and result.Normal then\n"
            "\t\t\t-- Wall found, compute run direction (along wall)\n"
            "\t\t\tlocal wallNormal = result.Normal\n"
            "\t\t\tlocal wallForward = wallNormal:Cross(Vector3.yAxis).Unit\n"
            "\t\t\t-- Align with player's move direction\n"
            "\t\t\tif wallForward:Dot(moveDir) < 0 then\n"
            "\t\t\t\twallForward = -wallForward\n"
            "\t\t\tend\n"
            "\t\t\treturn true, wallForward\n"
            "\t\tend\n"
            "\tend\n"
            "\treturn false, nil\n"
            "end\n",
            category="physics",
            difficulty=5,
        ),
    ]


# ---------------------------------------------------------------------------
# 12. Anti-cheat / server validation
# ---------------------------------------------------------------------------


def _anticheat_examples() -> list[dict]:
    return [
        _ex(
            "How do I validate client ability requests on the server to prevent cheating?",
            "<think>\n"
            "Never trust client data. Validate: 1) ability exists, 2) player owns it, "
            "3) cooldown has elapsed, 4) player state allows it (alive, not stunned), "
            "5) range/position plausible. Log suspicious requests for review.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "type ValidationResult = {\n"
            "\tvalid: boolean,\n"
            "\treason: string?,\n"
            "}\n\n"
            "local VALID_ABILITIES: { [string]: boolean } = table.freeze({\n"
            "\tability_grapple_v1 = true,\n"
            "\tability_glide_v1 = true,\n"
            "\tability_airdash_v1 = true,\n"
            "\tability_wallrun_v1 = true,\n"
            "})\n\n"
            "local function validateAbilityRequest(\n"
            "\tplayer: Player,\n"
            "\tabilityId: string\n"
            "): ValidationResult\n"
            "\t-- 1. Valid ability?\n"
            "\tif not VALID_ABILITIES[abilityId] then\n"
            "\t\twarn(`[AntiCheat] {player.Name} requested invalid ability: {abilityId}`)\n"
            '\t\treturn { valid = false, reason = "invalid_ability" }\n'
            "\tend\n\n"
            "\t-- 2. Player alive?\n"
            "\tlocal character = player.Character\n"
            "\tif not character then\n"
            '\t\treturn { valid = false, reason = "no_character" }\n'
            "\tend\n"
            '\tlocal humanoid = character:FindFirstChildOfClass("Humanoid")\n'
            "\tif not humanoid or humanoid.Health <= 0 then\n"
            '\t\treturn { valid = false, reason = "dead" }\n'
            "\tend\n\n"
            "\t-- 3. Cooldown check (delegated to cooldown module)\n"
            "\t-- 4. State check (not stunned, not in vehicle, etc.)\n\n"
            "\treturn { valid = true }\n"
            "end\n",
            category="service",
            difficulty=3,
        ),
        _ex(
            "How do I detect speed hackers on the server?",
            "<think>\n"
            "Track player position each heartbeat. If they move faster than physically "
            "possible (max speed + margin), flag them. Account for legitimate teleports "
            "(respawn, vehicle exit) by resetting tracking on those events.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n'
            'local Players = game:GetService("Players")\n\n'
            "local MAX_SPEED: number = 80 -- studs/second (generous margin)\n"
            "local TELEPORT_THRESHOLD: number = 200 -- ignore obvious teleports\n"
            "local FLAG_THRESHOLD: number = 3 -- violations before action\n\n"
            "local lastPositions: { [number]: Vector3 } = {}\n"
            "local violations: { [number]: number } = {}\n\n"
            "@native\n"
            "local function checkPlayerSpeed(dt: number)\n"
            "\tfor _, player in Players:GetPlayers() do\n"
            "\t\tlocal character = player.Character\n"
            "\t\tif not character then continue end\n"
            '\t\tlocal root = character:FindFirstChild("HumanoidRootPart") :: BasePart?\n'
            "\t\tif not root then continue end\n\n"
            "\t\tlocal pos = root.Position\n"
            "\t\tlocal lastPos = lastPositions[player.UserId]\n"
            "\t\tlastPositions[player.UserId] = pos\n\n"
            "\t\tif not lastPos then continue end\n"
            "\t\tlocal dist = (pos - lastPos).Magnitude\n"
            "\t\tif dist > TELEPORT_THRESHOLD then continue end -- legitimate teleport\n\n"
            "\t\tlocal speed = dist / dt\n"
            "\t\tif speed > MAX_SPEED then\n"
            "\t\t\tviolations[player.UserId] = (violations[player.UserId] or 0) + 1\n"
            "\t\t\tif violations[player.UserId] >= FLAG_THRESHOLD then\n"
            "\t\t\t\twarn(`[AntiCheat] {player.Name} speed violation: {speed:.0f} studs/s`)\n"
            "\t\t\tend\n"
            "\t\tend\n"
            "\tend\n"
            "end\n\n"
            "RunService.Heartbeat:Connect(checkPlayerSpeed)\n\n"
            "Players.PlayerRemoving:Connect(function(player: Player)\n"
            "\tlastPositions[player.UserId] = nil\n"
            "\tviolations[player.UserId] = nil\n"
            "end)\n",
            category="service",
            difficulty=4,
        ),
    ]


# ---------------------------------------------------------------------------
# 13. UI creation
# ---------------------------------------------------------------------------


def _ui_examples() -> list[dict]:
    return [
        _ex(
            "How do I create a basic ScreenGui with a button that prints when clicked?",
            "<think>\n"
            "ScreenGui goes under PlayerGui (or StarterGui for auto-clone). Create a Frame, "
            "add a TextButton, connect Activated (not MouseButton1Click, which doesn't work "
            "on mobile). Use :Init() to build the UI tree.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "local player = Players.LocalPlayer\n"
            'local playerGui = player:WaitForChild("PlayerGui") :: PlayerGui\n\n'
            'local screenGui = Instance.new("ScreenGui")\n'
            'screenGui.Name = "MyUI"\n'
            "screenGui.ResetOnSpawn = false\n"
            "screenGui.Parent = playerGui\n\n"
            'local frame = Instance.new("Frame")\n'
            "frame.Size = UDim2.fromScale(0.3, 0.2)\n"
            "frame.Position = UDim2.fromScale(0.35, 0.4)\n"
            "frame.BackgroundColor3 = Color3.fromRGB(30, 30, 40)\n"
            "frame.Parent = screenGui\n\n"
            'local corner = Instance.new("UICorner")\n'
            "corner.CornerRadius = UDim.new(0, 8)\n"
            "corner.Parent = frame\n\n"
            'local button = Instance.new("TextButton")\n'
            "button.Size = UDim2.fromScale(0.6, 0.4)\n"
            "button.Position = UDim2.fromScale(0.2, 0.3)\n"
            'button.Text = "Click Me"\n'
            "button.TextColor3 = Color3.new(1, 1, 1)\n"
            "button.BackgroundColor3 = Color3.fromRGB(60, 120, 200)\n"
            "button.Font = Enum.Font.GothamBold\n"
            "button.TextSize = 18\n"
            "button.Parent = frame\n\n"
            "-- Use Activated for cross-platform (mouse + touch + gamepad)\n"
            "button.Activated:Connect(function()\n"
            '\tprint("Button clicked!")\n'
            "end)\n",
            category="controller",
            difficulty=1,
        ),
        _ex(
            "How do I create a health bar UI that updates from the Humanoid?",
            "<think>\n"
            "Create a background frame with a fill frame inside. Listen to "
            "Humanoid.HealthChanged to update the fill width. Use math.clamp to keep "
            "the scale between 0 and 1. Color lerp from green to red based on health.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n'
            "local player = Players.LocalPlayer\n\n"
            "local function setupHealthBar(character: Model)\n"
            '\tlocal humanoid = character:WaitForChild("Humanoid") :: Humanoid\n'
            '\tlocal playerGui = player:WaitForChild("PlayerGui") :: PlayerGui\n\n'
            "\t-- Find or create UI\n"
            '\tlocal existing = playerGui:FindFirstChild("HealthUI")\n'
            "\tif existing then existing:Destroy() end\n\n"
            '\tlocal gui = Instance.new("ScreenGui")\n'
            '\tgui.Name = "HealthUI"\n'
            "\tgui.Parent = playerGui\n\n"
            '\tlocal bg = Instance.new("Frame")\n'
            "\tbg.Size = UDim2.new(0.2, 0, 0, 20)\n"
            "\tbg.Position = UDim2.fromScale(0.4, 0.92)\n"
            "\tbg.BackgroundColor3 = Color3.fromRGB(40, 40, 40)\n"
            "\tbg.Parent = gui\n\n"
            '\tlocal fill = Instance.new("Frame")\n'
            "\tfill.Size = UDim2.fromScale(1, 1)\n"
            "\tfill.BackgroundColor3 = Color3.fromRGB(80, 200, 80)\n"
            "\tfill.Parent = bg\n\n"
            "\thumanoid.HealthChanged:Connect(function(health: number)\n"
            "\t\tlocal fraction = math.clamp(health / humanoid.MaxHealth, 0, 1)\n"
            "\t\tfill.Size = UDim2.fromScale(fraction, 1)\n"
            "\t\tfill.BackgroundColor3 = Color3.fromRGB(\n"
            "\t\t\tmath.floor((1 - fraction) * 220),\n"
            "\t\t\tmath.floor(fraction * 200),\n"
            "\t\t\t40\n"
            "\t\t)\n"
            "\tend)\n"
            "end\n\n"
            "player.CharacterAdded:Connect(setupHealthBar)\n"
            "if player.Character then setupHealthBar(player.Character) end\n",
            category="controller",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# 14. Animation/tween patterns
# ---------------------------------------------------------------------------


def _tween_examples() -> list[dict]:
    return [
        _ex(
            "How do I tween a part's position and color smoothly?",
            "<think>\n"
            "TweenService:Create takes the instance, TweenInfo for timing/easing, and a "
            "property table. It interpolates from current values to targets. Use Enum.EasingStyle "
            "for different curves. Tweens are lightweight and run on the engine thread.\n"
            "</think>\n\n"
            "--!strict\n"
            'local TweenService = game:GetService("TweenService")\n\n'
            "local part = workspace.MyPart :: BasePart\n\n"
            "local tweenInfo = TweenInfo.new(\n"
            "\t1.5,                          -- duration\n"
            "\tEnum.EasingStyle.Quad,         -- easing style\n"
            "\tEnum.EasingDirection.InOut,     -- easing direction\n"
            "\t0,                             -- repeat count (0 = once)\n"
            "\tfalse,                         -- reverses\n"
            "\t0                              -- delay\n"
            ")\n\n"
            "local tween = TweenService:Create(part, tweenInfo, {\n"
            "\tPosition = Vector3.new(0, 20, 0),\n"
            "\tColor = Color3.fromRGB(255, 100, 50),\n"
            "})\n\n"
            "tween:Play()\n"
            "tween.Completed:Once(function()\n"
            '\tprint("Tween finished")\n'
            "end)\n",
            category="general_luau",
            difficulty=1,
        ),
        _ex(
            "How do I chain multiple tweens to create a sequence?",
            "<think>\n"
            "Connect to Completed to trigger the next tween. For complex chains, a helper "
            "function keeps things clean. Alternatively, use task.wait() in a spawned thread "
            "to sequence with delays.\n"
            "</think>\n\n"
            "--!strict\n"
            'local TweenService = game:GetService("TweenService")\n\n'
            "local part = workspace.MyPart :: BasePart\n\n"
            "type TweenStep = {\n"
            "\tduration: number,\n"
            "\tproperties: { [string]: any },\n"
            "\teasing: Enum.EasingStyle?,\n"
            "}\n\n"
            "local function playSequence(target: Instance, steps: { TweenStep })\n"
            "\ttask.spawn(function()\n"
            "\t\tfor _, step in steps do\n"
            "\t\t\tlocal info = TweenInfo.new(\n"
            "\t\t\t\tstep.duration,\n"
            "\t\t\t\tstep.easing or Enum.EasingStyle.Quad,\n"
            "\t\t\t\tEnum.EasingDirection.InOut\n"
            "\t\t\t)\n"
            "\t\t\tlocal tween = TweenService:Create(target, info, step.properties)\n"
            "\t\t\ttween:Play()\n"
            "\t\t\ttween.Completed:Wait()\n"
            "\t\tend\n"
            "\tend)\n"
            "end\n\n"
            "playSequence(part, {\n"
            "\t{ duration = 1, properties = { Position = Vector3.new(0, 20, 0) } },\n"
            "\t{ duration = 0.5, properties = { Size = Vector3.new(4, 4, 4) }, easing = Enum.EasingStyle.Back },\n"
            "\t{ duration = 1, properties = { Transparency = 1 } },\n"
            "})\n",
            category="general_luau",
            difficulty=3,
        ),
        _ex(
            "How do I play character animations from a script?",
            "<think>\n"
            "Load an Animation on the Humanoid's Animator. AnimationTrack controls playback "
            "speed, looping, weight, priority. Always load in :Init() or cache the track "
            "to avoid reloading every frame.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "local player = Players.LocalPlayer\n\n"
            "local function playAnimation(character: Model, animId: string, speed: number?)\n"
            '\tlocal humanoid = character:FindFirstChildOfClass("Humanoid")\n'
            "\tif not humanoid then return nil end\n"
            '\tlocal animator = humanoid:FindFirstChildOfClass("Animator")\n'
            "\tif not animator then return nil end\n\n"
            '\tlocal anim = Instance.new("Animation")\n'
            "\tanim.AnimationId = animId\n\n"
            "\tlocal track = animator:LoadAnimation(anim)\n"
            "\ttrack:Play()\n"
            "\tif speed then\n"
            "\t\ttrack:AdjustSpeed(speed)\n"
            "\tend\n"
            "\treturn track\n"
            "end\n\n"
            "-- Usage:\n"
            '-- local track = playAnimation(character, "rbxassetid://12345678", 1.5)\n'
            '-- track.Stopped:Once(function() print("Done") end)\n',
            category="controller",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# 15. Sound management
# ---------------------------------------------------------------------------


def _sound_examples() -> list[dict]:
    return [
        _ex(
            "How do I play positional 3D sounds attached to parts?",
            "<think>\n"
            "Parent a Sound instance to a part for 3D positional audio. Set RollOffMinDistance "
            "and RollOffMaxDistance to control falloff. The engine automatically attenuates "
            "based on camera distance. Pre-load sounds in :Init() for instant playback.\n"
            "</think>\n\n"
            "--!strict\n"
            'local SoundService = game:GetService("SoundService")\n'
            'local ContentProvider = game:GetService("ContentProvider")\n\n'
            "local function createPositionalSound(\n"
            "\tparent: BasePart,\n"
            "\tassetId: string,\n"
            "\tvolume: number?,\n"
            "\tmaxDist: number?\n"
            "): Sound\n"
            '\tlocal sound = Instance.new("Sound")\n'
            "\tsound.SoundId = assetId\n"
            "\tsound.Volume = volume or 0.5\n"
            "\tsound.RollOffMinDistance = 10\n"
            "\tsound.RollOffMaxDistance = maxDist or 100\n"
            "\tsound.RollOffMode = Enum.RollOffMode.InverseTapered\n"
            "\tsound.Parent = parent\n"
            "\treturn sound\n"
            "end\n\n"
            "-- Pre-load for instant playback\n"
            "local function preloadSounds(sounds: { Sound })\n"
            "\tlocal assets: { any } = {}\n"
            "\tfor _, s in sounds do\n"
            "\t\ttable.insert(assets, s)\n"
            "\tend\n"
            "\tContentProvider:PreloadAsync(assets)\n"
            "end\n",
            category="controller",
            difficulty=2,
        ),
        _ex(
            "How do I crossfade between two ambient music tracks?",
            "<think>\n"
            "Tween the volume of the outgoing track to 0 while tweening the incoming track "
            "from 0 to target volume. Both tweens run simultaneously for smooth crossfade. "
            "Parent ambient sounds to SoundService (not workspace) for non-positional audio.\n"
            "</think>\n\n"
            "--!strict\n"
            'local TweenService = game:GetService("TweenService")\n'
            'local SoundService = game:GetService("SoundService")\n\n'
            "local currentTrack: Sound? = nil\n"
            "local FADE_DURATION: number = 2\n\n"
            "local function crossfadeTo(newTrack: Sound, targetVolume: number?)\n"
            "\tlocal vol = targetVolume or 0.5\n"
            "\tlocal fadeInfo = TweenInfo.new(FADE_DURATION, Enum.EasingStyle.Quad)\n\n"
            "\t-- Fade out current\n"
            "\tif currentTrack and currentTrack.IsPlaying then\n"
            "\t\tlocal fadeOut = TweenService:Create(currentTrack, fadeInfo, { Volume = 0 })\n"
            "\t\tfadeOut:Play()\n"
            "\t\tlocal old = currentTrack\n"
            "\t\tfadeOut.Completed:Once(function()\n"
            "\t\t\told:Stop()\n"
            "\t\tend)\n"
            "\tend\n\n"
            "\t-- Fade in new\n"
            "\tnewTrack.Volume = 0\n"
            "\tnewTrack:Play()\n"
            "\tlocal fadeIn = TweenService:Create(newTrack, fadeInfo, { Volume = vol })\n"
            "\tfadeIn:Play()\n\n"
            "\tcurrentTrack = newTrack\n"
            "end\n",
            category="controller",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# 16. Terrain manipulation
# ---------------------------------------------------------------------------


def _terrain_examples() -> list[dict]:
    return [
        _ex(
            "How do I create terrain programmatically using FillBlock and FillBall?",
            "<think>\n"
            "workspace.Terrain exposes fill methods for procedural terrain. FillBlock takes "
            "a CFrame, size, and material. FillBall takes a center, radius, and material. "
            "These are server-side operations. Use them in builders for procedural world gen.\n"
            "</think>\n\n"
            "--!strict\n"
            "local terrain = workspace.Terrain\n\n"
            "-- Fill a rectangular region\n"
            "terrain:FillBlock(\n"
            "\tCFrame.new(0, -10, 0),        -- center position\n"
            "\tVector3.new(200, 5, 200),      -- size\n"
            "\tEnum.Material.Grass            -- material\n"
            ")\n\n"
            "-- Fill a spherical region (cave, crater)\n"
            "terrain:FillBall(\n"
            "\tVector3.new(50, 0, 50),        -- center\n"
            "\t20,                            -- radius\n"
            "\tEnum.Material.Air              -- Air = remove terrain (makes a cave)\n"
            ")\n\n"
            "-- Create a hill\n"
            "terrain:FillBall(\n"
            "\tVector3.new(-30, 5, 0),\n"
            "\t15,\n"
            "\tEnum.Material.Rock\n"
            ")\n\n"
            "-- Water pool\n"
            "terrain:FillBlock(\n"
            "\tCFrame.new(0, -2, -50),\n"
            "\tVector3.new(40, 4, 40),\n"
            "\tEnum.Material.Water\n"
            ")\n",
            category="general_luau",
            difficulty=2,
        ),
        _ex(
            "How do I read terrain material at a specific position?",
            "<think>\n"
            "Terrain:ReadVoxels reads a Region3 of voxels. Each voxel is 4x4x4 studs. "
            "The returned materials and occupancies are 3D arrays. For a single point, "
            "create a tiny region and read the center voxel.\n"
            "</think>\n\n"
            "--!strict\n"
            "local terrain = workspace.Terrain\n\n"
            "local function getMaterialAt(position: Vector3): Enum.Material\n"
            "\t-- Align to voxel grid (4-stud resolution)\n"
            "\tlocal resolution = 4\n"
            "\tlocal halfRes = resolution / 2\n"
            "\tlocal region = Region3.new(\n"
            "\t\tposition - Vector3.new(halfRes, halfRes, halfRes),\n"
            "\t\tposition + Vector3.new(halfRes, halfRes, halfRes)\n"
            "\t):ExpandToGrid(resolution)\n\n"
            "\tlocal materials, occupancies = terrain:ReadVoxels(region, resolution)\n"
            "\tif #materials > 0 and #materials[1] > 0 and #materials[1][1] > 0 then\n"
            "\t\treturn materials[1][1][1]\n"
            "\tend\n"
            "\treturn Enum.Material.Air\n"
            "end\n",
            category="general_luau",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# 17. Module require patterns
# ---------------------------------------------------------------------------


def _module_examples() -> list[dict]:
    return [
        _ex(
            "What's the correct way to require shared modules in Vertigo?",
            "<think>\n"
            "Vertigo anchors all shared requires on ReplicatedStorage.Shared. This ensures "
            "both client and server can access the same modules. Use WaitForChild for the "
            "root, then direct indexing for children. Config modules return multi-export tables.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n\n'
            "-- Config modules return multiple named exports\n"
            "local AbilitiesModule = require(Shared.Config.Abilities)\n"
            "local GrappleTuning = AbilitiesModule.GrappleTuning\n"
            "local GlideTuning = AbilitiesModule.GlideTuning\n\n"
            "-- Types module (types only, returns nil)\n"
            "local Types = require(Shared.Util.Types)\n"
            "-- Use types via: Types.PlayerProfile, Types.AbilityState, etc.\n\n"
            "-- Net layer\n"
            "local Remotes = require(Shared.Net.Remotes)\n\n"
            "-- Package dependencies\n"
            'local Packages = ReplicatedStorage:WaitForChild("Packages")\n'
            "local Signal = require(Packages.Signal)\n"
            "local Trove = require(Packages.Trove)\n",
            category="general_luau",
            difficulty=1,
        ),
        _ex(
            "How do I structure a module with Init/Start lifecycle in Vertigo?",
            "<think>\n"
            "Every service/controller follows the two-phase boot pattern. Init sets up state "
            "with no side effects. Start connects listeners and begins work. This ensures "
            "all modules are initialized before any start communicating. The module returns "
            "itself as a table.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
            'local Packages = ReplicatedStorage:WaitForChild("Packages")\n\n'
            "local Signal = require(Packages.Signal)\n"
            "local Trove = require(Packages.Trove)\n\n"
            "local MyService = {}\n\n"
            "-- Private state\n"
            "local _trove = Trove.new()\n"
            "local _playerData: { [number]: { [string]: any } } = {}\n"
            "local _onDataChanged = Signal.new()\n\n"
            "-- Public signal\n"
            "MyService.DataChanged = _onDataChanged\n\n"
            "function MyService:Init()\n"
            "\t-- Phase 1: setup state, no connections\n"
            '\tprint("[MyService] Init")\n'
            "end\n\n"
            "function MyService:Start()\n"
            "\t-- Phase 2: connect listeners\n"
            '\tlocal Players = game:GetService("Players")\n'
            "\t_trove:Connect(Players.PlayerAdded, function(player: Player)\n"
            "\t\t_playerData[player.UserId] = {}\n"
            "\t\t_onDataChanged:Fire(player.UserId)\n"
            "\tend)\n"
            "\t_trove:Connect(Players.PlayerRemoving, function(player: Player)\n"
            "\t\t_playerData[player.UserId] = nil\n"
            "\tend)\n"
            '\tprint("[MyService] Start")\n'
            "end\n\n"
            "return MyService\n",
            category="service",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# 18. Promise/async patterns
# ---------------------------------------------------------------------------


def _async_examples() -> list[dict]:
    return [
        _ex(
            "How do I use Promise for async operations in Roblox Luau?",
            "<think>\n"
            "The Promise library (from Packages) provides structured async flow. Promise.new "
            "takes a resolver. :andThen chains success, :catch handles errors. This is cleaner "
            "than nested pcall/spawn patterns. Vertigo uses Promises for DataStore, HTTP, and "
            "any yielding operations.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local Packages = ReplicatedStorage:WaitForChild("Packages")\n'
            "local Promise = require(Packages.Promise)\n"
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            'local playerStore = DataStoreService:GetDataStore("PlayerData_v2")\n\n'
            "local function loadDataAsync(userId: number)\n"
            "\treturn Promise.new(function(resolve, reject)\n"
            '\t\tlocal key = "player_" .. tostring(userId)\n'
            "\t\tlocal ok, result = pcall(function()\n"
            "\t\t\treturn playerStore:GetAsync(key)\n"
            "\t\tend)\n"
            "\t\tif ok then\n"
            "\t\t\tresolve(result or {})\n"
            "\t\telse\n"
            "\t\t\treject(result)\n"
            "\t\tend\n"
            "\tend)\n"
            "end\n\n"
            "-- Usage:\n"
            "loadDataAsync(12345)\n"
            "\t:andThen(function(data)\n"
            '\t\tprint("Loaded:", data)\n'
            "\tend)\n"
            "\t:catch(function(err)\n"
            '\t\twarn("Failed:", err)\n'
            "\tend)\n",
            category="general_luau",
            difficulty=3,
        ),
        _ex(
            "How do I run multiple async operations in parallel and wait for all?",
            "<think>\n"
            "Promise.all takes an array of promises and resolves when all complete. If any "
            "rejects, the whole group rejects. For cases where you want partial results, "
            "use Promise.allSettled instead. This is ideal for loading multiple DataStores "
            "or assets simultaneously.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local Packages = ReplicatedStorage:WaitForChild("Packages")\n'
            "local Promise = require(Packages.Promise)\n\n"
            "local function fetchFromStore(storeName: string, key: string)\n"
            "\treturn Promise.new(function(resolve, reject)\n"
            '\t\tlocal DataStoreService = game:GetService("DataStoreService")\n'
            "\t\tlocal store = DataStoreService:GetDataStore(storeName)\n"
            "\t\tlocal ok, result = pcall(function()\n"
            "\t\t\treturn store:GetAsync(key)\n"
            "\t\tend)\n"
            "\t\tif ok then resolve(result) else reject(result) end\n"
            "\tend)\n"
            "end\n\n"
            "-- Load multiple stores in parallel\n"
            "local function loadAllPlayerData(userId: number)\n"
            "\tlocal key = tostring(userId)\n"
            "\treturn Promise.all({\n"
            '\t\tfetchFromStore("PlayerData_v2", "player_" .. key),\n'
            '\t\tfetchFromStore("Inventory_v1", "inv_" .. key),\n'
            '\t\tfetchFromStore("Settings_v1", "settings_" .. key),\n'
            "\t}):andThen(function(results)\n"
            "\t\treturn {\n"
            "\t\t\tprofile = results[1],\n"
            "\t\t\tinventory = results[2],\n"
            "\t\t\tsettings = results[3],\n"
            "\t\t}\n"
            "\tend)\n"
            "end\n",
            category="general_luau",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# 19. Memory leak prevention
# ---------------------------------------------------------------------------


def _memory_leak_examples() -> list[dict]:
    return [
        _ex(
            "How do I prevent memory leaks from event connections in Roblox?",
            "<think>\n"
            "Every :Connect() returns an RBXScriptConnection that holds a reference. If you "
            "never disconnect, the callback and everything it captures stays in memory forever. "
            "Use Trove to batch-manage connections, or disconnect manually. :Once() auto-"
            "disconnects after one fire.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local Packages = ReplicatedStorage:WaitForChild("Packages")\n'
            "local Trove = require(Packages.Trove)\n\n"
            "-- BAD: leaked connection\n"
            "-- part.Touched:Connect(function() ... end) -- never disconnected!\n\n"
            "-- GOOD: Trove manages cleanup\n"
            "local trove = Trove.new()\n\n"
            "trove:Connect(part.Touched, function(other: BasePart)\n"
            '\tprint("Touched:", other.Name)\n'
            "end)\n\n"
            "-- When done (e.g., character removed, controller stopped):\n"
            "trove:Destroy() -- disconnects all, destroys all tracked instances\n\n"
            "-- GOOD: :Once() for one-shot events\n"
            "part.Touched:Once(function(other: BasePart)\n"
            '\tprint("First touch only")\n'
            "\t-- automatically disconnects\n"
            "end)\n\n"
            "-- GOOD: manual disconnect\n"
            "local conn: RBXScriptConnection\n"
            "conn = part.Touched:Connect(function(other: BasePart)\n"
            '\tprint("Done")\n'
            "\tconn:Disconnect() -- clean up immediately\n"
            "end)\n",
            category="general_luau",
            difficulty=2,
        ),
        _ex(
            "How do I avoid memory leaks with player data tables?",
            "<think>\n"
            "If you store per-player data in a table keyed by UserId and forget to clean it "
            "up on PlayerRemoving, the table grows forever as players join and leave. Always "
            "nil out entries when the player leaves. This applies to every per-player cache.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "-- Every per-player table MUST be cleaned up\n"
            "local playerStates: { [number]: { [string]: any } } = {}\n"
            "local playerCooldowns: { [number]: { [string]: number } } = {}\n"
            "local playerConnections: { [number]: { RBXScriptConnection } } = {}\n\n"
            "Players.PlayerAdded:Connect(function(player: Player)\n"
            "\tplayerStates[player.UserId] = {}\n"
            "\tplayerCooldowns[player.UserId] = {}\n"
            "\tplayerConnections[player.UserId] = {}\n"
            "end)\n\n"
            "Players.PlayerRemoving:Connect(function(player: Player)\n"
            "\t-- Disconnect all per-player connections\n"
            "\tlocal conns = playerConnections[player.UserId]\n"
            "\tif conns then\n"
            "\t\tfor _, conn in conns do\n"
            "\t\t\tconn:Disconnect()\n"
            "\t\tend\n"
            "\tend\n\n"
            "\t-- Nil out all per-player tables\n"
            "\tplayerStates[player.UserId] = nil\n"
            "\tplayerCooldowns[player.UserId] = nil\n"
            "\tplayerConnections[player.UserId] = nil\n"
            "end)\n",
            category="general_luau",
            difficulty=2,
        ),
        _ex(
            "My game's memory keeps growing. How do I find the leak?",
            "<think>\n"
            "Check these common sources: 1) event connections never disconnected, 2) per-player "
            "tables not cleaned on PlayerRemoving, 3) Instance references kept after Destroy, "
            "4) closures capturing large tables. Use Stats.GetMemoryUsageMbForTag and the "
            "Developer Console (F9) to identify which category is growing.\n"
            "</think>\n\n"
            "--!strict\n"
            "-- Diagnostic: track memory categories\n"
            'local Stats = game:GetService("Stats")\n\n'
            "local function logMemoryUsage()\n"
            "\tlocal categories = {\n"
            '\t\t"CoreMemory",\n'
            '\t\t"PlaceMemory",\n'
            '\t\t"Instances",\n'
            '\t\t"Signals",\n'
            '\t\t"LuaHeap",\n'
            "\t}\n"
            '\tprint("=== Memory Report ===")\n'
            "\tfor _, cat in categories do\n"
            "\t\tlocal mb = Stats:GetMemoryUsageMbForTag(cat)\n"
            "\t\tprint(`  {cat}: {mb:.2f} MB`)\n"
            "\tend\n"
            "end\n\n"
            "-- Run periodically to spot growth\n"
            "task.spawn(function()\n"
            "\twhile true do\n"
            "\t\tlogMemoryUsage()\n"
            "\t\ttask.wait(30)\n"
            "\tend\n"
            "end)\n\n"
            "-- Common checklist:\n"
            "-- 1. Every :Connect() has a matching :Disconnect() or uses Trove\n"
            "-- 2. Every playerData[userId] is set to nil on PlayerRemoving\n"
            "-- 3. Destroyed instances are not referenced from module-level tables\n"
            "-- 4. No table.insert in loops without corresponding removal\n",
            category="debugging",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# 20. Mobile input handling
# ---------------------------------------------------------------------------


def _mobile_input_examples() -> list[dict]:
    return [
        _ex(
            "How do I detect touch input and swipe gestures on mobile?",
            "<think>\n"
            "UserInputService handles touch events. TouchStarted/TouchMoved/TouchEnded give "
            "InputObjects with Position. Track the start position, compare with end to "
            "determine swipe direction. Use magnitude threshold to distinguish taps from swipes.\n"
            "</think>\n\n"
            "--!strict\n"
            'local UserInputService = game:GetService("UserInputService")\n\n'
            "local SWIPE_THRESHOLD: number = 50 -- pixels\n"
            "local touchStart: Vector3? = nil\n\n"
            "UserInputService.TouchStarted:Connect(function(input: InputObject, processed: boolean)\n"
            "\tif processed then return end\n"
            "\ttouchStart = input.Position\n"
            "end)\n\n"
            "UserInputService.TouchEnded:Connect(function(input: InputObject, processed: boolean)\n"
            "\tif processed then return end\n"
            "\tif not touchStart then return end\n\n"
            "\tlocal delta = input.Position - touchStart\n"
            "\tlocal dist = delta.Magnitude\n"
            "\ttouchStart = nil\n\n"
            "\tif dist < SWIPE_THRESHOLD then\n"
            '\t\tprint("Tap detected")\n'
            "\t\treturn\n"
            "\tend\n\n"
            "\t-- Determine swipe direction\n"
            "\tif math.abs(delta.X) > math.abs(delta.Y) then\n"
            "\t\tif delta.X > 0 then\n"
            '\t\t\tprint("Swipe right")\n'
            "\t\telse\n"
            '\t\t\tprint("Swipe left")\n'
            "\t\tend\n"
            "\telse\n"
            "\t\tif delta.Y > 0 then\n"
            '\t\t\tprint("Swipe down")\n'
            "\t\telse\n"
            '\t\t\tprint("Swipe up")\n'
            "\t\tend\n"
            "\tend\n"
            "end)\n",
            category="controller",
            difficulty=2,
        ),
        _ex(
            "How do I create on-screen touch buttons for mobile ability controls?",
            "<think>\n"
            "Mobile players need virtual buttons for abilities since they lack a keyboard. "
            "Create ImageButtons positioned in thumb-reach zones. Use Activated (not "
            "MouseButton1Click) for cross-platform compatibility. Size buttons appropriately "
            "for touch targets (min 44x44 logical pixels).\n"
            "</think>\n\n"
            "--!strict\n"
            'local UserInputService = game:GetService("UserInputService")\n'
            'local Players = game:GetService("Players")\n\n'
            "local player = Players.LocalPlayer\n"
            'local playerGui = player:WaitForChild("PlayerGui") :: PlayerGui\n\n'
            "-- Only show on mobile\n"
            "if not UserInputService.TouchEnabled then return end\n\n"
            'local gui = Instance.new("ScreenGui")\n'
            'gui.Name = "MobileControls"\n'
            "gui.ResetOnSpawn = false\n"
            "gui.Parent = playerGui\n\n"
            "type ButtonConfig = {\n"
            "\tname: string,\n"
            "\tposition: UDim2,\n"
            "\ttext: string,\n"
            "\tcallback: () -> (),\n"
            "}\n\n"
            "local function createMobileButton(config: ButtonConfig): TextButton\n"
            '\tlocal btn = Instance.new("TextButton")\n'
            "\tbtn.Name = config.name\n"
            "\tbtn.Size = UDim2.fromOffset(70, 70)\n"
            "\tbtn.Position = config.position\n"
            "\tbtn.AnchorPoint = Vector2.new(0.5, 0.5)\n"
            "\tbtn.Text = config.text\n"
            "\tbtn.TextColor3 = Color3.new(1, 1, 1)\n"
            "\tbtn.BackgroundColor3 = Color3.fromRGB(40, 40, 60)\n"
            "\tbtn.BackgroundTransparency = 0.3\n"
            "\tbtn.Font = Enum.Font.GothamBold\n"
            "\tbtn.TextSize = 16\n"
            "\tbtn.Parent = gui\n\n"
            '\tlocal corner = Instance.new("UICorner")\n'
            "\tcorner.CornerRadius = UDim.new(0.5, 0)\n"
            "\tcorner.Parent = btn\n\n"
            "\tbtn.Activated:Connect(config.callback)\n"
            "\treturn btn\n"
            "end\n\n"
            "createMobileButton({\n"
            '\tname = "GrappleBtn",\n'
            "\tposition = UDim2.fromScale(0.85, 0.6),\n"
            '\ttext = "Grapple",\n'
            "\tcallback = function()\n"
            '\t\tprint("Grapple activated")\n'
            "\tend,\n"
            "})\n\n"
            "createMobileButton({\n"
            '\tname = "GlideBtn",\n'
            "\tposition = UDim2.fromScale(0.85, 0.75),\n"
            '\ttext = "Glide",\n'
            "\tcallback = function()\n"
            '\t\tprint("Glide activated")\n'
            "\tend,\n"
            "})\n",
            category="controller",
            difficulty=2,
        ),
        _ex(
            "How do I detect which platform the player is on (PC, mobile, console)?",
            "<think>\n"
            "Check UserInputService properties: TouchEnabled for mobile, GamepadEnabled for "
            "console, KeyboardEnabled for PC. A device can have multiple (laptop with touch). "
            "Prioritize the primary input method. GuiService:IsTenFootInterface() detects "
            "console/TV mode.\n"
            "</think>\n\n"
            "--!strict\n"
            'local UserInputService = game:GetService("UserInputService")\n'
            'local GuiService = game:GetService("GuiService")\n\n'
            'export type Platform = "mobile" | "console" | "pc"\n\n'
            "local function detectPlatform(): Platform\n"
            "\tif GuiService:IsTenFootInterface() then\n"
            '\t\treturn "console"\n'
            "\telseif UserInputService.TouchEnabled and not UserInputService.KeyboardEnabled then\n"
            '\t\treturn "mobile"\n'
            "\telse\n"
            '\t\treturn "pc"\n'
            "\tend\n"
            "end\n\n"
            "local platform = detectPlatform()\n"
            "print(`[Input] Platform detected: {platform}`)\n\n"
            "-- Adjust UI based on platform\n"
            'if platform == "mobile" then\n'
            "\t-- Show touch controls, larger buttons\n"
            '\tprint("[Input] Enabling mobile controls")\n'
            'elseif platform == "console" then\n'
            "\t-- Show gamepad prompts, snap navigation\n"
            '\tprint("[Input] Enabling console controls")\n'
            "else\n"
            "\t-- Default keyboard/mouse\n"
            '\tprint("[Input] Using PC controls")\n'
            "end\n",
            category="controller",
            difficulty=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: DataStore edge cases
# ---------------------------------------------------------------------------


def _datastore_extra_examples() -> list[dict]:
    return [
        _ex(
            "How do I handle the DataStore budget and avoid throttling?",
            "<think>\n"
            "DataStoreService:GetRequestBudgetForRequestType returns remaining budget per "
            "request type. Check before making calls to avoid throttle errors. Budget "
            "replenishes over time. Batch operations and space out requests.\n"
            "</think>\n\n"
            "--!strict\n"
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            "local function hasBudget(requestType: Enum.DataStoreRequestType): boolean\n"
            "\tlocal budget = DataStoreService:GetRequestBudgetForRequestType(requestType)\n"
            "\treturn budget > 0\n"
            "end\n\n"
            "local function waitForBudget(requestType: Enum.DataStoreRequestType)\n"
            "\twhile not hasBudget(requestType) do\n"
            "\t\ttask.wait(1)\n"
            "\tend\n"
            "end\n\n"
            "-- Usage before any DataStore call:\n"
            "-- waitForBudget(Enum.DataStoreRequestType.GetAsync)\n"
            "-- store:GetAsync(key)\n",
            category="service",
            difficulty=3,
        ),
        _ex(
            "How do I auto-save player data periodically without flooding DataStore?",
            "<think>\n"
            "Run a loop that saves each player's dirty data at intervals. Stagger saves so "
            "they don't all fire at once. Track which profiles are dirty to skip unchanged ones. "
            "This avoids data loss if the server crashes between PlayerRemoving saves.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n'
            'local DataStoreService = game:GetService("DataStoreService")\n\n'
            'local playerStore = DataStoreService:GetDataStore("PlayerData_v2")\n'
            "local sessionData: { [number]: { [string]: any } } = {}\n"
            "local dirtyFlags: { [number]: boolean } = {}\n\n"
            "local SAVE_INTERVAL: number = 120 -- seconds\n"
            "local STAGGER_DELAY: number = 2 -- seconds between each player save\n\n"
            "local function markDirty(userId: number)\n"
            "\tdirtyFlags[userId] = true\n"
            "end\n\n"
            "task.spawn(function()\n"
            "\twhile true do\n"
            "\t\ttask.wait(SAVE_INTERVAL)\n"
            "\t\tfor _, player in Players:GetPlayers() do\n"
            "\t\t\tlocal userId = player.UserId\n"
            "\t\t\tif dirtyFlags[userId] and sessionData[userId] then\n"
            '\t\t\t\tlocal key = "player_" .. tostring(userId)\n'
            "\t\t\t\tlocal ok, err = pcall(function()\n"
            "\t\t\t\t\tplayerStore:SetAsync(key, sessionData[userId])\n"
            "\t\t\t\tend)\n"
            "\t\t\t\tif ok then\n"
            "\t\t\t\t\tdirtyFlags[userId] = nil\n"
            "\t\t\t\telse\n"
            "\t\t\t\t\twarn(`[AutoSave] Failed for {player.Name}: {err}`)\n"
            "\t\t\t\tend\n"
            "\t\t\t\ttask.wait(STAGGER_DELAY)\n"
            "\t\t\tend\n"
            "\t\tend\n"
            "\tend\n"
            "end)\n",
            category="service",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: More remote/networking patterns
# ---------------------------------------------------------------------------


def _networking_extra_examples() -> list[dict]:
    return [
        _ex(
            "How do I send different types of data through a single RemoteEvent?",
            "<think>\n"
            "Use a discriminated union pattern: send a table with a 'type' field. The server "
            "switches on the type to handle each payload shape. This reduces the number of "
            "RemoteEvent instances needed. Vertigo uses this with StateSync.\n"
            "</think>\n\n"
            "--!strict\n"
            "-- SERVER handler\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local remotes = ReplicatedStorage:WaitForChild("Remotes")\n\n'
            "type ActionPayload = {\n"
            "\ttype: string,\n"
            "\t[string]: any,\n"
            "}\n\n"
            "remotes.RequestUseAbility.OnServerEvent:Connect(function(player: Player, payload: ActionPayload)\n"
            '\tif payload.type == "grapple" then\n'
            "\t\tlocal target = payload.targetPos :: Vector3\n"
            "\t\tprint(`{player.Name} grapple to {target}`)\n"
            '\telseif payload.type == "glide" then\n'
            "\t\tprint(`{player.Name} started gliding`)\n"
            '\telseif payload.type == "airdash" then\n'
            "\t\tlocal dir = payload.direction :: Vector3\n"
            "\t\tprint(`{player.Name} air dash toward {dir}`)\n"
            "\telse\n"
            "\t\twarn(`[Net] Unknown action type: {payload.type}`)\n"
            "\tend\n"
            "end)\n",
            category="networking",
            difficulty=2,
        ),
        _ex(
            "How do I use UnreliableRemoteEvent for high-frequency updates?",
            "<think>\n"
            "UnreliableRemoteEvent skips the reliable ordering layer, making it faster for "
            "data that's OK to drop (position updates, VFX triggers). It has the same API "
            "as RemoteEvent but packets may arrive out of order or not at all. Use for "
            "cosmetic/visual data, never for gameplay-critical state.\n"
            "</think>\n\n"
            "--!strict\n"
            "-- UnreliableRemoteEvent for position interpolation\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local RunService = game:GetService("RunService")\n\n'
            'local unreliableRemote = ReplicatedStorage:WaitForChild("PositionSync") :: UnreliableRemoteEvent\n\n'
            "-- SERVER: broadcast positions at 20 Hz\n"
            "local SEND_RATE: number = 1 / 20\n"
            "local acc: number = 0\n\n"
            "@native\n"
            "local function onHeartbeat(dt: number)\n"
            "\tacc += dt\n"
            "\tif acc < SEND_RATE then return end\n"
            "\tacc = 0\n\n"
            '\tfor _, player in game:GetService("Players"):GetPlayers() do\n'
            "\t\tlocal character = player.Character\n"
            "\t\tif not character then continue end\n"
            '\t\tlocal root = character:FindFirstChild("HumanoidRootPart") :: BasePart?\n'
            "\t\tif not root then continue end\n"
            "\t\tunreliableRemote:FireAllClients(player.UserId, root.CFrame)\n"
            "\tend\n"
            "end\n\n"
            "RunService.Heartbeat:Connect(onHeartbeat)\n",
            category="networking",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: More camera/rendering
# ---------------------------------------------------------------------------


def _camera_extra_examples() -> list[dict]:
    return [
        _ex(
            "How do I make a camera shake effect for explosions?",
            "<think>\n"
            "Offset the camera CFrame by random amounts that decay over time. Use math.noise "
            "or random offsets applied each frame. Multiply by an intensity that decreases "
            "to zero. This is a client-only visual effect.\n"
            "</think>\n\n"
            "--!strict\n"
            'local RunService = game:GetService("RunService")\n\n'
            "local shakeIntensity: number = 0\n"
            "local shakeDuration: number = 0\n"
            "local shakeElapsed: number = 0\n"
            "local DECAY_RATE: number = 5\n\n"
            "local function triggerShake(intensity: number, duration: number)\n"
            "\tshakeIntensity = intensity\n"
            "\tshakeDuration = duration\n"
            "\tshakeElapsed = 0\n"
            "end\n\n"
            "@native\n"
            "local function applyShake(dt: number)\n"
            "\tif shakeIntensity <= 0.01 then return end\n\n"
            "\tshakeElapsed += dt\n"
            "\tlocal t = math.clamp(shakeElapsed / shakeDuration, 0, 1)\n"
            "\tlocal fade = 1 - t\n"
            "\tlocal magnitude = shakeIntensity * fade\n\n"
            "\tlocal offsetX = (math.random() - 0.5) * 2 * magnitude\n"
            "\tlocal offsetY = (math.random() - 0.5) * 2 * magnitude\n\n"
            "\tlocal cam = workspace.CurrentCamera\n"
            "\tcam.CFrame *= CFrame.new(offsetX, offsetY, 0)\n\n"
            "\tif t >= 1 then\n"
            "\t\tshakeIntensity = 0\n"
            "\tend\n"
            "end\n\n"
            "RunService.RenderStepped:Connect(applyShake)\n\n"
            "-- Usage: triggerShake(2, 0.5) -- 2 stud magnitude, 0.5s duration\n",
            category="controller",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: More physics patterns
# ---------------------------------------------------------------------------


def _physics_extra_examples() -> list[dict]:
    return [
        _ex(
            "How do I make a swing rope that the player can hang from?",
            "<think>\n"
            "Use a RopeConstraint between the anchor point and the player's root part. Set "
            "Length to the desired rope length. The physics engine handles pendulum motion "
            "naturally. Attach via Attachments for proper physics behavior.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local function attachRope(\n"
            "\tanchorPart: BasePart,\n"
            "\troot: BasePart,\n"
            "\tropeLength: number\n"
            "): RopeConstraint\n"
            '\tlocal anchorAttach = Instance.new("Attachment")\n'
            "\tanchorAttach.Parent = anchorPart\n\n"
            '\tlocal playerAttach = Instance.new("Attachment")\n'
            "\tplayerAttach.Parent = root\n\n"
            '\tlocal rope = Instance.new("RopeConstraint")\n'
            "\trope.Attachment0 = anchorAttach\n"
            "\trope.Attachment1 = playerAttach\n"
            "\trope.Length = ropeLength\n"
            "\trope.Restitution = 0.1\n"
            "\trope.Visible = true\n"
            "\trope.Thickness = 0.2\n"
            "\trope.Parent = anchorPart\n\n"
            "\treturn rope\n"
            "end\n\n"
            "local function detachRope(rope: RopeConstraint)\n"
            "\tlocal a0 = rope.Attachment0\n"
            "\tlocal a1 = rope.Attachment1\n"
            "\trope:Destroy()\n"
            "\tif a0 then a0:Destroy() end\n"
            "\tif a1 then a1:Destroy() end\n"
            "end\n",
            category="physics",
            difficulty=3,
        ),
        _ex(
            "How do I apply a launch/bounce force to a player?",
            "<think>\n"
            "Set AssemblyLinearVelocity directly for an instant launch. For a bounce, reflect "
            "the current velocity against the surface normal. Server-side for authoritative "
            "jump pads, or client-side for predicted local feel.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local LAUNCH_FORCE: number = 80\n\n"
            "local function launchPlayer(root: BasePart, direction: Vector3)\n"
            "\troot.AssemblyLinearVelocity = direction.Unit * LAUNCH_FORCE\n"
            "end\n\n"
            "local function bouncePlayer(root: BasePart, surfaceNormal: Vector3, bounciness: number)\n"
            "\tlocal vel = root.AssemblyLinearVelocity\n"
            "\tlocal reflected = vel - 2 * vel:Dot(surfaceNormal) * surfaceNormal\n"
            "\troot.AssemblyLinearVelocity = reflected * bounciness\n"
            "end\n\n"
            "-- Jump pad example (server script on the pad part):\n"
            "local pad = script.Parent :: BasePart\n"
            "pad.Touched:Connect(function(other: BasePart)\n"
            "\tlocal character = other.Parent\n"
            "\tif not character then return end\n"
            '\tlocal humanoid = character:FindFirstChildOfClass("Humanoid")\n'
            "\tif not humanoid then return end\n"
            '\tlocal root = character:FindFirstChild("HumanoidRootPart") :: BasePart?\n'
            "\tif root then\n"
            "\t\tlaunchPlayer(root, Vector3.new(0, 1, 0))\n"
            "\tend\n"
            "end)\n",
            category="physics",
            difficulty=2,
        ),
        _ex(
            "How do I create a sliding mechanic on slopes?",
            "<think>\n"
            "Detect the surface angle using the ground raycast normal. If the slope exceeds "
            "a threshold, apply a downhill force based on the projected gravity. The slide "
            "direction is gravity projected onto the surface plane.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local SLIDE_ANGLE_MIN: number = math.rad(30) -- start sliding above 30 degrees\n"
            "local SLIDE_ACCEL: number = 40\n"
            "local SLIDE_FRICTION: number = 0.95\n\n"
            "local rayParams = RaycastParams.new()\n"
            "rayParams.FilterType = Enum.RaycastFilterType.Exclude\n\n"
            "@native\n"
            "local function updateSlide(root: BasePart, dt: number): boolean\n"
            "\trayParams.FilterDescendantsInstances = { root.Parent :: Instance }\n"
            "\tlocal result = workspace:Raycast(\n"
            "\t\troot.Position,\n"
            "\t\tVector3.new(0, -4, 0),\n"
            "\t\trayParams\n"
            "\t)\n"
            "\tif not result then return false end\n\n"
            "\tlocal normal = result.Normal\n"
            "\tlocal slopeAngle = math.acos(normal:Dot(Vector3.yAxis))\n"
            "\tif slopeAngle < SLIDE_ANGLE_MIN then return false end\n\n"
            "\t-- Project gravity onto slope surface\n"
            "\tlocal gravity = Vector3.new(0, -workspace.Gravity, 0)\n"
            "\tlocal slideDir = gravity - gravity:Dot(normal) * normal\n"
            "\tif slideDir.Magnitude > 0 then\n"
            "\t\tslideDir = slideDir.Unit\n"
            "\tend\n\n"
            "\tlocal vel = root.AssemblyLinearVelocity\n"
            "\tlocal slideVel = vel + slideDir * SLIDE_ACCEL * dt\n"
            "\troot.AssemblyLinearVelocity = Vector3.new(\n"
            "\t\tslideVel.X * SLIDE_FRICTION,\n"
            "\t\tslideVel.Y,\n"
            "\t\tslideVel.Z * SLIDE_FRICTION\n"
            "\t)\n"
            "\treturn true\n"
            "end\n",
            category="physics",
            difficulty=4,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: CollectionService patterns
# ---------------------------------------------------------------------------


def _collection_service_examples() -> list[dict]:
    return [
        _ex(
            "How do I use CollectionService tags to manage interactive objects?",
            "<think>\n"
            "CollectionService tags let you label instances for batch processing. "
            "GetTagged returns all current instances, GetInstanceAddedSignal fires for new "
            "ones. This is how Vertigo discovers GrappleAnchors, PressurePlates, etc. "
            "Tags persist across Rojo sync and Studio edits.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n\n'
            'local TAG = "GrappleAnchor"\n\n'
            "local function setupAnchor(instance: Instance)\n"
            "\tlocal part = instance :: BasePart\n"
            "\tpart.Color = Color3.fromRGB(100, 200, 255)\n"
            "\tpart.Material = Enum.Material.Neon\n"
            "\tprint(`[GrappleSystem] Registered anchor: {part.Name}`)\n"
            "end\n\n"
            "local function teardownAnchor(instance: Instance)\n"
            "\tprint(`[GrappleSystem] Removed anchor: {instance.Name}`)\n"
            "end\n\n"
            "-- Process existing\n"
            "for _, anchor in CollectionService:GetTagged(TAG) do\n"
            "\tsetupAnchor(anchor)\n"
            "end\n\n"
            "-- Watch for new/removed\n"
            "CollectionService:GetInstanceAddedSignal(TAG):Connect(setupAnchor)\n"
            "CollectionService:GetInstanceRemovedSignal(TAG):Connect(teardownAnchor)\n",
            category="service",
            difficulty=2,
        ),
        _ex(
            "How do I add and remove tags at runtime?",
            "<think>\n"
            "CollectionService:AddTag and :RemoveTag modify tags at runtime. Tags replicate "
            "from server to client. Use HasTag to check. This is useful for state machines — "
            "tag a part as 'Active' or 'Disabled' and let systems react.\n"
            "</think>\n\n"
            "--!strict\n"
            'local CollectionService = game:GetService("CollectionService")\n\n'
            "local function activatePressurePlate(plate: BasePart)\n"
            '\tif CollectionService:HasTag(plate, "Active") then return end\n'
            '\tCollectionService:AddTag(plate, "Active")\n'
            '\tCollectionService:RemoveTag(plate, "Inactive")\n'
            "\tplate.Color = Color3.fromRGB(0, 255, 100)\n"
            "\tprint(`[Puzzle] Plate {plate.Name} activated`)\n"
            "end\n\n"
            "local function deactivatePressurePlate(plate: BasePart)\n"
            '\tCollectionService:RemoveTag(plate, "Active")\n'
            '\tCollectionService:AddTag(plate, "Inactive")\n'
            "\tplate.Color = Color3.fromRGB(100, 100, 100)\n"
            "\tprint(`[Puzzle] Plate {plate.Name} deactivated`)\n"
            "end\n",
            category="service",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: Attribute patterns
# ---------------------------------------------------------------------------


def _attribute_examples() -> list[dict]:
    return [
        _ex(
            "How do I iterate all attributes on an instance?",
            "<think>\n"
            "GetAttributes returns a dictionary of all attribute name-value pairs on an "
            "instance. Useful for serialization, debugging, or dynamic config reading.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local function dumpAttributes(instance: Instance)\n"
            "\tlocal attrs = instance:GetAttributes()\n"
            "\tprint(`Attributes on {instance:GetFullName()}:`)\n"
            "\tfor name, value in attrs do\n"
            "\t\tprint(`  {name} ({typeof(value)}) = {tostring(value)}`)\n"
            "\tend\n"
            "end\n\n"
            "-- Copy all attributes from one instance to another\n"
            "local function copyAttributes(source: Instance, target: Instance)\n"
            "\tfor name, value in source:GetAttributes() do\n"
            "\t\ttarget:SetAttribute(name, value)\n"
            "\tend\n"
            "end\n",
            category="general_luau",
            difficulty=1,
        ),
        _ex(
            "How do I use Instance Attributes for per-object configuration?",
            "<think>\n"
            "Attributes are key-value pairs on any Instance. They replicate server -> client "
            "automatically and support primitives (number, string, bool, Vector3, Color3, etc.). "
            "Use them instead of ValueObjects for config. GetAttributeChangedSignal watches "
            "for updates.\n"
            "</think>\n\n"
            "--!strict\n"
            "-- SERVER: configure a zone trigger\n"
            "local trigger = workspace.ZoneTrigger :: BasePart\n"
            'trigger:SetAttribute("ZoneId", "zone_canopy_v1")\n'
            'trigger:SetAttribute("DangerLevel", 3)\n'
            'trigger:SetAttribute("AmbientColor", Color3.fromRGB(120, 200, 140))\n'
            'trigger:SetAttribute("IsActive", true)\n\n'
            "-- CLIENT: read and react\n"
            'local zoneId = trigger:GetAttribute("ZoneId") :: string\n'
            'local danger = trigger:GetAttribute("DangerLevel") :: number\n'
            "print(`Entered zone {zoneId}, danger level {danger}`)\n\n"
            "-- Watch for changes\n"
            'trigger:GetAttributeChangedSignal("IsActive"):Connect(function()\n'
            '\tlocal active = trigger:GetAttribute("IsActive") :: boolean\n'
            "\tprint(`Zone active state: {tostring(active)}`)\n"
            "end)\n",
            category="general_luau",
            difficulty=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: Table/string patterns
# ---------------------------------------------------------------------------


def _luau_patterns_examples() -> list[dict]:
    return [
        _ex(
            "What are the best practices for table usage in Luau?",
            "<think>\n"
            "Key patterns: table.create(n) for pre-sized arrays, table.freeze for constants, "
            "table.clone for shallow copies, generalized for-in iteration (no ipairs/pairs "
            "needed in modern Luau). Avoid mixed tables (array + dict) — pick one.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- Pre-sized array (avoids rehashing)\n"
            "local positions: { Vector3 } = table.create(100)\n\n"
            "-- Frozen constants (immutable at runtime)\n"
            "local COLORS = table.freeze({\n"
            "\tRed = Color3.fromRGB(255, 0, 0),\n"
            "\tGreen = Color3.fromRGB(0, 255, 0),\n"
            "\tBlue = Color3.fromRGB(0, 0, 255),\n"
            "})\n\n"
            "-- Shallow clone\n"
            "local original = { 1, 2, 3 }\n"
            "local copy = table.clone(original)\n"
            "copy[1] = 99 -- doesn't affect original\n\n"
            "-- Modern iteration (no ipairs/pairs needed)\n"
            'local items = { "sword", "shield", "potion" }\n'
            "for i, item in items do\n"
            "\tprint(`{i}: {item}`)\n"
            "end\n\n"
            'local map = { name = "Player", level = 5 }\n'
            "for key, value in map do\n"
            "\tprint(`{key} = {value}`)\n"
            "end\n\n"
            "-- table.find for linear search\n"
            'local index = table.find(items, "shield") -- returns 2\n',
            category="general_luau",
            difficulty=1,
        ),
        _ex(
            "How do I use string interpolation and string.format in Luau?",
            "<think>\n"
            "Luau supports backtick string interpolation with {expression} syntax. It's "
            "cleaner than string.format for most cases. Use string.format when you need "
            "specific formatting like decimal places or padding.\n"
            "</think>\n\n"
            "--!strict\n"
            'local name = "Player1"\n'
            "local score = 1234\n"
            "local ratio = 0.756\n\n"
            "-- Backtick interpolation (preferred for readability)\n"
            "print(`{name} scored {score} points`)\n"
            "print(`Ratio: {ratio * 100:.1f}%`)\n\n"
            "-- string.format for precise formatting\n"
            'local padded = string.format("%05d", score) -- "01234"\n'
            'local fixed = string.format("%.2f", ratio) -- "0.76"\n'
            'local hex = string.format("0x%X", 255) -- "0xFF"\n\n'
            "-- string.split instead of gmatch (faster, no regex overhead)\n"
            'local csv = "apple,banana,cherry"\n'
            'local fruits = string.split(csv, ",")\n'
            "for _, fruit in fruits do\n"
            "\tprint(fruit)\n"
            "end\n",
            category="general_luau",
            difficulty=1,
        ),
        _ex(
            "How do I use type annotations and export types in Luau?",
            "<think>\n"
            "Luau's type system uses : for annotations and export type for sharing types "
            "across modules. In --!strict mode, all function params and returns should be "
            "typed. Vertigo keeps shared types in Util/Types.luau.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- Basic annotations\n"
            "local count: number = 0\n"
            'local name: string = "vertigo"\n'
            "local active: boolean = true\n\n"
            "-- Function types\n"
            "local function add(a: number, b: number): number\n"
            "\treturn a + b\n"
            "end\n\n"
            "-- Table types\n"
            "type PlayerState = {\n"
            "\tposition: Vector3,\n"
            "\thealth: number,\n"
            "\tinventory: { string },\n"
            "\tmetadata: { [string]: any }?,\n"
            "}\n\n"
            "-- Export for other modules\n"
            'export type AbilityId = "grapple" | "glide" | "airdash" | "wallrun"\n\n'
            "export type AbilityState = {\n"
            "\tid: AbilityId,\n"
            "\tactive: boolean,\n"
            "\tcooldownEnd: number,\n"
            "}\n\n"
            "-- Generic function\n"
            "local function findFirst<T>(list: { T }, predicate: (T) -> boolean): T?\n"
            "\tfor _, item in list do\n"
            "\t\tif predicate(item) then\n"
            "\t\t\treturn item\n"
            "\t\tend\n"
            "\tend\n"
            "\treturn nil\n"
            "end\n",
            category="general_luau",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: More anti-cheat
# ---------------------------------------------------------------------------


def _anticheat_extra_examples() -> list[dict]:
    return [
        _ex(
            "How do I validate that a client-sent position is reasonable?",
            "<think>\n"
            "Compare the client-claimed position with the server's last known position. "
            "If the delta exceeds what's physically possible given the time elapsed and "
            "max speed, reject the claim. Allow some margin for latency.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local MAX_MOVE_SPEED: number = 60 -- studs/second\n"
            "local LATENCY_MARGIN: number = 1.5 -- multiplier for network jitter\n"
            "local MAX_VERTICAL_SPEED: number = 120 -- higher for grapple launches\n\n"
            "local lastKnownPositions: { [number]: { pos: Vector3, time: number } } = {}\n\n"
            "local function validatePosition(\n"
            "\tuserId: number,\n"
            "\tclaimedPos: Vector3\n"
            "): boolean\n"
            "\tlocal last = lastKnownPositions[userId]\n"
            "\tif not last then\n"
            "\t\tlastKnownPositions[userId] = { pos = claimedPos, time = tick() }\n"
            "\t\treturn true\n"
            "\tend\n\n"
            "\tlocal dt = tick() - last.time\n"
            "\tlocal maxDist = MAX_MOVE_SPEED * dt * LATENCY_MARGIN\n"
            "\tlocal actualDist = (claimedPos - last.pos).Magnitude\n\n"
            "\tif actualDist > maxDist then\n"
            "\t\twarn(`[Validate] Position jump: {actualDist:.0f} studs in {dt:.2f}s`)\n"
            "\t\treturn false\n"
            "\tend\n\n"
            "\tlastKnownPositions[userId] = { pos = claimedPos, time = tick() }\n"
            "\treturn true\n"
            "end\n",
            category="service",
            difficulty=4,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: More UI patterns
# ---------------------------------------------------------------------------


def _ui_extra_examples() -> list[dict]:
    return [
        _ex(
            "How do I make a draggable UI element?",
            "<think>\n"
            "Track InputBegan on the frame, then follow InputChanged (mouse/touch movement) "
            "until InputEnded. Calculate the delta from the initial click position relative "
            "to the frame and apply it to the frame's Position.\n"
            "</think>\n\n"
            "--!strict\n"
            'local UserInputService = game:GetService("UserInputService")\n\n'
            "local function makeDraggable(frame: GuiObject)\n"
            "\tlocal dragging = false\n"
            "\tlocal dragStart: Vector3? = nil\n"
            "\tlocal startPos: UDim2? = nil\n\n"
            "\tframe.InputBegan:Connect(function(input: InputObject)\n"
            "\t\tif input.UserInputType == Enum.UserInputType.MouseButton1\n"
            "\t\t\tor input.UserInputType == Enum.UserInputType.Touch then\n"
            "\t\t\tdragging = true\n"
            "\t\t\tdragStart = input.Position\n"
            "\t\t\tstartPos = frame.Position\n"
            "\t\tend\n"
            "\tend)\n\n"
            "\tUserInputService.InputChanged:Connect(function(input: InputObject)\n"
            "\t\tif not dragging or not dragStart or not startPos then return end\n"
            "\t\tif input.UserInputType == Enum.UserInputType.MouseMovement\n"
            "\t\t\tor input.UserInputType == Enum.UserInputType.Touch then\n"
            "\t\t\tlocal delta = input.Position - dragStart\n"
            "\t\t\tframe.Position = UDim2.new(\n"
            "\t\t\t\tstartPos.X.Scale, startPos.X.Offset + delta.X,\n"
            "\t\t\t\tstartPos.Y.Scale, startPos.Y.Offset + delta.Y\n"
            "\t\t\t)\n"
            "\t\tend\n"
            "\tend)\n\n"
            "\tUserInputService.InputEnded:Connect(function(input: InputObject)\n"
            "\t\tif input.UserInputType == Enum.UserInputType.MouseButton1\n"
            "\t\t\tor input.UserInputType == Enum.UserInputType.Touch then\n"
            "\t\t\tdragging = false\n"
            "\t\tend\n"
            "\tend)\n"
            "end\n",
            category="controller",
            difficulty=2,
        ),
        _ex(
            "How do I create a notification/toast system that auto-dismisses?",
            "<think>\n"
            "Create a notification frame, tween it in, wait, tween it out, then destroy. "
            "Stack notifications vertically by tracking active count. Use task.spawn so "
            "multiple notifications can coexist.\n"
            "</think>\n\n"
            "--!strict\n"
            'local TweenService = game:GetService("TweenService")\n'
            'local Players = game:GetService("Players")\n\n'
            "local player = Players.LocalPlayer\n"
            'local playerGui = player:WaitForChild("PlayerGui") :: PlayerGui\n\n'
            "local notifContainer: ScreenGui\n"
            "local activeCount: number = 0\n\n"
            "local function ensureContainer(): ScreenGui\n"
            "\tif not notifContainer or not notifContainer.Parent then\n"
            '\t\tnotifContainer = Instance.new("ScreenGui")\n'
            '\t\tnotifContainer.Name = "Notifications"\n'
            "\t\tnotifContainer.ResetOnSpawn = false\n"
            "\t\tnotifContainer.Parent = playerGui\n"
            "\tend\n"
            "\treturn notifContainer\n"
            "end\n\n"
            "local function showNotification(text: string, duration: number?)\n"
            "\tlocal dur = duration or 3\n"
            "\tlocal gui = ensureContainer()\n"
            "\tlocal yOffset = activeCount * 50\n"
            "\tactiveCount += 1\n\n"
            '\tlocal frame = Instance.new("Frame")\n'
            "\tframe.Size = UDim2.new(0.3, 0, 0, 40)\n"
            "\tframe.Position = UDim2.new(0.35, 0, 0, -50) -- start offscreen\n"
            "\tframe.BackgroundColor3 = Color3.fromRGB(30, 30, 50)\n"
            "\tframe.BackgroundTransparency = 0.1\n"
            "\tframe.Parent = gui\n\n"
            '\tlocal label = Instance.new("TextLabel")\n'
            "\tlabel.Size = UDim2.fromScale(1, 1)\n"
            "\tlabel.Text = text\n"
            "\tlabel.TextColor3 = Color3.new(1, 1, 1)\n"
            "\tlabel.BackgroundTransparency = 1\n"
            "\tlabel.Font = Enum.Font.Gotham\n"
            "\tlabel.TextSize = 16\n"
            "\tlabel.Parent = frame\n\n"
            "\tlocal fadeInfo = TweenInfo.new(0.3, Enum.EasingStyle.Quad)\n"
            "\tTweenService:Create(frame, fadeInfo, {\n"
            "\t\tPosition = UDim2.new(0.35, 0, 0, 10 + yOffset),\n"
            "\t}):Play()\n\n"
            "\ttask.delay(dur, function()\n"
            "\t\tTweenService:Create(frame, fadeInfo, {\n"
            "\t\t\tPosition = UDim2.new(0.35, 0, 0, -50),\n"
            "\t\t}):Play()\n"
            "\t\ttask.wait(0.3)\n"
            "\t\tframe:Destroy()\n"
            "\t\tactiveCount -= 1\n"
            "\tend)\n"
            "end\n",
            category="controller",
            difficulty=3,
        ),
        _ex(
            "How do I make a scrolling frame with dynamically created list items?",
            "<think>\n"
            "ScrollingFrame with a UIListLayout handles vertical scrolling. Set "
            "CanvasSize to UDim2.new(0,0,0,0) and AutomaticCanvasSize to Y so it "
            "auto-expands. Add children and UIListLayout handles positioning.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n'
            "local player = Players.LocalPlayer\n"
            'local playerGui = player:WaitForChild("PlayerGui") :: PlayerGui\n\n'
            'local gui = Instance.new("ScreenGui")\n'
            'gui.Name = "InventoryUI"\n'
            "gui.Parent = playerGui\n\n"
            'local scroll = Instance.new("ScrollingFrame")\n'
            "scroll.Size = UDim2.fromScale(0.3, 0.5)\n"
            "scroll.Position = UDim2.fromScale(0.35, 0.25)\n"
            "scroll.CanvasSize = UDim2.new(0, 0, 0, 0)\n"
            "scroll.AutomaticCanvasSize = Enum.AutomaticSize.Y\n"
            "scroll.ScrollBarThickness = 6\n"
            "scroll.BackgroundColor3 = Color3.fromRGB(25, 25, 35)\n"
            "scroll.Parent = gui\n\n"
            'local layout = Instance.new("UIListLayout")\n'
            "layout.Padding = UDim.new(0, 4)\n"
            "layout.Parent = scroll\n\n"
            "local function addItem(name: string, count: number)\n"
            '\tlocal item = Instance.new("TextLabel")\n'
            "\titem.Size = UDim2.new(1, 0, 0, 36)\n"
            "\titem.Text = `  {name} x{count}`\n"
            "\titem.TextColor3 = Color3.new(1, 1, 1)\n"
            "\titem.TextXAlignment = Enum.TextXAlignment.Left\n"
            "\titem.BackgroundColor3 = Color3.fromRGB(40, 40, 55)\n"
            "\titem.Font = Enum.Font.Gotham\n"
            "\titem.TextSize = 14\n"
            "\titem.Parent = scroll\n"
            "end\n\n"
            "-- Populate\n"
            'addItem("Grapple Hook", 1)\n'
            'addItem("Health Potion", 5)\n'
            'addItem("Bloom Crystal", 12)\n'
            'addItem("Iron Ore", 30)\n',
            category="controller",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: Signal patterns
# ---------------------------------------------------------------------------


def _signal_examples() -> list[dict]:
    return [
        _ex(
            "How do I use the Signal library for custom events in Vertigo?",
            "<think>\n"
            "Signal (from Packages) is a lightweight event emitter. Create signals on services "
            "to decouple modules. Fire sends data to all connected listeners. Connect returns "
            "a connection that can be disconnected. This replaces BindableEvents.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local Packages = ReplicatedStorage:WaitForChild("Packages")\n'
            "local Signal = require(Packages.Signal)\n\n"
            "-- In a service module:\n"
            "local ZoneService = {}\n\n"
            "-- Public signals for other modules to listen to\n"
            "ZoneService.PlayerEnteredZone = Signal.new()\n"
            "ZoneService.PlayerExitedZone = Signal.new()\n\n"
            "function ZoneService:Start()\n"
            "\t-- When zone change detected:\n"
            '\tZoneService.PlayerEnteredZone:Fire(player, "zone_canopy_v1")\n'
            "end\n\n"
            "-- In a consumer (controller or other service):\n"
            "-- ZoneService.PlayerEnteredZone:Connect(function(player, zoneId)\n"
            "--     print(`{player.Name} entered {zoneId}`)\n"
            "-- end)\n\n"
            "return ZoneService\n",
            category="service",
            difficulty=2,
        ),
        _ex(
            "How do I use Trove for cleanup management?",
            "<think>\n"
            "Trove batches cleanup of connections, instances, and callbacks. Call :Add() to "
            "track resources, :Connect() for event connections, :Destroy() to clean everything. "
            "Essential for controllers that set up per-character state.\n"
            "</think>\n\n"
            "--!strict\n"
            'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
            'local RunService = game:GetService("RunService")\n'
            'local Packages = ReplicatedStorage:WaitForChild("Packages")\n'
            "local Trove = require(Packages.Trove)\n\n"
            "local GrappleController = {}\n"
            "local _trove = Trove.new()\n"
            "local _characterTrove: typeof(Trove.new())? = nil\n\n"
            "function GrappleController:Start()\n"
            '\tlocal Players = game:GetService("Players")\n'
            "\tlocal player = Players.LocalPlayer\n\n"
            "\t_trove:Connect(player.CharacterAdded, function(character: Model)\n"
            "\t\t-- Clean up previous character's resources\n"
            "\t\tif _characterTrove then\n"
            "\t\t\t_characterTrove:Destroy()\n"
            "\t\tend\n"
            "\t\t_characterTrove = Trove.new()\n\n"
            "\t\t-- Track per-character connections\n"
            '\t\tlocal root = character:WaitForChild("HumanoidRootPart") :: BasePart\n'
            "\t\t_characterTrove:Connect(RunService.Heartbeat, function(dt: number)\n"
            "\t\t\t-- per-frame logic\n"
            "\t\tend)\n\n"
            "\t\t-- Track instances created for this character\n"
            '\t\tlocal beam = Instance.new("Beam")\n'
            "\t\t_characterTrove:Add(beam)\n"
            "\tend)\n"
            "end\n\n"
            "return GrappleController\n",
            category="controller",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: Debugging workflow
# ---------------------------------------------------------------------------


def _debugging_workflow_examples() -> list[dict]:
    return [
        _ex(
            "How do I use print and warn effectively for debugging in Roblox?",
            "<think>\n"
            "Prefix all logs with a module tag like [ServiceName] for easy filtering. Use "
            "print for info, warn for non-fatal issues, error() for fatal. In production, "
            "wrap verbose logging behind a DEBUG flag to reduce output noise.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local DEBUG: boolean = true\n"
            'local MODULE_TAG = "[GrappleService]"\n\n'
            "local function debugLog(...: any)\n"
            "\tif DEBUG then\n"
            "\t\tprint(MODULE_TAG, ...)\n"
            "\tend\n"
            "end\n\n"
            "local function warnLog(...: any)\n"
            "\twarn(MODULE_TAG, ...)\n"
            "end\n\n"
            "-- Usage:\n"
            'debugLog("Player attached to anchor", anchorName)\n'
            'warnLog("Grapple validation failed: out of range")\n\n'
            "-- For structured logs:\n"
            "local function logEvent(event: string, data: { [string]: any })\n"
            "\tlocal parts: { string } = { `{MODULE_TAG} [{event}]` }\n"
            "\tfor key, value in data do\n"
            "\t\ttable.insert(parts, `{key}={tostring(value)}`)\n"
            "\tend\n"
            '\tprint(table.concat(parts, " "))\n'
            "end\n\n"
            'logEvent("grapple_attach", {\n'
            '\tplayer = "Player1",\n'
            '\tanchor = "Anchor_42",\n'
            "\tdistance = 85.3,\n"
            "})\n",
            category="debugging",
            difficulty=1,
        ),
        _ex(
            "How do I use breakpoints and the debugger in Roblox Studio?",
            "<think>\n"
            "Studio has a built-in debugger. Click the line number gutter to set breakpoints. "
            "When execution hits the breakpoint, Studio pauses and shows local variables, "
            "call stack, and watch expressions. Step Over (F10), Step Into (F11), Continue (F5). "
            "Use conditional breakpoints for specific scenarios.\n"
            "</think>\n\n"
            "Roblox Studio debugging workflow:\n\n"
            "**1. Set a breakpoint:** Click the left gutter next to a line number (red dot appears).\n\n"
            "**2. Run in Play mode:** Press F5 or the Play button.\n\n"
            "**3. When paused at breakpoint:**\n"
            "- **Watch** panel: hover variables or add custom watch expressions\n"
            "- **Call Stack** panel: see which functions called this one\n"
            "- **Locals** panel: all variables in current scope with values\n\n"
            "**4. Navigation:**\n"
            "- `F5` — Continue (resume execution)\n"
            "- `F10` — Step Over (next line, skip function internals)\n"
            "- `F11` — Step Into (enter function call)\n"
            "- `Shift+F11` — Step Out (finish current function)\n\n"
            "**5. Programmatic breakpoints:**\n"
            "```luau\n"
            "--!strict\n"
            "-- Add this where you want to pause:\n"
            "if someCondition then\n"
            '\tdebug.profilebegin("BREAKPOINT_HERE")\n'
            "\t-- Set a real breakpoint on this line\n"
            "\tdebug.profileend()\n"
            "end\n"
            "```\n\n"
            "**6. Output window** (View > Output): Shows print/warn/error output in real time.\n\n"
            "**7. MicroProfiler** (Ctrl+F6): Frame-level performance timeline.\n",
            category="debugging",
            difficulty=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: Spawning and respawn
# ---------------------------------------------------------------------------


def _spawn_examples() -> list[dict]:
    return [
        _ex(
            "How do I customize the player spawn location and respawn behavior?",
            "<think>\n"
            "SpawnLocation parts determine where players appear. Set AllowTeamChangeOnTouch "
            "to false to prevent accidental team changes. CharacterAutoLoads controls auto-"
            "respawn. For custom respawn, disable auto-load and spawn manually with "
            "Player:LoadCharacter().\n"
            "</think>\n\n"
            "--!strict\n"
            'local Players = game:GetService("Players")\n\n'
            "-- Disable auto-respawn for custom control\n"
            "Players.CharacterAutoLoads = false\n\n"
            "local RESPAWN_DELAY: number = 3\n\n"
            "Players.PlayerAdded:Connect(function(player: Player)\n"
            "\t-- Initial spawn\n"
            "\tplayer:LoadCharacter()\n\n"
            "\tplayer.CharacterAdded:Connect(function(character: Model)\n"
            '\t\tlocal humanoid = character:WaitForChild("Humanoid") :: Humanoid\n'
            "\t\thumanoid.Died:Connect(function()\n"
            "\t\t\tprint(`{player.Name} died, respawning in {RESPAWN_DELAY}s`)\n"
            "\t\t\ttask.wait(RESPAWN_DELAY)\n"
            "\t\t\tif player.Parent then -- still in game\n"
            "\t\t\t\tplayer:LoadCharacter()\n"
            "\t\t\tend\n"
            "\t\tend)\n"
            "\tend)\n"
            "end)\n\n"
            "-- Custom spawn position (instead of SpawnLocation part):\n"
            "-- After LoadCharacter, move the root part:\n"
            '-- local root = character:WaitForChild("HumanoidRootPart") :: BasePart\n'
            "-- root.CFrame = CFrame.new(0, 50, 0) -- custom spawn point\n",
            category="service",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: Particle/VFX patterns
# ---------------------------------------------------------------------------


def _vfx_examples() -> list[dict]:
    return [
        _ex(
            "How do I create a particle burst effect when a player collects something?",
            "<think>\n"
            "Pre-create a ParticleEmitter, disable it by default, then :Emit(count) for "
            "burst effects. This avoids creating instances at runtime. Parent to the part "
            "or an Attachment for positional effects.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local function createBurstEmitter(parent: BasePart): ParticleEmitter\n"
            '\tlocal emitter = Instance.new("ParticleEmitter")\n'
            "\temitter.Rate = 0 -- no continuous emission\n"
            "\temitter.Lifetime = NumberRange.new(0.5, 1)\n"
            "\temitter.Speed = NumberRange.new(10, 20)\n"
            "\temitter.SpreadAngle = Vector2.new(180, 180)\n"
            "\temitter.Size = NumberSequence.new({\n"
            "\t\tNumberSequenceKeypoint.new(0, 1),\n"
            "\t\tNumberSequenceKeypoint.new(1, 0),\n"
            "\t})\n"
            "\temitter.Color = ColorSequence.new({\n"
            "\t\tColorSequenceKeypoint.new(0, Color3.fromRGB(255, 220, 50)),\n"
            "\t\tColorSequenceKeypoint.new(1, Color3.fromRGB(255, 100, 0)),\n"
            "\t})\n"
            "\temitter.LightEmission = 1\n"
            "\temitter.Parent = parent\n"
            "\treturn emitter\n"
            "end\n\n"
            "-- Pre-create in :Init()\n"
            "local emitter = createBurstEmitter(coinPart)\n\n"
            "-- On collect:\n"
            "emitter:Emit(20) -- burst 20 particles\n",
            category="controller",
            difficulty=2,
        ),
        _ex(
            "How do I make a beam effect between two points for a grapple line?",
            "<think>\n"
            "Beam connects two Attachments and renders a textured strip between them. Create "
            "Attachments on the source (player) and target (anchor), parent a Beam to either. "
            "Set Texture, Width, Color, and FaceCamera for the visual style.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "local function createGrappleBeam(\n"
            "\tsource: BasePart,\n"
            "\ttarget: BasePart\n"
            "): (Beam, Attachment, Attachment)\n"
            '\tlocal sourceAttach = Instance.new("Attachment")\n'
            "\tsourceAttach.Parent = source\n\n"
            '\tlocal targetAttach = Instance.new("Attachment")\n'
            "\ttargetAttach.Parent = target\n\n"
            '\tlocal beam = Instance.new("Beam")\n'
            "\tbeam.Attachment0 = sourceAttach\n"
            "\tbeam.Attachment1 = targetAttach\n"
            "\tbeam.Width0 = 0.3\n"
            "\tbeam.Width1 = 0.1\n"
            "\tbeam.FaceCamera = true\n"
            "\tbeam.LightEmission = 0.8\n"
            "\tbeam.Color = ColorSequence.new({\n"
            "\t\tColorSequenceKeypoint.new(0, Color3.fromRGB(100, 200, 255)),\n"
            "\t\tColorSequenceKeypoint.new(1, Color3.fromRGB(50, 150, 255)),\n"
            "\t})\n"
            "\tbeam.Transparency = NumberSequence.new({\n"
            "\t\tNumberSequenceKeypoint.new(0, 0),\n"
            "\t\tNumberSequenceKeypoint.new(0.8, 0),\n"
            "\t\tNumberSequenceKeypoint.new(1, 1),\n"
            "\t})\n"
            "\tbeam.Parent = source\n\n"
            "\treturn beam, sourceAttach, targetAttach\n"
            "end\n\n"
            "local function destroyGrappleBeam(beam: Beam, a0: Attachment, a1: Attachment)\n"
            "\tbeam:Destroy()\n"
            "\ta0:Destroy()\n"
            "\ta1:Destroy()\n"
            "end\n",
            category="controller",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: Error handling patterns
# ---------------------------------------------------------------------------


def _error_handling_examples() -> list[dict]:
    return [
        _ex(
            "What's the difference between pcall and xpcall in Luau?",
            "<think>\n"
            "pcall catches errors and returns success + result. xpcall adds an error handler "
            "function that receives the error before the stack unwinds — useful for logging "
            "stack traces. Both are essential for production code wrapping DataStore, HTTP, "
            "and any yielding operation.\n"
            "</think>\n\n"
            "--!strict\n\n"
            "-- pcall: basic error catching\n"
            "local ok, result = pcall(function()\n"
            '\treturn workspace:FindFirstChild("NonExistent"):GetAttribute("x")\n'
            "end)\n"
            "if not ok then\n"
            '\twarn("pcall caught:", result) -- result is the error message string\n'
            "end\n\n"
            "-- xpcall: error handler gets the error + can add context\n"
            "local function errorHandler(err: string): string\n"
            "\tlocal trace = debug.traceback(err, 2)\n"
            '\twarn("[ErrorHandler] Stack trace:")\n'
            "\twarn(trace)\n"
            "\treturn trace -- returned as the 'result' from xpcall\n"
            "end\n\n"
            "local ok2, result2 = xpcall(function()\n"
            '\terror("Something went wrong")\n'
            "end, errorHandler)\n\n"
            "-- Best practice: wrap yielding operations\n"
            "local function safeFetch(store: DataStore, key: string): any?\n"
            "\tlocal ok3, value = pcall(function()\n"
            "\t\treturn store:GetAsync(key)\n"
            "\tend)\n"
            "\tif ok3 then\n"
            "\t\treturn value\n"
            "\telse\n"
            "\t\twarn(`[SafeFetch] Failed for key {key}: {value}`)\n"
            "\t\treturn nil\n"
            "\tend\n"
            "end\n",
            category="general_luau",
            difficulty=2,
        ),
    ]


# ---------------------------------------------------------------------------
# Extra: Lighting and effects
# ---------------------------------------------------------------------------


def _lighting_examples() -> list[dict]:
    return [
        _ex(
            "How do I change lighting properties for different zones/biomes?",
            "<think>\n"
            "Tween Lighting properties (Ambient, OutdoorAmbient, FogColor, FogEnd, etc.) "
            "when the player enters a new zone. This creates smooth atmospheric transitions. "
            "Store zone presets in config modules for data-driven design.\n"
            "</think>\n\n"
            "--!strict\n"
            'local Lighting = game:GetService("Lighting")\n'
            'local TweenService = game:GetService("TweenService")\n\n'
            "type ZoneLighting = {\n"
            "\tAmbient: Color3,\n"
            "\tOutdoorAmbient: Color3,\n"
            "\tFogColor: Color3,\n"
            "\tFogEnd: number,\n"
            "\tBrightness: number,\n"
            "\tClockTime: number,\n"
            "}\n\n"
            "local ZONE_PRESETS: { [string]: ZoneLighting } = table.freeze({\n"
            "\tcanopy = {\n"
            "\t\tAmbient = Color3.fromRGB(40, 60, 30),\n"
            "\t\tOutdoorAmbient = Color3.fromRGB(80, 120, 60),\n"
            "\t\tFogColor = Color3.fromRGB(100, 140, 80),\n"
            "\t\tFogEnd = 500,\n"
            "\t\tBrightness = 1.5,\n"
            "\t\tClockTime = 14,\n"
            "\t},\n"
            "\tabyss = {\n"
            "\t\tAmbient = Color3.fromRGB(10, 10, 20),\n"
            "\t\tOutdoorAmbient = Color3.fromRGB(5, 5, 15),\n"
            "\t\tFogColor = Color3.fromRGB(0, 0, 10),\n"
            "\t\tFogEnd = 150,\n"
            "\t\tBrightness = 0.3,\n"
            "\t\tClockTime = 0,\n"
            "\t},\n"
            "})\n\n"
            "local TRANSITION_TIME: number = 3\n\n"
            "local function transitionToZone(zoneName: string)\n"
            "\tlocal preset = ZONE_PRESETS[zoneName]\n"
            "\tif not preset then\n"
            "\t\twarn(`[Lighting] Unknown zone: {zoneName}`)\n"
            "\t\treturn\n"
            "\tend\n"
            "\tlocal info = TweenInfo.new(TRANSITION_TIME, Enum.EasingStyle.Quad)\n"
            "\tTweenService:Create(Lighting, info, preset :: any):Play()\n"
            "end\n",
            category="controller",
            difficulty=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    all_examples: list[dict] = []

    generators = [
        ("DataStore patterns", _datastore_examples),
        ("Remote debugging", _remote_debugging_examples),
        ("Camera follow systems", _camera_examples),
        ("Client/server boundary", _client_server_examples),
        ("Custom character controllers", _character_controller_examples),
        ("Performance optimization", _performance_examples),
        ("Leaderboards", _leaderboard_examples),
        ("Part.Touched detection", _touched_examples),
        ("Cooldown systems", _cooldown_examples),
        ("Raycasting", _raycast_examples),
        ("Vertigo physics (grapple/glide/vehicle)", _vertigo_physics_examples),
        ("Anti-cheat / server validation", _anticheat_examples),
        ("UI creation", _ui_examples),
        ("Animation/tween patterns", _tween_examples),
        ("Sound management", _sound_examples),
        ("Terrain manipulation", _terrain_examples),
        ("Module require patterns", _module_examples),
        ("Promise/async patterns", _async_examples),
        ("Memory leak prevention", _memory_leak_examples),
        ("Mobile input handling", _mobile_input_examples),
        ("DataStore extra", _datastore_extra_examples),
        ("Networking extra", _networking_extra_examples),
        ("Camera extra", _camera_extra_examples),
        ("Physics extra", _physics_extra_examples),
        ("CollectionService patterns", _collection_service_examples),
        ("Attribute patterns", _attribute_examples),
        ("Luau patterns", _luau_patterns_examples),
        ("Anti-cheat extra", _anticheat_extra_examples),
        ("UI extra", _ui_extra_examples),
        ("Error handling", _error_handling_examples),
        ("Lighting/effects", _lighting_examples),
        ("Signal patterns", _signal_examples),
        ("Debugging workflow", _debugging_workflow_examples),
        ("Spawning/respawn", _spawn_examples),
        ("VFX patterns", _vfx_examples),
    ]

    print("Generating DevForum Q&A training pairs...\n")

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

    # Difficulty breakdown
    diffs = Counter(ex.get("difficulty", 0) for ex in all_examples)
    print("\nDifficulty distribution:")
    for diff, count in sorted(diffs.items()):
        print(f"  level {diff}: {count}")


if __name__ == "__main__":
    main()
