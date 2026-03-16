#!/usr/bin/env python3
from __future__ import annotations

"""
Generate synthetic Vertigo training examples with <think> reasoning + Luau code.

Output: data/raw/synthetic.jsonl
"""

import argparse
import json
import textwrap
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "synthetic.jsonl"

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
# Seed tasks — 50 total across 5 categories
# ---------------------------------------------------------------------------

SEED_TASKS: list[dict] = [
    # === Services (10) ===
    {
        "instruction": "Write a DataService module that saves player profile data to DataStoreService with retry and session locking.",
        "category": "service",
        "difficulty": 4,
    },
    {
        "instruction": "Write a DataService:LoadProfile method that loads player data with version reconciliation and default backfill.",
        "category": "service",
        "difficulty": 3,
    },
    {
        "instruction": "Write an AbilityService:ValidateAbilityUse method that checks cooldown, stamina, and grounded state before allowing an ability.",
        "category": "service",
        "difficulty": 3,
    },
    {
        "instruction": "Write an AbilityService:ProcessGrapple method that validates anchor distance, line-of-sight, and applies the grapple state.",
        "category": "service",
        "difficulty": 4,
    },
    {
        "instruction": "Write a VehicleService:SpawnVehicle method that creates a vehicle from a template, assigns ownership, and registers cleanup.",
        "category": "service",
        "difficulty": 4,
    },
    {
        "instruction": "Write a VehicleService:DismountVehicle method that ejects the player, clears seat weld, and transitions state.",
        "category": "service",
        "difficulty": 3,
    },
    {
        "instruction": "Write a TelemetryService that batches player events and flushes to an HTTP endpoint every 30 seconds.",
        "category": "service",
        "difficulty": 3,
    },
    {
        "instruction": "Write a TelemetryService:RecordTraversal method that logs ability usage, duration, and distance traveled.",
        "category": "service",
        "difficulty": 2,
    },
    {
        "instruction": "Write a ZoneService:GetActiveZone method that determines which zone a player is in based on Y-level boundaries.",
        "category": "service",
        "difficulty": 2,
    },
    {
        "instruction": "Write a RewardService:ClaimReward method with server-side duplicate claim prevention and inventory update.",
        "category": "service",
        "difficulty": 3,
    },
    # === Physics (10) ===
    {
        "instruction": "Write a spring solver function that computes critically-damped spring motion for camera smoothing.",
        "category": "physics",
        "difficulty": 4,
    },
    {
        "instruction": "Write a Verlet rope simulation module with constraint satisfaction for grapple cable rendering.",
        "category": "physics",
        "difficulty": 5,
    },
    {
        "instruction": "Write a FlowField module that computes wind vectors on a 2D grid for glide steering.",
        "category": "physics",
        "difficulty": 4,
    },
    {
        "instruction": "Write an exponential damping function for velocity decay that is framerate-independent.",
        "category": "physics",
        "difficulty": 3,
    },
    {
        "instruction": "Write a trajectory prediction function that computes a ballistic arc given initial velocity and gravity.",
        "category": "physics",
        "difficulty": 3,
    },
    {
        "instruction": "Write a swept-sphere collision test for grapple hook projectile against terrain parts.",
        "category": "physics",
        "difficulty": 5,
    },
    {
        "instruction": "Write a quaternion slerp utility for smooth character rotation during wallrun transitions.",
        "category": "physics",
        "difficulty": 4,
    },
    {
        "instruction": "Write a buoyancy simulation function that applies vertical force based on submerged volume.",
        "category": "physics",
        "difficulty": 3,
    },
    {
        "instruction": "Write an angular velocity clamping function for vehicle spin-out prevention.",
        "category": "physics",
        "difficulty": 2,
    },
    {
        "instruction": "Write a PID controller module for hover-bike altitude hold.",
        "category": "physics",
        "difficulty": 4,
    },
    # === Builders (10) ===
    {
        "instruction": "Write a procedural zone builder that generates platform rings at specified Y-levels with random gap placement.",
        "category": "builder",
        "difficulty": 4,
    },
    {
        "instruction": "Write a crystal cavern builder that places BloomCrystal instances along cave walls using surface normal raycasts.",
        "category": "builder",
        "difficulty": 4,
    },
    {
        "instruction": "Write a grapple anchor placement function that distributes GrappleAnchor tags along vertical cliff faces.",
        "category": "builder",
        "difficulty": 3,
    },
    {
        "instruction": "Write a floating island builder that creates terrain voxels in an ellipsoid shape with moss and rock materials.",
        "category": "builder",
        "difficulty": 4,
    },
    {
        "instruction": "Write a bridge builder that spans two points with segmented planks and rope-like constraints.",
        "category": "builder",
        "difficulty": 3,
    },
    {
        "instruction": "Write a waterfall builder that creates cascading water parts with particle emitters and sound regions.",
        "category": "builder",
        "difficulty": 3,
    },
    {
        "instruction": "Write a vine network builder that connects anchor points with rope-like meshes for wallrun surfaces.",
        "category": "builder",
        "difficulty": 4,
    },
    {
        "instruction": "Write a mushroom grove builder that places scaled mushroom models with CollectionService tags for bounce pads.",
        "category": "builder",
        "difficulty": 3,
    },
    {
        "instruction": "Write a ruin pillar builder that generates broken column geometry with randomized erosion.",
        "category": "builder",
        "difficulty": 3,
    },
    {
        "instruction": "Write a checkpoint ring builder that places checkpoint parts at traversal waypoints with progressive numbering.",
        "category": "builder",
        "difficulty": 2,
    },
    # === Networking (10) ===
    {
        "instruction": "Write a Remotes module that defines all RemoteEvents and RemoteFunctions in a type-safe registry.",
        "category": "networking",
        "difficulty": 3,
    },
    {
        "instruction": "Write server-side validation for RequestUseAbility that checks ability exists, cooldown elapsed, and stamina sufficient.",
        "category": "networking",
        "difficulty": 3,
    },
    {
        "instruction": "Write a rate limiter module that throttles per-player remote calls with a token bucket algorithm.",
        "category": "networking",
        "difficulty": 4,
    },
    {
        "instruction": "Write a server handler for RequestSpawnVehicle that validates player eligibility, proximity to spawn pad, and vehicle limit.",
        "category": "networking",
        "difficulty": 3,
    },
    {
        "instruction": "Write a StateSync broadcaster that sends delta-compressed player state to nearby clients each heartbeat.",
        "category": "networking",
        "difficulty": 5,
    },
    {
        "instruction": "Write a client-side remote wrapper that queues requests and deduplicates rapid fire inputs.",
        "category": "networking",
        "difficulty": 3,
    },
    {
        "instruction": "Write server validation for RequestClaimReward that checks reward exists, player is in range, and not already claimed.",
        "category": "networking",
        "difficulty": 3,
    },
    {
        "instruction": "Write a RemoteFunction handler for GetPlayerProfile that returns sanitized profile data to the requesting client.",
        "category": "networking",
        "difficulty": 2,
    },
    {
        "instruction": "Write an EventNotify dispatcher that sends zone-enter and ability-unlock notifications to specific players.",
        "category": "networking",
        "difficulty": 2,
    },
    {
        "instruction": "Write a kick/ban enforcement module that validates punishment data and disconnects the player with a reason code.",
        "category": "networking",
        "difficulty": 3,
    },
    # === Config (10) ===
    {
        "instruction": "Write an Abilities config module that exports GrappleTuning with range, cooldown, speed, and stamina cost.",
        "category": "config",
        "difficulty": 2,
    },
    {
        "instruction": "Write a GlideTuning config with lift, drag, turn rate, max speed, and stamina drain per second.",
        "category": "config",
        "difficulty": 2,
    },
    {
        "instruction": "Write a Vehicles config module that exports DirtBikeTuning with top speed, acceleration, lean angle, and brake force.",
        "category": "config",
        "difficulty": 2,
    },
    {
        "instruction": "Write a HoverBikeTuning config with hover height, PID gains, max tilt, and boost multiplier.",
        "category": "config",
        "difficulty": 2,
    },
    {
        "instruction": "Write a Checkpoints config that defines checkpoint positions, required abilities, and reward tier per checkpoint.",
        "category": "config",
        "difficulty": 2,
    },
    {
        "instruction": "Write a Zones config that maps zone names to Y-level ranges, ambient color, fog density, and music asset IDs.",
        "category": "config",
        "difficulty": 3,
    },
    {
        "instruction": "Write a WallrunTuning config with max duration, min speed, gravity scale, and exit launch velocity.",
        "category": "config",
        "difficulty": 2,
    },
    {
        "instruction": "Write a SlideTuning config with speed boost, friction, max duration, and cooldown.",
        "category": "config",
        "difficulty": 2,
    },
    {
        "instruction": "Write a CombatTuning config with damage values, hit detection range, knockback force, and i-frame duration.",
        "category": "config",
        "difficulty": 3,
    },
    {
        "instruction": "Write a StaminaTuning config with max stamina, regen rate, regen delay, and per-ability costs.",
        "category": "config",
        "difficulty": 2,
    },
]

# ---------------------------------------------------------------------------
# Completions — keyed by instruction prefix (first 40 chars) for lookup
# ---------------------------------------------------------------------------


def _completions() -> dict[str, dict]:
    """Return dict mapping instruction[:40] -> {thinking, code, difficulty}."""
    C: dict[str, dict] = {}

    def add(instruction: str, thinking: str, code: str):
        C[instruction[:40]] = {"thinking": thinking, "code": code}

    # -- Services --
    add(
        SEED_TASKS[0]["instruction"],
        "Need retry logic with exponential backoff, session locking via a GUID to prevent concurrent writes, and UpdateAsync for atomic saves.",
        textwrap.dedent("""\
        --!strict
        local DataStoreService = game:GetService("DataStoreService")
        local HttpService = game:GetService("HttpService")
        local DataService = {}
        local store = DataStoreService:GetDataStore("PlayerProfiles_v2")
        local sessions: {[number]: string} = {}
        local MAX_RETRIES = 3

        function DataService:SaveProfile(userId: number, data: {[string]: any}): boolean
        \tlocal lockId = sessions[userId]
        \tif not lockId then return false end
        \tfor attempt = 1, MAX_RETRIES do
        \t\tlocal ok, err = pcall(function()
        \t\t\tstore:UpdateAsync(tostring(userId), function(old)
        \t\t\t\tif old and old._lockId ~= lockId then return nil end
        \t\t\t\tdata._lockId = lockId
        \t\t\t\tdata._saveTime = os.time()
        \t\t\t\treturn data
        \t\t\tend)
        \t\tend)
        \t\tif ok then return true end
        \t\tif attempt < MAX_RETRIES then task.wait(2 ^ attempt) end
        \tend
        \treturn false
        end

        function DataService:AcquireLock(userId: number): string
        \tlocal lockId = HttpService:GenerateGUID(false)
        \tsessions[userId] = lockId
        \treturn lockId
        end

        function DataService:ReleaseLock(userId: number)
        \tsessions[userId] = nil
        end

        return DataService"""),
    )

    add(
        SEED_TASKS[1]["instruction"],
        "LoadProfile needs pcall around GetAsync, version check against SCHEMA_VERSION, and backfill missing keys with defaults.",
        textwrap.dedent("""\
        --!strict
        local DataStoreService = game:GetService("DataStoreService")
        local store = DataStoreService:GetDataStore("PlayerProfiles_v2")
        local SCHEMA_VERSION = 2
        local DEFAULTS = table.freeze({coins = 0, abilities = {}, zone = "surface", schemaVersion = SCHEMA_VERSION})

        local DataService = {}

        function DataService:LoadProfile(userId: number): {[string]: any}?
        \tlocal ok, data = pcall(store.GetAsync, store, tostring(userId))
        \tif not ok then warn("[DataService] Load failed:", data); return nil end
        \tif not data then return table.clone(DEFAULTS) end
        \tif (data.schemaVersion or 0) < SCHEMA_VERSION then
        \t\tfor key, default in DEFAULTS do
        \t\t\tif data[key] == nil then data[key] = default end
        \t\tend
        \t\tdata.schemaVersion = SCHEMA_VERSION
        \tend
        \treturn data
        end

        return DataService"""),
    )

    add(
        SEED_TASKS[2]["instruction"],
        "Validate by checking ability registry, cooldown timestamp, stamina pool, and grounded state from Humanoid.FloorMaterial.",
        textwrap.dedent("""\
        --!strict
        local Shared = game:GetService("ReplicatedStorage"):WaitForChild("Shared")
        local Config = require(Shared.Config.Abilities)
        local AbilityService = {}
        local cooldowns: {[number]: {[string]: number}} = {}

        function AbilityService:ValidateAbilityUse(player: Player, abilityId: string, stamina: number): (boolean, string?)
        \tlocal tuning = Config.Abilities[abilityId]
        \tif not tuning then return false, "unknown_ability" end
        \tlocal now = os.clock()
        \tlocal playerCds = cooldowns[player.UserId] or {}
        \tif (playerCds[abilityId] or 0) > now then return false, "on_cooldown" end
        \tif stamina < tuning.staminaCost then return false, "no_stamina" end
        \tlocal char = player.Character
        \tif not char then return false, "no_character" end
        \tlocal hum = char:FindFirstChildOfClass("Humanoid")
        \tif tuning.requireGrounded and hum and hum.FloorMaterial == Enum.Material.Air then
        \t\treturn false, "not_grounded"
        \tend
        \tplayerCds[abilityId] = now + tuning.cooldown
        \tcooldowns[player.UserId] = playerCds
        \treturn true, nil
        end

        return AbilityService"""),
    )

    add(
        SEED_TASKS[3]["instruction"],
        "ProcessGrapple must validate anchor distance <= max range, raycast for line-of-sight, then set player grapple state with target position.",
        textwrap.dedent("""\
        --!strict
        local Shared = game:GetService("ReplicatedStorage"):WaitForChild("Shared")
        local Config = require(Shared.Config.Abilities)
        local AbilityService = {}

        function AbilityService:ProcessGrapple(player: Player, anchorPos: Vector3): (boolean, string?)
        \tlocal tuning = Config.GrappleTuning
        \tlocal char = player.Character
        \tif not char then return false, "no_character" end
        \tlocal root = char:FindFirstChild("HumanoidRootPart") :: BasePart?
        \tif not root then return false, "no_root" end
        \tlocal dist = (anchorPos - root.Position).Magnitude
        \tif dist > tuning.maxRange then return false, "out_of_range" end
        \tlocal ray = workspace:Raycast(root.Position, anchorPos - root.Position)
        \tif ray and not ray.Instance:HasTag("GrappleAnchor") then
        \t\treturn false, "obstructed"
        \tend
        \tchar:SetAttribute("GrappleTarget", anchorPos)
        \tchar:SetAttribute("GrappleState", "reeling")
        \treturn true, nil
        end

        return AbilityService"""),
    )

    add(
        SEED_TASKS[4]["instruction"],
        "SpawnVehicle clones from ServerStorage template, sets NetworkOwner, registers in a cleanup table with Trove.",
        textwrap.dedent("""\
        --!strict
        local ServerStorage = game:GetService("ServerStorage")
        local VehicleService = {}
        local activeVehicles: {[number]: Model} = {}

        function VehicleService:SpawnVehicle(player: Player, vehicleId: string, spawnCF: CFrame): Model?
        \tif activeVehicles[player.UserId] then self:DismountVehicle(player) end
        \tlocal template = ServerStorage:FindFirstChild("Vehicles") and ServerStorage.Vehicles:FindFirstChild(vehicleId)
        \tif not template then warn("[VehicleService] No template:", vehicleId); return nil end
        \tlocal vehicle = template:Clone() :: Model
        \tvehicle:PivotTo(spawnCF)
        \tvehicle.Parent = workspace
        \tlocal primary = vehicle.PrimaryPart
        \tif primary then primary:SetNetworkOwner(player) end
        \tvehicle:SetAttribute("OwnerId", player.UserId)
        \tactiveVehicles[player.UserId] = vehicle
        \treturn vehicle
        end

        return VehicleService"""),
    )

    add(
        SEED_TASKS[5]["instruction"],
        "DismountVehicle clears seat weld, resets character state, and destroys vehicle model from activeVehicles table.",
        textwrap.dedent("""\
        --!strict
        local VehicleService = {}
        local activeVehicles: {[number]: Model} = {}

        function VehicleService:DismountVehicle(player: Player): boolean
        \tlocal vehicle = activeVehicles[player.UserId]
        \tif not vehicle then return false end
        \tlocal seat = vehicle:FindFirstChildWhichIsA("VehicleSeat") :: VehicleSeat?
        \tif seat then seat:FindFirstChildOfClass("Weld")?.Destroy() end
        \tlocal char = player.Character
        \tif char then
        \t\tlocal root = char:FindFirstChild("HumanoidRootPart") :: BasePart?
        \t\tif root then root.CFrame = vehicle:GetPivot() * CFrame.new(5, 3, 0) end
        \tend
        \tvehicle:Destroy()
        \tactiveVehicles[player.UserId] = nil
        \treturn true
        end

        return VehicleService"""),
    )

    add(
        SEED_TASKS[6]["instruction"],
        "Batch events into a buffer, flush with HttpService:PostAsync on a timer, handle failures gracefully.",
        textwrap.dedent("""\
        --!strict
        local HttpService = game:GetService("HttpService")
        local RunService = game:GetService("RunService")
        local TelemetryService = {}
        local buffer: {{[string]: any}} = {}
        local FLUSH_INTERVAL = 30
        local ENDPOINT = "https://telemetry.vertigo.game/v1/events"

        function TelemetryService:Init()
        \ttask.spawn(function()
        \t\twhile true do
        \t\t\ttask.wait(FLUSH_INTERVAL)
        \t\t\tself:Flush()
        \t\tend
        \tend)
        end

        function TelemetryService:Record(event: {[string]: any})
        \tevent.timestamp = os.time()
        \ttable.insert(buffer, event)
        end

        function TelemetryService:Flush()
        \tif #buffer == 0 then return end
        \tlocal batch = buffer
        \tbuffer = {}
        \tlocal ok, err = pcall(HttpService.PostAsync, HttpService, ENDPOINT, HttpService:JSONEncode(batch))
        \tif not ok then warn("[Telemetry] Flush failed:", err) end
        end

        return TelemetryService"""),
    )

    add(
        SEED_TASKS[7]["instruction"],
        "Simple method that pushes a structured traversal event with ability name, duration, and distance into the telemetry buffer.",
        textwrap.dedent("""\
        --!strict
        local TelemetryService = {}

        function TelemetryService:RecordTraversal(player: Player, ability: string, duration: number, distance: number)
        \tself:Record({
        \t\ttype = "traversal",
        \t\tuserId = player.UserId,
        \t\tability = ability,
        \t\tduration = math.round(duration * 100) / 100,
        \t\tdistance = math.round(distance * 10) / 10,
        \t})
        end

        return TelemetryService"""),
    )

    add(
        SEED_TASKS[8]["instruction"],
        "Check player Y position against a sorted zone boundary table. Binary search or linear scan since there are few zones.",
        textwrap.dedent("""\
        --!strict
        local Shared = game:GetService("ReplicatedStorage"):WaitForChild("Shared")
        local Config = require(Shared.Config.Zones)
        local ZoneService = {}

        function ZoneService:GetActiveZone(player: Player): string?
        \tlocal char = player.Character
        \tif not char then return nil end
        \tlocal root = char:FindFirstChild("HumanoidRootPart") :: BasePart?
        \tif not root then return nil end
        \tlocal y = root.Position.Y
        \tfor _, zone in Config.Zones do
        \t\tif y >= zone.yMin and y < zone.yMax then return zone.name end
        \tend
        \treturn "void"
        end

        return ZoneService"""),
    )

    add(
        SEED_TASKS[9]["instruction"],
        "Check reward exists, player proximity to reward object, not already claimed via profile flag, then grant and mark claimed.",
        textwrap.dedent("""\
        --!strict
        local RewardService = {}
        local MAX_CLAIM_DIST = 20

        function RewardService:ClaimReward(player: Player, rewardId: string, rewardPart: BasePart, profile: {[string]: any}): (boolean, string?)
        \tlocal claimed: {string} = profile.claimedRewards or {}
        \tif table.find(claimed, rewardId) then return false, "already_claimed" end
        \tlocal char = player.Character
        \tif not char then return false, "no_character" end
        \tlocal root = char:FindFirstChild("HumanoidRootPart") :: BasePart?
        \tif not root then return false, "no_root" end
        \tif (root.Position - rewardPart.Position).Magnitude > MAX_CLAIM_DIST then
        \t\treturn false, "too_far"
        \tend
        \ttable.insert(claimed, rewardId)
        \tprofile.claimedRewards = claimed
        \tprofile.coins = (profile.coins or 0) + 100
        \treturn true, nil
        end

        return RewardService"""),
    )

    # -- Physics --
    add(
        SEED_TASKS[10]["instruction"],
        "Critically-damped spring: x'' = -2*omega*x' - omega^2*(x - target). Use @native for Heartbeat usage.",
        textwrap.dedent("""\
        --!strict
        export type Spring = {pos: number, vel: number, target: number, omega: number}

        local SpringSolver = {}

        @native
        function SpringSolver.step(s: Spring, dt: number): (number, number)
        \tlocal delta = s.pos - s.target
        \tlocal omega = s.omega
        \tlocal exp = math.exp(-omega * dt)
        \tlocal newPos = s.target + (delta + (s.vel + omega * delta) * dt) * exp
        \tlocal newVel = (s.vel * (1 - omega * dt) - omega * omega * delta * dt) * exp
        \treturn newPos, newVel
        end

        function SpringSolver.create(pos: number, target: number, omega: number?): Spring
        \treturn {pos = pos, vel = 0, target = target, omega = omega or 12}
        end

        return SpringSolver"""),
    )

    add(
        SEED_TASKS[11]["instruction"],
        "Verlet integration: store current and previous positions, solve distance constraints iteratively, @native for perf.",
        textwrap.dedent("""\
        --!strict
        export type RopePoint = {pos: vector, prev: vector, locked: boolean}
        local VerletRope = {}
        local ITERATIONS = 4

        function VerletRope.create(a: vector, b: vector, segments: number): {RopePoint}
        \tlocal pts = table.create(segments + 1)
        \tfor i = 0, segments do
        \t\tlocal t = i / segments
        \t\tlocal p = vector.create(math.lerp(a.x, b.x, t), math.lerp(a.y, b.y, t), math.lerp(a.z, b.z, t))
        \t\tpts[i + 1] = {pos = p, prev = p, locked = i == 0 or i == segments}
        \tend
        \treturn pts
        end

        @native
        function VerletRope.step(pts: {RopePoint}, gravity: vector, dt: number, segLen: number)
        \tfor _, p in pts do
        \t\tif not p.locked then
        \t\t\tlocal vel = p.pos - p.prev
        \t\t\tp.prev = p.pos
        \t\t\tp.pos = p.pos + vel + gravity * (dt * dt)
        \t\tend
        \tend
        \tfor _ = 1, ITERATIONS do
        \t\tfor i = 1, #pts - 1 do
        \t\t\tlocal a, b = pts[i], pts[i + 1]
        \t\t\tlocal delta = b.pos - a.pos
        \t\t\tlocal dist = vector.magnitude(delta)
        \t\t\tif dist > 0 then
        \t\t\t\tlocal correction = delta * ((dist - segLen) / dist * 0.5)
        \t\t\t\tif not a.locked then a.pos += correction end
        \t\t\t\tif not b.locked then b.pos -= correction end
        \t\t\tend
        \t\tend
        \tend
        end

        return VerletRope"""),
    )

    add(
        SEED_TASKS[12]["instruction"],
        "2D grid of wind vectors, sample with bilinear interpolation, update with curl noise for organic flow.",
        textwrap.dedent("""\
        --!strict
        local FlowField = {}
        FlowField.__index = FlowField

        export type Field = typeof(setmetatable({} :: {grid: {{vector}}, size: number, cellSize: number}, FlowField))

        function FlowField.new(size: number, cellSize: number): Field
        \tlocal grid = table.create(size)
        \tfor i = 1, size do
        \t\tlocal row = table.create(size)
        \t\tfor j = 1, size do row[j] = vector.zero end
        \t\tgrid[i] = row
        \tend
        \treturn setmetatable({grid = grid, size = size, cellSize = cellSize}, FlowField)
        end

        @native
        function FlowField.sample(self: Field, worldX: number, worldZ: number): vector
        \tlocal gx = math.clamp(worldX / self.cellSize + self.size / 2, 1, self.size - 1)
        \tlocal gz = math.clamp(worldZ / self.cellSize + self.size / 2, 1, self.size - 1)
        \tlocal ix, iz = math.floor(gx), math.floor(gz)
        \tlocal fx, fz = gx - ix, gz - iz
        \tlocal a = vector.create(math.lerp(self.grid[iz][ix].x, self.grid[iz][ix+1].x, fx),0,
        \t\tmath.lerp(self.grid[iz][ix].z, self.grid[iz][ix+1].z, fx))
        \tlocal b = vector.create(math.lerp(self.grid[iz+1][ix].x, self.grid[iz+1][ix+1].x, fx),0,
        \t\tmath.lerp(self.grid[iz+1][ix].z, self.grid[iz+1][ix+1].z, fx))
        \treturn vector.create(math.lerp(a.x,b.x,fz), 0, math.lerp(a.z,b.z,fz))
        end

        return FlowField"""),
    )

    add(
        SEED_TASKS[13]["instruction"],
        "v_new = v * exp(-damping * dt). Framerate-independent because we use continuous exponential decay.",
        textwrap.dedent("""\
        --!strict
        local Damping = {}

        @native
        function Damping.apply(velocity: vector, damping: number, dt: number): vector
        \tlocal factor = math.exp(-damping * dt)
        \treturn velocity * factor
        end

        @native
        function Damping.applyScalar(value: number, damping: number, dt: number): number
        \treturn value * math.exp(-damping * dt)
        end

        return Damping"""),
    )

    add(
        SEED_TASKS[14]["instruction"],
        "Standard kinematic equations: p(t) = p0 + v0*t + 0.5*g*t^2. Sample N points for the arc.",
        textwrap.dedent("""\
        --!strict
        local Trajectory = {}
        local GRAVITY = vector.create(0, -196.2, 0)

        @native
        function Trajectory.predict(origin: vector, velocity: vector, steps: number, dt: number): {vector}
        \tlocal points = table.create(steps)
        \tfor i = 1, steps do
        \t\tlocal t = i * dt
        \t\tpoints[i] = origin + velocity * t + GRAVITY * (0.5 * t * t)
        \tend
        \treturn points
        end

        function Trajectory.timeToGround(originY: number, velY: number, groundY: number): number?
        \tlocal a = 0.5 * GRAVITY.y
        \tlocal b = velY
        \tlocal c = originY - groundY
        \tlocal disc = b * b - 4 * a * c
        \tif disc < 0 then return nil end
        \tlocal t = (-b - math.sqrt(disc)) / (2 * a)
        \treturn if t > 0 then t else nil
        end

        return Trajectory"""),
    )

    add(
        SEED_TASKS[15]["instruction"],
        "Swept sphere = raycast with a radius. Use workspace:Spherecast for Roblox implementation.",
        textwrap.dedent("""\
        --!strict
        local SweptSphere = {}
        local HOOK_RADIUS = 0.5

        function SweptSphere.cast(origin: Vector3, direction: Vector3, radius: number?): RaycastResult?
        \tlocal r = radius or HOOK_RADIUS
        \tlocal params = RaycastParams.new()
        \tparams.FilterType = Enum.RaycastFilterType.Exclude
        \tparams.FilterDescendantsInstances = {}
        \treturn workspace:Spherecast(origin, r, direction, params)
        end

        function SweptSphere.castToAnchor(origin: Vector3, target: Vector3): (boolean, RaycastResult?)
        \tlocal dir = target - origin
        \tlocal result = SweptSphere.cast(origin, dir, HOOK_RADIUS)
        \tif not result then return false, nil end
        \tlocal hitAnchor = result.Instance:HasTag("GrappleAnchor")
        \treturn hitAnchor, result
        end

        return SweptSphere"""),
    )

    add(
        SEED_TASKS[16]["instruction"],
        "CFrame already does slerp via CFrame:Lerp, but for raw quaternion control use manual nlerp for speed.",
        textwrap.dedent("""\
        --!strict
        local QuatUtil = {}

        @native
        function QuatUtil.slerp(a: CFrame, b: CFrame, t: number): CFrame
        \treturn a:Lerp(b, t)
        end

        @native
        function QuatUtil.smoothRotation(current: CFrame, target: CFrame, speed: number, dt: number): CFrame
        \tlocal alpha = 1 - math.exp(-speed * dt)
        \treturn current:Lerp(target, alpha)
        end

        function QuatUtil.lookRotation(direction: Vector3, up: Vector3?): CFrame
        \treturn CFrame.lookAt(Vector3.zero, direction, up or Vector3.yAxis)
        end

        return QuatUtil"""),
    )

    add(
        SEED_TASKS[17]["instruction"],
        "Buoyancy force = density * submerged_fraction * gravity. Estimate submersion from Y overlap with water surface.",
        textwrap.dedent("""\
        --!strict
        local Buoyancy = {}
        local WATER_DENSITY = 1.2
        local GRAVITY = 196.2

        @native
        function Buoyancy.computeForce(partPos: vector, partSize: vector, waterSurfaceY: number): vector
        \tlocal halfH = partSize.y * 0.5
        \tlocal bottom = partPos.y - halfH
        \tlocal top = partPos.y + halfH
        \tif bottom >= waterSurfaceY then return vector.zero end
        \tlocal submerged = math.clamp((waterSurfaceY - bottom) / (top - bottom), 0, 1)
        \tlocal volume = partSize.x * partSize.y * partSize.z
        \tlocal force = WATER_DENSITY * volume * submerged * GRAVITY
        \treturn vector.create(0, force, 0)
        end

        return Buoyancy"""),
    )

    add(
        SEED_TASKS[18]["instruction"],
        "Clamp angular velocity magnitude while preserving direction. Simple vector operation with @native.",
        textwrap.dedent("""\
        --!strict
        local AngularClamp = {}

        @native
        function AngularClamp.clamp(angVel: vector, maxRad: number): vector
        \tlocal mag = vector.magnitude(angVel)
        \tif mag <= maxRad or mag < 1e-6 then return angVel end
        \treturn angVel * (maxRad / mag)
        end

        return AngularClamp"""),
    )

    add(
        SEED_TASKS[19]["instruction"],
        "PID = P*error + I*integral + D*derivative. Clamp integral to prevent windup. Use for hover altitude.",
        textwrap.dedent("""\
        --!strict
        export type PID = {kP: number, kI: number, kD: number, integral: number, prevError: number, maxIntegral: number}

        local PIDController = {}

        function PIDController.create(kP: number, kI: number, kD: number): PID
        \treturn {kP = kP, kI = kI, kD = kD, integral = 0, prevError = 0, maxIntegral = 50}
        end

        @native
        function PIDController.update(pid: PID, error: number, dt: number): number
        \tpid.integral = math.clamp(pid.integral + error * dt, -pid.maxIntegral, pid.maxIntegral)
        \tlocal derivative = (error - pid.prevError) / math.max(dt, 1e-6)
        \tpid.prevError = error
        \treturn pid.kP * error + pid.kI * pid.integral + pid.kD * derivative
        end

        return PIDController"""),
    )

    # -- Builders --
    add(
        SEED_TASKS[20]["instruction"],
        "Create ring of platforms at a Y-level. Use math.cos/sin for positions, randomize gaps by skipping segments.",
        textwrap.dedent("""\
        --!strict
        local CollectionService = game:GetService("CollectionService")
        local ZoneBuilder = {}
        local RNG = Random.new(42)

        function ZoneBuilder:BuildRing(center: Vector3, radius: number, platformCount: number, gapChance: number): {BasePart}
        \tlocal parts = {}
        \tlocal angleStep = math.pi * 2 / platformCount
        \tfor i = 0, platformCount - 1 do
        \t\tif RNG:NextNumber() < gapChance then continue end
        \t\tlocal angle = i * angleStep
        \t\tlocal pos = center + Vector3.new(math.cos(angle) * radius, 0, math.sin(angle) * radius)
        \t\tlocal part = Instance.new("Part")
        \t\tpart.Size = Vector3.new(12, 2, 8)
        \t\tpart.CFrame = CFrame.lookAt(pos, center) * CFrame.Angles(0, math.pi, 0)
        \t\tpart.Anchored = true
        \t\tpart.Material = Enum.Material.Slate
        \t\tpart.Parent = workspace
        \t\tCollectionService:AddTag(part, "ZonePlatform")
        \t\ttable.insert(parts, part)
        \tend
        \treturn parts
        end

        return ZoneBuilder"""),
    )

    add(
        SEED_TASKS[21]["instruction"],
        "Raycast along cave wall normals to place crystals. Tag with BloomCrystal for CollectionService discovery.",
        textwrap.dedent("""\
        --!strict
        local CollectionService = game:GetService("CollectionService")
        local CavernBuilder = {}
        local RNG = Random.new(7)

        function CavernBuilder:PlaceCrystals(wallParts: {BasePart}, density: number): number
        \tlocal placed = 0
        \tfor _, wall in wallParts do
        \t\tlocal cf = wall.CFrame
        \t\tlocal normal = cf.LookVector
        \t\tfor _ = 1, density do
        \t\t\tlocal offset = Vector3.new(RNG:NextNumber(-1,1), RNG:NextNumber(-1,1), 0) * wall.Size * 0.4
        \t\t\tlocal origin = cf:PointToWorldSpace(offset) + normal * 2
        \t\t\tlocal hit = workspace:Raycast(origin, -normal * 5)
        \t\t\tif hit then
        \t\t\t\tlocal crystal = Instance.new("Part")
        \t\t\t\tcrystal.Size = Vector3.new(1, RNG:NextNumber(2, 5), 1)
        \t\t\t\tcrystal.CFrame = CFrame.lookAt(hit.Position, hit.Position + hit.Normal)
        \t\t\t\tcrystal.Material = Enum.Material.Neon
        \t\t\t\tcrystal.Color = Color3.fromHSV(0.55, 0.8, 1)
        \t\t\t\tcrystal.Anchored = true
        \t\t\t\tcrystal.Parent = workspace
        \t\t\t\tCollectionService:AddTag(crystal, "BloomCrystal")
        \t\t\t\tplaced += 1
        \t\t\tend
        \t\tend
        \tend
        \treturn placed
        end

        return CavernBuilder"""),
    )

    add(
        SEED_TASKS[22]["instruction"],
        "Distribute anchors along vertical surfaces using raycasts downward from cliff top, tag with GrappleAnchor.",
        textwrap.dedent("""\
        --!strict
        local CollectionService = game:GetService("CollectionService")
        local AnchorPlacer = {}

        function AnchorPlacer:PlaceAlongCliff(topEdge: Vector3, bottomY: number, spacing: number, cliffNormal: Vector3): {BasePart}
        \tlocal anchors = {}
        \tlocal height = topEdge.Y - bottomY
        \tlocal count = math.floor(height / spacing)
        \tfor i = 0, count do
        \t\tlocal y = topEdge.Y - i * spacing
        \t\tlocal origin = Vector3.new(topEdge.X, y, topEdge.Z) + cliffNormal * 5
        \t\tlocal hit = workspace:Raycast(origin, -cliffNormal * 10)
        \t\tif hit then
        \t\t\tlocal anchor = Instance.new("Part")
        \t\t\tanchor.Size = Vector3.new(2, 2, 2)
        \t\t\tanchor.Shape = Enum.PartType.Ball
        \t\t\tanchor.Position = hit.Position + hit.Normal * 0.5
        \t\t\tanchor.Anchored = true
        \t\t\tanchor.Parent = workspace
        \t\t\tCollectionService:AddTag(anchor, "GrappleAnchor")
        \t\t\ttable.insert(anchors, anchor)
        \t\tend
        \tend
        \treturn anchors
        end

        return AnchorPlacer"""),
    )

    add(
        SEED_TASKS[23]["instruction"],
        "Create an ellipsoid of terrain voxels. Use Terrain:FillBall at sampled points inside the ellipsoid boundary.",
        textwrap.dedent("""\
        --!strict
        local IslandBuilder = {}

        function IslandBuilder:Build(center: Vector3, radii: Vector3, material: Enum.Material?): number
        \tlocal mat = material or Enum.Material.Rock
        \tlocal terrain = workspace.Terrain
        \tlocal step = 4
        \tlocal filled = 0
        \tfor x = -radii.X, radii.X, step do
        \t\tfor y = -radii.Y, radii.Y, step do
        \t\t\tfor z = -radii.Z, radii.Z, step do
        \t\t\t\tlocal nx = x / radii.X
        \t\t\t\tlocal ny = y / radii.Y
        \t\t\t\tlocal nz = z / radii.Z
        \t\t\t\tif nx*nx + ny*ny + nz*nz <= 1 then
        \t\t\t\t\tterrain:FillBall(center + Vector3.new(x, y, z), step * 0.6, mat)
        \t\t\t\t\tfilled += 1
        \t\t\t\tend
        \t\t\tend
        \t\tend
        \tend
        \treturn filled
        end

        return IslandBuilder"""),
    )

    add(
        SEED_TASKS[24]["instruction"],
        "Create parts between two points, segmented with HingeConstraints for swaying motion.",
        textwrap.dedent("""\
        --!strict
        local BridgeBuilder = {}

        function BridgeBuilder:Build(startPos: Vector3, endPos: Vector3, segments: number): Model
        \tlocal model = Instance.new("Model")
        \tmodel.Name = "Bridge"
        \tlocal dir = (endPos - startPos)
        \tlocal segLen = dir.Magnitude / segments
        \tlocal unitDir = dir.Unit
        \tlocal prevPart: BasePart? = nil
        \tfor i = 0, segments - 1 do
        \t\tlocal pos = startPos + unitDir * (i * segLen + segLen * 0.5)
        \t\tlocal plank = Instance.new("Part")
        \t\tplank.Size = Vector3.new(6, 0.4, segLen * 0.9)
        \t\tplank.CFrame = CFrame.lookAt(pos, pos + unitDir)
        \t\tplank.Anchored = i == 0 or i == segments - 1
        \t\tplank.Material = Enum.Material.Wood
        \t\tplank.Parent = model
        \t\tif prevPart and not plank.Anchored then
        \t\t\tlocal hinge = Instance.new("HingeConstraint")
        \t\t\tlocal a0 = Instance.new("Attachment"); a0.Parent = prevPart
        \t\t\tlocal a1 = Instance.new("Attachment"); a1.Parent = plank
        \t\t\thinge.Attachment0 = a0; hinge.Attachment1 = a1
        \t\t\thinge.Parent = plank
        \t\tend
        \t\tprevPart = plank
        \tend
        \tmodel.Parent = workspace
        \treturn model
        end

        return BridgeBuilder"""),
    )

    add(
        SEED_TASKS[25]["instruction"],
        "Create cascade of transparent blue parts with ParticleEmitter for mist. Add Sound for ambience.",
        textwrap.dedent("""\
        --!strict
        local WaterfallBuilder = {}

        function WaterfallBuilder:Build(top: Vector3, height: number, width: number): Model
        \tlocal model = Instance.new("Model"); model.Name = "Waterfall"
        \tlocal segments = math.ceil(height / 8)
        \tfor i = 0, segments - 1 do
        \t\tlocal part = Instance.new("Part")
        \t\tpart.Size = Vector3.new(width, 8, 2)
        \t\tpart.Position = top - Vector3.new(0, i * 8 + 4, 0)
        \t\tpart.Transparency = 0.4
        \t\tpart.Color = Color3.fromRGB(100, 160, 220)
        \t\tpart.Material = Enum.Material.Glass
        \t\tpart.Anchored = true
        \t\tpart.CanCollide = false
        \t\tpart.Parent = model
        \t\tlocal emitter = Instance.new("ParticleEmitter")
        \t\temitter.Rate = 30; emitter.Lifetime = NumberRange.new(1, 2)
        \t\temitter.Speed = NumberRange.new(2, 5)
        \t\temitter.Parent = part
        \tend
        \tlocal sound = Instance.new("Sound")
        \tsound.SoundId = "rbxassetid://0" -- placeholder
        \tsound.Looped = true; sound.Volume = 0.5
        \tsound.Parent = model
        \tmodel.Parent = workspace
        \treturn model
        end

        return WaterfallBuilder"""),
    )

    add(
        SEED_TASKS[26]["instruction"],
        "Connect points with beam-like mesh parts, tag surfaces for wallrun detection.",
        textwrap.dedent("""\
        --!strict
        local CollectionService = game:GetService("CollectionService")
        local VineBuilder = {}

        function VineBuilder:Connect(points: {Vector3}, thickness: number): {BasePart}
        \tlocal vines = {}
        \tfor i = 1, #points - 1 do
        \t\tlocal a, b = points[i], points[i + 1]
        \t\tlocal mid = (a + b) / 2
        \t\tlocal dist = (b - a).Magnitude
        \t\tlocal vine = Instance.new("Part")
        \t\tvine.Size = Vector3.new(thickness, thickness, dist)
        \t\tvine.CFrame = CFrame.lookAt(mid, b)
        \t\tvine.Shape = Enum.PartType.Cylinder
        \t\tvine.Material = Enum.Material.Grass
        \t\tvine.Color = Color3.fromRGB(50, 100, 30)
        \t\tvine.Anchored = true
        \t\tvine.Parent = workspace
        \t\tCollectionService:AddTag(vine, "WallrunSurface")
        \t\ttable.insert(vines, vine)
        \tend
        \treturn vines
        end

        return VineBuilder"""),
    )

    add(
        SEED_TASKS[27]["instruction"],
        "Place scaled mushroom models at positions, tag as BouncePad for gameplay interaction.",
        textwrap.dedent("""\
        --!strict
        local CollectionService = game:GetService("CollectionService")
        local ServerStorage = game:GetService("ServerStorage")
        local MushroomBuilder = {}
        local RNG = Random.new(99)

        function MushroomBuilder:Build(positions: {Vector3}): {Model}
        \tlocal template = ServerStorage:FindFirstChild("MushroomTemplate") :: Model?
        \tif not template then warn("No MushroomTemplate"); return {} end
        \tlocal mushrooms = table.create(#positions)
        \tfor _, pos in positions do
        \t\tlocal m = template:Clone() :: Model
        \t\tlocal scale = RNG:NextNumber(0.6, 1.8)
        \t\tm:ScaleTo(scale)
        \t\tm:PivotTo(CFrame.new(pos) * CFrame.Angles(0, RNG:NextNumber(0, math.pi * 2), 0))
        \t\tm.Parent = workspace
        \t\tlocal cap = m:FindFirstChild("Cap") :: BasePart?
        \t\tif cap then CollectionService:AddTag(cap, "BouncePad") end
        \t\ttable.insert(mushrooms, m)
        \tend
        \treturn mushrooms
        end

        return MushroomBuilder"""),
    )

    add(
        SEED_TASKS[28]["instruction"],
        "Generate broken column with randomized top cut and missing chunks using Part subtraction.",
        textwrap.dedent("""\
        --!strict
        local RuinPillarBuilder = {}
        local RNG = Random.new(13)

        function RuinPillarBuilder:Build(base: Vector3, maxHeight: number, radius: number): Model
        \tlocal model = Instance.new("Model"); model.Name = "RuinPillar"
        \tlocal height = maxHeight * RNG:NextNumber(0.4, 0.9)
        \tlocal pillar = Instance.new("Part")
        \tpillar.Size = Vector3.new(radius * 2, height, radius * 2)
        \tpillar.Position = base + Vector3.new(0, height / 2, 0)
        \tpillar.Shape = Enum.PartType.Cylinder
        \tpillar.CFrame = CFrame.new(pillar.Position) * CFrame.Angles(0, 0, math.rad(90))
        \tpillar.Material = Enum.Material.Limestone
        \tpillar.Anchored = true
        \tpillar.Parent = model
        \tfor _ = 1, RNG:NextInteger(1, 3) do
        \t\tlocal chip = Instance.new("Part")
        \t\tchip.Size = Vector3.new(radius, radius * 0.5, radius)
        \t\tchip.Position = base + Vector3.new(RNG:NextNumber(-radius, radius), RNG:NextNumber(0, height), RNG:NextNumber(-radius, radius))
        \t\tchip.Transparency = 1; chip.Anchored = true; chip.CanCollide = false
        \t\tchip.Parent = model
        \tend
        \tmodel.Parent = workspace
        \treturn model
        end

        return RuinPillarBuilder"""),
    )

    add(
        SEED_TASKS[29]["instruction"],
        "Place checkpoint ring parts at waypoints with numbered BillboardGui labels.",
        textwrap.dedent("""\
        --!strict
        local CollectionService = game:GetService("CollectionService")
        local CheckpointBuilder = {}

        function CheckpointBuilder:Build(waypoints: {Vector3}): {BasePart}
        \tlocal parts = table.create(#waypoints)
        \tfor i, pos in waypoints do
        \t\tlocal ring = Instance.new("Part")
        \t\tring.Size = Vector3.new(10, 10, 1)
        \t\tring.Shape = Enum.PartType.Cylinder
        \t\tring.Position = pos
        \t\tring.Transparency = 0.5
        \t\tring.Color = Color3.fromRGB(255, 200, 50)
        \t\tring.Material = Enum.Material.Neon
        \t\tring.Anchored = true; ring.CanCollide = false
        \t\tring.Parent = workspace
        \t\tring:SetAttribute("CheckpointIndex", i)
        \t\tCollectionService:AddTag(ring, "Checkpoint")
        \t\tlocal bb = Instance.new("BillboardGui")
        \t\tbb.Size = UDim2.fromOffset(60, 30); bb.StudsOffset = Vector3.new(0, 8, 0)
        \t\tbb.Parent = ring
        \t\tlocal label = Instance.new("TextLabel")
        \t\tlabel.Size = UDim2.fromScale(1, 1); label.Text = tostring(i)
        \t\tlabel.BackgroundTransparency = 1; label.TextColor3 = Color3.new(1,1,1)
        \t\tlabel.TextScaled = true; label.Parent = bb
        \t\ttable.insert(parts, ring)
        \tend
        \treturn parts
        end

        return CheckpointBuilder"""),
    )

    # -- Networking --
    add(
        SEED_TASKS[30]["instruction"],
        "Define all remotes in one module, create on server, WaitForChild on client. Type-safe registry table.",
        textwrap.dedent("""\
        --!strict
        local ReplicatedStorage = game:GetService("ReplicatedStorage")
        local RunService = game:GetService("RunService")
        local Remotes = {}

        local REMOTE_EVENTS = table.freeze({"RequestUseAbility","RequestEquipLoadout","RequestClaimReward","RequestSpawnVehicle","RequestDismountVehicle","RequestPuzzleReset","StateSync","EventNotify"})
        local REMOTE_FUNCTIONS = table.freeze({"GetPlayerProfile"})

        function Remotes:Init()
        \tlocal folder = Instance.new("Folder"); folder.Name = "Remotes"
        \tif RunService:IsServer() then
        \t\tfor _, name in REMOTE_EVENTS do
        \t\t\tlocal re = Instance.new("RemoteEvent"); re.Name = name; re.Parent = folder
        \t\tend
        \t\tfor _, name in REMOTE_FUNCTIONS do
        \t\t\tlocal rf = Instance.new("RemoteFunction"); rf.Name = name; rf.Parent = folder
        \t\tend
        \t\tfolder.Parent = ReplicatedStorage
        \telse
        \t\tfolder = ReplicatedStorage:WaitForChild("Remotes")
        \tend
        \tself._folder = folder
        end

        function Remotes:Get(name: string): Instance
        \treturn self._folder:WaitForChild(name)
        end

        return Remotes"""),
    )

    add(
        SEED_TASKS[31]["instruction"],
        "OnServerEvent handler: validate ability ID in registry, check cooldown timestamp, check stamina, then fire.",
        textwrap.dedent("""\
        --!strict
        local Shared = game:GetService("ReplicatedStorage"):WaitForChild("Shared")
        local Config = require(Shared.Config.Abilities)
        local cooldowns: {[number]: {[string]: number}} = {}

        local function onRequestUseAbility(player: Player, abilityId: string)
        \tlocal tuning = Config.Abilities[abilityId]
        \tif not tuning then warn("Invalid ability:", abilityId); return end
        \tlocal now = os.clock()
        \tlocal pcd = cooldowns[player.UserId] or {}
        \tif (pcd[abilityId] or 0) > now then return end
        \tlocal profile = {} -- fetched from DataService in real code
        \tif (profile.stamina or 100) < tuning.staminaCost then return end
        \tpcd[abilityId] = now + tuning.cooldown
        \tcooldowns[player.UserId] = pcd
        \t-- Execute ability logic
        end

        return onRequestUseAbility"""),
    )

    add(
        SEED_TASKS[32]["instruction"],
        "Token bucket: each player gets N tokens, refilled at a rate. Reject calls when bucket empty.",
        textwrap.dedent("""\
        --!strict
        export type Bucket = {tokens: number, lastRefill: number}
        local RateLimiter = {}
        local buckets: {[number]: Bucket} = {}
        local MAX_TOKENS = 10
        local REFILL_RATE = 2 -- tokens per second

        function RateLimiter.check(userId: number): boolean
        \tlocal now = os.clock()
        \tlocal b = buckets[userId]
        \tif not b then
        \t\tbuckets[userId] = {tokens = MAX_TOKENS - 1, lastRefill = now}
        \t\treturn true
        \tend
        \tlocal elapsed = now - b.lastRefill
        \tb.tokens = math.min(MAX_TOKENS, b.tokens + elapsed * REFILL_RATE)
        \tb.lastRefill = now
        \tif b.tokens < 1 then return false end
        \tb.tokens -= 1
        \treturn true
        end

        function RateLimiter.reset(userId: number)
        \tbuckets[userId] = nil
        end

        return RateLimiter"""),
    )

    add(
        SEED_TASKS[33]["instruction"],
        "Validate player has no active vehicle, is near a spawn pad tagged VehicleSpawn, and under vehicle limit.",
        textwrap.dedent("""\
        --!strict
        local CollectionService = game:GetService("CollectionService")
        local MAX_SPAWN_DIST = 15

        local function onRequestSpawnVehicle(player: Player, vehicleId: string)
        \tlocal char = player.Character
        \tif not char then return end
        \tlocal root = char:FindFirstChild("HumanoidRootPart") :: BasePart?
        \tif not root then return end
        \tif char:GetAttribute("ActiveVehicle") then return end
        \tlocal nearPad = false
        \tfor _, pad in CollectionService:GetTagged("VehicleSpawn") do
        \t\tif (pad.Position - root.Position).Magnitude <= MAX_SPAWN_DIST then
        \t\t\tnearPad = true; break
        \t\tend
        \tend
        \tif not nearPad then return end
        \t-- Delegate to VehicleService:SpawnVehicle
        end

        return onRequestSpawnVehicle"""),
    )

    add(
        SEED_TASKS[34]["instruction"],
        "Broadcast player positions as delta from last known. Only send to players within render distance.",
        textwrap.dedent("""\
        --!strict
        local Players = game:GetService("Players")
        local RunService = game:GetService("RunService")
        local RENDER_DIST = 256
        local lastPositions: {[number]: Vector3} = {}

        local function broadcastStateSync(syncRemote: RemoteEvent)
        \tlocal updates: {[number]: Vector3} = {}
        \tfor _, player in Players:GetPlayers() do
        \t\tlocal char = player.Character
        \t\tlocal root = char and char:FindFirstChild("HumanoidRootPart") :: BasePart?
        \t\tif root then
        \t\t\tlocal last = lastPositions[player.UserId]
        \t\t\tif not last or (root.Position - last).Magnitude > 0.5 then
        \t\t\t\tupdates[player.UserId] = root.Position
        \t\t\t\tlastPositions[player.UserId] = root.Position
        \t\t\tend
        \t\tend
        \tend
        \tfor _, receiver in Players:GetPlayers() do
        \t\tlocal rRoot = receiver.Character and receiver.Character:FindFirstChild("HumanoidRootPart")
        \t\tif not rRoot then continue end
        \t\tlocal nearby = {}
        \t\tfor uid, pos in updates do
        \t\t\tif uid ~= receiver.UserId and (pos - rRoot.Position).Magnitude <= RENDER_DIST then
        \t\t\t\tnearby[uid] = pos
        \t\t\tend
        \t\tend
        \t\tif next(nearby) then syncRemote:FireClient(receiver, nearby) end
        \tend
        end

        return broadcastStateSync"""),
    )

    add(
        SEED_TASKS[35]["instruction"],
        "Client wrapper that queues requests with a minimum interval, deduplicates by key.",
        textwrap.dedent("""\
        --!strict
        local RemoteQueue = {}
        RemoteQueue.__index = RemoteQueue
        local MIN_INTERVAL = 0.1

        export type Queue = typeof(setmetatable({} :: {remote: RemoteEvent, lastFire: number, pending: {[string]: any}}, RemoteQueue))

        function RemoteQueue.new(remote: RemoteEvent): Queue
        \treturn setmetatable({remote = remote, lastFire = 0, pending = {}}, RemoteQueue)
        end

        function RemoteQueue.send(self: Queue, key: string, ...: any)
        \tself.pending[key] = table.pack(...)
        \tlocal now = os.clock()
        \tif now - self.lastFire >= MIN_INTERVAL then
        \t\tself:flush()
        \telse
        \t\ttask.delay(MIN_INTERVAL - (now - self.lastFire), function() self:flush() end)
        \tend
        end

        function RemoteQueue.flush(self: Queue)
        \tfor key, args in self.pending do
        \t\tself.remote:FireServer(table.unpack(args))
        \tend
        \tself.pending = {}
        \tself.lastFire = os.clock()
        end

        return RemoteQueue"""),
    )

    add(
        SEED_TASKS[36]["instruction"],
        "Check reward registry, player proximity, and claimed set before granting.",
        textwrap.dedent("""\
        --!strict
        local CollectionService = game:GetService("CollectionService")
        local MAX_DIST = 20

        local function onRequestClaimReward(player: Player, rewardId: string, profile: {[string]: any})
        \tlocal claimed: {string} = profile.claimedRewards or {}
        \tif table.find(claimed, rewardId) then return false end
        \tlocal rewardPart: BasePart? = nil
        \tfor _, part in CollectionService:GetTagged("Reward") do
        \t\tif part:GetAttribute("RewardId") == rewardId then rewardPart = part :: BasePart; break end
        \tend
        \tif not rewardPart then return false end
        \tlocal char = player.Character
        \tlocal root = char and char:FindFirstChild("HumanoidRootPart") :: BasePart?
        \tif not root or (root.Position - rewardPart.Position).Magnitude > MAX_DIST then return false end
        \ttable.insert(claimed, rewardId)
        \tprofile.claimedRewards = claimed
        \treturn true
        end

        return onRequestClaimReward"""),
    )

    add(
        SEED_TASKS[37]["instruction"],
        "Return a sanitized copy of the profile — strip internal fields like _lockId and _saveTime.",
        textwrap.dedent("""\
        --!strict
        local STRIP_KEYS = table.freeze({"_lockId", "_saveTime", "_migrationLog"})

        local function onGetPlayerProfile(player: Player, profile: {[string]: any}): {[string]: any}
        \tlocal sanitized = table.clone(profile)
        \tfor _, key in STRIP_KEYS do
        \t\tsanitized[key] = nil
        \tend
        \treturn sanitized
        end

        return onGetPlayerProfile"""),
    )

    add(
        SEED_TASKS[38]["instruction"],
        "Fire EventNotify to specific players for zone-enter and ability-unlock events.",
        textwrap.dedent("""\
        --!strict
        local EventDispatcher = {}

        function EventDispatcher:NotifyZoneEnter(player: Player, zoneName: string, notifyRemote: RemoteEvent)
        \tnotifyRemote:FireClient(player, {type = "zone_enter", zone = zoneName, timestamp = os.clock()})
        end

        function EventDispatcher:NotifyAbilityUnlock(player: Player, abilityId: string, notifyRemote: RemoteEvent)
        \tnotifyRemote:FireClient(player, {type = "ability_unlock", ability = abilityId, timestamp = os.clock()})
        end

        function EventDispatcher:BroadcastEvent(message: string, notifyRemote: RemoteEvent)
        \tnotifyRemote:FireAllClients({type = "broadcast", message = message, timestamp = os.clock()})
        end

        return EventDispatcher"""),
    )

    add(
        SEED_TASKS[39]["instruction"],
        "Validate ban data, log the action, then Player:Kick with a formatted reason.",
        textwrap.dedent("""\
        --!strict
        local Players = game:GetService("Players")
        local Enforcement = {}

        function Enforcement:KickPlayer(userId: number, reason: string, duration: number?)
        \tlocal player = Players:GetPlayerByUserId(userId)
        \tif not player then return end
        \tlocal msg = string.format("[Vertigo] You have been removed. Reason: %s", reason)
        \tif duration then
        \t\tmsg ..= string.format(" Duration: %d minutes.", math.ceil(duration / 60))
        \tend
        \twarn(string.format("[Enforcement] Kicking %s (%d): %s", player.Name, userId, reason))
        \tplayer:Kick(msg)
        end

        return Enforcement"""),
    )

    # -- Config --
    add(
        SEED_TASKS[40]["instruction"],
        "Frozen tuning table with grapple parameters.",
        textwrap.dedent("""\
        --!strict
        local GrappleTuning = table.freeze({
        \tmaxRange = 120,
        \tcooldown = 1.5,
        \treelSpeed = 80,
        \tstaminaCost = 15,
        \thookProjectileSpeed = 200,
        \tautoReleaseDistance = 3,
        })

        local Abilities = table.freeze({ability_grapple_v1 = {staminaCost = 15, cooldown = 1.5, requireGrounded = false}})
        return {Abilities = Abilities, GrappleTuning = GrappleTuning}"""),
    )

    add(
        SEED_TASKS[41]["instruction"],
        "Frozen glide tuning table.",
        textwrap.dedent("""\
        --!strict
        local GlideTuning = table.freeze({
        \tliftCoefficient = 0.35,
        \tdragCoefficient = 0.12,
        \tturnRate = 2.5,
        \tmaxSpeed = 90,
        \tstaminaDrainPerSecond = 8,
        \tminAltitudeToActivate = 10,
        })
        return {GlideTuning = GlideTuning}"""),
    )

    add(
        SEED_TASKS[42]["instruction"],
        "Frozen dirt bike tuning table.",
        textwrap.dedent("""\
        --!strict
        local DirtBikeTuning = table.freeze({
        \ttopSpeed = 110,
        \tacceleration = 45,
        \tbrakeForce = 80,
        \tmaxLeanAngle = 35,
        \tturnSpeed = 3.2,
        \tsuspensionStiffness = 50,
        \tsuspensionDamping = 4,
        })

        local Vehicles = table.freeze({vehicle_dirtbike_v1 = DirtBikeTuning})
        return {Vehicles = Vehicles, DirtBikeTuning = DirtBikeTuning}"""),
    )

    add(
        SEED_TASKS[43]["instruction"],
        "Frozen hover bike tuning with PID gains.",
        textwrap.dedent("""\
        --!strict
        local HoverBikeTuning = table.freeze({
        \thoverHeight = 6,
        \tkP = 120,
        \tkI = 5,
        \tkD = 30,
        \tmaxTiltDegrees = 25,
        \tboostMultiplier = 1.8,
        \tboostDuration = 2.5,
        \tboostCooldown = 8,
        \tmaxSpeed = 80,
        })
        return {HoverBikeTuning = HoverBikeTuning}"""),
    )

    add(
        SEED_TASKS[44]["instruction"],
        "Frozen checkpoint definitions with positions, required abilities, and reward tiers.",
        textwrap.dedent("""\
        --!strict
        local Checkpoints = table.freeze({
        \t{id = "cp_01", position = Vector3.new(0, 30, 0), requiredAbility = nil, rewardTier = 1},
        \t{id = "cp_02", position = Vector3.new(50, 60, -20), requiredAbility = "ability_grapple_v1", rewardTier = 1},
        \t{id = "cp_03", position = Vector3.new(100, 90, -50), requiredAbility = "ability_grapple_v1", rewardTier = 2},
        \t{id = "cp_04", position = Vector3.new(80, 130, 10), requiredAbility = "ability_glide_v1", rewardTier = 2},
        \t{id = "cp_05", position = Vector3.new(30, 180, -80), requiredAbility = "ability_wallrun_v1", rewardTier = 3},
        })
        return {Checkpoints = Checkpoints}"""),
    )

    add(
        SEED_TASKS[45]["instruction"],
        "Zone definitions with Y ranges, colors, fog, and music.",
        textwrap.dedent("""\
        --!strict
        local Zones = table.freeze({
        \t{name = "Abyss", yMin = -120, yMax = -30, ambient = Color3.fromRGB(10, 5, 20), fogDensity = 0.08, musicId = "rbxassetid://0"},
        \t{name = "DeepCaverns", yMin = -30, yMax = 20, ambient = Color3.fromRGB(20, 30, 50), fogDensity = 0.05, musicId = "rbxassetid://0"},
        \t{name = "Surface", yMin = 20, yMax = 80, ambient = Color3.fromRGB(120, 140, 160), fogDensity = 0.01, musicId = "rbxassetid://0"},
        \t{name = "Canopy", yMin = 80, yMax = 150, ambient = Color3.fromRGB(60, 100, 40), fogDensity = 0.02, musicId = "rbxassetid://0"},
        \t{name = "SkyRing", yMin = 150, yMax = 210, ambient = Color3.fromRGB(180, 200, 240), fogDensity = 0.005, musicId = "rbxassetid://0"},
        })
        return {Zones = Zones}"""),
    )

    add(
        SEED_TASKS[46]["instruction"],
        "Frozen wallrun tuning table.",
        textwrap.dedent("""\
        --!strict
        local WallrunTuning = table.freeze({
        \tmaxDuration = 2.0,
        \tminEntrySpeed = 18,
        \tgravityScale = 0.3,
        \texitLaunchSpeed = 45,
        \texitLaunchAngle = 40,
        \tstaminaCost = 20,
        \tcooldown = 0.5,
        })
        return {WallrunTuning = WallrunTuning}"""),
    )

    add(
        SEED_TASKS[47]["instruction"],
        "Frozen slide tuning table.",
        textwrap.dedent("""\
        --!strict
        local SlideTuning = table.freeze({
        \tspeedBoost = 1.6,
        \tfriction = 0.92,
        \tmaxDuration = 1.8,
        \tcooldown = 1.0,
        \tminEntrySpeed = 12,
        \tstaminaCost = 10,
        })
        return {SlideTuning = SlideTuning}"""),
    )

    add(
        SEED_TASKS[48]["instruction"],
        "Frozen combat tuning table.",
        textwrap.dedent("""\
        --!strict
        local CombatTuning = table.freeze({
        \tbaseDamage = 25,
        \thitRange = 8,
        \tknockbackForce = 50,
        \tiFrameDuration = 0.6,
        \tattackCooldown = 0.8,
        \tcritMultiplier = 1.5,
        \tcritChance = 0.1,
        })
        return {CombatTuning = CombatTuning}"""),
    )

    add(
        SEED_TASKS[49]["instruction"],
        "Frozen stamina tuning with per-ability costs.",
        textwrap.dedent("""\
        --!strict
        local StaminaTuning = table.freeze({
        \tmaxStamina = 100,
        \tregenRate = 12,
        \tregenDelay = 2.0,
        \tcosts = table.freeze({
        \t\tgrapple = 15,
        \t\tglide = 8, -- per second
        \t\twallrun = 20,
        \t\tslide = 10,
        \t\tairDash = 25,
        \t}),
        })
        return {StaminaTuning = StaminaTuning}"""),
    )

    return C


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    "service": "service",
    "physics": "physics",
    "builder": "builder",
    "networking": "networking",
    "config": "config",
}


def build_example(seed: dict, completions: dict[str, dict]) -> dict | None:
    key = seed["instruction"][:40]
    comp = completions.get(key)
    if not comp:
        return None

    thinking = comp["thinking"]
    code = comp["code"]
    assistant_content = f"<think>\n{thinking}\n</think>\n\n```luau\n{code}\n```"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": seed["instruction"]},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": "synthetic",
        "category": CATEGORY_MAP.get(seed["category"], seed["category"]),
        "difficulty": seed.get("difficulty", 3),
        "has_reasoning": True,
        "verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Vertigo training data")
    parser.add_argument("--count", type=int, default=50, help="Number of examples to generate (max 50)")
    args = parser.parse_args()

    count = min(args.count, len(SEED_TASKS))
    completions = _completions()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(OUTPUT, "w") as f:
        for seed in SEED_TASKS[:count]:
            example = build_example(seed, completions)
            if example:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1

    print(f"[generate_synthetic] Wrote {written} examples to {OUTPUT}")


if __name__ == "__main__":
    main()
