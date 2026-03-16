#!/usr/bin/env python3
from __future__ import annotations

"""
Extract training data from Vertigo's Luau codebase into GRANULAR per-category files.

Produces separate datasets for each architectural layer:
  - codebase_services.jsonl    (37 server services)
  - codebase_controllers.jsonl (135+ client controllers)
  - codebase_builders.jsonl    (56 world builders)
  - codebase_config.jsonl      (33+ config modules)
  - codebase_physics.jsonl     (physics/util modules)
  - codebase_networking.jsonl  (remotes, types, net layer)
  - codebase_patterns.jsonl    (isolated architecture patterns)
  - codebase_other.jsonl       (everything else)

Each file is independently usable and combinable into a master dataset.
Every .luau file in src/ is covered — nothing is skipped.

Output: data/raw/codebase_*.jsonl
"""

import json
import re
from pathlib import Path
from collections import Counter

VERTIGO_SRC = Path(__file__).resolve().parent.parent.parent / "src"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

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
# Category routing — maps path prefixes to output file + instruction templates
# ---------------------------------------------------------------------------

CATEGORIES = {
    "Server/Services": {
        "file": "codebase_services.jsonl",
        "category": "service",
        "templates": [
            "Write a Vertigo server service that handles {desc}. Include Init/Start lifecycle, "
            "remote event handling, and server-authoritative validation.",
            "Implement the {desc} server service following Vertigo's two-phase boot pattern. "
            "Include typed remotes, rate limiting, and proper cleanup on PlayerRemoving.",
            "Create the server-side {desc} service with DataStore integration, "
            "per-player state tracking, and telemetry hooks.",
        ],
    },
    "Client/Controllers": {
        "file": "codebase_controllers.jsonl",
        "category": "controller",
        "templates": [
            "Write a Vertigo client controller for {desc}. Use RunService connections, "
            "Signal-based events, and proper cleanup with Trove.",
            "Implement the client-side {desc} controller with input handling, VFX management, and camera integration.",
            "Create a controller that manages {desc} on the client, connecting to "
            "server state via RemoteEvents and local prediction.",
        ],
    },
    "Server/World/Builders": {
        "file": "codebase_builders.jsonl",
        "category": "builder",
        "templates": [
            "Write a procedural world builder that generates {desc}. Use deterministic "
            "RNG (Random.new(seed)), CollectionService tags, and WeldConstraints.",
            "Create a zone builder for {desc} following Vertigo's builder pattern "
            "with BuildOptions type, color palettes, and part_builder shorthand.",
            "Implement the {desc} builder that procedurally generates geometry, "
            "assigns materials and lighting, and tags interactive elements.",
        ],
    },
    "Shared/Config": {
        "file": "codebase_config.jsonl",
        "category": "config",
        "templates": [
            "Write a data-driven configuration module for {desc}. Use named exports, "
            "table.freeze() for immutability, and full Luau type annotations.",
            "Create the Vertigo config module that defines {desc} tuning parameters. "
            "Include stable IDs (category_name_vN format) and typed constants.",
        ],
    },
    "Shared/Util/Physics": {
        "file": "codebase_physics.jsonl",
        "category": "physics",
        "templates": [
            "Implement a frame-rate independent physics module for {desc}. "
            "Use analytical solutions where possible, @native on hot paths, "
            "and SIMD vector math for performance.",
            "Write the {desc} physics system using Vertigo's math conventions: "
            "Dot products instead of Magnitude, math.exp for exponential damping, "
            "and time-corrected integration.",
        ],
    },
    "Shared/Util": {
        "file": "codebase_physics.jsonl",  # general utils go with physics
        "category": "physics",
        "templates": [
            "Write a utility module for {desc} following Vertigo's shared util conventions.",
        ],
    },
    "Shared/Net": {
        "file": "codebase_networking.jsonl",
        "category": "networking",
        "templates": [
            "Write the networking module for {desc}. Define typed RemoteEvents and "
            "RemoteFunctions with server/client separation and timeout handling.",
            "Implement the {desc} network layer with Vertigo's remote pattern: "
            "constants-first design, WaitForChild with timeout, and typed payloads.",
        ],
    },
    "Shared/Telemetry": {
        "file": "codebase_services.jsonl",  # telemetry is service-adjacent
        "category": "service",
        "templates": [
            "Write the telemetry module for {desc} with aggregate counters, funnel tracking, and histogram collection.",
        ],
    },
    "Server/World": {
        "file": "codebase_builders.jsonl",
        "category": "builder",
        "templates": [
            "Write the world management module for {desc} following Vertigo's zone system.",
        ],
    },
    "Client/UI": {
        "file": "codebase_controllers.jsonl",
        "category": "controller",
        "templates": [
            "Write the UI module for {desc} using ScreenGui, proper layering, and responsive layout patterns.",
        ],
    },
}

# Catch-all for files that don't match any prefix
DEFAULT_CATEGORY = {
    "file": "codebase_other.jsonl",
    "category": "general_luau",
    "templates": [
        "Write the Luau module `{path}` for the Vertigo experience that handles {desc}.",
    ],
}


def infer_description(filepath: Path, content: str) -> str:
    """Infer a human-readable description from filename."""
    name = filepath.stem
    for suffix in ("Service", "Controller", "Builder", "Module", "Util", "Helper"):
        name = name.replace(suffix, "")
    # PascalCase to spaced words
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower().strip()
    return spaced or filepath.stem.lower()


def detect_code_features(content: str) -> list[str]:
    """Detect notable code features for reasoning generation."""
    features = []
    checks = [
        ("--!strict", "Using --!strict mode for full type safety."),
        ("@native", "Annotating hot-path functions with @native for Luau NCG compilation."),
        ("table.freeze", "Freezing constant tables to prevent mutation and enable optimization."),
        (":Init()", "Following the two-phase boot: Init sets up state, Start connects listeners."),
        ("RemoteEvent", "Using typed RemoteEvents for client-server communication."),
        ("RemoteFunction", "Using RemoteFunctions for request-response patterns with timeout."),
        ("CollectionService", "Using CollectionService tags for runtime object discovery."),
        ("RunService", "Connecting to RunService for frame-locked updates."),
        ("DataStoreService", "Using DataStoreService with retry logic for persistence."),
        ("Random.new", "Using deterministic RNG for reproducible procedural generation."),
        ("WeldConstraint", "Assembling parts with WeldConstraints for structural integrity."),
        ("workspace:Raycast", "Using raycasting for spatial queries (wall/ground detection)."),
        ("math.exp(", "Using exponential damping for frame-rate independent smoothing."),
        (":Dot(", "Using dot product instead of Magnitude to avoid sqrt."),
        ("task.wait", "Using modern task library instead of deprecated wait()."),
        ("task.spawn", "Using task.spawn for coroutine management."),
        ("pcall", "Wrapping fallible operations in pcall for error handling."),
        ("export type", "Exporting typed interfaces for cross-module contracts."),
        ("table.create(", "Pre-sizing arrays for performance."),
        ("Signal", "Using Signal library for event dispatch."),
        ("Trove", "Using Trove for lifecycle/memory cleanup management."),
        ("Promise", "Using Promise library for async/await patterns."),
    ]
    for pattern, description in checks:
        if pattern in content:
            features.append(description)
    return features


def generate_reasoning(features: list[str], category: str) -> str:
    """Generate a reasoning trace from detected features."""
    if not features:
        features = [f"Implementing {category} module following Vertigo conventions."]
    return "<think>\n" + "\n".join(features[:8]) + "\n</think>\n\n"


def estimate_difficulty(content: str) -> int:
    """Estimate difficulty 1-5 based on code complexity."""
    lines = content.count("\n")
    services = sum(
        1
        for s in [
            "DataStoreService",
            "RunService",
            "CollectionService",
            "RemoteEvent",
            "RemoteFunction",
            "TweenService",
            "UserInputService",
            "Workspace",
        ]
        if s in content
    )

    if lines < 30:
        base = 1
    elif lines < 80:
        base = 2
    elif lines < 200:
        base = 3
    elif lines < 500:
        base = 4
    else:
        base = 5

    # Bump up for multi-service integration
    base = min(5, base + (services // 3))
    return base


def route_file(rel_path: str) -> tuple[str, dict]:
    """Route a file to the correct category config."""
    # Try longest prefix match first (more specific wins)
    sorted_prefixes = sorted(CATEGORIES.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if rel_path.startswith(prefix):
            return prefix, CATEGORIES[prefix]
    return "", DEFAULT_CATEGORY


def extract_all(src_dir: Path) -> dict[str, list[dict]]:
    """Extract training pairs from every .luau file, grouped by output file."""
    import random

    rng = random.Random(42)

    by_file: dict[str, list[dict]] = {}
    skipped = []

    for luau_file in sorted(src_dir.rglob("*.luau")):
        content = luau_file.read_text(encoding="utf-8", errors="ignore")
        rel = str(luau_file.relative_to(src_dir))

        # Skip init bootstrappers (these are just require() lists)
        if luau_file.name in ("init.server.luau", "init.client.luau") and len(content) < 5000:
            skipped.append(rel)
            continue

        # Skip truly empty files
        if len(content.strip()) < 50:
            skipped.append(rel)
            continue

        prefix, config = route_file(rel)
        output_file = config["file"]
        category = config["category"]
        template = rng.choice(config["templates"])

        desc = infer_description(luau_file, content)
        instruction = template.format(desc=desc, path=rel)

        features = detect_code_features(content)
        reasoning = generate_reasoning(features, category)
        difficulty = estimate_difficulty(content)

        pair = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": reasoning + content},
            ],
            "source": "vertigo_codebase",
            "category": category,
            "file_path": rel,
            "has_reasoning": True,
            "difficulty": difficulty,
        }

        by_file.setdefault(output_file, []).append(pair)

    if skipped:
        print(f"  Skipped {len(skipped)} files (bootstrappers/empty)")

    return by_file


def generate_architecture_patterns() -> list[dict]:
    """Generate isolated architecture pattern examples."""
    patterns = []

    # Pattern 1: Init/Start lifecycle
    patterns.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Show me the correct Init/Start lifecycle pattern for a Vertigo server service.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "Vertigo services follow a two-phase boot pattern:\n"
                        "1. Init() — setup state, cache references, no side effects\n"
                        "2. Start() — connect listeners, remotes, RunService loops\n"
                        "This decouples initialization from interconnection, preventing circular dependency issues.\n"
                        "The boot sequence is: Remotes.setup() → all :Init() → all :Start()\n"
                        "Services receive dependencies explicitly (no global singletons).\n"
                        "</think>\n\n"
                        "--!strict\n"
                        'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
                        'local Players = game:GetService("Players")\n\n'
                        'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
                        "local Remotes = require(Shared.Net.Remotes)\n"
                        "local Types = require(Shared.Util.Types)\n\n"
                        "export type PlayerState = {\n"
                        "\tjoinedAt: number,\n"
                        "\tlastAction: string,\n"
                        "}\n\n"
                        "local ExampleService = {}\n\n"
                        "function ExampleService:Init()\n"
                        "\tself._playerStates = {} :: { [Player]: PlayerState }\n"
                        "end\n\n"
                        "function ExampleService:Start()\n"
                        '\tlocal remote = Remotes.get("RequestExample")\n'
                        "\tremote.OnServerEvent:Connect(function(player: Player, action: string)\n"
                        "\t\tself:_handleRequest(player, action)\n"
                        "\tend)\n\n"
                        "\tPlayers.PlayerRemoving:Connect(function(player: Player)\n"
                        "\t\tself._playerStates[player] = nil\n"
                        "\tend)\n"
                        "end\n\n"
                        "function ExampleService:_handleRequest(player: Player, action: string)\n"
                        "\tlocal state = self._playerStates[player]\n"
                        "\tif not state then return end\n"
                        "\tstate.lastAction = action\n"
                        "end\n\n"
                        "return ExampleService\n"
                    ),
                },
            ],
            "source": "vertigo_codebase",
            "category": "service",
            "has_reasoning": True,
            "difficulty": 2,
        }
    )

    # Pattern 2: Server-authoritative validation with rate limiting
    patterns.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "How should I validate a client ability request on the server in Vertigo? Include rate limiting.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "Server-authoritative validation is non-negotiable in Vertigo. Never trust client state.\n"
                        "The validation pipeline:\n"
                        "1. Rate limit (10 msgs/sec, 16ms debounce)\n"
                        "2. Check ability unlocked in profile\n"
                        "3. Check cooldown\n"
                        "4. Spatial validation (KDTree for grapple anchors)\n"
                        "5. Fire confirmation or rejection via StateSync\n"
                        "Rejection includes a reason code for client-side feedback.\n"
                        "</think>\n\n"
                        "--!strict\n"
                        "local COOLDOWN = 0.3\n"
                        "local MAX_RPS = 10\n"
                        "local DEBOUNCE_MS = 16\n\n"
                        "function AbilityService:_validateRequest(\n"
                        "\tplayer: Player,\n"
                        "\tabilityId: string,\n"
                        "\ttarget: Vector3?\n"
                        "): (boolean, string?)\n"
                        "\tlocal now = tick()\n\n"
                        "\t-- Rate limit\n"
                        "\tlocal last = self._lastRequestTime[player]\n"
                        "\tif last and (now - last) < (1 / MAX_RPS) then\n"
                        '\t\treturn false, "rate_limited"\n'
                        "\tend\n"
                        "\tself._lastRequestTime[player] = now\n\n"
                        "\t-- Check unlock\n"
                        "\tlocal profile = self._dataService:GetProfile(player)\n"
                        '\tif not profile then return false, "no_profile" end\n'
                        "\tif not table.find(profile.unlockedAbilities, abilityId) then\n"
                        '\t\treturn false, "not_unlocked"\n'
                        "\tend\n\n"
                        "\t-- Check cooldown\n"
                        "\tlocal cd = self._cooldowns[player]\n"
                        "\tif cd and cd[abilityId] and (now - cd[abilityId]) < COOLDOWN then\n"
                        '\t\treturn false, "on_cooldown"\n'
                        "\tend\n\n"
                        "\t-- Spatial check (grapple anchor proximity)\n"
                        '\tif abilityId == "ability_grapple_v1" and target then\n'
                        "\t\tlocal nearest = self._anchorTree:findNearest(target)\n"
                        "\t\tif not nearest or (nearest.Position - target).Magnitude > 180 then\n"
                        '\t\t\treturn false, "no_anchor_in_range"\n'
                        "\t\tend\n"
                        "\tend\n\n"
                        "\treturn true, nil\n"
                        "end\n"
                    ),
                },
            ],
            "source": "vertigo_codebase",
            "category": "service",
            "has_reasoning": True,
            "difficulty": 3,
        }
    )

    # Pattern 3: Luau NCG optimization
    patterns.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What are Vertigo's Luau NCG optimization techniques for hot paths?"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "Luau's Native Code Generation (@native) compiles annotated functions to machine code.\n"
                        "Key rules for hot paths (Heartbeat, RenderStepped):\n"
                        "- @native annotation on the function\n"
                        "- SIMD vector.* API instead of Vector3 methods\n"
                        "- v:Dot(v) instead of v.Magnitude^2 (avoids sqrt)\n"
                        "- math.lerp(a,b,t) compiles to fused multiply-add\n"
                        "- math.exp(-c*dt) for frame-rate independent exponential damping\n"
                        "- table.create(n) for pre-sized arrays\n"
                        "- No closures in loops, no Instance.new() in Heartbeat\n"
                        "- Extract functions to module level, not inline\n"
                        "</think>\n\n"
                        "--!strict\n\n"
                        "-- BAD: closure per frame, Vector3 methods, no @native\n"
                        "--[[\n"
                        "RunService.Heartbeat:Connect(function(dt)\n"
                        "\tfor _, p in parts do\n"
                        "\t\tlocal dir = (target - p.Position).Unit\n"
                        "\t\tp.Position += dir * speed * dt\n"
                        "\tend\n"
                        "end)\n"
                        "]]\n\n"
                        "-- GOOD: @native, no closures, SIMD, pre-computed\n"
                        "--[[@native]]\n"
                        "local function updateParts(\n"
                        "\tparts: { BasePart },\n"
                        "\ttarget: Vector3,\n"
                        "\tspeed: number,\n"
                        "\tdt: number\n"
                        ")\n"
                        "\tfor i = 1, #parts do\n"
                        "\t\tlocal pos = parts[i].Position\n"
                        "\t\tlocal delta = target - pos\n"
                        "\t\tlocal distSq = delta:Dot(delta)\n"
                        "\t\tif distSq > 0.01 then\n"
                        "\t\t\tlocal invDist = 1 / math.sqrt(distSq)\n"
                        "\t\t\tparts[i].Position = pos + delta * (invDist * speed * dt)\n"
                        "\t\tend\n"
                        "\tend\n"
                        "end\n\n"
                        "RunService.Heartbeat:Connect(function(dt: number)\n"
                        "\tupdateParts(cachedParts, cachedTarget, SPEED, dt)\n"
                        "end)\n"
                    ),
                },
            ],
            "source": "vertigo_codebase",
            "category": "physics",
            "has_reasoning": True,
            "difficulty": 4,
        }
    )

    # Pattern 4: Remote definition pattern
    patterns.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Show me Vertigo's remote event definition and retrieval pattern."},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "Vertigo uses a constants-first remote design:\n"
                        "- REMOTE_EVENTS and REMOTE_FUNCTIONS are frozen tables defined at module load\n"
                        "- Server creates remotes in Remotes.setup()\n"
                        "- Client retrieves with Remotes.get(name) using WaitForChild with 10s timeout\n"
                        "- This prevents typos (frozen table errors on unknown keys) and timeout hangs\n"
                        "</think>\n\n"
                        "--!strict\n"
                        'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
                        'local RunService = game:GetService("RunService")\n\n'
                        "local REMOTE_EVENTS = table.freeze({\n"
                        '\t"RequestUseAbility",\n'
                        '\t"RequestSpawnVehicle",\n'
                        '\t"StateSync",\n'
                        '\t"EventNotify",\n'
                        '\t"WeatherSync",\n'
                        "})\n\n"
                        "local REMOTE_FUNCTIONS = table.freeze({\n"
                        '\t"GetPlayerProfile",\n'
                        '\t"GetNPCDialog",\n'
                        '\t"GetTraversalHint",\n'
                        "})\n\n"
                        "local Remotes = {}\n"
                        "local _folder: Folder?\n\n"
                        "function Remotes.setup()\n"
                        '\tassert(RunService:IsServer(), "Remotes.setup() must be called on server")\n'
                        '\t_folder = Instance.new("Folder")\n'
                        '\t_folder.Name = "Remotes"\n'
                        "\tfor _, name in REMOTE_EVENTS do\n"
                        '\t\tlocal r = Instance.new("RemoteEvent")\n'
                        "\t\tr.Name = name\n"
                        "\t\tr.Parent = _folder\n"
                        "\tend\n"
                        "\tfor _, name in REMOTE_FUNCTIONS do\n"
                        '\t\tlocal r = Instance.new("RemoteFunction")\n'
                        "\t\tr.Name = name\n"
                        "\t\tr.Parent = _folder\n"
                        "\tend\n"
                        "\t_folder.Parent = ReplicatedStorage\n"
                        "end\n\n"
                        "function Remotes.get(name: string): RemoteEvent | RemoteFunction\n"
                        "\tif RunService:IsServer() then\n"
                        '\t\tassert(_folder, "Call Remotes.setup() first")\n'
                        "\t\treturn _folder:FindFirstChild(name) :: any\n"
                        "\telse\n"
                        '\t\tlocal folder = ReplicatedStorage:WaitForChild("Remotes", 10)\n'
                        '\t\tassert(folder, "Remotes folder not found (10s timeout)")\n'
                        "\t\treturn folder:WaitForChild(name, 10) :: any\n"
                        "\tend\n"
                        "end\n\n"
                        "return Remotes\n"
                    ),
                },
            ],
            "source": "vertigo_codebase",
            "category": "networking",
            "has_reasoning": True,
            "difficulty": 3,
        }
    )

    # Pattern 5: DataStore with schema migration
    patterns.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Show me Vertigo's DataStore save pattern with schema versioning and migration.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "Vertigo's DataService uses:\n"
                        "- CURRENT_SCHEMA_VERSION = 7 with forward migration\n"
                        "- Auto-save every 120s + on PlayerRemoving + on meaningful unlock\n"
                        "- Per-player save lock to prevent concurrent DataStore writes\n"
                        "- 3 retries with 1s backoff on failures\n"
                        "- Profile normalization for backwards compatibility\n"
                        "</think>\n\n"
                        "--!strict\n"
                        'local DataStoreService = game:GetService("DataStoreService")\n\n'
                        'local STORE_NAME = "VertigoPlayerData_v1"\n'
                        "local CURRENT_SCHEMA = 7\n"
                        "local AUTO_SAVE_INTERVAL = 120\n"
                        "local MAX_RETRIES = 3\n"
                        "local RETRY_DELAY = 1\n\n"
                        "local store = DataStoreService:GetDataStore(STORE_NAME)\n\n"
                        "function DataService:_loadProfile(player: Player): Types.PlayerProfile?\n"
                        '\tlocal key = "player_" .. player.UserId\n'
                        "\tfor attempt = 1, MAX_RETRIES do\n"
                        "\t\tlocal ok, data = pcall(function()\n"
                        "\t\t\treturn store:GetAsync(key)\n"
                        "\t\tend)\n"
                        "\t\tif ok then\n"
                        "\t\t\tif data then\n"
                        "\t\t\t\treturn self:_migrateSchema(data)\n"
                        "\t\t\telse\n"
                        "\t\t\t\treturn self:_createDefaultProfile(player)\n"
                        "\t\t\tend\n"
                        "\t\tend\n"
                        "\t\tif attempt < MAX_RETRIES then\n"
                        "\t\t\ttask.wait(RETRY_DELAY)\n"
                        "\t\tend\n"
                        "\tend\n"
                        '\twarn("[DataService] Failed to load profile for", player.Name)\n'
                        "\treturn nil\n"
                        "end\n\n"
                        "function DataService:_migrateSchema(data: any): Types.PlayerProfile\n"
                        "\tlocal version = data.schemaVersion or 1\n"
                        "\twhile version < CURRENT_SCHEMA do\n"
                        "\t\tif version == 6 then\n"
                        "\t\t\t-- v6 → v7: add actorMemory field\n"
                        "\t\t\tdata.actorMemory = data.actorMemory or { interactions = {} }\n"
                        "\t\tend\n"
                        "\t\tversion += 1\n"
                        "\tend\n"
                        "\tdata.schemaVersion = CURRENT_SCHEMA\n"
                        "\treturn data :: Types.PlayerProfile\n"
                        "end\n"
                    ),
                },
            ],
            "source": "vertigo_codebase",
            "category": "service",
            "has_reasoning": True,
            "difficulty": 4,
        }
    )

    # Pattern 6: Builder pattern with deterministic RNG
    patterns.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Show me how to write a Vertigo zone builder with deterministic procedural generation.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "Vertigo builders use:\n"
                        "- Random.new(seed) for deterministic RNG (same seed = same world)\n"
                        "- BuildOptions type for configurable generation\n"
                        "- Color palettes as named constants\n"
                        "- WeldConstraints for structural assembly\n"
                        "- CollectionService tags for runtime discovery\n"
                        "- Part anchoring for static geometry\n"
                        "</think>\n\n"
                        "--!strict\n"
                        'local CollectionService = game:GetService("CollectionService")\n\n'
                        "export type BuildOptions = {\n"
                        "\torigin: Vector3?,\n"
                        "\tseed: number?,\n"
                        "\tbiomeTheme: string?,\n"
                        "}\n\n"
                        "local COLORS = table.freeze({\n"
                        "\tstone = Color3.fromRGB(140, 135, 130),\n"
                        "\tcrystal = Color3.fromHSV(0.75, 0.3, 0.9),\n"
                        "\tglow = Color3.fromHSV(0.55, 0.2, 1.0),\n"
                        "})\n\n"
                        "local ExampleBuilder = {}\n\n"
                        "function ExampleBuilder.build(options: BuildOptions?): Model\n"
                        "\tlocal opts = options or {}\n"
                        "\tlocal origin = opts.origin or Vector3.zero\n"
                        "\tlocal rng = Random.new(opts.seed or 12345)\n\n"
                        '\tlocal root = Instance.new("Model")\n'
                        '\troot.Name = "ExampleZone"\n\n'
                        "\tfor i = 1, 20 do\n"
                        '\t\tlocal part = Instance.new("Part")\n'
                        '\t\tpart.Name = "Platform_" .. i\n'
                        "\t\tpart.Size = Vector3.new(\n"
                        "\t\t\trng:NextNumber(8, 20),\n"
                        "\t\t\trng:NextNumber(1, 3),\n"
                        "\t\t\trng:NextNumber(8, 20)\n"
                        "\t\t)\n"
                        "\t\tpart.Position = origin + Vector3.new(\n"
                        "\t\t\trng:NextNumber(-100, 100),\n"
                        "\t\t\ti * 15,\n"
                        "\t\t\trng:NextNumber(-100, 100)\n"
                        "\t\t)\n"
                        "\t\tpart.Color = COLORS.stone\n"
                        "\t\tpart.Material = Enum.Material.Slate\n"
                        "\t\tpart.Anchored = true\n"
                        "\t\tpart.Parent = root\n\n"
                        "\t\t-- Add grapple anchor on every 3rd platform\n"
                        "\t\tif i % 3 == 0 then\n"
                        '\t\t\tlocal anchor = Instance.new("Part")\n'
                        '\t\t\tanchor.Name = "GrappleAnchor_" .. i\n'
                        "\t\t\tanchor.Size = Vector3.new(2, 2, 2)\n"
                        "\t\t\tanchor.Shape = Enum.PartType.Ball\n"
                        "\t\t\tanchor.Position = part.Position + Vector3.new(0, 3, 0)\n"
                        "\t\t\tanchor.Color = COLORS.glow\n"
                        "\t\t\tanchor.Material = Enum.Material.Neon\n"
                        "\t\t\tanchor.Anchored = true\n"
                        "\t\t\tanchor.Parent = root\n"
                        '\t\t\tCollectionService:AddTag(anchor, "GrappleAnchor")\n'
                        "\t\tend\n"
                        "\tend\n\n"
                        "\treturn root\n"
                        "end\n\n"
                        "return ExampleBuilder\n"
                    ),
                },
            ],
            "source": "vertigo_codebase",
            "category": "builder",
            "has_reasoning": True,
            "difficulty": 3,
        }
    )

    return patterns


def main():
    if not VERTIGO_SRC.exists():
        print(f"Error: Vertigo src not found at {VERTIGO_SRC}")
        return

    print("=== Granular Codebase Extraction ===")
    print(f"Source: {VERTIGO_SRC}\n")

    # Extract per-category files
    by_file = extract_all(VERTIGO_SRC)

    # Add architecture patterns
    patterns = generate_architecture_patterns()
    by_file.setdefault("codebase_patterns.jsonl", []).extend(patterns)

    # Write each output file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    print(f"{'Output File':<35} {'Examples':>8}")
    print("-" * 45)

    for filename, examples in sorted(by_file.items()):
        output_path = OUTPUT_DIR / filename
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"{filename:<35} {len(examples):>8}")
        total += len(examples)

    print("-" * 45)
    print(f"{'TOTAL':<35} {total:>8}")

    # Coverage report
    print("\n=== Coverage Report ===")
    all_luau = list(VERTIGO_SRC.rglob("*.luau"))
    covered = sum(len(exs) for exs in by_file.values()) - len(patterns)
    print(f"Total .luau files in src/: {len(all_luau)}")
    print(f"Files with training examples: {covered}")
    print(f"Architecture pattern examples: {len(patterns)}")

    # Category breakdown
    print("\n=== Category Breakdown ===")
    cats = Counter()
    for exs in by_file.values():
        for ex in exs:
            cats[ex.get("category", "unknown")] += 1
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    # Difficulty breakdown
    print("\n=== Difficulty Distribution ===")
    diffs = Counter()
    for exs in by_file.values():
        for ex in exs:
            diffs[ex.get("difficulty", 0)] += 1
    for d in sorted(diffs.keys()):
        bar = "█" * (diffs[d] // 2)
        print(f"  {d}: {diffs[d]:>4} {bar}")


if __name__ == "__main__":
    main()
