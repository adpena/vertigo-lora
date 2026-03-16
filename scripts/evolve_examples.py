#!/usr/bin/env python3
from __future__ import annotations

"""
Evolve simple training examples into harder ones using Evol-Instruct.

Strategy:
- Read existing instruction/completion pairs from data/raw/*.jsonl
- Apply evolution strategies to create more complex, production-grade examples
- Each evolved example includes a <think> reasoning trace and full Luau code
- Difficulty scored 1-5 based on services, length, error handling, multi-step logic

Falls back to a manual Evol-Instruct implementation when Distilabel is unavailable.

Output: data/raw/evolved.jsonl
"""

import argparse
import json
import random
import re
import textwrap
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT = RAW_DIR / "evolved.jsonl"

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
# Roblox / Vertigo domain knowledge for evolution
# ---------------------------------------------------------------------------

ROBLOX_SERVICES = [
    "DataStoreService",
    "MessagingService",
    "Players",
    "RunService",
    "CollectionService",
    "TweenService",
    "PhysicsService",
    "HttpService",
    "MarketplaceService",
    "TeleportService",
    "SoundService",
    "Lighting",
    "UserInputService",
    "ContextActionService",
    "StarterGui",
    "Teams",
    "BadgeService",
    "PolicyService",
    "TextService",
    "GroupService",
    "MemoryStoreService",
    "SocialService",
]

VERTIGO_CONCEPTS = [
    "Init/Start lifecycle",
    "Trove cleanup",
    "CollectionService tags",
    "server-authoritative validation",
    "deterministic RNG (Random.new(seed))",
    "WeldConstraints for assembly",
    "KDTree spatial queries",
    "RemoteEvent/RemoteFunction with rate limiting",
    "DataStore schema versioning",
    "procedural builder pattern",
    "@native NCG hot paths",
    "SIMD vector math",
    "frame-rate independent physics",
    "exponential damping (math.exp(-c*dt))",
    "part pooling in :Init()",
    "config modules with table.freeze()",
    "StateSync broadcast",
    "stable IDs (category_name_v1)",
    "two-phase boot",
]

ERROR_PATTERNS = [
    "pcall/xpcall wrapping",
    "retry with exponential backoff",
    "fallback defaults on failure",
    "timeout guards with task.delay",
    "rate limiting (N msgs/sec)",
    "input sanitization",
    "type validation on remote args",
    "graceful degradation",
    "error telemetry/logging",
    "circuit breaker pattern",
]

COMPLEXITY_ADDITIONS = [
    "concurrent request handling",
    "schema migration between versions",
    "cross-server communication via MessagingService",
    "ordered leaderboard with pagination",
    "real-time sync across servers",
    "spatial partitioning with octree/KDTree",
    "LOD (level of detail) system",
    "object pooling with warm/cold tiers",
    "priority queue scheduling",
    "A* pathfinding on navigation mesh",
    "bezier curve interpolation",
    "spring-damper physics simulation",
    "state machine with transitions",
    "observer pattern with typed signals",
    "command pattern with undo/redo",
    "ECS-style component composition",
    "promise chains with cancellation",
]

# ---------------------------------------------------------------------------
# Evolution strategies
# ---------------------------------------------------------------------------


def _extract_topic(instruction: str) -> str:
    """Pull the core topic from an instruction for use in evolution."""
    # Remove common prefixes
    cleaned = instruction
    for prefix in [
        "Write a Vertigo server service that ",
        "Write a Vertigo client controller that ",
        "Write a procedural world builder that generates ",
        "Create a zone builder for ",
        "Write a data-driven configuration module for ",
        "Implement a frame-rate independent physics module for ",
        "Write the networking module for ",
        "Write the Luau module ",
        "Implement the server-side ",
        "Implement the client-side controller for ",
        "Show me ",
        "How should I ",
        "What are ",
    ]:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :]
            break
    # Trim trailing punctuation
    cleaned = cleaned.rstrip(".!?")
    # Take first sentence if multi-sentence
    if ". " in cleaned:
        cleaned = cleaned.split(". ")[0]
    return cleaned.strip() or "a Vertigo system"


def evolve_add_constraints(instruction: str, rng: random.Random) -> str:
    """Add constraints: concurrent handling, retries, schema migration, etc."""
    topic = _extract_topic(instruction)
    constraints = rng.sample(
        [
            "handles concurrent access safely",
            "retries on failure with exponential backoff",
            "migrates between schema versions automatically",
            "validates all inputs server-side before processing",
            "rate-limits requests to prevent abuse (max 10/sec per player)",
            "uses table.freeze() for all constant config tables",
            "supports hot-reloading config without restarting",
            "implements graceful shutdown saving all pending state",
            "logs telemetry for every accept/reject decision",
            "enforces type-safe contracts via --!strict annotations",
        ],
        k=rng.randint(2, 4),
    )
    constraints_str = ", ".join(constraints[:-1]) + f", and {constraints[-1]}"
    templates = [
        (
            f"Write a Vertigo server service for {topic} that {constraints_str}. "
            f"Follow the Init/Start lifecycle and use modern Luau APIs."
        ),
        (
            f"Implement a production-grade {topic} module with these requirements: {constraints_str}. "
            f"Use --!strict mode, @native on hot paths, and Vertigo conventions."
        ),
    ]
    return rng.choice(templates)


def evolve_increase_complexity(instruction: str, rng: random.Random) -> str:
    """Increase complexity: multi-part procedural generation, advanced patterns."""
    topic = _extract_topic(instruction)
    additions = rng.sample(COMPLEXITY_ADDITIONS, k=rng.randint(2, 3))
    additions_str = ", ".join(additions)
    templates = [
        (
            f"Create a comprehensive {topic} system that incorporates {additions_str}. "
            f"Use --!strict mode, deterministic Random.new(seed) for reproducibility, "
            f"CollectionService tags for runtime discovery, and WeldConstraints for assembly."
        ),
        (
            f"Build an advanced {topic} module featuring {additions_str}. "
            f"Annotate hot-path functions with @native, use SIMD vector math, "
            f"and follow Vertigo's two-phase Init/Start boot pattern."
        ),
    ]
    return rng.choice(templates)


def evolve_add_error_handling(instruction: str, rng: random.Random) -> str:
    """Add comprehensive error handling requirements."""
    topic = _extract_topic(instruction)
    patterns = rng.sample(ERROR_PATTERNS, k=rng.randint(2, 4))
    patterns_str = ", ".join(patterns[:-1]) + f", and {patterns[-1]}"
    templates = [
        (
            f"Write a robust {topic} service with comprehensive error handling: {patterns_str}. "
            f"Every remote call must be server-authoritative. Use pcall for all "
            f"DataStore/network operations and log failures with structured context."
        ),
        (
            f"Implement a fault-tolerant {topic} module. Required error handling: {patterns_str}. "
            f"Follow Vertigo conventions: --!strict, Init/Start lifecycle, "
            f"and data-driven config with table.freeze()."
        ),
    ]
    return rng.choice(templates)


def evolve_multi_step(instruction: str, rng: random.Random) -> str:
    """Turn into a multi-step orchestration problem."""
    topic = _extract_topic(instruction)
    steps = rng.sample(
        [
            "initialize the subsystem with dependency injection in :Init()",
            "set up RemoteEvent listeners with rate limiting in :Start()",
            "validate all client inputs server-side before execution",
            "broadcast state changes to all clients via StateSync",
            "handle player join/leave lifecycle with proper cleanup via Trove",
            "persist state to DataStore with retry logic and schema versioning",
            "coordinate with other services through typed module APIs",
            "implement frame-locked updates via RunService.Heartbeat with @native",
            "add CollectionService tag-based discovery for runtime objects",
            "implement cooldown tracking with per-player debounce timers",
            "set network ownership correctly for physics-driven objects",
            "add rescue/recovery logic when the system enters an invalid state",
        ],
        k=rng.randint(4, 6),
    )
    numbered = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps))
    return (
        f"Implement a complete {topic} pipeline with these ordered steps:\n{numbered}\n\n"
        f"Each step must follow Vertigo conventions (--!strict, @native hot paths, "
        f"modern APIs). Include full type annotations and reasoning traces."
    )


def evolve_cross_cutting(instruction: str, rng: random.Random) -> str:
    """Combine multiple services/concerns into a cross-cutting feature."""
    topic = _extract_topic(instruction)
    services = rng.sample(ROBLOX_SERVICES, k=rng.randint(3, 5))
    services_str = ", ".join(services)
    concepts = rng.sample(VERTIGO_CONCEPTS, k=rng.randint(2, 3))
    concepts_str = ", ".join(concepts)
    templates = [
        (
            f"Build a cross-cutting {topic} system that integrates {services_str}. "
            f"Apply these Vertigo patterns: {concepts_str}. "
            f"The implementation must work across server/client boundary with "
            f"proper remote event handling and server-authoritative validation."
        ),
        (
            f"Design a {topic} feature spanning multiple Roblox services ({services_str}). "
            f"Requirements: {concepts_str}. "
            f"Use --!strict, @native on hot paths, and ensure the module "
            f"follows the Init/Start two-phase boot with Trove cleanup."
        ),
    ]
    return rng.choice(templates)


EVOLUTION_STRATEGIES = [
    ("add_constraints", evolve_add_constraints),
    ("increase_complexity", evolve_increase_complexity),
    ("add_error_handling", evolve_add_error_handling),
    ("multi_step", evolve_multi_step),
    ("cross_cutting", evolve_cross_cutting),
]

# ---------------------------------------------------------------------------
# Completion generation (manual Evol-Instruct fallback)
# ---------------------------------------------------------------------------

# Service-specific code templates used to assemble completions
_SERVICE_TEMPLATE = textwrap.dedent("""\
    --!strict
    local ReplicatedStorage = game:GetService("ReplicatedStorage")
    local Players = game:GetService("Players")
    {extra_services}
    local Shared = ReplicatedStorage:WaitForChild("Shared")
    local Remotes = require(Shared.Net.Remotes)
    {extra_requires}
    {type_block}
    {constants_block}
    local {name} = {{}}

    function {name}:Init()
    {init_body}
    end

    function {name}:Start()
    {start_body}
    end

    {methods}
    return {name}
""")

_CONTROLLER_TEMPLATE = textwrap.dedent("""\
    --!strict
    local ReplicatedStorage = game:GetService("ReplicatedStorage")
    local RunService = game:GetService("RunService")
    {extra_services}
    local Shared = ReplicatedStorage:WaitForChild("Shared")
    local Trove = require(ReplicatedStorage.Packages.Trove)
    {extra_requires}
    {type_block}
    {constants_block}
    local {name} = {{}}

    function {name}:Init()
    \tself._trove = Trove.new()
    {init_body}
    end

    function {name}:Start()
    {start_body}
    end

    function {name}:Destroy()
    \tself._trove:Destroy()
    end

    {methods}
    return {name}
""")

_BUILDER_TEMPLATE = textwrap.dedent("""\
    --!strict
    local CollectionService = game:GetService("CollectionService")
    {extra_services}
    {extra_requires}
    {type_block}
    {constants_block}
    local {name} = {{}}

    function {name}.build(options: BuildOptions): Model
    \tlocal rng = Random.new(options.seed)
    \tlocal root = Instance.new("Model")
    \troot.Name = "{short_name}"

    {build_body}
    \treturn root
    end

    {methods}
    return {name}
""")


def _pick_services(rng: random.Random, n: int) -> list[str]:
    """Pick n random Roblox services, excluding ones already in templates."""
    exclude = {"ReplicatedStorage", "Players", "RunService", "CollectionService"}
    pool = [s for s in ROBLOX_SERVICES if s not in exclude]
    return rng.sample(pool, k=min(n, len(pool)))


def _make_service_lines(services: list[str]) -> str:
    return "\n".join(f'local {s} = game:GetService("{s}")' for s in services)


def _generate_reasoning(strategy: str, evolved_instruction: str, services_used: list[str]) -> str:
    """Generate a <think> reasoning trace for the evolved completion."""
    lines = ["<think>"]

    # Strategy-specific reasoning
    if strategy == "add_constraints":
        lines.append("This implementation adds production constraints to harden the system.")
        lines.append("Each constraint is enforced at the module boundary, not sprinkled through callers.")
    elif strategy == "increase_complexity":
        lines.append("Increasing complexity requires careful architecture to keep the module maintainable.")
        lines.append("Using composition over inheritance — each concern is a focused sub-module.")
    elif strategy == "add_error_handling":
        lines.append("Robust error handling is critical for Roblox DataStore and network operations.")
        lines.append("Every external call is wrapped in pcall with structured error context for telemetry.")
    elif strategy == "multi_step":
        lines.append("Multi-step orchestration follows the Init/Start two-phase boot pattern.")
        lines.append("Each step is sequenced to respect dependency ordering — no side effects in Init.")
    elif strategy == "cross_cutting":
        lines.append("Cross-cutting features span multiple services, requiring careful interface design.")
        lines.append("Server-authoritative validation ensures the client cannot bypass game rules.")

    # Common conventions reasoning
    lines.append("Using --!strict for full type safety and compile-time error detection.")
    if services_used:
        lines.append(f"Integrating {', '.join(services_used)} with proper error boundaries.")
    lines.append("Following Vertigo conventions: @native on hot paths, table.freeze() for constants,")
    lines.append("modern APIs (task.wait, task.spawn), and no closures in per-frame loops.")
    lines.append("</think>")
    return "\n".join(lines)


def _generate_init_body(strategy: str, rng: random.Random) -> str:
    """Generate Init() body lines based on strategy."""
    parts = ["\tself._data = {} :: { [Player]: any }"]
    if strategy in ("add_constraints", "add_error_handling"):
        parts.append("\tself._retryQueue = {} :: { { player: Player, attempt: number, payload: any } }")
        parts.append("\tself._rateLimits = {} :: { [Player]: { count: number, windowStart: number } }")
    if strategy in ("multi_step", "cross_cutting"):
        parts.append("\tself._cooldowns = {} :: { [Player]: { [string]: number } }")
        parts.append("\tself._stateCache = {} :: { [string]: any }")
    if strategy == "increase_complexity":
        parts.append("\tself._pool = table.create(64)")
        parts.append("\tself._activeCount = 0")
    return "\n".join(parts)


def _generate_start_body(
    strategy: str,
    name: str,
    services: list[str],
    rng: random.Random,
) -> str:
    """Generate Start() body lines based on strategy."""
    parts = []

    # Remote setup
    parts.append(f'\tlocal remote = Remotes.get("Request{name.replace("Service", "")}")')
    parts.append("\tremote.OnServerEvent:Connect(function(player: Player, ...)")
    parts.append("\t\tself:_handleRequest(player, ...)")
    parts.append("\tend)")
    parts.append("")

    # Player lifecycle
    parts.append("\tPlayers.PlayerAdded:Connect(function(player: Player)")
    parts.append("\t\tself:_onPlayerAdded(player)")
    parts.append("\tend)")
    parts.append("\tPlayers.PlayerRemoving:Connect(function(player: Player)")
    parts.append("\t\tself:_onPlayerRemoving(player)")
    parts.append("\tend)")

    if strategy in ("multi_step", "cross_cutting") and "RunService" in services:
        parts.append("")
        parts.append("\tRunService.Heartbeat:Connect(function(dt: number)")
        parts.append("\t\tself:_update(dt)")
        parts.append("\tend)")

    return "\n".join(parts)


def _generate_methods(
    strategy: str,
    name: str,
    services: list[str],
    rng: random.Random,
) -> str:
    """Generate method implementations based on strategy."""
    methods = []

    # _handleRequest with validation
    handle_lines = [
        f"function {name}:_handleRequest(player: Player, action: string, payload: any)",
        "\t-- Rate limit check",
        "\tlocal now = os.clock()",
        "\tlocal limit = self._rateLimits[player]"
        if strategy in ("add_constraints", "add_error_handling")
        else "\tlocal now = os.clock()",
    ]
    if strategy in ("add_constraints", "add_error_handling"):
        handle_lines.extend(
            [
                "\tif limit then",
                "\t\tif (now - limit.windowStart) < 1 then",
                "\t\t\tlimit.count += 1",
                "\t\t\tif limit.count > 10 then",
                '\t\t\t\twarn("[' + name + '] Rate limited:", player.Name)',
                "\t\t\t\treturn",
                "\t\t\tend",
                "\t\telse",
                "\t\t\tlimit.windowStart = now",
                "\t\t\tlimit.count = 1",
                "\t\tend",
                "\telse",
                "\t\tself._rateLimits[player] = { count = 1, windowStart = now }",
                "\tend",
                "",
            ]
        )

    handle_lines.extend(
        [
            "\t-- Type validation",
            '\tif type(action) ~= "string" then',
            '\t\twarn("[' + name + '] Invalid action type from:", player.Name)',
            "\t\treturn",
            "\tend",
            "",
            "\t-- Dispatch",
        ]
    )

    if strategy == "add_error_handling":
        handle_lines.extend(
            [
                "\tlocal ok, err = pcall(function()",
                "\t\tself:_processAction(player, action, payload)",
                "\tend)",
                "\tif not ok then",
                '\t\twarn("[' + name + '] Error processing action:", err)',
                "\t\tself:_scheduleRetry(player, action, payload, 1)",
                "\tend",
            ]
        )
    else:
        handle_lines.append("\tself:_processAction(player, action, payload)")

    handle_lines.append("end")
    methods.append("\n".join(handle_lines))

    # _processAction
    process_lines = [
        f"function {name}:_processAction(player: Player, action: string, payload: any)",
    ]
    if "DataStoreService" in services:
        process_lines.extend(
            [
                '\tif action == "save" then',
                "\t\tself:_savePlayerData(player, payload)",
                '\telseif action == "load" then',
                "\t\tself:_loadPlayerData(player)",
                "\telse",
                f'\t\twarn("[{name}] Unknown action:", action)',
                "\tend",
            ]
        )
    else:
        process_lines.extend(
            [
                "\t-- Process the action with server-authoritative validation",
                "\tlocal playerData = self._data[player]",
                "\tif not playerData then return end",
                "",
                "\t-- Apply the action",
                "\tplayerData.lastAction = action",
                "\tplayerData.lastUpdate = os.clock()",
            ]
        )
    process_lines.append("end")
    methods.append("\n".join(process_lines))

    # Player lifecycle
    methods.append(
        "\n".join(
            [
                f"function {name}:_onPlayerAdded(player: Player)",
                "\tself._data[player] = {",
                "\t\tjoinedAt = os.clock(),",
                "\t\tlastAction = nil :: string?,",
                "\t\tlastUpdate = 0,",
                "\t}",
                "end",
            ]
        )
    )

    methods.append(
        "\n".join(
            [
                f"function {name}:_onPlayerRemoving(player: Player)",
                "\tself._data[player] = nil",
                "\tif self._rateLimits then",
                "\t\tself._rateLimits[player] = nil",
                "\tend",
                "\tif self._cooldowns then",
                "\t\tself._cooldowns[player] = nil",
                "\tend",
                "end",
            ]
        )
    )

    # Strategy-specific methods
    if strategy == "add_error_handling":
        methods.append(
            "\n".join(
                [
                    f"function {name}:_scheduleRetry(player: Player, action: string, payload: any, attempt: number)",
                    "\tlocal MAX_RETRIES = 3",
                    "\tif attempt > MAX_RETRIES then",
                    f'\t\twarn("[{name}] Max retries exceeded for:", player.Name, action)',
                    "\t\treturn",
                    "\tend",
                    "",
                    "\tlocal delay = math.min(2 ^ attempt, 30) -- exponential backoff, max 30s",
                    "\ttask.delay(delay, function()",
                    "\t\tif not player.Parent then return end -- player left",
                    "\t\tlocal ok, err = pcall(function()",
                    "\t\t\tself:_processAction(player, action, payload)",
                    "\t\tend)",
                    "\t\tif not ok then",
                    f'\t\t\twarn("[{name}] Retry", attempt, "failed:", err)',
                    "\t\t\tself:_scheduleRetry(player, action, payload, attempt + 1)",
                    "\t\tend",
                    "\tend)",
                    "end",
                ]
            )
        )

    if "DataStoreService" in services:
        methods.append(
            "\n".join(
                [
                    f"function {name}:_savePlayerData(player: Player, payload: any)",
                    '\tlocal store = DataStoreService:GetDataStore("PlayerData_v2")',
                    "\tlocal key = tostring(player.UserId)",
                    "",
                    "\tlocal ok, err = pcall(function()",
                    "\t\tstore:UpdateAsync(key, function(old)",
                    "\t\t\tlocal data = old or {}",
                    "\t\t\tdata.payload = payload",
                    "\t\t\tdata.lastSave = os.time()",
                    "\t\t\tdata.schemaVersion = 2",
                    "\t\t\treturn data",
                    "\t\tend)",
                    "\tend)",
                    "",
                    "\tif not ok then",
                    f'\t\twarn("[{name}] Save failed for", player.Name, ":", err)',
                    '\t\tself:_scheduleRetry(player, "save", payload, 1)',
                    "\tend",
                    "end",
                ]
            )
        )
        methods.append(
            "\n".join(
                [
                    f"function {name}:_loadPlayerData(player: Player)",
                    '\tlocal store = DataStoreService:GetDataStore("PlayerData_v2")',
                    "\tlocal key = tostring(player.UserId)",
                    "",
                    "\tlocal ok, data = pcall(function()",
                    "\t\treturn store:GetAsync(key)",
                    "\tend)",
                    "",
                    "\tif ok and data then",
                    "\t\t-- Schema migration",
                    "\t\tif (data.schemaVersion or 0) < 2 then",
                    "\t\t\tdata = self:_migrateSchema(data)",
                    "\t\tend",
                    "\t\tself._data[player] = data",
                    "\telse",
                    "\t\tif not ok then",
                    f'\t\t\twarn("[{name}] Load failed for", player.Name, ":", data)',
                    "\t\tend",
                    "\t\t-- Fallback defaults",
                    "\t\tself._data[player] = {",
                    "\t\t\tjoinedAt = os.clock(),",
                    "\t\t\tlastAction = nil,",
                    "\t\t\tlastUpdate = 0,",
                    "\t\t\tschemaVersion = 2,",
                    "\t\t}",
                    "\tend",
                    "end",
                ]
            )
        )
        methods.append(
            "\n".join(
                [
                    f"function {name}:_migrateSchema(data: any): any",
                    "\tlocal version = data.schemaVersion or 0",
                    "\tif version < 1 then",
                    "\t\t-- v0 -> v1: add lastUpdate field",
                    "\t\tdata.lastUpdate = data.lastUpdate or 0",
                    "\t\tdata.schemaVersion = 1",
                    "\tend",
                    "\tif version < 2 then",
                    "\t\t-- v1 -> v2: restructure payload",
                    "\t\tdata.payload = data.payload or {}",
                    "\t\tdata.schemaVersion = 2",
                    "\tend",
                    "\treturn data",
                    "end",
                ]
            )
        )

    if strategy in ("multi_step", "cross_cutting"):
        methods.append(
            "\n".join(
                [
                    "--[[@native]]",
                    f"function {name}:_update(dt: number)",
                    "\t-- Frame-rate independent update",
                    "\tlocal alpha = 1 - math.exp(-5 * dt)",
                    "\tfor player, data in self._data do",
                    "\t\tif data.lastUpdate and (os.clock() - data.lastUpdate) < 30 then",
                    "\t\t\t-- Active player: smooth state interpolation",
                    "\t\t\tif data.smoothedValue then",
                    "\t\t\t\tdata.smoothedValue = math.lerp(data.smoothedValue, data.targetValue or 0, alpha)",
                    "\t\t\tend",
                    "\t\tend",
                    "\tend",
                    "end",
                ]
            )
        )

    if "MessagingService" in services:
        methods.append(
            "\n".join(
                [
                    f"function {name}:_publishCrossServer(topic: string, payload: any)",
                    "\tlocal ok, err = pcall(function()",
                    "\t\tMessagingService:PublishAsync(topic, {",
                    "\t\t\tserverId = game.JobId,",
                    "\t\t\ttimestamp = os.time(),",
                    "\t\t\tdata = payload,",
                    "\t\t})",
                    "\tend)",
                    "\tif not ok then",
                    f'\t\twarn("[{name}] Cross-server publish failed:", err)',
                    "\tend",
                    "end",
                ]
            )
        )
        methods.append(
            "\n".join(
                [
                    f"function {name}:_subscribeCrossServer(topic: string)",
                    "\tlocal ok, err = pcall(function()",
                    "\t\tMessagingService:SubscribeAsync(topic, function(message)",
                    "\t\t\tlocal data = message.Data",
                    "\t\t\tif data.serverId == game.JobId then return end -- skip self",
                    "\t\t\ttask.spawn(function()",
                    "\t\t\t\tself:_handleCrossServerMessage(data)",
                    "\t\t\tend)",
                    "\t\tend)",
                    "\tend)",
                    "\tif not ok then",
                    f'\t\twarn("[{name}] Cross-server subscribe failed:", err)',
                    "\tend",
                    "end",
                ]
            )
        )

    return "\n\n".join(methods)


def _infer_module_kind(instruction: str) -> str:
    """Determine whether evolved instruction maps to service, controller, or builder."""
    lower = instruction.lower()
    if any(kw in lower for kw in ["server", "service", "datastore", "save", "load", "persist", "spawn", "leaderboard"]):
        return "service"
    if any(kw in lower for kw in ["client", "controller", "input", "camera", "hud", "ui", "render"]):
        return "controller"
    if any(kw in lower for kw in ["build", "generate", "procedural", "zone", "terrain", "world"]):
        return "builder"
    return "service"  # default


def _name_from_topic(topic: str) -> str:
    """Convert a topic string to a PascalCase module name."""
    words = re.sub(r"[^a-zA-Z0-9\s]", "", topic).split()
    pascal = "".join(w.capitalize() for w in words[:4])
    return pascal or "Evolved"


def generate_completion(
    evolved_instruction: str,
    strategy: str,
    rng: random.Random,
) -> tuple[str, int, list[str]]:
    """
    Generate a Luau completion for an evolved instruction.

    Returns (completion_text, difficulty_score, services_used).
    """
    kind = _infer_module_kind(evolved_instruction)
    topic = _extract_topic(evolved_instruction)
    raw_name = _name_from_topic(topic)

    # Pick services based on strategy complexity
    n_services = {
        "add_constraints": 2,
        "increase_complexity": 3,
        "add_error_handling": 2,
        "multi_step": 3,
        "cross_cutting": 4,
    }
    extra_services = _pick_services(rng, n_services.get(strategy, 2))

    # Force DataStoreService for persistence-related instructions
    lower_instr = evolved_instruction.lower()
    if any(kw in lower_instr for kw in ["save", "load", "persist", "datastore", "data"]):
        if "DataStoreService" not in extra_services:
            extra_services.insert(0, "DataStoreService")
    if any(kw in lower_instr for kw in ["cross-server", "messaging", "broadcast"]):
        if "MessagingService" not in extra_services:
            extra_services.insert(0, "MessagingService")

    all_services = list(extra_services)
    reasoning = _generate_reasoning(strategy, evolved_instruction, all_services)

    if kind == "service":
        suffix = "Service"
        name = raw_name + suffix
        code = _SERVICE_TEMPLATE.format(
            name=name,
            extra_services=_make_service_lines(extra_services),
            extra_requires="",
            type_block="",
            constants_block="local MAX_RETRIES = 3\nlocal RATE_LIMIT_WINDOW = 1\nlocal RATE_LIMIT_MAX = 10",
            init_body=_generate_init_body(strategy, rng),
            start_body=_generate_start_body(strategy, name, ["RunService"] + extra_services, rng),
            methods=_generate_methods(strategy, name, extra_services, rng),
        )
    elif kind == "controller":
        suffix = "Controller"
        name = raw_name + suffix
        code = _CONTROLLER_TEMPLATE.format(
            name=name,
            extra_services=_make_service_lines(extra_services),
            extra_requires="",
            type_block="",
            constants_block="local UPDATE_PRIORITY = Enum.RenderPriority.Camera.Value + 1",
            init_body=_generate_init_body(strategy, rng),
            start_body=_generate_start_body(strategy, name, ["RunService"] + extra_services, rng),
            methods=_generate_methods(strategy, name, extra_services, rng),
        )
    else:
        suffix = "Builder"
        name = raw_name + suffix
        build_body = textwrap.indent(
            "\n".join(
                [
                    'local model = Instance.new("Model")',
                    'model.Name = options.name or "Generated"',
                    "",
                    "-- Deterministic procedural generation",
                    "local partCount = rng:NextInteger(10, 50)",
                    "local parts = table.create(partCount)",
                    "",
                    "for i = 1, partCount do",
                    '\tlocal part = Instance.new("Part")',
                    "\tpart.Size = Vector3.new(",
                    "\t\trng:NextNumber(2, 10),",
                    "\t\trng:NextNumber(2, 20),",
                    "\t\trng:NextNumber(2, 10)",
                    "\t)",
                    "\tpart.Position = Vector3.new(",
                    "\t\trng:NextNumber(-100, 100),",
                    "\t\trng:NextNumber(0, 50),",
                    "\t\trng:NextNumber(-100, 100)",
                    "\t)",
                    "\tpart.Anchored = true",
                    "\tpart.Material = Enum.Material.SmoothPlastic",
                    '\tCollectionService:AddTag(part, "Generated")',
                    "\tpart.Parent = root",
                    "\tparts[i] = part",
                    "end",
                    "",
                    "-- Weld assembly",
                    "for i = 2, #parts do",
                    '\tlocal weld = Instance.new("WeldConstraint")',
                    "\tweld.Part0 = parts[1]",
                    "\tweld.Part1 = parts[i]",
                    "\tweld.Parent = parts[i]",
                    "end",
                ]
            ),
            "\t",
        )
        code = _BUILDER_TEMPLATE.format(
            name=name,
            short_name=raw_name,
            extra_services=_make_service_lines(extra_services),
            extra_requires="",
            type_block="export type BuildOptions = {\n\tseed: number,\n\tname: string?,\n\tscale: number?,\n}",
            constants_block="",
            build_body=build_body,
            methods=_generate_methods(strategy, name, extra_services, rng),
        )

    # Score difficulty
    difficulty = _score_difficulty(code, extra_services, strategy)

    completion = reasoning + "\n\n" + code.strip()
    return completion, difficulty, all_services


def _score_difficulty(code: str, services: list[str], strategy: str) -> int:
    """Score difficulty 1-5 based on multiple factors."""
    score = 1

    # Services used (each adds 0.5, cap at +2)
    score += min(2, len(services) * 0.5)

    # Code length
    line_count = code.count("\n")
    if line_count > 100:
        score += 1
    if line_count > 200:
        score += 0.5

    # Error handling complexity
    pcall_count = code.count("pcall")
    if pcall_count >= 2:
        score += 0.5
    if pcall_count >= 4:
        score += 0.5

    # Multi-step / cross-cutting strategies are inherently harder
    if strategy in ("multi_step", "cross_cutting"):
        score += 0.5

    return min(5, max(1, int(round(score))))


# ---------------------------------------------------------------------------
# Seed instructions (used when raw data is empty or as supplement)
# ---------------------------------------------------------------------------

SEED_INSTRUCTIONS = [
    "Write a DataStore save system",
    "Create a part",
    "Load player data",
    "Spawn a vehicle",
    "Make a leaderboard",
    "Handle player input",
    "Create an inventory system",
    "Build a shop UI",
    "Implement a cooldown system",
    "Write a chat command handler",
    "Create a teleport system",
    "Build a quest tracker",
    "Implement player respawning",
    "Write a damage system",
    "Create a notification system",
    "Build a friends list display",
    "Implement a voting system",
    "Write an admin command panel",
    "Create a weather system",
    "Build a pet follow system",
    "Implement a trading system",
    "Write a matchmaking service",
    "Create a zone transition system",
    "Build a particle effect manager",
    "Implement a sound manager",
    "Write a physics-based door",
    "Create a grapple hook system",
    "Build a glider controller",
    "Implement wall running",
    "Write a swimming controller",
    "Create a mount system",
    "Build a crafting system",
    "Implement a skill tree",
    "Write a buff/debuff system",
    "Create a dialogue system",
    "Build a minimap",
    "Implement a day/night cycle",
    "Write a procedural terrain generator",
    "Create an NPC patrol system",
    "Build a loot table system",
    "Implement a combo attack system",
    "Write a camera shake effect",
    "Create a badge award system",
    "Build a tutorial walkthrough",
    "Implement a replay system",
    "Write a server-hop system",
    "Create an anti-cheat validator",
    "Build a custom character loader",
    "Implement a ragdoll system",
    "Write an animation state machine",
]


def _try_distilabel_evolve(examples: list[dict]) -> list[dict] | None:
    """Attempt to use Distilabel's EvolInstruct for evolution. Returns None if unavailable."""
    try:
        from distilabel.steps.tasks import EvolInstruct
        from distilabel.llms import InferenceEndpointsLLM
    except ImportError:
        return None
    except Exception:
        return None

    # Distilabel is available — build and run the pipeline
    # This requires a running LLM endpoint; if configuration is missing, fall back.
    try:
        llm = InferenceEndpointsLLM(model_id="meta-llama/Meta-Llama-3-70B-Instruct")
        evol = EvolInstruct(
            llm=llm,
            num_evolutions=1,
            store_evolutions=True,
        )

        instructions = []
        for ex in examples:
            for msg in ex.get("messages", []):
                if msg["role"] == "user":
                    instructions.append(msg["content"])
                    break

        if not instructions:
            return None

        # Format input for Distilabel
        input_data = [{"instruction": inst} for inst in instructions]
        evol.load()
        results = list(evol.process(input_data))

        evolved = []
        rng = random.Random(42)
        for batch in results:
            for row in batch:
                evolved_inst = row.get("evolved_instruction") or row.get("instruction", "")
                if not evolved_inst:
                    continue

                completion, difficulty, services = generate_completion(
                    evolved_inst,
                    "increase_complexity",
                    rng,
                )
                evolved.append(
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": evolved_inst},
                            {"role": "assistant", "content": completion},
                        ],
                        "source": "evol_instruct_distilabel",
                        "category": _infer_module_kind(evolved_inst),
                        "has_reasoning": True,
                        "difficulty": difficulty,
                        "evolution_strategy": "distilabel_evol_instruct",
                    }
                )
        return evolved if evolved else None

    except Exception as exc:
        print(f"  Distilabel available but failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Manual fallback evolution pipeline
# ---------------------------------------------------------------------------


def load_raw_instructions(raw_dir: Path) -> list[str]:
    """Load user instructions from all JSONL files in raw_dir."""
    instructions = []
    if not raw_dir.exists():
        return instructions

    for jsonl_file in sorted(raw_dir.glob("*.jsonl")):
        # Skip our own output to avoid re-evolving
        if jsonl_file.name == "evolved.jsonl":
            continue
        try:
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    for msg in row.get("messages", []):
                        if msg["role"] == "user":
                            instructions.append(msg["content"])
                            break
        except (json.JSONDecodeError, KeyError):
            continue

    return instructions


def evolve_all(
    instructions: list[str],
    target_count: int = 120,
    seed: int = 42,
) -> list[dict]:
    """
    Apply Evol-Instruct evolution strategies to produce evolved training examples.

    Each source instruction is evolved with 1-3 randomly selected strategies,
    generating diverse, progressively harder examples.
    """
    rng = random.Random(seed)
    results: list[dict] = []
    seen_instructions: set[str] = set()

    # If we have fewer source instructions than target, cycle through them
    source_pool = list(instructions)
    if not source_pool:
        source_pool = list(SEED_INSTRUCTIONS)

    idx = 0
    while len(results) < target_count:
        base_instruction = source_pool[idx % len(source_pool)]
        idx += 1

        # Pick 1-2 evolution strategies per instruction
        n_evolutions = rng.randint(1, 2)
        strategies = rng.sample(EVOLUTION_STRATEGIES, k=min(n_evolutions, len(EVOLUTION_STRATEGIES)))

        for strategy_name, strategy_fn in strategies:
            if len(results) >= target_count:
                break

            evolved_instruction = strategy_fn(base_instruction, rng)

            # Deduplicate
            norm = evolved_instruction.strip().lower()
            if norm in seen_instructions:
                continue
            seen_instructions.add(norm)

            completion, difficulty, services = generate_completion(
                evolved_instruction,
                strategy_name,
                rng,
            )

            results.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": evolved_instruction},
                        {"role": "assistant", "content": completion},
                    ],
                    "source": "evol_instruct_manual",
                    "category": _infer_module_kind(evolved_instruction),
                    "has_reasoning": True,
                    "difficulty": difficulty,
                    "evolution_strategy": strategy_name,
                    "base_instruction": base_instruction[:120],
                }
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evolve simple training examples into harder ones using Evol-Instruct.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing to disk.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=120,
        help="Target number of evolved examples (default: 120).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    print("=== Evol-Instruct Example Evolution ===\n")

    # Step 1: Load existing instructions from raw data
    raw_instructions = load_raw_instructions(RAW_DIR)
    print(f"Loaded {len(raw_instructions)} source instructions from {RAW_DIR}")

    # Supplement with seed instructions if raw data is thin
    all_source = list(raw_instructions)
    if len(all_source) < 30:
        seed_used = [s for s in SEED_INSTRUCTIONS if s not in all_source]
        all_source.extend(seed_used)
        print(f"Added {len(seed_used)} seed instructions (total: {len(all_source)})")

    # Step 2: Try Distilabel first
    print("\nAttempting Distilabel EvolInstruct...")
    distilabel_results = _try_distilabel_evolve(
        [{"messages": [{"role": "user", "content": inst}]} for inst in all_source[:50]]
    )

    if distilabel_results:
        print(f"  Distilabel produced {len(distilabel_results)} examples")
        evolved = distilabel_results
        # Supplement with manual if needed
        if len(evolved) < args.count:
            remaining = args.count - len(evolved)
            print(f"  Supplementing with {remaining} manual evolutions...")
            manual = evolve_all(all_source, target_count=remaining, seed=args.seed)
            evolved.extend(manual)
    else:
        print("  Distilabel not available, using manual Evol-Instruct fallback")
        evolved = evolve_all(all_source, target_count=args.count, seed=args.seed)

    print(f"\nGenerated {len(evolved)} evolved examples")

    # Step 3: Stats
    from collections import Counter

    strategy_counts = Counter(e.get("evolution_strategy", "unknown") for e in evolved)
    category_counts = Counter(e.get("category", "unknown") for e in evolved)
    difficulty_counts = Counter(e.get("difficulty", 0) for e in evolved)

    print("\nEvolution strategies:")
    for strat, count in strategy_counts.most_common():
        print(f"  {strat}: {count}")

    print("\nCategories:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count}")

    print("\nDifficulty distribution:")
    for diff in sorted(difficulty_counts.keys()):
        bar = "#" * difficulty_counts[diff]
        print(f"  {diff}: {difficulty_counts[diff]:3d} {bar}")

    # Step 4: Dry run or write
    if args.dry_run:
        print("\n--- DRY RUN (first 3 examples) ---\n")
        for i, ex in enumerate(evolved[:3]):
            user_msg = ""
            for msg in ex["messages"]:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                    break
            print(f"Example {i + 1} [difficulty={ex['difficulty']}, strategy={ex.get('evolution_strategy')}]:")
            print(f"  Instruction: {user_msg[:200]}...")
            print(f"  Category: {ex['category']}")
            if ex.get("base_instruction"):
                print(f"  Evolved from: {ex['base_instruction']}")
            print()
        print(f"Would write {len(evolved)} examples to {OUTPUT}")
        print("Run without --dry-run to write.")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w") as f:
            for ex in evolved:
                f.write(json.dumps(ex) + "\n")
        print(f"\nWrote {len(evolved)} evolved examples -> {OUTPUT}")


if __name__ == "__main__":
    main()
