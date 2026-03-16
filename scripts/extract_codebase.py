#!/usr/bin/env python3
from __future__ import annotations

"""
Extract training data from Vertigo's Luau codebase.

Strategy (informed by research):
- Include reasoning traces explaining WHY each pattern is used
- Categorize by architecture role (service, controller, builder, config, physics)
- Generate multi-granularity examples:
  1. Full module examples (complete file with explanation)
  2. Pattern examples (isolated patterns like Init/Start, remote handling)
  3. Idiom examples (Luau NCG tricks, SIMD math, table patterns)
- All examples use --!strict mode conventions from CLAUDE.md

Output: data/raw/codebase.jsonl
"""

import json
import re
from pathlib import Path

VERTIGO_SRC = Path(__file__).resolve().parent.parent.parent / "src"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "codebase.jsonl"

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

# Architecture role detection
ROLE_PATTERNS = {
    "Server/Services": {
        "category": "service",
        "instruction_templates": [
            "Write a Vertigo server service that {desc}. Include Init/Start lifecycle, "
            "remote event handling, and server-authoritative validation.",
            "Implement the server-side {desc} service following Vertigo's two-phase boot pattern "
            "with explicit dependency injection.",
        ],
    },
    "Client/Controllers": {
        "category": "controller",
        "instruction_templates": [
            "Write a Vertigo client controller that {desc}. Use RunService connections and proper cleanup with Trove.",
            "Implement the client-side controller for {desc} with input handling and VFX management.",
        ],
    },
    "Server/World/Builders": {
        "category": "builder",
        "instruction_templates": [
            "Write a procedural world builder that generates {desc}. Use deterministic "
            "RNG (Random.new(seed)), CollectionService tags, and WeldConstraints.",
            "Create a zone builder for {desc} following Vertigo's builder pattern "
            "with BuildOptions type and part_builder shorthand.",
        ],
    },
    "Shared/Config": {
        "category": "config",
        "instruction_templates": [
            "Write a data-driven configuration module for {desc}. Use named exports, "
            "table.freeze() for immutability, and full type annotations.",
        ],
    },
    "Shared/Util/Physics": {
        "category": "physics",
        "instruction_templates": [
            "Implement a frame-rate independent physics module for {desc}. "
            "Use analytical solutions where possible, @native on hot paths, "
            "and SIMD vector math.",
        ],
    },
    "Shared/Net": {
        "category": "networking",
        "instruction_templates": [
            "Write the networking module for {desc}. Define RemoteEvents and "
            "RemoteFunctions with proper server/client separation and timeout handling.",
        ],
    },
}


def infer_description(filepath: Path, content: str) -> str:
    """Infer description from filename, removing suffixes."""
    name = filepath.stem
    for suffix in ("Service", "Controller", "Builder", "Module"):
        name = name.replace(suffix, "")
    # PascalCase to spaced words
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower().strip()
    return spaced or filepath.stem.lower()


def generate_reasoning(filepath: Path, content: str, category: str) -> str:
    """Generate a reasoning trace explaining the module's design choices."""
    reasoning_parts = []

    if "--!strict" in content:
        reasoning_parts.append("Using --!strict mode for full type safety.")

    if "@native" in content:
        reasoning_parts.append("Annotating hot-path functions with @native for Luau NCG compilation.")

    if "table.freeze" in content:
        reasoning_parts.append("Freezing constant tables to prevent accidental mutation and enable optimization.")

    if ":Init()" in content and ":Start()" in content:
        reasoning_parts.append(
            "Following the two-phase boot pattern: Init sets up state with no side effects, "
            "Start connects listeners and begins async operations."
        )

    if "RemoteEvent" in content or "RemoteFunction" in content:
        reasoning_parts.append(
            "Using typed remotes for client-server communication with server-authoritative validation."
        )

    if "CollectionService" in content:
        reasoning_parts.append(
            "Using CollectionService tags for runtime object discovery instead of path-based lookups."
        )

    if "RunService" in content:
        reasoning_parts.append("Connecting to RunService.Heartbeat/RenderStepped for frame-locked updates.")

    if "DataStoreService" in content:
        reasoning_parts.append("Using DataStoreService with retry logic and schema versioning for persistence.")

    if category == "builder":
        reasoning_parts.append(
            "Using deterministic Random.new(seed) for reproducible procedural generation. "
            "Parts are assembled with WeldConstraints and tagged via CollectionService."
        )

    if category == "physics":
        reasoning_parts.append(
            "Implementing frame-rate independent physics using analytical solutions "
            "rather than sub-stepping, ensuring consistent behavior across hardware."
        )

    if not reasoning_parts:
        reasoning_parts.append(f"Implementing {category} module following Vertigo conventions.")

    return "<think>\n" + "\n".join(reasoning_parts) + "\n</think>\n\n"


def detect_role(rel_path: str) -> tuple[str, dict] | None:
    """Detect the architecture role from file path."""
    for prefix, config in ROLE_PATTERNS.items():
        if rel_path.startswith(prefix):
            return prefix, config
    return None


def extract_module_pairs(src_dir: Path) -> list[dict]:
    """Walk the source tree and create instruction/completion pairs."""
    pairs = []
    import random

    rng = random.Random(42)

    for luau_file in sorted(src_dir.rglob("*.luau")):
        content = luau_file.read_text(encoding="utf-8", errors="ignore")

        # Skip tiny files and init bootstrappers
        if len(content.strip()) < 200:
            continue
        if luau_file.name in ("init.server.luau", "init.client.luau"):
            continue

        rel = str(luau_file.relative_to(src_dir))
        role = detect_role(rel)

        if role:
            prefix, config = role
            category = config["category"]
            template = rng.choice(config["instruction_templates"])
        else:
            category = "general_luau"
            template = "Write the Luau module `{desc}` for the Vertigo experience."

        desc = infer_description(luau_file, content)
        instruction = template.format(desc=desc)
        reasoning = generate_reasoning(luau_file, content, category)

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
            "difficulty": min(5, max(1, len(content) // 2000 + 1)),
        }
        pairs.append(pair)

    return pairs


def extract_pattern_examples(src_dir: Path) -> list[dict]:
    """Extract isolated pattern examples (Init/Start, remote handling, etc.)."""
    patterns = []

    # Pattern: Init/Start lifecycle
    patterns.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ("Show me the correct Init/Start lifecycle pattern for a Vertigo server service."),
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
                        "</think>\n\n"
                        "--!strict\n"
                        'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
                        'local Players = game:GetService("Players")\n\n'
                        'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
                        "local Remotes = require(Shared.Net.Remotes)\n\n"
                        "local ExampleService = {}\n\n"
                        "-- Phase 1: Setup state, no side effects\n"
                        "function ExampleService:Init()\n"
                        "\tself._playerData = {} :: { [Player]: PlayerState }\n"
                        "end\n\n"
                        "-- Phase 2: Connect listeners and begin operations\n"
                        "function ExampleService:Start()\n"
                        '\tlocal remote = Remotes.get("RequestExample")\n'
                        "\tremote.OnServerEvent:Connect(function(player: Player, ...)\n"
                        "\t\tself:_handleRequest(player, ...)\n"
                        "\tend)\n\n"
                        "\tPlayers.PlayerRemoving:Connect(function(player: Player)\n"
                        "\t\tself._playerData[player] = nil\n"
                        "\tend)\n"
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

    # Pattern: Server-authoritative validation
    patterns.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ("How should I validate a client ability request on the server in Vertigo?"),
                },
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "In Vertigo, the server is authoritative — never trust client state for abilities, "
                        "currency, or progression. The validation pattern is:\n"
                        "1. Check player has the ability unlocked (via DataService profile)\n"
                        "2. Check cooldown hasn't expired\n"
                        "3. Check spatial requirements (e.g., grapple anchor within range via KDTree)\n"
                        "4. Rate-limit requests (10 msgs/sec, 16ms debounce)\n"
                        "5. Fire confirmation or rejection back to client via StateSync\n"
                        "</think>\n\n"
                        "--!strict\n"
                        "local COOLDOWN_SECONDS = 0.3\n"
                        "local MAX_REQUESTS_PER_SEC = 10\n\n"
                        "function AbilityService:_validateAbilityRequest(\n"
                        "\tplayer: Player,\n"
                        "\tabilityId: string,\n"
                        "\ttargetPosition: Vector3?\n"
                        "): (boolean, string?)\n"
                        "\t-- 1. Rate limit\n"
                        "\tlocal now = tick()\n"
                        "\tlocal lastRequest = self._lastRequestTime[player]\n"
                        "\tif lastRequest and (now - lastRequest) < (1 / MAX_REQUESTS_PER_SEC) then\n"
                        '\t\treturn false, "rate_limited"\n'
                        "\tend\n"
                        "\tself._lastRequestTime[player] = now\n\n"
                        "\t-- 2. Check unlock\n"
                        "\tlocal profile = DataService:GetProfile(player)\n"
                        "\tif not profile or not table.find(profile.unlockedAbilities, abilityId) then\n"
                        '\t\treturn false, "not_unlocked"\n'
                        "\tend\n\n"
                        "\t-- 3. Check cooldown\n"
                        "\tlocal cooldowns = self._cooldowns[player]\n"
                        "\tif cooldowns and cooldowns[abilityId] then\n"
                        "\t\tif (now - cooldowns[abilityId]) < COOLDOWN_SECONDS then\n"
                        '\t\t\treturn false, "on_cooldown"\n'
                        "\t\tend\n"
                        "\tend\n\n"
                        "\t-- 4. Spatial validation (e.g., grapple anchor proximity)\n"
                        '\tif abilityId == "ability_grapple_v1" and targetPosition then\n'
                        "\t\tlocal nearest = self._anchorKDTree:findNearest(targetPosition)\n"
                        "\t\tif not nearest or (nearest.Position - targetPosition).Magnitude > GrappleTuning.maxRange then\n"
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

    # Pattern: Luau NCG optimization
    patterns.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ("What are the key Luau NCG optimization techniques used in Vertigo's hot paths?"),
                },
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "Vertigo uses several Luau Native Code Generation optimizations for performance:\n"
                        "- @native annotations on Heartbeat/RenderStepped callbacks\n"
                        "- SIMD vector.* API instead of Vector3 methods\n"
                        "- v:Dot(v) instead of v.Magnitude^2 to avoid sqrt\n"
                        "- math.lerp(a,b,t) compiles to fused multiply-add\n"
                        "- math.exp(-c*dt) for frame-rate independent exponential damping\n"
                        "- table.create(n) for pre-sized arrays\n"
                        "- No closures in loops, no Instance.new() in Heartbeat\n"
                        "</think>\n\n"
                        "--!strict\n\n"
                        "-- BAD: Creates closure every frame, uses Vector3 methods\n"
                        "--[[\n"
                        "RunService.Heartbeat:Connect(function(dt)\n"
                        "\tfor _, part in parts do\n"
                        "\t\tlocal dir = (target - part.Position).Unit\n"
                        "\t\tpart.Position = part.Position + dir * speed * dt\n"
                        "\tend\n"
                        "end)\n"
                        "]]\n\n"
                        "-- GOOD: @native, no closures, SIMD math, pre-computed values\n"
                        "--[[@native]]\n"
                        "local function updateParts(parts: { BasePart }, target: Vector3, speed: number, dt: number)\n"
                        "\tfor i = 1, #parts do\n"
                        "\t\tlocal part = parts[i]\n"
                        "\t\tlocal pos = part.Position\n"
                        "\t\tlocal delta = target - pos\n"
                        "\t\tlocal distSq = delta:Dot(delta)\n"
                        "\t\tif distSq > 0.01 then\n"
                        "\t\t\tlocal invDist = 1 / math.sqrt(distSq)\n"
                        "\t\t\tpart.Position = pos + delta * (invDist * speed * dt)\n"
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

    return patterns


def main():
    if not VERTIGO_SRC.exists():
        print(f"Error: Vertigo src not found at {VERTIGO_SRC}")
        return

    print(f"Scanning {VERTIGO_SRC}...")

    # Full module examples
    module_pairs = extract_module_pairs(VERTIGO_SRC)
    print(f"  Module examples: {len(module_pairs)}")

    # Isolated pattern examples
    pattern_pairs = extract_pattern_examples(VERTIGO_SRC)
    print(f"  Pattern examples: {len(pattern_pairs)}")

    all_pairs = module_pairs + pattern_pairs

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nTotal: {len(all_pairs)} training pairs -> {OUTPUT}")

    # Category breakdown
    from collections import Counter

    cats = Counter(p.get("category", "unknown") for p in all_pairs)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
