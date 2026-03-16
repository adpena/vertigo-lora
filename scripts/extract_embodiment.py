#!/usr/bin/env python3
from __future__ import annotations

"""
Extract embodiment and gameplay training data from Vertigo's agent systems.

Generates training examples for the "player" LoRA adapter — teaching the model
to play the game using the embodiment SDK, control agents, execute abilities,
navigate zones, and interact with the world.

Data sources:
1. Agent service patterns (EmbodimentService, AgentMotionService, AgentPatrolService)
2. Ability execution sequences (grapple, glide, wallrun, swim chains)
3. Spatial reasoning (zone navigation, waypoint routing, anchor finding)
4. Agent command protocol (fleet MCP embodiment tools)
5. Gameplay trajectories (traversal chains, vehicle routing, puzzle solving)

Output: data/raw/embodiment_gameplay.jsonl
"""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "embodiment_gameplay.jsonl"

PLAYER_SYSTEM_PROMPT = (
    "You are an embodied AI agent playing the Vertigo experience. You control an R15 avatar "
    "in a physics-driven exploration game. You can:\n"
    "- Execute traversal abilities: grapple (180 stud range), glide (0.55 lift), wallrun, slide, airdash, swim\n"
    "- Drive vehicles: dirt bike (95 max speed, turbo 140), glider kite (0.78 lift)\n"
    "- Navigate zones: Abyss (-120 to -56), Ocean, Surface/Hub, Caves, Ascent, Mountains, Sky, AetherSea, QuietRing\n"
    "- Interact with world: activate shrines, discover landmarks, place beacons, trigger events\n"
    "- Use MCP embodiment tools: embody_enter, embody_exit, embody_observe, embody_improve\n\n"
    "Think spatially about traversal chains. Plan routes that chain abilities for maximum flow:\n"
    "grapple → release at peak → airdash → glide → wallrun → slide landing.\n"
    "The world is vertical — you move up through layers, discovering new zones."
)

EXAMPLES = [
    # --- Spatial Navigation (10) ---
    {
        "messages": [
            {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Navigate from the Hub zone (Y=0) to the Ascent zone (Y=80). Plan your traversal chain.",
            },
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    "The Hub is at surface level (Y≈0) and the Ascent zone starts around Y=80. That's 80 studs of vertical "
                    "gain. I need to chain abilities efficiently:\n\n"
                    "1. Find GrappleAnchors leading upward from the Hub dock area\n"
                    "2. Grapple to first anchor → swing for momentum → release at peak\n"
                    "3. Airdash upward (1 use per airborne cycle) for ~15 studs vertical\n"
                    "4. If near a wall, wallrun for additional height\n"
                    "5. At apex, deploy glide to maintain altitude while steering toward next anchor\n"
                    "6. Chain grapple → release → glide → grapple for sustained climbing\n"
                    "7. The grapple-to-wallrun grace window is 0.65s — need to be near a wall within that time\n\n"
                    "Key constraints:\n"
                    "- Grapple maxRange is 180 studs, so anchors within range are reachable\n"
                    "- Airdash resets on grounding, so landing briefly on platforms resets it\n"
                    "- Glide has a 0.6s entry grace from grapple release\n"
                    "</think>\n\n"
                    "Route plan from Hub (Y=0) to Ascent (Y=80):\n\n"
                    "**Phase 1: Initial ascent (Y=0 → Y=30)**\n"
                    "- From Hub dock, locate nearest GrappleAnchor (should be within 180 studs)\n"
                    "- `embody_observe` to scan for anchors in upper hemisphere\n"
                    "- Grapple → swing toward cliffside → release at upward velocity peak\n"
                    "- Airdash upward immediately after release (+15 studs)\n\n"
                    "**Phase 2: Mid-climb (Y=30 → Y=60)**\n"
                    "- Deploy glide within 0.6s of airdash\n"
                    "- Steer toward next visible anchor cluster\n"
                    "- Grapple mid-glide → new swing arc → release higher\n"
                    "- If near cliff face, wallrun for 3-5 additional studs\n\n"
                    "**Phase 3: Final approach (Y=60 → Y=80)**\n"
                    "- Land on Ascent platform to reset airdash\n"
                    "- Final grapple chain to Ascent checkpoint\n"
                    "- Estimated total time: 15-25 seconds with clean chains"
                ),
            },
        ],
        "source": "embodiment_gameplay",
        "category": "controller",
        "difficulty": 4,
        "has_reasoning": True,
    },
    {
        "messages": [
            {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": "You're falling into the Abyss (Y=-80). How do you recover?"},
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    "The Abyss starts at Y=-56 and goes to Y=-120. I'm at Y=-80, which is deep but not at the bottom.\n"
                    "Recovery options:\n"
                    "1. Look for GrappleAnchors above — grapple range is 180 studs, so anything up to Y=100 is reachable\n"
                    "2. If no anchors visible, deploy glide to slow descent (0.55 lift counters ~55% of gravity)\n"
                    "3. Airdash upward if available (resets on grounding)\n"
                    "4. If near Abyss walls, wallrun to gain height\n"
                    "5. Last resort: if below Y=-120, hazard volume respawns to last checkpoint\n\n"
                    "Priority: arrest downward velocity first, then gain height.\n"
                    "</think>\n\n"
                    "Emergency recovery at Y=-80:\n\n"
                    "1. **Immediate:** Deploy glide to arrest fall (reduces descent rate by 55%)\n"
                    "2. **Scan:** `embody_observe` — look for any GrappleAnchor within 180 studs above\n"
                    "3. **If anchor found:** Cancel glide → grapple → swing upward\n"
                    "4. **If no anchor:** Steer glide toward Abyss walls → wallrun for height gain\n"
                    "5. **If near Y=-120:** Accept respawn at last checkpoint — don't waste time in the void\n\n"
                    "The key insight: glide + grapple is the fastest recovery combo. "
                    "Glide slows you down enough to aim a grapple accurately."
                ),
            },
        ],
        "source": "embodiment_gameplay",
        "category": "controller",
        "difficulty": 3,
        "has_reasoning": True,
    },
    # --- Ability Execution (8) ---
    {
        "messages": [
            {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": "Execute a perfect grapple swing release for maximum distance."},
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    "A perfect grapple release maximizes horizontal distance. The physics:\n"
                    "- Grapple creates a pendulum with the anchor as pivot\n"
                    "- Maximum velocity occurs at the bottom of the swing arc\n"
                    "- Optimal release angle is ~45° past the lowest point (for max range)\n"
                    "- Slingshot impulse adds 132 studs/sec on release\n"
                    "- Centripetal assist (0.08 strength) helps maintain swing arc\n"
                    "- After release, airdash within 0.6s for additional boost\n"
                    "- Then glide to convert velocity into distance\n"
                    "</think>\n\n"
                    "Perfect grapple release sequence:\n\n"
                    "1. **Hook:** Target anchor at ~120 studs (not max range — leaves room for swing arc)\n"
                    "2. **Swing:** Let pendulum build — don't reel immediately\n"
                    "3. **Apex timing:** Release when your velocity vector points ~45° upward\n"
                    "4. **Slingshot:** The 132 stud/sec impulse fires on release automatically\n"
                    "5. **Chain:** Within 0.6s, deploy airdash in your movement direction\n"
                    "6. **Glide:** Within 0.6s of airdash, deploy glide to sustain altitude\n"
                    "7. **Result:** ~200+ studs of horizontal travel from a single grapple\n\n"
                    "Common mistake: reeling in during the swing. This shortens the pendulum "
                    "and reduces velocity. Let the rope length stay natural."
                ),
            },
        ],
        "source": "embodiment_gameplay",
        "category": "physics",
        "difficulty": 4,
        "has_reasoning": True,
    },
    {
        "messages": [
            {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": "Chain a slide into a wallrun into an airdash. What are the timing windows?"},
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    "Ability chaining requires hitting precise timing windows:\n"
                    "- Slide: requires grounded + horizontal speed > 8 studs/sec. Duration 0.8s.\n"
                    "- Slide → hop: slideHopWindowSeconds = 0.3s. Gives 0.08 speed boost per chain.\n"
                    "- Wallrun: requires airborne + wall within 3 studs. Spam guard 0.3s.\n"
                    "- Airdash: requires airborne + not used since last grounding. One per airborne cycle.\n"
                    "- The key: slide builds speed → jump off slide → wallrun on nearby surface → "
                    "  airdash at wallrun exit for maximum height\n"
                    "</think>\n\n"
                    "Slide → Wallrun → Airdash chain timing:\n\n"
                    "```\n"
                    "T=0.0s    SLIDE (need >8 studs/sec horizontal speed, grounded)\n"
                    "T=0.3s    SLIDE HOP (jump within 0.3s window for speed boost)\n"
                    "T=0.35s   WALLRUN (must be airborne + wall within 3 studs)\n"
                    "          → 0.3s spam guard prevents re-triggering\n"
                    "T=0.8s    WALLRUN EXIT (jump off wall)\n"
                    "T=0.85s   AIRDASH (airborne, haven't used since grounding)\n"
                    "T=1.1s    GLIDE (within 0.6s of airdash for chain bonus)\n"
                    "```\n\n"
                    "Total chain: ~1.1 seconds for massive height + distance. "
                    "The slideHopBoostPerChain (0.08) stacks, so consecutive slide-hops "
                    "build speed up to the slideHopSpeedCap of 120 studs/sec."
                ),
            },
        ],
        "source": "embodiment_gameplay",
        "category": "physics",
        "difficulty": 5,
        "has_reasoning": True,
    },
    # --- Embodiment SDK (8) ---
    {
        "messages": [
            {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": "Enter the game as the Director agent and patrol the Hub zone."},
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    "To embody the Director agent:\n"
                    "1. Call embody_enter with agentId and identity\n"
                    "2. Set up patrol route with Hub zone waypoints\n"
                    "3. Configure speed and idle times for natural movement\n"
                    "4. The EmbodimentService will create an R15 avatar\n"
                    "5. AgentPatrolService handles waypoint routing\n"
                    "6. Need to send heartbeats every <5s to avoid stale detection\n"
                    "</think>\n\n"
                    "Embodiment sequence:\n\n"
                    "1. **Enter:** `embody_enter(agentId='director', identity='Director')`\n"
                    "   - Creates R15 avatar in workspace\n"
                    "   - Binds to EmbodimentService registry\n\n"
                    "2. **Configure patrol:**\n"
                    "   ```\n"
                    "   route = {\n"
                    "     waypoints = [\n"
                    "       Vector3(0, 5, 0),      -- Hub center\n"
                    "       Vector3(30, 5, 20),     -- Dock area\n"
                    "       Vector3(-20, 5, -30),   -- Garage entrance\n"
                    "       Vector3(10, 8, 40),     -- Overlook point\n"
                    "     ],\n"
                    "     speed = 0.7,  -- Leisurely walk\n"
                    "     idleTime = 3, -- Pause 3s at each waypoint\n"
                    "   }\n"
                    "   ```\n\n"
                    "3. **Heartbeat:** Send every 3s to prevent 5s stale timeout\n\n"
                    "4. **Observe:** Periodically call `embody_observe` to react to nearby players"
                ),
            },
        ],
        "source": "embodiment_gameplay",
        "category": "service",
        "difficulty": 3,
        "has_reasoning": True,
    },
    {
        "messages": [
            {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": "As an embodied agent, demonstrate the grapple ability to nearby players."},
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    "To demonstrate grapple as an embodied agent:\n"
                    "1. Use embody_observe to find nearby players\n"
                    "2. Find the nearest GrappleAnchor\n"
                    "3. Execute grapple via AgentInterfaceService command protocol\n"
                    "4. Perform a clean swing and release for visual impact\n"
                    "5. The demonstration should look intentional and fluid\n"
                    "6. Need to use the Fleet MCP studio tools for execution\n"
                    "</think>\n\n"
                    "Grapple demonstration sequence:\n\n"
                    "1. **Scan:** `embody_observe` → check for players within 50 studs\n"
                    "2. **Position:** Move to a spot with visible GrappleAnchors above\n"
                    '3. **Announce:** (via NPC thought bubble) "Watch this!"\n'
                    "4. **Execute:**\n"
                    "   - `agent_cmd('director', 'ability', { id = 'ability_grapple_v1', target = nearest_anchor })`\n"
                    "   - Wait for hook travel (hookSpeed=260, ~0.5s for 130 studs)\n"
                    "   - Swing through 1.5 pendulum cycles for dramatic arc\n"
                    "   - Release at upward angle → slingshot impulse fires (132 studs/sec)\n"
                    "   - Deploy glide for graceful landing\n"
                    "5. **React:** `embody_observe` → check if players saw it\n"
                    "6. **Idle:** Resume patrol or approach nearest player"
                ),
            },
        ],
        "source": "embodiment_gameplay",
        "category": "service",
        "difficulty": 4,
        "has_reasoning": True,
    },
    # --- Zone Knowledge (6) ---
    {
        "messages": [
            {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Describe the vertical layer structure of the Vertigo world and how to traverse between layers.",
            },
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    'Vertigo\'s world is a vertical "layer cake" with distinct biomes at different altitudes. '
                    "Each layer gates certain abilities — you need to unlock traversal tools to reach higher zones. "
                    "The progression is designed so that each new ability opens access to the next layer.\n"
                    "</think>\n\n"
                    "Vertigo's vertical world structure (bottom to top):\n\n"
                    "| Layer | Y Range | Zone | Required Ability |\n"
                    "|-------|---------|------|------------------|\n"
                    "| Abyss | -120 to -56 | Deep void, bioluminescent | Swim (ability_swim_v1) |\n"
                    "| Ocean | -56 to -10 | Kelp forests, reefs, coral | Swim |\n"
                    "| Surface/Hub | -10 to 20 | Central dock, garage, social | Base kit (slide, jump) |\n"
                    "| Caves | 0 to 40 | Crystal caverns, puzzles | Base kit |\n"
                    "| Ascent | 40 to 100 | Mountain climbing, cliffs | Grapple (ability_grapple_v1) |\n"
                    "| Mountains | 80 to 140 | Peaks, wind volumes | Grapple + Glide |\n"
                    "| Sky | 120 to 180 | Floating islands, thermals | Glide (ability_glide_v1) |\n"
                    "| AetherSea | 160 to 210 | Above-cloud realm | All abilities |\n"
                    "| QuietRing | 200 to 230 | Secret end-game space | Full mastery |\n\n"
                    "**Key transitions:**\n"
                    "- Hub → Ascent: Grapple chain up cliff face (GrappleAnchors placed along path)\n"
                    "- Ascent → Mountains: Grapple + glide combo over peaks\n"
                    "- Mountains → Sky: Thermal columns (WindVolumes) boost glide altitude\n"
                    "- Sky → AetherSea: Vehicle (glider kite) through cloud layer\n"
                    "- AetherSea → QuietRing: Hidden path requiring all abilities chained perfectly"
                ),
            },
        ],
        "source": "embodiment_gameplay",
        "category": "general_luau",
        "difficulty": 2,
        "has_reasoning": True,
    },
    # --- Vehicle Gameplay (4) ---
    {
        "messages": [
            {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": "How do I execute a perfect dirt bike drift with maximum quality score?"},
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    "Dirt bike drift scoring in Vertigo:\n"
                    "- DirtBikeTuning: maxSpeed=95, turbo=140, driftTurnMultiplier=1.5\n"
                    "- Drift quality depends on: angle consistency, speed maintenance, duration\n"
                    "- Enter drift at >60 studs/sec for best results\n"
                    "- Anti-tip stability keeps the bike upright during drift\n"
                    "- Score multiplier increases with continuous drift time\n"
                    "</think>\n\n"
                    "Perfect drift technique:\n\n"
                    "1. **Approach speed:** Hit 80+ studs/sec on a straight (turbo helps)\n"
                    "2. **Initiate:** Sharp turn input + brake tap to break rear traction\n"
                    "3. **Sustain:** Counter-steer to maintain ~30-45° drift angle\n"
                    "   - driftTurnMultiplier (1.5x) amplifies steering during drift\n"
                    "   - Keep throttle at ~70% to maintain speed without spinning out\n"
                    "4. **Duration bonus:** Hold drift >2 seconds for quality multiplier\n"
                    "5. **Exit clean:** Straighten steering gradually, release brake\n"
                    "6. **Chain:** Immediately initiate next drift for combo scoring\n\n"
                    "Server speed guard: stay under softLimit (155) to avoid warnings. "
                    "Hard limit (195) will teleport you back."
                ),
            },
        ],
        "source": "embodiment_gameplay",
        "category": "physics",
        "difficulty": 3,
        "has_reasoning": True,
    },
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for ex in EXAMPLES:
            f.write(json.dumps(ex) + "\n")

    from collections import Counter

    cats = Counter(ex["category"] for ex in EXAMPLES)
    diffs = Counter(ex["difficulty"] for ex in EXAMPLES)

    print(f"Generated {len(EXAMPLES)} embodiment/gameplay examples -> {OUTPUT}")
    print(f"\nCategories: {dict(cats)}")
    print(f"Difficulties: {dict(sorted(diffs.items()))}")


if __name__ == "__main__":
    main()
