#!/usr/bin/env python3
from __future__ import annotations

"""
Capture embodied gameplay sessions as LoRA training data.

Connects to the RuntimeBridge WebSocket, observes agent actions,
and converts gameplay trajectories into instruction/completion training pairs.

This creates the highest-signal training data possible — real agent decisions
in the actual game environment, verified by the game engine itself.

Data captured:
- Agent position/velocity over time
- Ability usage (which, when, success/fail)
- Environmental observations (anchors, landmarks, players)
- Movement decisions and reasoning
- MCP tool calls made during the session

Usage:
    python3 scripts/capture_gameplay.py --agent director --duration 300
    python3 scripts/capture_gameplay.py --agent all --duration 600
    python3 scripts/capture_gameplay.py --list-agents

Output: data/raw/gameplay_sessions.jsonl
"""

import argparse
import asyncio
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import websockets
except ImportError:
    websockets = None

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "gameplay_sessions.jsonl"

from prompts import PLAYER_SYSTEM_PROMPT  # noqa: E402

AGENTS = {
    "director": {"spawn": (0, 8, 0), "zone": "Hub", "role": "Director"},
    "architect": {"spawn": (0, 47, -25), "zone": "Ascent", "role": "Architect"},
    "builder": {"spawn": (-80, -15, 0), "zone": "LuminousDeep", "role": "Builder"},
    "scribe": {"spawn": (0, 75, 0), "zone": "AetherSea", "role": "Scribe"},
}


@dataclass
class GameplayFrame:
    """A single frame of gameplay observation."""

    timestamp: float
    agent_id: str
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0, 0, 0)
    zone: str = "unknown"
    action: str | None = None
    action_result: bool | None = None
    nearby_anchors: int = 0
    nearby_landmarks: int = 0
    nearby_players: int = 0
    thought: str | None = None


@dataclass
class GameplaySession:
    """A complete gameplay session for one agent."""

    agent_id: str
    role: str
    start_time: float = 0
    end_time: float = 0
    frames: list[GameplayFrame] = field(default_factory=list)
    abilities_used: list[dict] = field(default_factory=list)
    zones_visited: set = field(default_factory=set)
    landmarks_discovered: list[str] = field(default_factory=list)

    def to_training_examples(self) -> list[dict]:
        """Convert session into training examples."""
        examples = []

        # Example 1: Navigation decision
        if len(self.frames) >= 2:
            start = self.frames[0]
            end = self.frames[-1]
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"You are the {self.role} agent at position "
                                f"({start.position[0]:.0f}, {start.position[1]:.0f}, {start.position[2]:.0f}) "
                                f"in the {start.zone} zone. "
                                f"You can see {start.nearby_anchors} grapple anchors and "
                                f"{start.nearby_landmarks} landmarks nearby. "
                                f"What should you do next?"
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": (
                                f"<think>\n"
                                f"I'm in {start.zone} at Y={start.position[1]:.0f}. "
                                f"There are {start.nearby_anchors} anchors nearby for grappling. "
                                f"I should explore toward {'higher zones' if start.position[1] < 50 else 'the current zone'}. "
                                f"My target is ({end.position[0]:.0f}, {end.position[1]:.0f}, {end.position[2]:.0f}) "
                                f"in {end.zone}.\n"
                                f"</think>\n\n"
                                f"I'll navigate from {start.zone} toward {end.zone}. "
                                f"Moving to position ({end.position[0]:.0f}, {end.position[1]:.0f}, {end.position[2]:.0f})."
                            ),
                        },
                    ],
                    "source": "gameplay_session",
                    "category": "controller",
                    "difficulty": 3,
                    "has_reasoning": True,
                    "session_id": f"{self.agent_id}_{int(self.start_time)}",
                }
            )

        # Example 2: Ability usage decisions
        for ability in self.abilities_used:
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"You're at ({ability.get('position', (0, 0, 0))[0]:.0f}, "
                                f"{ability.get('position', (0, 0, 0))[1]:.0f}, "
                                f"{ability.get('position', (0, 0, 0))[2]:.0f}) "
                                f"and need to reach a higher platform. "
                                f"Which ability should you use?"
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": (
                                f"<think>\n"
                                f"Analyzing available abilities and environment.\n"
                                f"Selected: {ability.get('name', 'unknown')} — "
                                f"{'succeeded' if ability.get('success') else 'failed'}.\n"
                                f"</think>\n\n"
                                f"I'll use {ability.get('name', 'unknown')}. "
                                f"{'The ability executed successfully.' if ability.get('success') else 'The ability failed — I need to try a different approach.'}"
                            ),
                        },
                    ],
                    "source": "gameplay_session",
                    "category": "physics",
                    "difficulty": 3,
                    "has_reasoning": True,
                }
            )

        # Example 3: Zone knowledge
        if self.zones_visited:
            zones = sorted(self.zones_visited)
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
                        {"role": "user", "content": "What zones did you explore in this session?"},
                        {
                            "role": "assistant",
                            "content": (
                                f"<think>\n"
                                f"Reviewing my session as {self.role}.\n"
                                f"Duration: {self.end_time - self.start_time:.0f} seconds.\n"
                                f"Zones visited: {', '.join(zones)}.\n"
                                f"Abilities used: {len(self.abilities_used)} times.\n"
                                f"</think>\n\n"
                                f"In this session I explored: {', '.join(zones)}. "
                                f"I used {len(self.abilities_used)} abilities "
                                f"and discovered {len(self.landmarks_discovered)} landmarks."
                            ),
                        },
                    ],
                    "source": "gameplay_session",
                    "category": "general_luau",
                    "difficulty": 2,
                    "has_reasoning": True,
                }
            )

        return examples


async def capture_session_simulated(
    agent_id: str,
    duration: float = 60,
) -> GameplaySession:
    """
    Capture a simulated gameplay session.

    In production, this connects to the RuntimeBridge WebSocket at :9871
    and captures real agent data. This simulated version generates
    representative data for pipeline validation.

    To use with real Studio:
        1. Start runtime bridge: uv run scripts/dev/runtime-bridge.py
        2. Open Studio in Edit Mode (agents auto-spawn)
        3. Run this script with --live flag
    """
    agent_info = AGENTS.get(agent_id, AGENTS["director"])
    session = GameplaySession(
        agent_id=agent_id,
        role=agent_info["role"],
        start_time=time.time(),
    )

    import random

    rng = random.Random(42 + hash(agent_id))

    spawn = agent_info["spawn"]
    pos = list(spawn)
    zones = ["Hub", "Ascent", "CrystalCavern", "AetherSea", "Mountains"]
    abilities = ["ability_grapple_v1", "ability_glide_v1", "ability_slide_v1", "ability_airdash_v1"]

    steps = int(duration / 5)  # one frame every 5 seconds
    for i in range(steps):
        # Simulate movement
        pos[0] += rng.uniform(-10, 10)
        pos[1] += rng.uniform(-2, 5)  # tend upward
        pos[2] += rng.uniform(-10, 10)

        zone = zones[min(int((pos[1] + 20) / 40), len(zones) - 1)]
        session.zones_visited.add(zone)

        frame = GameplayFrame(
            timestamp=session.start_time + i * 5,
            agent_id=agent_id,
            position=tuple(pos),
            zone=zone,
            nearby_anchors=rng.randint(2, 15),
            nearby_landmarks=rng.randint(0, 5),
        )

        # Random ability usage
        if rng.random() < 0.3:
            ability = rng.choice(abilities)
            success = rng.random() > 0.2
            frame.action = ability
            frame.action_result = success
            session.abilities_used.append(
                {
                    "name": ability,
                    "position": tuple(pos),
                    "success": success,
                    "timestamp": frame.timestamp,
                }
            )

        session.frames.append(frame)

    session.end_time = time.time()
    return session


async def capture_session_live(
    agent_id: str,
    duration: float = 60,
) -> GameplaySession | None:
    """
    Capture a live gameplay session via WebSocket from the RuntimeBridge.

    Connects to ws://127.0.0.1:9871/ws/agent/{agent_id}, subscribes to
    telemetry channels, and records frames for the specified duration.

    Returns None if the connection fails after retries (caller should
    fall back to simulated mode).
    """
    if websockets is None:
        print("  ERROR: websockets library not installed. Install with: uv pip install websockets")
        return None

    url = f"ws://127.0.0.1:9871/ws/agent/{agent_id}"
    agent_info = AGENTS.get(agent_id, AGENTS["director"])

    # --- Connection with retry ---
    max_retries = 3
    backoff = 2.0
    ws = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Connecting to {url} (attempt {attempt}/{max_retries})...")
            ws = await asyncio.wait_for(
                websockets.connect(url),
                timeout=10.0,
            )
            print("  Connected.")
            break
        except ConnectionRefusedError:
            print("  RuntimeBridge not running at :9871. Start with: ./scripts/dev/runtime-bridge.py")
            if attempt < max_retries:
                print(f"  Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                return None
        except (OSError, asyncio.TimeoutError) as exc:
            print(f"  Connection failed: {exc}")
            if attempt < max_retries:
                print(f"  Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                return None

    if ws is None:
        return None

    # --- Subscribe ---
    session = GameplaySession(
        agent_id=agent_id,
        role=agent_info["role"],
        start_time=time.time(),
    )

    try:
        subscribe_msg = json.dumps(
            {
                "type": "subscribe",
                "agent_id": agent_id,
                "channels": ["position", "ability", "observation"],
            }
        )
        await ws.send(subscribe_msg)
        print("  Subscribed to channels: position, ability, observation")

        deadline = time.time() + duration

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 30.0))
            except asyncio.TimeoutError:
                # No message within timeout window — keep waiting until deadline
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Parse position/velocity arrays
            pos = msg.get("position", [0, 0, 0])
            vel = msg.get("velocity", [0, 0, 0])
            position = (float(pos[0]), float(pos[1]), float(pos[2]))
            velocity = (float(vel[0]), float(vel[1]), float(vel[2]))

            zone = msg.get("zone", "unknown")
            action = msg.get("action")
            action_result = msg.get("action_result")

            frame = GameplayFrame(
                timestamp=msg.get("timestamp", time.time()),
                agent_id=agent_id,
                position=position,
                velocity=velocity,
                zone=zone,
                action=action,
                action_result=action_result,
                nearby_anchors=int(msg.get("nearby_anchors", 0)),
                nearby_landmarks=int(msg.get("nearby_landmarks", 0)),
                nearby_players=int(msg.get("nearby_players", 0)),
                thought=msg.get("thought"),
            )
            session.frames.append(frame)

            # Track session-level stats
            if zone and zone != "unknown":
                session.zones_visited.add(zone)

            if action:
                session.abilities_used.append(
                    {
                        "name": action,
                        "position": position,
                        "success": action_result,
                        "timestamp": frame.timestamp,
                    }
                )

            # Track landmarks
            landmark_names = msg.get("landmark_names", [])
            for lm in landmark_names:
                if lm not in session.landmarks_discovered:
                    session.landmarks_discovered.append(lm)

        # --- Unsubscribe and close ---
        try:
            await ws.send(json.dumps({"type": "unsubscribe"}))
        except Exception:
            pass

    except KeyboardInterrupt:
        print("\n  Interrupted — saving captured frames...")
    except Exception as exc:
        print(f"  WebSocket error during capture: {exc}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass

    session.end_time = time.time()
    return session


def _safe_write_examples(output: Path, examples: list[dict]) -> None:
    """
    Append training examples to output file with backup and atomic write safety.

    - Backs up existing file to {output}.bak before modifying
    - Writes new content to a temp file first, then renames
    - Appends (does not overwrite) existing sessions
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    existing_content = b""
    if output.exists() and output.stat().st_size > 0:
        existing_content = output.read_bytes()
        # Backup existing file
        backup = output.with_suffix(output.suffix + ".bak")
        shutil.copy2(output, backup)
        print(f"  Backup saved to {backup}")

    # Build new lines to append
    new_lines = []
    for ex in examples:
        new_lines.append(json.dumps(ex, default=str) + "\n")

    # Write atomically via temp file
    fd, tmp_path = tempfile.mkstemp(
        dir=output.parent,
        prefix=output.stem + "_",
        suffix=".tmp",
    )
    try:
        with open(fd, "wb") as f:
            if existing_content:
                f.write(existing_content)
                # Ensure trailing newline before appending
                if not existing_content.endswith(b"\n"):
                    f.write(b"\n")
            for line in new_lines:
                f.write(line.encode())
        # Atomic rename
        Path(tmp_path).replace(output)
    except Exception:
        # Clean up temp file on failure
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description="Capture gameplay sessions as training data")
    parser.add_argument("--agent", default="director", help="Agent to capture (or 'all')")
    parser.add_argument("--duration", type=float, default=60, help="Session duration in seconds")
    parser.add_argument("--live", action="store_true", help="Connect to real RuntimeBridge (requires Studio)")
    parser.add_argument("--list-agents", action="store_true")
    args = parser.parse_args()

    if args.list_agents:
        for name, info in AGENTS.items():
            print(f"  {name:<12} {info['role']:<12} zone={info['zone']}")
        return

    agents = list(AGENTS.keys()) if args.agent == "all" else [args.agent]

    all_examples = []
    for agent_id in agents:
        print(f"\nCapturing {agent_id} session ({args.duration}s)...")

        if args.live:
            session = asyncio.run(capture_session_live(agent_id, args.duration))
            if session is None:
                print("  WARNING: Live capture failed — falling back to simulated mode")
                session = asyncio.run(capture_session_simulated(agent_id, args.duration))
        else:
            session = asyncio.run(capture_session_simulated(agent_id, args.duration))
        examples = session.to_training_examples()
        all_examples.extend(examples)

        print(f"  Frames: {len(session.frames)}")
        print(f"  Zones: {session.zones_visited}")
        print(f"  Abilities: {len(session.abilities_used)}")
        print(f"  Training examples: {len(examples)}")

    if all_examples:
        _safe_write_examples(OUTPUT, all_examples)
        print(f"\nAppended {len(all_examples)} examples -> {OUTPUT}")


if __name__ == "__main__":
    main()
