#!/usr/bin/env python3
from __future__ import annotations

"""
Record MCP tool-calling trajectories from a live Roblox Studio session.

Connects to the Studio MCP server, executes scripted exploration scenarios,
and records every request/response pair as ground-truth training data.

This is the highest-signal data source — real tool calls with real responses
from the actual running game.

Usage:
    python3 scripts/record_mcp_trajectories.py [--scenario all|explore|debug|build]
    python3 scripts/record_mcp_trajectories.py --list-scenarios
    python3 scripts/record_mcp_trajectories.py --dry-run

Requirements:
    - Roblox Studio running with MCP server enabled
    - Studio MCP server accessible (default: stdio via roblox-studio-mcp)

Output: data/raw/mcp_trajectories.jsonl
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "mcp_trajectories.jsonl"

SYSTEM_PROMPT = (
    "You are a Roblox Studio assistant for the Vertigo experience with access to MCP tools. "
    "You use tools to inspect, modify, and test the game. When asked to perform actions, "
    "think through which tools to use and in what order, then execute them."
)

# ---------------------------------------------------------------------------
# MCP Client (stdio transport)
# ---------------------------------------------------------------------------


@dataclass
class MCPClient:
    """Minimal MCP client using stdio transport."""

    process: subprocess.Popen | None = None
    request_id: int = 0

    def connect(self, command: str, args: list[str]):
        """Start the MCP server subprocess."""
        self.process = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Send initialize
        self._send(
            "initialize",
            {
                "protocolVersion": "2025-11-05",
                "capabilities": {},
                "clientInfo": {"name": "vertigo-lora-recorder", "version": "0.1.0"},
            },
        )
        resp = self._receive()
        if resp:
            print(f"  Connected. Server: {resp.get('result', {}).get('serverInfo', {}).get('name', 'unknown')}")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Call an MCP tool and return the result."""
        self._send("tools/call", {"name": name, "arguments": arguments})
        return self._receive()

    def list_tools(self) -> list[dict]:
        """List available tools."""
        self._send("tools/list", {})
        resp = self._receive()
        if resp and "result" in resp:
            return resp["result"].get("tools", [])
        return []

    def disconnect(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)

    def _send(self, method: str, params: dict):
        self.request_id += 1
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params,
            }
        )
        if self.process and self.process.stdin:
            self.process.stdin.write(msg + "\n")
            self.process.stdin.flush()

    def _receive(self) -> dict[str, Any] | None:
        if self.process and self.process.stdout:
            line = self.process.stdout.readline()
            if line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    return None
        return None


# ---------------------------------------------------------------------------
# Recording infrastructure
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryRecorder:
    """Records MCP tool call trajectories as training examples."""

    client: MCPClient
    trajectories: list[dict] = field(default_factory=list)
    tool_definitions: list[dict] = field(default_factory=list)

    def record_trajectory(
        self,
        user_instruction: str,
        reasoning: str,
        tool_calls: list[tuple[str, dict]],
        difficulty: int = 3,
    ):
        """Execute a sequence of tool calls and record the full trajectory."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_instruction},
        ]

        call_id = 0
        for tool_name, tool_args in tool_calls:
            call_id += 1
            call_id_str = f"call_{call_id}"

            # Record the assistant's tool call
            assistant_msg = {
                "role": "assistant",
                "content": reasoning if call_id == 1 else None,
                "tool_calls": [
                    {
                        "id": call_id_str,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            }
            messages.append(assistant_msg)

            # Execute the actual tool call
            result = self.client.call_tool(tool_name, tool_args)
            result_content = json.dumps(result.get("result", {}) if result else {"error": "no response"})

            # Record the tool response
            messages.append(
                {
                    "role": "tool",
                    "content": result_content,
                    "tool_call_id": call_id_str,
                }
            )

            # Brief pause between calls
            time.sleep(0.2)

        example = {
            "tools": self.tool_definitions,
            "messages": messages,
            "source": "mcp_trajectories",
            "category": "mcp_multi_step",
            "difficulty": difficulty,
            "has_reasoning": True,
            "verified": True,  # Ground truth from live Studio
        }
        self.trajectories.append(example)

    def save(self):
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w") as f:
            for t in self.trajectories:
                f.write(json.dumps(t) + "\n")
        print(f"Saved {len(self.trajectories)} trajectories -> {OUTPUT}")


# ---------------------------------------------------------------------------
# Exploration scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "explore": {
        "name": "World Exploration",
        "description": "Traverse the game tree, inspect zones, discover structure",
        "trajectories": [
            {
                "instruction": "Show me the full project structure of the Vertigo experience.",
                "reasoning": "I'll use get_file_tree to see the complete instance hierarchy.",
                "calls": [("get_file_tree", {})],
                "difficulty": 1,
            },
            {
                "instruction": "What zones exist under the SceneRoot?",
                "reasoning": "I'll get the children of Workspace.SceneRoot to see all zone models.",
                "calls": [("get_instance_children", {"path": "Workspace.SceneRoot"})],
                "difficulty": 1,
            },
            {
                "instruction": "How many GrappleAnchor tagged parts are in the game?",
                "reasoning": "I'll run code to count CollectionService tagged objects.",
                "calls": [
                    (
                        "run_code",
                        {
                            "code": 'local CS = game:GetService("CollectionService"); print("#GrappleAnchors:", #CS:GetTagged("GrappleAnchor"))'
                        },
                    )
                ],
                "difficulty": 2,
            },
            {
                "instruction": "Show me the properties of the Hub zone.",
                "reasoning": "I'll search for the Hub zone, then inspect its properties.",
                "calls": [
                    ("search_objects", {"query": "Hub"}),
                    ("get_instance_properties", {"path": "Workspace.SceneRoot.Hub"}),
                ],
                "difficulty": 2,
            },
            {
                "instruction": "List all RemoteEvents and RemoteFunctions in the game.",
                "reasoning": "I'll check ReplicatedStorage for the Remotes folder and list its children.",
                "calls": [
                    ("get_instance_children", {"path": "ReplicatedStorage.Remotes"}),
                ],
                "difficulty": 2,
            },
        ],
    },
    "debug": {
        "name": "Debug & Playtest",
        "description": "Start playtests, check for errors, diagnose issues",
        "trajectories": [
            {
                "instruction": "Run a quick playtest and check for any errors in the console.",
                "reasoning": "I'll start play mode, wait a moment, then check console output for errors.",
                "calls": [
                    ("start_stop_play", {"action": "start"}),
                    ("get_console_output", {}),
                    ("start_stop_play", {"action": "stop"}),
                ],
                "difficulty": 3,
            },
            {
                "instruction": "Check if the DataService initializes correctly during playtest.",
                "reasoning": "I'll start play mode, then check console for DataService init logs.",
                "calls": [
                    ("start_stop_play", {"action": "start"}),
                    (
                        "run_script_in_play_mode",
                        {
                            "code": 'print("[TEST] DataService loaded:", game:GetService("ServerScriptService"):FindFirstChild("Server") ~= nil)'
                        },
                    ),
                    ("get_console_output", {}),
                    ("start_stop_play", {"action": "stop"}),
                ],
                "difficulty": 3,
            },
            {
                "instruction": "Check the current Studio mode and verify we're in Edit mode.",
                "reasoning": "I'll use get_studio_mode to check the current state.",
                "calls": [("get_studio_mode", {})],
                "difficulty": 1,
            },
        ],
    },
    "build": {
        "name": "Build & Modify",
        "description": "Create objects, modify properties, build content",
        "trajectories": [
            {
                "instruction": "Create a new checkpoint part at position (50, 30, -100) and tag it.",
                "reasoning": "I'll create a Part, set its properties, then tag it with CollectionService via run_code.",
                "calls": [
                    (
                        "create_object",
                        {
                            "className": "Part",
                            "parent": "Workspace",
                            "properties": {
                                "Name": "Checkpoint_Test",
                                "Position": [50, 30, -100],
                                "Size": [4, 1, 4],
                                "Anchored": True,
                                "Material": "Neon",
                                "Color": [0.2, 0.8, 0.4],
                            },
                        },
                    ),
                    (
                        "run_code",
                        {
                            "code": 'game:GetService("CollectionService"):AddTag(workspace.Checkpoint_Test, "Checkpoint"); print("Tagged Checkpoint_Test")'
                        },
                    ),
                ],
                "difficulty": 3,
            },
            {
                "instruction": "Find all parts with Transparency > 0.5 and make them fully transparent.",
                "reasoning": "I'll search by property to find semi-transparent parts, then mass-set their transparency to 1.",
                "calls": [
                    ("search_by_property", {"property": "Transparency", "value": 0.5}),
                    (
                        "run_code",
                        {
                            "code": 'local count = 0; for _, d in workspace:GetDescendants() do if d:IsA("BasePart") and d.Transparency > 0.5 then d.Transparency = 1; count += 1 end end; print("Made", count, "parts fully transparent")'
                        },
                    ),
                ],
                "difficulty": 3,
            },
            {
                "instruction": "Insert the pine tree model (asset ID 123456) into the Hub zone.",
                "reasoning": "I'll use insert_model to add the asset from the Creator Store.",
                "calls": [
                    ("insert_model", {"assetId": 123456, "parent": "Workspace.SceneRoot.Hub"}),
                ],
                "difficulty": 2,
            },
        ],
    },
}


def run_scenario(recorder: TrajectoryRecorder, scenario_name: str, dry_run: bool = False):
    """Run all trajectories in a scenario."""
    scenario = SCENARIOS.get(scenario_name)
    if not scenario:
        print(f"Unknown scenario: {scenario_name}")
        return

    print(f"\n=== {scenario['name']} ===")
    print(f"  {scenario['description']}")

    for i, traj in enumerate(scenario["trajectories"], 1):
        print(f"\n  [{i}] {traj['instruction'][:70]}...")

        if dry_run:
            print(f"      Tools: {[c[0] for c in traj['calls']]}")
            continue

        try:
            recorder.record_trajectory(
                user_instruction=traj["instruction"],
                reasoning=traj["reasoning"],
                tool_calls=traj["calls"],
                difficulty=traj["difficulty"],
            )
            print(f"      Recorded ({len(traj['calls'])} tool calls)")
        except Exception as e:
            print(f"      Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Record MCP trajectories from live Studio")
    parser.add_argument("--scenario", default="all", help="Scenario to run: explore, debug, build, all")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions without executing")
    parser.add_argument("--mcp-command", default=None, help="MCP server command (default: from .mcp.json)")
    args = parser.parse_args()

    if args.list_scenarios:
        print("Available scenarios:")
        for name, scenario in SCENARIOS.items():
            count = len(scenario["trajectories"])
            print(f"  {name:<12} {scenario['name']:<25} ({count} trajectories)")
        total = sum(len(s["trajectories"]) for s in SCENARIOS.values())
        print(f"\n  Total: {total} trajectories across {len(SCENARIOS)} scenarios")
        return

    if args.dry_run:
        print("=== DRY RUN ===\n")
        scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
        total = 0
        for name in scenarios:
            scenario = SCENARIOS[name]
            count = len(scenario["trajectories"])
            print(f"Scenario: {name} ({count} trajectories)")
            for traj in scenario["trajectories"]:
                tools = [c[0] for c in traj["calls"]]
                print(f"  D{traj['difficulty']}: {traj['instruction'][:60]}... → {tools}")
            total += count
        print(f"\nTotal: {total} trajectories to record")
        return

    # Resolve MCP command
    mcp_cmd = args.mcp_command
    if not mcp_cmd:
        mcp_json = Path(__file__).resolve().parent.parent.parent / ".mcp.json"
        if mcp_json.exists():
            cfg = json.loads(mcp_json.read_text())
            studio_cfg = cfg.get("mcpServers", {}).get("roblox-studio-mcp", {})
            mcp_cmd = studio_cfg.get("command")
            mcp_args = studio_cfg.get("args", [])
        else:
            print("error: No .mcp.json found. Pass --mcp-command or configure .mcp.json")
            sys.exit(1)

    print("Connecting to Studio MCP server...")
    client = MCPClient()
    try:
        client.connect(mcp_cmd, mcp_args)
    except Exception as e:
        print(f"error: Could not connect to Studio MCP: {e}")
        print("Make sure Roblox Studio is running with the MCP plugin enabled.")
        sys.exit(1)

    # Get tool definitions for training data
    tools = client.list_tools()
    tool_defs = [{"type": "function", "function": t} for t in tools]

    recorder = TrajectoryRecorder(client=client, tool_definitions=tool_defs)

    # Run scenarios
    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    for name in scenarios:
        run_scenario(recorder, name)

    recorder.save()
    client.disconnect()

    print(f"\nDone. Total trajectories: {len(recorder.trajectories)}")


if __name__ == "__main__":
    main()
