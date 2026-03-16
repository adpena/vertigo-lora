#!/usr/bin/env python3
from __future__ import annotations

"""
Build MCP tool-calling training examples using MLX's native tools format.

Follows the TOUCAN/ToolACE research patterns:
- Uses proper OpenAI-compatible tool definitions with JSON Schema parameters
- Includes multi-step tool chains
- Includes error recovery scenarios
- Includes "when NOT to call tools" examples (NVIDIA When2Call pattern)
- Reasoning traces explain tool selection logic

Output: data/raw/mcp_tools.jsonl
"""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "mcp_tools.jsonl"

# ---------------------------------------------------------------------------
# Roblox Studio MCP Tool Definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_file_tree",
            "description": "Get the full project file/instance tree from Roblox Studio",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_objects",
            "description": "Search for instances by name pattern in the Studio data model",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Name or pattern to search for"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_property",
            "description": "Find instances with a specific property value",
            "parameters": {
                "type": "object",
                "properties": {
                    "property": {"type": "string", "description": "Property name"},
                    "value": {"description": "Expected property value"},
                },
                "required": ["property", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_instance_children",
            "description": "List all children of a specific instance path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Instance path (e.g., 'Workspace.SceneRoot')"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_instance_properties",
            "description": "Get all properties of an instance",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Instance path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_object",
            "description": "Create a new instance in the data model",
            "parameters": {
                "type": "object",
                "properties": {
                    "className": {"type": "string", "description": "Roblox class name (Part, Model, Script, etc.)"},
                    "parent": {"type": "string", "description": "Parent instance path"},
                    "properties": {"type": "object", "description": "Initial property values"},
                },
                "required": ["className", "parent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mass_create_objects",
            "description": "Create multiple instances atomically (supports undo as single operation)",
            "parameters": {
                "type": "object",
                "properties": {
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "className": {"type": "string"},
                                "parent": {"type": "string"},
                                "properties": {"type": "object"},
                            },
                        },
                    },
                },
                "required": ["objects"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_object",
            "description": "Remove an instance from the data model",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Instance path to delete"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_property",
            "description": "Set a property value on an instance",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "property": {"type": "string"},
                    "value": {"description": "New value"},
                },
                "required": ["path", "property", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mass_set_property",
            "description": "Set a property on multiple instances atomically",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "property": {"type": "string"},
                    "value": {"description": "New value"},
                },
                "required": ["paths", "property", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Luau code in Studio's command bar context",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Luau code to execute"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_script_in_play_mode",
            "description": "Execute Luau code during an active playtest session",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_stop_play",
            "description": "Start or stop a playtest session in Studio",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "stop"]},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_console_output",
            "description": "Get recent output from the Studio console/output window",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_model",
            "description": "Insert a model from the Creator Store by asset ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "assetId": {"type": "number", "description": "Creator Store asset ID"},
                    "parent": {"type": "string", "description": "Parent instance path"},
                },
                "required": ["assetId", "parent"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a Roblox Studio assistant for the Vertigo experience with access to MCP tools. "
    "Vertigo is a physics-driven exploration game with:\n"
    "- 56 procedural zone builders (Abyss, Ocean, Hub, Ascent, AetherSea, etc.)\n"
    "- Traversal abilities: grapple (180 stud range), glide (0.55 lift), wallrun, swim, slide, airdash\n"
    "- Vehicles: dirt bike (95 max speed), glider kite (0.78 lift)\n"
    "- Service/controller architecture with Init/Start lifecycle\n"
    "- CollectionService tags: GrappleAnchor, WindVolume, CurrentVolume, Checkpoint, etc.\n"
    "- DataStore schema v7 with auto-save and migration\n\n"
    "Think through which tools to use and in what order. Use the minimum number of tool calls needed."
)


def make_tool_call(call_id: str, name: str, args: dict) -> dict:
    """Create a tool_call in OpenAI format."""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


def build_scenarios() -> list[dict]:
    """Build diverse MCP tool-calling training scenarios."""
    scenarios = []
    call_counter = [0]

    def next_id():
        call_counter[0] += 1
        return f"call_{call_counter[0]}"

    def cur_id():
        return f"call_{call_counter[0]}"

    # ===================================================================
    # SINGLE-TOOL EXAMPLES (10 scenarios)
    # ===================================================================

    # 1. get_file_tree
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Show me the project structure."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(next_id(), "get_file_tree", {}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_single_tool",
            "difficulty": 1,
        }
    )

    # 2. get_instance_children
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What scripts are in ServerScriptService?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(next_id(), "get_instance_children", {"path": "ServerScriptService"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_single_tool",
            "difficulty": 1,
        }
    )

    # 3. get_instance_properties
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What are the properties of the Baseplate?"},
                {
                    "role": "assistant",
                    "content": "I'll look up the properties of the Baseplate part.",
                    "tool_calls": [
                        make_tool_call(next_id(), "get_instance_properties", {"path": "Workspace.Baseplate"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_single_tool",
            "difficulty": 1,
        }
    )

    # 4. search_objects
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Find all parts named 'SpawnLocation' in the game."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(next_id(), "search_objects", {"query": "SpawnLocation"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_single_tool",
            "difficulty": 1,
        }
    )

    # 5. search_by_property
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Which parts in the game are unanchored?"},
                {
                    "role": "assistant",
                    "content": ("I'll search for parts where Anchored is false. Note this may return a large set."),
                    "tool_calls": [
                        make_tool_call(next_id(), "search_by_property", {"property": "Anchored", "value": False}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_single_tool",
            "difficulty": 1,
        }
    )

    # 6. get_console_output
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Are there any errors in the output console right now?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(next_id(), "get_console_output", {}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_single_tool",
            "difficulty": 1,
        }
    )

    # 7. insert_model
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Insert the tree model (asset ID 12345678) into the Hub zone."},
                {
                    "role": "assistant",
                    "content": "I'll insert that model from the Creator Store into the Hub zone folder.",
                    "tool_calls": [
                        make_tool_call(
                            next_id(), "insert_model", {"assetId": 12345678, "parent": "Workspace.SceneRoot.HubZone"}
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_single_tool",
            "difficulty": 1,
        }
    )

    # 8. delete_object
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Remove the old test part at Workspace.SceneRoot.TestPart."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(next_id(), "delete_object", {"path": "Workspace.SceneRoot.TestPart"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_single_tool",
            "difficulty": 1,
        }
    )

    # 9. set_property
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Change the color of Workspace.SceneRoot.HubZone.Beacon to bright red."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "set_property",
                            {
                                "path": "Workspace.SceneRoot.HubZone.Beacon",
                                "property": "Color",
                                "value": [1, 0, 0],
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_single_tool",
            "difficulty": 1,
        }
    )

    # 10. mass_set_property
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Make all three zone markers transparent (0.8 transparency)."},
                {
                    "role": "assistant",
                    "content": ("I'll use mass_set_property to update all three markers in one atomic operation."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "mass_set_property",
                            {
                                "paths": [
                                    "Workspace.SceneRoot.HubZone.Marker",
                                    "Workspace.SceneRoot.AscentZone.Marker",
                                    "Workspace.SceneRoot.AbyssZone.Marker",
                                ],
                                "property": "Transparency",
                                "value": 0.8,
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_single_tool",
            "difficulty": 2,
        }
    )

    # ===================================================================
    # MULTI-STEP CHAINS (15 scenarios)
    # ===================================================================

    # 11. Search -> inspect -> modify chain
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Find the DirtBike template and increase its max speed property."},
                {
                    "role": "assistant",
                    "content": (
                        "I'll first search for the DirtBike template, then inspect its properties to find "
                        "the current max speed, and then update it."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "search_objects", {"query": "DirtBike"}),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"results": [{"path": "Workspace.VehicleTemplates.DirtBike", "className": "Model"}]}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(), "get_instance_properties", {"path": "Workspace.VehicleTemplates.DirtBike"}
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"properties": {"Name": "DirtBike", "MaxSpeed": 95, "Acceleration": 40, "TurnSpeed": 2.5}}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": ("The DirtBike template currently has MaxSpeed = 95. I'll increase it to 110."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "set_property",
                            {
                                "path": "Workspace.VehicleTemplates.DirtBike",
                                "property": "MaxSpeed",
                                "value": 110,
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 3,
        }
    )

    # 12. Create -> tag -> verify chain
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Add a new checkpoint at position (100, 50, -200) in the Ascent zone "
                        "and tag it with CollectionService."
                    ),
                },
                {
                    "role": "assistant",
                    "content": ("I'll create the checkpoint part, then run code to add the CollectionService tag."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "create_object",
                            {
                                "className": "Part",
                                "parent": "Workspace.SceneRoot.AscentZone",
                                "properties": {
                                    "Name": "Checkpoint_Ascent_New",
                                    "Position": [100, 50, -200],
                                    "Size": [4, 1, 4],
                                    "Anchored": True,
                                    "Transparency": 0.5,
                                    "Material": "Neon",
                                },
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"success": true, "path": "Workspace.SceneRoot.AscentZone.Checkpoint_Ascent_New"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": "Now I'll add the Checkpoint tag via CollectionService.",
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'game:GetService("CollectionService"):AddTag('
                                    'workspace.SceneRoot.AscentZone.Checkpoint_Ascent_New, "Checkpoint")\n'
                                    'print("Tagged Checkpoint_Ascent_New as Checkpoint")'
                                ),
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"output": "Tagged Checkpoint_Ascent_New as Checkpoint"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "get_instance_properties",
                            {"path": "Workspace.SceneRoot.AscentZone.Checkpoint_Ascent_New"},
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"properties": {"Name": "Checkpoint_Ascent_New", "Position": [100, 50, -200], "Size": [4, 1, 4], "Anchored": true, "Transparency": 0.5, "Material": "Neon", "Tags": ["Checkpoint"]}}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Done. Created Checkpoint_Ascent_New at (100, 50, -200) in the Ascent zone with the "
                        "Checkpoint CollectionService tag. Verified the properties are set correctly."
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 3,
        }
    )

    # 13. Playtest -> debug -> fix -> retest
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Run a quick playtest and check for any errors."},
                {
                    "role": "assistant",
                    "content": ("I'll start a playtest, wait a moment, check the console for errors, then stop."),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "start"}),
                    ],
                },
                {"role": "tool", "content": '{"success": true, "status": "playing"}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(next_id(), "get_console_output", {}),
                    ],
                },
                {
                    "role": "tool",
                    "content": (
                        '{"output": ["[INFO] DataService:Init complete", '
                        '"[INFO] AbilityService:Init complete", '
                        '"[ERROR] VehicleService:Start - Failed to find DirtBike template: '
                        'Workspace.VehicleTemplates.DirtBike not found", '
                        '"[INFO] 37 services started"]}'
                    ),
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Found an error: VehicleService can't find the DirtBike template at "
                        "`Workspace.VehicleTemplates.DirtBike`. Let me stop the playtest and investigate."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "stop"}),
                    ],
                },
                {"role": "tool", "content": '{"success": true, "status": "stopped"}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(next_id(), "search_objects", {"query": "DirtBike"}),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"results": [{"path": "ServerStorage.VehicleTemplates.DirtBike", "className": "Model"}]}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Found it -- the DirtBike template is in ServerStorage.VehicleTemplates, not "
                        "Workspace.VehicleTemplates. The service is looking in the wrong location. "
                        "This is a code-level fix in VehicleService. The template path constant needs updating."
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 4,
        }
    )

    # 14. Mass operations with mass_create_objects
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Create a ring of 8 grapple anchor points around the Hub at radius 60 studs, "
                        "Y=25, evenly spaced."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'll use mass_create_objects to create all 8 anchors atomically. "
                        "Positions calculated as cos/sin at 45-degree increments around (0, 25, 0)."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "mass_create_objects",
                            {
                                "objects": [
                                    {
                                        "className": "Part",
                                        "parent": "Workspace.SceneRoot.HubZone",
                                        "properties": {
                                            "Name": "GrappleAnchor_Ring_1",
                                            "Position": [60, 25, 0],
                                            "Size": [3, 3, 3],
                                            "Anchored": True,
                                            "Shape": "Ball",
                                        },
                                    },
                                    {
                                        "className": "Part",
                                        "parent": "Workspace.SceneRoot.HubZone",
                                        "properties": {
                                            "Name": "GrappleAnchor_Ring_2",
                                            "Position": [42.4, 25, 42.4],
                                            "Size": [3, 3, 3],
                                            "Anchored": True,
                                            "Shape": "Ball",
                                        },
                                    },
                                    {
                                        "className": "Part",
                                        "parent": "Workspace.SceneRoot.HubZone",
                                        "properties": {
                                            "Name": "GrappleAnchor_Ring_3",
                                            "Position": [0, 25, 60],
                                            "Size": [3, 3, 3],
                                            "Anchored": True,
                                            "Shape": "Ball",
                                        },
                                    },
                                    {
                                        "className": "Part",
                                        "parent": "Workspace.SceneRoot.HubZone",
                                        "properties": {
                                            "Name": "GrappleAnchor_Ring_4",
                                            "Position": [-42.4, 25, 42.4],
                                            "Size": [3, 3, 3],
                                            "Anchored": True,
                                            "Shape": "Ball",
                                        },
                                    },
                                    {
                                        "className": "Part",
                                        "parent": "Workspace.SceneRoot.HubZone",
                                        "properties": {
                                            "Name": "GrappleAnchor_Ring_5",
                                            "Position": [-60, 25, 0],
                                            "Size": [3, 3, 3],
                                            "Anchored": True,
                                            "Shape": "Ball",
                                        },
                                    },
                                    {
                                        "className": "Part",
                                        "parent": "Workspace.SceneRoot.HubZone",
                                        "properties": {
                                            "Name": "GrappleAnchor_Ring_6",
                                            "Position": [-42.4, 25, -42.4],
                                            "Size": [3, 3, 3],
                                            "Anchored": True,
                                            "Shape": "Ball",
                                        },
                                    },
                                    {
                                        "className": "Part",
                                        "parent": "Workspace.SceneRoot.HubZone",
                                        "properties": {
                                            "Name": "GrappleAnchor_Ring_7",
                                            "Position": [0, 25, -60],
                                            "Size": [3, 3, 3],
                                            "Anchored": True,
                                            "Shape": "Ball",
                                        },
                                    },
                                    {
                                        "className": "Part",
                                        "parent": "Workspace.SceneRoot.HubZone",
                                        "properties": {
                                            "Name": "GrappleAnchor_Ring_8",
                                            "Position": [42.4, 25, -42.4],
                                            "Size": [3, 3, 3],
                                            "Anchored": True,
                                            "Shape": "Ball",
                                        },
                                    },
                                ],
                            },
                        ),
                    ],
                },
                {"role": "tool", "content": '{"success": true, "created": 8}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": "Now I'll tag them all as GrappleAnchors via CollectionService.",
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local CS = game:GetService("CollectionService")\n'
                                    "for i = 1, 8 do\n"
                                    '    local anchor = workspace.SceneRoot.HubZone:FindFirstChild("GrappleAnchor_Ring_" .. i)\n'
                                    '    if anchor then CS:AddTag(anchor, "GrappleAnchor") end\n'
                                    "end\n"
                                    'print("Tagged 8 grapple anchor ring parts")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 3,
        }
    )

    # 15. mass_set_property chain: search then bulk update
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Find all neon parts in the Abyss zone and change their material to Glass.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'll search for parts with Neon material in the Abyss zone, then use mass_set_property "
                        "to change them all to Glass atomically."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local results = {}\n"
                                    "for _, part in workspace.SceneRoot.AbyssZone:GetDescendants() do\n"
                                    '    if part:IsA("BasePart") and part.Material == Enum.Material.Neon then\n'
                                    "        table.insert(results, part:GetFullName())\n"
                                    "    end\n"
                                    "end\n"
                                    'print(game:GetService("HttpService"):JSONEncode(results))'
                                ),
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"output": "["Workspace.SceneRoot.AbyssZone.Crystal1","Workspace.SceneRoot.AbyssZone.Crystal2","Workspace.SceneRoot.AbyssZone.LightOrb"]"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": "Found 3 neon parts. Changing them all to Glass.",
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "mass_set_property",
                            {
                                "paths": [
                                    "Workspace.SceneRoot.AbyssZone.Crystal1",
                                    "Workspace.SceneRoot.AbyssZone.Crystal2",
                                    "Workspace.SceneRoot.AbyssZone.LightOrb",
                                ],
                                "property": "Material",
                                "value": "Glass",
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 3,
        }
    )

    # 16. set_property with calculated value (set_calculated_property pattern)
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ("Double the size of the main platform in the Hub zone.")},
                {
                    "role": "assistant",
                    "content": (
                        "I need to read the current size first, then set it to double. Let me get the properties."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(), "get_instance_properties", {"path": "Workspace.SceneRoot.HubZone.MainPlatform"}
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"properties": {"Name": "MainPlatform", "Size": [40, 2, 40], "Position": [0, 10, 0], "Anchored": true}}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": "Current size is (40, 2, 40). Doubling to (80, 4, 80).",
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "set_property",
                            {
                                "path": "Workspace.SceneRoot.HubZone.MainPlatform",
                                "property": "Size",
                                "value": [80, 4, 80],
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 2,
        }
    )

    # 17. run_script_in_play_mode usage: runtime testing
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Start a playtest and check if the grapple ability activates correctly when "
                        "pointing at a GrappleAnchor."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'll start a playtest and then run a diagnostic script in play mode to simulate "
                        "a grapple validation check."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "start"}),
                    ],
                },
                {"role": "tool", "content": '{"success": true, "status": "playing"}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_script_in_play_mode",
                            {
                                "code": (
                                    'local CS = game:GetService("CollectionService")\n'
                                    'local anchors = CS:GetTagged("GrappleAnchor")\n'
                                    'local player = game:GetService("Players").LocalPlayer\n'
                                    "local char = player.Character\n"
                                    'if not char then print("[TEST] No character found") return end\n'
                                    'local hrp = char:FindFirstChild("HumanoidRootPart")\n'
                                    "local nearest, nearDist = nil, math.huge\n"
                                    "for _, a in anchors do\n"
                                    "    local d = (a.Position - hrp.Position).Magnitude\n"
                                    "    if d < nearDist then nearest, nearDist = a, d end\n"
                                    "end\n"
                                    "if nearest then\n"
                                    '    print("[TEST] Nearest GrappleAnchor: " .. nearest.Name .. " at " .. string.format("%.1f", nearDist) .. " studs")\n'
                                    '    print("[TEST] In range: " .. tostring(nearDist <= 180))\n'
                                    "else\n"
                                    '    print("[TEST] No GrappleAnchors found")\n'
                                    "end"
                                ),
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"output": "[TEST] Nearest GrappleAnchor: Anchor_Hub_3 at 45.2 studs\\n[TEST] In range: true"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The grapple validation test passed. The nearest GrappleAnchor (Anchor_Hub_3) is "
                        "45.2 studs away, well within the 180-stud max range. Let me stop the playtest."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "stop"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 4,
        }
    )

    # 18. Search -> inspect children -> modify child
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What children does the OceanZone have and what are their classes?"},
                {
                    "role": "assistant",
                    "content": "I'll list the children of the OceanZone.",
                    "tool_calls": [
                        make_tool_call(next_id(), "get_instance_children", {"path": "Workspace.SceneRoot.OceanZone"}),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"children": [{"name": "WaterVolume", "className": "Part"}, {"name": "CoralCluster_1", "className": "Model"}, {"name": "CoralCluster_2", "className": "Model"}, {"name": "CurrentVolume_1", "className": "Part"}, {"name": "Checkpoint_Ocean", "className": "Part"}, {"name": "OceanFloor", "className": "Part"}]}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The OceanZone contains 6 children:\n"
                        "- **WaterVolume** (Part) -- the swim volume\n"
                        "- **CoralCluster_1**, **CoralCluster_2** (Model) -- decorative coral\n"
                        "- **CurrentVolume_1** (Part) -- water current push zone\n"
                        "- **Checkpoint_Ocean** (Part) -- respawn checkpoint\n"
                        "- **OceanFloor** (Part) -- terrain base"
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 2,
        }
    )

    # 19. File tree -> navigate -> inspect
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "I need to understand the shared config structure. Show me what's in the Config folder.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(next_id(), "get_instance_children", {"path": "ReplicatedStorage.Shared.Config"}),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"children": [{"name": "Abilities", "className": "ModuleScript"}, {"name": "Vehicles", "className": "ModuleScript"}, {"name": "Zones", "className": "ModuleScript"}, {"name": "BroadcastCam", "className": "ModuleScript"}, {"name": "AgentPatrols", "className": "ModuleScript"}]}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The Config folder under ReplicatedStorage.Shared contains 5 ModuleScripts:\n"
                        "- **Abilities** -- GrappleTuning, GlideTuning, SlideTuning, AirDashTuning\n"
                        "- **Vehicles** -- DirtBikeTuning, GliderKiteTuning\n"
                        "- **Zones** -- zone builder configs for all 56 zones\n"
                        "- **BroadcastCam** -- camera shot presets and zone palettes\n"
                        "- **AgentPatrols** -- NPC patrol route definitions"
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 2,
        }
    )

    # 20. Create object, set properties, then delete (full lifecycle)
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Create a test part, make it neon green, then remove it."},
                {
                    "role": "assistant",
                    "content": "I'll create a test part, modify its color, then clean it up.",
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "create_object",
                            {
                                "className": "Part",
                                "parent": "Workspace",
                                "properties": {"Name": "TestPart_Temp", "Anchored": True},
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"success": true, "path": "Workspace.TestPart_Temp"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "set_property",
                            {
                                "path": "Workspace.TestPart_Temp",
                                "property": "Color",
                                "value": [0, 1, 0],
                            },
                        ),
                        make_tool_call(
                            next_id(),
                            "set_property",
                            {
                                "path": "Workspace.TestPart_Temp",
                                "property": "Material",
                                "value": "Neon",
                            },
                        ),
                    ],
                },
                {"role": "tool", "content": '{"success": true}', "tool_call_id": f"call_{call_counter[0] - 1}"},
                {"role": "tool", "content": '{"success": true}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": "Part is now neon green. Removing it.",
                    "tool_calls": [
                        make_tool_call(next_id(), "delete_object", {"path": "Workspace.TestPart_Temp"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 2,
        }
    )

    # 21. run_code for complex query + follow-up modification
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Count how many parts are in each zone and tell me which zone has the most.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'll run a Luau script to iterate through all zone children under SceneRoot "
                        "and count descendants per zone."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local counts = {}\n"
                                    "for _, zone in workspace.SceneRoot:GetChildren() do\n"
                                    "    local n = 0\n"
                                    "    for _, d in zone:GetDescendants() do\n"
                                    '        if d:IsA("BasePart") then n += 1 end\n'
                                    "    end\n"
                                    "    counts[zone.Name] = n\n"
                                    "end\n"
                                    'local maxName, maxCount = "", 0\n'
                                    "for name, c in counts do\n"
                                    "    if c > maxCount then maxName, maxCount = name, c end\n"
                                    "end\n"
                                    'print("Zone part counts:")\n'
                                    'for name, c in counts do print("  " .. name .. ": " .. c) end\n'
                                    'print("Largest: " .. maxName .. " with " .. maxCount .. " parts")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 3,
        }
    )

    # 22. Playtest -> runtime script -> check output -> stop
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Test if the swim ability activates when the player enters the OceanZone water volume.",
                },
                {
                    "role": "assistant",
                    "content": ("I'll start a playtest and run a diagnostic to check swim state detection."),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "start"}),
                    ],
                },
                {"role": "tool", "content": '{"success": true, "status": "playing"}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_script_in_play_mode",
                            {
                                "code": (
                                    'local player = game:GetService("Players").LocalPlayer\n'
                                    "local char = player.Character or player.CharacterAdded:Wait()\n"
                                    'local hrp = char:WaitForChild("HumanoidRootPart")\n'
                                    "-- Teleport into water volume\n"
                                    'local waterVol = workspace.SceneRoot.OceanZone:FindFirstChild("WaterVolume")\n'
                                    "if waterVol then\n"
                                    "    hrp.CFrame = waterVol.CFrame\n"
                                    "    task.wait(0.5)\n"
                                    "    -- Check if swim state is active\n"
                                    '    local stateAttr = char:GetAttribute("TraversalState")\n'
                                    '    print("[TEST] TraversalState: " .. tostring(stateAttr))\n'
                                    '    print("[TEST] Swim active: " .. tostring(stateAttr == "Swimming"))\n'
                                    "else\n"
                                    '    print("[TEST] WaterVolume not found in OceanZone")\n'
                                    "end"
                                ),
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"output": "[TEST] TraversalState: Swimming\\n[TEST] Swim active: true"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Swim ability is working correctly. When the player is teleported into the OceanZone "
                        "WaterVolume, the TraversalState transitions to Swimming. Stopping the playtest."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "stop"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 4,
        }
    )

    # 23. Search by property -> inspect -> bulk modify
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Find all parts with Transparency = 1 (invisible) and make them semi-transparent (0.5) so I can see them.",
                },
                {
                    "role": "assistant",
                    "content": ("I'll search for invisible parts and then make them semi-transparent for debugging."),
                    "tool_calls": [
                        make_tool_call(next_id(), "search_by_property", {"property": "Transparency", "value": 1}),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"results": [{"path": "Workspace.SceneRoot.HubZone.TriggerZone_A"}, {"path": "Workspace.SceneRoot.HubZone.TriggerZone_B"}, {"path": "Workspace.SceneRoot.AscentZone.WindTrigger"}]}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": "Found 3 invisible parts. Making them semi-transparent.",
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "mass_set_property",
                            {
                                "paths": [
                                    "Workspace.SceneRoot.HubZone.TriggerZone_A",
                                    "Workspace.SceneRoot.HubZone.TriggerZone_B",
                                    "Workspace.SceneRoot.AscentZone.WindTrigger",
                                ],
                                "property": "Transparency",
                                "value": 0.5,
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 3,
        }
    )

    # 24. Insert model -> configure -> tag
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Insert the lantern model (asset 98765432) into the AetherSea zone, scale it up, and tag it as a light source.",
                },
                {
                    "role": "assistant",
                    "content": ("I'll insert the model, then adjust its scale and add a CollectionService tag."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "insert_model",
                            {"assetId": 98765432, "parent": "Workspace.SceneRoot.AetherSeaZone"},
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"success": true, "path": "Workspace.SceneRoot.AetherSeaZone.Lantern"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local lantern = workspace.SceneRoot.AetherSeaZone.Lantern\n"
                                    "lantern:ScaleTo(2.5)\n"
                                    'game:GetService("CollectionService"):AddTag(lantern, "LightSource")\n'
                                    'print("Lantern scaled to 2.5x and tagged as LightSource")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 3,
        }
    )

    # 25. Multi-tool: get_file_tree + search_objects (parallel investigation)
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "I'm trying to find where the ability configs are stored. Check both the file tree and search for 'AbilityConfig'.",
                },
                {
                    "role": "assistant",
                    "content": ("I'll search for AbilityConfig by name and also look at the Config folder structure."),
                    "tool_calls": [
                        make_tool_call(next_id(), "search_objects", {"query": "Abilities"}),
                        make_tool_call(next_id(), "get_instance_children", {"path": "ReplicatedStorage.Shared.Config"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 2,
        }
    )

    # ===================================================================
    # VERTIGO-SPECIFIC SCENARIOS (15 scenarios)
    # ===================================================================

    # 26. Grapple anchor management: search, resize, retag
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ("Find all GrappleAnchor tagged parts and make them 20% larger.")},
                {
                    "role": "assistant",
                    "content": (
                        "I'll use run_code to find all tagged parts and resize them, since "
                        "CollectionService tags can't be searched via search_objects directly."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local CS = game:GetService("CollectionService")\n'
                                    'local anchors = CS:GetTagged("GrappleAnchor")\n'
                                    "for _, anchor in anchors do\n"
                                    "    anchor.Size = anchor.Size * 1.2\n"
                                    "end\n"
                                    'print("Resized " .. #anchors .. " GrappleAnchor parts")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 3,
        }
    )

    # 27. Grapple anchor retag: migrate tags
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Rename the CollectionService tag 'GrapplePoint' to 'GrappleAnchor' on all instances that have it.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'll use run_code to find all instances with the old tag, remove it, and add the new one."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local CS = game:GetService("CollectionService")\n'
                                    'local old = CS:GetTagged("GrapplePoint")\n'
                                    "for _, inst in old do\n"
                                    '    CS:RemoveTag(inst, "GrapplePoint")\n'
                                    '    CS:AddTag(inst, "GrappleAnchor")\n'
                                    "end\n"
                                    'print("Migrated " .. #old .. " instances from GrapplePoint to GrappleAnchor")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 2,
        }
    )

    # 28. Zone builder inspection: get children + check properties
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "What does the Abyss zone contain? List all its parts and their approximate positions.",
                },
                {
                    "role": "assistant",
                    "content": ("I'll get the Abyss zone's children, then use run_code to get a position summary."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local abyss = workspace.SceneRoot:FindFirstChild("AbyssZone")\n'
                                    'if not abyss then print("AbyssZone not found") return end\n'
                                    'print("AbyssZone contents:")\n'
                                    "for _, child in abyss:GetChildren() do\n"
                                    '    local pos = "N/A"\n'
                                    '    if child:IsA("BasePart") then\n'
                                    '        pos = string.format("(%.0f, %.0f, %.0f)", child.Position.X, child.Position.Y, child.Position.Z)\n'
                                    '    elseif child:IsA("Model") and child.PrimaryPart then\n'
                                    "        local p = child.PrimaryPart.Position\n"
                                    '        pos = string.format("(%.0f, %.0f, %.0f)", p.X, p.Y, p.Z)\n'
                                    "    end\n"
                                    '    print("  " .. child.Name .. " [" .. child.ClassName .. "] @ " .. pos)\n'
                                    "end"
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 2,
        }
    )

    # 29. Vehicle template creation/modification
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Create a new vehicle template called 'Hoverboard' with max speed 120, "
                        "lift 0.6, and acceleration 55. Put it in ServerStorage.VehicleTemplates."
                    ),
                },
                {
                    "role": "assistant",
                    "content": ("I'll create the Hoverboard model with configuration values stored as attributes."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "create_object",
                            {
                                "className": "Model",
                                "parent": "ServerStorage.VehicleTemplates",
                                "properties": {"Name": "Hoverboard"},
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"success": true, "path": "ServerStorage.VehicleTemplates.Hoverboard"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": "Now I'll set the tuning attributes on the template.",
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local hb = game.ServerStorage.VehicleTemplates.Hoverboard\n"
                                    'hb:SetAttribute("MaxSpeed", 120)\n'
                                    'hb:SetAttribute("Lift", 0.6)\n'
                                    'hb:SetAttribute("Acceleration", 55)\n'
                                    'hb:SetAttribute("VehicleType", "hover")\n'
                                    'hb:SetAttribute("StableID", "vehicle_hoverboard_v1")\n'
                                    'print("Hoverboard template configured")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 3,
        }
    )

    # 30. Checkpoint placement and tagging
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Place checkpoints at the entrance and exit of the CrystalCavern zone."},
                {
                    "role": "assistant",
                    "content": (
                        "I'll first inspect the CrystalCavern to understand its bounds, then place "
                        "checkpoints at logical entry/exit points."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local zone = workspace.SceneRoot:FindFirstChild("CrystalCavern")\n'
                                    'if not zone then print("CrystalCavern not found") return end\n'
                                    "local minY, maxY = math.huge, -math.huge\n"
                                    "local minX, maxX, minZ, maxZ = math.huge, -math.huge, math.huge, -math.huge\n"
                                    "for _, d in zone:GetDescendants() do\n"
                                    '    if d:IsA("BasePart") then\n'
                                    "        local p = d.Position\n"
                                    "        if p.Y < minY then minY = p.Y end\n"
                                    "        if p.Y > maxY then maxY = p.Y end\n"
                                    "        if p.X < minX then minX = p.X end\n"
                                    "        if p.X > maxX then maxX = p.X end\n"
                                    "        if p.Z < minZ then minZ = p.Z end\n"
                                    "        if p.Z > maxZ then maxZ = p.Z end\n"
                                    "    end\n"
                                    "end\n"
                                    'print(string.format("Bounds: X[%.0f,%.0f] Y[%.0f,%.0f] Z[%.0f,%.0f]", minX, maxX, minY, maxY, minZ, maxZ))'
                                ),
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"output": "Bounds: X[-80,80] Y[-60,-30] Z[-80,80]"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The CrystalCavern spans X[-80,80], Y[-60,-30], Z[-80,80]. I'll place an entrance "
                        "checkpoint near the top (Y=-30) and an exit checkpoint near the bottom (Y=-58)."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "mass_create_objects",
                            {
                                "objects": [
                                    {
                                        "className": "Part",
                                        "parent": "Workspace.SceneRoot.CrystalCavern",
                                        "properties": {
                                            "Name": "Checkpoint_CrystalCavern_Entrance",
                                            "Position": [0, -30, 0],
                                            "Size": [6, 1, 6],
                                            "Anchored": True,
                                            "Transparency": 0.5,
                                            "Material": "Neon",
                                        },
                                    },
                                    {
                                        "className": "Part",
                                        "parent": "Workspace.SceneRoot.CrystalCavern",
                                        "properties": {
                                            "Name": "Checkpoint_CrystalCavern_Exit",
                                            "Position": [0, -58, 0],
                                            "Size": [6, 1, 6],
                                            "Anchored": True,
                                            "Transparency": 0.5,
                                            "Material": "Neon",
                                        },
                                    },
                                ],
                            },
                        ),
                    ],
                },
                {"role": "tool", "content": '{"success": true, "created": 2}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local CS = game:GetService("CollectionService")\n'
                                    'CS:AddTag(workspace.SceneRoot.CrystalCavern.Checkpoint_CrystalCavern_Entrance, "Checkpoint")\n'
                                    'CS:AddTag(workspace.SceneRoot.CrystalCavern.Checkpoint_CrystalCavern_Exit, "Checkpoint")\n'
                                    'print("Both checkpoints tagged")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 3,
        }
    )

    # 31. Crystal formation generation
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Create 20 crystal formations in the CrystalCavern zone at random positions "
                        "between Y=-30 and Y=-60."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'll use run_code with a loop to generate crystals at random positions "
                        "within the cavern depth range, using deterministic seeding for reproducibility."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local CS = game:GetService("CollectionService")\n'
                                    "local rng = Random.new(12345)\n"
                                    "local parent = workspace.SceneRoot.CrystalCavern\n"
                                    "for i = 1, 20 do\n"
                                    '    local crystal = Instance.new("Part")\n'
                                    '    crystal.Name = "Crystal_" .. i\n'
                                    "    crystal.Size = Vector3.new(\n"
                                    "        rng:NextNumber(2, 6),\n"
                                    "        rng:NextNumber(4, 12),\n"
                                    "        rng:NextNumber(2, 6)\n"
                                    "    )\n"
                                    "    crystal.Position = Vector3.new(\n"
                                    "        rng:NextNumber(-100, 100),\n"
                                    "        rng:NextNumber(-60, -30),\n"
                                    "        rng:NextNumber(-100, 100)\n"
                                    "    )\n"
                                    "    crystal.Material = Enum.Material.Neon\n"
                                    "    crystal.Color = Color3.fromHSV(0.75, 0.3, 0.9)\n"
                                    "    crystal.Anchored = true\n"
                                    "    crystal.Parent = parent\n"
                                    '    CS:AddTag(crystal, "CrystalFormation")\n'
                                    "end\n"
                                    'print("Created 20 crystal formations in CrystalCavern")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 3,
        }
    )

    # 32. Lighting/atmosphere changes per zone
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Set the Lighting atmosphere to a deep ocean feel -- dark blue ambient, heavy fog, low brightness.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'll modify Lighting properties and the Atmosphere instance for a deep ocean atmosphere."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local lighting = game:GetService("Lighting")\n'
                                    "lighting.Ambient = Color3.fromRGB(10, 20, 50)\n"
                                    "lighting.OutdoorAmbient = Color3.fromRGB(5, 10, 30)\n"
                                    "lighting.Brightness = 0.3\n"
                                    "lighting.ClockTime = 0\n"
                                    "lighting.FogColor = Color3.fromRGB(8, 15, 40)\n"
                                    "lighting.FogStart = 20\n"
                                    "lighting.FogEnd = 200\n"
                                    "-- Update Atmosphere if it exists\n"
                                    'local atmo = lighting:FindFirstChildOfClass("Atmosphere")\n'
                                    "if atmo then\n"
                                    "    atmo.Density = 0.6\n"
                                    "    atmo.Offset = 0.2\n"
                                    "    atmo.Color = Color3.fromRGB(10, 25, 60)\n"
                                    "    atmo.Decay = Color3.fromRGB(5, 10, 30)\n"
                                    "    atmo.Glare = 0\n"
                                    "    atmo.Haze = 8\n"
                                    "end\n"
                                    'print("Deep ocean atmosphere applied")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 2,
        }
    )

    # 33. DataStore inspection via run_code
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Check the DataStore schema version for the currently loaded player profile.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'll use run_code to query the DataService for the current schema version. "
                        "Note: DataStore access requires a running server context, so I need to start a playtest first."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "start"}),
                    ],
                },
                {"role": "tool", "content": '{"success": true, "status": "playing"}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_script_in_play_mode",
                            {
                                "code": (
                                    "local DataService = require(game.ServerScriptService.Server.Services.DataService)\n"
                                    'local players = game:GetService("Players"):GetPlayers()\n'
                                    "if #players > 0 then\n"
                                    "    local profile = DataService:GetProfile(players[1])\n"
                                    "    if profile then\n"
                                    '        print("Schema version: " .. tostring(profile.schemaVersion))\n'
                                    '        print("Auto-save: " .. tostring(profile.autoSave))\n'
                                    "    else\n"
                                    '        print("No profile loaded for " .. players[1].Name)\n'
                                    "    end\n"
                                    "else\n"
                                    '    print("No players in game")\n'
                                    "end"
                                ),
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"output": "Schema version: 7\\nAuto-save: true"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The DataStore is running schema v7 with auto-save enabled, which matches the "
                        "expected configuration. Stopping the playtest."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "stop"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 4,
        }
    )

    # 34. Ability config reading
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What are the current grapple tuning values in the live config?"},
                {
                    "role": "assistant",
                    "content": (
                        "I'll run code to require the Abilities config module and print the GrappleTuning table."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local Shared = game:GetService("ReplicatedStorage"):WaitForChild("Shared")\n'
                                    "local config = require(Shared.Config.Abilities)\n"
                                    "local tuning = config.GrappleTuning\n"
                                    'local HttpService = game:GetService("HttpService")\n'
                                    "print(HttpService:JSONEncode(tuning))"
                                ),
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"output": "{"maxRange":180,"hookSpeed":260,"reelSpeed":75,"gravityScale":1.15,"baumgarteBeta":0.22,"centripetalStrength":0.08,"slingshotImpulse":132}"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Current GrappleTuning values from the live config:\n"
                        "- **maxRange**: 180 studs\n"
                        "- **hookSpeed**: 260 studs/sec\n"
                        "- **reelSpeed**: 75 studs/sec\n"
                        "- **gravityScale**: 1.15x\n"
                        "- **baumgarteBeta**: 0.22\n"
                        "- **centripetalStrength**: 0.08\n"
                        "- **slingshotImpulse**: 132 studs/sec"
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 2,
        }
    )

    # 35. Wind volume creation for glide zones
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Create an updraft wind volume in the Ascent zone to help gliders gain altitude. Make it 50 studs wide and 100 studs tall.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'll create a wind volume part and tag it with WindVolume for the traversal system to detect."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "create_object",
                            {
                                "className": "Part",
                                "parent": "Workspace.SceneRoot.AscentZone",
                                "properties": {
                                    "Name": "Updraft_Ascent_1",
                                    "Position": [0, 80, 0],
                                    "Size": [50, 100, 50],
                                    "Anchored": True,
                                    "Transparency": 1,
                                    "CanCollide": False,
                                },
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"success": true, "path": "Workspace.SceneRoot.AscentZone.Updraft_Ascent_1"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local vol = workspace.SceneRoot.AscentZone.Updraft_Ascent_1\n"
                                    'vol:SetAttribute("WindDirection", Vector3.new(0, 1, 0))\n'
                                    'vol:SetAttribute("WindStrength", 45)\n'
                                    'game:GetService("CollectionService"):AddTag(vol, "WindVolume")\n'
                                    'print("Updraft wind volume created and tagged")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 3,
        }
    )

    # 36. Remotes inspection
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "List all the RemoteEvents and RemoteFunctions currently in the game."},
                {
                    "role": "assistant",
                    "content": ("I'll use run_code to enumerate all remotes in ReplicatedStorage."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local rs = game:GetService("ReplicatedStorage")\n'
                                    'print("RemoteEvents:")\n'
                                    "for _, r in rs:GetDescendants() do\n"
                                    '    if r:IsA("RemoteEvent") then\n'
                                    '        print("  " .. r:GetFullName())\n'
                                    "    end\n"
                                    "end\n"
                                    'print("RemoteFunctions:")\n'
                                    "for _, r in rs:GetDescendants() do\n"
                                    '    if r:IsA("RemoteFunction") then\n'
                                    '        print("  " .. r:GetFullName())\n'
                                    "    end\n"
                                    "end"
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 2,
        }
    )

    # 37. Zone health audit: count parts per tag
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Give me a health report: how many GrappleAnchors, Checkpoints, WindVolumes, and CurrentVolumes are in the game?",
                },
                {
                    "role": "assistant",
                    "content": ("I'll use run_code to count each CollectionService tag category."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local CS = game:GetService("CollectionService")\n'
                                    'local tags = {"GrappleAnchor", "Checkpoint", "WindVolume", "CurrentVolume", "BloomCrystal", "PressurePlate"}\n'
                                    'print("CollectionService Tag Census:")\n'
                                    "for _, tag in tags do\n"
                                    "    local count = #CS:GetTagged(tag)\n"
                                    '    print("  " .. tag .. ": " .. count)\n'
                                    "end"
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 2,
        }
    )

    # 38. Current volume creation for swim zones
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Add a water current that pushes swimmers eastward in the OceanZone."},
                {
                    "role": "assistant",
                    "content": (
                        "I'll create a CurrentVolume part inside the OceanZone with an eastward push direction."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "create_object",
                            {
                                "className": "Part",
                                "parent": "Workspace.SceneRoot.OceanZone",
                                "properties": {
                                    "Name": "CurrentVolume_East",
                                    "Position": [20, -15, 0],
                                    "Size": [80, 30, 40],
                                    "Anchored": True,
                                    "Transparency": 1,
                                    "CanCollide": False,
                                },
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"success": true, "path": "Workspace.SceneRoot.OceanZone.CurrentVolume_East"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local vol = workspace.SceneRoot.OceanZone.CurrentVolume_East\n"
                                    'vol:SetAttribute("CurrentDirection", Vector3.new(1, 0, 0))\n'
                                    'vol:SetAttribute("CurrentStrength", 30)\n'
                                    'game:GetService("CollectionService"):AddTag(vol, "CurrentVolume")\n'
                                    'print("Eastward current volume created")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 3,
        }
    )

    # 39. Wallrun surface setup
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Tag the tall wall in the Hub zone as a wallrun surface."},
                {
                    "role": "assistant",
                    "content": ("I'll search for the wall, then tag it with the WallrunSurface CollectionService tag."),
                    "tool_calls": [
                        make_tool_call(next_id(), "search_objects", {"query": "TallWall"}),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"results": [{"path": "Workspace.SceneRoot.HubZone.TallWall", "className": "Part"}]}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local CS = game:GetService("CollectionService")\n'
                                    "local wall = workspace.SceneRoot.HubZone.TallWall\n"
                                    'CS:AddTag(wall, "WallrunSurface")\n'
                                    "wall.Material = Enum.Material.Concrete\n"
                                    'print("TallWall tagged as WallrunSurface, material set to Concrete")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 2,
        }
    )

    # 40. Service boot order inspection
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Show me the server boot initialization order by checking the init script.",
                },
                {
                    "role": "assistant",
                    "content": ("I'll inspect the server init script source to show the boot order."),
                    "tool_calls": [
                        make_tool_call(next_id(), "get_instance_children", {"path": "ServerScriptService"}),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"children": [{"name": "Server", "className": "Script"}]}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local source = game.ServerScriptService.Server.Source\n"
                                    "-- Print lines containing require or :Init or :Start\n"
                                    'for line in source:gmatch("[^\\n]+") do\n'
                                    '    if line:find("require") or line:find(":Init") or line:find(":Start") then\n'
                                    "        print(line)\n"
                                    "    end\n"
                                    "end"
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 3,
        }
    )

    # ===================================================================
    # WHEN NOT TO CALL TOOLS (5 scenarios)
    # ===================================================================

    # 41. Luau syntax question
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What's the difference between RemoteEvent and RemoteFunction?"},
                {
                    "role": "assistant",
                    "content": (
                        "RemoteEvent is fire-and-forget (one-way communication), while RemoteFunction "
                        "is request-response (blocks until the other side returns a value).\n\n"
                        "In Vertigo:\n"
                        "- **RemoteEvent** is used for 51 event types like `RequestUseAbility`, "
                        "`StateSync`, `EventNotify` -- situations where the client fires and moves on\n"
                        "- **RemoteFunction** is used for 3 request types: `GetPlayerProfile`, "
                        "`GetNPCDialog`, `GetTraversalHint` -- where the client needs data back\n\n"
                        "General rule: prefer RemoteEvent unless you need a return value. "
                        "RemoteFunction blocks the calling thread and is vulnerable to the other side "
                        "never returning (causing a hang). Vertigo uses WaitForChild with 10s timeouts "
                        "to prevent indefinite hangs."
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_no_tool",
            "difficulty": 2,
        }
    )

    # 42. Vertigo architecture question
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "How does Vertigo's grapple swing physics work?"},
                {
                    "role": "assistant",
                    "content": (
                        "Vertigo's grapple uses a constraint-based pendulum with these key parameters "
                        "(from GrappleTuning config):\n\n"
                        "- **maxRange**: 180 studs -- furthest hook distance\n"
                        "- **hookSpeed**: 260 studs/sec -- how fast the hook travels\n"
                        "- **reelSpeed**: 75 studs/sec -- pull-in speed\n"
                        "- **gravityScale**: 1.15x -- slightly heavier than normal for satisfying arcs\n"
                        "- **baumgarteBeta**: 0.22 -- oscillation damping (Baumgarte stabilization)\n"
                        "- **centripetalStrength**: 0.08 -- assist multiplier for pendulum control\n"
                        "- **slingshotImpulse**: 132 studs/sec -- release velocity boost\n\n"
                        "The Spring module in `Shared/Util/Physics/Spring.luau` handles the constraint "
                        "solving using a closed-form analytical solution (no sub-stepping needed), "
                        "discriminating between under-damped, critically-damped, and over-damped regimes "
                        "based on the damping ratio zeta = c / (2 * sqrt(k*m)).\n\n"
                        "Ability chaining is supported via grace windows: grapple-to-wallrun has a "
                        "0.65s grace, grapple-to-glide has 0.6s."
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_no_tool",
            "difficulty": 2,
        }
    )

    # 43. Roblox API behavior question
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "How does CollectionService:GetTagged() work? Does it return a static snapshot or a live list?",
                },
                {
                    "role": "assistant",
                    "content": (
                        "CollectionService:GetTagged(tag) returns a **static snapshot** (a regular Luau "
                        "table), not a live-updating list. If instances are tagged or untagged after the "
                        "call, the returned table does not update.\n\n"
                        "For live tracking, connect to:\n"
                        "- `CollectionService:GetInstanceAddedSignal(tag)` -- fires when an instance gets the tag\n"
                        "- `CollectionService:GetInstanceRemovedSignal(tag)` -- fires when an instance loses the tag\n\n"
                        "In Vertigo, the world builders call GetTagged during Init to get the initial set, "
                        "then connect the Added/Removed signals to maintain a live cache. This avoids "
                        "re-querying every frame, which would be expensive with 200+ GrappleAnchor parts."
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_no_tool",
            "difficulty": 2,
        }
    )

    # 44. Luau type annotation question
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "How do I write generic type annotations in Luau?"},
                {
                    "role": "assistant",
                    "content": (
                        "Luau generics use angle-bracket syntax:\n\n"
                        "```luau\n"
                        "-- Generic function\n"
                        "local function map<T, U>(arr: {T}, fn: (T) -> U): {U}\n"
                        "    local result = table.create(#arr)\n"
                        "    for i, v in arr do\n"
                        "        result[i] = fn(v)\n"
                        "    end\n"
                        "    return result\n"
                        "end\n\n"
                        "-- Generic type alias\n"
                        "type Result<T> = { ok: true, value: T } | { ok: false, error: string }\n\n"
                        "-- Bounded generics (constrained)\n"
                        "type Container<T> = {\n"
                        "    items: {T},\n"
                        "    add: (self: Container<T>, item: T) -> (),\n"
                        "}\n"
                        "```\n\n"
                        "Key rules:\n"
                        "- Generics appear after the function name or type name\n"
                        "- `{T}` means array of T, `{[string]: T}` means dictionary with string keys\n"
                        "- Union types use `|`, intersection types use `&`\n"
                        "- In Vertigo, all shared types live in `src/Shared/Util/Types.luau` using `export type`"
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_no_tool",
            "difficulty": 2,
        }
    )

    # 45. Vertigo architecture: service pattern
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Explain the Init/Start lifecycle pattern used in Vertigo services."},
                {
                    "role": "assistant",
                    "content": (
                        "Vertigo uses a two-phase boot pattern for all services (server) and controllers (client):\n\n"
                        "**Phase 1: Init**\n"
                        "- Called sequentially in explicit order\n"
                        "- Setup internal state, create data structures, register config\n"
                        "- No side effects -- no event connections, no network calls, no spawning\n"
                        "- Other services may not be initialized yet, so no cross-service calls\n\n"
                        "**Phase 2: Start**\n"
                        "- Called after ALL services have completed Init\n"
                        "- Connect event listeners, bind remotes, start loops\n"
                        "- Safe to call other services since everything is initialized\n\n"
                        "```luau\n"
                        "--!strict\n"
                        "local MyService = {}\n\n"
                        "function MyService:Init()\n"
                        "    self._cache = {}\n"
                        "    self._config = require(Shared.Config.MyConfig)\n"
                        "end\n\n"
                        "function MyService:Start()\n"
                        "    RunService.Heartbeat:Connect(function(dt)\n"
                        "        self:_update(dt)\n"
                        "    end)\n"
                        "end\n\n"
                        "return MyService\n"
                        "```\n\n"
                        "Boot order matters. Both `init.server.luau` and `init.client.luau` require and call "
                        "services in an explicit, hardcoded order -- there is no auto-discovery or dependency injection."
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_no_tool",
            "difficulty": 2,
        }
    )

    # ===================================================================
    # ERROR RECOVERY SCENARIOS (5 scenarios)
    # ===================================================================

    # 46. Tool returns error, model adjusts approach
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Delete the broken checkpoint at Workspace.SceneRoot.HubZone.OldCheckpoint.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(), "delete_object", {"path": "Workspace.SceneRoot.HubZone.OldCheckpoint"}
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"error": "Instance not found at path: Workspace.SceneRoot.HubZone.OldCheckpoint"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The path doesn't exist. Let me search for it -- it might be named differently or "
                        "in a different location."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "search_objects", {"query": "OldCheckpoint"}),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"results": [{"path": "Workspace.SceneRoot.HubZone.Checkpoints.OldCheckpoint", "className": "Part"}]}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": "Found it nested under a Checkpoints folder. Deleting now.",
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "delete_object",
                            {"path": "Workspace.SceneRoot.HubZone.Checkpoints.OldCheckpoint"},
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_error_recovery",
            "difficulty": 3,
        }
    )

    # 47. Property set error, model inspects first
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Set the WalkSpeed on the grapple anchor to 50."},
                {
                    "role": "assistant",
                    "content": (
                        "I'll try to set WalkSpeed, but GrappleAnchors are typically Parts which don't have "
                        "a WalkSpeed property. Let me check what kind of instance it is first."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "search_objects", {"query": "GrappleAnchor"}),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"results": [{"path": "Workspace.SceneRoot.HubZone.GrappleAnchor_1", "className": "Part"}]}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "As expected, GrappleAnchor_1 is a Part, which doesn't have a WalkSpeed property. "
                        "WalkSpeed is a property of the Humanoid class. Did you mean to modify a Humanoid, "
                        "or did you want to set a custom attribute on the anchor? For example, I could set "
                        "a 'PullSpeed' attribute via `SetAttribute`."
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_error_recovery",
            "difficulty": 3,
        }
    )

    # 48. Playtest reveals errors, model debugs
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "The glide ability isn't working. Can you diagnose it?"},
                {
                    "role": "assistant",
                    "content": ("I'll start a playtest and check the console for glide-related errors."),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "start"}),
                    ],
                },
                {"role": "tool", "content": '{"success": true, "status": "playing"}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_script_in_play_mode",
                            {
                                "code": (
                                    'local player = game:GetService("Players").LocalPlayer\n'
                                    "local char = player.Character or player.CharacterAdded:Wait()\n"
                                    'print("[DIAG] Character: " .. char.Name)\n'
                                    'print("[DIAG] HRP exists: " .. tostring(char:FindFirstChild("HumanoidRootPart") ~= nil))\n'
                                    "-- Check ability module state\n"
                                    'local Shared = game:GetService("ReplicatedStorage"):WaitForChild("Shared")\n'
                                    "local config = require(Shared.Config.Abilities)\n"
                                    'print("[DIAG] GlideTuning.lift: " .. tostring(config.GlideTuning.lift))\n'
                                    'print("[DIAG] GlideTuning.enabled: " .. tostring(config.GlideTuning.enabled))'
                                ),
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"output": "[DIAG] Character: Player1\\n[DIAG] HRP exists: true\\n[DIAG] GlideTuning.lift: 0.55\\n[DIAG] GlideTuning.enabled: false"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Found the issue: `GlideTuning.enabled` is set to `false`. The glide ability is "
                        "disabled in the config. This needs to be changed in `src/Shared/Config/Abilities.luau` "
                        "by setting `enabled = true` in the GlideTuning table. Let me stop the playtest."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "stop"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_error_recovery",
            "difficulty": 4,
        }
    )

    # 49. run_code fails, model tries alternative approach
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Get the total memory usage of the game."},
                {
                    "role": "assistant",
                    "content": ("I'll use run_code to check stats via the Stats service."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local stats = game:GetService("Stats")\n'
                                    'print("Memory: " .. stats:GetTotalMemoryUsageMb() .. " MB")'
                                ),
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"error": "GetTotalMemoryUsageMb is not a valid member of Stats"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "That API doesn't exist in the command bar context. Let me try using "
                        "the MemoryStoreService stats or a playtest to get runtime memory."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "start"}),
                    ],
                },
                {"role": "tool", "content": '{"success": true, "status": "playing"}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_script_in_play_mode",
                            {
                                "code": (
                                    'local stats = game:GetService("Stats")\n'
                                    "local mem = stats:GetMemoryUsageMbForTag(Enum.DeveloperMemoryTag.Internal)\n"
                                    'print("Internal memory: " .. string.format("%.1f", mem) .. " MB")\n'
                                    "local instances = stats:GetMemoryUsageMbForTag(Enum.DeveloperMemoryTag.Instances)\n"
                                    'print("Instance memory: " .. string.format("%.1f", instances) .. " MB")'
                                ),
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"output": "Internal memory: 245.3 MB\\nInstance memory: 89.7 MB"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Memory usage during playtest:\n"
                        "- Internal: 245.3 MB\n"
                        "- Instances: 89.7 MB\n\n"
                        "The Stats memory API is only available in a running game context, not from the "
                        "edit-mode command bar. Let me stop the playtest."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "stop"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_error_recovery",
            "difficulty": 4,
        }
    )

    # 50. Search returns empty, model broadens search
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Find the pressure plate puzzle script."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(next_id(), "search_objects", {"query": "PressurePlatePuzzle"}),
                    ],
                },
                {"role": "tool", "content": '{"results": []}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": "No exact match. Let me try a broader search.",
                    "tool_calls": [
                        make_tool_call(next_id(), "search_objects", {"query": "PressurePlate"}),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"results": [{"path": "Workspace.SceneRoot.HubZone.PuzzleArea.PressurePlate_1", "className": "Part"}, {"path": "Workspace.SceneRoot.HubZone.PuzzleArea.PressurePlate_2", "className": "Part"}]}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Found PressurePlate parts but no dedicated script. Let me check if there's a "
                        "puzzle script in the PuzzleArea folder."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(), "get_instance_children", {"path": "Workspace.SceneRoot.HubZone.PuzzleArea"}
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"children": [{"name": "PressurePlate_1", "className": "Part"}, {"name": "PressurePlate_2", "className": "Part"}, {"name": "PuzzleController", "className": "Script"}, {"name": "RewardChest", "className": "Model"}]}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The pressure plate logic is in `Workspace.SceneRoot.HubZone.PuzzleArea.PuzzleController` "
                        "(a Script). The PuzzleArea contains two PressurePlate parts, the controller script, "
                        "and a RewardChest model. The plates are likely tagged with the `PressurePlate` "
                        "CollectionService tag and detected by the PuzzleController."
                    ),
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_error_recovery",
            "difficulty": 3,
        }
    )

    # ===================================================================
    # ADDITIONAL SCENARIOS to reach 55+ total
    # ===================================================================

    # 51. Slide surface setup
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Create a slide ramp in the Hub zone -- a long angled part that players can slide down.",
                },
                {
                    "role": "assistant",
                    "content": ("I'll create an angled part and tag it for slide detection."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "create_object",
                            {
                                "className": "Part",
                                "parent": "Workspace.SceneRoot.HubZone",
                                "properties": {
                                    "Name": "SlideRamp_Hub_1",
                                    "Size": [10, 1, 60],
                                    "Anchored": True,
                                    "Material": "SmoothPlastic",
                                },
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"success": true, "path": "Workspace.SceneRoot.HubZone.SlideRamp_Hub_1"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": ("Now I'll angle it at roughly 30 degrees and tag it as a SlideSurface."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local ramp = workspace.SceneRoot.HubZone.SlideRamp_Hub_1\n"
                                    "ramp.CFrame = CFrame.new(30, 20, 0) * CFrame.Angles(math.rad(-30), 0, 0)\n"
                                    'game:GetService("CollectionService"):AddTag(ramp, "SlideSurface")\n'
                                    'print("Slide ramp positioned and tagged")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 3,
        }
    )

    # 52. Air dash ring creation
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Create a series of air dash boost rings leading up through the Ascent zone.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'll create 5 ring-shaped boost volumes at ascending Y positions using run_code "
                        "since creating torus shapes requires procedural geometry."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    'local CS = game:GetService("CollectionService")\n'
                                    "local parent = workspace.SceneRoot.AscentZone\n"
                                    "local positions = {\n"
                                    "    Vector3.new(10, 40, 0),\n"
                                    "    Vector3.new(-5, 60, 15),\n"
                                    "    Vector3.new(20, 80, -10),\n"
                                    "    Vector3.new(0, 100, 5),\n"
                                    "    Vector3.new(-15, 120, -5),\n"
                                    "}\n"
                                    "for i, pos in positions do\n"
                                    '    local ring = Instance.new("Part")\n'
                                    '    ring.Name = "AirDashRing_" .. i\n'
                                    "    ring.Shape = Enum.PartType.Cylinder\n"
                                    "    ring.Size = Vector3.new(2, 12, 12)\n"
                                    "    ring.Position = pos\n"
                                    "    ring.Anchored = true\n"
                                    "    ring.CanCollide = false\n"
                                    "    ring.Transparency = 0.6\n"
                                    "    ring.Material = Enum.Material.Neon\n"
                                    "    ring.Color = Color3.fromHSV(0.55, 0.8, 1)\n"
                                    '    ring:SetAttribute("BoostStrength", 80)\n'
                                    "    ring.Parent = parent\n"
                                    '    CS:AddTag(ring, "AirDashBoost")\n'
                                    "end\n"
                                    'print("Created 5 air dash boost rings in AscentZone")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 3,
        }
    )

    # 53. Physics audit: find floating unanchored parts
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Are there any unanchored parts floating in mid-air that might fall when the game starts?",
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'll run a physics audit to find unanchored parts above ground level that aren't "
                        "in a physics assembly."
                    ),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local suspicious = {}\n"
                                    "for _, part in workspace:GetDescendants() do\n"
                                    '    if part:IsA("BasePart") and not part.Anchored and part.Position.Y > 5 then\n'
                                    "        -- Check if it has constraints or welds\n"
                                    "        local hasConstraint = false\n"
                                    "        for _, child in part:GetChildren() do\n"
                                    '            if child:IsA("Constraint") or child:IsA("JointInstance") then\n'
                                    "                hasConstraint = true\n"
                                    "                break\n"
                                    "            end\n"
                                    "        end\n"
                                    "        if not hasConstraint then\n"
                                    "            table.insert(suspicious, {\n"
                                    "                path = part:GetFullName(),\n"
                                    '                y = string.format("%.0f", part.Position.Y),\n'
                                    "            })\n"
                                    "        end\n"
                                    "    end\n"
                                    "end\n"
                                    'print("Suspicious floating parts: " .. #suspicious)\n'
                                    "for _, s in suspicious do\n"
                                    '    print("  " .. s.path .. " (Y=" .. s.y .. ")")\n'
                                    "end"
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 3,
        }
    )

    # 54. Mass anchor fix
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Anchor all parts in the HubZone that are currently unanchored."},
                {
                    "role": "assistant",
                    "content": ("I'll use run_code to find and anchor all unanchored parts in the HubZone."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local count = 0\n"
                                    "for _, part in workspace.SceneRoot.HubZone:GetDescendants() do\n"
                                    '    if part:IsA("BasePart") and not part.Anchored then\n'
                                    "        part.Anchored = true\n"
                                    "        count += 1\n"
                                    "    end\n"
                                    "end\n"
                                    'print("Anchored " .. count .. " parts in HubZone")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 2,
        }
    )

    # 55. Traversal smoke test via playtest
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Run the traversal smoke test suite and report results."},
                {
                    "role": "assistant",
                    "content": ("I'll start a playtest and trigger the DevAutomationController smoke suite."),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "start"}),
                    ],
                },
                {"role": "tool", "content": '{"success": true, "status": "playing"}', "tool_call_id": cur_id()},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_script_in_play_mode",
                            {
                                "code": (
                                    'local ReplicatedStorage = game:GetService("ReplicatedStorage")\n'
                                    'local Shared = ReplicatedStorage:WaitForChild("Shared")\n'
                                    "-- Trigger the automation controller\n"
                                    'local autoEvent = ReplicatedStorage:FindFirstChild("DevAutomationRun")\n'
                                    "if autoEvent then\n"
                                    "    autoEvent:Fire()\n"
                                    "    task.wait(5)\n"
                                    "else\n"
                                    '    print("[SMOKE] DevAutomationRun event not found")\n'
                                    "end"
                                ),
                            },
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"output": "[DevAutomationController] Tool smoke finished\\n  PASS grapple_valid_confirm\\n  PASS grapple_invalid_reject\\n  PASS glide_valid_confirm_airborne\\n  FAIL slide_surface_detection - expected Sliding, got Idle\\n  PASS wallrun_attach"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Traversal smoke test results:\n"
                        "- PASS: grapple_valid_confirm\n"
                        "- PASS: grapple_invalid_reject\n"
                        "- PASS: glide_valid_confirm_airborne\n"
                        "- **FAIL**: slide_surface_detection -- expected Sliding state but got Idle\n"
                        "- PASS: wallrun_attach\n\n"
                        "4/5 tests passed. The slide_surface_detection failure suggests the slide trigger "
                        "isn't detecting SlideSurface-tagged parts correctly, or the surface angle check "
                        "is rejecting the test surface. Let me stop the playtest."
                    ),
                    "tool_calls": [
                        make_tool_call(next_id(), "start_stop_play", {"action": "stop"}),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 5,
        }
    )

    # 56. Batch delete obsolete parts
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Clean up all parts named 'DEBUG_*' from the entire workspace."},
                {
                    "role": "assistant",
                    "content": ("I'll use run_code to find and remove all debug parts in one pass."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local toDelete = {}\n"
                                    "for _, inst in workspace:GetDescendants() do\n"
                                    '    if inst.Name:sub(1, 6) == "DEBUG_" then\n'
                                    "        table.insert(toDelete, inst)\n"
                                    "    end\n"
                                    "end\n"
                                    'print("Found " .. #toDelete .. " debug objects")\n'
                                    "for _, inst in toDelete do\n"
                                    "    inst:Destroy()\n"
                                    "end\n"
                                    'print("Deleted all DEBUG_ objects")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_vertigo",
            "difficulty": 2,
        }
    )

    # 57. set_calculated_property with CFrame
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Move the Hub beacon 20 studs higher."},
                {
                    "role": "assistant",
                    "content": ("I'll read the current position, then set the new position 20 studs higher."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(), "get_instance_properties", {"path": "Workspace.SceneRoot.HubZone.Beacon"}
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"properties": {"Name": "Beacon", "Position": [0, 30, 0], "Size": [5, 20, 5], "Anchored": true}}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": "Currently at Y=30. Moving to Y=50.",
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "set_property",
                            {
                                "path": "Workspace.SceneRoot.HubZone.Beacon",
                                "property": "Position",
                                "value": [0, 50, 0],
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 2,
        }
    )

    # 58. insert_model + reconfigure
    scenarios.append(
        {
            "tools": MCP_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Insert the rock formation model (asset 55555555) into the Abyss zone, make it 3x bigger, and anchor it.",
                },
                {
                    "role": "assistant",
                    "content": ("I'll insert the model and then scale and anchor it."),
                    "tool_calls": [
                        make_tool_call(
                            next_id(), "insert_model", {"assetId": 55555555, "parent": "Workspace.SceneRoot.AbyssZone"}
                        ),
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"success": true, "path": "Workspace.SceneRoot.AbyssZone.RockFormation"}',
                    "tool_call_id": cur_id(),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        make_tool_call(
                            next_id(),
                            "run_code",
                            {
                                "code": (
                                    "local rock = workspace.SceneRoot.AbyssZone.RockFormation\n"
                                    "rock:ScaleTo(3)\n"
                                    "for _, part in rock:GetDescendants() do\n"
                                    '    if part:IsA("BasePart") then\n'
                                    "        part.Anchored = true\n"
                                    "    end\n"
                                    "end\n"
                                    'print("Rock formation scaled 3x and all parts anchored")'
                                ),
                            },
                        ),
                    ],
                },
            ],
            "source": "mcp_tool_calling",
            "category": "mcp_multi_step",
            "difficulty": 3,
        }
    )

    return scenarios


def main():
    scenarios = build_scenarios()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for s in scenarios:
            f.write(json.dumps(s) + "\n")

    print(f"Generated {len(scenarios)} MCP tool-calling examples -> {OUTPUT}")

    # Stats
    from collections import Counter

    cats = Counter(s.get("category", "unknown") for s in scenarios)
    diffs = Counter(s.get("difficulty", 0) for s in scenarios)
    print(f"Categories: {dict(cats)}")
    print(f"Difficulty distribution: {dict(sorted(diffs.items()))}")

    tools_used = set()
    for s in scenarios:
        for m in s["messages"]:
            if "tool_calls" in m and m["tool_calls"]:
                for tc in m["tool_calls"]:
                    tools_used.add(tc["function"]["name"])
    print(f"Tools covered: {len(tools_used)}/{len(MCP_TOOLS)} — {sorted(tools_used)}")


if __name__ == "__main__":
    main()
