#!/usr/bin/env python3
from __future__ import annotations

"""
Create the Vertigo LoRA epic and sub-issues in Linear.

Usage:
  LINEAR_API_KEY=... uv run scripts/create_linear_issue.py [--dry-run]
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

API_URL = "https://api.linear.app/graphql"


def gql(api_key: str, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": api_key},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
    obj = json.loads(body)
    if obj.get("errors"):
        raise RuntimeError(json.dumps(obj["errors"]))
    return obj["data"]


EPIC_TITLE = "LoRA Fine-Tuning Pipeline: Vertigo-Specialized Local Coding Model"

EPIC_DESCRIPTION = """## Summary

Build and deploy a LoRA-adapted local LLM (Qwen3.5-27B on MLX) specialized for Vertigo development + Roblox Studio MCP tool calling. The model runs on the M5 Max 36GB via LM Studio, providing unlimited, free, offline agentic coding with full project context.

## Strategic Context

- **ROI:** 15-30% velocity gain on Vertigo-specific tasks (research-backed)
- **Cost:** $0 compute (MLX on local hardware), ~2-3 days pipeline investment
- **Risk mitigation:** Train v0.1 immediately on existing 1,033 examples, iterate
- **Roadmap fit:** Force multiplier for Phase A beta readiness (Mar-Apr 2026)

## Current State (v0.3)

| Metric | Value |
|---|---|
| Training examples | 1,033 |
| Estimated tokens | ~2M |
| Composite quality score | 70.2% |
| Roblox service coverage | 95% (20/21) |
| Vertigo pattern coverage | 100% (24/24) |
| MCP tool coverage | 100% (15/15) |

## Dataset Sources (Modular)

- `codebase_*.jsonl` — 315 examples from all 309 .luau files (granular per-category)
- `api_docs.jsonl` — 100 Roblox API training pairs
- `mcp_tools.jsonl` — 58 MCP tool-calling examples (all 15 tools)
- `devforum_qa.jsonl` — 80 community Q&A patterns
- `bugfix_pairs.jsonl` — 50 bug detection/correction pairs
- `evolved.jsonl` — 120 Evol-Instruct complexity-scaled examples

## Architecture

```
vertigo/lora/
├── configs/default.yaml          # MLX LoRA config (rank 32, all attn+MLP)
├── scripts/                      # Full pipeline (extract → validate → dedup → train)
├── data/raw/                     # Modular per-source JSONL files
├── data/processed/               # MLX-ready train/valid/test splits
├── adapters/                     # Timestamped LoRA checkpoints
└── docs/snapshots/               # Dataset quality snapshots for tracking
```

## Success KPIs

- Eval loss >10% lower than base model
- First-try acceptance rate >60%
- Convention compliance >90% (--!strict, @native, Init/Start)
- MCP tool accuracy >80%

## References

- Pipeline: `vertigo/lora/README.md`
- Strategic analysis: saved in Claude memory
- Framework analysis: Nevermore/Knit/jecs evaluated, recommendation is cherry-pick not adopt
"""

SUB_ISSUES = [
    {
        "title": "Train and deploy LoRA v0.1 baseline adapter",
        "description": (
            "Run first training on existing 1,033 examples.\n\n"
            "- Install MLX: `uv pip install mlx-lm`\n"
            "- Run: `cd lora && ./scripts/train.sh v0.1-baseline`\n"
            "- Consider reducing iters from 600→400 to avoid overfitting\n"
            "- Evaluate on 49 test examples\n"
            "- Serve via `mlx_lm.server` and test with Qwen Code\n"
            "- Keep acceptance rate log for 1 week"
        ),
        "priority": 1,
    },
    {
        "title": "Split AbilityController into sub-controllers",
        "description": (
            "AbilityController is 4,231 lines — too large to maintain.\n\n"
            "Split into:\n"
            "- GrappleController (~1200 lines)\n"
            "- GlideController (~800 lines)\n"
            "- SlideController (~400 lines)\n"
            "- AirDashController (~300 lines)\n"
            "- WallRunController (~300 lines)\n"
            "- AbilityController (orchestrator, ~500 lines)\n\n"
            "Use LoRA v0.1 to assist. Document where it helps vs hallucinated."
        ),
        "priority": 2,
    },
    {
        "title": "Add RemoteMiddleware for centralized validation",
        "description": (
            "Rate limiting, input validation, and logging are duplicated across services.\n\n"
            "Create a RemoteMiddleware layer that:\n"
            "- Centralizes rate limiting (10 msgs/sec, 16ms debounce)\n"
            "- Validates payload types before reaching service handlers\n"
            "- Logs all remote traffic for debugging\n"
            "- Pluggable per-remote configuration"
        ),
        "priority": 2,
    },
    {
        "title": "Lightweight ServiceRegistry with auto-discovery",
        "description": (
            "Replace hard-coded service list in init.server.luau with auto-discovery.\n\n"
            "- Scan Services/ directory for modules with :Init()/:Start()\n"
            "- Optional dependency declaration for init ordering\n"
            "- Graceful error isolation (one service failure doesn't block others)\n"
            "- Inspired by Nevermore's ServiceBag pattern"
        ),
        "priority": 3,
    },
    {
        "title": "Standardize TagCache across all services",
        "description": (
            "Some services use TagCache for CollectionService lookups, others don't.\n\n"
            "- Create shared TagCache utility in Shared/Util/\n"
            "- Migrate all services to use it consistently\n"
            "- Add invalidation signals for dynamic tag changes"
        ),
        "priority": 3,
    },
    {
        "title": "MCP trajectory recording for ground-truth training data",
        "description": (
            "Build a script that connects to the running Studio MCP server, executes "
            "real operations, and records request/response pairs as training data.\n\n"
            "This produces verified ground-truth examples (not synthetic) from actual "
            "Studio state — the single highest-signal data source possible.\n\n"
            "Output: `data/raw/mcp_trajectories.jsonl`"
        ),
        "priority": 2,
    },
    {
        "title": "OSS Roblox codebase extraction for LoRA training",
        "description": (
            "Extract training data from open-source Roblox projects:\n"
            "- NevermoreEngine (270 packages, DI, Binder, Blend, camera, IK)\n"
            "- jecs (high-performance ECS, entity relationships)\n"
            "- Matter (ECS, hot-reload, debugger)\n"
            "- Reflex (Redux-like state management)\n"
            "- ByteNet (buffer-based networking)\n"
            "- Knit (archived but clean service/controller reference)\n"
            "- luau-lang/luau (language tests and examples)\n\n"
            "Output: `data/raw/oss_roblox.jsonl`"
        ),
        "priority": 3,
    },
    {
        "title": "Retrain LoRA v0.2 on improved codebase + expanded dataset",
        "description": (
            "After architecture improvements, re-extract and retrain:\n\n"
            "- Re-run `./scripts/retrain.sh`\n"
            "- Add OSS Roblox data, MCP trajectories, telemetry patterns\n"
            "- Fill dataset gaps identified from v0.1 usage\n"
            "- Expand eval set to 100+ examples\n"
            "- Compare v0.1 vs v0.2 via `analyze_dataset.py --compare`\n"
            "- Target composite score >85%"
        ),
        "priority": 2,
    },
    {
        "title": "Full local coding stack integration",
        "description": (
            "Wire up the complete local coding workflow:\n\n"
            "- LM Studio serving LoRA-adapted model on localhost:1234\n"
            "- Qwen Code → LM Studio (terminal agent)\n"
            "- OpenCode → LM Studio (alternative terminal agent)\n"
            "- Cline/Roo Code VS Code extension → LM Studio\n"
            "- Roblox Studio MCP server connected\n"
            "- Test full loop: ask model to modify Vertigo code via MCP"
        ),
        "priority": 2,
    },
    {
        "title": "GRPO reinforcement learning with Luau CLI verifier",
        "description": (
            "Use the Luau CLI as a reward signal for reinforcement learning:\n\n"
            "- Correct Luau code = positive reward\n"
            "- Syntax errors = negative reward\n"
            "- Use mlx-lm-lora fork for GRPO support\n"
            "- Generate DPO preference pairs from verified vs unverified code\n"
            "- Target: v0.3 adapter with RL-tuned code correctness"
        ),
        "priority": 4,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Vertigo LoRA Linear issues")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions only")
    args = parser.parse_args()

    api_key = os.environ.get("LINEAR_API_KEY", "").strip()
    if not api_key:
        print("error: LINEAR_API_KEY is required", file=sys.stderr)
        print("Set it in your .env file or export it directly.")
        return 1

    print("Fetching workspace snapshot...")

    if args.dry_run:
        print("\n=== DRY RUN ===\n")
        print(f"EPIC: {EPIC_TITLE}")
        print(f"  Description: {len(EPIC_DESCRIPTION)} chars")
        print()
        for i, sub in enumerate(SUB_ISSUES, 1):
            print(f"  SUB-ISSUE {i} [P{sub['priority']}]: {sub['title']}")
        print(f"\nTotal: 1 epic + {len(SUB_ISSUES)} sub-issues")
        return 0

    # Fetch team and states
    snapshot = gql(
        api_key,
        """
        query {
          teams(first: 50) {
            nodes { id key name states { nodes { id name type } } }
          }
          issues(first: 500) {
            nodes { id identifier title }
          }
        }
    """,
    )

    # Find the VER team
    team = None
    for t in snapshot["teams"]["nodes"]:
        if t["key"] == "VER":
            team = t
            break

    if not team:
        # Try first team
        team = snapshot["teams"]["nodes"][0]
        print(f"VER team not found, using: {team['key']} ({team['name']})")
    else:
        print(f"Found team: {team['key']} ({team['name']})")

    team_id = team["id"]

    # Find "Backlog" or "Triage" state
    state_id = None
    for s in team["states"]["nodes"]:
        if s["name"].lower() in ("backlog", "triage", "todo"):
            state_id = s["id"]
            break
    if not state_id:
        state_id = team["states"]["nodes"][0]["id"]

    # Check if epic already exists
    existing = {i["title"]: i for i in snapshot["issues"]["nodes"]}
    if EPIC_TITLE in existing:
        epic = existing[EPIC_TITLE]
        print(f"Epic already exists: {epic['identifier']} - {EPIC_TITLE}")
    else:
        # Create epic
        data = gql(
            api_key,
            """
            mutation IssueCreate($input: IssueCreateInput!) {
              issueCreate(input: $input) {
                success
                issue { id identifier title url }
              }
            }
        """,
            {
                "input": {
                    "teamId": team_id,
                    "title": EPIC_TITLE,
                    "description": EPIC_DESCRIPTION,
                    "priority": 1,
                    "stateId": state_id,
                }
            },
        )
        epic = data["issueCreate"]["issue"]
        print(f"Created epic: {epic['identifier']} - {EPIC_TITLE}")
        print(f"  URL: {epic.get('url', '')}")

    # Create sub-issues
    print(f"\nCreating {len(SUB_ISSUES)} sub-issues...")
    for sub in SUB_ISSUES:
        if sub["title"] in existing:
            issue = existing[sub["title"]]
            print(f"  exists: {issue['identifier']} - {sub['title']}")
            continue

        data = gql(
            api_key,
            """
            mutation IssueCreate($input: IssueCreateInput!) {
              issueCreate(input: $input) {
                success
                issue { id identifier title url }
              }
            }
        """,
            {
                "input": {
                    "teamId": team_id,
                    "title": sub["title"],
                    "description": sub["description"],
                    "priority": sub["priority"],
                    "stateId": state_id,
                    "parentId": epic["id"],
                }
            },
        )
        issue = data["issueCreate"]["issue"]
        print(f"  created: {issue['identifier']} - {sub['title']}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
