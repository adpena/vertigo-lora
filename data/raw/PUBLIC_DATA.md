# Training Data Classification

Files in `data/raw/` classified for public repository inclusion.

## PUBLIC (included in public repo)

| File | License | Notes |
|---|---|---|
| `roblox_creator_docs.jsonl` | CC-BY-4.0 | Extracted from Roblox Creator documentation |
| `api_docs.jsonl` | CC-BY-4.0 | Roblox API usage examples |
| `luau_tests.jsonl` | MIT | From luau-lang/luau test suite |
| `rojo_architecture.jsonl` | MPL-2.0 | From rojo-rbx/rojo |
| `selene_lint.jsonl` | MPL-2.0 | From Kampfkarren/selene |
| `roblox_ts_patterns.jsonl` | MIT | From roblox-ts/roblox-ts |
| `opengameeval_tasks.jsonl` | MIT | OpenGameEval benchmark tasks |
| `star_verified.jsonl` | Generated | Self-generated via STaR verification |
| `composition_tasks.jsonl` | Generated | Hand-crafted composition examples |
| `rejection_sampled.jsonl` | Generated | Rejection sampling from base model |
| `teacher_distillation.jsonl` | Generated | Distilled from teacher model outputs |
| `bugfix_pairs.jsonl` | Generated | Synthetic bug/fix pairs |
| `synthetic.jsonl` | Generated | Synthetically generated examples |
| `evolved.jsonl` | Generated | Evol-Instruct complexity-scaled examples |
| `critic_repair.jsonl` | Generated | Critic/repair self-improvement examples |
| `tool_calling_sft.jsonl` | Generated | MCP tool-calling SFT pairs |
| `studio_trajectories.jsonl` | Generated | Studio MCP session trajectories |
| `gameplay_sessions.jsonl` | Generated | Embodied gameplay session captures |
| `embodiment_gameplay.jsonl` | Generated | Embodiment gameplay examples |
| `mcp_tools.jsonl` | Generated | MCP tool schema examples |
| `devforum_qa.jsonl` | Fair use (rewritten) | DevForum Q&A, rewritten to avoid verbatim copying |

## PRIVATE (excluded from public repo)

These files contain proprietary Vertigo codebase data and are excluded via `.gitignore`.

| File | Reason |
|---|---|
| `codebase.jsonl` | Vertigo proprietary source code |
| `codebase_controllers.jsonl` | Vertigo controller modules |
| `codebase_builders.jsonl` | Vertigo zone builder modules |
| `codebase_services.jsonl` | Vertigo service modules |
| `codebase_config.jsonl` | Vertigo config modules |
| `codebase_physics.jsonl` | Vertigo physics systems |
| `codebase_patterns.jsonl` | Vertigo code patterns |
| `codebase_networking.jsonl` | Vertigo networking modules |
| `codebase_other.jsonl` | Vertigo miscellaneous modules |
| `oss_roblox.jsonl` | Mixed OSS licenses, needs per-file audit before public release |
