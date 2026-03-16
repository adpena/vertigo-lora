[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Model-yellow)](https://huggingface.co/adpena/Vertigo-Qwen3.5-4B-v0.5-4bit)
[![Results](https://img.shields.io/badge/Results-lora.vertigo.build-brightgreen)](https://lora.vertigo.build)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-Apple_Silicon-black)](https://github.com/ml-explore/mlx)

# Vertigo LoRA

Domain-specialized LoRA fine-tuning pipeline for Roblox/Luau code generation on Apple Silicon.

**[Live Results →](https://lora.vertigo.build)** | **[Models →](https://huggingface.co/adpena/Vertigo-Qwen3.5-4B-v0.5-4bit)** | **[Methodology →](data/eval/METHODOLOGY.md)**

## Key Results

| Category | Qwen3.5-4B Base | Vertigo-4B v0.5 | Delta |
|---|---|---|---|
| Coding | 63.7% | 72.5% | +8.8pp |
| Bugfix | 83.3% | 90.0% | +6.7pp |
| Architecture | 67.5% | 76.6% | +9.1pp |
| MCP | 97.5% | 85.8% | -11.7pp |
| Embodiment | 75.0% | 100.0% | +25.0pp |
| **Overall** | **75.1%** | **82.9%** | **+7.8pp** |

OpenGameEval dry-run: **83.0% pass rate** on 47 held-out Roblox tasks (pattern-match scoring, not execution-verified — see [methodology](data/eval/METHODOLOGY.md) for caveats).

## Prerequisites

- **Apple Silicon Mac** — M1+ for inference, M4/M5 with 36GB+ unified memory for training
- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv) package manager**
- **~10GB disk space** for models + training data

## Quick Start

```bash
git clone https://github.com/adpena/vertigo-lora
cd vertigo-lora
uv pip install -e ".[dev]"

# Train an adapter
./scripts/auto_train.sh 4b

# Benchmark
uv run python scripts/run_benchmark.py --adapter adapters/latest
```

## Architecture

The pipeline uses a **3-dataset system** to produce high-quality, rights-clean training data:

1. **SFT pairs** — instruction/completion pairs extracted from documentation, open-source repos, and hand-crafted examples covering Roblox API usage, Luau patterns, and framework conventions.
2. **Trajectories** — multi-turn tool-calling sequences from MCP sessions, gameplay captures, and Studio interactions that teach the model realistic agentic workflows.
3. **Critic/repair** — self-generated verification and correction examples via STaR (Self-Taught Reasoner) and rejection sampling that improve the model's ability to detect and fix errors.

All examples include reasoning traces (`<think>` blocks) and are scored on a 1-5 difficulty scale for balanced curriculum training. The curation pipeline applies Pydantic validation, quality scoring, exact + MinHash deduplication, and category balancing.

## Training Data

### Included in this repo (rights-clean)

| Source | Examples | License |
|---|---|---|
| Roblox Creator Docs | 806 | CC-BY-4.0 |
| OSS Roblox repos | 1,301 | Various OSS |
| Luau, Rojo, Wally, Selene, roblox-ts | 61 | MIT/MPL-2.0 |
| STaR-verified (self-generated) | 49 | Generated |
| Composition tasks (hand-crafted) | 20 | Generated |
| Rejection sampled | 34 | Generated |
| Teacher distillation | 19 | Generated |
| Bugfix pairs | 50 | Generated |
| Synthetic/evolved | 170 | Generated |
| Critic/repair | 28 | Generated |
| API docs | 100 | CC-BY-4.0 |
| DevForum Q&A | 80 | Fair use (rewritten) |

### Proprietary data used for training (not included)

The production adapter (v0.5) was additionally trained on proprietary source code from **[Vertigo](https://vertigo.build)** — a platform for modern Roblox development tooling, AI-assisted game building, and persistent AI embodiment within Roblox experiences. Vertigo provides developer infrastructure including a sub-millisecond sync engine (Vertigo Sync), a DSL compiler for agent-driven world building (Suit SDK), MCP-instrumented Studio tooling, an SDK and harness for persistent AI agents embodied as characters in live Roblox worlds, autonomous multi-agent workflows, and a portable agent identity system (OAC). The reference experience is a physics-driven vertical exploration game with chained traversal abilities, procedural world generation, — serving as both a showcase and a testbed for the tooling and embodiment stack.

The project spans multiple languages and systems:

- **117K lines of `--!strict` Luau** — the game runtime: services, controllers, zone builders, physics, abilities, networking, UI
- **Rust (Strata / Suit SDK / Vertigo Sync)** — the core innovations of the platform:
  - **Strata**: the foundational framework providing the Suit SDK runtime, compilation pipeline, and cross-language codegen
  - **Suit SDK & DSL**: a domain-specific language and compiler that transpiles high-level agent instructions (natural language, expression trees, Logo turtle geometry) to optimized Luau — enabling AI agents to build and modify game worlds through a compiled, type-safe interface rather than raw code generation. Includes a Rust-to-Python codegen bridge with 412 differential tests
  - **Qualia Synthesizer**: a perceptual layer that transforms raw Studio MCP telemetry (coordinates, distances, materials, colors) into rich spatial-temporal descriptions that give embodied agents a genuine sense of *being in a place* — converting data snapshots into felt experience through spatial grounding (egocentric relationships), temporal differencing (what changed since last frame), and aesthetic synthesis (material feel, light, atmosphere)
  - **Vertigo Sync**: a sub-millisecond source sync engine replacing Rojo for real-time Studio development, with native FSEvents watching, incremental snapshot caching, mmap, Luau validation, and HTTP/SSE/WebSocket server
- **Python (Fleet/Agent)** — a 5-machine local AI fleet with Obelisk inference proxy, Qwen Design Council (4-agent continuous deliberation), Discord relay, MCP server (99+ tools), and Toga/BeeWare cross-platform native UI (macOS, iOS/iPad)
- **TypeScript (Site/WebMCP)** — an interactive Three.js showcase site with procedural crystal geometry, GPU particle systems, and zone-reactive atmospheric effects; plus WebMCP browser-native AI agent polyfill
- **[OAC (Open Agent Capsule)](https://github.com/adpena/oac)** — a portable agent identity and projection system that compiles capsules to 8 harness targets (Claude Code, Codex, Gemini, OpenCode, OpenClaw, MCP, WebMCP, Roblox Embodiment)

This proprietary codebase data is not included in the public repository but contributed significantly to the model's understanding of production Roblox/Luau patterns:

- **Service/controller architecture** (37 services, 135 controllers) — Init/Start two-phase boot pattern with explicit dependency ordering, service-controller separation for server/client responsibility, and module-level `@native` optimization on hot paths
- **Zone builders** (56 procedural world builders) — CollectionService tagging for runtime discovery, procedural geometry generation with `Instance.new` pooling, crystal/terrain/vegetation placement, and per-zone atmospheric configuration
- **Physics systems** — Critically-damped spring solvers for camera and movement interpolation, Baumgarte-stabilized constraints, vehicle controllers (dirt bike, glider) with fixed-step PreSimulation physics loops using `vector.*` SIMD operations
- **Config modules** — Named exports pattern (`return { GrappleTuning, GlideTuning, ... }`), `table.freeze` for immutable configs, `export type` definitions for type-safe consumption, and runtime attribute-based hot-reload
- **Networking patterns** — Server-authoritative RemoteEvent validation (cooldown checks, range validation, line-of-sight, player state), structured client-server request/response flow (`RequestUseAbility` → validate → `FireClient`), and replication-safe state synchronization
- **Ability system** — Full traversal ability stack (grapple hook with spring-based travel and momentum preservation, air dash with burst acceleration, glide with planar velocity steering, wall run with surface detection, slide with terrain-following), cooldown management, input buffering, and graceful state transitions
- **Embodied agent runtime** — Edit Mode Runtime engine that animates AI agents at 60Hz in Studio, WebSocket-based RuntimeBridge for multi-agent coordination, patrol systems with waypoint navigation, and agent-to-agent encounter behaviors
- **MCP instrumentation** — 12+ Studio MCP tools for programmatic game development (run_code, get_console_output, run_script_in_play_mode), Fleet DSL wrappers for structured queries (world stats, zone health, physics audit), and automated traversal smoke testing

This proprietary data comprised ~631 examples (19% of the curated training set). The model can be retrained without it using only the included rights-clean data, though performance may differ from the published benchmarks — the proprietary data provides the densest signal for Vertigo-specific conventions that general Roblox documentation does not cover.

## Evaluation

See [`data/eval/METHODOLOGY.md`](data/eval/METHODOLOGY.md) for the full evaluation methodology, benchmark descriptions, and known caveats.

Live results are published at [lora.vertigo.build](https://lora.vertigo.build).

## Pipeline

The automated pipeline runs 7 steps end-to-end:

1. **Train** — QLoRA fine-tuning via MLX on Apple Silicon (rank 32, all attention + MLP layers)
2. **Benchmark** — Run held-out eval suite (Luau syntax, API usage, service patterns, type coverage)
3. **Promote** — If metrics improve, symlink `adapters/latest` to the new checkpoint
4. **Analyze** — Generate per-category performance reports and regression analysis
5. **Log** — Write timestamped run metadata to `docs/runs/`
6. **Fuse** — Merge adapter weights into base model for faster inference
7. **Done** — Final validation and cleanup

## Models

| Model | Size | Link |
|---|---|---|
| Vertigo-Qwen3.5-4B-v0.5-4bit (fused) | 2.2 GB | [HuggingFace](https://huggingface.co/adpena/Vertigo-Qwen3.5-4B-v0.5-4bit) |
| Vertigo-Qwen3.5-4B-v0.5-lora (adapter) | 62 MB | [HuggingFace](https://huggingface.co/adpena/Vertigo-Qwen3.5-4B-v0.5-lora) |

## Hardware Requirements

- **Training:** Apple Silicon with 36GB+ unified memory (M5 Max, M4 Ultra, etc.)
- **Inference:** Any Apple Silicon Mac (8GB+ for 4B model)
- **Recommended:** 128GB for 9B training and full-rank 4B

## Contributing

Contributions welcome — please open an issue first to discuss. Areas of interest:
- Additional rights-clean Roblox/Luau training data
- Benchmark task contributions
- Evaluation methodology improvements
- Bug reports and fixes

## Citation

```bibtex
@software{pena2026vertigo_lora,
  author = {Pe\~{n}a, Alejandro},
  title = {Vertigo LoRA: Domain-Specialized Fine-Tuning for Roblox/Luau Code Generation},
  year = {2026},
  url = {https://github.com/adpena/vertigo-lora},
}
```

## License

Apache 2.0 (same as base Qwen3.5 model)

## Acknowledgments

- [Apple MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) for making on-device fine-tuning practical
- [Qwen](https://huggingface.co/Qwen) team for the excellent Qwen3.5 base models (Apache 2.0)
- [Roblox OpenGameEval](https://github.com/Roblox/open-game-eval) for the evaluation benchmark and leaderboard
- [Roblox Creator Documentation](https://create.roblox.com/docs) (CC-BY-4.0) for high-quality training data
- The open-source Roblox community for publicly available code and tools
- [Claude](https://anthropic.com) (Anthropic) for development assistance

## Contact

Alejandro Pena — adpena@vertigo.build
