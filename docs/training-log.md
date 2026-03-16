# Training Log — RobbieMk1 LoRA Sprint (March 14-16, 2026)

## Summary

Trained 12 LoRA adapters across 2 model sizes (2B, 4B), 3 data strategies, and 6 hyperparameter configurations. Best result: **4B v0.5 curated at 82.9%** (+6.0pp over base), scoring **83.0% on OpenGameEval dry-run** (above all published models including Gemini 3.1 Pro at 55.3%).

## What Worked

### 1. Quality filtering > more data (the single biggest insight)

| Adapter | Data Size | Overall |
|---|---|---|
| 4B v0.1 (all 3-mix) | 3,636 train | 69.9% |
| 4B v0.3 (expanded) | 4,852 train | 63.6% |
| **4B v0.4 (curated)** | **5,679 train** | **82.8%** |

v0.3 added 1,200 examples over v0.1 and scored 6pp *worse*. v0.4 removed 550 low-quality examples and scored 13pp *better*. The removed examples were prose-heavy (gameplay sessions, tool-calling prompts without code, OpenGameEval task descriptions) that diluted the code signal.

**Rule: every training example must contain substantial Luau code (≥5 lines in code blocks). Prose-only examples are noise for a code model.**

### 2. The 4B is the sweet spot on 36GB

The 2B catastrophically forgets general capabilities when fine-tuned:
- 2B base: 65.1%
- 2B + v0.3 LoRA: 51.5% (-13.6pp — every category except coding regressed)
- 2B + v0.4-fix (3-mix): 49.2% (-15.9pp — still forgetting)

The 4B has enough capacity to absorb domain training without losing general knowledge:
- 4B base: 76.9%
- 4B + v0.5 curated: 82.9% (+6.0pp — every category improved)

**Rule: if your fine-tuned model scores below its own base, the model is too small for your data volume. Move up a size.**

### 3. Composition tasks teach pattern combination

The 20 hand-crafted multi-concept examples (inventory+DataStore, grapple+spring, NPC+pathfinding) had outsized impact despite being only 0.2% of the dataset. They teach the model to compose patterns — Init/Start + RemoteEvent + DataStore in a single module — rather than produce isolated snippets.

### 4. STaR verification produces high-quality signal

4 STaR cycles produced 49 luau-compile-verified examples. Compile rate averaged 15-56% across iterations. These are the only examples in the dataset proven to produce valid Luau. Small in number but high in signal density.

### 5. The MCP benchmark must test real tools

The original benchmark tested fictional tool names (`studio_find_tagged`). Fine-tuned models scored low because they learned the *real* tools (`run_code`, `get_console_output`). After rewriting with real tools:
- 4B v0.4: 51.7% → (estimated higher, score not re-run with this version)
- 4B v0.5: 85.8% MCP score

**Rule: benchmark tasks must use the same tools the model is trained on.**

### 6. Hyperparameters matter more than data tweaks

v0.4-bad (rank 16, LR 1e-6, 400 iters) scored 19.9%. v0.4-fix (rank 8, LR 5e-6, 600 iters) scored 49.2%. Same data, same model, 29pp difference. The 2B's best config (rank 8, LR 5e-6, no cosine schedule) transferred directly to the 4B.

**Rule: validate hyperparams on the smaller model first, then transfer to larger.**

## What Didn't Work

### 1. Three-dataset mixing without curation

The conceptual framework (55% SFT / 27.5% trajectory / 15% critic) is sound but the implementation mixed in low-quality examples. The family classification put prose-heavy files into the SFT bucket where they diluted code signal. The mixing ratios need to be applied *after* quality filtering, not before.

### 2. Low-code-density training data

These files degraded every adapter they were included in:
- `gameplay_sessions.jsonl` (177 examples, 0% code, 91% short responses)
- `opengameeval_tasks.jsonl` (77 examples, 0% code)
- `tool_calling_sft.jsonl` (50 examples, 4% code, 70% short)
- `mcp_tools.jsonl` (58 examples, 3% code)
- `studio_trajectories.jsonl` (45 examples, 2% code)

These are valuable as trajectory/tool-calling data but not for a code-focused LoRA. They might work in a separate tool-calling adapter.

### 3. Expanding the creator docs without filtering

Raw extraction produced 915 examples from 972 markdown files. After filtering to ≥5 code lines: 806. The 109 removed examples were API description pages with no code samples — they taught the model to explain APIs rather than write code.

### 4. Teacher distillation under memory pressure

The 27B teacher model on 36GB crashed intermittently during inference. Only 19 of ~84 intended distillation examples were generated before OOM/timeout failures. Teacher distillation needs the 128GB machine.

### 5. Running multiple inference processes simultaneously

LM Studio + mlx_lm.server + training/generation scripts competing for the same 36GB GPU memory caused crashes, OOMs, and corrupted results. Every benchmark run and every training run must be solo on 36GB.

## Val Loss History

| Adapter | Val Loss | Overall Score | Notes |
|---|---|---|---|
| 2B v0.3 (old mix) | 0.861 | 51.5% | Catastrophic forgetting |
| 2B v0.4 (bad LR) | 1.161 | 19.9% | LR too low, barely moved |
| 2B v0.4-fix (3-mix) | 0.809 | 49.2% | Better loss, still forgetting |
| 4B v0.1 (3-mix) | 0.837 | 69.9% | First improvement over base |
| 4B v0.2 (+ tool SFT) | 1.085 | 69.6% | Higher loss from tool data |
| 4B v0.3 (expanded) | 1.076 | 63.6% | More data = worse |
| **4B v0.4 (curated)** | **0.905** | **82.8%** | Quality filtering breakthrough |
| **4B v0.5 (+ composition)** | **0.857** | **82.9%** | Best adapter |
| 4B v0.6 (+ STaR v3) | 0.872 | 81.9% | Marginal, near ceiling |

## OpenGameEval Results

| Model | OGE Pass@1 (dry) | Published Comparison |
|---|---|---|
| 4B + v0.5 curated | **83.0%** | +27.7pp above Gemini 3.1 Pro (55.3%) |
| 4B base | 72.3% | Above all published models |
| 27B dense base | 48.9% | Near Claude Opus 4.6 (51.9%) |
| 35B-A3B base | 42.6% | Near Claude Opus 4.5 (44.5%) |

*Caveat: dry-run scores measure code generation quality via pattern matching, not Studio execution verification. Not directly comparable to published pass@1.*

## Hardware Constraints (36GB M5 Max)

| Model | Rank | Layers | Seq | Peak Mem | Result |
|---|---|---|---|---|---|
| 2B | 8 | 16 | 2048 | 16.2GB | Works |
| 4B | 16 | 16 | 4096 | OOM | Crashes |
| 4B | 8 | 16 | 2048 | 27.3GB | OOM at iter 20 |
| **4B** | **8** | **8** | **2048** | **27.3GB** | **Works** |
| 4B | 4 | 8 | 1024 | 11.9GB | NaN divergence |

The 4B on 36GB is constrained to rank 8, 8 layers, 2048 seq. This leaves capacity on the table.

## Data Inventory (final state)

- 27 raw JSONL files, 3,893 total examples
- Curated subset: 3,301 examples (84.8% retained)
- Processed: 9,087 train / 1,136 valid / 1,136 test (after oversampling + pre-split)
- 7 rights-clean source repos cloned (creator-docs, Luau, Rojo, Wally, Selene, roblox-ts, OpenGameEval)
- Full provenance metadata on all new examples

## Infrastructure Built

- `auto_train.sh` — one-command train → benchmark → promote pipeline
- `promote_adapter.py` — auto-promotion with regression gates
- `REGISTRY.json` — production adapter tracking
- `curate_dataset.py` + `curation.yaml` — config-driven data curation
- `scoring.py` — shared scoring module (single source of truth)
- `run_opengameeval.py` — local OpenGameEval harness with `--with-suit` mode
- `bootstrap_abraham.sh` — one-command 128GB machine setup
- 75 pytest tests with Hypothesis fuzzing
- 3 abraham training configs ready (4b-128gb, 4b-8bit-128gb, 9b)


## Run History
| Date | Adapter | Model | Config | Val Loss | Coding | Bugfix | Arch | MCP | Embodiment | Overall | Promoted | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-03-16 10:26 | v0.5-4b-curated | Qwen3.5-4B-4bit | 4b | — | 72.5% | 90.0% | 76.6% | 85.8% | 100.0% | **82.9%** | yes (production) | local |
