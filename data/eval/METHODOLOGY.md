# Evaluation Methodology

Version: 0.2.1 | Scoring method: `pattern_match_v2`
Last updated: 2026-03-15

This document describes how Vertigo LoRA adapters are evaluated, what the scores mean, and — critically — what they do not mean. It is the authoritative evaluation reference for the project.

> **Honesty note:** This benchmark is a v0.1 pattern-matching baseline. It tells us whether the adapter shifts output toward expected patterns, but it cannot tell us whether generated code is correct, compiles, or behaves as intended. Execution-based evaluation (v1.0) is planned.

---

## 1. Benchmark Design

### Task Inventory

The benchmark consists of 30 structured tasks across 5 categories:

| Category | Count | Difficulty Range | Rationale |
|---|---|---|---|
| Coding | 10 | 2-5 | Core use case: generating Luau services, controllers, and systems following Vertigo conventions |
| MCP Tool Calling | 5 | 2-3 | Adapter should improve the model's ability to select and invoke the correct Studio MCP tool |
| Bugfix | 5 | 2-3 | Real-world usage pattern: diagnosing and fixing broken Luau code with Vertigo-specific knowledge |
| Architecture | 5 | 4-5 | Higher-order reasoning: boot ordering, service decomposition, rate limiting, config-driven systems |
| Embodiment | 5 | 3-4 | Agent-in-world tasks: traversal planning, ability chaining, spatial reasoning in the Vertigo world |

### Task Selection Rationale

Each category maps directly to a source of LoRA training data and an intended downstream use case:

- **Coding** maps to the bulk of the training corpus: Luau source files, service modules, config patterns extracted from the Vertigo codebase. These tasks test whether the adapter has internalized the structural patterns that define a Vertigo module.
- **MCP Tool Calling** maps to MCP interaction traces recorded during development sessions. The adapter should learn which Fleet/Studio DSL tool to invoke for a given intent, and how to structure the invocation.
- **Bugfix** maps to commit diffs and code review data where broken code was identified and corrected. These tasks present deliberately broken Luau and ask the model to diagnose and fix.
- **Architecture** maps to design documents and multi-file patterns in the codebase. These test whether the adapter transfers structural knowledge (boot ordering, service boundaries, remote flows).
- **Embodiment** maps to agent play traces and traversal recordings. These test whether the adapter has internalized the spatial structure of the Vertigo world (zone layout, anchor positions, ability mechanics).

### Difficulty Ratings

Difficulty is rated 2-5 (no trivial difficulty-1 tasks):

- **2**: Single-concept tasks with clear expected output (e.g., write a config module, wrap code in pcall)
- **3**: Multi-concept tasks requiring integration of 2-3 Vertigo patterns (e.g., DataStore + retry + lifecycle)
- **4**: Design tasks requiring architectural reasoning (e.g., boot ordering, request validation flows)
- **5**: Open-ended design tasks with many valid solutions (e.g., config-driven ability system, monolith decomposition)

### Expected Patterns

Each task defines an `expected_patterns` array of regex patterns used for automated scoring. For example, a task asking for a service module might expect:

```json
["--!strict", ":Init\\(\\)", ":Start\\(\\)", "return\\s+\\w+Service"]
```

These patterns are hand-authored proxies for correctness. They check for the presence of key structural elements, not for logical correctness or compilability. The benchmark file is versioned in the repo at `data/eval/benchmark.jsonl`.

---

## 2. Scoring Methodology v0.2

**Scoring method tag:** `pattern_match_v2`

### Dimensions

Each response is scored on up to four dimensions. Not all dimensions apply to every task category.

**Correctness** (weight: 0.4)
Fraction of `expected_patterns` regex matches found in the response (case-insensitive). This is a proxy for semantic correctness. Pattern matching can confirm the presence of expected structural elements but cannot verify that the code is logically correct, compiles, or does what the prompt asked.

**Convention Adherence** (weight: 0.2, coding/bugfix/architecture only)
Checks for Vertigo-specific patterns relevant to the task category:
- `coding`: `--!strict`, `@native`, `:Init()`, `:Start()`
- `bugfix`: `--!strict`
- `architecture`: `--!strict`, `:Init()`, `:Start()`
- `mcp_tool_calling`: not scored (empty convention set)
- `embodiment`: not scored (not applicable)

Only conventions relevant to the category are checked. The convention score is the fraction of applicable conventions found in the response.

**Tool Selection** (weight: 0.2, MCP tasks only)
Whether the model mentions or calls the correct MCP tool by name. For tasks that provide a `tools` array, the score is the fraction of expected tool names that appear anywhere in the response text (including in JSON tool_call blocks). Non-MCP tasks receive `null` for this dimension.

**Code Presence** (weight: 0.2, code-producing tasks only)
Whether the response contains actual Luau code rather than prose-only answers. Checked via the presence of fenced code blocks (` ``` `) or Luau syntactic indicators (`local`, `function`, `return`, `end`, `--!strict`). Categories where code is expected: coding, bugfix, architecture, mcp_tool_calling. Embodiment tasks are not penalized for prose-only answers.

### Weight Normalization

Dimensions that return `null` (not applicable to the task category) are excluded from the weighted average. The applicable weights are renormalized to sum to 1.0. For example:

- A **coding** task uses correctness (0.4) + convention (0.2) + code_presence (0.2) = 0.8 total weight, renormalized to 1.0
- An **MCP** task uses correctness (0.4) + tool_selection (0.2) + code_presence (0.2) = 0.8 total weight, renormalized to 1.0
- An **embodiment** task uses correctness (0.4) only, renormalized to 1.0

### Failure Penalty Multiplier

A `score_failure_penalty` function detects degenerate responses:
- Empty or near-empty responses (< 10 chars): 0.0 multiplier
- Timeout/error sentinel responses: 0.0 multiplier
- Very short responses (< 50 chars): 0.3 multiplier

This penalty is applied as a multiplier on the overall weighted score, ensuring that obviously broken generations are not rewarded by incidental pattern matches.

**Implementation note:** As of `run_benchmark.py` at `SCORING_METHOD=pattern_match_v2`, the `score_task()` function has a known issue where it does not properly handle `None`-valued dimensions in the weighted average. The `score_code_presence` and `score_failure_penalty` functions exist in the codebase but are not yet fully wired into the `score_task` aggregation. Results should be interpreted with this caveat until the scoring code is corrected.

---

## 3. Data Contamination Audit

This section documents a systematic audit of all 30 benchmark tasks for overlap with the LoRA training data. The audit checks whether benchmark prompts, expected outputs, or task structure appear in the training corpus.

### Definitions

- **CLEAN**: No meaningful overlap between the benchmark task and any training example. The task tests novel generation ability.
- **PARTIAL**: Domain knowledge overlap — the model has seen similar code, patterns, or concepts in training, but the specific task formulation and expected output are novel. This is expected and intentional for tasks that test learned codebase patterns.
- **CONTAMINATED**: The benchmark task (prompt + expected output) appears verbatim or near-verbatim in the training data. The task is effectively memorization, not generation.

### Audit Results

**18 CLEAN tasks:**

| Task ID | Category | Rationale |
|---|---|---|
| code_01 | Coding | Novel prompt, no matching training example |
| code_09 | Coding | Novel prompt, no matching training example |
| code_10 | Coding | Novel prompt, no matching training example |
| mcp_01 | MCP Tool Calling | All MCP tasks use novel tool-calling scenarios |
| mcp_02 | MCP Tool Calling | Novel tool selection scenario |
| mcp_03 | MCP Tool Calling | Novel tool selection scenario |
| mcp_04 | MCP Tool Calling | Novel tool selection scenario |
| mcp_05 | MCP Tool Calling | Novel tool selection scenario |
| fix_01 | Bugfix | Novel bug scenario not in training diffs |
| fix_03 | Bugfix | Novel bug scenario |
| fix_04 | Bugfix | Novel bug scenario |
| fix_05 | Bugfix | Novel bug scenario |
| arch_01 | Architecture | Novel design problem |
| arch_03 | Architecture | Novel design problem |
| arch_04 | Architecture | Novel design problem |
| arch_05 | Architecture | Novel design problem |
| play_03 | Embodiment | Novel traversal scenario |
| play_05 | Embodiment | Novel embodiment scenario |

**12 PARTIAL tasks (domain knowledge overlap):**

| Task ID | Category | Overlap Type |
|---|---|---|
| code_02 | Coding | Codebase pattern overlap — tests learned service structure |
| code_03 | Coding | Codebase recall — tests reproduction of known module patterns |
| code_04 | Coding | Codebase pattern overlap |
| code_05 | Coding | Codebase pattern overlap |
| code_06 | Coding | Codebase pattern overlap |
| code_07 | Coding | Codebase recall — tests reproduction of known config patterns |
| code_08 | Coding | Codebase recall — tests reproduction of known lifecycle patterns |
| fix_02 | Bugfix | Bug pattern seen in training diffs |
| arch_02 | Architecture | Design pattern overlap with training docs |
| play_01 | Embodiment | Zone/spatial knowledge from training data |
| play_02 | Embodiment | Ability mechanics from training data |
| play_04 | Embodiment | Agent behavior patterns from training data |

**0 CONTAMINATED tasks.**

### Interpretation

The 12 PARTIAL overlaps are domain knowledge overlaps, not task contamination. The model learns from the Vertigo codebase; the benchmark tests whether it can reproduce and apply those patterns in novel combinations. This is by design — a LoRA adapter trained on codebase patterns *should* perform better on tasks that require those patterns. The distinction is between:

- **Memorization** (contaminated): the model has seen the exact question-answer pair and can regurgitate it.
- **Transfer** (partial): the model has seen related code and must synthesize it into a new context.

Both are valid signals, but they measure different things. When interpreting results, distinguish:

- **Codebase recall tasks** (code_03, code_07, code_08): Primarily test whether the adapter has memorized specific module structures. High scores indicate successful knowledge injection. Low scores on base model confirm the adapter's contribution.
- **Novel synthesis tasks** (all CLEAN tasks): Test whether the adapter enables the model to generalize learned patterns to new problems. These are the stronger signal for adapter quality.

### Recommendation

Report results separately for CLEAN and PARTIAL task subsets when making claims about adapter effectiveness. The CLEAN subset (18 tasks) provides the uncontaminated signal. The PARTIAL subset (12 tasks) provides the "did the adapter learn the codebase" signal. Both are useful; neither should be hidden.

---

## 4. Known Limitations

These are real constraints on what the benchmark can and cannot tell us. Read this section before drawing conclusions from any results.

**Pattern matching is a weak proxy for correctness.** A model could produce syntactically broken code that contains the right keywords and score 100% on correctness. Conversely, correct code using different patterns, variable names, or approaches than expected will score low. The expected_patterns were chosen to be relatively general, but there is no escaping this fundamental limitation of regex-based evaluation.

**No runtime verification.** We do not compile or execute the generated Luau code. A response could contain code that parses but crashes at runtime, has type errors in strict mode, or produces incorrect behavior. The `verify_luau.py` script exists for compilation checking but is not integrated into the benchmark pipeline.

**No semantic equivalence checking.** Two implementations can solve the same problem correctly using completely different patterns, variable names, and code structures. Our scoring will reward only the approach that matches the hand-authored expected patterns. This systematically underscores creative or alternative solutions.

**Small sample size (30 tasks) — wide confidence intervals.** With only 30 tasks (and as few as 5 per category), individual outliers have disproportionate impact on category averages. A single lucky or unlucky response can swing a category score by 20 percentage points. No confidence intervals are computed. Do not treat category-level differences smaller than ~15% as meaningful.

**System prompt primes conventions (inflates convention scores).** The benchmark system prompt explicitly mentions Vertigo conventions (`--!strict`, `Init/Start`, `@native`, `vector.*`, `CollectionService`). This inflates convention adherence scores because the model is told what patterns to use. The convention dimension measures "did the model follow the instructions in the system prompt" more than "did the adapter teach the model Vertigo conventions."

**API vs local inference may differ.** Results from LM Studio API mode and local mlx_lm can differ due to: quantization differences, sampling implementation details, chat template handling (tools are inlined into the system prompt for API mode), and temperature interpretation. Do not compare API and local results as if they are equivalent.

**Embodiment scored on keywords, not physical feasibility.** Embodiment tasks are scored on keyword matching (e.g., does the response mention "grapple", "anchor", "momentum"). There is no way to verify whether a traversal plan is physically feasible in the Vertigo world geometry, whether the described ability chain is mechanically valid, or whether the spatial reasoning is correct.

**Expected patterns may be incomplete.** The patterns were hand-authored by a single person and reflect one valid approach to each task. They may miss legitimate alternative solutions or be too specific to a particular coding style. No inter-rater reliability check has been performed.

**Partial contamination in 12/30 tasks (domain knowledge overlap).** As documented in Section 3, 40% of tasks have domain knowledge overlap with training data. While this is expected and intentional for a codebase-specialized adapter, it means aggregate scores conflate "learned the codebase" with "can generalize to new problems."

**No test for catastrophic forgetting on general capabilities.** The benchmark measures only Vertigo-specific performance. It does not check whether fine-tuning has degraded the base model's general coding ability, reasoning, instruction following, or safety properties. A model could ace this benchmark while becoming worse at everything else.

**Convention scoring is coarse-grained.** The convention dimension checks for the presence of broad patterns (e.g., does `@native` appear anywhere in the response) without verifying that the annotation is applied correctly (e.g., on the right function, not in a comment, not in prose explanation).

---

## 5. Comparison to Industry Benchmarks

| Feature | Our Benchmark v0.1 | SWE-Bench | HumanEval | BigCodeBench | OpenGameEval | ToolBench |
|---|---|---|---|---|---|---|
| Task count | 30 | 2,294 | 164 | 1,140 | 47 | 16,464 |
| Verification | Pattern matching | Test execution | Test execution | Test execution | Studio execution | LLM judge |
| Domain-specific | Yes (Vertigo/Luau) | General Python | General Python | Multi-library | Roblox/Luau | REST APIs |
| Language | Luau | Python | Python | Python | Luau | Python |
| Contamination control | Audit + partial overlap tracking | Real GitHub issues (time-stamped) | Known contaminated | Curated post-cutoff | Manually curated | Time-stamped |
| Statistical rigor | Low (n=30, no CI) | High (n=2294, stratified) | Medium (n=164) | High (n=1140, CI reported) | Medium (n=47) | High (n=16464) |
| Multi-turn | No | Yes (agent loop) | No | Some | No | Yes (agent loop) |
| Tool use | 5 tasks | No | No | No | No | Yes (primary focus) |
| Open-ended tasks | Yes (architecture, design) | No (patch generation) | No (function completion) | No (function completion) | No (script generation) | No (API call sequence) |

### Key Takeaways

1. **Our task count is an order of magnitude below credible benchmarks.** SWE-Bench, BigCodeBench, and ToolBench have hundreds to thousands of tasks. Our 30 tasks cannot support fine-grained claims.
2. **Pattern matching is the weakest verification method in this table.** Every established benchmark uses some form of execution-based verification (test suites, runtime checks, or at minimum LLM-as-judge). We are the only entry relying purely on regex.
3. **Our domain specificity is a strength for our use case** but makes cross-benchmark comparison meaningless. Scores on this benchmark are not comparable to scores on HumanEval or SWE-Bench.
4. **OpenGameEval is the closest comparable.** It targets Roblox/Luau, uses Studio execution for verification, and has a similar task count (47). Our v1.0 roadmap should converge toward their methodology.

---

## 6. Roadmap to Rigorous Evaluation (v1.0)

Based on research into SWE-Bench (ICLR 2024), OpenGameEval (Roblox, 2025), EvalPlus, BigCodeBench (ICLR 2025), ToolBench/ToolLLM, LiveCodeBench, and AgentBench.

### Tier 1: Luau Compilation Gate (immediate, no Studio needed)

Run `luau-compile` on all generated code blocks:
- Binary pass/fail — does the code parse without syntax errors?
- Catches syntax errors, type annotation issues, malformed expressions
- Zero infrastructure cost — `luau-compile` is already available locally
- Report compilation rate alongside pattern-match scores
- This single addition would move us from "keyword presence" to "syntactically valid keyword presence," which is a meaningful improvement

### Tier 2: Standalone Unit Tests (next sprint)

- Write test cases for each benchmark task (minimum 5 per task, target 15+)
- Execute via `luau` CLI with stub globals (mock `game`, `workspace`, standard Roblox APIs)
- Measures functional correctness, not just pattern presence
- Follow the EvalPlus approach: augment hand-written tests with LLM-generated edge cases to improve coverage
- Requires building a Roblox API stub layer — nontrivial but bounded effort
- pass@k metric with k=1,5 to measure reliability

### Tier 3: Studio Execution Tests (after 128GB machine)

- Use Roblox Studio MCP tools (`run_code`, `run_script_in_play_mode`, `get_console_output`)
- Verify runtime behavior: instance creation, physics responses, event firing, DataStore operations
- Adapt the OpenGameEval framework for Vertigo-specific tasks
- pass@k with k=1,3,5
- Requires a running Studio instance and careful sandboxing to prevent side effects between tests
- Studio MCP is already operational — the infrastructure exists, the test harness does not

### Tier 4: Agent/Embodiment Verification (future)

- Runtime bridge + `capture_gameplay.py` for behavioral assertions
- Did the agent actually reach the target zone?
- Did ability chains execute in the correct sequence and timing?
- Were physics constraints respected (momentum, gravity, cooldowns)?
- WebArena/AgentBench-style goal completion verification
- This is the gold standard for embodiment evaluation but requires significant infrastructure

### Task Expansion Target: 150-200 Tasks

| Category | Count | Focus |
|---|---|---|
| Luau syntax/semantics | 30 | Language features, strict mode, type annotations, NCG patterns |
| Roblox API knowledge | 30 | Services, instances, events, data types |
| Game mechanics | 25 | Physics, collision, tweening, raycasting, spatial queries |
| NPC/Agent behavior | 20 | State machines, patrol, interaction, memory, decision-making |
| Client-server patterns | 20 | Remotes, replication, validation, anti-cheat, DataStore |
| Tool-calling (MCP) | 25 | Fleet DSL, Studio MCP, multi-tool sequences, error handling |
| Multi-step reasoning/debugging | 20 | Diagnosis chains, root cause analysis, performance profiling |
| Choreography/Animation | 10 | Ability VFX, camera sequences, agent coordination |

Expansion priorities:
1. More CLEAN tasks (reduce partial overlap ratio below 20%)
2. Difficulty-1 baseline tasks (sanity checks the base model should pass)
3. Adversarial tasks (common Luau pitfalls, Roblox API gotchas)
4. Regression tasks (does fine-tuning break general Luau ability?)

---

## 7. Reproducibility

### Result Format

All results are saved as JSONL files with one JSON object per line per task. Each result record includes:

```json
{
  "id": "code_01",
  "category": "coding",
  "difficulty": 2,
  "prompt": "...",
  "response": "...(full model output)...",
  "scores": {
    "correctness": 0.75,
    "convention_score": 0.5,
    "tool_selection": null,
    "code_presence": 1.0,
    "failure_penalty": 1.0,
    "overall": 0.525
  },
  "adapter": "adapters/v0.3-2b-oss",
  "model": "mlx-community/Qwen3.5-35B-A3B-4bit+v0.3-2b-oss",
  "benchmark_version": "0.2.1",
  "scoring_method": "pattern_match_v2",
  "timestamp": "2026-03-14T13:35:15.123456"
}
```

### Versioning

- Benchmark file: `data/eval/benchmark.jsonl` (versioned in repo)
- Scoring code: `scripts/run_benchmark.py` with `BENCHMARK_VERSION` and `SCORING_METHOD` constants
- Result files include `benchmark_version` and `scoring_method` tags for traceability
- Output files are never overwritten — a timestamp suffix is appended if the file already exists

### Inference Parameters

- Temperature: 0.1 (deterministic-ish — MLX and LM Studio have implementation-dependent sampling)
- Max tokens: 2048 per response
- Generation timeout: 60 seconds (local) / 120 seconds (API)
- System prompt: identical across all tasks (the Vertigo coding assistant prompt)

### Running the Benchmark

```bash
# Local model + adapter (requires mlx_lm installed)
uv run python scripts/run_benchmark.py --adapter adapters/v0.3-2b-oss

# Via API (LM Studio or any OpenAI-compatible endpoint)
uv run python scripts/run_benchmark.py --api --api-model qwen3.5-35b-a3b

# Filter to specific categories
uv run python scripts/run_benchmark.py --api --categories coding bugfix

# Compare results from multiple runs
uv run python scripts/run_benchmark.py --compare data/eval/results_a.jsonl data/eval/results_b.jsonl
```

API mode auto-detects the model name from the `/v1/models` endpoint if `--api-model` is not specified. For API mode, tool definitions are inlined into the system prompt rather than passed via the `tools` API field, since many local API servers do not support it reliably.

---

## 8. Baseline Results

### Vertigo Pattern-Matching Benchmark (v0.2, scoring: `pattern_match_v2`)

| Model | Params Active | Adapter | Coding | Bugfix | Architecture | MCP Tools | Embodiment | **Overall** |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3.5-27B dense | 27B | none (base) | 80.6% | 96.7% | 79.1% | 100% | 95.0% | **88.7%** |
| Qwen3.5-35B-A3B | 3B active | none (base) | 79.2% | 75.3% | 66.5% | 96.7% | 77.5% | **79.1%** |
| Qwen3.5-2B | 2B | none (base) | 45.0% | 81.7% | 54.2% | 70.0% | 95.0% | **65.1%** |
| Qwen3.5-9B | 9B | none (base) | 25.6% | 76.7% | 61.6% | 96.7% | 95.0% | **63.5%** |
| Qwen3.5-2B | 2B | **v0.3 LoRA** | 50.3% | 49.0% | 39.5% | 45.0% | 75.0% | **51.5%** |

*Collected 2026-03-15. Inference via LM Studio (API mode) and mlx_lm (local mode). Temperature 0.1, max_tokens 2048.*

### OpenGameEval Dry-Run (code generation quality, no Studio execution)

| Model | Params Active | Pass@1 (dry) | Nearest Published |
|---|:---:|:---:|---|
| Qwen3.5-35B-A3B base | 3B active | **42.6%** | Claude Opus 4.5 (44.5%) |

*Dry-run scores code generation quality only (does response contain valid-looking Luau that references appropriate Roblox APIs). Studio execution verification pending.*

### Key Findings

1. **Catastrophic forgetting detected.** The v0.3 LoRA adapter (51.5%) underperforms the 2B base model (65.1%) on this benchmark. Fine-tuning improved coding (+5.3pp) but degraded bugfix (-32.7pp), MCP tools (-25.0pp), and embodiment (-20.0pp). This is a known risk of LoRA training on domain-specific data without general instruction mixing.

2. **35B-A3B is a strong Roblox base.** At 42.6% on OpenGameEval (dry-run), the 3B-active MoE model competes with Claude Opus 4.5 (44.5%) on Roblox code generation — with 1000x fewer active parameters.

3. **9B underperforms 2B on coding.** The 9B base scores 25.6% on coding vs 45.0% for the 2B. This may reflect the 9B's tendency to overthink (Qwen3.5-9B enables `<think>` by default which can crowd out code).

4. **Embodiment scores are inflated.** All base models score 95% on embodiment tasks, suggesting the patterns are too easy to match. Embodiment evaluation needs behavioral verification (Tier 4).

### Implications for 35B Training (March 19)

- Mix general instruction data to prevent catastrophic forgetting
- Evaluate at every checkpoint (save_every=200) to catch regression early
- Maintain the 2B base as a regression anchor
- Target: beat 79.1% overall (35B base) without regressing below 65.1% (2B base) on any category

Raw result files in `data/eval/`:
- `results_2b_base.jsonl`, `results_9b_base.jsonl`, `results_27b_base.jsonl`, `results_35b_base.jsonl`
- `results_2b_v03_lora.jsonl` (renamed from `results_20260314_134408.jsonl`)
- `opengameeval_35b_dryrun.jsonl`
- `results_27b_base.jsonl`
- `results_35b_base.jsonl`
- Various timestamped runs

These results were generated with the current scoring aggregation (which has the `None` dimension handling issue noted in Section 2) and should be re-run after the scoring code is corrected.

---

## 9. References

### Primary Benchmarks Studied

- **SWE-bench**: Jimenez, C.E., et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" ICLR 2024. [swe-bench.com](https://www.swe-bench.com/) — Gold standard for repository-level code generation evaluation using real GitHub issues with test-based verification.

- **OpenGameEval**: Roblox, 2025. Benchmark for evaluating LLMs on Roblox/Luau game development tasks with Studio execution verification. 47 tasks across script generation, debugging, and game mechanics. Most directly comparable to our domain.

- **BigCodeBench**: Zhuo, T.Y., et al. "BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions." ICLR 2025. [bigcode-bench.github.io](https://bigcode-bench.github.io/) — 1,140 tasks testing multi-library Python code generation with execution-based verification.

- **EvalPlus**: Liu, J., et al. "Is Your Code Generated by ChatGPT Really Correct?" NeurIPS 2023. [evalplus.github.io](https://evalplus.github.io/) — Framework for augmenting code benchmarks with LLM-generated test cases. Demonstrated that HumanEval+ (with augmented tests) drops pass rates by 10-20% vs vanilla HumanEval.

- **LiveCodeBench**: Jain, N., et al. "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code." 2024. Continuously updated from competitive programming platforms to avoid contamination. Time-stamped tasks enable pre/post-training-cutoff analysis.

- **HumanEval**: Chen, M., et al. "Evaluating Large Language Models Trained on Code." 2021. The original code generation benchmark. 164 Python function completion tasks. Known to be contaminated in most modern LLM training sets.

### Tool Use and Agent Benchmarks

- **ToolBench / ToolLLM**: Qin, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs." 2023. 16,464 tasks across real REST APIs with multi-step tool use chains. Uses LLM-as-judge (ToolEval) for verification.

- **TOUCAN**: Li, J., et al. "TOUCAN: Token-Efficient Tool Use with Calibrated Actions." 2024. Focuses on token-efficient tool invocation patterns.

- **ToolACE**: Liu, H., et al. "ToolACE: Winning the Points of LLM Function Calling." 2024. Automated pipeline for generating diverse, high-quality tool-calling training data.

- **AgentBench**: Liu, X., et al. "AgentBench: Evaluating LLMs as Agents." 2023. Multi-environment agent evaluation across OS, database, web, and game environments.

- **WebArena**: Zhou, S., et al. "WebArena: A Realistic Web Environment for Building Autonomous Agents." 2023. Functional end-to-end web agent evaluation with real websites and goal completion verification.

### Data Quality and Selection

- **DEITA**: Liu, M., et al. "What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning." 2024. Data quality scoring and selection methods for instruction tuning. Relevant to our training data curation pipeline.
