# 9B Experiment Plan — Abraham (128GB)

## Hypothesis

The 4B at rank 8 / 8 layers / 2048 seq reached 82.9% on our benchmark. It's constrained by:
1. Low rank (8) limits adapter expressiveness
2. Only 8 of 28 layers adapted — most of the model is frozen
3. 2048 seq truncates the longest examples

The 9B with 128GB headroom can run at full capacity. The question: **does 2.25x more model capacity produce proportionally better results, or does the 4B curated approach plateau?**

## Planned Experiments (in order)

### Experiment 1: 4B at Full Capacity (baseline)

Before training the 9B, re-train the 4B with the constraints removed. This isolates whether the 4B's ceiling is capacity-limited or data-limited.

```bash
./scripts/auto_train.sh 4b-128gb v0.7-4b-full
```

Config: rank 32, 16 layers, 4096 seq, batch 2, grad_accum 4 (effective batch 8), cosine LR decay.

**Expected outcome:** If this scores significantly above 82.9%, the 4B was capacity-limited and the 9B should do even better. If it scores ~83%, the ceiling is data-limited and we need more/better data before scaling the model.

### Experiment 2: 9B Curated (same data, bigger model)

Same curated dataset, 9B model.

```bash
./scripts/auto_train.sh 9b v0.1-9b-curated
```

Config: rank 16, 16 layers, 4096 seq, batch 1, grad_accum 8.

**Key question:** Does the 9B resist catastrophic forgetting like the 4B, or does it need even more careful curation?

**Benchmark the 9B base first** via LM Studio to establish the true baseline.

### Experiment 3: 9B with Full Uncurated Data

If the 9B handles the curated set well, try the full uncurated `--mix-strategy three-dataset` data. The 9B may have enough capacity to absorb the prose-heavy trajectory/tool data that degraded the 4B.

```bash
# First rebuild with full data
uv run python scripts/merge_dataset.py --mix-strategy three-dataset
./scripts/auto_train.sh 9b v0.2-9b-full
```

### Experiment 4: 4B 8-bit Training

Train the 4B at 8-bit quantization instead of 4-bit. Higher precision gradients should produce a more accurate adapter at the cost of 2x memory (~18GB model vs ~9GB).

```bash
./scripts/auto_train.sh 4b-8bit-128gb v0.1-4b-8bit
```

## Data Strategy for 9B

### What might change

The 4B needed aggressive curation (84.8% retention) because low-code-density examples diluted signal in a capacity-constrained model. The 9B has 2.25x capacity and may:

1. **Handle trajectory data** — the studio_trajectories.jsonl and tool_calling_sft.jsonl that degraded the 4B might work in a model with more parameters to partition tool-calling vs code-generation knowledge.

2. **Benefit from longer sequences** — at 4096 seq (vs 2048), the 9B can see full multi-file examples and composition tasks without truncation. This may unlock better multi-module reasoning.

3. **Need less oversampling** — the curated pipeline oversamples minority categories. With more capacity, the 9B may learn minority patterns from fewer repetitions.

### What probably stays the same

1. **Code density requirement** — examples without code are still noise for a code model, regardless of model size.
2. **Quality > quantity** — the fundamental finding that adding mediocre data hurts more than it helps is unlikely to change at 9B.
3. **Composition tasks** — multi-concept examples teach pattern combination, which is model-size-independent.

### Recommended approach

Start with the curated dataset (proven recipe), then incrementally add data categories and measure impact:

```
Experiment 2: curated only (baseline)
Experiment 3: curated + trajectory data (add tool-calling signal)
Experiment 4: curated + trajectory + expanded creator docs (add more SFT)
Experiment 5: full uncurated (stress test)
```

Track per-category scores at each step. If any category drops >2pp below base, stop adding data in that direction.

## Hyperparameter Strategy

### Transfer from 4B

The 4B's best config used:
- LR 2e-6 (lower than 2B's 5e-6 — larger models need gentler LR)
- Rank 8 (constrained by memory)
- No cosine schedule (flat LR worked)
- 600 iters

For 9B, start with:
- LR 1e-6 (half again — 9B is 2.25x larger)
- Rank 16 (2x what 4B used, still conservative for 9B)
- Cosine decay with 100-step warmup (safer for larger model)
- 600 iters (same — the dataset size hasn't changed)

### If training diverges (NaN)

Reduce in this order:
1. LR → 5e-7
2. Rank → 8
3. Add gradient clipping (1.0)
4. Reduce num_layers → 8

### If training converges too slowly

Increase in this order:
1. LR → 2e-6
2. Rank → 32
3. Add warmup (100 iters)
4. Increase iters → 1000

## Evaluation Plan

### Before training

1. Benchmark 9B base via LM Studio (get true baseline)
2. Run OpenGameEval dry-run on 9B base (compare to 4B's 72.3%)

### After each training run

1. `auto_train.sh` handles benchmark + promote automatically
2. Run OpenGameEval dry-run on each adapter
3. Compare to 4B v0.5 (82.9% benchmark, 83.0% OGE)

### Success criteria

- **Minimum:** Beat 4B v0.5 (82.9%) on our benchmark
- **Good:** Beat 4B v0.5 on OpenGameEval dry-run (83.0%)
- **Great:** No category regression below 4B base (76.9%)
- **Exceptional:** OpenGameEval execution-verified pass@1 competitive with Claude Haiku (35.7%)

## Timeline

Day 1 (March 19):
- Bootstrap abraham (`./scripts/bootstrap_abraham.sh`)
- Benchmark 9B base
- Train 4B at full capacity (Experiment 1)
- Train 9B curated (Experiment 2)

Day 2:
- Evaluate results
- If 9B curated works: try full uncurated (Experiment 3)
- If 9B curated forgetting: debug data mix
- Try 4B 8-bit (Experiment 4)

Day 3+:
- Iterate STaR loop on best adapter
- OpenGameEval with Studio execution (if Studio is running)
- Suit SDK comparison (`--with-suit` mode)
