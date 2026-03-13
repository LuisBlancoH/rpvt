# Experiment Findings

## v2.6: Synthetic Recall Task

**Setup**: Small transformer (4-layer, 256-dim) trained from scratch on synthetic store/recall task. Store a key-value pair in chunk 0, recall after K filler chunks. Whole-doc training (gradient flows through M within single forward call).

### Key Discovery 1: Decay Rate Confound

Early results appeared to show 100% recall with memory. This was **memorization in model weights**, not M-based retrieval.

- With decay 0.99: M decays to ~0.006 by recall time (effectively dead). Model memorized key→value mappings in weights.
- With decay 0.999: M retains ~28% signal at gap 8. M is alive but adds noise when untrained.

**Lesson**: A dead M (fast decay) produces fake-good results because it gets out of the way. Always use decay where M actually persists, or results are meaningless.

### Key Discovery 2: Pair Space Determines Memorization vs Retrieval

The model will memorize key→value mappings in weights if the pair space is small enough:

| Config | Possible pairs | Train docs | Train/eval overlap | Can memorize? |
|--------|---------------|------------|-------------------|---------------|
| 32 keys × 64 values | 2,048 | 2,000 | **63%** | Yes — most pairs seen |
| 64 keys × 128 values | 8,192 | 2,000 | **22%** | No — most eval pairs novel |
| 64 keys × 256 values | 16,384 | 2,000 | ~11% | No |

### Key Discovery 3: The 100% Predictive Result Was a Fluke

The predictive M 100% result (128v, gap 5-20, decay 0.999) **did not reproduce**. Rerun with identical config gave 1.2% (chance). The original run was a lucky seed.

### Ablation Results (all decay 0.999, 128v, gap 5-20)

| Ablation | Values stored | Output | Recall |
|----------|-------------|--------|--------|
| A: Regular (uniform) | current chunk | additive | 0% |
| B: Future-only | next chunk | additive | 0.2% |
| C: Subtract-only | current chunk | delta (pred - current) | 0.4% |
| D: Predictive (both) | next chunk | delta | 1.2% |
| D (original, non-reproducible) | next chunk | delta | ~~100%~~ |

### Loss Dilution Problem (root cause)

The recall token is 1 out of N tokens in the sequence. With uniform loss weighting, the recall gradient is ~1/N of total. For a 1344-token sequence (gap 20), the recall signal is 0.07% of the gradient.

Added `--recall-loss-weight` flag to amplify recall signal.

### Key Discovery 4: Loss Weighting Solves Single-Pair Retrieval

**Results** (all: decay 0.999, 64 keys, 128 values, gap 5-20, 20 epochs, whole-doc):

| Experiment | Fix | Recall |
|---|---|---|
| No memory + 100x weight | Loss weight (control) | 0.4% |
| **Regular M + 100x weight** | **Loss weight** | **100%** |
| **Predictive M + 100x weight** | **Loss weight** | **100%** |
| Regular M + W_out 0.01 | W_out init | 0.2% |
| Predictive M + W_out 0.01 | W_out init | 0.8% |

**Conclusions:**
1. **Loss weighting is necessary and sufficient** for single-pair retrieval.
2. **W_out init alone does nothing.** Sufficient gradient overcomes zero init.
3. **Both regular M and predictive M achieve 100%.** Architecture not the bottleneck.

### Key Discovery 5: Multi-Pair Capacity Problem

Testing M with multiple store/recall pairs per document (all: decay 0.999, 64 keys, 128 values, gap 5-10, 20 epochs, whole-doc, 100x recall weight):

**Uniform write mode + 100x weight:**

| Pairs | Recall | Behavior |
|---|---|---|
| 1 | **100%** | Perfect retrieval |
| 2 | 1.1% | Chance (bad seed) |
| 4 | 24.9% | Gets 1/4 right consistently |
| 8 | 5.4% | ~0.4/8 right |
| 16 | 13.3% | ~2/16 right |

**Key finding**: Model learns a single fixed retrieval pattern, not key-dependent retrieval. It always retrieves the same pair regardless of query key.

**Debug analysis** (from recall-only loss 2-pair run, 47.7%): Pair 0 always correct, pair 1 always gets pair 0's value. The model doesn't use the query key to select which pair to retrieve.

### Key Discovery 6: Write Gate Enables 2-Pair Retrieval

**Gate write mode** (learned sigmoid gate on write strength) + 100x weight:

| Pairs | Uniform + 100x | Gate + 100x |
|---|---|---|
| 1 | 100% | — |
| 2 | 1.1% | **94.4%** |
| 4 | 24.9% | 25.4% |
| 8 | 5.4% | 3.4% |

**The gate enables key-dependent retrieval for 2 pairs** (debug shows both pairs correct with different predictions). But it doesn't help for 4+ pairs.

**Why the gate helps**: It learns to write strongly on STORE chunks and weakly on fillers, reducing noise in M. But it can't solve the W_key/W_query coordination problem for 4+ keys.

**Why it fails at 4+ pairs**: W_key and W_query are separate projections. The model must learn them to map the same key token (in STORE vs RECALL context) to matching vectors. This coordination is easy for 2 keys but hard for 4+.

### Approaches Tried for Multi-Pair (4 pairs)

| Approach | 2 pairs | 4 pairs | Notes |
|---|---|---|---|
| Uniform + 100x weight | 1.1% | 24.9% | Single retrieval pattern |
| Recall-only loss | 47.7% | — | Gets 1/2 right (pair 0 always) |
| Key at position -2 | 1.4% | 0.7% | Key visibility not the issue |
| Curriculum (1→N pairs) | 11.0% | 13.7% | Disrupts learned retrieval |
| Gate + recall-only | 47.1% | — | Same as recall-only |
| **Gate + 100x** | **94.4%** | 25.4% | Works for 2 but not 4 |
| Gate + 100x, 512d model | — | 0.8% | Larger model worse (optimization) |
| Gate + 100x, 8-layer | — | 22.1% | Depth doesn't help |
| Gate + 100x, 16 pairs | — | 5.2% | Chance level |

### Key Discovery 7: Hard Gate Result Was a Lucky Seed

**Architectural fix results** (all: gate + 100x, 4 pairs, decay 0.999, 20 epochs):

| Approach | 4-pair Recall | Notes |
|---|---|---|
| Gate (bias=-2) + 100x | 25.4% | Baseline (1/4 pattern) |
| Hard gate (bias=-5) + 100x, seed=42 | **55.6%** | Lucky seed |
| All three (tied+delta+hard) | 13.7% | Fixes interfere with each other |
| Tied Q=K | 0.9% | Catastrophic |
| Delta rule | 0.6% | Catastrophic |
| Tied Q=K + delta | 0.65% | Catastrophic |

**Gate bias sweep** (all: gate + 100x, 4 pairs, seed=42):

| Bias | sigmoid(bias) | 4-pair Recall |
|---|---|---|
| -2 | 0.12 | 25.4% |
| -3 | 0.047 | 0.7% |
| -4 | 0.018 | 0.6% |
| **-5** | **0.007** | **55.6%** |
| -8 | 0.0003 | 0.6% |
| -10 | 0.00005 | 0.8% |

No consistent pattern — bias=-5 worked but -3, -4, -8, -10 all failed.

### Key Discovery 8: The 55.6% Result Was a Fluke (Confirmed by Multi-Seed)

**Multi-seed test** (gate + 100x, bias=-5, 4 pairs):

| Seed | 4-pair Recall |
|---|---|
| 42 (original) | **55.6%** |
| 1 | 0.6% |
| 2 | 0.9% |
| 3 | 0.5% |
| 4 | 0.7% |

**The 55.6% was a lucky seed.** 4/4 new seeds failed at chance level. The optimization landscape is the fundamental bottleneck — the "use M" basin is extremely rare regardless of gate bias.

### Key Discovery 9: Gate Analysis — Failed Models Learn Inverted Gate

Gate values from a trained model that failed (1.3% recall, bias=-5):

| Chunk type | Gate value |
|---|---|
| STORE | 0.019 (low — suppresses writes!) |
| FILLER | 0.999 (high — writes everything!) |
| RECALL | 0.016 (low) |

**Opposite of our hypothesis.** Failed models learn to suppress STORE writes and amplify filler writes. The gate only learns the correct pattern (store≫filler) when the model successfully finds the "use M" basin — which almost never happens.

### Alternative Architectures Tested

| Architecture | 4-pair Recall | Notes |
|---|---|---|
| **Hopfield** (softmax retrieval, 32 slots) | 1.1% | Same optimization problem |
| **Slot** (content-based addressing, 32 slots) | running | |

**Hopfield failed** — same as M-matrix. The problem is not retrieval quality (linear vs softmax) but that the optimizer never discovers how to use memory at all. Softmax retrieval is theoretically better (exponential capacity) but can't help if the model doesn't learn to write meaningful content.

### The Core Problem: Optimization, Not Architecture

Every architecture and fix we've tried fails for the same reason: the "ignore M" local minimum is much easier to find than the "use M" basin. The model can always minimize loss by:
1. Predicting filler tokens from context (easy, dominates the loss)
2. Ignoring recall tokens (small fraction of total loss, even with 100x weight)

**This is not an architecture problem — it's an optimization problem.** Changing retrieval (Hopfield), write rules (delta), key matching (tied Q=K), or gate bias doesn't help because the optimizer never even tries to use M.

### Currently Running Experiments

**New approach: force M usage through training tricks:**

1. **Recall bottleneck** — Zero out transformer hidden states at recall positions. M becomes the ONLY path for recall information. Forces the optimizer to use M.
2. **Memory supervision** — Add 10x extra CE loss directly on recall positions. Stronger gradient signal for M.
3. **Both combined** — Bottleneck + supervision.
4. **Slot memory** (32 slots) — Still running from earlier batch.

### Open Questions

1. ~~Does recall-weighted loss enable retrieval?~~ **Yes, for 1 pair.**
2. ~~Does M scale to multiple pairs?~~ **No — 55.6% was a fluke. Fails consistently at 4+ pairs.**
3. ~~Can tied Q=K, delta rule, or hard gate solve 4-pair retrieval?~~ **No. All fail or are seed-dependent.**
4. ~~Can Hopfield (softmax retrieval) solve it?~~ **No. Same optimization problem.**
5. Can information bottleneck (forcing M usage) solve the optimization problem?
6. Can direct memory supervision provide enough gradient signal?
7. If bottleneck works: does it generalize when the bottleneck is removed?
8. Does predictive M offer advantages on anticipation/planning tasks?
9. Multi-timescale memory for agent planning (long-term vision)

---
*Last updated: 2026-03-13*
