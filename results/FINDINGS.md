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
| **Slot** (content-based addressing, 32 slots) | 16.7% | Better than chance but not great |

Both alternative architectures hit the same optimization problem — the model doesn't learn to use memory regardless of retrieval mechanism.

### Optimization Tricks Tested

| Approach | 4-pair Recall | Notes |
|---|---|---|
| Recall bottleneck (zero hidden states) | 0.0% | Too harsh — kills everything |
| Memory supervision (10x extra CE) | 0.9% | Extra loss didn't help |
| Both combined | 0.0% | Same as bottleneck |

### Key Discovery 10: Two-Phase Training Solves Gate Learning

**Two-phase**: Phase 1 = freeze transformer, train only memory. Phase 2 = unfreeze all.

| Approach | 4-pair Recall | Notes |
|---|---|---|
| Two-phase (fast weight M) | 5.4% | **Correct gate pattern learned!** |
| **Two-phase (Hopfield)** | **18.1%** | **Best non-fluke result** |

**Gate values from two-phase fast weight model:**

| Chunk type | Gate value |
|---|---|
| **STORE** | **0.55** |
| **FILLER** | **0.000** |
| **RECALL** | **0.16** |

**Breakthrough:** Store/Filler ratio = 1.6 million. The gate learned exactly what we hypothesized — write strongly on STORE chunks, completely suppress fillers. This is the first time we've seen the correct gate pattern. Two-phase training forces M to learn because the transformer can't adapt to compensate.

**Why two-phase works:** In normal training, the transformer has ~2M params that can minimize loss without M. M starts at zero (W_out zero-init) and contributes nothing initially. The transformer learns to solve the task without M, and once that solution is found, there's no gradient to start using M. Two-phase prevents this by freezing the transformer, so M is the only thing that can improve the loss.

**Why Hopfield outperformed M-matrix (18.1% vs 5.4%):** Once the gate learns correctly (which two-phase ensures), Hopfield's softmax retrieval is strictly better at selecting the right stored pair. The gate solves "what to write," Hopfield solves "how to read."

### The Core Problem: Optimization, Not Architecture

The "ignore M" local minimum is much easier to find than the "use M" basin. Two-phase training is the first approach that reliably forces the optimizer into the correct basin by eliminating the "ignore M" option.

### Currently Running Experiments

**Three-phase training** (pretrain transformer → freeze + train memory → unfreeze all):

1. **Three-phase fast weight** (33/33/33 split, 20 epochs) — M learns from useful representations instead of random weights
2. **Three-phase Hopfield** (33/33/33, 20 epochs) — best architecture + best optimization
3. **Three-phase Hopfield long-p1** (25/50/25, 20 epochs) — more time for memory to learn
4. **Two-phase Hopfield 40 epochs** (50/50) — test if 18.1% improves with more training

### Open Questions

1. ~~Does recall-weighted loss enable retrieval?~~ **Yes, for 1 pair.**
2. ~~Does M scale to multiple pairs?~~ **Not with standard training. Two-phase + Hopfield gets 18.1%.**
3. ~~Can architecture changes solve it?~~ **No — optimization is the bottleneck.**
4. ~~Can two-phase training solve the optimization problem?~~ **Partially — correct gate pattern learned, 18.1% with Hopfield.**
5. Can three-phase training improve on two-phase? (transformer pretrained → better M features)
6. Does more training (40 epochs) push two-phase Hopfield higher?
7. Once solved for 4 pairs: does it scale to 8, 16?
8. Predictive Hopfield for planning (store transitions, chain retrievals)
9. Multi-timescale memory for agent planning (long-term vision)

---
*Last updated: 2026-03-13*
