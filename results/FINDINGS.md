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
| Gate + 100x, 8-layer | — | running | |

### Currently Running Experiments

**Testing architectural fixes for 4-pair retrieval** (all: gate + 100x, 4 pairs, decay 0.999):

1. **Tied Q=K** (`--tie-qk`): Share W_query = W_key so same key token produces matching vectors for both storage and retrieval
2. **Delta rule** (`--delta-rule`): `M += k⊗(v - Mk)` — subtract old association before writing to reduce interference
3. **Hard gate** (`--gate-bias -5`): Gate init at sigmoid(-5)≈0.007 instead of sigmoid(-2)≈0.12 for more aggressive filler suppression
4. **Tied Q=K + delta**: Both fixes combined
5. **All three**: Kitchen sink

Also running:
- Gate + 100x, 16 pairs (capacity test)
- Gate + 100x, 4 pairs, 8-layer model (depth test)

### High Variance Problem

Same config gives wildly different results depending on random init:
- 2 pairs uniform: 1.1% vs 47.7% (different loss modes but same arch)
- 8 pairs key-fix: 44.6% vs original 5.4%
- Original predictive: 100% vs 1.2%

The optimization landscape has multiple basins. Most don't involve M. The model easily falls into "ignore M" local minima.

### Open Questions

1. ~~Does recall-weighted loss enable retrieval?~~ **Yes, for 1 pair.**
2. ~~Does M scale to multiple pairs?~~ **Not yet — fails at 4+ pairs.**
3. Can tied Q=K, delta rule, or hard gate solve 4-pair retrieval?
4. Is the problem fundamentally optimization (need more seeds/training) or architectural?
5. Does predictive M offer advantages on anticipation/planning tasks?
6. Multi-timescale memory for agent planning (long-term vision)

---
*Last updated: 2026-03-12*
