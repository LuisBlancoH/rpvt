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

**Conclusion**: No memory mode achieves reliable retrieval. Neither future values nor subtraction nor their combination works. The recall signal is too diluted in the loss (~1/1344 of total gradient).

### Controlled Results (all decay 0.999)

| Config | No Memory | Regular M | Predictive M |
|--------|-----------|-----------|-------------|
| 64v, gap 2-8 | 2.0% | 1.6% | 0.8% |
| 128v, gap 5-20 | — | 0.2% | 1.2% |

All at chance level. M provides no benefit over no-memory baseline.

### Loss Dilution Problem (root cause)

The recall token is 1 out of N tokens in the sequence. With uniform loss weighting, the recall gradient is ~1/N of total. For a 1344-token sequence (gap 20), the recall signal is 0.07% of the gradient.

The model spends all capacity learning to predict random filler tokens (fixed entropy ~5.5) and gets negligible gradient from the one recall token that matters.

Added `--recall-loss-weight` flag to amplify recall signal. **Not yet tested.**

### Reproducibility Issues

- No-memory baseline: 24% in one run, 2% in another (same config, different random init)
- Predictive M: 100% in one run, 1.2% in rerun (same config)

Single runs are unreliable. Need multiple seeds per config.

### Open Questions

1. Does recall-weighted loss (e.g. 100x) enable any mode to learn retrieval?
2. If weighted loss works, does it work for regular M too, or only predictive?
3. Should we compute loss ONLY on the recall position?
4. Need 3+ seeds per config for reliable results

---
*Last updated: 2026-03-12*
