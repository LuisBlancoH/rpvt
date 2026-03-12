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

The 128v predictive M result (100%) is **confirmed real retrieval** — only 22% of eval pairs appeared in training. The model generalized to 78% unseen pairs.

### Controlled Results (all decay 0.999)

| Config | No Memory | Regular M (uniform) | Predictive M (nudge) |
|--------|-----------|---------------------|----------------------|
| 64v, gap 2-8 (2048 pairs) | 2.0% | 1.6% | 0.8% |
| 128v, gap 5-20 (8192 pairs) | — | 0.2% | **100%** |

At 64v: all modes fail because pair space is small enough to attempt memorization, but decay 0.999 M interferes with it. The task becomes a conflict between two learning strategies.

At 128v: memorization is impossible (too many pairs). Regular M fails. **Predictive M succeeds** — the only configuration that achieves real retrieval.

### Predictive M (nudge) vs Regular M

Predictive mode: `output = W_out(M @ query) - W_nudge(x)` (delta toward predicted state)
Regular mode: `output = W_out(M @ query)` (additive)

Predictive M also stores future states: `value = W_value(next_chunk_hidden)`.

On the medium task (128 values, gap 5-20, decay 0.999):
- Regular M: 0.2% (chance)
- Predictive M: **100%** (verified not memorization — 78% of eval pairs unseen in training)

### Why Predictive Works (hypotheses, not yet tested)

1. **Richer stored values**: future states contain attention-processed info from the store chunk
2. **Stronger gradient signal**: the subtraction creates a direct comparison forcing M to be useful
3. **Extra parameters**: W_nudge (256×256 per layer = 262K extra) may provide better learning landscape
4. **Architecture prevents ignoring M**: delta output can't be trivially zeroed like additive output

### Invalidated Results (decay 0.99, M was dead)

| Config | Regular M | Predictive M |
|--------|-----------|-------------|
| 64v, gap 2-8 | 100% (memorized) | 100% (memorized) |
| 256v, gap 2-8 | 0.2% (M dead, can't memorize) | 0.2% (same) |

### Loss Dilution Problem

The recall token is 1 out of N tokens in the sequence. With uniform loss weighting, the recall gradient is ~1/N of total. Added `--recall-loss-weight` flag to amplify recall signal. Not yet tested in clean sweep.

### Open Questions

1. Why does predictive mode work when regular doesn't? (need ablations: subtraction only, future values only, extra params only)
2. Does recall-weighted loss fix regular M?
3. Does predictive M hold up at longer gaps (10-40)?
4. Reproducibility: need multiple seeds per config (no-memory baseline varied 2% to 24% across runs)

---
*Last updated: 2026-03-12*
