# Experiment Findings

## v2.6: Synthetic Recall Task

**Setup**: Small transformer (4-layer, 256-dim) trained from scratch on synthetic store/recall task. Store a key-value pair in chunk 0, recall after K filler chunks. Whole-doc training (gradient flows through M within single forward call).

### Key Discovery: Decay Rate Confound

Early results appeared to show 100% recall with memory. This was **memorization in model weights**, not M-based retrieval.

- With decay 0.99: M decays to ~0.006 by recall time (effectively dead). Model memorized key→value mappings in weights (32 keys × 64 values = 2048 pairs, 2000 training docs = enough to memorize).
- With decay 0.999: M retains ~28% signal at gap 8. M is alive but untrained, adds noise. Model can't memorize as cleanly.

**Lesson**: A dead M (fast decay) produces fake-good results because it gets out of the way. A live-but-useless M (slow decay) adds noise and hurts. Always use decay where M actually persists, or results are meaningless.

### Controlled Results (all decay 0.999)

| Config | No Memory | Regular M (uniform) | Predictive M (nudge) |
|--------|-----------|---------------------|----------------------|
| 64v, gap 2-8 | 2.0% | 1.6% | *pending* |
| 128v, gap 5-20 | — | 0.2% | **100%** |
| 128v, gap 10-40 (decay 0.9999) | — | 0.4% | *killed before completion* |

### Predictive M (nudge) vs Regular M

Predictive mode: `output = W_out(M @ query) - W_nudge(x)` (delta toward predicted state)
Regular mode: `output = W_out(M @ query)` (additive)

Predictive M also stores future states: `value = W_value(next_chunk_hidden)`.

On the medium task (128 values, gap 5-20, decay 0.999):
- Regular M: 0.2% (chance)
- Predictive M: 100% (perfect)

This is the only clean result so far. Needs memorization test to confirm (train on even values, eval on odd).

### Invalidated Results (decay 0.99, M was dead)

These results are meaningless — M had decayed to ~0 and model memorized:

| Config | Regular M | Predictive M |
|--------|-----------|-------------|
| 64v, gap 2-8, decay 0.99 | 100% (memorized) | 100% (memorized) |
| 256v, gap 2-8, decay 0.99 | 0.2% (too many pairs to memorize, M dead) | 0.2% (same) |

### Loss Dilution Problem

The recall token is 1 out of N tokens in the sequence. With uniform loss weighting, the recall gradient is ~1/N of total. For longer sequences, this signal gets buried. Added `--recall-loss-weight` flag to amplify recall signal. Not yet tested in clean sweep.

### Open Questions

1. Does predictive M generalize to unseen key-value pairs? (memorization test needed)
2. Why does predictive mode work when regular doesn't? Hypotheses:
   - Richer stored values (future states contain attention-processed info from store)
   - Stronger gradient signal from subtraction
   - Extra parameters (W_nudge)
3. Does recall-weighted loss fix regular M?
4. Reproducibility: no-memory baseline gave 24% in one run, 2% in another (same config). Need multiple seeds.

---
*Last updated: 2026-03-12*
