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

### Key Discovery 11: Three-Phase Training Worse Than Two-Phase

**Three-phase** (pretrain transformer → freeze + train memory → unfreeze all):

| Approach | 4-pair Recall | Gate ratio |
|---|---|---|
| Three-phase fast weight (33/33/33) | 0.5% | 12.6x |
| Three-phase Hopfield (33/33/33) | 0.6% | — |
| Three-phase Hopfield long-p1 (25/50/25) | 0.9% | — |
| **Two-phase Hopfield 20ep** | **18.1%** | **1.6M** |

**Why three-phase is worse:** A pretrained transformer already solves filler prediction. When M starts training in phase 1, there's weak gradient pressure because loss is already low. A *helpless* (random) transformer is the best teacher for M — maximum gradient forces M to learn.

### Key Discovery 12: More Training Helps Two-Phase (25.8%)

| Epochs | Two-phase Hopfield 4-pair |
|---|---|
| 20 | 18.1% |
| **40** | **25.8%** |

Consistent across gaps (23-29% per gap). The model retrieves ~1 of 4 pairs correctly. Still stuck in "retrieve one fixed pair" pattern, but more training pushes it higher.

---

## v3.1: Pretrained Model + LoRA + Hopfield Memory

**Setup**: Qwen2.5-3B (frozen, bf16) + LoRA (rank 16, q_proj/v_proj) + Hopfield memory (256-dim, 64 slots) attached at layer 18/36.

### Key Discovery 13: Per-Chunk Processing Required

With whole-document processing, the pretrained model's self-attention sees all tokens — memory is redundant. LoRA alone gets 100% on 4 pairs (full attention) or 92% (chunk-local attention mask, learning from shared weights).

**Per-chunk processing**: each chunk gets its own `model()` call. Memory state persists across chunks (with gradient flow). No cross-chunk attention — memory is the only information channel.

| Approach | 1-pair Recall | Notes |
|---|---|---|
| LoRA only (per-chunk) | **1.0%** | Chance — confirms memory is necessary |
| LoRA + Memory (per-chunk, joint) | **0.0%** | Gate perfect, retrieval fails |

### Key Discovery 14: Gate Learns Perfectly, Retrieval Fails

Per-chunk joint training (LoRA + Memory, 5 epochs, 500 docs):

| Chunk type | Gate value |
|---|---|
| **STORE** | **0.71** |
| **FILLER** | **0.000** |
| **RECALL** | **0.002** |
| Store/Filler ratio | **1.8 billion x** |

The gate learns the perfect write pattern immediately — far cleaner than any v2.6 result. But recall accuracy is 0%. The bottleneck is **query-key alignment**: W_query applied to RECALL hidden states doesn't match W_key applied to STORE hidden states. The pretrained model produces very different representations for the same key token in different contexts.

### Key Discovery 15: Shared Q=K Init and Longer Training Both Fail

| Experiment | Epochs | Recall | Gate Pattern |
|---|---|---|---|
| Joint (baseline) | 5 | 0% | Perfect (store=0.71, filler=0.000) |
| Shared Q=K init | 10 | 0% | Collapsed (all gates → 0.000) |
| Random Q/K, more training | 10 | 1% | Saturated (all gates → 0.984) |

**The good gate pattern is transient.** 5 epochs produces perfect gate, but 10 epochs destroys it in both init modes. Shared Q=K init makes it worse — gates collapse to zero immediately.

### Key Discovery 16: Hidden State Diagnostic — Signal Exists But Is Weak

Analyzed cosine similarity of pretrained hidden states at layer 18 between STORE and RECALL chunks:

| Comparison | Cosine Similarity |
|---|---|
| Same key token, store vs recall context | 0.747 (std 0.034) |
| Different key tokens, store vs recall | 0.538 (std 0.042) |
| **Separation** | **0.209** |
| Same key, same context, diff docs | ~1.0 |
| Different keys, same context | 0.64-0.70 |

The pretrained model *does* encode key identity across contexts (0.209 separation), but the signal is modest. Within-context distinctiveness is strong (~1.0 for same key), but cross-context similarity is high baseline (~0.5+). A learned projection should amplify this, but the optimization landscape prevents it.

### Pivot: Synthetic Task → Natural Language (v3.2)

**Root cause of v3.1 failure**: The synthetic recall task has a fundamental loss dilution problem — only 1 token per document requires memory. Even with 100x weighting, the gradient signal is dominated by filler/padding prediction. The pretrained model is too competent at filler prediction, leaving no gradient pressure for memory retrieval.

**Solution**: Switch to SQuAD-based extractive QA where:
- Loss is computed ONLY on answer tokens (no dilution by construction)
- Multiple QA pairs per passage → 15+ answer tokens per document (vs 1)
- Natural language plays to pretrained model's strengths
- Per-chunk processing still makes memory the only cross-chunk channel

---

## v3.2: Natural Language Recall (SQuAD + Hopfield Memory)

**Setup**: Qwen2.5-3B (frozen, bf16) + LoRA (rank 16) + HopfieldMemory (256-dim, 64 slots) at layer 18/36. Per-chunk processing (chunk_size=128).

**Document structure**:
- 1-4 passage chunks (SQuAD context)
- 2-6 filler chunks (WikiText paragraphs)
- 1 QA chunk: "Q: ... A: ... Q: ... A: ..." (3 QA pairs)

Loss = answer-only (masked cross-entropy on answer tokens only).

### Key Discovery 17: Memory Adds Nothing on SQuAD — Parametric Knowledge Dominates

| Condition | Best Token Accuracy | Final Token Accuracy |
|---|---|---|
| No memory (LoRA only) | 57.1% (ep 2) | 54.8% (ep 3) |
| Memory + LoRA (bias=-2) | 57.0% (ep 2) | 51.6% (ep 5) |
| Memory + LoRA (bias=0) | 57.0% (ep 1) | 54.1% (ep 3) |
| Untrained baseline | 55.5% | — |

**All three conditions are identical.** The untrained model already gets 55.5% from parametric knowledge — Qwen has seen SQuAD during pretraining. LoRA improves QA formatting slightly (+1.5%), then overfits. Memory has zero contribution.

**Gate values**: Passage=0.53, Filler=0.48 (ratio=1.1x). The gate cannot distinguish passage from filler — both are natural language text, unlike the synthetic task where STORE chunks had unique tokens.

**Root cause**: The task doesn't genuinely require memory. Most SQuAD answers are in the model's weights. Memory needs a task where the **only** way to answer is to retrieve from the passage.

### Key Discovery 18: Template Memorization With Small Pools

First synthetic generator used small name/field/city pools (30×20×15). LoRA memorized the template mapping without needing memory:

| Condition | Synthetic v1 (small pools) |
|---|---|
| Untrained baseline | 34.8% |
| Memory + LoRA | 77.3% |
| **No memory (LoRA only)** | **77.4%** (identical!) |

LoRA learned to predict answers from question structure alone. Not a memory result.

### Key Discovery 19: Truly Unique Facts — Gate Learns, Retrieval Fails

Second synthetic generator: random names, random 3-digit codes, random cities. Every fact is unique per document. No template memorization possible.

| Condition | Token Accuracy | Gate Ratio |
|---|---|---|
| Untrained baseline | 10.2% | — |
| Memory + LoRA (5 ep) | **13.5%** | **2.6x** (passage=0.63, filler=0.24) |

**Two findings:**
1. **Gate learned to discriminate** passage from filler on natural language (2.6x ratio vs 1.1x on SQuAD). Synthetic passages have different statistics than WikiText filler.
2. **Retrieval still fails** — only +3.3% over baseline despite correct gating. Same problem as v3.1: W_query/W_key alignment doesn't learn.

**Pattern across all experiments**: The gate is easy to train. Retrieval alignment is the fundamental bottleneck. This holds for both synthetic token recall (v3.1) and natural language recall (v3.2).

### Key Discovery 20: Attention Alignment Is Already Perfect — Value Encoding Is the Bottleneck

Replaced cosine retrieval loss with attention supervision: cross-entropy loss pushing Hopfield attention toward passage slots in K_mem. Result: **attention was already 100% on passage slots before any supervision.**

| Metric | Value |
|---|---|
| Attention on passage slots | **100%** (epoch 1, no supervision needed) |
| Token accuracy | 14.2% (vs 10.2% baseline) |
| Cosine retrieval loss | Collapsed to trivial solution (0.001) — was solving a non-problem |
| Attention supervision loss | 0.0000 after 50 steps — already satisfied |

**Diagnosis updated:** The bottleneck is NOT attention alignment (Q/K matching). The memory correctly identifies and attends to passage slots. The problem is **value utilization**: the chunk-aggregated, normalized, 256-dim vectors in V_mem don't retain enough factual information to predict specific answers. A single V_mem slot can't simultaneously encode "code=847", "city=Xilob", "year=1923".

**Implications for architecture:**
1. Need higher-capacity value encoding (larger memory_size, or store raw hidden states)
2. Need per-token storage instead of chunk aggregation (one slot per token, not per chunk)
3. May need to bypass W_value projection and store raw hidden states
4. The W_out projection (256→2048, initialized to zero) may be too constrained

### Key Discovery 21: Memory Contributes Zero — Frozen Model Can't Use W_out Injection

Definitive controlled experiment: trained LoRA for 5 epochs **without memory** and got the same accuracy as all memory configurations. The entire ~15% accuracy comes from LoRA learning from passage/filler LM loss. Memory adds nothing.

| Config | Token Acc (ep3) | Memory contribution |
|---|---|---|
| Untrained baseline | 10.2% | — |
| **No memory, trained LoRA** | **15.5%** | **N/A** |
| Memory, mean-pool (256-dim) | 14.9% | 0% |
| Memory, mean-pool (1024-dim) | 15.0% | 0% |
| Memory, mean-pool (2048-dim) | 14.9% | 0% |
| Memory, n_extract=4 | 14.8% | 0% |
| **Memory, full LoRA (all linear, 30M params)** | **14.4%** | **0% (actively ignored)** |

Furthermore, 100% vs 3% attention on passage slots makes no difference — the W_out output is noise either way. With full LoRA (6 targets, 30M params, 0.99% of model), attn_passage dropped from 1.4% → 0.0% — the model actively learned to IGNORE memory.

**Root cause**: Additive residual injection is fundamentally incompatible with pretrained transformers. The model's residual stream has a learned distribution at each layer. Adding a foreign signal (even a small one) disrupts downstream processing. More LoRA capacity doesn't help — the model uses that capacity to suppress the memory signal, not utilize it.

**Implications for architecture**:
1. Additive residual injection doesn't work, even with full LoRA on all linear layers
2. Need cross-attention injection: memory slots become extra KV pairs in transformer attention
3. Or train from scratch so the model learns with memory from the start
4. The extraction queries and gate mechanisms are sound — revisit once injection works

### Key Discovery 22: Learned Extraction Queries (n_extract=4)

Replaced mean-pooling with k=4 learned queries that cross-attend over chunk tokens to produce diverse summary vectors. Each query could specialize in different info types (entities, numbers, etc.).

Result: same accuracy as mean-pooling. But this was moot since Discovery 21 shows the injection path itself is broken. The extraction mechanism should be revisited once injection works.

### Open Questions

1. ~~Does recall-weighted loss enable retrieval?~~ **Yes, for 1 pair (v2.6).**
2. ~~Does M scale to multiple pairs?~~ **Two-phase + Hopfield gets 25.8% at 40ep.**
3. ~~Can three-phase improve on two-phase?~~ **No — pretrained transformer hurts M learning.**
4. ~~Does per-chunk processing make memory necessary?~~ **Yes — LoRA alone at chance.**
5. ~~Can shared Q=K init solve alignment?~~ **No — gates collapse.**
6. ~~Does answer-only loss on NL QA produce memory-dependent retrieval?~~ **No — all improvement is LoRA.**
7. ~~Does the gate learn passage vs filler discrimination on natural text?~~ **Yes — 2.6x ratio.**
8. ~~Is attention alignment the bottleneck?~~ **No — 100% attention on passage slots already.**
9. ~~Does larger memory_size help?~~ **No — 256/1024/2048 all identical.**
10. ~~Does learned extraction (multi-slot) help?~~ **No — injection path itself is broken.**
11. ~~Can full LoRA (all linear layers) help the model use memory output?~~ **No — model uses extra capacity to suppress memory.**
12. ~~Does cross-attention injection work?~~ **YES — 42.5% vs 15.5% no-memory. BREAKTHROUGH.**

### Key Discovery 23: Cross-Attention Memory Injection — First Working Memory on Pretrained Model

Replaced additive W_out injection with cross-attention: memory hidden states become extra KV pairs in layer 19's attention. The model's own attention mechanism decides whether to attend to memory.

| Epoch | Token Accuracy | Improvement over no-memory |
|---|---|---|
| 1 | 15.4% | +0% (warming up) |
| 2 | 20.0% | +4.5% |
| 3 | 30.3% | +14.8% |
| 4 | 39.5% | +24.0% |
| **5** | **42.5%** | **+27.0%** |

**Architecture**: Write@layer 18 (gate → store hidden states in MemoryBank), Read@layer 19 (MemoryAugmentedAttention concatenates memory KVs before softmax). No RoPE on memory KVs (content-based, position-independent). Only 2,049 new params (gate only) — reuses model's own k_proj/v_proj via LoRA.

**Why it works**: Instead of adding a foreign signal to the residual stream (which the model ignores/suppresses), cross-attention uses the model's existing attention mechanism. The model already knows how to attend to KV pairs — memory KVs are just more KV pairs. LoRA on q_proj/v_proj learns to query/value-project memory naturally.

**Implications**:
1. The memory architecture is validated — gate + storage + cross-attention retrieval works
2. Next: add learned extraction queries to improve what gets stored
3. Next: test with more QA pairs, longer gaps, harder tasks
4. Next: hierarchical compression (compress chunks, then compress compressed chunks)

### Key Discovery 24: 1.5B Model Outperforms 3B — 97.6% Token Accuracy

Migrated to Qwen2.5-1.5B on 3080 Ti (12.9GB VRAM). The smaller model significantly outperforms the 3B on cross-attention memory.

**1.5B cross-attention, 15 epochs** (500 train, 100 eval, chunk_size=128, 3 QA pairs):

| Epoch | Token Accuracy | Exact Match |
|---|---|---|
| Baseline (untrained) | 10.8% | 0.0% |
| **No memory (LoRA, 5 ep)** | **15.9%** | **0.0%** |
| 1 | 29.5% | 0.0% |
| 2 | 56.8% | 1.0% |
| 3 | 75.2% | 9.0% |
| 4 | 88.7% | 29.0% |
| 5 | 92.7% | 48.0% |
| 6 | 94.7% | 59.0% |
| 7 | 96.1% | 67.0% |
| 8 | 96.7% | 73.0% |
| 9 | 97.2% | 76.0% |
| 10 | 97.4% | 78.0% |
| 11-15 | ~97.5-97.7% | 77-80% |

**Comparison: 1.5B vs 3B at epoch 5:**

| Model | Token Accuracy | Exact Match | Trainable params |
|---|---|---|---|
| Qwen2.5-3B | 42.5% | — | 3.7M LoRA + 2,049 mem |
| **Qwen2.5-1.5B** | **92.7%** | **48.0%** | **2.2M LoRA + 1,537 mem** |

**Why 1.5B outperforms 3B:**
1. LoRA has more relative influence (0.14% of 1.5B vs 0.12% of 3B)
2. Shorter gradient path — 28 layers vs 36, memory at layers 14/15 vs 18/19
3. The task doesn't need 3B's extra capacity — it's factual retrieval, not reasoning

**What the errors look like at 97.6%:**
- Numbers (codes, years) are nearly 100% correct
- Random words (cities, org names) are the remaining errors — novel subword tokens that the model must reproduce exactly
- Debug output shows all 10 sample docs getting every token correct at epoch 15

**Plateau behavior:** Accuracy saturated around epoch 10 (97.4%). Epochs 10-15 show marginal gains (97.4→97.7%), suggesting diminishing returns with current architecture on this task difficulty.

### Key Discovery 25: Learned Extraction Queries (n_extract=4) — Marginal Improvement

Added learned extraction queries to MemoryBank for cross-attention mode. 4 queries cross-attend over chunk tokens to produce 4 diverse summary vectors instead of mean-pooling into a single slot.

**Comparison at 15 epochs** (1.5B, 3 QA pairs):

| Config | Token Accuracy | Exact Match | Memory Params |
|---|---|---|---|
| n_extract=1 (mean-pool) | 97.6% | 79% | 1,537 |
| **n_extract=4 (learned)** | **98.1%** | **83%** | **7,681** |

Small but real improvement (+0.5% token, +4% exact match). The extraction queries help with novel subword tokens but don't fundamentally change the picture — a single 1536-dim vector already captures enough for 3 QA pairs.

### Key Discovery 26: 6 QA Pairs — Architecture Scales With No Degradation

Doubled the number of QA pairs per passage from 3 to 6. All 6 question types asked: city, organization, result code, year, output code, field.

**Comparison: 3 vs 6 QA pairs** (1.5B, cross-attention, 15 epochs, n_extract=1):

| Epoch | 3 QA (token/exact) | 6 QA (token/exact) |
|---|---|---|
| Baseline (no mem) | 10.8% / 0% | 11.7% / 0% |
| 1 | 29.5% / 0% | 35.4% / 0% |
| 2 | 56.8% / 1% | 64.3% / 0% |
| 3 | 75.2% / 9% | 91.8% / 17% |
| 4 | 88.7% / 29% | 96.1% / 48% |
| 5 | 92.7% / 48% | 97.9% / 66% |
| 6 | 94.7% / 59% | 98.7% / 77% |
| 7 | 96.1% / 67% | 98.5% / 74% |
| 8 | 96.7% / 73% | 99.0% / 81% |
| 9 | 97.2% / 76% | **99.2% / 86%** |
| 10 | 97.4% / 78% | 99.0% / 81% |
| 15 | 97.7% / 80% | 99.2% / 86% |

**Key findings:**
1. **Token accuracy is HIGHER with 6 QA** — more answer tokens = richer gradient signal per doc
2. **Exact match is lower** but this is expected — 6 pairs all correct is harder than 3 all correct
3. **Learning is faster** — 91.8% at epoch 3 (vs 75.2%) due to double the training signal
4. **Memory capacity is not the bottleneck** — a single 1536-dim vector successfully encodes 6 distinct facts

### Key Discovery 27: Long Gaps (6-12 filler chunks) — No Degradation

Tested with 6-12 filler chunks between passage and QA (vs default 2-6). More filler means more writes to the circular buffer that could overwrite passage memories.

**Result** (1.5B, 6 QA pairs, 15 epochs):

| Gap range | Best Token Acc | Best Exact Match |
|---|---|---|
| 2-6 (default) | 99.2% | 86% |
| **6-12 (long)** | **99.6%** | **92%** |

**Actually better with longer gaps.** More chunks per doc = more forward passes = richer gradient signal. The gate successfully suppresses filler writes — passage memories are not corrupted despite up to 12 filler chunks.

### Key Discovery 28: Multi-Passage Recall — 99.8% Token / 97% Exact Match

Two passages about different people, with interleaved QA about both. Tests whether memory can keep facts from different passages separate (interference test).

**Document structure**: [Passage A] [filler 2-6] [Passage B] [filler 2-6] [QA about both]

**Result** (1.5B, 3 QA per person = 6 total, 15 epochs, n_extract=1):

| Epoch | Token Accuracy | Exact Match |
|---|---|---|
| Baseline (no mem) | 12.2% | 0% |
| 2 | 90.4% | 17% |
| 5 | 99.2% | 86% |
| 9 | 99.7% | 95% |
| 12 | **99.8%** | **97%** |
| 15 | 99.8% | 97% |

**Best result across all experiments.** Only 3 token errors in 1881 tokens across 100 documents. The model perfectly distinguishes which facts belong to which person — no interference between passages in memory. Debug output shows all 10 sample docs completely correct, including novel subword tokens.

**Why this matters:**
1. Memory slots from different passages occupy separate slots in the circular buffer
2. Cross-attention naturally selects the right slot based on the person's name in the question
3. No architectural changes needed — the same mechanism scales from 1 to 2 passages

**Verification**: Confirmed zero data leakage between chunks. Each chunk is a separate `model()` call with independent KV computation. The no-memory baseline (12.2%) proves the only cross-chunk channel is memory.

### Key Discovery 29: Confusable Facts — Shared Attributes Make No Difference

Tested with 2 passages sharing 1-2 attributes (organization, field, city) to check whether the model relies on keyword matching or genuine entity binding.

**Result** (1.5B, 2 passages, 3 QA per person = 6 total, 15 epochs):

| Condition | Token Accuracy | Exact Match |
|---|---|---|
| Multi-passage (distinct) | 99.8% | 97% |
| **Multi-passage (confusable)** | **99.8%** | **97%** |

**Identical performance.** The model binds facts to entities via the person's name, not keyword matching. Shared attributes between passages cause zero confusion — cross-attention selects the correct memory slot based on the entity name in the question, not surface-level attribute overlap.

### Key Discovery 30: 5-Passage Scaling — 100% Perfect Score

Scaled from 2 to 5 passages per document, each about a different person. 10 QA pairs total (2 per person).

**Result** (1.5B, 5 passages, 2 QA per person = 10 total, 15 epochs):

| Epoch | Token Accuracy | Exact Match |
|---|---|---|
| Baseline (no mem) | ~10% | 0% |
| 3 | ~98% | ~90% |
| 7 | **100.0%** | **100.0%** |
| 8-15 | 100.0% | 100.0% |

**Perfect score from epoch 7 onward.** No degradation whatsoever going from 2 to 5 passages. The circular buffer and cross-attention mechanism scale naturally — each passage occupies its own memory slots, and the model selects the right slots for each question.

### Key Discovery 31: Generalization — Memory Skill Transfers Across Formats

Tested whether the memory skill learned on single-passage synthetic data transfers to unseen formats without any retraining.

**Cross-format evaluation** (trained on single-passage synthetic, evaluated on other formats):

| Train format | Eval format | Token Accuracy |
|---|---|---|
| Single-passage synthetic | Multi-passage (2 passages) | 98.0% |
| Single-passage synthetic | 5-passage | 98.1% |
| Single-passage synthetic | SQuAD (real Wikipedia) | 86.0% |
| No memory baseline | SQuAD | 55.5% |

**The model learned "how to use memory" as a general skill**, not a format-specific trick. Zero retraining needed — a checkpoint trained only on single-passage synthetic facts can recall information from multiple passages and even real Wikipedia articles. The +30.5% improvement on SQuAD over the no-memory baseline is especially striking given the model was never trained on that format.

### Key Discovery 32: Natural-Style Training Improves Generalization

Trained with more diverse passage templates (6 varied templates) and question phrasings (3 per fact type) to improve cross-format transfer.

**Result** (1.5B, 15 epochs, natural-style training):

| Eval format | Natural-trained | Synthetic-trained | No memory |
|---|---|---|---|
| Natural format | 99.2% | — | ~10% |
| Synthetic format (cross-format) | 98.5% | 97.7% | ~10% |
| SQuAD (real Wikipedia) | **90.4%** | 86.0% | 55.5% |

**Key findings:**
1. **+35 point improvement on real Wikipedia from memory alone** (90.4% vs 55.5% no-memory)
2. Natural-style training improves SQuAD generalization by +4.4% over synthetic training (90.4% vs 86.0%)
3. Cross-format transfer works in both directions — natural-trained model gets 98.5% on synthetic format
4. Template and question diversity during training helps the model generalize to real-world text

### Key Discovery 33: Streaming Document QA on Real Wikipedia

End-to-end test: process real Wikipedia articles chunk-by-chunk (streaming), accumulate memory across chunks, then answer questions about the article content.

**Result** (1.5B, natural-trained checkpoint, zero additional training):

| Question type | Token Accuracy | Exact Match | Questions | Articles |
|---|---|---|---|---|
| Numbers only | **100.0%** | **100.0%** | 227 | 48 |
| Diverse (entities, locations, numbers) | **97.3%** | **73.9%** | 364 | 49 |
| No-memory baseline | 39.7% | — | — | — |

**Key findings:**
1. **Perfect recall on numerical facts** — 100%/100% on 227 questions across 48 Wikipedia articles
2. **97.3% token accuracy on diverse questions** — entities and locations are harder due to novel subword tokens
3. **73.9% exact match on diverse questions** — lower because entity names must match every character exactly
4. **+57.6% token accuracy over no-memory baseline** (97.3% vs 39.7%)
5. **Zero additional training** — uses the natural-trained checkpoint directly on real Wikipedia
6. This is a genuine streaming architecture — each chunk is processed independently, memory is the only cross-chunk channel

**Summary table — all 1.5B cross-attention results:**

| Experiment | Passages | QA pairs | Gap | Best Token Acc | Best Exact Match |
|---|---|---|---|---|---|
| No memory baseline | 1 | 3 | 2-6 | 15.9% | 0% |
| Cross-attn 15ep | 1 | 3 | 2-6 | 97.7% | 80% |
| Cross-attn n_extract=4 | 1 | 3 | 2-6 | 98.2% | 84% |
| Cross-attn 15ep | 1 | 6 | 2-6 | 99.2% | 86% |
| Cross-attn 15ep | 1 | 6 | 6-12 | 99.6% | 92% |
| Cross-attn multi | 2 | 6 (3+3) | 2-6 | 99.8% | 97% |
| Cross-attn confusable | 2 | 6 (3+3) | 2-6 | 99.8% | 97% |
| Cross-attn 5-passage | 5 | 10 (2×5) | 2-6 | 100.0% | 100.0% |
| Generalization: multi-passage | 2 | 6 (3+3) | 2-6 | 98.0% | — |
| Generalization: 5-passage | 5 | 10 (2×5) | 2-6 | 98.1% | — |
| Generalization: SQuAD | 1 | varies | — | 86.0% | — |
| Natural-trained: synthetic | 1 | varies | 2-6 | 98.5% | — |
| Natural-trained: SQuAD | 1 | varies | — | 90.4% | — |
| **Streaming Wikipedia (numbers)** | **1** | **varies** | **—** | **100.0%** | **100.0%** |
| **Streaming Wikipedia (diverse)** | **1** | **varies** | **—** | **97.3%** | **73.9%** |

### Open Questions (Updated)

1. ~~Does learned extraction + cross-attention improve further?~~ **Marginal — +0.5% token, +4% exact match with n_extract=4.**
2. ~~How does accuracy scale with more QA pairs per passage?~~ **No degradation — 6 QA pairs gives higher token accuracy than 3.**
3. ~~How does accuracy degrade with longer gaps (more filler chunks)?~~ **No degradation — 99.6%/92% with gap 6-12.**
4. ~~Can this generalize to multi-passage recall?~~ **Yes — 99.8%/97% with 2 passages, no interference.**
5. ~~Does accuracy hold with confusable facts (shared attributes between passages)?~~ **Yes — 99.8%/97%, identical to distinct passages. Model binds facts by entity name.**
6. ~~How does accuracy scale to 5-10 passages?~~ **5 passages: 100%/100% from epoch 7. No degradation.**
7. Can this generalize to multi-hop reasoning?
8. Does hierarchical memory compression work?
9. What does the attention pattern look like — does it attend to specific memory slots for specific questions?
10. ~~Does extended training find a higher ceiling?~~ **Yes — 97.6% at epoch 10-15 (vs 69.8% at epoch 5 with cosine decay).**
11. ~~Does the 1.5B model work as well as the 3B?~~ **Better — 92.7% at ep 5 vs 42.5% for 3B.**
12. ~~Does the memory skill transfer across formats?~~ **Yes — train on synthetic, eval on multi-passage (98%), SQuAD (86%), no retraining needed.**
13. ~~Does natural-style training improve generalization?~~ **Yes — 90.4% on SQuAD (vs 86% from synthetic training). +35 points over no-memory baseline.**
14. ~~Does streaming document QA work on real Wikipedia?~~ **Yes — 100%/100% on numbers, 97.3%/73.9% on diverse questions. Zero additional training.**
15. How does accuracy scale to 10+ passages?
16. Can the architecture handle documents longer than ~2K tokens?
17. Does fine-tuning on real Wikipedia QA data further improve streaming performance?

---
*Last updated: 2026-03-15*
