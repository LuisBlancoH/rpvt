# RPVT — Recurrent Predictive Value Transformer

## What This Is

Research project building **architectural-level memory for LLM agents**. Not scaffolding — we modify the model's forward pass to add a differentiable external memory module (Hopfield memory) that persists across chunks of text.

**Broader vision**: Agents that can plan, learn, evaluate, and have goals. Memory (M) is the first building block. Five minimal mechanisms: prediction, error signal, persistent goal state, multiple timescales, recurrence.

**Philosophy**: Work at architecture level (not scaffolding), take brain *principles* not implementation details, bitter lesson (make mechanisms learnable, don't hard-code), verify foundation before building more.

## Current Architecture

- **Base model**: Qwen2.5-3B (frozen, bf16) — can also use 1.5B for faster iteration
- **LoRA**: rank 16 on q_proj, v_proj
- **HopfieldMemory**: Attached at layer 18/36 (middle). Slot-based memory with:
  - W_gate (learned write gate) — controls what gets stored
  - W_key, W_value — project hidden states for storage
  - W_query — project hidden states for retrieval
  - K_mem, V_mem — the actual memory banks (n_slots x memory_size)
  - Softmax attention retrieval with learned temperature (beta)
  - W_out — project retrieved values back to residual stream
- **Per-chunk processing**: Each chunk is a separate `model()` call. Memory is the ONLY cross-chunk information channel.

## Key Files

- `rpvt/model/hopfield_memory.py` — HopfieldMemory module
- `rpvt/experiments/exp_v3_2_nlp_recall.py` — Current NLP recall experiment (SQuAD + synthetic)
- `rpvt/experiments/exp_v3_1_pretrained_recall.py` — Synthetic token recall (has MemoryWrapper, build_model, etc.)
- `results/FINDINGS.md` — Comprehensive experiment findings (20 key discoveries)

## What Works

1. **Write gating**: Gate reliably learns to discriminate passage from filler (2.6x ratio)
2. **Attention alignment**: 100% of retrieval attention falls on passage slots (not filler)
3. **Per-chunk processing**: Forces genuine memory dependence (LoRA-only baseline at chance: 10.2%)
4. **Answer-only loss**: Loss computed only on answer tokens after "A:" markers — no dilution

## What Doesn't Work (The Core Bottleneck)

**Value encoding/utilization is the unsolved problem.** The memory correctly identifies what to store (gate) and correctly retrieves the right slots (attention), but the retrieved values don't help predict answers.

- memory_size=256: 14.2% (vs 10.2% baseline)
- memory_size=1024: 15.6%
- memory_size=2048: 14.3%
- Increasing value dimensionality doesn't help

**Hypotheses for why**:
1. Chunk aggregation: averaging 128 tokens into one vector destroys token-level facts
2. W_out injection: frozen model can't use the memory signal in the residual stream
3. Both

## Experiment History Summary

- **v2.6**: Small transformer from scratch, synthetic store/recall. Proved basic Hopfield retrieval works for 1 pair. Scales poorly to 4+ pairs.
- **v3.1**: Pretrained model (Qwen2.5-3B) + synthetic token recall. Per-chunk processing works but retrieval alignment appears stuck.
- **v3.2**: Pivoted to NLP recall (SQuAD + synthetic facts). Discovered:
  - SQuAD contaminated by parametric knowledge (55.5% baseline)
  - Synthetic facts with truly unique entities work (10.2% baseline)
  - Gate learns, attention aligns, but values don't help
  - Attention supervision proves alignment was never the bottleneck
  - Value capacity (memory_size) is not the bottleneck either

## Next Steps to Try

1. **Per-token storage**: Write each token to its own slot (no chunk aggregation). Tests whether aggregation is the problem.
2. **Multi-slot learned selection**: k slots per chunk with learned routing of which tokens to store.
3. **Larger W_out / multi-layer W_out**: More expressive value-to-residual projection.
4. **LoRA at memory layer**: Let the layer receiving memory output adapt to use it.
5. **Cross-attention readout**: Instead of additive injection, use cross-attention to read from memory.

## Environment

- Python: use system python or venv with torch, transformers, peft, datasets
- GPU: works on 12GB+ (3080 Ti with 1.5B model, or 24GB+ for 3B)
- Always push to GitHub after code changes
- Update results/FINDINGS.md with experiment outcomes

## Synthetic Data Task

The current benchmark: generate passages with truly unique facts (random names, random 3-digit codes, random cities). Model sees passage in early chunks, filler chunks (WikiText), then QA chunk. Must answer from memory.

```
Passage: "Balcorfen Torushvel was a researcher at the Criho Institute in Xilob..."
Q: What was Balcorfen Torushvel's result code? A: 847
```

Baseline (no memory, per-chunk): 10.2%. Current best with memory: ~14-15%.
