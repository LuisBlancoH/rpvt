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

**The memory module adds zero value.** Controlled experiment: training LoRA without memory gives 15.5% — identical to all memory configs (~14.8%). The +5% over untrained baseline (10.2%) is entirely from LoRA learning on passage/filler LM loss. Memory output is ignored.

**Root cause**: Additive W_out injection into the residual stream of a frozen model doesn't work. Layers 19-35 were never trained to interpret memory signals. LoRA rank 16 on q_proj/v_proj is insufficient to adapt.

| Config | Token Acc | Memory adds |
|---|---|---|
| No memory, trained LoRA | 15.5% | N/A |
| Memory (any size/config) | ~14.8% | 0% |

What DOES work: gate discrimination (2.6x passage/filler), attention alignment (100% on passage slots). The read/write pipeline is functional — the injection point is broken.

## Experiment History Summary

- **v2.6**: Small transformer from scratch, synthetic store/recall. Proved basic Hopfield retrieval works for 1 pair. Scales poorly to 4+ pairs.
- **v3.1**: Pretrained model (Qwen2.5-3B) + synthetic token recall. Per-chunk processing works but retrieval alignment appears stuck.
- **v3.2**: Pivoted to NLP recall (SQuAD + synthetic facts). Discovered:
  - SQuAD contaminated by parametric knowledge (55.5% baseline)
  - Synthetic facts with truly unique entities work (10.2% baseline)
  - Gate works (2.6x), attention works (100%), but memory output ignored
  - memory_size doesn't matter (256=1024=2048)
  - Learned extraction queries (n_extract=4) don't matter
  - **Critical**: no-memory trained LoRA matches memory configs → W_out injection is broken

## Next Steps to Try (Injection Problem)

1. **LoRA at memory injection layer**: Add LoRA to layers 18-20 so the model can learn to use memory signals
2. **Cross-attention injection**: Replace additive W_out with a cross-attention layer that queries memory
3. **Output-space injection**: Inject at logits level instead of hidden state level
4. **Gated injection**: `output = x + gate * W_out(retrieved)` where gate is learned per-token
5. **Train from scratch on smaller model**: Eliminates the frozen-model problem entirely

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
