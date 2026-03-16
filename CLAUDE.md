# RPVT — Recurrent Predictive Value Transformer

## What This Is

Research project building **architectural-level memory for LLM agents**. Not scaffolding — we modify the model's forward pass to add a differentiable external memory module that persists across chunks of text. The model processes text in chunks, with memory as the ONLY cross-chunk information channel.

**Broader vision**: Agents that can plan, learn, evaluate, and have goals. Memory (M) is the first building block. Five minimal mechanisms: prediction, error signal, persistent goal state, multiple timescales, recurrence.

**Philosophy**: Work at architecture level (not scaffolding), take brain *principles* not implementation details, bitter lesson (make mechanisms learnable, don't hard-code), verify foundation before building more.

**Owner**: Luis Blanco, SWE at Google, independent ML researcher.

## Current Working Architecture (v3.2/v3.3 cross-attention)

**This is the architecture that works. Previous additive injection approaches all failed.**

- **Base model**: Qwen2.5-1.5B (frozen, bf16) on 3080 Ti
- **LoRA**: rank 16 on q_proj, v_proj (2.18M params)
- **MemoryBank** (`rpvt/model/cross_attention_memory.py`): Stores gated hidden states
  - W_gate: learned write gate (bias=-2.0) — controls what gets stored
  - Circular buffer of n_slots=64 hidden state vectors (raw, no projection)
  - Only 1,537 trainable params (just the gate)
  - Optional n_extract learned extraction queries for multi-slot writes
- **WriteWrapper**: Wraps layer 14 — passes output through unchanged, then writes gated hidden states to MemoryBank
- **MemoryAugmentedAttention**: Replaces layer 15's self_attn — concatenates memory hidden states as extra KV pairs before attention computation
  - Projects memory through layer 15's own k_proj/v_proj (with LoRA)
  - No RoPE on memory KVs (content-based, position-independent retrieval)
  - Model's own attention mechanism decides whether to attend to memory
- **Per-chunk processing**: Each chunk is a separate `model()` call. Memory is the ONLY cross-chunk information channel.
- **128× compression**: 128 tokens → 1 memory vector (1536-dim)

## Key Results

| Experiment | Token Acc | Exact Match | Notes |
|---|---|---|---|
| No memory baseline | 10.8-15.9% | 0% | LoRA only, per-chunk |
| Single passage, 3 QA | 97.7% | 80% | 15 epochs |
| Single passage, 6 QA | 99.2% | 86% | More QA = richer signal |
| Long gaps (6-12 filler) | 99.6% | 92% | Gate suppresses filler |
| 2 passages, confusable | 99.8% | 97% | Shared attributes no effect |
| **5 passages, 10 QA** | **100.0%** | **100.0%** | Perfect from epoch 7 |
| Generalization: train single → eval multi | 98.0% | 68% | No retraining |
| Generalization: train single → eval 5-passage | 98.1% | 68% | No retraining |
| Natural → SQuAD (real Wikipedia) | 90.4% | 48.7% | +35 pts over no-memory |
| **Streaming Wikipedia QA** | **97.3%** | **73.9%** | Real articles, diverse Q types |

## Key Files

- `rpvt/model/cross_attention_memory.py` — MemoryBank, WriteWrapper, MemoryAugmentedAttention (THE WORKING APPROACH)
- `rpvt/model/hopfield_memory.py` — HopfieldMemory module (additive injection — DOES NOT WORK on pretrained models)
- `rpvt/experiments/exp_v3_2_nlp_recall.py` — Synthetic/natural recall experiments (all data sources)
- `rpvt/experiments/exp_v3_3_streaming.py` — Streaming document QA on real Wikipedia
- `rpvt/experiments/exp_v3_1_pretrained_recall.py` — Has build_model(), MemoryWrapper, reset_memories, etc.
- `results/FINDINGS.md` — Comprehensive experiment findings (33 key discoveries)
- `results/RESULTS_SUMMARY.md` — Clean summary of what works (paper-ready)
- `results/checkpoints/` — Saved LoRA + memory weights

## What Works

1. **Cross-attention memory injection**: Memory as extra KV pairs in attention — up to 100% on synthetic, 97.3% on real Wikipedia
2. **1.5B model outperforms 3B**: Shorter gradient path + higher LoRA influence ratio
3. **Scales to 5+ passages**: Perfect score with 5 passages, 10 interleaved QA
4. **Confusable facts**: Shared attributes between passages don't degrade accuracy
5. **Generalizes across formats**: Train on single-passage → 98% on multi-passage without retraining
6. **Transfers to real text**: 90.4% on SQuAD, 97.3% on streaming Wikipedia QA
7. **Write gating**: Gate discriminates passage from filler (works on natural text too)
8. **Streaming mode**: Process real documents chunk-by-chunk, answer from accumulated memory

## What Does NOT Work (Critical Lessons)

1. **Additive residual injection (W_out → residual stream)**: Pretrained models ignore/suppress the signal. Even with full LoRA (30M params), the model suppresses memory.
2. **Increasing memory_size**: 256, 1024, 2048 all identical — not a capacity problem
3. **n_extract queries**: Marginal improvement (+0.5% token, +4% exact match). Mean-pool sufficient for current task.
4. **SQuAD as sole benchmark**: Parametric knowledge dominates (55% baseline). Use synthetic facts for clean evaluation.
5. **Small entity pools**: LoRA memorizes template mappings. Use truly random names/codes/cities.

## Data Sources

- `synthetic` — rigid template, random entities/codes/years (original)
- `synthetic_natural` — 6 varied templates, 3 question phrasings per type (better for generalization)
- `synthetic_multi` — 2 passages about different people, interleaved QA
- `synthetic_confusable` — 2 passages sharing 1-2 attributes (org/field/city)
- `synthetic_n_N` — N passages (e.g. `synthetic_n_5`, `synthetic_n_10`)
- `squad` — real SQuAD v2 Wikipedia passages

## Experiment History (Condensed)

### v2.6: Small model from scratch
- Proved Hopfield retrieval works for 1 pair (100% with loss weighting)
- Scales poorly to 4+ pairs (25.8% best with two-phase training)
- Additive injection works when training from scratch

### v3.1: Pretrained model + synthetic tokens
- Per-chunk processing works — memory is the only information channel
- Gate learns perfectly (1.8 billion x store/filler ratio)
- But retrieval fails — additive injection doesn't work on pretrained models

### v3.2: Pretrained model + NLP recall (BREAKTHROUGH)
- Cross-attention injection: 42.5% on 3B (5 ep), **97.6% on 1.5B (15 ep)**
- Scales to 6 QA (99.2%), long gaps (99.6%), 2 passages (99.8%), 5 passages (100%)
- Confusable facts: no degradation (99.8%)
- Generalizes: single→multi 98%, single→5-passage 98.1%
- Natural training: 99.2% on format, 90.4% on real Wikipedia

### v3.3: Streaming document QA
- Real Wikipedia articles processed chunk-by-chunk
- 97.3% token accuracy on diverse questions (numbers, entities, locations)
- No-memory baseline: 39.7%
- Uses trained checkpoint with zero additional training

## Next Steps (Priority Order)

1. ~~More training~~: DONE — ceiling ~97-100% depending on task
2. ~~Scaling tests~~: DONE — 6 QA, long gaps, 5 passages all work
3. ~~Multi-passage~~: DONE — 100% on 5 passages
4. ~~Generalization~~: DONE — transfers across formats and to real text
5. ~~Streaming QA~~: DONE — 97.3% on real Wikipedia
6. **Interactive demo** — feed any document, chat about it from memory
7. **Recurrence** — output feeds back to input (see memory/project_recurrence_ideas.md)
8. **Port to other models** — Llama, Mistral, Phi
9. **Agent integration** — memory in an agent loop across tool calls
10. **Hierarchical compression** — multi-timescale memory
11. **Paper** — write up results

## Environment

- Python 3.12, venv at `.venv/` — activate with `source .venv/bin/activate` or use `.venv/bin/python3`
- Packages: PyTorch 2.6+cu124, transformers 5.3.0, peft 0.18.1, datasets 4.7.0
- GPU: RTX 3080 Ti (~12.9GB VRAM), bf16 supported, WSL2
- Use Qwen2.5-1.5B for experiments (3B needs 24GB+)
- Always push to GitHub after code changes
- Update `results/FINDINGS.md` with experiment outcomes
- Use `--memory-mode cross_attn` flag for the working architecture
- Use `--data-source synthetic_natural` for best generalization

## Running Experiments

```bash
# Train with checkpoint saving
python -m rpvt.experiments.exp_v3_2_nlp_recall \
  --model-name Qwen/Qwen2.5-1.5B \
  --data-source synthetic_natural \
  --memory-mode cross_attn \
  --epochs 15 \
  --n-train 500 --n-eval 100 \
  --save-checkpoint results/checkpoints/my_checkpoint.pt \
  --output-dir results/my_experiment

# Cross-format evaluation (no retraining)
python -m rpvt.experiments.exp_v3_2_nlp_recall \
  --model-name Qwen/Qwen2.5-1.5B \
  --data-source synthetic_natural \
  --memory-mode cross_attn \
  --load-checkpoint results/checkpoints/natural_15ep.pt \
  --eval-data-source synthetic_multi \
  --output-dir results/my_crosseval

# Streaming document QA
python -m rpvt.experiments.exp_v3_3_streaming \
  --load-checkpoint results/checkpoints/natural_15ep.pt \
  --n-articles 50 --n-questions 8 \
  --output-dir results/my_streaming

# No-memory baseline
python -m rpvt.experiments.exp_v3_2_nlp_recall \
  --model-name Qwen/Qwen2.5-1.5B \
  --data-source synthetic \
  --no-memory --epochs 5 \
  --output-dir results/my_baseline
```

## Architecture Diagram

```
Input text split into chunks: [Content1] [Content2] ... [Filler1] ... [QA]

For each chunk:
  Layers 0-13: frozen Qwen + LoRA
  Layer 14 (WriteWrapper):
    → normal layer forward
    → gate(hidden_states) → store in MemoryBank circular buffer
  Layer 15 (MemoryAugmentedAttention):
    → compute Q, K, V from current chunk
    → project stored memories through k_proj, v_proj (no RoPE)
    → concatenate: K = [mem_K | chunk_K], V = [mem_V | chunk_V]
    → standard softmax attention (model decides what to attend to)
  Layers 16-27: frozen Qwen + LoRA
  → output logits

Loss: cross-entropy on answer tokens only (after "A:" markers)
Memory: 128 tokens → 1 vector (1536-dim), 128× compression
```
