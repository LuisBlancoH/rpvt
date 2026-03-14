# RPVT — Recurrent Predictive Value Transformer

## What This Is

Research project building **architectural-level memory for LLM agents**. Not scaffolding — we modify the model's forward pass to add a differentiable external memory module that persists across chunks of text. The model processes text in chunks, with memory as the ONLY cross-chunk information channel.

**Broader vision**: Agents that can plan, learn, evaluate, and have goals. Memory (M) is the first building block. Five minimal mechanisms: prediction, error signal, persistent goal state, multiple timescales, recurrence.

**Philosophy**: Work at architecture level (not scaffolding), take brain *principles* not implementation details, bitter lesson (make mechanisms learnable, don't hard-code), verify foundation before building more.

**Owner**: Luis Blanco, SWE at Google, independent ML researcher.

## Current Working Architecture (v3.2 cross-attention)

**This is the architecture that works. Previous additive injection approaches all failed.**

- **Base model**: Qwen2.5-3B (frozen, bf16) — can also use 1.5B for faster iteration on 3080 Ti
- **LoRA**: rank 16 on q_proj, v_proj (3.7M params)
- **MemoryBank** (`rpvt/model/cross_attention_memory.py`): Stores gated hidden states
  - W_gate: learned write gate (bias=-2.0) — controls what gets stored
  - Circular buffer of n_slots=64 hidden state vectors (raw, no projection)
  - Only 2,049 trainable params (just the gate)
- **WriteWrapper**: Wraps layer 18 — passes output through unchanged, then writes gated hidden states to MemoryBank
- **MemoryAugmentedAttention**: Replaces layer 19's self_attn — concatenates memory hidden states as extra KV pairs before attention computation
  - Projects memory through layer 19's own k_proj/v_proj (with LoRA)
  - No RoPE on memory KVs (content-based, position-independent retrieval)
  - Model's own attention mechanism decides whether to attend to memory
- **Per-chunk processing**: Each chunk is a separate `model()` call. Memory is the ONLY cross-chunk information channel.

**Result**: 97.6% token accuracy, 79% exact match on 1.5B (vs 15.9% no-memory, 10.8% untrained). 42.5% on 3B at 5 epochs. First working memory on a pretrained model.

## Key Files

- `rpvt/model/cross_attention_memory.py` — MemoryBank, WriteWrapper, MemoryAugmentedAttention (THE WORKING APPROACH)
- `rpvt/model/hopfield_memory.py` — HopfieldMemory module (additive injection — DOES NOT WORK on pretrained models)
- `rpvt/experiments/exp_v3_2_nlp_recall.py` — Current NLP recall experiment (SQuAD + synthetic data)
- `rpvt/experiments/exp_v3_1_pretrained_recall.py` — Has build_model(), MemoryWrapper, reset_memories, etc.
- `results/FINDINGS.md` — Comprehensive experiment findings (23 key discoveries)

## What Works

1. **Cross-attention memory injection**: Memory as extra KV pairs in attention → 97.6% token acc / 79% exact match on 1.5B (15 ep), 42.5% on 3B (5 ep)
2. **1.5B model outperforms 3B**: Shorter gradient path + higher LoRA influence ratio → faster learning
3. **Write gating**: Gate reliably discriminates passage from filler (2.6x ratio on NL text)
4. **Per-chunk processing**: Forces genuine memory dependence (LoRA-only at chance: 10.8%)
5. **Answer-only loss**: Loss computed only on answer tokens after "A:" markers — no dilution
6. **Synthetic facts with truly unique entities**: Random names, codes, cities prevent template memorization

## What Does NOT Work (Critical Lessons)

1. **Additive residual injection (W_out → residual stream)**: Pretrained models ignore/suppress the signal. Even with full LoRA (30M params on all linear layers), the model actively learns to suppress memory. The residual stream has a tightly learned distribution — foreign signals are noise.
2. **Increasing memory_size**: 256, 1024, 2048 all give identical results — not a capacity problem
3. **Learned extraction queries (n_extract=4)**: Multi-slot writes don't help when injection is broken. Should be revisited with cross-attention.
4. **SQuAD as benchmark**: Parametric knowledge dominates (55.5% baseline). Use synthetic facts.
5. **Small entity pools**: LoRA memorizes template mappings. Use truly random names/codes/cities.

## Experiment History (Condensed)

### v2.6: Small model from scratch
- Proved Hopfield retrieval works for 1 pair (100% with loss weighting)
- Scales poorly to 4+ pairs (25.8% best with two-phase training at 40 epochs)
- Key insight: two-phase training (freeze transformer, train memory first) solves gate learning
- Additive injection works when training from scratch (model learns WITH memory)

### v3.1: Pretrained model + synthetic tokens
- Per-chunk processing works — memory is genuinely the only information channel
- Gate learns perfectly (1.8 billion x store/filler ratio)
- But retrieval fails — W_query/W_key alignment doesn't learn

### v3.2: Pretrained model + NLP recall
- **SQuAD**: contaminated by parametric knowledge (55.5% baseline)
- **Synthetic v1** (small pools): template memorization (77.4% no-memory = 77.3% with memory)
- **Synthetic v2** (truly unique facts): proper benchmark (10.2% baseline)
- **Attention supervision**: proved attention already 100% on passage slots — never the bottleneck
- **Memory_size sweep**: 256=1024=2048, all ~14.8% — value dimensionality not the issue
- **No-memory trained control**: 15.5% — proved memory contributes zero with additive injection
- **Full LoRA (30M params)**: model SUPPRESSES memory (attn→0%)
- **BREAKTHROUGH: Cross-attention injection**: 42.5% on 3B at epoch 5, **97.6% on 1.5B at epoch 15** (29.5→56.8→75.2→88.7→92.7→...→97.6)

## Synthetic Data Task (Current Benchmark)

Generate passages with truly unique facts (random names, random 3-digit codes, random cities). Model sees passage in early chunks, filler chunks (WikiText), then QA chunk. Must answer from memory.

```
Passage: "Balcorfen Torushvel was a researcher at the Criho Institute in Xilob.
Their work on gomilu dynamics produced result code 847. The project started in
1923 and generated output code 412..."

Q: What was Balcorfen Torushvel's result code? A: 847
Q: Where did Balcorfen Torushvel work? A: Xilob
Q: When did Balcorfen Torushvel's project start? A: 1923
```

Baseline (no memory, per-chunk): 10.8% (1.5B). No-memory trained LoRA: 15.9%. Cross-attention memory: 97.6% token / 79% exact match (1.5B, 15 ep).

## Next Steps (Priority Order)

1. ~~**More training**~~: DONE — ceiling is ~97.6% token / 79% exact match at epoch 10-15
2. **Learned extraction + cross-attention**: Combine n_extract queries with cross-attention (--n-extract 4). Attacks remaining errors on novel subword tokens by storing multiple aspects per chunk instead of mean-pooling
3. **Scaling tests**: More QA pairs per passage (5-6), longer gaps (8-12), harder questions
4. **Multi-passage recall**: Two separate passages, questions about both — tests memory interference
5. **Attention analysis**: What do the cross-attention patterns look like? Does the model attend to specific memory slots for specific questions?
6. **Multi-hop reasoning**: Questions that require combining facts from multiple passages
7. **Hierarchical compression**: Compress chunks, then compress compressed chunks — same extraction mechanism at multiple timescales

## Environment

- Python 3.12, venv at `.venv/` — activate with `source .venv/bin/activate` or use `.venv/bin/python3`
- Packages: PyTorch 2.6+cu124, transformers 5.3.0, peft 0.18.1, datasets 4.7.0
- GPU: RTX 3080 Ti (~12.9GB VRAM), bf16 supported, WSL2
- Use Qwen2.5-1.5B for experiments (3B needs 24GB+)
- Always push to GitHub after code changes
- Update `results/FINDINGS.md` with experiment outcomes
- Use `--memory-mode cross_attn` flag for the working architecture
- Use `--data-source synthetic` for the proper benchmark (not squad)

## Running Experiments

```bash
# Cross-attention memory (THE WORKING APPROACH)
python -m rpvt.experiments.exp_v3_2_nlp_recall \
  --data-source synthetic \
  --memory-mode cross_attn \
  --epochs 5 \
  --n-train 500 \
  --n-eval 100 \
  --output-dir results/exp_v3_2_cross_attn

# No-memory baseline (for comparison)
python -m rpvt.experiments.exp_v3_2_nlp_recall \
  --data-source synthetic \
  --no-memory \
  --epochs 5 \
  --output-dir results/exp_v3_2_no_memory

# For 1.5B model (fits on 3080 Ti 12GB)
python -m rpvt.experiments.exp_v3_2_nlp_recall \
  --model-name Qwen/Qwen2.5-1.5B \
  --data-source synthetic \
  --memory-mode cross_attn \
  --epochs 5 \
  --output-dir results/exp_v3_2_cross_attn_1.5b
```

## Architecture Diagram

```
Input text split into chunks: [Passage] [Filler1] [Filler2] ... [QA]

For each chunk:
  Layers 0-17: frozen Qwen + LoRA
  Layer 18 (WriteWrapper):
    → normal layer forward
    → gate(hidden_states) → store in MemoryBank circular buffer
  Layer 19 (MemoryAugmentedAttention):
    → compute Q, K, V from current chunk
    → project stored memories through k_proj, v_proj (no RoPE)
    → concatenate: K = [mem_K | chunk_K], V = [mem_V | chunk_V]
    → standard softmax attention (model decides what to attend to)
  Layers 20-35: frozen Qwen + LoRA
  → output logits

Loss: cross-entropy on answer tokens only (after "A:" markers)
```
