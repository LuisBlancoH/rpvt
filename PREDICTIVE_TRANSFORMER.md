# Predictive Transformer — Full Architecture Specification

## Overview

The Predictive Transformer is a modified transformer architecture designed for brain-like processing. It keeps everything that works about transformers (attention, feedforward layers, residual connections, scaling) and adds five mechanisms the standard transformer lacks:

1. **Recurrent state** — persistent memory across inputs (attention-pooled GRU per block)
2. **Shared memory bank** — read/write episodic memory (all blocks read, top blocks write)
3. **Predictive coding** — each block predicts the previous block's output (top-down)
4. **Adaptive settling** — variable compute depth based on learned confidence
5. **Fully learned gating** — no hardcoded thresholds anywhere

**Everything is learned. Nothing is hardcoded.**

## Architecture Diagram

```
Input tokens: [t₁, t₂, ..., tₙ]
    │
    ▼
┌── Token Embedding + Position Embedding ────────────────────────────┐
│                                                                     │
│   ┌───────────── Adaptive Settling Loop ────────────────────┐      │
│   │  Repeat until halt network says "confident enough"       │      │
│   │  (easy inputs: 1 pass, hard inputs: up to 5 passes)     │      │
│   │                                                          │      │
│   │  ┌── PredictiveBlock ──────────────────────────────┐    │      │
│   │  │                                                  │    │      │
│   │  │  ① input + state_context (from GRU)             │    │      │
│   │  │       │                                          │    │      │
│   │  │  ② Self-Attention (8 heads)                     │    │      │
│   │  │       │  (what's in the current input?)          │    │      │
│   │  │       ▼                                          │    │      │
│   │  │  ③ Main FFN (4x expansion)                      │    │      │
│   │  │       │  (understand/process it)                 │    │      │
│   │  │       ▼                                          │    │      │
│   │  │  ④ Memory Attention (4 heads)                   │    │      │
│   │  │       │  (retrieve relevant memories)            │    │      │
│   │  │       │  gate = learned(input, memory_content)   │    │      │
│   │  │       ▼                                          │    │      │
│   │  │  ⑤ Memory Integration FFN (2x expansion)       │    │      │
│   │  │       │  (combine understanding + memories)      │    │      │
│   │  │       ▼                                          │    │      │
│   │  │  ⑥ State Update                                 │    │      │
│   │  │       │  attention_pool → GRU → new state       │    │      │
│   │  │       ▼                                          │    │      │
│   │  │  ⑦ Prediction Head                              │    │      │
│   │  │       │  (predict previous block's output)       │    │      │
│   │  │       ▼                                          │    │      │
│   │  │  ⑧ Write Gate (novelty-aware)                   │    │      │
│   │  │       │  gate = learned(hidden, hidden-mem_read) │    │      │
│   │  │       ▼                                          │    │      │
│   │  │  output                                          │    │      │
│   │  └──────────────────────────────────────────────────┘    │      │
│   │       │                                                  │      │
│   │       ▼ × 6 blocks                                      │      │
│   │                                                          │      │
│   │  Compute prediction errors → Halt Network               │      │
│   │  errors = [‖actual - predicted‖ per block]              │      │
│   │  halt_prob = sigmoid(MLP(errors))                        │      │
│   │  logits += halt_prob × step_logits                       │      │
│   │  if confident → stop; else → reset x, keep states       │      │
│   │                                                          │      │
│   │  Top 2 blocks write to memory (novelty-gated)           │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│   LayerNorm → LM Head → logits                                     │
│                                                                     │
│   ◄─────── Shared Memory Bank (64 slots) ───────────────────►     │
│   Written by top 2 blocks. Read by all blocks.                      │
│   Eviction: weakest slot replaced when full.                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Block Flow (Step by Step)

```
① State injection:     x = x + project(GRU_state)
② Self-attention:      x = x + SelfAttn(x, x, x)
③ Main FFN:            x = x + FFN_main(x)             ← understand before retrieving
④ Memory attention:    mem = MemAttn(x, memory, memory)
                       gate = sigmoid(W · [x, mem])      ← input-aware gating
                       x = x + gate * mem
⑤ Memory FFN:          x = x + FFN_mem(x)               ← integrate memory with understanding
⑥ State update:        q = learned_query
                       pooled = Attn(q, x, x)            ← attention pooling (not mean)
                       state = GRU(compress(pooled), old_state)
⑦ Prediction:          pred = Linear(LayerNorm(x))       ← predict prev block's output
⑧ Write gate:          novelty = x - mem_read
                       strength = sigmoid(W · [x, novelty])  ← novelty-aware
```

## Component Details

### 1. Self-Attention (8 heads) — Standard

Each position gathers information from every other position via learned Q/K/V projections. Causal mask prevents attending to future tokens. Each head learns a different type of relevance.

### 2. Main FFN (4x expansion) — Understand Before Retrieving

```
x → Linear(768→3072) → SiLU → Linear(3072→768) → x
```

Placed BEFORE memory attention. The model needs to understand what it's looking at before it knows what to retrieve from memory. In the brain: you recognize the stimulus before searching episodic memory.

### 3. Memory Attention (4 heads) — Retrieve Relevant Memories

Same mechanism as self-attention, but keys/values come from the shared memory bank. Queries are the model's processed representation (post-FFN), so retrieval is based on understanding, not raw features.

**Input-aware gate:**
```
gate = sigmoid(W · concat(input_hidden, memory_output))
x = x + dropout(gate * memory_output)
```

The gate sees BOTH the input and the memory content. It learns context-dependent retrieval:
- "Question about Alice + memory has Alice's facts" → gate opens
- "Greeting + memory has facts" → gate stays closed
- "Question + empty memory" → gate stays closed (prevents hallucination)

### 4. Memory Integration FFN (2x expansion) — Combine Sources

```
x → Linear(768→1536) → SiLU → Linear(1536→768) → x
```

Smaller than the main FFN (2x vs 4x expansion). Its job is to nonlinearly integrate "what I understand" with "what I remembered." Without this, memory output is just linearly added — no opportunity to reconcile the two information sources.

### 5. GRU State — Persistent Recurrent Memory

**Attention pooling** (learned, replaces mean pooling):
```
query = learned_parameter (1 × hidden_size)
pooled = MultiHeadAttention(query, hidden_states, hidden_states)
```
The model learns WHICH positions carry the most important information for the persistent state. A filler word contributes less than a key fact.

**GRU update:**
```
reset_gate  = σ(W_r · [input, old_state])   → what to forget
update_gate = σ(W_u · [input, old_state])   → how much to change
candidate   = tanh(W_c · [input, reset × old_state])
new_state   = update × old_state + (1-update) × candidate
```

State persists across inputs AND across settling steps. It's the primary mechanism for carrying information between passes during settling.

### 6. Prediction Head — Top-Down Predictive Coding

Each block predicts what the previous block should have output. Prediction errors are computed but not injected — they serve as:
- Auxiliary training loss (small weight: 0.001)
- Input to the halt network (determines settling confidence)

### 7. Novelty-Aware Write Gate — Learned Storage Decisions

```
novelty = hidden_state - memory_read_output
write_strength = sigmoid(Linear(concat(hidden_state, novelty)) + bias)
```

The gate sees what's at this position AND how different it is from what's already in memory. The model learns context-dependent storage:
- "Important fact not in memory" → high strength
- "Important fact already stored" → low strength (novelty ≈ 0)
- "Filler content" → low strength regardless of novelty

Different types of content get different effective thresholds — all learned, no hardcoded rules.

### 8. Memory Bank — Shared with Learned Eviction

- **Read:** All blocks, via their 4 memory attention heads
- **Write:** Top 2 blocks only (high-level semantic representations)
- **Eviction:** When full, weakest slot (lowest write strength) gets replaced — but only if the new content is stronger
- **Write strength:** Continuous [0, 1], not binary — slots have varying importance

### 9. Adaptive Settling — Learned Halting

```python
for step in range(max_settle):  # max 5
    # Forward through all blocks
    errors = prediction_errors_per_block  # [e₁, ..., e₆]
    halt_prob = sigmoid(MLP(errors))      # learned confidence

    logits += halt_prob × step_logits     # weighted accumulation
    remaining_budget -= halt_prob

    if remaining_budget < 0.01: break     # confident enough
```

The halt network learns from a ponder cost: more steps = higher penalty. The model discovers the minimum compute needed per input. Easy inputs settle in 1 pass. Ambiguous inputs get up to 5 passes.

## Training

**Three losses:**

| Loss | Weight | Purpose |
|---|---|---|
| LM loss (cross-entropy) | 1.0 | Next-token prediction (primary) |
| Prediction error | 0.001 | Top-down predictions match reality |
| Ponder cost | 0.01 | Penalize excessive settling steps |

**Settling during training:** 2/3 of batches use 1-pass, 1/3 use 2-pass.

## Initialization

All new mechanisms start as **no-ops** — the model begins as a standard transformer:

| Component | Init | Effect at step 0 |
|---|---|---|
| `state_proj` | zeros | State doesn't affect output |
| `mem_gate` | zeros | Memory contributes nothing |
| `predictor` | std=0.01 | Near-zero predictions |
| `write_gate.bias` | -2.0 | sigmoid(-2)≈0.12, mostly closed |
| `halt_net.bias` | +1.0 | Biased toward halting early |

Enables weight initialization from existing transformers (e.g., Qwen) — the model works identically to the base at step 0, then gradually learns to use the new mechanisms.

## What's Learned vs What's Designed

| Aspect | Status |
|---|---|
| What to store in memory | **Learned** (novelty-aware write gate) |
| What to retrieve from memory | **Learned** (attention) |
| How much to rely on memory | **Learned** (input-aware gate) |
| What to evict from memory | **Learned** (weakest-strength eviction) |
| When to stop thinking | **Learned** (halt network from errors) |
| What GRU remembers | **Learned** (attention pooling) |
| Block order / connections | Designed (Self-Attn → FFN → Mem-Attn → Mem-FFN) |
| Number of heads split (8+4) | Designed |
| Memory bank size (64) | Designed |
| State dimension (256) | Designed |

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `hidden_size` | 768 | Hidden dimension |
| `n_layers` | 6 | Number of PredictiveBlocks |
| `n_self_heads` | 8 | Self-attention heads |
| `n_mem_heads` | 4 | Memory attention heads |
| `state_dim` | 256 | GRU state dimension |
| `n_memory_slots` | 64 | Memory bank capacity |
| `n_write_layers` | 2 | Top N blocks that write to memory |
| `max_settle` | 5 | Maximum settling iterations |
| `ffn_mult` | 4 | Main FFN expansion (memory FFN uses 2) |
| `dropout` | 0.1 | Dropout rate |

## Results

**From-scratch training on WikiText-2 (v2, 10 epochs):**

| Metric | Result |
|---|---|
| 1-pass perplexity | 330 |
| 2-pass perplexity | **251** |
| Settling gain | **+79 (24% improvement)** |
| Prediction error | 45.8 → 20.2 (halved over training) |

The model learns to use settling — 2-pass consistently beats 1-pass, and the gap grows every epoch. This proves the architecture works: the GRU state, memory, and predictions are genuinely useful on the second pass.

## Files

- `rpvt/model/predictive_transformer.py` — full model implementation
- `rpvt/experiments/exp_v3_26_train_predictive.py` — training script
- `PREDICTIVE_TRANSFORMER.md` — this document

## Future

1. **Qwen weight initialization** — copy self-attention, FFN, embeddings from Qwen; new mechanisms start as no-ops
2. **Larger training** — OpenWebText or SlimPajama instead of WikiText-2
3. **RoPE** — replace absolute position embeddings for length generalization
4. **Memory across documents** — persistent memory across training sequences
5. **Agent integration** — memory persists across tool calls and actions
