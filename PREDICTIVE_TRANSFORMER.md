# Predictive Transformer — Full Architecture Specification

## Overview

The Predictive Transformer is a modified transformer architecture designed for brain-like processing. It keeps everything that works about transformers (attention, feedforward layers, residual connections, scaling) and adds five mechanisms the standard transformer lacks:

1. **Recurrent state** — persistent memory across inputs (GRU per block)
2. **Shared memory bank** — read/write episodic memory (all blocks read, top blocks write)
3. **Predictive coding** — each block predicts the previous block's output (top-down)
4. **Adaptive settling** — variable compute depth based on learned confidence
5. **Novelty-aware gating** — learned write decisions, no hardcoded thresholds

All decisions are learned. Nothing is hardcoded.

## Architecture Diagram

```
Input tokens: [t₁, t₂, ..., tₙ]
    │
    ▼
┌── Token Embedding + Position Embedding ──────────────────────────────┐
│   x = Embed(tokens) + PosEmbed(positions)                           │
│   Shape: (batch, seq_len, hidden_size)                              │
│                                                                      │
│   ┌─────────────── Settling Loop ──────────────────────────────┐    │
│   │  Repeat until confident (adaptive) or N times (fixed)       │    │
│   │                                                             │    │
│   │  ┌── PredictiveBlock 1 ──────────────────────────────┐     │    │
│   │  │                                                    │     │    │
│   │  │  state₁ ──► Attention Pool ──► GRU ──► new state₁ │     │    │
│   │  │               │                                    │     │    │
│   │  │  x + state_context                                 │     │    │
│   │  │       │                                            │     │    │
│   │  │       ├── Self-Attention (8 heads) ──┐            │     │    │
│   │  │       │   (attends to input tokens)  │            │     │    │
│   │  │       │                              │            │     │    │
│   │  │       └── Memory-Attention (4 heads)─┤            │     │    │
│   │  │           (attends to memory bank)   │            │     │    │
│   │  │                                      ▼            │     │    │
│   │  │                                 ── FFN ──         │     │    │
│   │  │                                      │            │     │    │
│   │  │                    ┌─────────────────┤            │     │    │
│   │  │                    │         │       │            │     │    │
│   │  │              Predict      Write    output₁       │     │    │
│   │  │           (block 0's    (novelty-                │     │    │
│   │  │            output)       aware)                   │     │    │
│   │  └────────────────────────────────────────────────────┘     │    │
│   │       │                                                     │    │
│   │       ▼                                                     │    │
│   │  ┌── PredictiveBlock 2 ... Block 6 (same structure) ──┐   │    │
│   │  └────────────────────────────────────────────────────┘   │    │
│   │       │                                                     │    │
│   │  Compute prediction errors per block                        │    │
│   │  errors = [‖actual_i - predicted_i‖ for each block]        │    │
│   │       │                                                     │    │
│   │  ┌── Halt Network ─────────────────────────────────────┐   │    │
│   │  │  halt_prob = sigmoid(MLP(errors))                    │   │    │
│   │  │  if confident enough → stop settling                 │   │    │
│   │  │  else → reset x to embeddings, keep states, repeat   │   │    │
│   │  └──────────────────────────────────────────────────────┘   │    │
│   │                                                             │    │
│   │  Output logits accumulated with halt-weighted average       │    │
│   └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│   LayerNorm → LM Head → logits                                      │
│                                                                      │
│   ◄──────────── Shared Memory Bank (64 slots) ──────────────────►   │
│   Written by top 2 blocks only. Read by all blocks.                  │
│   Learned eviction: weakest slot replaced when full.                 │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Token + Position Embeddings (Standard)

```python
x = TokenEmbed(input_ids) + PosEmbed(positions)
# TokenEmbed: (vocab_size, hidden_size) — learned lookup table
# PosEmbed: (max_seq_len, hidden_size) — absolute position encoding
# Output: (batch, seq_len, hidden_size)
```

Each token is mapped to a dense vector. Position information is added so the model knows token order. Token and LM head weights are tied (same matrix).

### 2. Self-Attention (8 heads, standard transformer)

```
For each position i in the sequence:
    Q_i = W_Q · x_i     (what am I looking for?)
    K_j = W_K · x_j     (what does position j contain?)
    V_j = W_V · x_j     (what information does position j carry?)

    score_ij = Q_i · K_j / √d_head    (how relevant is j to i?)
    weights = softmax(scores)           (normalize across all j)
    output_i = Σ weights_ij · V_j       (weighted sum)
```

Each of the 8 heads learns a different "type of relevance." Head 1 might learn subject-verb agreement, head 2 might learn coreference, etc. A causal mask prevents attending to future positions (for autoregressive generation).

**Shape flow:**
- Input: `(batch, seq_len, hidden_size)`
- Per head: `(batch, seq_len, head_dim)` where `head_dim = hidden_size / n_heads`
- Output (all heads concatenated): `(batch, seq_len, hidden_size)`

### 3. Memory Attention (4 heads, reads from shared bank)

```
For each position i in the input:
    Q_i = W_Q_mem · x_i          (what am I looking for in memory?)
    K_m = W_K_mem · memory_m     (what does each memory slot contain?)
    V_m = W_V_mem · memory_m     (what information does each slot carry?)

    score_im = Q_i · K_m / √d    (how relevant is memory slot m?)
    weights = softmax(scores)     (over all memory slots)
    output_i = Σ weights_im · V_m (retrieve relevant memories)
```

Same mechanism as self-attention, but the keys and values come from the memory bank instead of the current input. Each position in the input queries all memory slots and retrieves what's relevant.

**Gated addition** (starts as no-op, learns to open):
```python
mem_contribution = sigmoid(W_gate · mem_output) * mem_output
x = x + mem_contribution
```

The gate starts at zero (sigmoid(0) = 0.5 × zero-init weights ≈ 0) so memory doesn't disrupt learning early on. As training progresses, the model gradually learns to rely on memory.

### 4. Feedforward Network (per-position computation)

```python
ffn(x) = W_2 · SiLU(W_1 · x)
# W_1: (hidden_size, 4 × hidden_size) — expand
# SiLU: smooth nonlinearity (x · sigmoid(x))
# W_2: (4 × hidden_size, hidden_size) — contract
```

Attention routes information between positions. FFN transforms information at each position independently. The 4x expansion creates a larger space for computation — the network can represent more complex transformations in the expanded space before compressing back down.

### 5. GRU State (recurrent memory per block)

Each block maintains a persistent state vector that carries information across inputs and settling steps.

**Attention pooling** (instead of mean pooling):
```python
# Learned query asks "what's most important in this sequence?"
query = learned_parameter  # (1, 1, hidden_size)
pooled = MultiHeadAttention(query, x, x)  # (batch, 1, hidden_size)
# The model learns to focus on task-relevant positions
```

**GRU update:**
```python
state_input = Linear(pooled)  # compress to state_dim (256)

# GRU gates:
reset  = sigmoid(W_r · [state_input, old_state])   # what to forget
update = sigmoid(W_u · [state_input, old_state])   # how much to change
candidate = tanh(W_c · [state_input, reset * old_state])  # new content
new_state = update * old_state + (1 - update) * candidate
```

The reset gate controls forgetting (reset=0 → ignore old state). The update gate controls integration (update=1 → keep old state unchanged, update=0 → fully replace with new content). The model learns this tradeoff through training.

**State injection** (adds context to input):
```python
state_context = W_proj · state  # (batch, hidden_size)
x = x + state_context.unsqueeze(1)  # broadcast across sequence
```

Initialized to zero so state doesn't affect output initially. The model gradually learns to use state information.

**Brain analog:** Working memory in the prefrontal cortex. A compressed, continuously-updated representation of "what I know so far." Unlike attention (which recomputes everything each time), the GRU accumulates information over time.

### 6. Prediction Head (top-down, predictive coding)

```python
prediction = LayerNorm(x) → Linear(hidden_size, hidden_size)
# Block i predicts what block (i-1) output
# Error = actual_(i-1) - prediction_i

error = ‖actual_hidden_states[i-1] - prediction_i‖
```

Each block generates a prediction of what the block below it should have output. This is the **top-down** signal — higher layers telling lower layers "this is what I expect you to produce."

**Training loss:** The prediction error is an auxiliary loss (weighted small: 0.001). The model is rewarded for making accurate top-down predictions, but the primary objective is language modeling.

**During settling:** Predictions improve across passes. Pass 1 has rough predictions (high error). Pass 2, with updated GRU states, has more accurate predictions (lower error). When predictions match reality (low error), the model has "settled" on a consistent interpretation.

**Brain analog:** Predictive coding in the cortex. Higher areas (e.g., IT cortex) send predictions to lower areas (e.g., V1). Prediction errors propagate upward. The brain "settles" when predictions match sensory input.

### 7. Memory Write Gate (novelty-aware, learned)

```python
mem_read = memory_attention_output   # what memory already knows
novelty = hidden_state - mem_read    # what's new here
write_input = concat(hidden_state, novelty)  # (hidden * 2)
write_strength = sigmoid(Linear(write_input) + bias)
# bias = -2 → sigmoid(-2) ≈ 0.12, mostly closed initially
```

The write gate sees two things:
1. **The hidden state** — what information is at this position
2. **The novelty** — how different this is from what memory already contains

If memory already has this information → `novelty ≈ 0` → low write strength.
If this is genuinely new → `novelty` is large → high write strength.

The model learns through training what combinations of content and novelty warrant storage. Different types of content naturally get different effective thresholds.

**Only the top 2 blocks write** — they have the most abstract, semantic-level representations. Lower blocks have token-level features that aren't useful to store.

**Brain analog:** Hippocampal encoding. The hippocampus checks for pattern completion (recall) before encoding. If recall succeeds (already known), encoding is suppressed. If recall fails (novel), encoding is triggered.

### 8. Memory Bank (shared, learned eviction)

```python
class MemoryBank:
    slots: (n_slots, hidden_size)      # stored vectors
    strength: (n_slots,)               # importance of each slot [0, 1]
```

**Read:** All blocks read via their memory attention heads. Each block extracts what it needs through learned attention weights.

**Write:** When a new vector is written:
- If free slots exist → use the next free slot
- If full → compare new vector's strength to all existing slots
- If new strength > weakest existing → evict weakest, store new
- If new strength ≤ weakest → don't store (not important enough)

**Brain analog:** Hippocampal memory with consolidation. Important memories (high strength) persist. Less important ones get overwritten by newer, more important content.

### 9. Adaptive Settling (learned halting)

Instead of a fixed number of forward passes, the model decides when to stop:

```python
for step in range(max_settle):
    # Forward through all blocks
    # Compute prediction errors per block
    errors = [e₁, e₂, ..., e₆]

    # Halt network: should we stop?
    halt_prob = sigmoid(MLP(errors))

    # Weighted accumulation (Adaptive Computation Time)
    logits += halt_prob * step_logits
    remaining_budget -= halt_prob

    if remaining_budget < 0.01:
        break  # confident enough, stop

# Ponder cost: penalty for using more steps
loss += 0.01 * (steps_used / max_settle)
```

**Easy inputs** (low prediction error) → halt early (1-2 passes) → fast.
**Hard inputs** (high prediction error) → more passes → more compute.

The halt network learns from the ponder cost: using more steps is penalized, so the model learns to settle as quickly as possible while still maintaining accuracy.

**Brain analog:** Neural settling in cortical circuits. Simple stimuli are recognized quickly. Ambiguous stimuli trigger more recurrent processing. The brain adaptively allocates compute.

## Training

**Three losses combined:**

1. **Language modeling loss** (primary) — standard next-token prediction cross-entropy
2. **Prediction error loss** (auxiliary, weight=0.001) — encourages accurate top-down predictions, normalized by hidden_size to keep scale manageable
3. **Ponder cost** (regularizer, weight=0.01) — penalizes using more settling steps, encourages efficient computation

**Settling schedule during training:**
- 2/3 of batches: `n_settle=1` (standard forward pass)
- 1/3 of batches: `n_settle=2` (settling, so the model learns to use it)

## Initialization

All new mechanisms initialize as **no-ops** so they don't interfere with base transformer learning:

| Component | Initialization | Effect at step 0 |
|---|---|---|
| `state_proj` | zeros | State doesn't affect hidden states |
| `mem_gate` | zeros | Memory attention contributes nothing |
| `predictor` | std=0.01 | Predictions are near-zero |
| `write_gate.bias` | -2.0 | sigmoid(-2)≈0.12, most positions don't write |
| `halt_net.bias` | +1.0 | Biased toward halting early (1 pass) |

This means the model starts as a standard transformer and gradually "discovers" how to use state, memory, prediction, and settling as training progresses.

## Key Design Decisions

### Why attention pooling for GRU (not mean pooling)?
Mean pooling gives equal weight to every position. Attention pooling lets the model learn which positions carry the most important information for the persistent state. A filler word shouldn't contribute as much as a key fact.

### Why only top blocks write to memory?
Lower blocks produce token-level features (character patterns, subword combinations). Upper blocks produce semantic-level representations (facts, relationships, concepts). Memory should store meaning, not surface features.

### Why shared memory (not per-block)?
The brain's hippocampus is shared across all cortical areas. Each area reads from it through its own projections. Similarly, each block's memory attention heads learn to extract what that block needs from the shared semantic store.

### Why prediction goes backward (block i → block i-1)?
This is top-down prediction — higher levels predict lower levels. In the brain, higher cortical areas send predictions to lower areas. The mismatch (prediction error) is what carries information upward. Only what's unexpected needs to propagate — expected information is "explained away" by the prediction.

### Why adaptive settling (not fixed N passes)?
The brain doesn't process every stimulus for the same duration. Recognition of familiar patterns is fast (single feedforward sweep). Ambiguous or novel inputs trigger recurrent processing. The model should allocate compute proportionally to difficulty.

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `hidden_size` | 768 | Hidden dimension per position |
| `n_layers` | 6 | Number of PredictiveBlocks |
| `n_self_heads` | 8 | Self-attention heads per block |
| `n_mem_heads` | 4 | Memory attention heads per block |
| `state_dim` | 256 | GRU state dimension |
| `n_memory_slots` | 64 | Shared memory bank capacity |
| `n_write_layers` | 2 | How many top blocks write to memory |
| `max_settle` | 5 | Maximum settling iterations |
| `ffn_mult` | 4 | FFN expansion ratio |
| `dropout` | 0.1 | Dropout rate |
| `max_seq_len` | 512 | Maximum sequence length |

## Files

- `rpvt/model/predictive_transformer.py` — full model implementation
- `rpvt/experiments/exp_v3_26_train_predictive.py` — training on WikiText-2

## Future: Qwen Weight Initialization

The architecture is designed to be compatible with existing transformer weights:
- Self-attention, FFN, embeddings, LM head → copy from Qwen
- Memory attention, GRU, prediction, write gate, halt network → initialize as no-ops

A Qwen-initialized Predictive Transformer would behave identically to Qwen at step 0, then gradually learn to use the new mechanisms through continued training.
