# Predictive Transformer — Full Architecture Specification

## Overview

The Predictive Transformer wraps a frozen pretrained transformer (Qwen 0.5B) with brain-inspired mechanisms. The base model's weights never change — all learning happens in the new components.

**Design principles:**
1. Everything learned, nothing hardcoded (bitter lesson)
2. New mechanisms start as no-ops — model is exactly Qwen at step 0
3. Base model = cortex (slow, stable knowledge), new mechanisms = hippocampus (fast, adaptive)
4. Architecture provides structure, learning provides policy

## What's New vs What's Qwen

```
QWEN (frozen, 494M params):          NEW MECHANISMS (trainable, 203M params):
─────────────────────────            ──────────────────────────────────────
Token embeddings                     Memory attention (2 heads per block)
24 self-attention layers             Memory gate (input-aware, per block)
24 SwiGLU FFN layers                 Memory integration FFN (per block)
RoPE position embeddings             GRU state (persistent, per block)
RMSNorm layers                       State injection (per block)
LM head                              Attention pooling (per block)
                                     Prediction head (per block)
                                     Write gate (novelty-aware, per block)
                                     Goal query projection (per block)
                                     ──────────────────────────
                                     Shared memory bank (64 slots)
                                     Halt network
                                     Value head
                                     Reward network
                                     Goal state GRU
```

## Architecture Diagram

```
Input tokens: [t₁, t₂, ..., tₙ]
    │
    ▼
┌── Token Embedding (Qwen, frozen) ─────────────────────────────────────┐
│                                                                        │
│   ┌───────────── Adaptive Settling Loop ────────────────────────┐     │
│   │  Repeat until halt network says "confident enough"           │     │
│   │  max_settle: 1→2→3→5 over training (curriculum)             │     │
│   │                                                              │     │
│   │  ┌── PredictiveBlock (×24) ──────────────────────────────┐  │     │
│   │  │                                                        │  │     │
│   │  │  ① State injection (from GRU, starts as zero)         │  │     │
│   │  │  ② Qwen self-attention + FFN (frozen, untouched)      │  │     │
│   │  │  ③ Memory attention (goal-biased queries)             │  │     │
│   │  │  ④ Memory gate (input-aware: sigmoid(W·[x, mem]))    │  │     │
│   │  │  ⑤ Memory FFN (integrate memory with understanding)   │  │     │
│   │  │  ⑥ Attention pooling → GRU state update               │  │     │
│   │  │  ⑦ Prediction head (predicts prev block's output)     │  │     │
│   │  │  ⑧ Write gate (novelty-aware gated pooling)           │  │     │
│   │  │                                                        │  │     │
│   │  └────────────────────────────────────────────────────────┘  │     │
│   │                                                              │     │
│   │  Prediction errors → Halt Network → stop or settle again    │     │
│   │  Top 2 blocks write gated-pooled vectors to memory          │     │
│   │  Prediction errors + TD error → Goal state update           │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                        │
│   RMSNorm → LM Head → logits → next token                            │
│   Prediction errors + GRU states → Value Head → state value           │
│   Error dynamics (prev + current) → Reward Network → intrinsic reward │
│                                                                        │
│   ◄──────── Shared Memory Bank (64 slots, strength-based eviction) ──►│
└────────────────────────────────────────────────────────────────────────┘
```

## Block Flow (Step by Step)

Each of the 24 PredictiveBlocks does this:

```
① State injection:
   x = x + state_proj(GRU_state)
   state_proj initialized to zeros → no-op at step 0

② Qwen layer (FROZEN):
   x = QwenLayer(x)  — self-attention + SwiGLU FFN, exact original forward pass
   This is where language understanding happens. We never touch it.

③ Memory attention (2 heads):
   queries = W_q · x + W_goal · goal_state    ← goal biases retrieval
   keys    = W_k · memory_slots
   values  = W_v · memory_slots
   mem_out = CrossAttention(queries, keys, values)
   o_proj initialized to zeros → output is zero at step 0

④ Memory gate (input-aware):
   gate = sigmoid(W · [x, mem_out])
   x = x + gate * mem_out
   W initialized to zeros → gate = 0.5, but mem_out = 0 at init → no effect

⑤ Memory FFN (SwiGLU, 2× expansion):
   x = x + MemFFN(x)
   down_proj initialized with std=0.001 → near-zero output at init

⑥ State update:
   query = learned_parameter                    ← what positions matter?
   pooled = attention(query, x, x)              ← weighted summary of sequence
   compressed = W_compress · pooled             ← project to state_dim (224)
   new_state = GRU(compressed, old_state)       ← update persistent state

⑦ Prediction:
   prediction = W · RMSNorm(x)                  ← predict prev block's output
   error = ‖actual_prev_output - prediction‖    ← feeds halt network

⑧ Write gate:
   novelty = x - mem_read                       ← how different from stored?
   write_scores = sigmoid(W · [x, novelty])     ← per-position importance
   gated = write_scores * x                     ← gate positions (differentiable)
   pooled_write = mean(gated)                   ← one vector per input
   strength = mean(write_scores)                ← low if nothing matters → evicted
```

## Settling (Adaptive Halting)

The model can run all 24 blocks **multiple times** on the same input. Between passes:
- Input embeddings reset (same starting point)
- GRU states persist (carry forward what was learned)
- Memory persists (reads from what was written in previous passes)

```
Pass 1:  embed → 24 blocks → logits₁     (states: fresh, memory: empty)
Pass 2:  embed → 24 blocks → logits₂     (states: updated, memory: has pass 1)
Pass 3:  embed → 24 blocks → logits₃     (states: refined, memory: richer)

Each pass:
  errors = prediction_errors_per_block       [e₁, ..., e₂₄]
  halt_prob = sigmoid(MLP(errors))           how confident?
  logits_final += halt_prob × logitsₙ        weighted contribution
  budget -= halt_prob                        spend confidence
  budget < 0.01 → stop

Curriculum (during training):
  Epochs 1-3:   max_settle = 1   (just learn language, no settling)
  Epochs 4-6:   max_settle = 2   (learn to use memory from pass 1)
  Epochs 7-8:   max_settle = 3   (deeper refinement)
  Epochs 9-10:  max_settle = 5   (full settling)
```

## Memory System

### Writing (gated pooling, differentiable)

Top 2 blocks write one pooled vector per input:

```
For each position in the sequence:
  write_score = sigmoid(W · [hidden, hidden - mem_read])
     ↑ novelty signal: how different is this from what's already stored?

gated_hidden = write_score × hidden        ← scale each position
pooled_vector = mean(gated_hidden)          ← single vector
strength = mean(write_scores)              ← eviction priority

If all scores ≈ 0:  pooled ≈ zero, strength ≈ 0 → evicted immediately
If some scores high: pooled focuses on those positions → persists
```

Gradients flow through the gating (differentiable). The model learns what to focus on when writing.

### Storage (64 slots, strength-based eviction)

```
Write: store pooled vector with strength score
Full?  → replace weakest slot (if new entry is stronger)
Update: TD error modulates all stored strengths
         positive δ → strengthen all entries (things are going well)
         negative δ → weaken all entries (things are going badly)
         Over time: useful memories get stronger, useless ones get evicted
```

### Reading (cross-attention, goal-biased)

All 24 blocks read from memory via cross-attention:

```
Query = W_q · hidden + W_goal · goal_state    ← what do I need? (biased by goal)
Key   = W_k · memory_slots                    ← what's stored?
Value = W_v · memory_slots                    ← stored content

Attention selects relevant memories. Gate decides how much to use:
  gate = sigmoid(W · [hidden, mem_output])
  output = gate × mem_output
```

### Multi-Chunk Training

Memory persists across consecutive chunks within a training group:

```
Group of 4 chunks from same document:
  chunk 1: "Alice works at Acme..."     → write to memory, backward, detach
  chunk 2: "She moved to Tokyo..."      → read chunk 1's memory, write more, backward, detach
  chunk 3: "Her colleague Bob..."       → read chunks 1-2, write more, backward, detach
  chunk 4: "Where does Alice work?"     → read all, answer benefits from memory

Memory and GRU states persist across chunks.
Gradients are contained per-chunk (truncated BPTT).
Memory resets between groups.
```

## Value / Reward / Goal System

These components are designed for an **agent loop** but can pre-train on self-supervised signals.

### Value Head

```
Input:  prediction_errors (24) + pooled_GRU_states (224) = 248-dim
Output: scalar value estimate

Training (Phase 1 — self-supervised):
  value_target = -lm_loss   (lower loss = higher value)
  value_loss = (V(state) - value_target)²

Training (Phase 2 — agent loop):
  TD learning: V(s) ← V(s) + α·(reward + γ·V(s') - V(s))
```

The value head learns: "what does a good internal state look like?" States with relevant memories and low prediction errors → high value.

### Reward Network

```
Input:  [prev_prediction_errors, current_prediction_errors] = 48-dim
Output: scalar intrinsic reward

Learns what constitutes "progress" — not hardcoded.
Maybe decreasing errors = good. Maybe certain blocks' errors matter more.
The network discovers it.
```

### Goal State

A slow-updating persistent vector that biases memory retrieval:

```
Input:  prediction_errors (24) + TD_error (1) = 25-dim
GRU:    goal_gru(input, old_goal) → new_goal

Update gate biased to 0.95 (changes slowly — persistent objective)
Goal biases memory queries: Q = W_q·hidden + W_goal·goal
```

Without an agent loop, the goal state tracks "what kind of text am I processing" — shifts at topic boundaries, stable within a topic. With an agent loop, it tracks "what task am I solving."

## Training Phases

### Phase 1: Self-Supervised (current)

Train on WikiText-2 with multi-chunk groups. Loss:

| Component | Weight | Signal |
|-----------|--------|--------|
| LM loss (cross-entropy) | 1.0 | Next-token prediction |
| Prediction error | 0.001 | Top-down predictions match reality |
| Ponder cost | 0.01 | Penalize excessive settling |
| Value loss | 0.01 | Value head predicts -lm_loss |

**What trains:** memory attention, write gate, prediction heads, halt network, value head, reward net, goal state. Memory FFN, state injection learn gradually.

**What doesn't train yet:** GRU state (no gradient until max_settle ≥ 2), goal-biased retrieval (goal state is noisy early), strength updates (TD signal is weak).

### Phase 2: Document QA (planned)

Synthetic QA with stored facts:
```
Passage: "Alice works at Acme Corp in Tokyo"
Question: "Where does Alice work?"
Reward: 1 if answer contains "Acme Corp", 0 otherwise
```

This gives real reward signal → TD learning → value head learns → goal state learns to focus on QA → memory strengths update based on usefulness.

### Phase 3: Agent Loop (planned)

Multi-step tasks with tool use:
```
Task → generate action → execute → observe result → reward
  ↑                                                    │
  └────── goal, memory, value guide next action ───────┘
```

External reward drives all motivation components. The architecture is ready — just needs the training loop.

## Initialization (All No-Ops)

| Component | Init | Effect at step 0 |
|-----------|------|-------------------|
| state_proj | zeros | GRU state doesn't affect output |
| mem_attn.o_proj | zeros | Memory attention outputs zero |
| mem_gate | zeros | Gate = 0.5, but mem output = 0 |
| mem_ffn.down_proj | std=0.001 | Near-zero output |
| predictor | std=0.01 | Small predictions |
| write_gate | bias=0.0 | Neutral (sigmoid(0)=0.5) |
| goal_query_proj | zeros | Goal doesn't bias queries |
| value_head | zeros | Outputs 0 |
| reward_net | zeros | Outputs 0 |
| halt_net | bias=1.0 | Biased toward halting (sigmoid(1)≈0.73) |
| goal_gru update gate | bias=3.0 | sigmoid(3)≈0.95, barely changes |

## Capabilities

| Capability | Mechanism | Trainable Now? |
|------------|-----------|---------------|
| Store facts across inputs | Memory bank + write gate | Yes (multi-chunk) |
| Retrieve relevant facts | Memory attention + gate | Yes (multi-chunk) |
| Track context over time | GRU state | Partially (needs settle ≥ 2) |
| Think harder on hard inputs | Adaptive settling | Yes (curriculum) |
| Detect surprise | Prediction heads + errors | Yes |
| Decide what to store | Write gate (novelty-aware) | Yes |
| Evaluate state quality | Value head | Yes (proxy: -lm_loss) |
| Detect progress | Reward network | Yes (error dynamics) |
| Maintain objectives | Goal state | Partially (needs agent loop) |
| Reinforce useful memories | TD-modulated strengths | Partially (weak signal) |

## Files

| File | Contents |
|------|----------|
| `rpvt/model/predictive_transformer.py` | Full model: PredictiveBlock, PredictiveTransformer, MemoryBank |
| `rpvt/experiments/exp_v3_28_qwen_wrapped.py` | Training script with multi-chunk, debug logging |
| `PREDICTIVE_TRANSFORMER.md` | This document |
| `architecture_diagram.md` | Visual diagrams |

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base model | Qwen2.5-0.5B | 24 layers, hidden=896, 14 heads |
| Frozen params | 494M | All Qwen weights |
| Trainable params | 203M | All new mechanisms |
| Memory slots | 64 | Shared bank capacity |
| Memory heads | 2 | Per-block cross-attention heads |
| State dim | 224 | GRU hidden dimension |
| Goal dim | 64 | Goal state dimension |
| Write layers | 2 | Top N blocks that write |
| Max settle | 5 | Maximum settling iterations |
| Memory FFN | 2× expansion | SwiGLU |
| Dropout | 0.1 | On memory attention and FFN |

## Brain Analogy

```
Brain                    │  Predictive Transformer
─────────────────────────┼───────────────────────────────
Cortical column          │  PredictiveBlock (×24)
Feedforward path         │  Qwen self-attention + FFN (frozen)
Feedback path            │  Prediction head (top-down)
Prediction error         │  ‖actual - predicted‖
Neural settling          │  Adaptive settling loop
Working memory (PFC)     │  GRU state (per block, dim=224)
Hippocampus              │  Shared memory bank (64 slots)
Encoding gate            │  Novelty-aware write gate
Pattern completion       │  Memory attention (retrieval)
Novelty detection        │  hidden - mem_read signal
Consolidation            │  Strength-based eviction + TD updates
Top-down attention       │  Input-aware memory gate
Dopamine / reward        │  TD error from value head
Intrinsic motivation     │  Learned reward network
Goal maintenance (PFC)   │  Goal state GRU (slow-updating)
Value judgment            │  Value head
```
