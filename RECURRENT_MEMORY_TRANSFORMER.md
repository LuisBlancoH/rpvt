# Recurrent Memory Transformer

## What It Is

A frozen Qwen 0.5B transformer wrapped with memory + recurrence. The transformer adapts via LoRA on all layers. Two mechanisms do everything: **attention** (retrieval) and **FFN** (representation).

23.5M trainable params. 494M frozen. Fits on a 3080 Ti in 1.1GB.

## Architecture

```
Input tokens
    │
    ▼
┌── Embed (Qwen, frozen) ──────────────────────────────────────┐
│                                                               │
│   ┌─── Settling Loop (model decides when to stop) ────────┐  │
│   │                                                        │  │
│   │  Qwen Layer 1 (frozen + LoRA on all linear layers)     │  │
│   │    Self-Attention → FFN                                │  │
│   │    + memory cross-attention (2 heads, gated)           │  │
│   │                                                        │  │
│   │  Qwen Layer 2 ... same                                 │  │
│   │  ...                                                   │  │
│   │  Qwen Layer 24 ... same                                │  │
│   │       │                                                │  │
│   │       ▼                                                │  │
│   │  Memory Extraction                                     │  │
│   │    16 learned queries attend to output                 │  │
│   │    → FFN transforms → 16 memory vectors               │  │
│   │    → priority scores (learned importance)              │  │
│   │    → confidence score (halt signal)                    │  │
│   │       │                                                │  │
│   │       ├── confident → stop, output logits              │  │
│   │       └── not confident → loop (same input, new memory)│  │
│   │                                                        │  │
│   └── Only final extraction stored in persistent buffer ───┘  │
│                                                               │
│   LM Head → next token                                        │
│                                                               │
│   ◄── Memory Buffer (256 slots, priority-based eviction) ──►  │
└───────────────────────────────────────────────────────────────┘
```

## Components

### 1. Qwen 0.5B (frozen base, 494M params)
24 transformer layers. Self-attention + SwiGLU FFN. RoPE positions. Provides all language understanding. Never modified directly.

### 2. LoRA on all linear layers (4.4M params)
Rank 8 adaptation on every linear layer in every block:
- **Attention**: q_proj, k_proj, v_proj, o_proj — learn to attend to memory vectors
- **FFN**: gate_proj, up_proj, down_proj — learn to process memory-augmented hidden states

The entire transformer adapts to work with memory. Without this, frozen attention patterns can't reach memory content and frozen FFNs can't process memory-augmented representations.

### 3. Per-layer memory cross-attention (13.4M params)
Each of the 24 layers has a small cross-attention (2 heads, head_dim=64):
- Q from hidden states, K/V from memory vectors
- Output projection zero-initialized (no-op at start)
- Per-layer learned gate (sigmoid, starts at 0.5)
- Gated residual: `x = x + gate × cross_attn(x, memory)`

The model's own attention (via LoRA) retrieves broadly. The cross-attention retrieves specifically from memory. Both are learned.

### 4. Memory Extractor (5.6M params)
Runs once after all 24 layers. Compresses output into memory:

```
16 learned query vectors (896-dim each)
    │
    ▼ cross-attention to output hidden states
    │
    ▼ SwiGLU FFN (transforms into memory representation)
    │
    ├── 16 memory vectors (stored in buffer)
    ├── 16 priority scores (for eviction)
    └── 1 confidence score (for halting)
```

**Extraction queries** learn WHAT to compress. They attend to the full output and each produce one vector. Different queries learn to extract different things — entities, relationships, context, state.

**FFN** transforms extracted vectors into a representation optimized for future retrieval, not for token prediction. Memory vectors don't need to look like hidden states.

**Priority** scores determine eviction order. High priority = important, survives when buffer is full. Low priority = filler, evicted first.

**Confidence** determines halting. Same mechanism as extraction — just one more query that asks "am I done?" Trained by the LM loss (no separate signal needed).

### 5. Memory Buffer (0 params, just storage)
FIFO when space available, priority-based eviction when full.

- Max 256 entries (16 chunks of history)
- Each entry: 896-dim vector + priority score
- Settling: only final extraction stored (intermediate passes overwrite)
- Cross-chunk: accumulates (16 new entries per chunk)

## How It Works

### Single chunk (no settling)

```
Input: "Alice works at Acme Corp in Tokyo as an engineer"
Memory: [vectors from previous chunks, if any]

1. Embed tokens
2. Run through 24 Qwen layers
   - Each layer: self-attention + FFN (frozen + LoRA)
   - Each layer: cross-attend to memory (if memory exists)
3. Extract: 16 queries compress output → FFN → 16 memory vectors
4. Store in buffer with priorities
5. LM Head → logits → next token
```

### Multi-chunk (memory accumulates)

```
Chunk 1: "Alice works at Acme Corp in Tokyo"
  → process → extract → buffer: [16 Alice vectors]

Chunk 2: "Bob joined Globex in Paris"
  → process WITH chunk 1 memory → extract → buffer: [16 Alice, 16 Bob]

Chunk 3: "Where does Alice work?"
  → process WITH 32 memory vectors
  → cross-attention retrieves Alice-related vectors
  → generates "Acme Corp"
```

### Settling (model decides when to stop)

```
Pass 1:
  visible_memory = persistent buffer (from previous chunks)
  → process → extract → confidence = 0.75
  → logits₁ weighted by 0.75
  → temporary extraction (for next pass)

Pass 2:
  visible_memory = persistent buffer + pass 1 extraction
  → process (differently — memory changed) → extract → confidence = 0.25
  → logits₂ weighted by 0.25
  → budget spent → stop

Final logits = 0.75 × logits₁ + 0.25 × logits₂

Only pass 2's extraction stored in persistent buffer.
```

The model learns through the LM loss:
- Easy input → high confidence on pass 1 → 1 pass (fast)
- Hard input → spread confidence → 2-3 passes (thorough)

## What Memory Can Represent

Memory vectors are 896-dim, passed through a learned FFN. They can encode anything:

| Content | Example |
|---------|---------|
| Facts | "Alice works at Acme" |
| State | "I'm processing a QA task" |
| Numbers | "running total is 42" |
| Plans | "next step: check the database" |
| Experience | "last action got reward 0.8" |
| Context | "this is a science article" |

The model decides what to store through training. Train on QA → learns to store facts. Train on math → learns to store intermediates. Train on agent tasks → learns to store plans and experience.

## Parameter Budget

| Component | Params | What it does |
|-----------|--------|-------------|
| Qwen base (frozen) | 494.0M | Language understanding |
| LoRA (all linear, rank 8) | 4.4M | Adapt transformer to memory |
| Memory cross-attention (×24) | 13.4M | Per-layer retrieval from memory |
| Memory extractor | 5.6M | Compress output → memory vectors |
| Gates + halt + priority | 0.1M | Control signals |
| **Total trainable** | **23.5M** | |
| **Total model** | **517.5M** | |

## What's Designed vs Learned

| Designed (structure) | Learned (policy) |
|---------------------|-----------------|
| Memory as extra KV pairs | What to extract (queries) |
| Cross-attention per layer | What to retrieve (attention) |
| Extraction with FFN | How to represent memory (FFN) |
| Priority for eviction | What's important (priority scores) |
| Confidence for halting | When to stop (confidence query) |
| LoRA on all layers | How to process memory signals |
| FIFO + priority buffer | Everything else |

## Compared to Predictive Transformer

| | Predictive Transformer | Recurrent Memory Transformer |
|---|---|---|
| Trainable params | 212M | 23.5M |
| Mechanisms | 10+ (GRU, prediction, halt net, write gate, value head, reward net, goal state, state modulation...) | 2 (attention + FFN) |
| Memory write | Gated pooling per block | Learned extraction queries |
| State | Separate GRU (224-dim) | Encoded in memory vectors |
| Halting | Separate halt network from prediction errors | Confidence extraction query |
| QA accuracy (15 epochs) | 68% | TBD |

## Future Capabilities (no architecture changes needed)

| Capability | How | Training needed |
|-----------|-----|-----------------|
| QA retrieval | Memory stores facts, attention retrieves | QA data (current) |
| Multi-hop reasoning | Settling chains retrievals across passes | Multi-hop QA data |
| Planning | Memory as scratchpad, settling as deliberation | Planning tasks |
| RL / reward-seeking | Reward as input token, stored in memory | Agent loop + trajectories |
| Continual learning | Online LoRA updates from experience | Agent loop |
| Value estimation | One more extraction query | Agent loop + reward signal |

The architecture is a general-purpose substrate. Training determines capability.

## Files

| File | Contents |
|------|----------|
| `rpvt/model/recurrent_memory.py` | Full model implementation |
| `rpvt/experiments/eval_suite.py` | Evaluation suite (QA, bAbI, LAMBADA, SQuAD, long-doc PPL) |
| `RECURRENT_MEMORY_TRANSFORMER.md` | This document |
