# Recurrent Memory Transformer

## What It Is

A frozen Qwen transformer wrapped with memory, recurrence, and three deep control tokens. The transformer adapts via LoRA on all layers. Two core mechanisms do everything: **attention** (retrieval) and **FFN** (representation).

28.7M trainable params. 494M frozen. Fits on a 3080 Ti.

## Architecture

```
Input: [value_token, confidence_token, manager_token, t₁, t₂, ..., tₙ]
         ↓              ↓                ↓
   ┌── 24 Qwen Layers (frozen + LoRA) ─────────────────────────────┐
   │   Each layer:                                                  │
   │     Self-Attention (all tokens attend to each other + memory)  │
   │     FFN (process memory-augmented representations)             │
   │     + memory cross-attention (2 heads, gated, to stored memory)│
   └────────────────────────────────────────────────────────────────┘
         ↓              ↓                ↓                ↓
   value_hidden   conf_hidden    manager_hidden     output_hidden
         ↓              ↓                ↓                ↓
   MLP(value,action)  sigmoid       score each         LM head
     → value          → halt        memory entry       → logits
                                    → update priorities
         ↓              ↓                ↓                ↓
   ┌── After Halting ──────────────────────────────────────────────┐
   │  1. Manager updates all memory priorities (deep re-evaluation)│
   │  2. Extraction: 16 queries compress output → memory vectors   │
   │  3. Store in persistent buffer with temporal encoding          │
   └───────────────────────────────────────────────────────────────┘
```

## Control Tokens

Three learned tokens processed through ALL 24 transformer layers alongside the input. Each attends to input, memory, and each other. All decisions made with full depth.

### Value Token
- **Asks**: "Given my state and proposed action, how much reward do I expect?"
- **Sees**: input (through self-attention) + memory (through cross-attention)
- **Output**: `value = MLP([value_hidden, last_position_hidden])` → scalar
- **Used for**: advantage estimation in RL, value-guided settling search
- **During settling**: value stored in temporary memory → next pass sees "my value was 0.2, try better"

### Confidence Token
- **Asks**: "Am I confident enough to stop thinking?"
- **Sees**: same as value — full depth of input + memory
- **Output**: `confidence = sigmoid(linear(confidence_hidden))` → [0, 1]
- **Used for**: halting decision (budget-weighted settling)
- **During settling**: stored in temporary memory → next pass sees previous confidence

### Manager Token
- **Asks**: "Given current context, how relevant is each stored memory entry?"
- **Sees**: same as value/confidence — full depth
- **Output**: per-entry priority scores via learned projections
- **Used for**: memory management (what to keep, what to evict)
- **When**: ONLY after halting. Manager thinks during settling but acts once at the end.

## Memory System

### Writing (extraction)
After all layers, 16 learned queries compress output into memory vectors:
```
16 queries attend to output hidden states → cross-attention → FFN → 16 vectors
Each vector gets a priority score (learned importance)
Temporal step encoding added (sinusoidal, no learned params)
Stored in persistent buffer
```

### Storage (buffer)
- FIFO when space available, priority-based eviction when full
- Max 256 entries (16 per chunk = ~16 chunks of history)
- Each entry has: vector (896-dim) + priority (scalar) + temporal encoding (baked in)

### Reading (per-layer cross-attention)
All 24 layers read from memory via cross-attention (2 heads each):
```
Q = current hidden state
K, V = stored memory entries
output = gated residual: x = x + gate × cross_attn(x, memory)
```
Gate is learned per layer (when to use memory at each depth).

### Priority Management (deep, model-driven)
After halting, the manager token re-evaluates ALL stored entries:
```
manager_query = query_proj(manager_hidden)     ← 24 layers deep
entry_key_i   = key_proj(memory_entry_i)       ← learned projection
score_i       = sigmoid(query · key_i / √d)    ← per-entry relevance
→ replaces priority for all entries
```
Model learns: "goals always high priority, filler always low, facts depend on context."

### Temporal Encoding
Sinusoidal step signal added to each vector at storage time:
```
signal = 0.1 × sin/cos(step × frequencies)
```
- No learned parameters, generalizes to any step count
- Model can distinguish "stored 1 chunk ago" from "stored 10 chunks ago"
- Enables ordering: "A then B" ≠ "B then A"
- Only increments across chunks, NOT during settling

## Settling (Adaptive Computation)

```
Pass 1:
  [value, confidence, manager, input] + persistent memory → 24 layers
  → value = 0.2 (low)    → confidence = 0.3 (not halting)
  → extract temporary memory (model's current understanding)
  → control states stored in temporary memory for next pass

Pass 2:
  [value, confidence, manager, input] + persistent + temporary memory
  → value = 0.7 (better!) → confidence = 0.5
  → model sees its own previous value/confidence
  → refines understanding, different output

Pass 3:
  → value = 0.9           → confidence = 0.7 → budget spent → HALT
  → manager scores all entries → priorities updated
  → final extraction stored permanently
```

Key properties:
- Temporary memory overwrites each pass (settling refines, doesn't accumulate)
- Only final extraction stored permanently
- Control states visible during settling, not persisted across chunks
- Step counter does NOT increment during settling (same temporal position)

## Multi-Chunk Processing

```
Chunk 1: "Alice works at Acme in Tokyo"
  → 24 layers → extract 16 vectors (step 0) → store

Chunk 2: "Bob joined Globex in Paris"
  → 24 layers + memory from chunk 1 → extract (step 1) → store
  → manager re-scores chunk 1 entries (still relevant? → yes)

Chunk 3: "Where does Alice work?"
  → 24 layers + memory from chunks 1,2
  → cross-attention retrieves Alice-Acme entries
  → value token: "high value — I can answer this"
  → generates "Acme Corp"
  → manager re-scores: Alice entries = high priority, Bob = moderate
```

## RL Training

### Supervised phase (demonstrations)
Standard cross-entropy on QA, code, interactive task demonstrations.

### RL phase (policy gradient with advantage)
```
action = model.generate(observation, temperature=0.7)
reward = environment.evaluate(action)

advantage = reward - value    (value from value token)
policy_loss = -(advantage × log_prob(action))
value_loss = (value - td_target)²

Both losses train through the full model:
  policy_loss → LoRA weights, extraction, memory cross-attention
  value_loss → value token embedding, value MLP, LoRA
```

### Value-guided settling
During settling, the model searches for high-value actions:
```
Pass 1: candidate A → value 0.2 → "not good enough"
Pass 2: candidate B → value 0.7 → "better, but can I do more?"
Pass 3: candidate C → value 0.9 → "good" → halt → commit to C
```

## Parameter Budget

| Component | Params | Purpose |
|-----------|--------|---------|
| Qwen 0.5B-Instruct (frozen) | 494.0M | Language understanding |
| LoRA (all linear, rank 8) | 4.4M | Adapt to memory signals |
| Memory cross-attention (×24) | 13.4M | Per-layer retrieval |
| Memory extractor (queries+attn+FFN) | 5.6M | Compress → memory |
| Value token + MLP | 1.8M | State+action → value |
| Confidence token + proj | 0.9K | Halt decision |
| Manager token + projections | 1.6M | Memory priority management |
| Control token embeddings | 2.7K | 3 × 896 |
| **Total trainable** | **~28.7M** | |

## Files

| File | Contents |
|------|----------|
| `rpvt/model/recurrent_memory.py` | Full model implementation |
| `rpvt/experiments/exp_v3_30_rmt_qa.py` | QA training (0.5B) |
| `rpvt/experiments/exp_v3_32_agent.py` | Agent training (3B, all levels) |
| `rpvt/experiments/exp_v3_33_agent_rl.py` | RL training with value/advantage |
| `rpvt/experiments/eval_suite.py` | Evaluation across datasets |
| `RECURRENT_MEMORY_TRANSFORMER.md` | This document |

## Design Principles

1. **Bitter lesson**: minimal structure, maximal learning. Two mechanisms (attention + FFN) everywhere.
2. **Deep decisions**: all critical decisions (value, confidence, memory management) through full transformer depth via control tokens.
3. **Model manages itself**: memory priorities updated by the model's own learned assessment, not prescribed rules.
4. **Temporal awareness**: sinusoidal encoding gives ordering without learned parameters.
5. **Settling = search**: multiple passes with value feedback enable candidate search.
6. **Memory = state**: no separate GRU/goal — memory vectors carry everything.
