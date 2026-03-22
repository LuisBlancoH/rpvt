# Predictive Transformer — Architecture Diagrams

## Full System

```
                            ┌─────────────────────────┐
                            │   Shared Memory Bank     │
                            │   64 slots × 896 dim     │
                            │                          │
                            │  ┌───┬───┬───┬───┬───┐  │
                            │  │ s₁│ s₂│ s₃│...│s₆₄│  │
                            │  └───┴───┴───┴───┴───┘  │
                            │  strength: [0.8, 0.3, …] │
                            │                          │
                            │  write: top 2 blocks     │
                            │  read:  all 24 blocks    │
                            │  evict: weakest slot     │
                            │  update: TD error        │
                            └────▲──────────┬──────────┘
                                 │          │
                          write  │          │ read
                       (gated    │          │ (goal-biased
                        pooling) │          │  attention)
                                 │          │
 ┌───────────────────────────────┴──────────┴───────────────────────────┐
 │                                                                      │
 │  Input tokens ──► Embed (Qwen, frozen) ──► x                       │
 │                                                                      │
 │  ┌─── Adaptive Settling Loop ─────────────────────────────────────┐ │
 │  │  max_settle: 1→2→3→5 (curriculum)                               │ │
 │  │                                                                  │ │
 │  │  ┌─── PredictiveBlock (×24) ──────────────────────────────┐    │ │
 │  │  │                                                         │    │ │
 │  │  │  ① State injection   x = x + proj(GRU_state)          │    │ │
 │  │  │  ② Qwen layer        x = SelfAttn + SwiGLU (frozen)   │    │ │
 │  │  │  ③ Memory attention  mem = CrossAttn(x+goal, memory)  │    │ │
 │  │  │  ④ Memory gate       x = x + sigmoid(W·[x,mem])·mem   │    │ │
 │  │  │  ⑤ Memory FFN        x = x + MemFFN(x)                │    │ │
 │  │  │  ⑥ State update      pool → compress → GRU → state    │    │ │
 │  │  │  ⑦ Prediction        pred = W·RMSNorm(x) → error      │    │ │
 │  │  │  ⑧ Write gate        scores·x → pool → memory.write   │    │ │
 │  │  │                                                         │    │ │
 │  │  └─────────────────────────────────────────────────────────┘    │ │
 │  │                                                                  │ │
 │  │  errors = [e₁, ..., e₂₄] ──► Halt Network ──► stop/continue   │ │
 │  │                                                                  │ │
 │  │  logits_final = Σ (halt_prob × step_logits)                     │ │
 │  └──────────────────────────────────────────────────────────────────┘ │
 │                                                                      │
 │  ┌─── Output Heads ─────────────────────────────────────────────┐   │
 │  │                                                               │   │
 │  │  RMSNorm → LM Head → next token logits                      │   │
 │  │                                                               │   │
 │  │  errors + GRU states → Value Head → "how good is my state?" │   │
 │  │                                                               │   │
 │  │  [prev_errors, errors] → Reward Net → intrinsic reward      │   │
 │  │                                                               │   │
 │  │  errors + TD error → Goal GRU → updated goal state          │   │
 │  │                                                               │   │
 │  └───────────────────────────────────────────────────────────────┘   │
 └──────────────────────────────────────────────────────────────────────┘
```

## Single PredictiveBlock

```
 input   state   goal                 memory bank
   │       │       │                      │
   ▼       ▼       │                      │
 ┌─────────────┐   │                      │
 │ x += proj(  │   │                      │
 │   state)    │   │                      │
 │ (no-op@init)│   │                      │
 └──────┬──────┘   │                      │
        │          │                      │
        ▼          │                      │
 ┌──────────────┐  │                      │
 │ QWEN LAYER   │  │                      │
 │ (frozen)     │  │                      │
 │              │  │                      │
 │ self-attn    │  │                      │
 │ + SwiGLU FFN │  │                      │
 │              │  │                      │
 │ 896-dim      │  │                      │
 └──────┬───────┘  │                      │
        │          │                      │
        ▼          ▼                      ▼
 ┌───────────────────────────────────────────┐
 │  MEMORY ATTENTION (2 heads)               │
 │                                           │
 │  Q = W_q·x + W_goal·goal  ← goal biases │
 │  K = W_k·memory                          │
 │  V = W_v·memory                          │
 │  mem_out = Attn(Q, K, V)                 │
 │  (o_proj=zeros@init → output=0)          │
 │                                           │
 │  GATE: g = σ(W·[x, mem_out])            │
 │  x = x + g·mem_out                       │
 └──────┬────────────────────────────────────┘
        │
        ▼
 ┌──────────────┐
 │ MEMORY FFN   │
 │ 896→1792→896 │
 │ SwiGLU       │
 │ (tiny@init)  │
 └──────┬───────┘
        │
        ▼
 ┌───────────────────────────────────────┐
 │  STATE UPDATE                         │
 │                                       │
 │  query = learned_param                │
 │  pooled = Attn(query, x, x)          │
 │  compressed = W·pooled  (→224-dim)   │
 │                                       │
 │  GRU:                                │
 │    reset  = σ(W·[in, old])           │
 │    update = σ(W·[in, old])           │
 │    cand   = tanh(W·[in, r·old])     │
 │    new = u·old + (1-u)·cand          │
 │                        ──► persists  │
 └──────┬────────────────────────────────┘
        │
        ▼
 ┌───────────────────────────────────────┐
 │  PREDICTION + WRITE                   │
 │                                       │
 │  PREDICTION HEAD:                     │
 │    pred = W·RMSNorm(x)               │
 │    error = ‖prev_block_out - pred‖   │
 │    → feeds halt network              │
 │                                       │
 │  WRITE GATE:                          │
 │    novelty = x - mem_read             │
 │    scores = σ(W·[x, novelty])        │
 │    gated = scores·x                   │
 │    pooled = mean(gated)  → to memory │
 │    strength = mean(scores)            │
 │                                       │
 │    scores≈0 → weak write → evicted   │
 │    scores↑  → focused write → persists│
 └──────┬────────────────────────────────┘
        │
        ▼
     output → next block
```

## Settling (Adaptive Halting)

```
  Pass 1                    Pass 2                    Pass 3
  ─────                    ─────                    ─────

  embed ──► 24 blocks      embed ──► 24 blocks      embed ──► 24 blocks
  state: fresh             state: updated            state: refined
  memory: empty/prev       memory: += pass 1         memory: += pass 2
       │                        │                        │
       ▼                        ▼                        ▼
  errors = [HIGH]           errors = [MEDIUM]        errors = [LOW]
       │                        │                        │
       ▼                        ▼                        ▼
  halt = 0.2               halt = 0.3               halt = 0.5
  (not confident)          (getting there)           (confident!)
       │                        │                        │
       ▼                        ▼                        ▼
  logits × 0.2             logits × 0.3              logits × 0.5
       │                        │                        │
       └───────────┬────────────┘────────────────────────┘
                   ▼
          final = Σ (halt_prob × step_logits)
```

## Multi-Chunk Training

```
  Group: 4 consecutive 128-token chunks from same document

  chunk 1: "Alice works at Acme Corp as an engineer..."
           │
           ▼
      forward → loss₁ → backward → detach_state
      memory: writes 2 vectors (from blocks 23, 24)
      state:  24 GRU states updated
           │
           ▼ (memory + state persist, gradients don't)
  chunk 2: "She moved to Tokyo last year..."
           │
           ▼
      forward (reads chunk 1's memory!) → loss₂ → backward → detach
      memory: writes 2 more vectors (now 4 stored)
           │
           ▼
  chunk 3: "Her colleague Bob joined the team..."
           │
           ▼
      forward (reads chunks 1-2 memory) → loss₃ → backward → detach
      memory: writes 2 more (now 6 stored)
           │
           ▼
  chunk 4: "Where does Alice work?"
           │
           ▼
      forward (reads all 6 memories!) → loss₄ → backward
      Answer quality depends on memory retrieval

  ─── reset_state() ─── next group ───
```

## Value / Reward / Goal Flow

```
                    ┌──────────────┐
                    │  GOAL STATE   │◄── goal_gru(errors, δ)
                    │  dim=64       │    slow-updating (sigmoid(3)≈0.95)
                    │  biases memory│
                    │  queries      │
                    └──────┬───────┘
                           │
      chunk t-1            │           chunk t
      ───────              │           ───────
                           │
  errors_{t-1} ────────────┼──► errors_t
       │                   │        │
       │                   │        ▼
       │              ┌────┴──────────────┐
       └─────────────►│  REWARD NETWORK    │
                      │  [e_{t-1}, e_t]    │
                      │  → intrinsic reward│
                      └────────┬───────────┘
                               │
    V(s_{t-1}) ◄───  V(s_t) ◄─┤
    (detached)   │  (in graph) │
                 │      │      │
                 │      ▼      ▼
                 │  ┌──────────────────┐
                 │  │  VALUE HEAD       │
                 │  │  errors + states  │
                 │  │  → scalar value   │
                 │  │                   │
                 │  │  trained to       │
                 │  │  predict -lm_loss │
                 │  └──────────────────┘
                 │
                 ▼
          TD error: δ = reward + γ·V(t) - V(t-1)
                 │
                 ├──► goal_gru update
                 └──► memory strength update
```

## Memory Lifecycle Example

```
  Input 1: "Alice works at Acme Corp as an engineer in Tokyo"

  Block 23: write_scores focus on "Alice/Acme/engineer" → WRITE
  Block 24: write_scores focus on "Tokyo" → WRITE

  Memory: [Alice-Acme-eng(0.52), Tokyo(0.48), ...]

  ─────────────────────────────────────────────────

  Input 2 (same group): "She also volunteers at a local school"

  Block 23: write_scores focus on "volunteers/school" → WRITE
  Block 24: write_scores neutral → WEAK WRITE

  Memory: [Alice-Acme-eng(0.52), Tokyo(0.48),
           volunteers-school(0.45), weak(0.31), ...]

  ─────────────────────────────────────────────────

  Input 3 (same group): "What company does Alice work for?"

  All blocks: Memory attention retrieves Alice-Acme-eng slot
  Goal state: biases queries toward person/company info
  Memory gate: question + relevant memory → OPEN
  Model generates: "Acme Corp"

  TD update: correct → δ positive → strengthen retrieved slots
```
