# Predictive Transformer — Architecture Diagram

## Full System

```
                            ┌─────────────────────────┐
                            │   Shared Memory Bank     │
                            │   64 slots × 768 dim     │
                            │                          │
                            │  ┌───┬───┬───┬───┬───┐  │
                            │  │ s₁│ s₂│ s₃│...│s₆₄│  │
                            │  └───┴───┴───┴───┴───┘  │
                            │  strength: [0.8, 0.3, …] │
                            │                          │
                            │  write: top 2 blocks     │
                            │  read:  all blocks       │
                            │  evict: weakest slot     │
                            └────▲──────────┬──────────┘
                                 │          │
                          write  │          │ read
                       (novelty- │          │ (attention)
                        gated)   │          │
                                 │          │
 ┌───────────────────────────────┴──────────┴───────────────────────────┐
 │                                                                      │
 │  Input tokens ──► Embed + PosEmbed ──► x                            │
 │                                                                      │
 │  ┌─── Adaptive Settling Loop ─────────────────────────────────────┐ │
 │  │                                                                 │ │
 │  │   ┌── Block 1 ──┐  ┌── Block 2 ──┐       ┌── Block 6 ──┐    │ │
 │  │   │  state₁ GRU  │  │  state₂ GRU  │  ...  │  state₆ GRU  │    │ │
 │  │   │              │  │              │       │              │    │ │
 │  │   │ ①Self-Attn   │  │ ①Self-Attn   │       │ ①Self-Attn   │    │ │
 │  │   │ ②FFN         │→│ ②FFN         │→ … →│ ②FFN         │    │ │
 │  │   │ ③Mem-Attn    │  │ ③Mem-Attn    │       │ ③Mem-Attn    │    │ │
 │  │   │ ④Mem-FFN     │  │ ④Mem-FFN     │       │ ④Mem-FFN     │    │ │
 │  │   │ ⑤State+Pred  │  │ ⑤State+Pred  │       │ ⑤State+Pred  │    │ │
 │  │   │ ⑥Write Gate  │  │ ⑥Write Gate  │       │ ⑥Write Gate  │    │ │
 │  │   └──────────────┘  └──────────────┘       └──────┬───────┘    │ │
 │  │                                                     │           │ │
 │  │   Prediction errors: [e₁, e₂, e₃, e₄, e₅, e₆] ◄──┘           │ │
 │  │          │                                                      │ │
 │  │          ▼                                                      │ │
 │  │   ┌── Halt Network ─────────────┐                              │ │
 │  │   │  halt = σ(MLP(errors))      │                              │ │
 │  │   │  confident? → stop          │                              │ │
 │  │   │  uncertain? → settle again  │                              │ │
 │  │   └─────────────────────────────┘                              │ │
 │  └─────────────────────────────────────────────────────────────────┘ │
 │                                                                      │
 │  LayerNorm ──► LM Head ──► logits ──► next token                    │
 └──────────────────────────────────────────────────────────────────────┘
```

## Single PredictiveBlock (detailed)

```
 input   state                    memory bank
   │       │                          │
   ▼       ▼                          │
 ┌─────────────────┐                  │
 │  x = x + W·state│  ← state        │
 │    (context)     │    injection     │
 └────────┬────────┘                  │
          │                           │
          ▼                           │
 ┌────────────────────────┐           │
 │  ① SELF-ATTENTION      │           │
 │  8 heads               │           │
 │                        │           │
 │  Q ← x  (what am I    │           │
 │          looking for?) │           │
 │  K ← x  (what does    │           │
 │          each pos      │           │
 │          contain?)     │           │
 │  V ← x  (info to      │           │
 │          gather)       │           │
 │                        │           │
 │  causal mask applied   │           │
 │  x = x + attn_out      │           │
 └────────┬───────────────┘           │
          │                           │
          ▼                           │
 ┌────────────────────────┐           │
 │  ② MAIN FFN            │           │
 │  768 → 3072 → 768      │           │
 │  (4x expansion)        │           │
 │                        │           │
 │  Understand input      │           │
 │  BEFORE retrieving     │           │
 │  from memory           │           │
 │                        │           │
 │  x = x + FFN(x)        │           │
 └────────┬───────────────┘           │
          │                           │
          ▼                           ▼
 ┌──────────────────────────────────────┐
 │  ③ MEMORY ATTENTION                  │
 │  4 heads                             │
 │                                      │
 │  Q ← x       (what do I need?)      │
 │  K ← memory  (what's stored?)       │
 │  V ← memory  (stored info)          │
 │                                      │
 │  mem_out = Attention(Q, K_mem, V_mem)│
 │                                      │
 │  ┌────────────────────────────────┐ │
 │  │  INPUT-AWARE GATE              │ │
 │  │                                │ │
 │  │  gate = σ(W · [x, mem_out])    │ │
 │  │                                │ │
 │  │  sees BOTH:                    │ │
 │  │   • what I'm processing (x)   │ │
 │  │   • what memory returned       │ │
 │  │                                │ │
 │  │  learns WHEN to use memory:    │ │
 │  │   question + relevant mem → ON │ │
 │  │   greeting + any mem → OFF     │ │
 │  │   question + empty mem → OFF   │ │
 │  └────────────────────────────────┘ │
 │                                      │
 │  x = x + gate * mem_out             │
 └────────┬─────────────────────────────┘
          │
          ▼
 ┌────────────────────────┐
 │  ④ MEMORY FFN           │
 │  768 → 1536 → 768      │
 │  (2x expansion)        │
 │                        │
 │  Integrate memory      │
 │  with understanding    │
 │  (nonlinear combine)   │
 │                        │
 │  x = x + MemFFN(x)     │
 └────────┬───────────────┘
          │
          ▼
 ┌────────────────────────────────────────────┐
 │  ⑤ STATE UPDATE                            │
 │                                            │
 │  ┌─────────────────────────────┐          │
 │  │  ATTENTION POOLING           │          │
 │  │  (learned, not mean pool)    │          │
 │  │                              │          │
 │  │  query = learned_param       │          │
 │  │  pooled = Attn(query, x, x)  │          │
 │  │  → focuses on important pos  │          │
 │  └──────────────┬──────────────┘          │
 │                  │                         │
 │                  ▼                         │
 │  ┌──────────────────────────────┐         │
 │  │  GRU                         │         │
 │  │                              │         │
 │  │  input = compress(pooled)    │         │
 │  │                              │         │
 │  │  reset  = σ(W·[in, old])    │         │
 │  │  update = σ(W·[in, old])    │         │
 │  │  cand   = tanh(W·[in,       │         │
 │  │              reset·old])     │         │
 │  │  new = update·old +          │         │
 │  │       (1-update)·cand        │         │
 │  │                              │  ──► new state
 │  │  Persists across inputs      │   (carries to
 │  │  AND settling steps          │    next input)
 │  └──────────────────────────────┘         │
 └────────┬───────────────────────────────────┘
          │
          ▼
 ┌────────────────────────────────────────────┐
 │  ⑥ PREDICTION + WRITE                      │
 │                                            │
 │  ┌─────────────────────┐   ┌────────────┐│
 │  │ PREDICTION HEAD      │   │ WRITE GATE  ││
 │  │                      │   │             ││
 │  │ pred = LN(x) → W·x  │   │ novelty =   ││
 │  │                      │   │  x - mem_rd ││
 │  │ predicts block (i-1) │   │             ││
 │  │ output (top-down)    │   │ strength =  ││
 │  │                      │   │  σ(W·[x,    ││
 │  │ error = ‖actual -    │   │    novelty])││
 │  │          predicted‖  │   │             ││
 │  │                      │   │ high if new ││
 │  │ → feeds halt network │   │ low if known││
 │  └─────────────────────┘   └─────┬──────┘│
 │                                   │       │
 │                            (top 2 blocks  │
 │                             write to      │
 │                             memory bank)  │
 └────────┬──────────────────────────────────┘
          │
          ▼
       output → next block
```

## Settling (Adaptive Halting)

```
  Pass 1                    Pass 2                    Pass 3
  ─────                    ─────                    ─────

  embed ──► blocks         embed ──► blocks         embed ──► blocks
  (states = init)          (states = updated)       (states = refined)
  (memory = empty/prev)    (memory += pass 1)       (memory += pass 2)
       │                        │                        │
       ▼                        ▼                        ▼
  errors = [HIGH]           errors = [MEDIUM]        errors = [LOW]
       │                        │                        │
       ▼                        ▼                        ▼
  halt_prob = 0.2           halt_prob = 0.3          halt_prob = 0.5
  (not confident)           (getting there)          (confident!)
       │                        │                        │
       ▼                        ▼                        ▼
  logits × 0.2              logits × 0.3             logits × 0.5
       │                        │                        │
       └────────────┬───────────┘────────────────────────┘
                    ▼
           final logits = Σ (halt_prob × step_logits)
                    ▼
              next token
```

## Memory Lifecycle

```
  Input 1: "Alice works at Acme Corp as an engineer in Tokyo"

  Block 5: gate=0.82 for "Alice/Acme" positions → WRITE
  Block 6: gate=0.76 for "engineer/Tokyo" positions → WRITE

  Memory: [Alice-Acme(0.82), engineer-Tokyo(0.76), ...]

  ─────────────────────────────────────────────────

  Input 2: "Alice works at Acme Corp as an engineer in Tokyo"  (repeated)

  Block 5: mem_read ≈ hidden → novelty ≈ 0 → gate=0.08 → SKIP
  Block 6: mem_read ≈ hidden → novelty ≈ 0 → gate=0.05 → SKIP

  Memory: unchanged (no duplicates stored)

  ─────────────────────────────────────────────────

  Input 3: "Bob joined Globex as a designer in Paris"

  Block 5: mem_read ≠ hidden → novelty HIGH → gate=0.79 → WRITE
  Block 6: mem_read ≠ hidden → novelty HIGH → gate=0.71 → WRITE

  Memory: [Alice-Acme(0.82), engineer-Tokyo(0.76),
           Bob-Globex(0.79), designer-Paris(0.71), ...]

  ─────────────────────────────────────────────────

  Input 4: "Where does Alice work?"

  All blocks: Memory-Attention retrieves Alice-Acme slot
  Input-aware gate: question + relevant memory → OPEN
  Model generates: "Acme Corp"
```

## Brain Analogy Map

```
  ┌──────────────────────┬──────────────────────────────────┐
  │  Brain               │  Predictive Transformer           │
  ├──────────────────────┼──────────────────────────────────┤
  │  Cortical column     │  PredictiveBlock                  │
  │  Feedforward path    │  Self-Attention + FFN              │
  │  Feedback path       │  Prediction Head (top-down)       │
  │  Prediction error    │  ‖actual - predicted‖             │
  │  Neural settling     │  Adaptive settling loop           │
  │  Working memory (PFC)│  GRU state (per block)            │
  │  Hippocampus         │  Shared Memory Bank               │
  │  Encoding gate       │  Novelty-aware write gate         │
  │  Pattern completion  │  Memory attention (retrieval)     │
  │  Novelty detection   │  hidden - mem_read signal         │
  │  Consolidation       │  Strength-based eviction          │
  │  Attention (top-down)│  Input-aware memory gate          │
  │  Neuromodulation     │  Learned gates (context-dependent)│
  └──────────────────────┴──────────────────────────────────┘
```
