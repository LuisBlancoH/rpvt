# RPVT Roadmap: From Memory to Continuous Learning Agents

## Vision

Build an agent architecture that overcomes the fundamental limitations of current AI agents: no memory, no learning from experience, no improvement over time. The system should operate like biological intelligence — accumulate experience, learn from mistakes, consolidate knowledge, and get better every day.

## Foundations (Completed)

### Memory System (v3.2-v3.3)
- Cross-attention memory injection: 97-100% recall on base model
- 128× compression (128 tokens → 1 vector)
- Write gating, importance-based eviction
- Generalizes across formats, works on real Wikipedia
- **Status: DONE**

### Architecture Validation
- Additive injection fails on pretrained models (23 experiments)
- Cross-attention works because it uses the model's own attention mechanism
- Smaller models learn faster (1.5B > 3B)
- Memory skill generalizes: train simple → deploy complex
- **Status: DONE**

## Phase 1: Instruct + Memory (In Progress)

### The Problem
Any weight modification (LoRA) on an instruct model that enables memory reading also destroys generation quality. Frozen models achieve ~33% recall max.

### Approaches Tried
1. LoRA on instruct model → 86% recall, broken generation
2. Joint training (v3.5) → 79% recall, broken generation (insufficient instruct data)
3. Parallel cross-attention, no LoRA → 33% recall, perfect generation
4. Parallel cross-attention + LoRA → 81% recall, broken generation
5. Soft prompts, no LoRA → 33% recall, coherent generation
6. Surprise-weighted LoRA (v3.10) → running, TBD
7. From-scratch training (v3.9) → ready to run (10k instruct + 500 memory, ~15h)

### Key Insight (from neuroscience)
The brain never has this problem because memory (hippocampus) and skills (neocortex) develop together from birth. Adding memory post-hoc to a trained model is like wiring a new sense into an adult brain.

### Solution: Train From Scratch
Base model + LoRA + memory + 10k instruct examples, all learning together. No prior distribution to protect. This is v3.9.

## Phase 2: Predictive Routing (Designed)

### Architecture
Replace learned gate with prediction-error routing:
- Predictor (GRU) models the input stream
- Prediction error = surprise signal
- Write: surprising chunks stored, predictable chunks skipped
- Read: surprising retrievals amplified, expected ones suppressed
- No LoRA needed during inference — model stays frozen
- One signal (prediction error) drives everything

### Key Files
- `rpvt/model/predictive_memory.py` — PredictiveMemoryBank, PredictiveWriteWrapper
- `rpvt/data/inference_tasks.py` — Multi-hop, comparison, constraint, temporal, aggregation tasks

## Phase 3: Thinking Module (Designed)

### Architecture
Recurrent deliberation over memory before generation:
- N cross-attention steps over memory bank
- GRU updates thought state each step
- Thought state injected into model for generation
- Trained end-to-end with answer loss

### What It Enables
- Multi-hop reasoning: A→B, B→C, answer A→C
- Comparison: retrieve two facts, compare
- Constraint satisfaction: check facts against rules
- Adaptive computation: harder questions get more thinking steps

### Key Files
- `rpvt/model/thinking.py` — ThinkingModule, ThoughtInjector
- `rpvt/experiments/exp_v3_7_thinking.py` — Full experiment with inference tasks

## Phase 4: Continuous Learning Agent

### The Full Loop
```
WORK (online, no weight changes):
  Task → predict outcome → retrieve memories → choose action → execute
  → observe outcome → compute surprise → store experience
  → update predictor

SLEEP (offline, gradual weight changes):
  Replay memories → model predicts → prediction error
  → high error: update weights (not yet learned)
  → low error: release memory (already consolidated)
  → model absorbs patterns, memory bank freed for new experiences
```

### Timescale Separation
- Milliseconds: model inference (frozen weights)
- Seconds: memory storage/retrieval (state updates)
- Hours: predictor adaptation (online, lightweight)
- Days: weight consolidation (offline, sleep)

### Components Needed
1. Action-outcome memory (store what happened, not just facts)
2. Online predictor (learn to predict outcomes from actions)
3. Consolidation loop (replay + surprise-gated weight updates)
4. Memory management (release consolidated memories, make room)

## Phase 5: World Model

### Predictive Thinking (v3.8 concept)
- Predict consequences of actions before taking them
- Training signal: prediction error between predicted and actual outcome
- Enables planning: simulate multiple actions, pick best

### Hierarchy
```
Level 1: predict next chunk (input statistics)
Level 2: predict action outcomes (causal model)
Level 3: predict episode patterns (schema)
Level 4: predict general rules (theory)
```

## Phase 6: Motivation and Goal Persistence

### What's Missing
Everything above is reactive. The agent processes what it's given. Motivation means the agent decides what to pay attention to, what to learn, what to work on.

### Architecture
- Persistent goal state that biases all processing
- Prediction error computed relative to goals, not just statistical expectation
- Curiosity: seek high prediction error states (explore the unknown)
- Goal hierarchy: top-level objectives decompose into subtasks

## Theoretical Foundation

### Neuroscience Principles
1. **Complementary Learning Systems**: fast episodic (hippocampus) + slow semantic (neocortex)
2. **Predictive Coding**: cortex predicts, only errors propagate, surprise drives everything
3. **Sleep Consolidation**: offline replay, gradual transfer from episodic to semantic
4. **Neuromodulation**: dynamic learning rates based on arousal/surprise/reward

### One Signal Drives Everything
Prediction error serves as:
- **Write gate**: high error → store (surprising input)
- **Read gate**: high error → amplify (useful retrieval)
- **Learning rate**: high error → update more (wrong prediction)
- **Forgetting signal**: low error → release (already consolidated)
- **Curiosity signal**: seek states with high expected prediction error

### Philosophical Position
- Current LLMs are stateless pattern matchers (Montañez critique is valid)
- Memory alone is a filing cabinet — necessary but not sufficient
- Predictive coding + recurrence + consolidation = qualitatively different system
- Whether it constitutes "understanding" is open — but the behavioral gap shrinks with each mechanism added
- The question isn't "can machines think" but "can machines build world models that enable effective action"

## Full Brain-Like Architecture

```
┌─────────────────────────────────────────────────────┐
│                    THE AGENT                         │
│                                                      │
│  CORTEX (model weights)  ◄──►  HIPPOCAMPUS (memory) │
│  slow learner, general        fast store, specific   │
│  implicit knowledge           episodic experiences   │
│                                                      │
│  PREDICTIVE CODING (prediction error)                │
│  one signal drives everything:                       │
│    storage, retrieval, consolidation, attention      │
│                                                      │
│  AMYGDALA (valence tags)    PREFRONTAL (goals)       │
│  good/bad, instant          planning, reflection     │
│  no reasoning needed        temporary text scaffolds │
│                                                      │
│  SLEEP (offline consolidation)                       │
│  replay raw experiences → update weights             │
│  high valence = more replay                          │
│  NEVER train on own output → prevents model collapse │
└─────────────────────────────────────────────────────┘
```

### Operating Modes

**Awake:** perceive → predict → act → observe → tag (good/bad) → store raw experience → update predictor

**Thinking:** retrieve memories → prediction error routes relevant ones → multi-step reasoning → answer

**Sleep:** replay raw experiences → cortex predicts → error → gradual weight update → release consolidated memories

### Model Collapse Prevention
- Episodic memory stores RAW experiences only (ground truth)
- Reflected rules are used as CONTEXT, never as training targets
- Consolidation always updates toward raw experience, never model output
- Reality is the anchor — derived things can be wrong, raw experience can't

## Environment
- Hardware: RTX 3080 Ti (12GB), WSL2
- Model: Qwen2.5-1.5B (base + instruct)
- Training: ~2h per experiment (500 docs), ~15h for from-scratch (10k instruct)
- All experiments reproducible from this repo
