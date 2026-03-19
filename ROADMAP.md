# RPVT Roadmap: From Memory to Continuous Learning Agents

## Vision

Build a brain-like agent architecture where complex capabilities — planning, reflection, creativity, learning from mistakes — **emerge** from the interaction of simple primitives: prediction, memory, valence, and recurrent thought. Not programmed, not hardcoded. Emergent.

The goal: an agent that can do real work in society, overcoming the fundamental limitations of current AI agents.

## The Primitives

Five simple mechanisms. Everything else emerges from their interaction.

| Primitive | Brain analog | What it does |
|---|---|---|
| Memory | Hippocampus | Fast episodic storage of raw experiences |
| Prediction | Cortical hierarchy | Predict next input, predict outcomes, predict retrieval |
| Valence | Amygdala + dopamine | Tag experiences good/bad, instant, no reasoning |
| Recurrent thought | Prefrontal cortex | Multi-step computation over memory before acting |
| Consolidation | Sleep | Offline replay, gradual weight updates toward raw experience |

## What Emerges

| Emergent capability | Primitives required | How it arises |
|---|---|---|
| Planning | prediction + valence + memory | Simulate outcomes, evaluate, choose best |
| Reflection | memory + prediction error + thinking | Review past actions, compare predicted vs actual |
| Learning from mistakes | memory + valence + consolidation | Tag failure, replay, update weights |
| Creativity | random replay + prediction + valence | Novel combinations during consolidation |
| Curiosity | prediction error (seek high error states) | Explore where predictions are worst |
| Caution | memory + negative valence | Retrieve past failures before acting |
| Expertise | consolidated procedural knowledge | Slow knowledge from many replayed experiences |
| Knowing what you don't know | prediction error magnitude | High error = uncertain = ask for help |

None of these are programmed. They emerge from primitives interacting during training and inference.

## Build Plan

### Step 0: Foundation — Memory That Works (DONE)
- Cross-attention memory: 97-100% recall on base model
- 128× compression, write gating, importance-based eviction
- Generalizes across formats, works on real Wikipedia
- **Status: COMPLETE**

### Step 1: Instruct + Memory (IN PROGRESS)
**Goal:** A model that can both use memory AND generate coherently.

**Key finding:** You cannot add memory to an instruct model post-hoc. LoRA destroys generation regardless of approach (8 approaches tried, all failed). The brain develops memory and skills together from birth.

**Solution:** Train from scratch — base model + LoRA + memory + 10k instruct examples.
- v3.9 running overnight (ETA ~10h)
- Same recipe scales to any base model (7B, 30B-A3B, etc.)
- **Status: RUNNING**

### Step 2: Predictive Memory (~1 day)
**Goal:** Replace learned bias gate with prediction-error routing.

Replace the fixed gate (bias=-2.0) with a predictor that models the input stream. Prediction error becomes the write gate — surprising chunks stored, predictable chunks skipped. This is the first brain-like primitive: prediction drives storage.

- Predictor: GRU that maintains running model of input stream
- Write gate = prediction error magnitude (can't collapse, content-dependent)
- Same training, better filtering, drop-in replacement
- Also scale memory READ signal by prediction error (retrieval routing)
- **Key files:** `rpvt/model/predictive_memory.py`

### Step 3: Thinking Module (~1-2 days)
**Goal:** Recurrent deliberation over memory before generation.

N cross-attention steps over memory bank with GRU state update. The thought state changes the query each step — step 1 retrieves fact A, step 2 (informed by A) retrieves fact B, step 3 combines them.

- Train on inference tasks: multi-hop, comparison, constraint, temporal, aggregation
- Test: does N=4 beat N=1 on multi-hop? If yes, recurrence is working.
- Adaptive halting (PonderNet): harder questions get more thinking steps
- **Key files:** `rpvt/model/thinking.py`, `rpvt/data/inference_tasks.py`

### Step 4: Valence Tagging (~1 day)
**Goal:** Distinguish good from bad outcomes.

A scalar tag stored with each memory: positive (good outcome), negative (bad outcome), neutral. This is the amygdala — instant, no reasoning needed.

- Store: (state, action, outcome, valence, surprise)
- Before acting: retrieve high-valence memories for similar states
- Negative valence → caution. Positive valence → repeat.
- Prioritize replay of high-valence experiences during consolidation
- Simple implementation: `memory.write(content, valence=reward_signal)`

### Step 5: Agent Loop (~2-3 days)
**Goal:** Put the model in an environment where it acts and observes.

The agent interacts with a real environment (sandboxed code execution, APIs, file system). Actions have real outcomes. Memory stores action-outcome-valence triples.

```
perceive task → predict outcome → retrieve relevant memories
→ think (multi-step reasoning) → choose action → execute
→ observe real outcome → compute surprise → tag valence
→ store raw experience → update predictor → next task
```

- Environment: sandboxed Python/bash execution, API calls
- Reward signal: task success/failure (code runs, test passes, user approves)
- Memory stores RAW experiences (ground truth), not model interpretations

### Step 6: Sleep Consolidation (~2-3 days)
**Goal:** Offline learning that makes the model smarter over time.

After a work session, replay stored experiences to the model. The model predicts outcomes from weights alone. Prediction error → small weight update. Over many sleep cycles, patterns transfer from episodic memory to model weights.

```
for experience in memory_bank:
    predicted = model.predict(experience.state, experience.action)
    actual = experience.outcome  # RAW ground truth
    error = compare(predicted, actual)

    if error > threshold:
        update_weights(toward=actual)  # not toward model output
    else:
        release_memory(experience)  # already consolidated
```

**Model collapse prevention:**
- ONLY update toward raw experiences (ground truth)
- NEVER train on model's own output
- Interleave old and new experiences during replay
- Prediction error gates update magnitude (protects learned knowledge)

### Step 7: Scale Up
**Goal:** Prove the architecture on larger, more capable models.

- Dual RTX 3090 (48GB NVLink): train 7-13B models locally
- Qwen3-30B-A3B: MoE architecture, 3B active params, fits in 8-bit on dual 3090
- Cloud A100: for bigger runs and publication-quality results
- Architecture is model-agnostic — same code, bigger model

### Step 8: World Model (research frontier)
**Goal:** Predict consequences of actions before taking them.

The thinking module simulates actions using the prediction model:
```
for each possible action:
    predicted_outcome = predictor.predict(state, action)
    predicted_value = valence_model(predicted_outcome)
choose action with highest predicted value
```

This is planning without a planner. The prediction + valence primitives combine to produce goal-directed behavior.

### Step 9: Intrinsic Motivation (research frontier)
**Goal:** The agent decides what to pay attention to, what to learn, what to work on.

- Curiosity: seek states with high expected prediction error
- Competence: seek tasks at the edge of current ability
- Goal persistence: maintain objectives across sessions, decompose into subtasks
- The agent generates its own training curriculum

## Theoretical Foundation

### Core Principle: One Signal Drives Everything
Prediction error serves as:
- **Write gate**: high error → store (surprising input)
- **Read gate**: high error → amplify (useful retrieval)
- **Learning rate**: high error → update more (wrong prediction)
- **Forgetting signal**: low error → release (already consolidated)
- **Curiosity signal**: seek states with high expected prediction error
- **Valence modulation**: error × valence = prioritized learning

### Neuroscience Basis
1. **Complementary Learning Systems** (McClelland 1995): fast hippocampus + slow neocortex
2. **Predictive Coding** (Friston, Rao & Ballard): cortex predicts, errors propagate, surprise drives learning
3. **Sleep Consolidation**: offline replay, gradual episodic → semantic transfer
4. **Dopamine as Reward Prediction Error** (Schultz 1997): valence = predicted reward error

### The Emergence Hypothesis
Complex agent capabilities (planning, reflection, creativity) are NOT separate modules. They emerge from the interaction of simple, general primitives (prediction, memory, valence, recurrence). The architecture provides the primitives. Training provides the pressure. Behavior emerges.

### Model Collapse Prevention
- Episodic memory stores RAW experiences only (ground truth from environment)
- Reflected rules used as CONTEXT at inference, never as training targets
- Consolidation updates toward raw experience, never model output
- Reality is the anchor — the environment provides ground truth

## Full Brain-Like Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       THE AGENT                           │
│                                                           │
│  ┌─────────────┐         ┌──────────────┐                │
│  │   CORTEX    │  ◄──►   │ HIPPOCAMPUS  │                │
│  │  (weights)  │         │  (memory)    │                │
│  │ slow learn  │         │ fast store   │                │
│  │ implicit    │         │ episodic     │                │
│  └──────┬──────┘         └──────┬───────┘                │
│         │                       │                         │
│  ┌──────┴───────────────────────┴────────┐               │
│  │         PREDICTIVE CODING              │               │
│  │  predict → compare → error             │               │
│  │  one signal drives everything          │               │
│  └──────┬───────────────────────┬────────┘               │
│         │                       │                         │
│  ┌──────┴──────┐  ┌────────────┴─────────┐              │
│  │  AMYGDALA   │  │     PREFRONTAL       │              │
│  │  (valence)  │  │  (thinking + goals)  │              │
│  │ good / bad  │  │  recurrent thought   │              │
│  │ instant tag │  │  planning            │              │
│  └─────────────┘  └──────────────────────┘              │
│                                                           │
│  ┌────────────────────────────────────────┐              │
│  │              SLEEP                      │              │
│  │  replay raw experiences                 │              │
│  │  update weights where surprised         │              │
│  │  release consolidated memories          │              │
│  │  NEVER train on own output              │              │
│  └─────────────────────────────────────────┘              │
│                                                           │
│  ┌────────────────────────────────────────┐              │
│  │           ENVIRONMENT                   │              │
│  │  act → observe → ground truth           │              │
│  │  the anchor that prevents collapse      │              │
│  └─────────────────────────────────────────┘              │
└───────────────────────────────────────────────────────────┘
```

### Operating Modes

**Awake (online):** perceive → predict → retrieve → think → act → observe → tag valence → store raw experience → update predictor

**Sleep (offline):** replay raw experiences → predict from weights → error → update weights → release consolidated memories

### Timeline Estimate
- Steps 0-1: Foundation (DONE / running overnight)
- Steps 2-4: Brain primitives (~1 week)
- Steps 5-6: Agent + consolidation (~1 week)
- Step 7: Scale up (needs hardware)
- Steps 8-9: Research frontier (ongoing)

A working brain-like agent on 1.5B in ~2-3 weeks. Scale to 7B+ with better hardware.

## Environment
- Hardware: RTX 3080 Ti (12GB), WSL2
- Model: Qwen2.5-1.5B (base)
- Training: ~2h per experiment (500 docs), ~15h for from-scratch (10k instruct)
- Target: dual RTX 3090 (48GB NVLink) for 7B+ models
- All experiments reproducible from this repo
