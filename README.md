# RPVT: Recurrent Plastic Value-Driven Transformer

A continuously learning transformer. A frozen pretrained transformer (Qwen 4B) with lightweight plastic adapters at every layer that update via local backprop during inference. The system learns from every interaction, accumulates knowledge at multiple timescales, routes to specialized experts, and consolidates based on intrinsic value signals.

## Core Idea

Each layer of the transformer trains its own adapters using only a local loss signal, with no gradients flowing between layers. This enables learning at inference time without an external training pipeline.

## Architecture Overview

- **Frozen base model:** Qwen2.5-3B/4B with all original weights frozen
- **Plastic adapters:** Multi-timescale LoRA (fast/medium/slow) at every weight matrix
- **Local learning:** Per-layer loss via logit lens projections — no global backprop needed
- **Mixture of Experts:** Spanning experts (shared frozen weights, separate adapter sets) with dynamic growth
- **Value system:** Intrinsic value signals (coherence, calibration, engagement, resolution) drive reward prediction error, which gates long-term consolidation
- **Settling:** Multiple forward passes with per-layer recurrent state for hard inputs
- **Anti-forgetting:** Timescale separation, expert isolation, sparse activations, EWC, expert splitting, sleep consolidation

## Experimental Plan

Each experiment validates one component incrementally:

1. **Local backprop at one layer** — validate that logit lens loss works (THIS IS FIRST)
2. **All layers simultaneously** — scale to full network
3. **Continuous learning** — establish forgetting baseline
4. **Multi-timescale adapters** — reduce forgetting
5. **Prediction loss** — inter-layer coordination
6. **Settling** — recurrent refinement
7. **MoE experts** — capacity and isolation
8. **Value and consolidation** — intrinsic reward-driven learning

## Project Structure

```
rpvt/
├── model/          # Model loading, adapter modules, logit lens
├── training/       # Training loops, local backprop, losses
├── evaluation/     # Benchmarks, metrics, comparison tools
└── experiments/    # Experiment configs and scripts
```

## Setup

```bash
pip install -e .
```

## Status

🔬 Experiment 1: Local backprop at one layer — in progress
