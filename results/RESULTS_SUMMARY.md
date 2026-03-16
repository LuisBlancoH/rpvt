# Cross-Attention Memory for Pretrained Transformers — Results Summary

## The Problem

Pretrained LLMs are stateless. Each forward pass is independent. We add a differentiable external memory module that enables persistent state across chunks, turning a stateless model into a stateful one.

## The Architecture

- **Base model**: Qwen2.5-1.5B (frozen, bf16)
- **Trainable**: LoRA rank 16 on q_proj, v_proj (2.18M params) + memory gate (1,537 params)
- **Memory**: Write gate at layer 14, cross-attention read at layer 15
- **Mechanism**: Memory hidden states become extra KV pairs in attention (no RoPE)
- **Compression**: 128 tokens → 1 vector (1536-dim), 128× compression ratio

Total trainable: 2,180,609 params (0.14% of model)

## Why Cross-Attention (Not Additive Injection)

We ran 23 experiments proving that additive residual injection fails on pretrained models. The model's residual stream has a learned distribution — foreign signals are noise. Even with full LoRA (30M params on all linear layers), the model actively learns to suppress memory output.

Cross-attention works because it uses the model's existing attention mechanism. Memory KVs are just additional KV pairs — the model already knows how to attend to KV pairs. LoRA adapts q_proj/v_proj to query memory naturally.

## Synthetic Recall Results

All results on Qwen2.5-1.5B, 15 epochs, 500 train / 100 eval documents.

### Scaling

| Experiment | Passages | QA pairs | Gap | Token Acc | Exact Match |
|---|---|---|---|---|---|
| No memory baseline | 1 | 3 | 2-6 | 15.9% | 0% |
| Single passage | 1 | 3 | 2-6 | 97.7% | 80% |
| More QA pairs | 1 | 6 | 2-6 | 99.2% | 86% |
| Long gaps | 1 | 6 | 6-12 | 99.6% | 92% |
| 2 passages | 2 | 6 (3+3) | 2-6 | 99.8% | 97% |
| Confusable facts | 2 | 6 (3+3) | 2-6 | 99.8% | 97% |
| **5 passages** | **5** | **10 (2×5)** | **2-6** | **100.0%** | **100.0%** |

Every scaling test improves accuracy. More passages = more chunks = richer gradient signal.

### Generalization (No Retraining)

Train on single-passage synthetic, evaluate on unseen formats:

| Eval format | Token Acc | Notes |
|---|---|---|
| Same format (single passage) | 97.0% | Control |
| Multi-passage (2 people) | 98.0% | Never seen during training |
| 5-passage (5 people) | 98.1% | Never seen during training |
| Natural-style templates | 97.7% | Different phrasing |
| SQuAD (real Wikipedia) | 86.0% | Real text, no synthetic training |

The model learned "how to use memory" as a general skill, not the specific task format.

### Natural Training (Varied Templates)

Training on 6 diverse passage templates with 3 question phrasings per fact type:

| Eval format | Token Acc | Notes |
|---|---|---|
| Natural (same format) | 99.2% | Control |
| Synthetic (rigid template) | 98.5% | Cross-format |
| Multi-passage | 98.4% | Cross-format |
| **SQuAD (real Wikipedia)** | **90.4%** | **+35 pts over no-memory (55%)** |

## Real-World Results

### Streaming Document QA (v3.3)

Process real Wikipedia articles chunk-by-chunk, accumulate memory, answer questions:

| Condition | Token Acc | Exact Match | Questions |
|---|---|---|---|
| No memory | 39.7% | 4.8% | 227 |
| **Memory (numbers only)** | **100.0%** | **100.0%** | **227** |
| **Memory (diverse Q types)** | **97.3%** | **73.9%** | **364** |

- 48-49 real articles (Tatwine, Nerva, Olmec heads, Imagism, etc.)
- Question types: numbers, entity descriptions, locations, relationships
- Uses natural-trained checkpoint with zero additional training
- Each chunk is a separate forward pass — context cleared between chunks

## Key Insights

1. **Cross-attention is the right interface** between external memory and pretrained transformers. Additive injection fails (23 experiments). Cross-attention uses the model's existing attention mechanism.

2. **The memory skill generalizes**. Train on the simplest format, deploy on complex formats and real text. The model learns "use memory" not "answer synthetic QA."

3. **Smaller models learn faster**. 1.5B outperforms 3B (97.6% vs 42.5% at epoch 5). Shorter gradient path + higher LoRA influence ratio.

4. **More data = better** (counterintuitively). More passages, more QA pairs, longer gaps all improve accuracy. More chunks per document means richer gradient.

5. **1,537 parameters is enough**. The memory gate is the only memory-specific parameter. Everything else (KV projection, attention) is reused from the base model via LoRA.

6. **128× compression with ~99% fidelity**. A 128-token chunk compressed to a single 1536-dim vector retains enough information to answer specific factual questions about names, dates, numbers, locations, and organizations.

## Comparison with Prior Work

| Approach | Trained from scratch? | Parameters added | Works on pretrained? |
|---|---|---|---|
| Memorizing Transformers (Wu 2022) | N/A (kNN lookup) | Stores all past KVs | Yes but no compression |
| Recurrent Memory Transformer (Bulatov 2022) | Yes | Memory tokens | Not tested on pretrained |
| Compressive Transformer (Rae 2019) | Yes | Compression network | Not tested on pretrained |
| **This work** | **No (LoRA only)** | **1,537 memory + 2.18M LoRA** | **Yes — 97-100%** |

## Reproducibility

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft datasets safetensors einops
pip install -e .

# Train (15 min on 3080 Ti)
python -m rpvt.experiments.exp_v3_2_nlp_recall \
  --model-name Qwen/Qwen2.5-1.5B \
  --data-source synthetic_natural \
  --memory-mode cross_attn \
  --epochs 15 --n-train 500 --n-eval 100 \
  --save-checkpoint results/checkpoints/natural_15ep.pt

# Streaming eval on real Wikipedia
python -m rpvt.experiments.exp_v3_3_streaming \
  --load-checkpoint results/checkpoints/natural_15ep.pt \
  --n-articles 50 --n-questions 8
```
