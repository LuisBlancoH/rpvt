# Cross-Attention Memory Injection for Stateful Pretrained Transformers

## Paper Outline

### Title Options
1. "Cross-Attention Memory Injection: Adding Persistent State to Frozen Pretrained Transformers"
2. "From Stateless to Stateful: Differentiable External Memory for Pretrained LLMs"
3. "Memory as Extra KV Pairs: A Minimal Architecture for Persistent State in Transformers"

### Abstract (draft)

We present a method for adding differentiable external memory to frozen pretrained transformers, enabling persistent state across context boundaries. Our approach injects memory as additional key-value pairs into a single attention layer, allowing the model's existing attention mechanism to naturally decide when and what to retrieve. Unlike additive residual injection — which we show fails comprehensively across 23 experiments — cross-attention memory injection works because it operates within the model's learned attention distribution rather than disrupting its residual stream.

On a frozen Qwen2.5-1.5B model with only 1,537 memory-specific parameters and 2.18M LoRA parameters (0.14% of model), we achieve: 100% exact match on 5-passage factual recall, 97.3% token accuracy on streaming Wikipedia QA, and 99.8% on cross-document reasoning — all with 128× compression (128 tokens → 1 vector). The memory skill generalizes across formats without retraining: a model trained on single-passage synthetic data achieves 98% on multi-passage evaluation and 90.4% on real Wikipedia. Our results demonstrate that pretrained transformers can be efficiently augmented with genuine persistent state, bridging the gap between stateless language models and stateful agents.

### 1. Introduction

- LLMs are stateless — each forward pass independent
- Context window is not real memory (re-reads everything, finite, expensive)
- Agents need persistent state across interactions
- Our contribution: minimal architecture that adds genuine memory to pretrained models
- Key insight: cross-attention (not additive injection) is the right interface

### 2. Related Work

- Memorizing Transformers (Wu et al., 2022) — kNN over past KVs, no compression
- Recurrent Memory Transformer (Bulatov et al., 2022-2023) — memory tokens, trained from scratch
- Compressive Transformer (Rae et al., 2019) — compression network, trained from scratch
- Block-Recurrent Transformer (Hutchins et al., 2022)
- ∞-former (Martins et al., 2022) — continuous memory
- RAG approaches — external retrieval, not differentiable end-to-end
- LoRA (Hu et al., 2021) — parameter-efficient fine-tuning
- **Our distinction**: works on frozen pretrained models, minimal parameters, 128× compression

### 3. Why Additive Injection Fails on Pretrained Models

- Section presenting the negative result (valuable contribution)
- The residual stream has a learned distribution — foreign signals are noise
- Experimental evidence:
  - Gate learns perfectly (1.8B× store/filler ratio) but retrieval fails
  - Attention is already 100% on passage slots — not an alignment problem
  - Memory size doesn't matter (256=1024=2048)
  - Even full LoRA (30M params) → model suppresses memory
- Root cause analysis: additive injection works from scratch (v2.6) but not on pretrained models
- This explains why prior work trains from scratch

### 4. Cross-Attention Memory Architecture

#### 4.1 Architecture
- MemoryBank: circular buffer + learned write gate
- WriteWrapper: gated hidden states → buffer (layer N)
- MemoryAugmentedAttention: memory as extra KV pairs (layer N+1)
- No RoPE on memory KVs (content-based retrieval)
- LoRA on q_proj, v_proj for adaptation

#### 4.2 Why It Works
- Uses the model's existing attention mechanism
- Memory KVs are just additional KV pairs — model already knows how to attend
- LoRA adapts projections naturally
- Gate controls write selectivity
- Model decides read relevance via softmax

#### 4.3 Training
- Per-chunk processing: each chunk is a separate forward pass
- Answer-only loss on QA tokens
- Memory is the only cross-chunk information channel
- Cosine LR schedule, AdamW

#### 4.4 Compression
- 128 tokens → 1 vector (1536-dim) = 128× compression
- Compared to full KV cache storage or RAG approaches

### 5. Experiments

#### 5.1 Synthetic Factual Recall
- Setup: random entities, codes, years — no parametric knowledge possible
- Single passage: 97.7% token accuracy (vs 15.9% no-memory)
- Scaling: 6 QA pairs (99.2%), long gaps (99.6%)
- Table: full scaling results

#### 5.2 Multi-Passage Recall
- 2 passages: 99.8% / 97% exact match
- Confusable facts (shared attributes): 99.8% — entity binding, not keyword matching
- 5 passages: 100% / 100% from epoch 7
- Table: passage scaling

#### 5.3 Generalization
- Train on single-passage → eval on multi-passage: 98.0%
- Train on single-passage → eval on 5-passage: 98.1%
- Train on synthetic → eval on real Wikipedia (SQuAD): 86.0-90.4%
- The model learns "how to use memory" as a general skill
- Table: cross-format generalization matrix

#### 5.4 Streaming Document QA
- Real Wikipedia articles processed chunk-by-chunk
- 100% on numerical questions, 97.3% on diverse questions
- No-memory baseline: 39.7%
- Zero additional training — uses synthetic checkpoint

#### 5.5 Cross-Document Reasoning
- Comparison: "Who started first?" — requires retrieving 2 years, comparing
- Arithmetic: "Sum of codes?" — requires retrieving 2 numbers, adding
- Zero-shot (not trained on reasoning): 99.8% token / 98% exact match
- Memory naturally supports multi-step reasoning

#### 5.6 Ablations
- n_extract (1 vs 4): marginal improvement
- 1.5B vs 3B: smaller model learns faster
- Memory size: no effect (256=1024=2048) for additive, N/A for cross-attention
- Gate bias sweep
- Two-phase training (for from-scratch models)

### 6. Analysis

#### 6.1 What Gets Stored
- Gate values by chunk type (passage vs filler)
- Compression: what information survives 128× compression
- Numbers nearly 100%, novel subwords ~97-99%

#### 6.2 Leakage Verification
- Each chunk is independent forward pass
- No KV cache reuse (verified experimentally)
- No-memory baseline confirms memory is only cross-chunk channel

#### 6.3 Limitations
- Evaluated on factual recall and simple reasoning
- Fixed chunk size (128 tokens)
- Single write layer, single read layer
- No online learning / weight updates at inference
- Circular buffer eventually overwrites old memories

### 7. Discussion

- From stateless to stateful: what this enables for agents
- Compression vs fidelity tradeoff (like biological memory)
- Path to hierarchical memory, recurrence, goal persistence
- Broader implications for agent architectures

### 8. Conclusion

- Cross-attention is the right interface for memory in pretrained transformers
- 1,537 parameters is enough for a working memory system
- The skill generalizes: train simple, deploy complex
- Foundation for stateful AI agents

### Appendix
- Full hyperparameters
- All 33 experimental findings (condensed)
- Additional ablation results
- Code availability
