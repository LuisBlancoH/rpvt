"""Thinking module: recurrent deliberation over memory before generation.

The model processes chunks and accumulates memory as before. But before
generating a response, it runs N "thinking steps" that:
  1. Cross-attend to memory (read relevant information)
  2. Update a recurrent thought state (accumulate reasoning)
  3. Optionally consolidate memory (rewrite/add derived facts)

This enables multi-step reasoning over stored experiences — the model
can combine facts, detect patterns, and derive new information before
producing any output tokens.

Architecture:
  [chunks processed, memory accumulated]
      ↓
  thought_0 = init(query_hidden_states)
  for step in range(N):
      retrieved = cross_attend(thought, memory)
      thought = GRU(retrieved, thought)
      memory = consolidate(memory, thought)  # optional
      ↓
  thought_N → injected into model for generation

Future extensions:
  - Adaptive halting (PonderNet): learn when to stop thinking
  - Predictive loss: predict next memory state (world model)
  - Counterfactual branches: simulate multiple actions
  - Schema extraction: compress episodes into reusable rules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThinkingModule(nn.Module):
    """Recurrent deliberation over memory.

    Runs N cross-attention steps over the memory bank, accumulating
    a thought state via GRU. The thought state is then used to
    condition generation by injecting into the residual stream.

    Args:
        hidden_size: model hidden dimension (e.g. 1536 for Qwen2.5-1.5B)
        n_think_steps: number of deliberation steps
        inner_dim: cross-attention inner dimension
        n_heads: number of attention heads
        consolidate: whether to update memory during thinking
    """

    def __init__(
        self,
        hidden_size: int,
        n_think_steps: int = 4,
        inner_dim: int = 512,
        n_heads: int = 8,
        consolidate: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_think_steps = n_think_steps
        self.inner_dim = inner_dim
        self.n_heads = n_heads
        self.head_dim = inner_dim // n_heads
        self.consolidate = consolidate

        # Cross-attention: thought → memory
        self.q_proj = nn.Linear(hidden_size, inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, inner_dim, bias=False)
        self.attn_out = nn.Linear(inner_dim, hidden_size, bias=False)

        # Recurrent state update (GRU cell)
        self.gru = nn.GRUCell(hidden_size, hidden_size)

        # Layer norms for stability
        self.ln_thought = nn.LayerNorm(hidden_size)
        self.ln_retrieved = nn.LayerNorm(hidden_size)

        # Optional: memory consolidation
        # Thought state writes back to memory, creating derived facts
        if consolidate:
            self.consolidation_gate = nn.Linear(hidden_size, 1, bias=True)
            nn.init.constant_(self.consolidation_gate.bias, -3.0)  # start conservative
            self.consolidation_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Initialize
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.attn_out.weight, std=0.01)

    def _cross_attend(self, thought, memory):
        """Cross-attention from thought state to memory bank.

        Args:
            thought: (hidden_size,) — current thought state
            memory: (n_mem, hidden_size) — active memory slots

        Returns:
            (hidden_size,) — retrieved information
        """
        n_mem = memory.shape[0]

        # Query from thought
        q = self.q_proj(thought.unsqueeze(0))  # (1, inner_dim)
        q = q.view(1, self.n_heads, self.head_dim)  # (1, n_heads, head_dim)

        # Key/Value from memory
        k = self.k_proj(memory)  # (n_mem, inner_dim)
        v = self.v_proj(memory)  # (n_mem, inner_dim)
        k = k.view(n_mem, self.n_heads, self.head_dim)  # (n_mem, n_heads, head_dim)
        v = v.view(n_mem, self.n_heads, self.head_dim)

        # Attention: (1, n_heads, head_dim) x (n_mem, n_heads, head_dim)^T
        # Per-head dot product
        scale = self.head_dim ** -0.5
        attn_weights = torch.einsum('qnh,mnh->nqm', q, k) * scale  # (n_heads, 1, n_mem)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum
        attn_out = torch.einsum('nqm,mnh->qnh', attn_weights, v)  # (1, n_heads, head_dim)
        attn_out = attn_out.reshape(1, self.inner_dim)

        return self.attn_out(attn_out).squeeze(0)  # (hidden_size,)

    def forward(self, query_hidden, memory_bank):
        """Run N thinking steps over memory.

        Args:
            query_hidden: (hidden_size,) — initial thought seed
                (e.g. mean-pooled hidden states from the question)
            memory_bank: MemoryBank instance with accumulated memories

        Returns:
            thought: (hidden_size,) — final thought state after N steps
            think_states: list of (hidden_size,) — intermediate thought states
                (useful for analysis and future adaptive halting)
        """
        mem_states, n_active = memory_bank.get_active_memories()
        if n_active == 0:
            return query_hidden, [query_hidden]

        mem = mem_states.to(dtype=query_hidden.dtype)
        thought = query_hidden
        think_states = [thought]

        for step in range(self.n_think_steps):
            # 1. Cross-attend to memory
            normed_thought = self.ln_thought(thought)
            retrieved = self._cross_attend(normed_thought, mem)
            retrieved = self.ln_retrieved(retrieved)

            # 2. Update thought state via GRU
            thought = self.gru(retrieved.unsqueeze(0), thought.unsqueeze(0)).squeeze(0)

            # 3. Optional: consolidate memory
            if self.consolidate and hasattr(self, 'consolidation_gate'):
                gate = torch.sigmoid(self.consolidation_gate(thought))
                if gate.item() > 0.1:  # only consolidate if gate is meaningfully open
                    derived = self.consolidation_proj(thought)
                    # Add derived fact as new memory slot
                    # Use the memory bank's write mechanism
                    new_mem = gate * derived
                    mem = torch.cat([mem, new_mem.unsqueeze(0)], dim=0)

            think_states.append(thought)

        return thought, think_states


class ThoughtInjector(nn.Module):
    """Injects thought state into the model's forward pass.

    After thinking, the thought state needs to influence generation.
    This module projects the thought state and adds it to the hidden
    states at a target layer during the generation forward pass.

    Similar to ParallelCrossAttentionWrapper but uses a pre-computed
    thought vector instead of cross-attending to memory on the fly.
    """

    def __init__(self, original_layer, hidden_size: int):
        super().__init__()
        self.layer = original_layer
        self.thought_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln = nn.LayerNorm(hidden_size)

        # Initialize small
        nn.init.normal_(self.thought_proj.weight, std=0.01)

        # Storage for thought state (set before generation forward pass)
        self.thought_state = None

    def set_thought(self, thought):
        """Set the thought state to inject during next forward pass."""
        self.thought_state = thought

    def clear_thought(self):
        """Clear thought state."""
        self.thought_state = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        outputs = self.layer(*args, **kwargs)

        if self.thought_state is not None:
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            # Project thought and add to all positions
            thought_signal = self.thought_proj(self.ln(self.thought_state))
            # Broadcast to all sequence positions
            thought_signal = thought_signal.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden)
            thought_signal = thought_signal.expand_as(hidden_states)
            hidden_states = hidden_states + thought_signal

            if isinstance(outputs, tuple):
                return (hidden_states,) + outputs[1:]
            return hidden_states

        return outputs
