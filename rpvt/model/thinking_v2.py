"""Unified Thinking Module: one mechanism, all modes emerge.

One recurrent attention loop over a unified set of states.
What the attention targets determines the type of thinking:
  - Attend to memory slots → retrieval
  - Attend to working memory → computation
  - Attend to previous thoughts → refinement/settling

The model learns which attention patterns produce correct answers.
We don't program retrieval vs computation — it emerges from training.

Architecture:
  state_set = [memory_slots | working_memory | thought_history]

  for step in range(n_steps):
      thought = cross_attend(query, state_set)
      query = GRU(thought, query)         # update query for next step
      working_memory = update(thought)     # write intermediate results
      thought_history.append(thought)      # record thinking trace

  final_thought → used for answer generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedThinkingModule(nn.Module):
    """One recurrent attention mechanism. Everything emerges.

    The thinking module maintains:
      - A query state (what it's looking for)
      - Working memory (intermediate results)
      - Thought history (previous thinking steps)

    At each step, it cross-attends to ALL available states
    (external memory + working memory + thought history).
    The attention pattern determines what type of thinking happens.

    Args:
        hidden_size: dimension of states
        n_heads: attention heads
        n_work_slots: working memory capacity
        max_think_steps: maximum thinking iterations
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 8,
        n_work_slots: int = 4,
        max_think_steps: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.n_work_slots = n_work_slots
        self.max_think_steps = max_think_steps

        # Cross-attention: query attends to state set
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Query update (GRU — recurrent state)
        self.gru = nn.GRUCell(hidden_size, hidden_size)

        # Working memory write gate
        self.work_gate = nn.Linear(hidden_size * 2, n_work_slots)
        self.work_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Layer norms
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_kv = nn.LayerNorm(hidden_size)
        self.ln_out = nn.LayerNorm(hidden_size)

        # Learned initial working memory
        self.init_work = nn.Parameter(torch.zeros(n_work_slots, hidden_size))

        # Initialize
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02)

    def _cross_attend(self, query, states):
        """Multi-head cross-attention from query to state set.

        Args:
            query: (hidden_size,) — current thought/query
            states: (n_states, hidden_size) — unified state set

        Returns:
            (hidden_size,) — retrieved/computed result
        """
        q = self.q_proj(self.ln_q(query)).unsqueeze(0)  # (1, hidden)
        kv = self.ln_kv(states)
        k = self.k_proj(kv)  # (n_states, hidden)
        v = self.v_proj(kv)

        # Reshape for multi-head
        q = q.view(1, self.n_heads, self.head_dim)
        k = k.view(-1, self.n_heads, self.head_dim)
        v = v.view(-1, self.n_heads, self.head_dim)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('qnh,snh->nqs', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('nqs,snh->qnh', attn, v)

        out = out.reshape(1, self.hidden_size)
        return self.o_proj(out).squeeze(0), attn.squeeze(1)  # (hidden,), (n_heads, n_states)

    def forward(self, query_seed, memory_states=None, n_steps=None):
        """Think: iteratively attend to unified state set.

        Args:
            query_seed: (hidden_size,) — initial thought seed
                (e.g., mean-pooled question hidden states)
            memory_states: (n_mem, hidden_size) — external memory slots
                (from KV cache mean-pooled representations, or MemoryBank)
            n_steps: override max_think_steps

        Returns:
            final_thought: (hidden_size,) — result of thinking
            thinking_trace: list of intermediate thoughts
            attention_trace: list of attention patterns per step
        """
        if n_steps is None:
            n_steps = self.max_think_steps

        device = query_seed.device
        dtype = query_seed.dtype

        # Initialize working memory
        working_mem = self.init_work.to(dtype=dtype)  # (n_work_slots, hidden)

        # Thought history
        thought_history = []
        attention_trace = []

        query = query_seed

        for step in range(n_steps):
            # Build unified state set
            states_list = [working_mem]
            if memory_states is not None:
                states_list.append(memory_states.to(dtype=dtype))
            if thought_history:
                states_list.append(torch.stack(thought_history))

            all_states = torch.cat(states_list, dim=0)  # (n_total, hidden)

            # Cross-attend: query → all states
            thought, attn_weights = self._cross_attend(query, all_states)
            thought = self.ln_out(thought + query)  # residual

            # Update query for next step (GRU)
            query = self.gru(
                thought.unsqueeze(0), query.unsqueeze(0)
            ).squeeze(0)

            # Update working memory
            # Gate decides which working memory slot to write to
            gate_input = torch.cat([thought, query])
            gate = F.softmax(self.work_gate(gate_input), dim=-1)  # (n_work_slots,)
            new_content = self.work_proj(thought)  # (hidden,)

            # Soft write: blend new content into selected slot
            gate_2d = gate.unsqueeze(1)  # (n_work_slots, 1)
            working_mem = working_mem * (1 - gate_2d) + gate_2d * new_content.unsqueeze(0)

            thought_history.append(thought.detach())
            attention_trace.append(attn_weights.detach())

        return query, thought_history, attention_trace
