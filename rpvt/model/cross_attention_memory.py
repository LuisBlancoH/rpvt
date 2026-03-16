"""Cross-attention memory injection for pretrained transformers.

Instead of additive residual injection (which pretrained models ignore),
memory is injected as extra KV pairs into a transformer layer's attention.
The model's own attention mechanism naturally decides whether to attend
to memory or regular context.

Architecture:
  Layer N (write): hidden states → gate → store in MemoryBank
  Layer N+1 (read): self_attn gets extra KV pairs from MemoryBank

Memory KVs are projected through the read layer's own k_proj/v_proj,
so LoRA adaptations apply naturally. No RoPE on memory KVs (content-based
retrieval, not position-based).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """Circular buffer storing gated hidden states for cross-attention.

    Stores raw hidden states (not projected) so the read layer can
    project them through its own k_proj/v_proj.
    """

    def __init__(
        self,
        hidden_size: int,
        n_slots: int = 64,
        gate_bias: float = -2.0,
        decay: float = 0.999,
        n_extract: int = 1,
        eviction: str = "circular",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        self.decay = decay
        self.n_extract = n_extract
        self.eviction = eviction  # "circular" or "importance"

        # Write gate
        self.W_gate = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.W_gate.weight)
        nn.init.constant_(self.W_gate.bias, gate_bias)

        # Learned extraction queries: k queries that cross-attend over chunk tokens
        # to produce k diverse hidden state vectors instead of mean-pooling
        if n_extract > 1:
            self.extract_queries = nn.Parameter(
                torch.randn(n_extract, hidden_size) * 0.02
            )

        # Storage for raw hidden states
        self.register_buffer("mem_states", torch.zeros(n_slots, hidden_size))
        self.register_buffer("mem_strength", torch.zeros(n_slots))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))

        self.persistent_grad = False

    def reset(self):
        self.mem_states.zero_()
        self.mem_strength.zero_()
        self.write_ptr.zero_()

    def _get_write_slot(self, write_ptr, mem_strength, param_dtype):
        """Get the slot index to write to based on eviction strategy."""
        if self.eviction == "importance" and write_ptr >= self.n_slots:
            # Buffer full — evict the slot with lowest strength
            slot_idx = mem_strength.argmin()
        else:
            # Circular or buffer not yet full
            slot_idx = write_ptr % self.n_slots
        return slot_idx

    def write(self, hidden_states: torch.Tensor):
        """Store gated hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_size) from write layer output
        """
        param_dtype = self.W_gate.weight.dtype
        x = hidden_states.to(dtype=param_dtype)

        gate = torch.sigmoid(self.W_gate(x))  # (batch, seq_len, 1)
        weights = gate.squeeze(-1)  # (batch, seq_len)
        write_str = weights.mean().item()

        # Working copies for autograd compatibility
        mem_states = self.mem_states.clone().to(dtype=param_dtype)
        mem_strength = self.mem_strength.clone().to(dtype=param_dtype)
        write_ptr = self.write_ptr.clone()

        # Decay
        seq_len = hidden_states.shape[1]
        mem_strength = mem_strength * (self.decay ** seq_len)

        if self.n_extract > 1:
            # Learned extraction: k queries cross-attend over chunk tokens
            # to produce k diverse hidden state vectors
            eq = self.extract_queries.to(dtype=param_dtype)  # (k, hidden_size)
            extract_scores = torch.matmul(
                eq.unsqueeze(0), x.transpose(-1, -2)
            ) / (self.hidden_size ** 0.5)  # (batch, k, seq_len)

            # Apply gate as mask on extraction attention
            gate_mask = weights.unsqueeze(1)  # (batch, 1, seq_len)
            extract_scores = extract_scores + torch.log(gate_mask.clamp(min=1e-8))

            extract_weights = F.softmax(extract_scores, dim=-1)  # (batch, k, seq_len)

            # Extract k summary vectors in hidden space
            extracted = torch.matmul(extract_weights, x)  # (batch, k, hidden_size)
            # Mean across batch
            extracted = extracted.mean(dim=0)  # (k, hidden_size)

            # Write k slots
            for ei in range(self.n_extract):
                slot_idx = self._get_write_slot(write_ptr, mem_strength, param_dtype)
                mask = F.one_hot(slot_idx, self.n_slots).to(dtype=param_dtype)
                mask_2d = mask.unsqueeze(1)
                mem_states = mem_states * (1 - mask_2d) + mask_2d * extracted[ei].unsqueeze(0)
                mem_strength = mem_strength * (1 - mask) + mask * write_str
                write_ptr = write_ptr + 1
        else:
            # Single aggregated write (gate-weighted mean-pool)
            w_sum = weights.sum(dim=(0, 1)).clamp(min=1e-8)
            aggregated = (x * weights.unsqueeze(-1)).sum(dim=(0, 1)) / w_sum  # (hidden_size,)

            slot_idx = self._get_write_slot(write_ptr, mem_strength, param_dtype)
            mask = F.one_hot(slot_idx, self.n_slots).to(dtype=param_dtype)
            mask_2d = mask.unsqueeze(1)
            mem_states = mem_states * (1 - mask_2d) + mask_2d * aggregated.unsqueeze(0)
            mem_strength = mem_strength * (1 - mask) + mask * write_str
            write_ptr = write_ptr + 1

        # Store
        if self.persistent_grad:
            self.mem_states = mem_states
            self.mem_strength = mem_strength
            self.write_ptr = write_ptr.detach()
        else:
            self.mem_states = mem_states.detach()
            self.mem_strength = mem_strength.detach()
            self.write_ptr = write_ptr.detach()

    def get_active_memories(self):
        """Return active memory states and count.

        Returns:
            mem: (n_active, hidden_size) tensor of active memories
            n_active: number of active slots
        """
        active_mask = self.mem_strength > 1e-8
        n_active = active_mask.sum().item()
        if n_active == 0:
            return None, 0
        return self.mem_states[active_mask], int(n_active)

    def detach_state(self):
        self.mem_states = self.mem_states.detach()
        self.mem_strength = self.mem_strength.detach()


class WriteWrapper(nn.Module):
    """Wraps a transformer layer — passes output through unchanged, then writes to memory."""

    def __init__(self, original_layer, memory_bank: MemoryBank):
        super().__init__()
        self.layer = original_layer
        self.memory_bank = memory_bank

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        outputs = self.layer(*args, **kwargs)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        self.memory_bank.write(hidden_states)
        return outputs  # pass through unchanged — no additive injection


def _get_rotary_and_attention_fns(attn_module):
    """Dynamically import RoPE and attention functions for the model type."""
    module_name = type(attn_module).__module__
    if "qwen2" in module_name:
        from transformers.models.qwen2.modeling_qwen2 import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        return apply_rotary_pos_emb, eager_attention_forward
    elif "llama" in module_name:
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        return apply_rotary_pos_emb, eager_attention_forward
    elif "phi" in module_name:
        from transformers.models.phi.modeling_phi import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        return apply_rotary_pos_emb, eager_attention_forward
    elif "mistral" in module_name:
        from transformers.models.mistral.modeling_mistral import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        return apply_rotary_pos_emb, eager_attention_forward
    else:
        # Generic fallback — most HF models have these in their module
        import importlib
        mod = importlib.import_module(module_name)
        return mod.apply_rotary_pos_emb, mod.eager_attention_forward


class MemoryAugmentedAttention(nn.Module):
    """Wraps any HuggingFace attention layer to inject memory KV pairs.

    Memory hidden states are projected through the layer's own k_proj/v_proj,
    then concatenated to the regular KV pairs before attention computation.
    No RoPE applied to memory KVs (content-based, position-independent).

    The model's own query heads naturally decide whether to attend to
    memory or regular context via softmax attention weights.

    Supports Qwen2, Llama, Phi, Mistral, and other HF models with
    standard q_proj/k_proj/v_proj/o_proj attention layers.
    """

    def __init__(self, original_attn, memory_bank: MemoryBank):
        super().__init__()
        self.attn = original_attn
        self.memory_bank = memory_bank
        self._rope_fn, self._attn_fn = _get_rotary_and_attention_fns(original_attn)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.attn, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask=None,
        **kwargs,
    ):
        apply_rotary_pos_emb = self._rope_fn
        eager_attention_forward = self._attn_fn

        input_shape = hidden_states.shape[:-1]
        batch_size = input_shape[0]
        hidden_shape = (*input_shape, -1, self.attn.head_dim)

        # Regular Q, K, V
        query_states = self.attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply RoPE to regular tokens only
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Inject memory KV pairs (no RoPE — content-based retrieval)
        mem_states, n_active = self.memory_bank.get_active_memories()
        if n_active > 0:
            mem_hidden = mem_states.unsqueeze(0).to(dtype=hidden_states.dtype)  # (1, n_mem, hidden)
            head_dim = self.attn.head_dim
            # Derive n_kv_heads from k_proj output size
            kv_dim = self.attn.k_proj.out_features
            n_kv_heads = kv_dim // head_dim

            # Project through same k_proj, v_proj (with LoRA if applied)
            mem_k = self.attn.k_proj(mem_hidden)  # (1, n_mem, kv_dim)
            mem_v = self.attn.v_proj(mem_hidden)  # (1, n_mem, kv_dim)

            mem_k = mem_k.view(1, n_active, n_kv_heads, head_dim).transpose(1, 2)
            mem_v = mem_v.view(1, n_active, n_kv_heads, head_dim).transpose(1, 2)

            # Concatenate: [memory_KVs | regular_KVs]
            key_states = torch.cat(
                [mem_k.expand(batch_size, -1, -1, -1), key_states], dim=2
            )
            value_states = torch.cat(
                [mem_v.expand(batch_size, -1, -1, -1), value_states], dim=2
            )

            # Extend attention mask — memory positions always attendable
            if attention_mask is not None:
                mem_mask = torch.zeros(
                    *attention_mask.shape[:-1], n_active,
                    device=attention_mask.device, dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([mem_mask, attention_mask], dim=-1)

        # Standard attention computation (handles GQA via repeat_kv)
        attn_output, attn_weights = eager_attention_forward(
            self.attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attn.attention_dropout,
            scaling=self.attn.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.attn.o_proj(attn_output)
        return attn_output, attn_weights
