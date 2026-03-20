"""Memory decoder: dedicated pathway from memory bank to model input.

Brain-inspired architecture: the hippocampus communicates to the cortex
through a dedicated output pathway with its own parameters. The cortex
doesn't need to change — it just receives processed input.

The MemoryDecoder is a small transformer that:
1. Cross-attends to memory bank (reads relevant memories)
2. Processes through 2-3 self-attention layers (organizes information)
3. Produces N output tokens (rich context for the main model)
4. These tokens are prepended to the main model's input

The main model is 100% frozen. The decoder learns to "translate" memory
into a format the model naturally understands. Trained end-to-end
through the answer loss.

This is NOT soft prompts (which are just 8 learned vectors).
This is a full transformer decoder with cross-attention to memory,
producing 32+ rich context tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MemoryDecoderLayer(nn.Module):
    """Single layer: self-attention + cross-attention to memory + FFN."""

    def __init__(self, hidden_size, n_heads=8, ffn_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        if ffn_dim is None:
            ffn_dim = hidden_size * 4

        # Self-attention (between output tokens)
        self.self_attn_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_attn_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_attn_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_attn_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_self = nn.LayerNorm(hidden_size)

        # Cross-attention (output tokens → memory)
        self.cross_attn_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_attn_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_attn_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_attn_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_cross = nn.LayerNorm(hidden_size)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size, bias=False),
        )
        self.ln_ffn = nn.LayerNorm(hidden_size)

    def _attention(self, q, k, v):
        """Multi-head attention."""
        bsz = q.shape[0]
        seq_q = q.shape[1]
        seq_k = k.shape[1]

        q = q.view(bsz, seq_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_k, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        return out.transpose(1, 2).contiguous().view(bsz, seq_q, self.hidden_size)

    def forward(self, x, memory):
        """
        Args:
            x: (batch, n_tokens, hidden_size) — output tokens being built
            memory: (batch, n_mem, hidden_size) — memory bank contents

        Returns:
            x: (batch, n_tokens, hidden_size) — updated output tokens
        """
        # Self-attention
        residual = x
        x_norm = self.ln_self(x)
        q = self.self_attn_q(x_norm)
        k = self.self_attn_k(x_norm)
        v = self.self_attn_v(x_norm)
        x = residual + self.self_attn_o(self._attention(q, k, v))

        # Cross-attention to memory
        residual = x
        x_norm = self.ln_cross(x)
        q = self.cross_attn_q(x_norm)
        k = self.cross_attn_k(memory)
        v = self.cross_attn_v(memory)
        x = residual + self.cross_attn_o(self._attention(q, k, v))

        # FFN
        residual = x
        x = residual + self.ffn(self.ln_ffn(x))

        return x


class MemoryDecoder(nn.Module):
    """Dedicated transformer that reads memory and produces context tokens.

    Like the hippocampal output pathway: takes raw memory vectors and
    translates them into a format the main model naturally understands.

    Architecture:
        learned_queries (n_output_tokens, hidden_size)
            ↓
        DecoderLayer × n_layers
          - self-attention between output tokens
          - cross-attention to memory bank
          - FFN
            ↓
        output_tokens (n_output_tokens, hidden_size)
            ↓
        prepend to main model input

    Args:
        hidden_size: must match main model's hidden size
        n_output_tokens: number of context tokens to produce
        n_layers: depth of the decoder
        n_heads: attention heads per layer
    """

    def __init__(
        self,
        hidden_size: int,
        n_output_tokens: int = 32,
        n_layers: int = 2,
        n_heads: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_output_tokens = n_output_tokens

        # Learned output queries — these become the context tokens
        self.output_queries = nn.Parameter(
            torch.randn(n_output_tokens, hidden_size) * 0.02
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            MemoryDecoderLayer(hidden_size, n_heads=n_heads)
            for _ in range(n_layers)
        ])

        # Final projection + layer norm
        self.output_ln = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, memory_states, n_active):
        """Read from memory and produce context tokens.

        Args:
            memory_states: (n_active, hidden_size) active memory slots
            n_active: number of active memory slots

        Returns:
            context: (1, n_output_tokens, hidden_size) tokens to prepend
            Returns None if no active memories
        """
        if n_active == 0:
            return None

        # Prepare inputs
        mem = memory_states.unsqueeze(0).to(dtype=self.output_queries.dtype)
        # (1, n_mem, hidden)

        x = self.output_queries.unsqueeze(0)
        # (1, n_output_tokens, hidden)

        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, mem)

        # Final normalization
        x = self.output_ln(x)

        return x  # (1, n_output_tokens, hidden_size)

    def param_count(self):
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


class DecoderInjectionWrapper(nn.Module):
    """Wraps a transformer layer to inject decoded memory as extra KV pairs.

    Like MemoryAugmentedAttention but uses decoded tokens from the
    MemoryDecoder instead of raw memory vectors. The decoder has already
    processed and organized the memory — these are rich, contextualized
    representations, not raw compressed hidden states.

    The decoded tokens are projected through the layer's own k_proj/v_proj
    (same as the original cross-attention approach) but the input is
    much higher quality because the decoder organized it.
    """

    def __init__(self, original_layer, decoder, memory_bank):
        super().__init__()
        self.layer = original_layer
        self.decoder = decoder
        self.memory_bank = memory_bank

        # Cache decoded memory to avoid recomputing for each layer call
        self._cached_context = None
        self._cache_valid = False

    def invalidate_cache(self):
        """Call after memory changes (new chunk processed)."""
        self._cached_context = None
        self._cache_valid = False

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def _get_decoded_memory(self):
        """Get decoded memory tokens, using cache if valid."""
        if self._cache_valid and self._cached_context is not None:
            return self._cached_context

        mem_states, n_active = self.memory_bank.get_active_memories()
        if n_active == 0:
            self._cached_context = None
            self._cache_valid = True
            return None

        context = self.decoder(mem_states, n_active)
        self._cached_context = context
        self._cache_valid = True
        return context

    def forward(self, *args, **kwargs):
        """Run normal layer forward, then add decoded memory to residual.

        Uses the ParallelCrossAttention pattern: run the layer normally,
        then add a cross-attention output from decoded memory. The decoder
        produces rich context that gets cross-attended to by the layer output.
        """
        outputs = self.layer(*args, **kwargs)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        decoded = self._get_decoded_memory()
        if decoded is not None:
            # Cross-attend: hidden_states query → decoded memory KV
            decoded = decoded.to(dtype=hidden_states.dtype)
            bsz, seq_len, hdim = hidden_states.shape
            n_decoded = decoded.shape[1]

            # Simple scaled dot-product attention
            # Q from hidden states, K/V from decoded memory
            scale = hdim ** -0.5
            attn = torch.matmul(hidden_states, decoded.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)  # (bsz, seq_len, n_decoded)
            mem_output = torch.matmul(attn, decoded)  # (bsz, seq_len, hdim)

            # Scale down to not overwhelm — decoder learns the right magnitude
            hidden_states = hidden_states + 0.1 * mem_output

            if isinstance(outputs, tuple):
                return (hidden_states,) + outputs[1:]
            return hidden_states

        return outputs
