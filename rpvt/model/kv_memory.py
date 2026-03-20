"""KV Cache Memory: brain-like pattern reinstatement.

Instead of injecting foreign signals (which frozen models can't use),
store the actual KV cache from passage processing and restore it at
query time. The model "re-experiences" the passage through its own
native representations.

This is how the brain works: the hippocampus stores a compressed index
to the original cortical activity pattern. At retrieval, it reinstates
that pattern. The cortex processes it normally because it IS normal
activity — just replayed.

Architecture:
  During passage processing:
    model(passage_tokens, use_cache=True)
    → KV cache captured at all layers
    → gate selects important token positions
    → selected KV pairs stored in KVMemoryBank

  During question answering:
    stored KV pairs restored as past_key_values
    model(question_tokens, past_key_values=stored_kvs)
    → model's attention naturally attends to stored passage KVs
    → no foreign signals, no LoRA, no adaptation needed
    → 100% compatible because it's the model's own representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVMemoryBank(nn.Module):
    """Stores selected KV cache entries from processed chunks.

    Instead of storing compressed hidden state vectors, stores actual
    key-value pairs from the model's own forward pass. These are
    native representations the model already knows how to attend to.

    The write gate selects which token positions are important enough
    to store (same gating principle as MemoryBank, but stores KVs
    instead of hidden states).

    Args:
        n_layers: number of model layers to store KVs for
        n_kv_heads: number of KV heads per layer
        head_dim: dimension per head
        max_entries: maximum number of token positions to store
        hidden_size: model hidden size (for the gate)
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_entries: int = 128,
        hidden_size: int = 1536,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_entries = max_entries
        self.hidden_size = hidden_size

        # Write gate — decides which token positions to store
        self.gate = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

        # Storage: KV pairs for each layer
        # keys[layer]: (max_entries, n_kv_heads, head_dim)
        # values[layer]: (max_entries, n_kv_heads, head_dim)
        for layer_idx in range(n_layers):
            self.register_buffer(
                f"keys_{layer_idx}",
                torch.zeros(max_entries, n_kv_heads, head_dim),
            )
            self.register_buffer(
                f"values_{layer_idx}",
                torch.zeros(max_entries, n_kv_heads, head_dim),
            )

        self.register_buffer("n_stored", torch.tensor(0, dtype=torch.long))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_tokens_seen", torch.tensor(0, dtype=torch.long))

    def reset(self):
        """Clear all stored KV pairs."""
        for layer_idx in range(self.n_layers):
            getattr(self, f"keys_{layer_idx}").zero_()
            getattr(self, f"values_{layer_idx}").zero_()
        self.n_stored.zero_()
        self.write_ptr.zero_()
        self.total_tokens_seen.zero_()

    def write(self, hidden_states, kv_cache):
        """Store selected KV pairs based on gate importance.

        Args:
            hidden_states: (batch, seq_len, hidden_size) — for computing gate
            kv_cache: DynamicCache from model forward pass
        """
        param_dtype = self.gate.weight.dtype
        x = hidden_states.to(dtype=param_dtype)

        # Compute importance per token
        gate_scores = torch.sigmoid(self.gate(x))  # (batch, seq_len, 1)
        gate_scores = gate_scores.squeeze(0).squeeze(-1)  # (seq_len,)

        # Select top-k most important positions
        seq_len = gate_scores.shape[0]
        n_select = min(seq_len, self.max_entries - self.n_stored.item())
        if n_select <= 0:
            return

        _, top_indices = gate_scores.topk(min(n_select, seq_len))

        # Store KV pairs for selected positions across all layers
        for layer_idx in range(self.n_layers):
            layer_cache = kv_cache.layers[layer_idx]
            # keys shape: (batch, n_kv_heads, seq_len, head_dim)
            layer_keys = layer_cache.keys[0]  # (n_kv_heads, seq_len, head_dim)
            layer_values = layer_cache.values[0]

            # Select important positions
            selected_keys = layer_keys[:, top_indices, :]  # (n_kv_heads, n_select, head_dim)
            selected_values = layer_values[:, top_indices, :]

            # Store (transposed to entries-first)
            start = self.write_ptr.item()
            end = start + len(top_indices)
            keys_buf = getattr(self, f"keys_{layer_idx}")
            values_buf = getattr(self, f"values_{layer_idx}")

            keys_buf[start:end] = selected_keys.transpose(0, 1).detach()
            values_buf[start:end] = selected_values.transpose(0, 1).detach()

        self.write_ptr = self.write_ptr + len(top_indices)
        self.n_stored = torch.clamp(self.write_ptr, max=self.max_entries)

    def store_all(self, kv_cache):
        """Store ALL KV pairs from a forward pass (no gating).

        The model's own attention will decide what's relevant at query time.
        This is the simplest approach — just accumulate KV entries.
        """
        for layer_idx in range(self.n_layers):
            layer_cache = kv_cache.layers[layer_idx]
            layer_keys = layer_cache.keys[0]  # (n_kv_heads, seq_len, head_dim)
            layer_values = layer_cache.values[0]
            seq_len = layer_keys.shape[1]

            n_to_store = min(seq_len, self.max_entries - self.write_ptr.item())
            if n_to_store <= 0:
                break

            start = self.write_ptr.item()
            end = start + n_to_store
            keys_buf = getattr(self, f"keys_{layer_idx}")
            values_buf = getattr(self, f"values_{layer_idx}")

            # Store (transposed: entries first)
            keys_buf[start:end] = layer_keys[:, :n_to_store, :].transpose(0, 1).detach()
            values_buf[start:end] = layer_values[:, :n_to_store, :].transpose(0, 1).detach()

        actual_stored = min(seq_len, self.max_entries - self.write_ptr.item())
        self.write_ptr = self.write_ptr + actual_stored
        self.n_stored = torch.clamp(self.write_ptr, max=self.max_entries)
        self.total_tokens_seen = self.total_tokens_seen + seq_len

    def skip(self, n_tokens):
        """Record that tokens were processed but not stored.

        Updates total_tokens_seen for correct position encoding
        at query time.
        """
        self.total_tokens_seen = self.total_tokens_seen + n_tokens

    def get_past_key_values(self, device, dtype):
        """Reconstruct past_key_values for model forward pass.

        Returns a DynamicCache-compatible object that the model
        can use as past_key_values.

        Returns None if no entries stored.
        """
        n = self.n_stored.item()
        if n == 0:
            return None

        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()

        for layer_idx in range(self.n_layers):
            keys_buf = getattr(self, f"keys_{layer_idx}")[:n]  # (n, n_kv_heads, head_dim)
            values_buf = getattr(self, f"values_{layer_idx}")[:n]

            # Reshape to (batch=1, n_kv_heads, n_entries, head_dim)
            k = keys_buf.transpose(0, 1).unsqueeze(0).to(device=device, dtype=dtype)
            v = values_buf.transpose(0, 1).unsqueeze(0).to(device=device, dtype=dtype)

            cache.update(k, v, layer_idx)

        return cache
