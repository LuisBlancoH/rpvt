"""KV Cache Compressor/Decompressor — brain-like memory compression.

Like the hippocampus: compress full cortical activity into a sparse
representation, then reconstruct at recall time. The cortex processes
the reconstruction natively because it's in the right format.

Compressor: full KV cache (128 tokens × 28 layers) → compressed vector(s)
Decompressor: compressed vector(s) → approximate KV cache

The model stays frozen. Only the compressor and decompressor are trained.
Loss: reconstruction error on KV pairs + answer accuracy.

This gives both:
  - Compression (128 tokens → N compressed slots, where N << 128)
  - Compatibility (decompressed KVs are in the model's native format)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCompressor(nn.Module):
    """Compresses full KV cache into a small set of memory vectors.

    Takes the KV pairs from all layers for a chunk and compresses
    them into n_compressed slots. Each slot captures a different
    aspect of the chunk content.

    Args:
        n_layers: number of model layers
        n_kv_heads: KV heads per layer
        head_dim: dimension per head
        n_compressed: number of compressed memory slots (e.g. 8)
        compress_dim: internal compression dimension
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        n_compressed: int = 8,
        compress_dim: int = 256,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_compressed = n_compressed

        # Per-token feature: flatten KV across a few representative layers
        # Don't use all 28 layers — too expensive. Sample 4 layers.
        self.sample_layers = [0, 7, 14, 21]  # evenly spaced
        n_sample = len(self.sample_layers)
        kv_feature_dim = n_sample * n_kv_heads * head_dim * 2  # keys + values

        # Compress: pool across tokens using learned queries
        self.compress_queries = nn.Parameter(
            torch.randn(n_compressed, compress_dim) * 0.02
        )
        self.token_proj = nn.Linear(kv_feature_dim, compress_dim, bias=False)
        self.compress_ln = nn.LayerNorm(compress_dim)

        # Initialize
        nn.init.normal_(self.token_proj.weight, std=0.02)

    def forward(self, kv_cache):
        """Compress a full KV cache into n_compressed memory vectors.

        Args:
            kv_cache: DynamicCache from model forward pass

        Returns:
            compressed: (n_compressed, compress_dim) memory vectors
        """
        # Extract KV features from sampled layers
        features = []
        for layer_idx in self.sample_layers:
            layer = kv_cache.layers[layer_idx]
            k = layer.keys[0]   # (n_kv_heads, seq_len, head_dim)
            v = layer.values[0]  # (n_kv_heads, seq_len, head_dim)
            seq_len = k.shape[1]
            # Flatten: (seq_len, n_kv_heads * head_dim)
            k_flat = k.transpose(0, 1).reshape(seq_len, -1)
            v_flat = v.transpose(0, 1).reshape(seq_len, -1)
            features.append(k_flat)
            features.append(v_flat)

        # (seq_len, kv_feature_dim)
        token_features = torch.cat(features, dim=-1).to(dtype=self.token_proj.weight.dtype)

        # Project to compress_dim
        token_features = self.token_proj(token_features)  # (seq_len, compress_dim)

        # Cross-attention: queries attend to tokens → compressed slots
        scale = self.compress_queries.shape[-1] ** -0.5
        attn = torch.matmul(
            self.compress_queries, token_features.transpose(-2, -1)
        ) * scale  # (n_compressed, seq_len)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum → compressed vectors
        compressed = torch.matmul(attn, token_features)  # (n_compressed, compress_dim)
        compressed = self.compress_ln(compressed + self.compress_queries)  # residual

        return compressed


class KVDecompressor(nn.Module):
    """Reconstructs approximate KV cache from compressed memory vectors.

    Takes compressed memory slots and produces KV pairs for all layers.
    The reconstructed KVs can be used as past_key_values in the model.

    Args:
        n_layers: number of model layers
        n_kv_heads: KV heads per layer
        head_dim: dimension per head
        n_decompressed: number of KV positions to produce
        compress_dim: dimension of compressed vectors
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        n_decompressed: int = 16,
        compress_dim: int = 256,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_decompressed = n_decompressed

        kv_out_dim = n_kv_heads * head_dim

        # Learned position queries — one per output position
        self.position_queries = nn.Parameter(
            torch.randn(n_decompressed, compress_dim) * 0.02
        )

        # Cross-attention: position queries attend to compressed memory
        self.query_ln = nn.LayerNorm(compress_dim)

        # Per-layer KV projection (each layer gets different KVs)
        self.key_projections = nn.ModuleList([
            nn.Linear(compress_dim, kv_out_dim, bias=False)
            for _ in range(n_layers)
        ])
        self.value_projections = nn.ModuleList([
            nn.Linear(compress_dim, kv_out_dim, bias=False)
            for _ in range(n_layers)
        ])

        # Initialize small
        for proj in list(self.key_projections) + list(self.value_projections):
            nn.init.normal_(proj.weight, std=0.02)

    def forward(self, compressed, device, dtype):
        """Reconstruct KV cache from compressed memory.

        Args:
            compressed: (n_compressed, compress_dim) memory vectors
            device: target device
            dtype: target dtype

        Returns:
            DynamicCache with reconstructed KV pairs, or None
        """
        if compressed is None:
            return None

        from transformers.cache_utils import DynamicCache

        compressed = compressed.to(dtype=self.position_queries.dtype)

        # Cross-attend: position queries → compressed memory
        q = self.query_ln(self.position_queries)
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, compressed.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Per-position features
        position_features = torch.matmul(attn, compressed)  # (n_decompressed, compress_dim)

        # Reconstruct KV pairs for each layer
        cache = DynamicCache()

        for layer_idx in range(self.n_layers):
            # Project to key/value space
            keys = self.key_projections[layer_idx](position_features)
            values = self.value_projections[layer_idx](position_features)

            # Reshape: (n_decompressed, n_kv_heads * head_dim) → (1, n_kv_heads, n_decompressed, head_dim)
            k = keys.view(self.n_decompressed, self.n_kv_heads, self.head_dim)
            v = values.view(self.n_decompressed, self.n_kv_heads, self.head_dim)
            k = k.transpose(0, 1).unsqueeze(0).to(device=device, dtype=dtype)
            v = v.transpose(0, 1).unsqueeze(0).to(device=device, dtype=dtype)

            cache.update(k, v, layer_idx)

        return cache


class KVMemorySystem(nn.Module):
    """Complete compress → store → decompress pipeline.

    Wraps compressor + decompressor + circular buffer storage.
    Drop-in replacement for KVMemoryBank that adds compression.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        n_slots: int = 64,
        n_compressed: int = 8,
        n_decompressed: int = 16,
        compress_dim: int = 256,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.n_compressed = n_compressed
        self.compress_dim = compress_dim

        self.compressor = KVCompressor(
            n_layers, n_kv_heads, head_dim,
            n_compressed=n_compressed,
            compress_dim=compress_dim,
        )
        self.decompressor = KVDecompressor(
            n_layers, n_kv_heads, head_dim,
            n_decompressed=n_decompressed,
            compress_dim=compress_dim,
        )

        # Circular buffer for compressed vectors
        self.register_buffer(
            "memory", torch.zeros(n_slots, n_compressed, compress_dim)
        )
        self.register_buffer("n_stored", torch.tensor(0, dtype=torch.long))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_tokens_seen", torch.tensor(0, dtype=torch.long))

    def reset(self):
        self.memory.zero_()
        self.n_stored.zero_()
        self.write_ptr.zero_()
        self.total_tokens_seen.zero_()

    def store(self, kv_cache, seq_len=128):
        """Compress and store a chunk's KV cache."""
        compressed = self.compressor(kv_cache)  # (n_compressed, compress_dim)

        slot = self.write_ptr.item() % self.n_slots
        self.memory[slot] = compressed.detach()
        self.write_ptr = self.write_ptr + 1
        self.n_stored = torch.clamp(self.write_ptr, max=self.n_slots)
        self.total_tokens_seen = self.total_tokens_seen + seq_len

        return compressed  # for reconstruction loss during training

    def skip(self, n_tokens):
        """Record skipped tokens for position tracking."""
        self.total_tokens_seen = self.total_tokens_seen + n_tokens

    def reconstruct(self, device, dtype):
        """Decompress all stored memory into a KV cache."""
        n = self.n_stored.item()
        if n == 0:
            return None

        # Collect all stored compressed vectors
        all_compressed = self.memory[:n]  # (n_slots_used, n_compressed, compress_dim)

        # Flatten: treat all compressed slots as one set of memory
        flat_compressed = all_compressed.reshape(-1, self.compress_dim)
        # (n_slots_used * n_compressed, compress_dim)

        return self.decompressor(flat_compressed, device, dtype)

    def get_n_decompressed(self):
        """Return total number of decompressed positions."""
        return self.decompressor.n_decompressed
