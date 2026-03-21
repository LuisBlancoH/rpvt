"""KV Cache Autoencoder: compress and decompress KV pairs.

The KV cache is the model's own native representation (74% recall on
frozen instruct). But it's too large (3.5MB per chunk for Qwen2.5-1.5B).

This autoencoder compresses KV pairs into a compact latent space,
then decompresses them for use as past_key_values. Trained end-to-end
with answer loss — the encoder learns WHAT information matters, not
just pixel-perfect reconstruction.

Like the hippocampus: stores a compressed index that's good enough
to reinstate the cortical pattern when needed.

Architecture:
  Encoder: KV pairs → hidden states → compressed vectors
  Decoder: compressed vectors → reconstructed KV pairs

  Per token: 28 layers × 2 heads × 128 dim × 2 (K+V) = 14,336 values
  Compressed to: 1 vector of size `latent_dim` (e.g., 1536)
  Compression ratio: ~10x per token

  Additionally, we compress across tokens:
  128 tokens → `n_latent` vectors (e.g., 16)
  Total compression: 128 × 14,336 → 16 × 1,536 = ~75x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVEncoder(nn.Module):
    """Compress KV pairs into latent vectors.

    Takes KV pairs from all layers for a chunk of tokens and produces
    a small number of latent vectors.
    """

    def __init__(self, n_layers=28, n_kv_heads=2, head_dim=128,
                 latent_dim=1536, n_latent=16):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.n_latent = n_latent

        # Per-token KV size: n_layers × n_kv_heads × head_dim × 2 (K+V)
        self.kv_dim = n_layers * n_kv_heads * head_dim * 2

        # Project each token's KV pairs to latent_dim
        self.token_proj = nn.Linear(self.kv_dim, latent_dim, bias=False)

        # Learned extraction queries to compress seq_len tokens → n_latent
        self.extract_queries = nn.Parameter(
            torch.randn(n_latent, latent_dim) * 0.02
        )
        self.extract_attn = nn.MultiheadAttention(
            latent_dim, num_heads=8, batch_first=True, bias=False,
        )
        self.extract_norm = nn.LayerNorm(latent_dim)

    def forward(self, kv_pairs):
        """Encode KV pairs into compressed latent vectors.

        Args:
            kv_pairs: dict with keys 'keys_{i}' and 'values_{i}'
                      each of shape (n_tokens, n_kv_heads, head_dim)
                      OR a list of (keys, values) tuples per layer

        Returns:
            latent: (1, n_latent, latent_dim) compressed representation
        """
        # Flatten KV pairs: (n_tokens, kv_dim)
        all_kvs = []
        if isinstance(kv_pairs, dict):
            for i in range(self.n_layers):
                k = kv_pairs[f"keys_{i}"]    # (n_tokens, n_kv_heads, head_dim)
                v = kv_pairs[f"values_{i}"]
                all_kvs.append(k.reshape(k.shape[0], -1))
                all_kvs.append(v.reshape(v.shape[0], -1))
        else:
            # List of (keys, values) per layer
            for keys, values in kv_pairs:
                all_kvs.append(keys.reshape(keys.shape[0], -1))
                all_kvs.append(values.reshape(values.shape[0], -1))

        flat_kv = torch.cat(all_kvs, dim=-1)  # (n_tokens, kv_dim)
        flat_kv = flat_kv.unsqueeze(0)  # (1, n_tokens, kv_dim)

        # Project to latent dim
        token_latent = self.token_proj(flat_kv)  # (1, n_tokens, latent_dim)

        # Cross-attention: n_latent queries extract from n_tokens
        queries = self.extract_queries.unsqueeze(0)  # (1, n_latent, latent_dim)
        queries = self.extract_norm(queries)
        latent, _ = self.extract_attn(
            queries, token_latent, token_latent,
            need_weights=False,
        )

        return latent  # (1, n_latent, latent_dim)


class KVDecoder(nn.Module):
    """Decompress latent vectors back into KV pairs.

    Produces KV pairs for all layers that can be used as past_key_values.
    """

    def __init__(self, n_layers=28, n_kv_heads=2, head_dim=128,
                 latent_dim=1536, n_output_tokens=32):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_output_tokens = n_output_tokens

        self.kv_dim = n_layers * n_kv_heads * head_dim * 2

        # Learned output queries: produce n_output_tokens positions
        self.output_queries = nn.Parameter(
            torch.randn(n_output_tokens, latent_dim) * 0.02
        )
        self.decode_attn = nn.MultiheadAttention(
            latent_dim, num_heads=8, batch_first=True, bias=False,
        )
        self.decode_norm = nn.LayerNorm(latent_dim)

        # Project back to KV pairs
        self.output_proj = nn.Linear(latent_dim, self.kv_dim, bias=False)

    def forward(self, latent):
        """Decode latent vectors into KV pairs.

        Args:
            latent: (1, n_latent, latent_dim) from encoder

        Returns:
            list of (keys, values) tuples per layer
            keys/values shape: (1, n_kv_heads, n_output_tokens, head_dim)
        """
        # Cross-attention: output queries read from latent
        queries = self.output_queries.unsqueeze(0)  # (1, n_output, latent_dim)
        queries = self.decode_norm(queries)
        decoded, _ = self.decode_attn(
            queries, latent, latent,
            need_weights=False,
        )

        # Project to KV pairs
        kv_flat = self.output_proj(decoded)  # (1, n_output, kv_dim)
        kv_flat = kv_flat.squeeze(0)  # (n_output, kv_dim)

        # Reshape into per-layer KV pairs
        kv_per_layer = kv_flat.reshape(
            self.n_output_tokens,
            self.n_layers, 2, self.n_kv_heads, self.head_dim,
        )

        result = []
        for layer_idx in range(self.n_layers):
            keys = kv_per_layer[:, layer_idx, 0, :, :]   # (n_out, n_heads, head_dim)
            values = kv_per_layer[:, layer_idx, 1, :, :]

            # Reshape to (1, n_kv_heads, n_output_tokens, head_dim)
            keys = keys.permute(1, 0, 2).unsqueeze(0)
            values = values.permute(1, 0, 2).unsqueeze(0)
            result.append((keys, values))

        return result


class KVAutoencoder(nn.Module):
    """Full autoencoder: encode KV cache → latent → decode KV cache.

    Trained end-to-end: encode passage KVs, decode, use as past_key_values,
    check if model can still answer correctly.
    """

    def __init__(self, n_layers=28, n_kv_heads=2, head_dim=128,
                 latent_dim=1536, n_latent=16, n_output_tokens=32):
        super().__init__()
        self.encoder = KVEncoder(
            n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
            latent_dim=latent_dim, n_latent=n_latent,
        )
        self.decoder = KVDecoder(
            n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
            latent_dim=latent_dim, n_output_tokens=n_output_tokens,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  KVAutoencoder: latent={n_latent}×{latent_dim}, "
              f"output_tokens={n_output_tokens}, params={n_params:,}")
        print(f"  Compression: {n_layers}L × 128 tokens → "
              f"{n_latent} latent → {n_output_tokens} output tokens")

    def forward(self, kv_pairs):
        """Encode then decode KV pairs.

        Args:
            kv_pairs: dict or list of per-layer (keys, values)

        Returns:
            reconstructed: list of (keys, values) per layer
            latent: the compressed representation
        """
        latent = self.encoder(kv_pairs)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, kv_pairs):
        return self.encoder(kv_pairs)

    def decode(self, latent):
        return self.decoder(latent)

    def to_past_key_values(self, reconstructed):
        """Convert decoder output to DynamicCache-compatible past_key_values."""
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for layer_idx, (keys, values) in enumerate(reconstructed):
            cache.update(keys, values, layer_idx)
        return cache
