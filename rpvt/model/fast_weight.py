"""Fast weight memory module.

A linear associative memory that updates on every forward pass.
No backprop at inference — the write rule is learned during pretraining
and frozen at deployment.

Memory update per token:
    key = W_key @ hidden_state
    value = W_value @ hidden_state
    M = decay * M + write_strength * outer(value, key)

Write modes:
    "uniform":  write_strength = 1 (all tokens write equally)
    "gate":     write_strength = sigmoid(W_gate @ hidden_state) (learned gate)
    "surprise": write_strength = sigmoid(scale * ||error|| + bias)
                where error = hidden_state - W_out(M @ query)
                Surprise-driven: write strongly when M's prediction is wrong.

Memory retrieval per token:
    query = W_query @ hidden_state
    retrieved = M @ query
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastWeightMemory(nn.Module):
    """Single-timescale linear fast weight memory.

    Maintains a matrix M that accumulates outer products of (value, key) pairs.
    Retrieval is linear: M @ query.
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 256,
        decay: float = 0.99,
        write_mode: str = "uniform",
        # Legacy compat
        use_write_gate: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.decay = decay

        # Handle legacy use_write_gate parameter
        if use_write_gate and write_mode == "uniform":
            write_mode = "gate"
        self.write_mode = write_mode

        # Projections: hidden_size -> memory_size
        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
        self.W_key = nn.Linear(hidden_size, memory_size, bias=False)
        self.W_value = nn.Linear(hidden_size, memory_size, bias=False)

        # Output projection: memory_size -> hidden_size
        # Zero-initialized so memory starts with no contribution,
        # and the model learns to use it through W_out weights directly.
        self.W_out = nn.Linear(memory_size, hidden_size, bias=False)
        nn.init.zeros_(self.W_out.weight)

        # Write gate: learns which tokens are worth writing to M
        if write_mode == "gate":
            self.use_write_gate = True  # legacy compat
            self.W_gate = nn.Linear(hidden_size, 1, bias=True)
            nn.init.zeros_(self.W_gate.weight)
            nn.init.constant_(self.W_gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12
        else:
            self.use_write_gate = write_mode == "gate"

        # Surprise-driven write strength
        if write_mode == "surprise":
            # Two learned scalars: how to map error magnitude to write strength
            # sigmoid(scale * (||error|| / sqrt(hidden_size)) + bias)
            # ||error|| / sqrt(hidden_size) ≈ 1.0 for typical hidden states
            # So sigmoid(1.0 * 1.0 + (-2.0)) = sigmoid(-1.0) ≈ 0.27 baseline
            # As W_out trains and predictions improve, errors shrink → writes decrease
            self.surprise_scale = nn.Parameter(torch.tensor(1.0))
            self.surprise_bias = nn.Parameter(torch.tensor(-2.0))
            self.error_norm_scale = math.sqrt(hidden_size)  # not learned, just normalization

        # Memory matrix — not a parameter, not saved, starts at zero
        self.register_buffer("M", torch.zeros(memory_size, memory_size))

    def reset_memory(self):
        """Reset memory to zero."""
        self.M.zero_()

    def forward(self, x: torch.Tensor, chunk_size: int = 64) -> tuple[torch.Tensor, float | None]:
        """Process a sequence in chunks, updating memory between chunks.

        Within each chunk, all tokens read from the same M state (beginning of chunk).
        After the chunk, M is updated with all writes from that chunk.

        Args:
            x: (batch, seq_len, hidden_size)
            chunk_size: Number of tokens to process at once.

        Returns:
            output: (batch, seq_len, hidden_size) — memory output to add to residual
            mean_ws: mean write strength for logging (None if uniform)
        """
        batch, seq_len, _ = x.shape

        query = self.W_query(x)   # (batch, seq_len, memory_size)
        key = self.W_key(x)       # (batch, seq_len, memory_size)
        value = self.W_value(x)   # (batch, seq_len, memory_size)

        # L2-normalize keys and values to prevent M from exploding
        key = F.normalize(key, dim=-1)
        value = F.normalize(value, dim=-1)

        # Pre-compute gate write strengths (only for "gate" mode)
        if self.write_mode == "gate":
            gate_strengths = torch.sigmoid(self.W_gate(x))  # (batch, seq_len, 1)

        retrieved_chunks = []
        write_strength_log = []
        M = self.M.clone()  # (memory_size, memory_size)

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            c_len = chunk_end - chunk_start

            # ── READ: all tokens in this chunk read from current M ──
            q_chunk = query[:, chunk_start:chunk_end, :]   # (batch, c_len, memory_size)
            r_chunk = torch.matmul(q_chunk, M.T)           # (batch, c_len, memory_size)
            retrieved_chunks.append(r_chunk)

            # ── COMPUTE WRITE STRENGTH ──
            k_chunk = key[:, chunk_start:chunk_end, :]     # (batch, c_len, memory_size)
            v_chunk = value[:, chunk_start:chunk_end, :]   # (batch, c_len, memory_size)

            if self.write_mode == "uniform":
                pass  # no scaling

            elif self.write_mode == "gate":
                ws_chunk = gate_strengths[:, chunk_start:chunk_end, :]
                k_chunk = k_chunk * ws_chunk
                v_chunk = v_chunk * ws_chunk
                write_strength_log.append(ws_chunk.mean().item())

            elif self.write_mode == "surprise":
                # Prediction: what does M think the hidden state should be?
                prediction = self.W_out(r_chunk)  # (batch, c_len, hidden_size)
                actual = x[:, chunk_start:chunk_end, :]  # (batch, c_len, hidden_size)
                error = actual - prediction

                # Normalized error magnitude per token
                # Divide by sqrt(hidden_size) so typical values are ~1.0
                error_norm = error.norm(dim=-1, keepdim=True) / self.error_norm_scale

                # Map normalized error to write strength via learned scale + bias
                ws_chunk = torch.sigmoid(
                    self.surprise_scale * error_norm + self.surprise_bias
                )  # (batch, c_len, 1)

                k_chunk = k_chunk * ws_chunk
                v_chunk = v_chunk * ws_chunk
                write_strength_log.append(ws_chunk.mean().item())

            # ── WRITE ──
            chunk_outer = torch.einsum("bci,bcj->ij", v_chunk, k_chunk) / (batch * c_len)
            M = (self.decay ** c_len) * M + chunk_outer

        # Store updated memory (detach — no gradient through memory across calls)
        self.M = M.detach()

        retrieved = torch.cat(retrieved_chunks, dim=1)  # (batch, seq_len, memory_size)
        output = self.W_out(retrieved)  # (batch, seq_len, hidden_size)

        mean_ws = sum(write_strength_log) / len(write_strength_log) if write_strength_log else None
        return output, mean_ws


class TransformerLayerWithMemory(nn.Module):
    """Wraps an existing transformer layer, adding fast weight memory.

    The forward pass:
        1. Run the original transformer layer
        2. Read from fast weight memory using the layer's output
        3. Add gated memory output to the residual stream
        4. Write to fast weight memory
    """

    def __init__(self, original_layer: nn.Module, hidden_size: int, memory_size: int = 256,
                 decay: float = 0.99, write_mode: str = "uniform", use_write_gate: bool = False):
        super().__init__()
        self.layer = original_layer
        self.memory = FastWeightMemory(
            hidden_size=hidden_size,
            memory_size=memory_size,
            decay=decay,
            write_mode=write_mode,
            use_write_gate=use_write_gate,
        )

    def forward(self, *args, **kwargs):
        """Run the original layer, then add memory output."""
        # Run original transformer layer
        outputs = self.layer(*args, **kwargs)

        # Handle both tuple outputs (Qwen) and direct tensor outputs (GPT-2)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        # Read from and write to memory
        memory_output, gate_value = self.memory(hidden_states)

        # Add memory output to hidden states
        modified = hidden_states + memory_output

        # Return in same format as original layer
        if isinstance(outputs, tuple):
            return (modified,) + outputs[1:]
        else:
            return modified

    def reset_memory(self):
        self.memory.reset_memory()


def attach_fast_weight_memory(layers, hidden_size, layer_indices=None, memory_size=256, decay=0.99,
                              write_mode="uniform", use_write_gate=False):
    """Attach fast weight memory modules to specified transformer layers.

    Args:
        layers: nn.ModuleList of transformer layers (e.g., model.transformer.h)
        hidden_size: Hidden dimension of the model.
        layer_indices: Which layers to add memory to. None = all.
        memory_size: Dimension of key/value/query projections.
        decay: Memory decay rate per token.
        write_mode: "uniform", "gate", or "surprise".

    Returns:
        List of FastWeightMemory modules (for parameter access).
    """
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    memory_modules = []
    for idx in layer_indices:
        original_layer = layers[idx]
        wrapped = TransformerLayerWithMemory(
            original_layer,
            hidden_size=hidden_size,
            memory_size=memory_size,
            decay=decay,
            write_mode=write_mode,
            use_write_gate=use_write_gate,
        )
        # Move to same device/dtype as model
        device = next(original_layer.parameters()).device
        dtype = next(original_layer.parameters()).dtype
        wrapped.memory = wrapped.memory.to(device=device, dtype=dtype)
        layers[idx] = wrapped
        memory_modules.append(wrapped.memory)

    return memory_modules


def remove_fast_weight_memory(layers, layer_indices=None):
    """Remove fast weight memory modules, restoring original layers."""
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    for idx in layer_indices:
        layer = layers[idx]
        if isinstance(layer, TransformerLayerWithMemory):
            layers[idx] = layer.layer


def reset_all_memories(layers, layer_indices=None):
    """Reset all fast weight memories to zero."""
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    for idx in layer_indices:
        layer = layers[idx]
        if isinstance(layer, TransformerLayerWithMemory):
            layer.reset_memory()
