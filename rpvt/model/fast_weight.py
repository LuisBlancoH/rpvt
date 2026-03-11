"""Fast weight memory module.

A linear associative memory that updates on every forward pass.
No backprop at inference — the write rule is learned during pretraining
and frozen at deployment.

Memory update per token:
    key = W_key @ hidden_state
    value = W_value @ hidden_state
    M = decay * M + outer(value, key)

Memory retrieval per token:
    query = W_query @ hidden_state
    retrieved = M @ query

This is the simplest version: single timescale, linear (identity) feature map,
no write gating. Just: does the model learn to use the memory at all?
"""

import torch
import torch.nn as nn


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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.decay = decay

        # Projections: hidden_size -> memory_size
        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
        self.W_key = nn.Linear(hidden_size, memory_size, bias=False)
        self.W_value = nn.Linear(hidden_size, memory_size, bias=False)

        # Output projection: memory_size -> hidden_size
        self.W_out = nn.Linear(memory_size, hidden_size, bias=False)

        # Learnable gate that controls how much memory output mixes in
        self.gate = nn.Parameter(torch.zeros(1))

        # Memory matrix — not a parameter, not saved, starts at zero
        self.register_buffer("M", torch.zeros(memory_size, memory_size))

    def reset_memory(self):
        """Reset memory to zero."""
        self.M.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence, updating memory token by token.

        Args:
            x: (batch, seq_len, hidden_size)

        Returns:
            retrieved: (batch, seq_len, hidden_size) — memory output for each token
            gate_value: scalar gate value (for logging)
        """
        batch, seq_len, _ = x.shape

        query = self.W_query(x)   # (batch, seq_len, memory_size)
        key = self.W_key(x)       # (batch, seq_len, memory_size)
        value = self.W_value(x)   # (batch, seq_len, memory_size)

        # Process token by token (memory is recurrent)
        retrieved_list = []
        M = self.M.clone()  # (memory_size, memory_size)

        for t in range(seq_len):
            # Read: retrieve from current memory
            q_t = query[:, t, :]  # (batch, memory_size)
            r_t = torch.matmul(q_t, M.T)  # (batch, memory_size)
            retrieved_list.append(r_t)

            # Write: update memory with mean over batch
            k_t = key[:, t, :]    # (batch, memory_size)
            v_t = value[:, t, :]  # (batch, memory_size)

            # Outer product, averaged over batch
            outer = torch.einsum("bi,bj->ij", v_t, k_t) / batch
            M = self.decay * M + outer

        # Store updated memory (detach — no gradient through memory across calls)
        self.M = M.detach()

        retrieved = torch.stack(retrieved_list, dim=1)  # (batch, seq_len, memory_size)
        output = self.W_out(retrieved)  # (batch, seq_len, hidden_size)

        gate_value = torch.sigmoid(self.gate)
        return output * gate_value, gate_value


class TransformerLayerWithMemory(nn.Module):
    """Wraps an existing transformer layer, adding fast weight memory.

    The forward pass:
        1. Run the original transformer layer
        2. Read from fast weight memory using the layer's output
        3. Add gated memory output to the residual stream
        4. Write to fast weight memory
    """

    def __init__(self, original_layer: nn.Module, hidden_size: int, memory_size: int = 256, decay: float = 0.99):
        super().__init__()
        self.layer = original_layer
        self.memory = FastWeightMemory(
            hidden_size=hidden_size,
            memory_size=memory_size,
            decay=decay,
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


def attach_fast_weight_memory(layers, hidden_size, layer_indices=None, memory_size=256, decay=0.99):
    """Attach fast weight memory modules to specified transformer layers.

    Args:
        layers: nn.ModuleList of transformer layers (e.g., model.transformer.h)
        hidden_size: Hidden dimension of the model.
        layer_indices: Which layers to add memory to. None = all.
        memory_size: Dimension of key/value/query projections.
        decay: Memory decay rate per token.

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
