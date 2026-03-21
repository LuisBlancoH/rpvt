"""Predictive Coding v2: Forward + Inverse with persistent state.

The inverse transformer maintains a persistent hidden state across
chunks — a running world model that encodes accumulated context
and expectations. Prediction errors at each layer tell us what's
new, what's expected, and what's surprising.

No memory. No recall. Just prediction and understanding.

Components:
  Forward: frozen instruct model (bottom-up)
  Inverse: trainable, persistent state (top-down)

The inverse has a GRU that accumulates context across chunks.
Its predictions for each layer are conditioned on this accumulated
context — not just the current output, but everything it's seen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseLayer(nn.Module):
    """Single layer of the inverse transformer."""

    def __init__(self, hidden_size, n_heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, n_heads, batch_first=True, bias=False
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        ffn_dim = hidden_size * 2
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size, bias=False),
        )

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class PersistentInverseTransformer(nn.Module):
    """Inverse transformer with persistent state across chunks.

    The GRU accumulates context: after processing chunk 1, the state
    encodes "I've seen a briefing about VIPER-371." This state
    conditions predictions for chunk 2 — the inverse EXPECTS
    certain patterns and is surprised by others.

    Args:
        hidden_size: must match forward model
        n_inverse_layers: depth of inverse transformer
        target_layers: which forward layers to predict
        state_dim: dimension of persistent context state
        n_heads: attention heads
    """

    def __init__(
        self,
        hidden_size: int,
        n_inverse_layers: int = 3,
        target_layers: list = None,
        state_dim: int = 512,
        n_heads: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_layers = target_layers or [7, 14, 21]
        self.state_dim = state_dim

        # Persistent context state (GRU accumulates across chunks)
        self.context_gru = nn.GRUCell(hidden_size, state_dim)
        self.context_proj = nn.Linear(state_dim, hidden_size, bias=False)

        # Inverse transformer layers
        self.layers = nn.ModuleList([
            InverseLayer(hidden_size, n_heads)
            for _ in range(n_inverse_layers)
        ])

        # Per-target prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(len(self.target_layers))
        ])

        # Persistent state
        self.persistent_state = None

        # Initialize small
        for head in self.prediction_heads:
            nn.init.normal_(head.weight, std=0.01)
        nn.init.normal_(self.context_proj.weight, std=0.02)

    def reset_state(self):
        """Reset persistent state (new document/session)."""
        self.persistent_state = None

    def update_context(self, output_hidden):
        """Update persistent state with new chunk information.

        Args:
            output_hidden: (batch, seq_len, hidden_size) from forward output layer
        """
        # Pool the chunk to a single vector
        chunk_summary = output_hidden.mean(dim=1).squeeze(0)  # (hidden_size,)

        # Compress to state_dim for GRU
        if self.persistent_state is not None:
            state = self.persistent_state
        else:
            state = torch.zeros(
                self.state_dim,
                device=chunk_summary.device,
                dtype=chunk_summary.dtype,
            )

        # Update persistent state
        # GRU input needs (1, hidden_size) → compress first
        gru_input = chunk_summary[:self.state_dim].unsqueeze(0)
        if gru_input.shape[1] < self.state_dim:
            gru_input = F.pad(gru_input, (0, self.state_dim - gru_input.shape[1]))

        new_state = self.context_gru(gru_input, state.unsqueeze(0))
        self.persistent_state = new_state.squeeze(0).detach()

    def predict(self, output_hidden):
        """Generate predictions for target layers.

        Predictions are conditioned on:
        1. The current chunk's output (what just happened)
        2. The persistent state (what happened before)

        Args:
            output_hidden: (batch, seq_len, hidden_size)

        Returns:
            predictions: {layer_idx: (batch, seq_len, hidden_size)}
        """
        x = output_hidden

        # Condition on persistent state if available
        if self.persistent_state is not None:
            context = self.context_proj(self.persistent_state)  # (hidden_size,)
            # Add context to every position
            context = context.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
            x = x + context.expand_as(x)

        # Run inverse layers, generate predictions
        predictions = {}
        targets_rev = list(reversed(self.target_layers))
        heads_rev = list(reversed(list(self.prediction_heads)))

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(targets_rev):
                predictions[targets_rev[i]] = heads_rev[i](x)

        return predictions


class HiddenStateCapture(nn.Module):
    """Wraps a forward layer to capture hidden states for error computation.

    Transparent — doesn't modify the forward pass at all.
    Just stores the output for later comparison with inverse predictions.
    """

    def __init__(self, original_layer, layer_idx):
        super().__init__()
        self.layer = original_layer
        self.layer_idx = layer_idx
        self.captured = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        outputs = self.layer(*args, **kwargs)
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        self.captured = hidden.detach()
        return outputs  # completely unchanged


class PredictiveCodingSystem(nn.Module):
    """Forward + Inverse with persistent state.

    No memory. No modulation. Pure predictive coding.
    The inverse learns to predict the forward model and builds
    a persistent world model across chunks.
    """

    def __init__(self, forward_model, inverse, target_layers, captures):
        super().__init__()
        self.forward_model = forward_model
        self.inverse = inverse
        self.target_layers = target_layers
        self.captures = {c.layer_idx: c for c in captures}

    def reset(self):
        """Reset for new document."""
        self.inverse.reset_state()

    def process_chunk(self, input_ids):
        """Process one chunk: forward → capture → predict → errors → update state.

        Returns:
            output: model output
            errors: {layer_idx: error_tensor}
            magnitudes: {layer_idx: float}
        """
        # Forward pass (captures hidden states via wrappers)
        output = self.forward_model(input_ids)

        # Get captured hidden states
        hidden_dict = {
            li: c.captured for li, c in self.captures.items()
            if c.captured is not None
        }

        # Get the output layer's hidden states for inverse input
        # Use the last target layer's captured states as proxy for output
        max_layer = max(self.target_layers)
        if max_layer in hidden_dict:
            output_hidden = hidden_dict[max_layer]
        else:
            return output, {}, {}

        # Inverse predicts (conditioned on persistent state)
        predictions = self.inverse.predict(output_hidden)

        # Compute errors
        errors = {}
        magnitudes = {}
        for layer_idx in self.target_layers:
            if layer_idx in predictions and layer_idx in hidden_dict:
                actual = hidden_dict[layer_idx]
                pred = predictions[layer_idx]
                error = actual - pred
                errors[layer_idx] = error
                magnitudes[layer_idx] = error.pow(2).mean().item()

        # Update persistent state
        self.inverse.update_context(output_hidden)

        return output, errors, magnitudes

    def prediction_loss(self, input_ids):
        """Compute prediction loss for training the inverse.

        Forward pass → capture states → inverse predicts → MSE loss.
        Only the inverse gets gradients (forward is frozen).
        """
        with torch.no_grad():
            self.forward_model(input_ids)

        hidden_dict = {
            li: c.captured for li, c in self.captures.items()
            if c.captured is not None
        }

        max_layer = max(self.target_layers)
        if max_layer not in hidden_dict:
            return torch.tensor(0.0)

        output_hidden = hidden_dict[max_layer]
        predictions = self.inverse.predict(output_hidden)

        loss = torch.tensor(0.0, device=input_ids.device)
        n = 0
        for layer_idx in self.target_layers:
            if layer_idx in predictions and layer_idx in hidden_dict:
                actual = hidden_dict[layer_idx].detach()
                pred = predictions[layer_idx]
                loss = loss + F.mse_loss(pred, actual)
                n += 1

        # Update persistent state (detached — don't backprop through state)
        self.inverse.update_context(output_hidden.detach())

        return loss / max(n, 1)
