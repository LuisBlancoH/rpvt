"""Pattern Reinstatement Memory — the simplest brain-like predictive coding.

Store hidden states from passage processing at target layers.
At query time, compute error = stored_pattern - current_state.
Modulate the forward pass with this error.

This IS predictive coding:
  - Stored pattern = top-down prediction (what the layer should look like)
  - Current state = bottom-up processing (what the layer produces now)
  - Error = prediction error (what's missing from memory)
  - Modulation = pattern reinstatement (shift toward remembered state)

No inverse transformer. No learned predictor. The stored hidden states
ARE the prediction. The model's own representations, stored and replayed.

Combined with KV cache for native attention-based recall.
"""

import torch
import torch.nn as nn


class PatternMemoryBank(nn.Module):
    """Stores hidden state patterns at target layers for reinstatement.

    Like the hippocampus: stores compressed cortical activity patterns.
    At recall, the error between stored and current pattern IS the
    memory signal.

    Args:
        hidden_size: model hidden dimension
        n_slots: max number of patterns to store per layer
        target_layers: which layers to store patterns for
    """

    def __init__(self, hidden_size, n_slots=64, target_layers=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        self.target_layers = target_layers or [7, 14, 21]

        for layer_idx in self.target_layers:
            self.register_buffer(
                f"patterns_{layer_idx}",
                torch.zeros(n_slots, hidden_size)
            )
        self.register_buffer("n_stored", torch.tensor(0, dtype=torch.long))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))

    def reset(self):
        for layer_idx in self.target_layers:
            getattr(self, f"patterns_{layer_idx}").zero_()
        self.n_stored.zero_()
        self.write_ptr.zero_()

    def store(self, hidden_states_dict):
        """Store mean-pooled hidden states from target layers.

        Args:
            hidden_states_dict: {layer_idx: (batch, seq_len, hidden)} for target layers
        """
        slot = self.write_ptr.item() % self.n_slots
        for layer_idx in self.target_layers:
            if layer_idx in hidden_states_dict:
                h = hidden_states_dict[layer_idx]
                pooled = h.mean(dim=(0, 1))  # (hidden_size,)
                buf = getattr(self, f"patterns_{layer_idx}")
                buf[slot] = pooled.detach()

        self.write_ptr = self.write_ptr + 1
        self.n_stored = torch.clamp(self.write_ptr, max=self.n_slots)

    def get_pattern(self, layer_idx):
        """Get mean pattern across all stored slots for a layer.

        Returns the "remembered" cortical pattern — what this layer
        should look like when the passage is being processed.
        """
        n = self.n_stored.item()
        if n == 0:
            return None
        buf = getattr(self, f"patterns_{layer_idx}")[:n]
        return buf.mean(dim=0)  # (hidden_size,) averaged across stored chunks


class PatternModulationWrapper(nn.Module):
    """Wraps a transformer layer with pattern-based error modulation.

    During query processing:
      current = what the layer produces from the question
      stored = what the layer produced from the passage (from memory)
      error = stored - current = what the question is missing
      modulated = current + scale * error

    When no stored pattern → no modulation → exact original behavior.
    """

    def __init__(self, original_layer, layer_idx, pattern_memory, scale=0.1):
        super().__init__()
        self.layer = original_layer
        self.layer_idx = layer_idx
        self.pattern_memory = pattern_memory
        self.scale = nn.Parameter(torch.tensor(scale))
        self.modulate = False  # only modulate during QA, not during passage processing

    def set_modulate(self, enabled):
        self.modulate = enabled

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        outputs = self.layer(*args, **kwargs)

        if self.modulate:
            stored_pattern = self.pattern_memory.get_pattern(self.layer_idx)
            if stored_pattern is not None:
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs

                # Current state: mean-pool current hidden states
                current_pattern = hidden.mean(dim=1, keepdim=True)  # (batch, 1, hidden)

                # Error: what's missing (stored - current)
                error = stored_pattern.to(dtype=hidden.dtype) - current_pattern.squeeze(1)

                # Modulate: shift ALL positions toward the stored pattern
                # Broadcast error across sequence positions
                modulation = self.scale * error.unsqueeze(1).expand_as(hidden)
                hidden = hidden + modulation

                if isinstance(outputs, tuple):
                    return (hidden,) + outputs[1:]
                return hidden

        return outputs
