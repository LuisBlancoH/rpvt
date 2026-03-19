"""Predictive coding memory for pretrained transformers.

Instead of a learned gate bias, memory writes are controlled by prediction
error — only surprising information gets stored. This naturally suppresses
filler (predictable) and amplifies novel facts (unpredictable).

The predictor maintains a running state (GRU) that models the input stream.
At each chunk, it predicts what the chunk will contain. The prediction error
magnitude serves as the write gate — replacing the fixed bias gate.

When used with LoRA on instruct models, the prediction error can also scale
the training loss per-token, so:
  - Tokens the model already predicts correctly → low error → tiny gradient → instruct preserved
  - Tokens requiring memory → high error → full gradient → memory learned
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveMemoryBank(nn.Module):
    """Memory bank with prediction-error gated writes.

    Replaces MemoryBank's learned gate with a prediction-error gate.
    The predictor learns the statistics of the input stream, so:
      - Filler chunks are predictable → low error → weak write
      - Novel facts are unpredictable → high error → strong write

    Args:
        hidden_size: model hidden dimension
        n_slots: number of memory slots
        pred_dim: predictor hidden dimension
        decay: memory decay rate per chunk
    """

    def __init__(
        self,
        hidden_size: int,
        n_slots: int = 64,
        pred_dim: int = 512,
        decay: float = 0.999,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        self.pred_dim = pred_dim
        self.decay = decay

        # Predictor: GRU that maintains running state of input stream
        self.pred_gru = nn.GRUCell(hidden_size, pred_dim)
        # Project prediction state → predicted hidden state
        self.pred_proj = nn.Linear(pred_dim, hidden_size, bias=False)
        # Compress input for GRU update
        self.input_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Surprise = normalized L2 prediction error (not learned — can't collapse)
        # Scaling factor learned to calibrate the magnitude
        self.surprise_scale = nn.Parameter(torch.tensor(1.0))

        # Storage
        self.register_buffer("mem_states", torch.zeros(n_slots, hidden_size))
        self.register_buffer("mem_strength", torch.zeros(n_slots))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))

        # Prediction state (not a buffer — needs to persist across chunks but reset per doc)
        self.pred_state = None

        self.persistent_grad = False

        # Initialize
        nn.init.normal_(self.pred_proj.weight, std=0.02)
        nn.init.normal_(self.input_proj.weight, std=0.02)

    def reset(self):
        """Reset memory and prediction state for new document."""
        self.mem_states.zero_()
        self.mem_strength.zero_()
        self.write_ptr.zero_()
        self.pred_state = None

    def predict(self, dtype, device):
        """Generate prediction for next chunk.

        Returns:
            predicted: (hidden_size,) predicted hidden state, or zeros if no state
        """
        if self.pred_state is not None:
            return self.pred_proj(self.pred_state)
        return torch.zeros(self.hidden_size, dtype=dtype, device=device)

    def write(self, hidden_states: torch.Tensor):
        """Process chunk: predict, compute error, store, update predictor.

        Args:
            hidden_states: (batch, seq_len, hidden_size) from write layer

        Returns:
            surprise: scalar surprise magnitude for this chunk
            prediction_error: (hidden_size,) error vector
            predicted: (hidden_size,) what was predicted
        """
        param_dtype = self.pred_proj.weight.dtype
        x = hidden_states.to(dtype=param_dtype)

        # Pool chunk to single vector
        actual = x.mean(dim=(0, 1))  # (hidden_size,)

        # Predict
        predicted = self.predict(param_dtype, actual.device)

        # Compute prediction error
        prediction_error = actual - predicted

        # Surprise = normalized L2 prediction error (can't collapse to 0)
        error_norm = prediction_error.norm() / (self.hidden_size ** 0.5)
        surprise = torch.sigmoid(self.surprise_scale * error_norm)  # [0, 1]
        surprise_val = surprise.item()

        # Write to memory — surprise replaces the fixed gate
        mem_states = self.mem_states.clone().to(dtype=param_dtype)
        mem_strength = self.mem_strength.clone().to(dtype=param_dtype)
        write_ptr = self.write_ptr.clone()

        # Decay
        seq_len = hidden_states.shape[1]
        mem_strength = mem_strength * (self.decay ** seq_len)

        # Write: store actual hidden state, weighted by surprise
        slot_idx = write_ptr % self.n_slots
        mask = F.one_hot(slot_idx, self.n_slots).to(dtype=param_dtype)
        mask_2d = mask.unsqueeze(1)

        # Weight the stored representation by surprise
        weighted_actual = actual * surprise.squeeze()
        mem_states = mem_states * (1 - mask_2d) + mask_2d * weighted_actual.unsqueeze(0)
        mem_strength = mem_strength * (1 - mask) + mask * surprise_val
        write_ptr = write_ptr + 1

        # Update prediction state
        actual_for_gru = self.input_proj(actual).unsqueeze(0)
        if self.pred_state is not None:
            new_pred = self.pred_gru(actual_for_gru, self.pred_state.unsqueeze(0))
        else:
            new_pred = self.pred_gru(
                actual_for_gru,
                torch.zeros(1, self.pred_dim, dtype=param_dtype, device=actual.device),
            )
        self.pred_state = new_pred.squeeze(0)

        # Store
        if self.persistent_grad:
            self.mem_states = mem_states
            self.mem_strength = mem_strength
            self.write_ptr = write_ptr.detach()
        else:
            self.mem_states = mem_states.detach()
            self.mem_strength = mem_strength.detach()
            self.write_ptr = write_ptr.detach()
            self.pred_state = self.pred_state.detach()

        return surprise.squeeze(), prediction_error, predicted

    def get_active_memories(self):
        """Return active memory states and count."""
        active_mask = self.mem_strength > 1e-8
        n_active = active_mask.sum().item()
        if n_active == 0:
            return None, 0
        return self.mem_states[active_mask], int(n_active)

    def detach_state(self):
        """Detach all state from computation graph."""
        self.mem_states = self.mem_states.detach()
        self.mem_strength = self.mem_strength.detach()
        if self.pred_state is not None:
            self.pred_state = self.pred_state.detach()


class PredictiveWriteWrapper(nn.Module):
    """Wraps a transformer layer — writes to predictive memory after forward."""

    def __init__(self, original_layer, memory: PredictiveMemoryBank):
        super().__init__()
        self.layer = original_layer
        self.memory = memory

        # Store last surprise for logging/loss scaling
        self.last_surprise = 0.0
        self.last_pred_error = None
        self.last_predicted = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        outputs = self.layer(*args, **kwargs)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        surprise, pred_error, predicted = self.memory.write(hidden_states)
        self.last_surprise = surprise
        self.last_pred_error = pred_error
        self.last_predicted = predicted

        return outputs  # pass through unchanged
