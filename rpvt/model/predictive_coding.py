"""Recurrent Predictive Coding for Transformers.

The frozen transformer runs bottom-up. A trainable inverse transformer
runs top-down, predicting each layer's hidden states. Prediction errors
modulate the activations on a second forward pass — injecting memory
and context information WITHOUT changing any weights.

When no prediction errors exist → zero modulation → exact instruct model.
When errors exist → modulation proportional to surprise → memory-augmented.

Architecture:
  Pass 1 (observe):   input → frozen transformer → hidden states h₀...h₂₈
  Inverse (predict):  h₂₈ → inverse transformer → predictions p₀, p₇, p₁₄, p₂₁
  Errors:             e = h - p at target layers
  Pass 2 (modulate):  input → frozen transformer + error modulation → output

The inverse transformer uses the same operations as the forward
(self-attention + FFN) but flows in the opposite direction through
the layer hierarchy. Trained end-to-end from answer loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseTransformerLayer(nn.Module):
    """Single layer of the inverse (top-down) transformer.
    Same operations as forward: self-attention + FFN + residual."""

    def __init__(self, hidden_size, n_heads=8, ffn_mult=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, n_heads, batch_first=True, bias=False,
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_mult, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size * ffn_mult, hidden_size, bias=False),
        )

    def forward(self, x):
        normed = self.ln1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class InverseTransformer(nn.Module):
    """Top-down transformer that predicts target layer hidden states.

    Takes the output of the forward transformer and generates predictions
    for what specific layers should have produced. The predictions flow
    top-down: output → layer 21 → layer 14 → layer 7 → layer 0.

    Args:
        hidden_size: must match the forward model
        n_inverse_layers: number of inverse transformer layers
        target_layers: which forward layers to predict
        n_heads: attention heads per inverse layer
    """

    def __init__(
        self,
        hidden_size: int,
        n_inverse_layers: int = 4,
        target_layers: list = None,
        n_heads: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_layers = target_layers or [0, 7, 14, 21]
        self.n_targets = len(self.target_layers)

        # Inverse transformer layers
        self.layers = nn.ModuleList([
            InverseTransformerLayer(hidden_size, n_heads=n_heads)
            for _ in range(n_inverse_layers)
        ])

        # Per-target projection heads
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(self.n_targets)
        ])

        # Initialize small
        for head in self.prediction_heads:
            nn.init.normal_(head.weight, std=0.01)

    def forward(self, output_hidden):
        """Generate predictions for target layers.

        Args:
            output_hidden: (batch, seq_len, hidden_size) from last forward layer

        Returns:
            predictions: dict mapping layer_idx → (batch, seq_len, hidden_size)
        """
        x = output_hidden
        predictions = {}

        # Process top-down, generating predictions at each step
        targets_reversed = list(reversed(self.target_layers))
        heads_reversed = list(reversed(list(self.prediction_heads)))

        for inv_layer_idx, inv_layer in enumerate(self.layers):
            x = inv_layer(x)

            if inv_layer_idx < self.n_targets:
                target = targets_reversed[inv_layer_idx]
                head = heads_reversed[inv_layer_idx]
                predictions[target] = head(x)

        return predictions


class ModulationWrapper(nn.Module):
    """Wraps a frozen transformer layer to add prediction error modulation.

    During the modulated forward pass, adds the prediction error
    (scaled) to the layer's hidden states. When no error is set,
    the layer behaves exactly as the original.
    """

    def __init__(self, original_layer, layer_idx, scale=0.1):
        super().__init__()
        self.layer = original_layer
        self.layer_idx = layer_idx
        self.scale = nn.Parameter(torch.tensor(scale))
        self.error = None  # set before modulated forward pass

    def set_error(self, error):
        self.error = error

    def clear_error(self):
        self.error = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        outputs = self.layer(*args, **kwargs)

        if self.error is not None:
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            # Modulate by prediction error — zero error = zero change
            hidden = hidden + self.scale * self.error
            self.error = None  # clear after use

            if isinstance(outputs, tuple):
                return (hidden,) + outputs[1:]
            return hidden

        return outputs


class RecurrentPredictiveCoding(nn.Module):
    """Complete predictive coding system.

    Orchestrates the forward model, inverse transformer, and modulation.
    Handles the observe → predict → error → modulate cycle.
    """

    def __init__(self, model, inverse_transformer, target_layers,
                 modulation_wrappers, kv_memory=None):
        super().__init__()
        self.model = model
        self.inverse = inverse_transformer
        self.target_layers = target_layers
        self.wrappers = {w.layer_idx: w for w in modulation_wrappers}
        self.kv_memory = kv_memory

    def observe(self, input_ids):
        """Pass 1: Run forward model, capture hidden states."""
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True,
                use_cache=True,
            )
        return outputs

    def predict_and_error(self, hidden_states):
        """Run inverse transformer, compute per-layer errors."""
        # Get the final hidden state
        final_hidden = hidden_states[-1]

        # Inverse transformer predicts target layers
        predictions = self.inverse(final_hidden)

        # Compute errors at each target layer
        errors = {}
        error_magnitudes = {}
        for layer_idx, prediction in predictions.items():
            actual = hidden_states[layer_idx]
            error = actual.detach() - prediction
            errors[layer_idx] = error
            error_magnitudes[layer_idx] = error.pow(2).mean().item()

        return errors, error_magnitudes

    def modulated_forward(self, input_ids, errors, past_key_values=None,
                          position_ids=None, attention_mask=None):
        """Pass 2: Forward with prediction error modulation."""
        # Set errors on modulation wrappers
        for layer_idx, error in errors.items():
            if layer_idx in self.wrappers:
                self.wrappers[layer_idx].set_error(error)

        # Forward pass — modulation happens inside wrappers
        kwargs = {}
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        output = self.model(input_ids, **kwargs)

        # Clear any unused errors
        for wrapper in self.wrappers.values():
            wrapper.clear_error()

        return output

    def forward(self, input_ids, n_cycles=2, past_key_values=None,
                position_ids=None, attention_mask=None):
        """Full predictive coding cycle: observe → predict → modulate.

        Args:
            input_ids: token ids
            n_cycles: number of predict-modulate cycles
            past_key_values: stored KV cache (memory)
            position_ids: for KV cache offset
            attention_mask: for KV cache

        Returns:
            output: model output from final modulated pass
            all_errors: list of error magnitudes per cycle
        """
        all_errors = []

        # Pass 1: observe (no modulation)
        obs = self.observe(input_ids)
        hidden_states = obs.hidden_states

        for cycle in range(n_cycles):
            # Predict + compute errors
            errors, magnitudes = self.predict_and_error(hidden_states)
            all_errors.append(magnitudes)

            # Pass 2+: modulated forward
            output = self.modulated_forward(
                input_ids, errors,
                past_key_values=past_key_values,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            # For next cycle, use modulated hidden states
            if cycle < n_cycles - 1:
                with torch.no_grad():
                    mod_obs = self.model(
                        input_ids, output_hidden_states=True,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                    )
                hidden_states = mod_obs.hidden_states

        return output, all_errors
