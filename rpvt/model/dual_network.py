"""Dual Network Predictive Coding.

Two networks, one signal, everything emerges.

Forward transformer (instruct + LoRA): processes input, generates output
Inverse transformer (trainable): predicts forward's hidden states top-down

Errors between them drive: memory, thinking, learning, attention.

Training:
  Inverse: fast learner, predicts forward hidden states (prediction loss)
  LoRA: slow learner, makes representations more predictable (consistency loss)
  Both: answer loss on memory recall tasks (task performance)

The LoRA changes are constrained by consistency — not wild distortion
toward specific answers, but gentle structuring toward predictability.

Operating modes:
  AWAKE: forward + inverse → errors → memory selection + modulation
  THINKING: multiple cycles until errors settle
  SLEEP: replay + slow LoRA updates at high-error layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseLayer(nn.Module):
    """Single layer of the inverse (top-down) transformer."""

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


class InverseTransformer(nn.Module):
    """Top-down transformer predicting forward hidden states.

    Takes the output of the forward pass and generates predictions
    for target layers, flowing top-down through inverse layers.
    """

    def __init__(self, hidden_size, n_layers=4, target_layers=None, n_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_layers = target_layers or [0, 7, 14, 21]

        self.layers = nn.ModuleList([
            InverseLayer(hidden_size, n_heads) for _ in range(n_layers)
        ])
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(len(self.target_layers))
        ])

        for head in self.prediction_heads:
            nn.init.normal_(head.weight, std=0.01)

    def forward(self, output_hidden):
        """Predict target layer hidden states from the output.

        Args:
            output_hidden: (batch, seq_len, hidden_size)

        Returns:
            predictions: dict {layer_idx: (batch, seq_len, hidden_size)}
        """
        x = output_hidden
        predictions = {}

        targets_rev = list(reversed(self.target_layers))
        heads_rev = list(reversed(list(self.prediction_heads)))

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(targets_rev):
                predictions[targets_rev[i]] = heads_rev[i](x)

        return predictions


class DualModulationWrapper(nn.Module):
    """Wraps a forward layer to add error modulation during QA.

    During encoding: passes through unchanged, captures hidden states.
    During QA: modulates activations by prediction error from inverse.
    """

    def __init__(self, original_layer, layer_idx):
        super().__init__()
        self.layer = original_layer
        self.layer_idx = layer_idx
        self.scale = nn.Parameter(torch.tensor(0.0))  # start at zero — no modulation
        self.error = None
        self.modulate_enabled = False
        self.last_hidden = None  # for capturing hidden states during encoding

    def enable_modulation(self):
        self.modulate_enabled = True

    def disable_modulation(self):
        self.modulate_enabled = False
        self.error = None

    def set_error(self, error):
        self.error = error

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        outputs = self.layer(*args, **kwargs)
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs

        # Always capture for inverse transformer training
        self.last_hidden = hidden.detach()

        # Modulate only during QA, only if error exists
        if self.modulate_enabled and self.error is not None:
            modulation = torch.tanh(self.scale) * self.error
            hidden = hidden + modulation
            self.error = None

            if isinstance(outputs, tuple):
                return (hidden,) + outputs[1:]
            return hidden

        return outputs


class DualNetworkSystem(nn.Module):
    """Complete dual network predictive coding system.

    Manages the forward model, inverse transformer, modulation wrappers,
    and the error computation. Supports awake, thinking, and sleep modes.
    """

    def __init__(self, forward_model, inverse_transformer,
                 target_layers, wrappers):
        super().__init__()
        self.forward_model = forward_model
        self.inverse = inverse_transformer
        self.target_layers = target_layers
        self.wrappers = {w.layer_idx: w for w in wrappers}

    def compute_errors(self, hidden_states_dict):
        """Compute prediction errors at target layers.

        Args:
            hidden_states_dict: {layer_idx: hidden_states} from forward pass

        Returns:
            errors: {layer_idx: error_tensor}
            magnitudes: {layer_idx: float}
            predictions: {layer_idx: prediction_tensor}
        """
        # Get the highest target layer's hidden states as input to inverse
        max_layer = max(self.target_layers)
        if max_layer not in hidden_states_dict:
            # Use output from the last available layer
            max_layer = max(hidden_states_dict.keys())

        inverse_input = hidden_states_dict[max_layer]
        predictions = self.inverse(inverse_input)

        errors = {}
        magnitudes = {}
        for layer_idx in self.target_layers:
            if layer_idx in predictions and layer_idx in hidden_states_dict:
                actual = hidden_states_dict[layer_idx].detach()
                pred = predictions[layer_idx]
                error = actual - pred
                errors[layer_idx] = error
                magnitudes[layer_idx] = error.pow(2).mean().item()

        return errors, magnitudes, predictions

    def forward_with_modulation(self, input_ids, errors=None, **kwargs):
        """Run forward pass with optional error modulation at target layers."""
        if errors is not None:
            for layer_idx, error in errors.items():
                if layer_idx in self.wrappers:
                    self.wrappers[layer_idx].set_error(error)
                    self.wrappers[layer_idx].enable_modulation()

        output = self.forward_model(input_ids, **kwargs)

        for w in self.wrappers.values():
            w.disable_modulation()

        return output

    def get_captured_hidden_states(self):
        """Get hidden states captured by wrappers during last forward pass."""
        return {
            layer_idx: w.last_hidden
            for layer_idx, w in self.wrappers.items()
            if w.last_hidden is not None
        }

    def awake_step(self, input_ids, n_cycles=1, **kwargs):
        """Awake mode: process input with optional thinking cycles.

        Cycle 1: observe (no modulation) → compute errors
        Cycle 2+: modulate with errors → recompute → refine

        Returns:
            output: model output
            errors: final errors at each layer
            magnitudes: error magnitudes per cycle
        """
        all_magnitudes = []

        # First pass: observe without modulation
        for w in self.wrappers.values():
            w.disable_modulation()

        output = self.forward_model(
            input_ids, output_hidden_states=False, **kwargs
        )
        hidden_dict = self.get_captured_hidden_states()

        for cycle in range(n_cycles):
            errors, magnitudes, predictions = self.compute_errors(hidden_dict)
            all_magnitudes.append(magnitudes)

            if cycle < n_cycles - 1:
                # Modulated forward pass
                output = self.forward_with_modulation(
                    input_ids, errors, **kwargs
                )
                hidden_dict = self.get_captured_hidden_states()

        # Final modulated pass
        if n_cycles > 0 and errors:
            output = self.forward_with_modulation(
                input_ids, errors, **kwargs
            )

        return output, errors, all_magnitudes

    def prediction_loss(self, hidden_states_dict):
        """Compute prediction loss for inverse transformer training.

        Self-supervised: inverse tries to predict forward hidden states.
        """
        _, _, predictions = self.compute_errors(hidden_states_dict)

        loss = torch.tensor(0.0, device=next(self.inverse.parameters()).device)
        n = 0
        for layer_idx in self.target_layers:
            if layer_idx in predictions and layer_idx in hidden_states_dict:
                actual = hidden_states_dict[layer_idx].detach()
                pred = predictions[layer_idx]
                loss = loss + F.mse_loss(pred, actual)
                n += 1

        return loss / max(n, 1)

    def consistency_loss(self, hidden_states_dict):
        """Compute consistency loss for LoRA training.

        Forward model should produce representations predictable by inverse.
        This is the REVERSE of prediction loss — gradients flow to LoRA.
        """
        max_layer = max(self.target_layers)
        if max_layer not in hidden_states_dict:
            max_layer = max(hidden_states_dict.keys())

        inverse_input = hidden_states_dict[max_layer]
        predictions = self.inverse(inverse_input.detach())

        loss = torch.tensor(0.0, device=inverse_input.device)
        n = 0
        for layer_idx in self.target_layers:
            if layer_idx in predictions and layer_idx in hidden_states_dict:
                actual = hidden_states_dict[layer_idx]  # NOT detached — gradient to LoRA
                pred = predictions[layer_idx].detach()   # detached — no gradient to inverse
                loss = loss + F.mse_loss(actual, pred)
                n += 1

        return loss / max(n, 1)
