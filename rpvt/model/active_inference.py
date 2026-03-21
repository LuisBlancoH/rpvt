"""Active Inference for LLM agents.

Two networks that train online, driving each other toward understanding:

Forward model (Qwen + LoRA): processes input bottom-up → hidden states
Inverse model (small, trainable): predicts forward hidden states top-down

Settling loop:
  1. Forward pass → capture hidden states
  2. Inverse predicts → compute prediction error
  3. Error injected via cross-attention → forward pass again
  4. Repeat until forward/inverse agree (settling)
  5. Generate from settled representation

The primary objective is UNDERSTANDING (prediction error minimization),
not task completion. Correct answers emerge from understanding.
Actions (code execution) serve as information gathering to reduce
prediction error — not as a reward signal.

This is Karl Friston's Free Energy Principle applied to an LLM agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseModel(nn.Module):
    """Small inverse transformer that predicts forward hidden states.

    Takes hidden states from a high layer and predicts what a lower
    layer's hidden states should look like. Maintains a GRU state
    that accumulates understanding across steps.

    Args:
        hidden_size: must match forward model (e.g., 1536 for Qwen2.5-1.5B)
        n_layers: depth of inverse transformer
        n_heads: attention heads
        state_dim: GRU state dimension (compressed representation)
    """

    def __init__(self, hidden_size=1536, n_layers=2, n_heads=8,
                 state_dim=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim

        # Compress input to state_dim for GRU
        self.input_compress = nn.Linear(hidden_size, state_dim, bias=False)

        # GRU: accumulates understanding across settling steps and actions
        self.context_gru = nn.GRU(
            input_size=state_dim,
            hidden_size=state_dim,
            batch_first=True,
        )

        # Project GRU state back to hidden_size for prediction
        self.context_proj = nn.Linear(state_dim, hidden_size, bias=False)

        # Inverse transformer layers
        self.layers = nn.ModuleList([
            InverseLayer(hidden_size, n_heads)
            for _ in range(n_layers)
        ])

        # Prediction head: predicts target layer hidden states
        self.prediction_head = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.prediction_head.weight, std=0.01)

        # Persistent GRU state
        self._gru_state = None

    def reset_state(self):
        """Reset accumulated understanding (new task/episode)."""
        self._gru_state = None

    def predict(self, high_layer_hidden):
        """Predict lower layer hidden states from higher layer.

        Args:
            high_layer_hidden: [batch, seq_len, hidden_size] from a high layer

        Returns:
            predicted: [batch, seq_len, hidden_size] predicted lower layer states
            state_info: dict with GRU state norm etc. for monitoring
        """
        x = high_layer_hidden

        # Update GRU with compressed input (accumulate understanding)
        compressed = self.input_compress(x.mean(dim=1, keepdim=True))
        if self._gru_state is not None:
            gru_out, self._gru_state = self.context_gru(
                compressed, self._gru_state.detach().to(compressed.device)
            )
        else:
            gru_out, self._gru_state = self.context_gru(compressed)
        # Detach to prevent graph accumulation across steps
        self._gru_state = self._gru_state.detach()

        # Add context to all positions
        context = self.context_proj(gru_out)  # [batch, 1, hidden_size]
        x = x + context

        # Process through inverse layers
        for layer in self.layers:
            x = layer(x)

        # Predict target layer hidden states
        predicted = self.prediction_head(x)

        return predicted, {
            "gru_state_norm": self._gru_state.norm().item(),
        }


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


class ErrorMemoryBank(nn.Module):
    """Stores prediction errors for cross-attention injection.

    Instead of storing memories, stores prediction errors — the model
    attends to "where it's confused" rather than "what it's seen."
    """

    def __init__(self, hidden_size, max_errors=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_errors = max_errors

        # Error buffer
        self.register_buffer(
            'errors', torch.zeros(max_errors, hidden_size)
        )
        self.n_stored = 0
        self.write_ptr = 0

    def reset(self):
        self.errors.zero_()
        self.n_stored = 0
        self.write_ptr = 0

    def store_errors(self, error_vectors):
        """Store prediction error vectors.

        Args:
            error_vectors: [batch, seq_len, hidden_size] prediction errors
        """
        # Mean-pool across sequence to get summary errors
        error_summary = error_vectors[0]  # [seq_len, hidden_size]

        # Store top-k errors by magnitude
        error_norms = error_summary.norm(dim=-1)  # [seq_len]
        k = min(8, error_summary.shape[0], self.max_errors - self.n_stored)
        if k <= 0:
            return

        _, top_indices = error_norms.topk(k)
        top_errors = error_summary[top_indices]

        for i in range(k):
            self.errors[self.write_ptr] = top_errors[i].detach()
            self.write_ptr = (self.write_ptr + 1) % self.max_errors
            self.n_stored = min(self.n_stored + 1, self.max_errors)

    def get_errors(self):
        """Return stored errors for cross-attention.

        Returns: [n_stored, hidden_size] or None if empty
        """
        if self.n_stored == 0:
            return None
        return self.errors[:self.n_stored].unsqueeze(0)  # [1, n, hidden]


class ActiveInferenceEngine(nn.Module):
    """Combines forward model, inverse model, and settling loop.

    Usage:
        engine = ActiveInferenceEngine(forward_model, ...)
        engine.reset()  # new task

        # Settling: process input and iterate until stable
        settled_hidden, info = engine.settle(input_ids, n_steps=3)

        # Generate from settled representation
        tokens = engine.generate(input_ids)

        # After code execution, process result (online learning)
        engine.observe_and_learn(result_ids)
    """

    def __init__(self, forward_model, hidden_size=1536,
                 source_layer=27, target_layer=14,
                 inject_layer=15, n_inverse_layers=2,
                 inverse_lr=1e-3):
        super().__init__()
        self.forward_model = forward_model
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.inject_layer = inject_layer

        # Inverse model
        self.inverse = InverseModel(
            hidden_size=hidden_size,
            n_layers=n_inverse_layers,
        ).to(dtype=torch.bfloat16, device=next(forward_model.parameters()).device)

        # Error memory for cross-attention injection
        self.error_bank = ErrorMemoryBank(
            hidden_size=hidden_size,
        ).to(dtype=torch.bfloat16, device=next(forward_model.parameters()).device)

        # Online optimizer for inverse model
        self.inverse_optimizer = torch.optim.Adam(
            self.inverse.parameters(), lr=inverse_lr
        )

        # Hooks for capturing hidden states
        self._captured_hidden = {}
        self._hooks = []
        self._install_capture_hooks()

        # Stats
        self.settling_history = []

        n_params = sum(p.numel() for p in self.inverse.parameters())
        print(f"  ActiveInference: source=layer{source_layer}, "
              f"target=layer{target_layer}, inject=layer{inject_layer}, "
              f"inverse_params={n_params:,}")

    def _get_layers(self):
        model = self.forward_model
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            base = model.base_model.model
            return base.model.layers if hasattr(base, 'model') else base.layers
        if hasattr(model, 'model'):
            if hasattr(model.model, 'layers'):
                return model.model.layers
            return model.model.model.layers
        return model.layers

    def _install_capture_hooks(self):
        """Install hooks to capture hidden states at source and target layers."""
        layers = self._get_layers()

        for layer_idx in [self.source_layer, self.target_layer]:
            if layer_idx < len(layers):
                h = layers[layer_idx].register_forward_hook(
                    lambda module, input, output, idx=layer_idx:
                        self._capture_hook(idx, output)
                )
                self._hooks.append(h)

    def _capture_hook(self, layer_idx, output):
        hidden = output[0]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        self._captured_hidden[layer_idx] = hidden

    def reset(self):
        """Reset for a new task/episode."""
        self.inverse.reset_state()
        self.error_bank.reset()
        self._captured_hidden.clear()
        self.settling_history.clear()

    def _forward_pass(self, input_ids, attention_mask=None):
        """Run forward model and capture hidden states."""
        self._captured_hidden.clear()
        with torch.no_grad():
            outputs = self.forward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        return outputs

    def compute_prediction_error(self, input_ids, attention_mask=None):
        """Run forward + inverse, compute prediction error.

        Returns:
            error: [batch, seq_len, hidden_size] prediction error vectors
            error_magnitude: scalar, mean error norm
            info: dict with diagnostics
        """
        # Forward pass
        self._forward_pass(input_ids, attention_mask)

        source_hidden = self._captured_hidden.get(self.source_layer)
        target_hidden = self._captured_hidden.get(self.target_layer)

        if source_hidden is None or target_hidden is None:
            raise ValueError(
                f"Failed to capture hidden states. "
                f"Got layers: {list(self._captured_hidden.keys())}"
            )

        # Inverse prediction
        predicted_target, inv_info = self.inverse.predict(
            source_hidden.detach()
        )

        # Prediction error
        error = target_hidden.detach() - predicted_target
        error_magnitude = error.norm(dim=-1).mean().item()

        return error, error_magnitude, {
            "source_norm": source_hidden.norm(dim=-1).mean().item(),
            "target_norm": target_hidden.norm(dim=-1).mean().item(),
            "prediction_norm": predicted_target.norm(dim=-1).mean().item(),
            "error_magnitude": error_magnitude,
            **inv_info,
        }

    def settle(self, input_ids, attention_mask=None, n_steps=3):
        """Iterative settling: forward/inverse loop until agreement.

        Each step:
        1. Forward pass → hidden states
        2. Inverse predicts → error
        3. Store errors → available for cross-attention on next pass
        4. Train inverse (online, one gradient step)

        Returns:
            final_error: scalar
            history: list of error magnitudes per step
        """
        history = []

        for step in range(n_steps):
            # Compute prediction error
            error, error_mag, info = self.compute_prediction_error(
                input_ids, attention_mask
            )
            history.append(error_mag)

            # Store errors for cross-attention injection
            self.error_bank.store_errors(error.detach())

            # Online training: update inverse to reduce prediction error
            self._train_inverse_step(input_ids, attention_mask)

        self.settling_history.append(history)
        return history[-1], history

    def _train_inverse_step(self, input_ids, attention_mask=None):
        """One gradient step on the inverse model."""
        # Forward pass (no grad for forward model)
        self._forward_pass(input_ids, attention_mask)

        source_hidden = self._captured_hidden[self.source_layer].detach()
        target_hidden = self._captured_hidden[self.target_layer].detach()

        # Inverse prediction (WITH grad)
        predicted, _ = self.inverse.predict(source_hidden)

        # Loss: predict target hidden states
        loss = F.mse_loss(predicted, target_hidden)

        self.inverse_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.inverse.parameters(), 1.0)
        self.inverse_optimizer.step()

        return loss.item()

    def observe_and_learn(self, new_input_ids, attention_mask=None):
        """Process new information (e.g., code execution result).

        Updates the inverse model's understanding of the world.
        The GRU state accumulates this new information.
        """
        error, error_mag, info = self.compute_prediction_error(
            new_input_ids, attention_mask
        )
        # Store errors
        self.error_bank.store_errors(error.detach())

        # Train inverse on this new observation
        loss = self._train_inverse_step(new_input_ids, attention_mask)

        return {
            "error_magnitude": error_mag,
            "inverse_loss": loss,
            **info,
        }

    def get_uncertainty(self, input_ids, attention_mask=None):
        """Get scalar uncertainty estimate for current input.

        High value = model doesn't understand this → should explore
        Low value = model understands this → can act confidently
        """
        _, error_mag, _ = self.compute_prediction_error(
            input_ids, attention_mask
        )
        return error_mag

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
