"""Active Inference v2: Settling via cross-attention to inverse predictions.

The forward model sees BOTH its own hidden states (bottom-up) AND the
inverse model's predictions (top-down) via cross-attention at layer 15.

The attention mechanism naturally computes the mismatch — where the
bottom-up and top-down views agree or disagree. The model decides
what to do with the information.

Settling loop:
  1. Forward pass → capture hidden states at layers 14 and 27
  2. Inverse predicts layer 14 hidden states from layer 27
  3. Store predictions in memory bank (as KV pairs for layer 15)
  4. Forward pass again — layer 15 now attends to predictions
  5. New hidden states → inverse predicts again → updated predictions
  6. Repeat — the two views converge

Same cross-attention mechanism as our working memory system.
Different content: inverse predictions instead of stored memories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rpvt.model.active_inference import InverseModel
from rpvt.model.cross_attention_memory import (
    MemoryAugmentedAttention,
    MemoryBank,
)


class PredictionBank(nn.Module):
    """Stores inverse predictions for cross-attention.

    Simpler than MemoryBank — no gating, no extraction queries.
    Just stores the inverse model's predictions as raw hidden states
    that layer 15 can attend to.
    """

    def __init__(self, hidden_size, n_slots=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots = n_slots

        self.register_buffer("mem_states", torch.zeros(n_slots, hidden_size))
        self.register_buffer("mem_strength", torch.zeros(n_slots))
        self.n_stored = 0

    def reset(self):
        self.mem_states.zero_()
        self.mem_strength.zero_()
        self.n_stored = 0

    def store_predictions(self, predictions):
        """Store inverse predictions.

        Args:
            predictions: [batch, seq_len, hidden_size]
        """
        pred = predictions[0].detach()  # [seq_len, hidden_size]
        seq_len = pred.shape[0]

        # Store the mean-pooled prediction as one slot
        # (keeps it simple — one summary vector per settling step)
        if self.n_stored < self.n_slots:
            self.mem_states[self.n_stored] = pred.mean(dim=0)
            self.mem_strength[self.n_stored] = 1.0
            self.n_stored += 1

    def store_per_position(self, predictions, top_k=8):
        """Store top-k most confident predictions (by norm).

        Args:
            predictions: [batch, seq_len, hidden_size]
            top_k: number of positions to store
        """
        pred = predictions[0].detach()  # [seq_len, hidden_size]
        norms = pred.norm(dim=-1)  # [seq_len]
        k = min(top_k, pred.shape[0], self.n_slots - self.n_stored)
        if k <= 0:
            return

        _, top_idx = norms.topk(k)
        for i in range(k):
            if self.n_stored < self.n_slots:
                self.mem_states[self.n_stored] = pred[top_idx[i]]
                self.mem_strength[self.n_stored] = 1.0
                self.n_stored += 1

    def get_active_memories(self):
        """Return stored predictions for cross-attention."""
        if self.n_stored == 0:
            return None, 0
        return self.mem_states[:self.n_stored], self.n_stored


class ActiveInferenceSettler(nn.Module):
    """Active inference with settling via cross-attention.

    Uses the existing MemoryAugmentedAttention mechanism but feeds it
    inverse predictions instead of stored memories.
    """

    def __init__(self, model, hidden_size=1536,
                 source_layer=27, target_layer=14, read_layer=15,
                 n_inverse_layers=2, n_prediction_slots=32,
                 inverse_lr=1e-3):
        super().__init__()
        self.model = model
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.read_layer = read_layer

        device = next(model.parameters()).device

        # Inverse model: predicts layer 14 from layer 27
        self.inverse = InverseModel(
            hidden_size=hidden_size,
            n_layers=n_inverse_layers,
        ).to(dtype=torch.bfloat16, device=device)

        # Prediction bank: stores inverse predictions for cross-attention
        self.prediction_bank = PredictionBank(
            hidden_size=hidden_size,
            n_slots=n_prediction_slots,
        ).to(dtype=torch.bfloat16, device=device)

        # Replace layer 15's self_attn with memory-augmented version
        layers = self._get_layers()
        read_layer_module = layers[read_layer]
        if hasattr(read_layer_module, 'self_attn'):
            original_attn = read_layer_module.self_attn
        elif hasattr(read_layer_module, 'layer'):
            original_attn = read_layer_module.layer.self_attn
            read_layer_module = read_layer_module.layer
        else:
            raise ValueError(f"Can't find self_attn in layer {read_layer}")

        self.aug_attn = MemoryAugmentedAttention(
            original_attn, self.prediction_bank
        )
        read_layer_module.self_attn = self.aug_attn

        # Online optimizer for inverse
        self.inverse_optimizer = torch.optim.Adam(
            self.inverse.parameters(), lr=inverse_lr
        )

        # Capture hooks
        self._captured = {}
        self._hooks = []
        self._install_hooks()

        n_inv_params = sum(p.numel() for p in self.inverse.parameters())
        print(f"  ActiveInferenceSettler: source=L{source_layer}, "
              f"target=L{target_layer}, read=L{read_layer}")
        print(f"  Inverse params: {n_inv_params:,}, "
              f"prediction slots: {n_prediction_slots}")

    def _get_layers(self):
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
            base = self.model.base_model.model
            return base.model.layers if hasattr(base, 'model') else base.layers
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                return self.model.model.layers
            return self.model.model.model.layers
        return self.model.layers

    def _install_hooks(self):
        layers = self._get_layers()
        for idx in [self.source_layer, self.target_layer]:
            if idx < len(layers):
                h = layers[idx].register_forward_hook(
                    lambda mod, inp, out, i=idx: self._capture(i, out)
                )
                self._hooks.append(h)

    def _capture(self, layer_idx, output):
        h = output[0]
        if h.dim() == 2:
            h = h.unsqueeze(0)
        self._captured[layer_idx] = h

    def reset(self):
        """Reset for new task."""
        self.inverse.reset_state()
        self.prediction_bank.reset()
        self._captured.clear()

    def settle(self, input_ids, attention_mask=None, n_steps=3):
        """Run the settling loop.

        Each step:
        1. Forward pass (layer 15 attends to stored predictions)
        2. Capture hidden states at source and target layers
        3. Inverse predicts target from source
        4. Store predictions in bank (available for next forward pass)
        5. Train inverse online (one gradient step)

        Returns:
            error_history: list of prediction error magnitudes
            info: dict with diagnostics
        """
        error_history = []

        for step in range(n_steps):
            # Forward pass — layer 15 sees predictions from previous step
            self._captured.clear()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            source_h = self._captured.get(self.source_layer)
            target_h = self._captured.get(self.target_layer)

            if source_h is None or target_h is None:
                raise ValueError(
                    f"Failed to capture. Got: {list(self._captured.keys())}"
                )

            # Inverse predicts target layer from source layer
            predicted, inv_info = self.inverse.predict(source_h.detach())

            # Compute prediction error
            error = (target_h.detach() - predicted).norm(dim=-1).mean().item()
            error_history.append(error)

            # Store predictions for next forward pass
            self.prediction_bank.store_per_position(predicted, top_k=8)

            # Online training: one gradient step on inverse
            self._train_inverse(source_h.detach(), target_h.detach())

        return error_history, {
            "initial_error": error_history[0],
            "final_error": error_history[-1],
            "n_predictions_stored": self.prediction_bank.n_stored,
        }

    def _train_inverse(self, source_hidden, target_hidden):
        """One gradient step on the inverse model."""
        predicted, _ = self.inverse.predict(source_hidden)
        loss = F.mse_loss(predicted, target_hidden)

        self.inverse_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.inverse.parameters(), 1.0)
        self.inverse_optimizer.step()

        return loss.item()

    def get_logits(self, input_ids, attention_mask=None):
        """Get logits after settling (for generation or evaluation)."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        return outputs.logits

    def generate(self, input_ids, attention_mask=None, max_new_tokens=50,
                 n_settle=3, temperature=0.0):
        """Settle, then generate token by token.

        Settling happens once on the full input. Then generation proceeds
        normally (predictions remain in the bank as context).
        """
        # Settle on the input
        self.settle(input_ids, attention_mask, n_steps=n_settle)

        # Generate
        generated = []
        current_ids = input_ids

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            logits = out.logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            token_id = next_token.item()
            config = self.model.config if hasattr(self.model, 'config') else self.model.base_model.model.config
            if token_id == config.eos_token_id:
                break

            generated.append(token_id)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=attention_mask.device,
                               dtype=attention_mask.dtype),
                ], dim=1)

        return generated

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
