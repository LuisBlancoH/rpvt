"""Recurrent depth: loop upper layers of a transformer for deeper reasoning.

Instead of one pass through layers 0-27, run layers 14-27 multiple times.
A learned projection maps layer 27 output back to layer 14 input space.

This gives the model N× the reasoning depth from the same parameters.
LoRA on the looped layers learns to leverage the extra passes.

Approach: Use a pre-hook on the split layer and a post-hook on the last
layer. After the first full pass, re-run upper layers by directly calling
them (without hooks triggering recursively).
"""

import torch
import torch.nn as nn


class RecurrentDepthWrapper(nn.Module):
    """Wraps a transformer model to add recurrent depth via hooks."""

    def __init__(self, model, split_layer=14, n_loops=2, residual_scale=0.1):
        super().__init__()
        self.model = model
        self.split_layer = split_layer
        self.n_loops = n_loops
        self.residual_scale = residual_scale

        config = self._get_config()
        hidden_size = config.hidden_size
        device = next(model.parameters()).device

        # Learned projection: layer N output → layer split_layer input
        self.project_back = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.project_back.weight)
        self.project_back = self.project_back.to(dtype=torch.bfloat16, device=device)

        # Loop embedding so LoRA can distinguish passes
        self.loop_embed = nn.Embedding(n_loops, hidden_size)
        nn.init.zeros_(self.loop_embed.weight)
        self.loop_embed = self.loop_embed.to(dtype=torch.bfloat16, device=device)

        # For capturing state during forward pass
        self._lower_output = None
        self._position_embeddings = None
        self._active_loops = n_loops
        self._hooks = []

        self._install_hooks()

        n_params = (hidden_size * hidden_size) + (n_loops * hidden_size)
        print(f"  RecurrentDepth: split@layer {split_layer}, "
              f"n_loops={n_loops}, params={n_params:,}")

    def _get_config(self):
        if hasattr(self.model, 'config'):
            return self.model.config
        return self.model.base_model.model.config

    def _get_layers(self):
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
            base = self.model.base_model.model
            return base.model.layers if hasattr(base, 'model') else base.layers
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                return self.model.model.layers
            return self.model.model.model.layers
        return self.model.layers

    def _get_inner_model(self):
        """Get the inner transformer model (the one with .layers and .norm)."""
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
            base = self.model.base_model.model
            return base.model if hasattr(base, 'model') else base
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                return self.model.model
            return self.model.model.model
        return self.model

    def _get_rotary_emb(self):
        inner = self._get_inner_model()
        return inner.rotary_emb

    def _install_hooks(self):
        layers = self._get_layers()

        # Save the output of the layer before split (lower layers output)
        if self.split_layer > 0:
            h = layers[self.split_layer - 1].register_forward_hook(
                self._capture_lower_output
            )
            self._hooks.append(h)

    def _capture_lower_output(self, module, input, output):
        """Save hidden states from the layer before the split."""
        self._lower_output = output[0]

    def forward(self, input_ids, attention_mask=None, labels=None,
                n_loops=None, **kwargs):
        """Forward pass with recurrent depth."""
        active_loops = n_loops if n_loops is not None else self.n_loops
        self._lower_output = None

        # Forward pass with hidden state capture
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=(active_loops > 1),
            use_cache=False,
            **kwargs,
        )

        if active_loops <= 1:
            if labels is not None:
                outputs.loss = self._compute_loss(outputs.logits, labels)
            return outputs

        # Get hidden states for recurrence
        hidden_states = outputs.hidden_states[-1]
        lower_output = self._lower_output
        if lower_output is None:
            lower_output = outputs.hidden_states[self.split_layer]

        layers = self._get_layers()
        inner_model = self._get_inner_model()

        # Compute position embeddings for re-running upper layers
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        rotary_emb = self._get_rotary_emb()
        position_embeddings = rotary_emb(hidden_states, position_ids)

        for loop in range(1, active_loops):
            # Project back
            projected = self.project_back(hidden_states)
            hidden_states = lower_output + self.residual_scale * (
                projected - lower_output
            )

            # Add loop embedding
            loop_emb = self.loop_embed(
                torch.tensor(loop, device=hidden_states.device)
            )
            hidden_states = hidden_states + loop_emb

            # Re-run upper layers directly (no hooks — avoid recursion)
            for i in range(self.split_layer, len(layers)):
                layer_out = layers[i](
                    hidden_states,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_out[0]
                # PEFT wrapping can drop batch dim — restore it
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)

        # Final norm + lm_head
        hidden_states = inner_model.norm(hidden_states)
        if hasattr(self.model, 'base_model'):
            logits = self.model.base_model.model.lm_head(hidden_states)
        elif hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(hidden_states)
        else:
            logits = self.model.model.lm_head(hidden_states)

        # Replace logits in outputs
        outputs.logits = logits

        if labels is not None:
            outputs.loss = self._compute_loss(logits, labels)

        return outputs

    def _compute_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def generate_with_loops(self, input_ids, attention_mask=None,
                            max_new_tokens=100, n_loops=None,
                            temperature=0.0):
        """Autoregressive generation with recurrent depth."""
        generated = []
        current_ids = input_ids

        for step in range(max_new_tokens):
            with torch.no_grad():
                out = self.forward(
                    current_ids,
                    attention_mask=attention_mask,
                    n_loops=n_loops,
                )

            logits = out.logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            token_id = next_token.item()
            if token_id == self._get_config().eos_token_id:
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

    def get_trainable_params(self):
        return list(self.project_back.parameters()) + \
               list(self.loop_embed.parameters())

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
