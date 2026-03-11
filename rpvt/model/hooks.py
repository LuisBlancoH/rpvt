"""Forward hooks for capturing intermediate residual streams."""

import torch
from rpvt.model.base import get_layers


def capture_residual_stream(model, input_ids, layer_indices):
    """Run a forward pass and capture residual stream states at specified layers.

    Args:
        model: The Qwen model.
        input_ids: (batch, seq_len) input token IDs.
        layer_indices: List of layer indices to capture.

    Returns:
        Dict mapping layer_idx -> residual stream tensor (batch, seq_len, hidden_size).
    """
    layers = get_layers(model)
    captured = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            # Qwen layers return (hidden_states, ...) tuple
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden.detach()
        return hook_fn

    for idx in layer_indices:
        h = layers[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return captured


class ResidualStreamCapture:
    """Context manager for capturing residual streams during forward passes that need gradients.

    Unlike capture_residual_stream, this does NOT detach the tensors,
    so gradients can flow through them.
    """

    def __init__(self, model, layer_indices):
        self.model = model
        self.layer_indices = layer_indices
        self.captured = {}
        self._hooks = []

    def __enter__(self):
        layers = get_layers(self.model)
        for idx in self.layer_indices:
            h = layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.captured[idx] = hidden
        return hook_fn
