"""LoRA adapter for plastic weight modification."""

import math
import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """Low-rank adapter that modifies a frozen linear layer's output.

    Computes: y = frozen_linear(x) + (x @ A) @ B
    Where A is (in_features, rank) and B is (rank, out_features).
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 32, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Initialize A with kaiming, B with zeros (standard LoRA init)
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the adapter's additive contribution.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Adapter output (..., out_features) to be added to frozen layer output.
        """
        return (x @ self.A @ self.B) * self.scale


class AdaptedLayer(nn.Module):
    """Wraps a frozen linear layer with a LoRA adapter.

    Replaces the frozen layer in the model's forward pass:
        y = frozen(x) + adapter(x)
    """

    def __init__(self, frozen_linear: nn.Linear, rank: int = 32, alpha: float = 1.0):
        super().__init__()
        self.frozen = frozen_linear
        self.adapter = LoRAAdapter(
            frozen_linear.in_features,
            frozen_linear.out_features,
            rank=rank,
            alpha=alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            frozen_out = self.frozen(x)
        return frozen_out + self.adapter(x)

    @property
    def weight(self):
        """Expose weight for any code that accesses .weight directly."""
        return self.frozen.weight

    @property
    def bias(self):
        return self.frozen.bias


def attach_adapter(model, layer_idx: int, target: str = "mlp_down", rank: int = 32):
    """Attach a LoRA adapter to a specific weight matrix in a specific layer.

    Args:
        model: The Qwen model.
        layer_idx: Which transformer layer.
        target: Which weight matrix. Options:
            - "mlp_down": MLP down projection (gate_proj in Qwen)
            - "mlp_up": MLP up projection (up_proj in Qwen)
            - "mlp_out": MLP output projection (down_proj in Qwen — confusing naming!)
            - "q_proj", "k_proj", "v_proj", "o_proj": Attention projections
        rank: LoRA rank.

    Returns:
        The AdaptedLayer wrapping the target.
    """
    from rpvt.model.base import get_layers

    layers = get_layers(model)
    layer = layers[layer_idx]

    target_map = {
        "mlp_down": "mlp.gate_proj",
        "mlp_up": "mlp.up_proj",
        "mlp_out": "mlp.down_proj",
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
    }

    if target not in target_map:
        raise ValueError(f"Unknown target '{target}'. Options: {list(target_map.keys())}")

    attr_path = target_map[target].split(".")
    parent = layer
    for attr in attr_path[:-1]:
        parent = getattr(parent, attr)

    frozen_linear = getattr(parent, attr_path[-1])
    adapted = AdaptedLayer(frozen_linear, rank=rank)
    adapted = adapted.to(device=frozen_linear.weight.device, dtype=frozen_linear.weight.dtype)
    setattr(parent, attr_path[-1], adapted)

    return adapted
