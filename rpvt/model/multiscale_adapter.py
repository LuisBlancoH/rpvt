"""Multi-timescale LoRA adapter.

Three timescales of adaptation:
  Fast:   high lr, rapid decay   — handles current context, forgets quickly
  Medium: moderate lr/decay      — accumulates over sessions
  Slow:   low lr, minimal decay  — long-term knowledge, resists change

The combined adapter output is the sum of all three timescales:
  delta_W = A_fast @ B_fast + A_med @ B_med + A_slow @ B_slow
"""

import math
import torch
import torch.nn as nn


class MultiScaleLoRA(nn.Module):
    """Multi-timescale LoRA adapter with fast, medium, and slow components."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank

        # Fast timescale: learns quickly, decays quickly
        self.A_fast = nn.Parameter(torch.empty(in_features, rank))
        self.B_fast = nn.Parameter(torch.zeros(rank, out_features))

        # Medium timescale
        self.A_med = nn.Parameter(torch.empty(in_features, rank))
        self.B_med = nn.Parameter(torch.zeros(rank, out_features))

        # Slow timescale: learns slowly, resists forgetting
        self.A_slow = nn.Parameter(torch.empty(in_features, rank))
        self.B_slow = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A matrices
        for A in [self.A_fast, self.A_med, self.A_slow]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute combined adapter output from all timescales."""
        fast = x @ self.A_fast @ self.B_fast
        med = x @ self.A_med @ self.B_med
        slow = x @ self.A_slow @ self.B_slow
        return (fast + med + slow) * self.scale

    def get_param_groups(self, base_lr: float, fast_lr_mult: float = 10.0, slow_lr_mult: float = 0.1):
        """Return parameter groups with different learning rates per timescale."""
        return [
            {"params": [self.A_fast, self.B_fast], "lr": base_lr * fast_lr_mult, "timescale": "fast"},
            {"params": [self.A_med, self.B_med], "lr": base_lr, "timescale": "medium"},
            {"params": [self.A_slow, self.B_slow], "lr": base_lr * slow_lr_mult, "timescale": "slow"},
        ]

    def decay_fast(self, rate: float = 0.99):
        """Decay fast adapter weights toward zero."""
        with torch.no_grad():
            self.A_fast.mul_(rate)
            self.B_fast.mul_(rate)

    def decay_medium(self, rate: float = 0.9999):
        """Decay medium adapter weights toward zero."""
        with torch.no_grad():
            self.A_med.mul_(rate)
            self.B_med.mul_(rate)

    def get_timescale_norms(self):
        """Return the Frobenius norm of each timescale's contribution."""
        with torch.no_grad():
            fast_norm = (self.A_fast @ self.B_fast).norm().item()
            med_norm = (self.A_med @ self.B_med).norm().item()
            slow_norm = (self.A_slow @ self.B_slow).norm().item()
        return {"fast": fast_norm, "medium": med_norm, "slow": slow_norm}


class AdaptedLayerMultiScale(nn.Module):
    """Wraps a frozen linear layer with a multi-timescale LoRA adapter."""

    def __init__(self, frozen_linear: nn.Linear, rank: int = 32, alpha: float = 1.0):
        super().__init__()
        self.frozen = frozen_linear
        self.adapter = MultiScaleLoRA(
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
        return self.frozen.weight

    @property
    def bias(self):
        return self.frozen.bias


def attach_multiscale_adapters(model, layer_indices=None, targets=None, rank=32):
    """Attach multi-timescale adapters to specified layers and weight matrices.

    Args:
        model: The Qwen model.
        layer_indices: List of layer indices. None = all layers.
        targets: List of target weight matrices. None = ["mlp_out"].
        rank: LoRA rank per timescale.

    Returns:
        List of AdaptedLayerMultiScale modules.
    """
    from rpvt.model.base import get_layers, get_num_layers

    layers = get_layers(model)
    num_layers = get_num_layers(model)

    if layer_indices is None:
        layer_indices = list(range(num_layers))
    if targets is None:
        targets = ["mlp_out"]

    target_map = {
        "mlp_down": "mlp.gate_proj",
        "mlp_up": "mlp.up_proj",
        "mlp_out": "mlp.down_proj",
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
    }

    adapted_modules = []

    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for target in targets:
            attr_path = target_map[target].split(".")
            parent = layer
            for attr in attr_path[:-1]:
                parent = getattr(parent, attr)

            frozen_linear = getattr(parent, attr_path[-1])
            adapted = AdaptedLayerMultiScale(frozen_linear, rank=rank)
            adapted = adapted.to(
                device=frozen_linear.weight.device,
                dtype=frozen_linear.weight.dtype,
            )
            setattr(parent, attr_path[-1], adapted)
            adapted_modules.append(adapted)

    return adapted_modules


def remove_multiscale_adapters(model, layer_indices=None, targets=None):
    """Remove all multi-timescale adapters, restoring frozen weights."""
    from rpvt.model.base import get_layers, get_num_layers

    layers = get_layers(model)
    num_layers = get_num_layers(model)

    if layer_indices is None:
        layer_indices = list(range(num_layers))
    if targets is None:
        targets = ["mlp_out"]

    target_map = {
        "mlp_down": "mlp.gate_proj",
        "mlp_up": "mlp.up_proj",
        "mlp_out": "mlp.down_proj",
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
    }

    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for target in targets:
            attr_path = target_map[target].split(".")
            parent = layer
            for attr in attr_path[:-1]:
                parent = getattr(parent, attr)
            adapted = getattr(parent, attr_path[-1])
            if isinstance(adapted, AdaptedLayerMultiScale):
                setattr(parent, attr_path[-1], adapted.frozen)
