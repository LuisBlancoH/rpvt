"""Synthetic gradient modules.

Each layer gets a small network that predicts what gradient backprop
would send to that layer's output. The adapter uses this predicted
gradient instead of actual backprop through downstream layers.

Training:
1. First, run global backprop to collect real gradients at each layer.
2. Train gradient predictors to match the real gradients.
3. Transition: use predicted gradients instead of real backprop.
"""

import torch
import torch.nn as nn


class GradientPredictor(nn.Module):
    """Predicts the gradient of the loss w.r.t. the hidden state at a given layer.

    Takes the hidden state as input, outputs a predicted gradient of the same shape.
    Small MLP: hidden_size -> bottleneck -> hidden_size.
    """

    def __init__(self, hidden_size, bottleneck=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, bottleneck, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_size, bias=False),
        )
        # Initialize to near-zero so predictions start small
        nn.init.normal_(self.net[0].weight, std=0.01)
        nn.init.normal_(self.net[2].weight, std=0.01)

    def forward(self, hidden_states):
        """Predict gradient given hidden states.

        Args:
            hidden_states: (batch, seq, hidden_size)

        Returns:
            predicted_gradient: (batch, seq, hidden_size)
        """
        return self.net(hidden_states)
