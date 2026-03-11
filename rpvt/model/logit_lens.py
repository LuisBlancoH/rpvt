"""Logit lens: linear projections from intermediate residual streams to vocab logits."""

import torch
import torch.nn as nn


class LogitLens(nn.Module):
    """A learned linear projection from hidden states at a given layer to vocabulary logits.

    This is trained in Stage 1 to give each layer a direct loss signal
    by projecting its residual stream to next-token predictions.
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logits.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return self.proj(hidden_states)


def train_logit_lens(
    model,
    tokenizer,
    layer_idx: int,
    dataset,
    hidden_size: int,
    vocab_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    lr: float = 1e-3,
    num_steps: int = 1000,
    batch_size: int = 4,
    seq_len: int = 512,
    log_every: int = 50,
):
    """Train a logit lens projection for a specific layer.

    Runs the frozen model, captures the residual stream at `layer_idx`,
    and trains a linear projection to predict next-token logits.
    """
    from rpvt.model.base import get_layers, get_lm_head
    from rpvt.model.hooks import capture_residual_stream
    from torch.utils.data import DataLoader

    lens = LogitLens(hidden_size, vocab_size).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(lens.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    step = 0

    for batch in dataloader:
        if step >= num_steps:
            break

        input_ids = batch["input_ids"][:, :seq_len].to(device)
        labels = input_ids[:, 1:]  # next token targets

        # Capture residual stream at target layer
        with torch.no_grad():
            residuals = capture_residual_stream(model, input_ids, [layer_idx])
            hidden = residuals[layer_idx][:, :-1]  # align with labels

        # Train logit lens
        logits = lens(hidden.to(dtype))
        loss = loss_fn(logits.reshape(-1, vocab_size), labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % log_every == 0:
            print(f"  [logit lens] step {step}/{num_steps}, loss={loss.item():.4f}")

    return lens
