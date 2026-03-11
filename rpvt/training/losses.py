"""Loss functions for local and global backprop training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def local_logit_lens_loss(hidden_states, labels, logit_lens, vocab_size):
    """Compute local loss using logit lens projection.

    This is the simplified L_usefulness for Experiment 1:
    L = cross_entropy(logit_lens(hidden_states), next_token_target)

    Args:
        hidden_states: (batch, seq_len, hidden_size) — residual stream at the adapted layer.
        labels: (batch, seq_len) — target token IDs (shifted by 1 from input).
        logit_lens: LogitLens module for this layer.
        vocab_size: Vocabulary size.

    Returns:
        Scalar loss.
    """
    logits = logit_lens(hidden_states)
    return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))


def native_logit_lens_loss(hidden_states, labels, model):
    """Compute local loss using the model's own final norm + lm_head.

    Instead of a separately trained projection, apply the model's actual
    output pathway (RMSNorm + lm_head) to the intermediate residual stream.
    This is the classic "logit lens" from interpretability and should be
    better aligned with downstream computation.

    Args:
        hidden_states: (batch, seq_len, hidden_size) — residual stream at adapted layer.
        labels: (batch, seq_len) — target token IDs.
        model: The full model (need access to model.model.norm and model.lm_head).

    Returns:
        Scalar loss.
    """
    normed = model.model.norm(hidden_states)
    logits = model.lm_head(normed)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )


def contribution_usefulness_loss(hidden_with, hidden_without, labels, model):
    """Full contribution usefulness loss from the architecture spec.

    Measures whether the adapter's modification to the residual stream
    actually helped the output, by comparing projections with and without
    the layer's contribution.

    L = CE(proj(r_with), target) - CE(proj(r_without), target)

    Minimizing this maximizes the helpfulness of the adapter's contribution.
    The "without" term acts as a baseline — the gradient only rewards changes
    that improve over what was already there.

    Args:
        hidden_with: (batch, seq_len, hidden_size) — residual stream WITH adapter contribution.
        hidden_without: (batch, seq_len, hidden_size) — residual stream WITHOUT adapter (frozen only).
        labels: (batch, seq_len) — target token IDs.
        model: The full model (for norm + lm_head).

    Returns:
        Scalar loss.
    """
    normed_with = model.model.norm(hidden_with)
    logits_with = model.lm_head(normed_with)
    loss_with = F.cross_entropy(
        logits_with.reshape(-1, logits_with.size(-1)),
        labels.reshape(-1),
    )

    with torch.no_grad():
        normed_without = model.model.norm(hidden_without)
        logits_without = model.lm_head(normed_without)
        loss_without = F.cross_entropy(
            logits_without.reshape(-1, logits_without.size(-1)),
            labels.reshape(-1),
        )

    return loss_with - loss_without


def global_loss(logits, labels, vocab_size):
    """Standard next-token prediction loss through the full model.

    Args:
        logits: (batch, seq_len, vocab_size) — model output logits.
        labels: (batch, seq_len) — target token IDs.

    Returns:
        Scalar loss.
    """
    return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))
