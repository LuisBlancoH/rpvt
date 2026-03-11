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


def global_loss(logits, labels, vocab_size):
    """Standard next-token prediction loss through the full model.

    Args:
        logits: (batch, seq_len, vocab_size) — model output logits.
        labels: (batch, seq_len) — target token IDs.

    Returns:
        Scalar loss.
    """
    return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))
