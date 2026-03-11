"""Load and wrap the frozen Qwen base model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model(
    model_name: str = "Qwen/Qwen2.5-3B",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load frozen Qwen model and tokenizer.

    Returns the model with all parameters frozen, plus the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model, tokenizer


def get_layers(model):
    """Get the transformer layer list from a Qwen model."""
    return model.model.layers


def get_embed_tokens(model):
    """Get the token embedding layer."""
    return model.model.embed_tokens


def get_lm_head(model):
    """Get the final language model head (unembedding)."""
    return model.lm_head


def get_hidden_size(model):
    """Get the hidden dimension of the model."""
    return model.config.hidden_size


def get_vocab_size(model):
    """Get vocabulary size."""
    return model.config.vocab_size


def get_num_layers(model):
    """Get number of transformer layers."""
    return model.config.num_hidden_layers
