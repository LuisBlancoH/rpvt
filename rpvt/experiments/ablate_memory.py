"""Run memory ablations on a trained model.

Tests whether the model uses M's accumulated content (real memory)
or just the projections as extra compute (fake memory).

Usage:
    python -m rpvt.experiments.ablate_memory --checkpoint results/exp_v2_1/memory_model.pt --device cuda
"""

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer

from rpvt.model.fast_weight import (
    attach_fast_weight_memory,
    TransformerLayerWithMemory,
)


def load_data(tokenizer, seq_len=512, max_tokens=5_000_000):
    from datasets import load_dataset
    print("Loading wikitext-103 validation...")
    raw_val = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    all_tokens = []
    batch_texts, batch_chars = [], 0
    for text in raw_val["text"]:
        if len(text.strip()) < 50:
            continue
        batch_texts.append(text)
        batch_chars += len(text)
        if batch_chars >= 500_000:
            encoded = tokenizer(batch_texts, return_attention_mask=False, truncation=False)
            for ids in encoded["input_ids"]:
                all_tokens.extend(ids)
            batch_texts, batch_chars = [], 0
            if len(all_tokens) >= max_tokens:
                break
    if batch_texts:
        encoded = tokenizer(batch_texts, return_attention_mask=False, truncation=False)
        for ids in encoded["input_ids"]:
            all_tokens.extend(ids)
    tokens = torch.tensor(all_tokens[:max_tokens], dtype=torch.long)
    n = len(tokens) // seq_len
    return tokens[:n * seq_len].reshape(n, seq_len)


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        return {"input_ids": self.chunks[idx]}


def evaluate(model, val_loader, device, max_batches=100, reset_between_batches=False):
    """Evaluate with control over memory reset behavior."""
    model.eval()
    total, count = 0.0, 0
    # Reset once at start
    for layer in model.transformer.h:
        if isinstance(layer, TransformerLayerWithMemory):
            layer.reset_memory()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            if reset_between_batches and i > 0:
                for layer in model.transformer.h:
                    if isinstance(layer, TransformerLayerWithMemory):
                        layer.reset_memory()
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            total += outputs.loss.item()
            count += 1
    return total / max(count, 1)


def evaluate_no_within_seq_memory(model, val_loader, device, max_batches=100):
    """Evaluate with M zeroed before EACH forward pass.

    M gets written during the forward pass but is discarded after.
    This disables both within-sequence and cross-sequence memory.
    """
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            # Zero M before every forward pass
            for layer in model.transformer.h:
                if isinstance(layer, TransformerLayerWithMemory):
                    layer.reset_memory()
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            total += outputs.loss.item()
            count += 1
    return total / max(count, 1)


def evaluate_no_memory_output(model, val_loader, device, max_batches=100):
    """Evaluate with W_out zeroed (no memory contribution at all)."""
    saved = {}
    for i, layer in enumerate(model.transformer.h):
        if isinstance(layer, TransformerLayerWithMemory):
            saved[i] = layer.memory.W_out.weight.data.clone()
            layer.memory.W_out.weight.data.zero_()

    loss = evaluate(model, val_loader, device, max_batches)

    for i, w in saved.items():
        model.transformer.h[i].memory.W_out.weight.data = w
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/exp_v2_1/memory_model.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    val_chunks = load_data(tokenizer)
    val_loader = DataLoader(ChunkDataset(val_chunks), batch_size=args.batch_size,
                            shuffle=False, drop_last=True)
    print(f"  {len(val_chunks)} val chunks")

    # Build model with memory
    print("Building model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad = False
    attach_fast_weight_memory(
        model.transformer.h, hidden_size=model.config.n_embd,
        memory_size=args.memory_size, decay=args.decay,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)

    # ── Condition 1: M accumulates across batches (full memory) ──
    print(f"\n{'='*60}")
    print("CONDITION 1: M accumulates across batches")
    print(f"{'='*60}")
    loss_accumulate = evaluate(model, val_loader, device, reset_between_batches=False)
    print(f"  Loss: {loss_accumulate:.4f}")

    # ── Condition 2: M resets between batches (within-sequence only) ──
    print(f"\n{'='*60}")
    print("CONDITION 2: M resets between batches (within-sequence memory only)")
    print(f"{'='*60}")
    loss_reset_between = evaluate(model, val_loader, device, reset_between_batches=True)
    print(f"  Loss: {loss_reset_between:.4f}")

    # ── Condition 3: M resets before every forward pass (no memory) ──
    # M starts at zero, accumulates within the forward pass (chunks),
    # but is discarded after. Tests within-sequence chunk-to-chunk memory.
    print(f"\n{'='*60}")
    print("CONDITION 3: M resets before each forward pass (no carryover)")
    print(f"{'='*60}")
    loss_no_carryover = evaluate_no_within_seq_memory(model, val_loader, device)
    print(f"  Loss: {loss_no_carryover:.4f}")

    # ── Condition 4: W_out zeroed (no memory pathway at all) ──
    print(f"\n{'='*60}")
    print("CONDITION 4: W_out zeroed (memory pathway disabled)")
    print(f"{'='*60}")
    loss_no_memory = evaluate_no_memory_output(model, val_loader, device)
    print(f"  Loss: {loss_no_memory:.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  1. M accumulates across batches:  {loss_accumulate:.4f}")
    print(f"  2. M resets between batches:       {loss_reset_between:.4f}  (cross-seq contribution: {loss_reset_between - loss_accumulate:+.4f})")
    print(f"  3. M resets each forward pass:     {loss_no_carryover:.4f}  (within-seq contribution: {loss_no_carryover - loss_reset_between:+.4f})")
    print(f"  4. No memory pathway:              {loss_no_memory:.4f}  (projection contribution: {loss_no_memory - loss_no_carryover:+.4f})")
    print()

    total = loss_no_memory - loss_accumulate
    cross_seq = loss_reset_between - loss_accumulate
    within_seq = loss_no_carryover - loss_reset_between
    projections = loss_no_memory - loss_no_carryover

    print(f"  BREAKDOWN of total {total:+.4f} improvement:")
    if total > 0.001:
        print(f"    Cross-sequence M:    {cross_seq:+.4f}  ({cross_seq/total:.0%})")
        print(f"    Within-sequence M:   {within_seq:+.4f}  ({within_seq/total:.0%})")
        print(f"    Projections alone:   {projections:+.4f}  ({projections/total:.0%})")
    else:
        print(f"    No meaningful improvement from memory.")


if __name__ == "__main__":
    main()
