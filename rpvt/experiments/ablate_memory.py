"""Run memory ablations on a trained model.

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


def evaluate(model, val_loader, device, max_batches=100):
    model.eval()
    total, count = 0.0, 0
    for layer in model.transformer.h:
        if isinstance(layer, TransformerLayerWithMemory):
            layer.reset_memory()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            total += outputs.loss.item()
            count += 1
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/exp_v2_1/memory_model.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", action="store_true", default=True)
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

    # ── Ablation 1: Normal eval ──
    print(f"\n{'='*60}")
    print("EVAL: Normal (memory active)")
    print(f"{'='*60}")
    loss_normal = evaluate(model, val_loader, device)
    print(f"  Loss: {loss_normal:.4f}")

    # ── Ablation 2: Zero W_out ──
    print(f"\n{'='*60}")
    print("ABLATION 1: Zero W_out (disable all memory output)")
    print(f"{'='*60}")
    saved_wout = {}
    for i, layer in enumerate(model.transformer.h):
        if isinstance(layer, TransformerLayerWithMemory):
            saved_wout[i] = layer.memory.W_out.weight.data.clone()
            layer.memory.W_out.weight.data.zero_()
    loss_no_wout = evaluate(model, val_loader, device)
    for i, w in saved_wout.items():
        model.transformer.h[i].memory.W_out.weight.data = w
    print(f"  Loss: {loss_no_wout:.4f}")
    print(f"  Memory contribution: {loss_no_wout - loss_normal:+.4f}")

    # ── Ablation 3: Reset M to zero (keep projections) ──
    print(f"\n{'='*60}")
    print("ABLATION 2: Reset M to zero (keep W_out, test if M content matters)")
    print(f"{'='*60}")
    saved_M = {}
    for i, layer in enumerate(model.transformer.h):
        if isinstance(layer, TransformerLayerWithMemory):
            saved_M[i] = layer.memory.M.clone()
            layer.memory.M.zero_()
    loss_no_M = evaluate(model, val_loader, device)
    for i, m in saved_M.items():
        model.transformer.h[i].memory.M = m
    print(f"  Loss: {loss_no_M:.4f}")
    print(f"  M content contribution: {loss_no_M - loss_normal:+.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Normal (memory active):  {loss_normal:.4f}")
    print(f"  W_out zeroed:            {loss_no_wout:.4f}  (delta: {loss_no_wout - loss_normal:+.4f})")
    print(f"  M content zeroed:        {loss_no_M:.4f}  (delta: {loss_no_M - loss_normal:+.4f})")
    print()
    if loss_no_M - loss_normal > 0.01:
        print("  RESULT: Model uses M's stored content. REAL MEMORY.")
    elif loss_no_wout - loss_normal > 0.01:
        print("  RESULT: Model uses memory projections but not M content. FAKE MEMORY.")
    else:
        print("  RESULT: Model doesn't use memory at all.")


if __name__ == "__main__":
    main()
