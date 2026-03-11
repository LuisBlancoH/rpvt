"""Experiment v2.2b: Sweep decay rates on sequential data.

Tests whether slower decay increases M's benefit at later document positions.

Decay 0.99:   effective memory ~8 chunks  (from v2.2: flat benefit ~0.13)
Decay 0.999:  effective memory ~80 chunks (should show growing benefit)
Decay 0.9999: effective memory ~800 chunks (very long range)
Decay 1.0:    no decay (M grows forever — may degrade from noise)
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer

from rpvt.model.fast_weight import (
    attach_fast_weight_memory,
    remove_fast_weight_memory,
    TransformerLayerWithMemory,
)
from rpvt.experiments.exp_v2_2_sequential import (
    SequentialDocDataset,
    sequential_collate_fn,
    train_sequential,
    evaluate_by_position,
    evaluate_by_position_no_memory,
    reset_memories,
)


def main():
    parser = argparse.ArgumentParser(description="v2.2b: decay rate sweep")
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--decays", type=float, nargs="+", default=[0.99, 0.999, 0.9999, 1.0])
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v2_2_sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load data once
    print("Loading sequential document data...")
    seq_dataset = SequentialDocDataset(
        tokenizer, seq_len=args.seq_len, max_docs=args.max_docs,
    )

    # First: get no-memory baseline (pretrained GPT-2 without memory)
    print(f"\n{'='*60}")
    print("BASELINE: Pretrained GPT-2 (no memory)")
    print(f"{'='*60}")
    model_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device, dtype=dtype)
    model_base.eval()
    loader = DataLoader(seq_dataset, batch_size=1, shuffle=False,
                        collate_fn=sequential_collate_fn)
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            outputs = model_base(input_ids, labels=input_ids)
            total += outputs.loss.item()
            count += 1
    baseline_loss = total / count
    print(f"  Baseline loss: {baseline_loss:.4f}")
    del model_base
    torch.cuda.empty_cache()

    all_results = {"baseline": baseline_loss, "decays": {}}

    for decay in args.decays:
        print(f"\n{'='*60}")
        print(f"DECAY = {decay}")
        print(f"{'='*60}")

        # Build fresh model with this decay
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device, dtype=dtype)
        for p in model.parameters():
            p.requires_grad = False

        memory_modules = attach_fast_weight_memory(
            model.transformer.h,
            hidden_size=model.config.n_embd,
            memory_size=args.memory_size,
            decay=decay,
        )

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_trainable:,}")

        # Train
        train_losses, log_data = train_sequential(
            model, seq_dataset, device,
            num_epochs=args.epochs, lr=args.lr,
            warmup_steps=200, log_every=args.log_every,
            model_name=f"decay={decay}",
        )

        # Evaluate by position
        print(f"\n  Evaluating by position...")
        pos_with = evaluate_by_position(model, seq_dataset, device)
        pos_without = evaluate_by_position_no_memory(model, seq_dataset, device)

        # Print comparison
        print(f"\n  {'Pos':>5} | {'With M':>8} | {'No M':>8} | {'Benefit':>8}")
        print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        benefits = {}
        for pos in sorted(set(pos_with) & set(pos_without)):
            b = pos_without[pos] - pos_with[pos]
            benefits[pos] = b
            label = f"{pos}" if pos < 20 else "20+"
            print(f"  {label:>5} | {pos_with[pos]:.4f} | {pos_without[pos]:.4f} | {b:+.4f}")

        # Summary stats
        early_benefit = sum(benefits.get(p, 0) for p in range(1, 4)) / 3
        late_benefit = sum(benefits.get(p, 0) for p in range(10, 20) if p in benefits)
        late_count = sum(1 for p in range(10, 20) if p in benefits)
        late_benefit = late_benefit / max(late_count, 1)
        growth = late_benefit - early_benefit

        print(f"\n  Early benefit (chunks 1-3): {early_benefit:+.4f}")
        print(f"  Late benefit (chunks 10-19): {late_benefit:+.4f}")
        print(f"  Growth (late - early): {growth:+.4f}")
        if growth > 0.01:
            print(f"  Memory benefit GROWS with position.")
        elif growth > -0.01:
            print(f"  Memory benefit is FLAT.")
        else:
            print(f"  Memory benefit DECREASES with position.")

        all_results["decays"][str(decay)] = {
            "pos_with_M": {str(k): v for k, v in pos_with.items()},
            "pos_without_M": {str(k): v for k, v in pos_without.items()},
            "benefits": {str(k): v for k, v in benefits.items()},
            "early_benefit": early_benefit,
            "late_benefit": late_benefit,
            "growth": growth,
        }

        # Save checkpoint
        torch.save(model.state_dict(), Path(args.output_dir) / f"model_decay_{decay}.pt")

        del model
        torch.cuda.empty_cache()

    # ── Final comparison ──
    print(f"\n{'='*60}")
    print("FINAL COMPARISON ACROSS DECAY RATES")
    print(f"{'='*60}")
    print(f"  Baseline (no memory): {baseline_loss:.4f}")
    print(f"\n  {'Decay':>8} | {'Early':>8} | {'Late':>8} | {'Growth':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for decay in args.decays:
        r = all_results["decays"][str(decay)]
        print(f"  {decay:>8} | {r['early_benefit']:+.4f} | {r['late_benefit']:+.4f} | {r['growth']:+.4f}")

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
