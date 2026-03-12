"""Experiment v2.3: Surprise-driven write gating.

Tests whether error-driven writes improve M's quality compared to
uniform writes and learned gating.

Hypothesis: Writing strongly when M's prediction is wrong (surprise)
and weakly when M predicts correctly (expected) should produce
better memory content than writing uniformly or with a learned gate.

Write modes compared:
    "uniform":  All tokens write equally (baseline from v2.2)
    "gate":     sigmoid(W_gate @ hidden_state) — learned, but stuck at init
    "surprise": sigmoid(scale * ||actual - prediction|| + bias)
                prediction = W_out(M @ query)
                Writes driven by prediction error, not learned from scratch.

Success criteria:
    1. At decay 0.999: surprise benefit grows more with position than uniform
    2. At decay 0.9999: surprise keeps M stable (like gating) while uniform explodes
    3. surprise_scale and surprise_bias actually move during training
       (unlike the gate which was stuck at sigmoid(-2) = 0.1191)
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
    evaluate_memory_reset_ablation,
    reset_memories,
)


def main():
    parser = argparse.ArgumentParser(description="v2.3: surprise-driven write gating")
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--decay", type=float, default=0.999,
                        help="Decay rate (single value, run all write modes at this decay)")
    parser.add_argument("--decays", type=float, nargs="+", default=None,
                        help="Multiple decay rates to sweep (overrides --decay)")
    parser.add_argument("--write-modes", type=str, nargs="+",
                        default=["uniform", "surprise"],
                        choices=["uniform", "gate", "surprise", "surprise-fwd", "surprise-fwd-store"],
                        help="Write modes to compare")
    parser.add_argument("--max-m-norm", type=float, default=10.0,
                        help="Cap on M's Frobenius norm (0 = no cap)")
    parser.add_argument("--chunk-aggs", type=str, nargs="+",
                        default=["token"],
                        choices=["token", "mean", "last", "surprise", "learned"],
                        help="Chunk aggregation methods to compare")
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-docs", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v2_3_surprise")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    dtype = torch.bfloat16

    decay_rates = args.decays if args.decays else [args.decay]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load data once
    print("Loading sequential document data...")
    seq_dataset = SequentialDocDataset(
        tokenizer, seq_len=args.seq_len, max_docs=args.max_docs,
    )

    # Get no-memory baseline
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

    all_results = {"baseline": baseline_loss, "runs": {}}

    for decay in decay_rates:
        for write_mode in args.write_modes:
            for chunk_agg in args.chunk_aggs:
                # Skip invalid combo: surprise agg only works with surprise write modes
                if chunk_agg == "surprise" and write_mode not in ("surprise", "surprise-fwd", "surprise-fwd-store"):
                    print(f"\n  Skipping chunk_agg=surprise with write_mode={write_mode} (no surprise scores)")
                    continue

                run_key = f"decay={decay}_mode={write_mode}_agg={chunk_agg}"
                print(f"\n{'='*60}")
                print(f"DECAY = {decay}, WRITE MODE = {write_mode}, CHUNK AGG = {chunk_agg}")
                print(f"{'='*60}")

                # Build fresh model
                model = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device, dtype=dtype)
                for p in model.parameters():
                    p.requires_grad = False

                memory_modules = attach_fast_weight_memory(
                    model.transformer.h,
                    hidden_size=model.config.n_embd,
                    memory_size=args.memory_size,
                    decay=decay,
                    write_mode=write_mode,
                    max_m_norm=args.max_m_norm,
                    chunk_agg=chunk_agg,
                )

                n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Trainable params: {n_trainable:,}")

                # Show surprise hyperparams
                if write_mode in ("surprise", "surprise-fwd", "surprise-fwd-store"):
                    mem = memory_modules[0]
                    print(f"  surprise_scale: {mem.surprise_scale:.1f}")
                    print(f"  surprise_bias:  {mem.surprise_bias:.1f}")

                # Train
                train_losses, log_data = train_sequential(
                    model, seq_dataset, device,
                    num_epochs=args.epochs, lr=args.lr,
                    warmup_steps=200, log_every=args.log_every,
                    model_name=run_key,
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
                late_positions = [p for p in range(10, 21) if p in benefits]
                late_benefit = sum(benefits[p] for p in late_positions) / max(len(late_positions), 1)
                growth = late_benefit - early_benefit

                print(f"\n  Early benefit (chunks 1-3): {early_benefit:+.4f}")
                print(f"  Late benefit (chunks 10+):  {late_benefit:+.4f}")
                print(f"  Growth (late - early):      {growth:+.4f}")
                if growth > 0.01:
                    print(f"  -> Memory benefit GROWS with position")
                elif growth > -0.01:
                    print(f"  -> Memory benefit is FLAT")
                else:
                    print(f"  -> Memory benefit DECREASES with position")

                # ── Memory reset ablation: is M real memory or just adapter? ──
                print(f"\n  Memory reset ablation (adapter vs real memory)...")
                reset_ablation = evaluate_memory_reset_ablation(
                    model, seq_dataset, device, reset_at_positions=(3, 5, 8, 12),
                )
                print(f"\n  {'Reset@':>8} | {'Normal':>8} | {'Reset':>8} | {'No M':>8} | {'Reset Cost':>10}")
                print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
                for pos, r in sorted(reset_ablation.items()):
                    print(f"  {f'K={pos}':>8} | {r['normal']:.4f} | {r['reset']:.4f} | "
                          f"{r['no_memory']:.4f} | {r['reset_cost']:+.4f}")
                print(f"\n  If reset_cost grows with K -> M stores real document info")
                print(f"  If reset_cost is flat/zero -> M is just an adapter")

                all_results["runs"][run_key] = {
                    "decay": decay,
                    "write_mode": write_mode,
                    "chunk_agg": chunk_agg,
                    "pos_with_M": {str(k): v for k, v in pos_with.items()},
                    "pos_without_M": {str(k): v for k, v in pos_without.items()},
                    "benefits": {str(k): v for k, v in benefits.items()},
                    "early_benefit": early_benefit,
                    "late_benefit": late_benefit,
                    "growth": growth,
                    "reset_ablation": {str(k): v for k, v in reset_ablation.items()},
                    "training_log": log_data,
                }

                # Save checkpoint
                torch.save(model.state_dict(),
                           Path(args.output_dir) / f"model_{run_key}.pt")

                del model
                torch.cuda.empty_cache()

    # ── Final comparison ──
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  Baseline (no memory): {baseline_loss:.4f}")
    print(f"\n  {'Run':>30} | {'Early':>8} | {'Late':>8} | {'Growth':>8}")
    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for run_key, r in all_results["runs"].items():
        print(f"  {run_key:>30} | {r['early_benefit']:+.4f} | {r['late_benefit']:+.4f} | {r['growth']:+.4f}")

    # Highlight winner per decay rate
    for decay in decay_rates:
        runs_at_decay = {k: v for k, v in all_results["runs"].items() if v["decay"] == decay}
        if len(runs_at_decay) > 1:
            best_key = max(runs_at_decay, key=lambda k: runs_at_decay[k]["late_benefit"])
            print(f"\n  Best at decay={decay}: {best_key} "
                  f"(late benefit: {runs_at_decay[best_key]['late_benefit']:+.4f})")

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
