"""Experiment 1 sweep: test local backprop across multiple layers and learning rates.

Quick diagnostic to find where local backprop works best.
"""

import argparse
import json
import os
from pathlib import Path
import math

import torch
from torch.utils.data import DataLoader

from rpvt.model.base import (
    load_base_model,
    get_hidden_size,
    get_num_layers,
    get_vocab_size,
)
from rpvt.model.adapter import LoRAAdapter, attach_adapter, AdaptedLayer
from rpvt.model.hooks import ResidualStreamCapture, capture_residual_stream
from rpvt.model.logit_lens import LogitLens, train_logit_lens
from rpvt.training.data import TokenizedDataset
from rpvt.training.losses import global_loss, local_logit_lens_loss


def evaluate(model, dataloader, vocab_size, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = input_ids[:, 1:]
            logits = model(input_ids).logits[:, :-1]
            loss = global_loss(logits, labels, vocab_size)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def reset_adapter(adapted_layer):
    """Re-initialize adapter weights to fresh state."""
    nn.init.kaiming_uniform_(adapted_layer.adapter.A, a=math.sqrt(5))
    adapted_layer.adapter.B.data.zero_()


def remove_adapter(model, layer_idx, target, original_linear):
    """Remove adapter and restore original frozen linear."""
    from rpvt.model.base import get_layers

    layers = get_layers(model)
    layer = layers[layer_idx]

    target_map = {
        "mlp_down": "mlp.gate_proj",
        "mlp_up": "mlp.up_proj",
        "mlp_out": "mlp.down_proj",
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
    }

    attr_path = target_map[target].split(".")
    parent = layer
    for attr in attr_path[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, attr_path[-1], original_linear)


import torch.nn as nn


def train_and_eval_local(
    model, layer_idx, logit_lens, train_loader, eval_loader,
    vocab_size, device, num_steps, lr, target, rank, log_every=100,
):
    """Train adapter with local backprop and return eval loss."""
    adapted = attach_adapter(model, layer_idx, target=target, rank=rank)
    original_linear = adapted.frozen  # save reference for cleanup

    optimizer = torch.optim.Adam(adapted.adapter.parameters(), lr=lr)

    # Freeze logit lens
    for p in logit_lens.parameters():
        p.requires_grad = False

    model.train()
    step = 0
    for batch in train_loader:
        if step >= num_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = input_ids[:, 1:]

        with ResidualStreamCapture(model, [layer_idx]) as cap:
            model(input_ids)
            hidden = cap.captured[layer_idx][:, :-1]

        loss = local_logit_lens_loss(hidden, labels, logit_lens, vocab_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % log_every == 0:
            print(f"    step {step}/{num_steps}, local_loss={loss.item():.4f}")

    eval_loss = evaluate(model, eval_loader, vocab_size, device)

    # Cleanup: remove adapter
    remove_adapter(model, layer_idx, target, original_linear)

    return eval_loss


def train_and_eval_global(
    model, layer_idx, train_loader, eval_loader,
    vocab_size, device, num_steps, lr, target, rank, log_every=100,
):
    """Train adapter with global backprop and return eval loss."""
    adapted = attach_adapter(model, layer_idx, target=target, rank=rank)
    original_linear = adapted.frozen

    optimizer = torch.optim.Adam(adapted.adapter.parameters(), lr=lr)

    model.train()
    step = 0
    for batch in train_loader:
        if step >= num_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = input_ids[:, 1:]

        logits = model(input_ids).logits[:, :-1]
        loss = global_loss(logits, labels, vocab_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % log_every == 0:
            print(f"    step {step}/{num_steps}, global_loss={loss.item():.4f}")

    eval_loss = evaluate(model, eval_loader, vocab_size, device)

    # Cleanup
    remove_adapter(model, layer_idx, target, original_linear)

    return eval_loss


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 15, 20, 25, 30, 33])
    parser.add_argument("--local-lrs", type=float, nargs="+", default=[1e-4, 5e-4, 1e-3, 5e-3])
    parser.add_argument("--global-lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--target", default="mlp_out")
    parser.add_argument("--lens-steps", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp1_sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    dtype = torch.bfloat16

    print(f"Loading {args.model}...")
    model, tokenizer = load_base_model(args.model, device=device, dtype=dtype)
    hidden_size = get_hidden_size(model)
    vocab_size = get_vocab_size(model)
    num_layers = get_num_layers(model)
    print(f"  {num_layers} layers, hidden_size={hidden_size}, vocab_size={vocab_size}")

    print("Loading dataset...")
    train_ds = TokenizedDataset(tokenizer, seq_len=args.seq_len, max_tokens=5_000_000)
    eval_ds = TokenizedDataset(tokenizer, split="validation", seq_len=args.seq_len, max_tokens=500_000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Baseline
    print("Evaluating baseline...")
    baseline_loss = evaluate(model, eval_loader, vocab_size, device)
    print(f"  Baseline: {baseline_loss:.4f}")

    results = {"baseline": baseline_loss, "layers": {}}

    for layer_idx in args.layers:
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*60}")

        # Train logit lens
        print(f"  Training logit lens...")
        logit_lens = train_logit_lens(
            model, tokenizer, layer_idx, train_ds,
            hidden_size, vocab_size,
            device=device, dtype=dtype,
            lr=1e-3, num_steps=args.lens_steps,
            batch_size=args.batch_size, seq_len=args.seq_len,
            log_every=200,
        )
        lens_eval = evaluate_logit_lens(model, logit_lens, layer_idx, eval_loader, vocab_size, device)
        print(f"  Logit lens eval loss: {lens_eval:.4f}")

        # Global baseline for this layer
        print(f"  Global backprop (lr={args.global_lr})...")
        global_eval = train_and_eval_global(
            model, layer_idx, train_loader, eval_loader,
            vocab_size, device,
            num_steps=args.train_steps, lr=args.global_lr,
            target=args.target, rank=args.rank,
        )
        global_improvement = baseline_loss - global_eval
        print(f"  Global eval: {global_eval:.4f} (improvement: {global_improvement:+.4f})")

        layer_results = {
            "logit_lens_loss": lens_eval,
            "global_eval": global_eval,
            "global_improvement": global_improvement,
            "local": {},
        }

        # Local at various learning rates
        for lr in args.local_lrs:
            print(f"  Local backprop (lr={lr})...")
            local_eval = train_and_eval_local(
                model, layer_idx, logit_lens, train_loader, eval_loader,
                vocab_size, device,
                num_steps=args.train_steps, lr=lr,
                target=args.target, rank=args.rank,
            )
            local_improvement = baseline_loss - local_eval
            ratio = local_improvement / global_improvement if global_improvement > 0 else 0
            print(f"    eval: {local_eval:.4f} (improvement: {local_improvement:+.4f}, ratio: {ratio:.1%})")

            layer_results["local"][str(lr)] = {
                "eval_loss": local_eval,
                "improvement": local_improvement,
                "ratio": ratio,
            }

        results["layers"][str(layer_idx)] = layer_results

        # Save after each layer
        with open(Path(args.output_dir) / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_loss:.4f}\n")
    print(f"{'Layer':>6} | {'Lens Loss':>10} | {'Global':>10} | ", end="")
    for lr in args.local_lrs:
        print(f"{'Local lr='+str(lr):>16} | ", end="")
    print()
    print("-" * (40 + 19 * len(args.local_lrs)))

    for layer_idx in args.layers:
        lr_key = str(layer_idx)
        if lr_key not in results["layers"]:
            continue
        r = results["layers"][lr_key]
        print(f"{layer_idx:>6} | {r['logit_lens_loss']:>10.4f} | {r['global_improvement']:>+10.4f} | ", end="")
        for lr in args.local_lrs:
            local_r = r["local"].get(str(lr), {})
            ratio = local_r.get("ratio", 0)
            print(f"{ratio:>15.1%} | ", end="")
        print()

    print(f"\nResults saved to {args.output_dir}/sweep_results.json")


def evaluate_logit_lens(model, logit_lens, layer_idx, eval_loader, vocab_size, device):
    """Evaluate logit lens quality on eval data."""
    from rpvt.model.hooks import capture_residual_stream

    total_loss = 0.0
    count = 0
    logit_lens.eval()
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 50:
                break
            input_ids = batch["input_ids"].to(device)
            labels = input_ids[:, 1:]
            residuals = capture_residual_stream(model, input_ids, [layer_idx])
            hidden = residuals[layer_idx][:, :-1]
            logits = logit_lens(hidden)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


if __name__ == "__main__":
    main()
