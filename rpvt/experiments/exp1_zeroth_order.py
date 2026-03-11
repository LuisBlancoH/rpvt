"""Experiment 1g: Zeroth-order gradient estimation.

Estimate gradients via random perturbation — no backprop, no activation storage.
Uses the REAL final model loss (not a proxy), so it can't be gamed.

For K random directions v:
    loss+ = forward(params + eps * v)
    loss- = forward(params - eps * v)
    grad_estimate += (loss+ - loss-) / (2 * eps) * v
grad_estimate /= K

Cost: 2K forward passes per step (no backward pass).
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rpvt.model.base import (
    load_base_model,
    get_hidden_size,
    get_num_layers,
    get_vocab_size,
)
from rpvt.model.adapter import attach_adapter, AdaptedLayer
from rpvt.model.base import get_layers
from rpvt.training.data import TokenizedDataset
from rpvt.training.losses import global_loss


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


def remove_adapter(model, layer_idx, target):
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
    adapted = getattr(parent, attr_path[-1])
    if isinstance(adapted, AdaptedLayer):
        setattr(parent, attr_path[-1], adapted.frozen)


def compute_loss_no_grad(model, input_ids, vocab_size):
    """Single forward pass, return scalar loss."""
    with torch.no_grad():
        logits = model(input_ids).logits[:, :-1]
        labels = input_ids[:, 1:]
        return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1)).item()


def zeroth_order_step(model, adapted, input_ids, vocab_size, K, epsilon):
    """Estimate gradient via K random perturbations and apply update.

    Returns the estimated gradient for A and B, and the base loss.
    """
    A_orig = adapted.adapter.A.data.clone()
    B_orig = adapted.adapter.B.data.clone()

    grad_A = torch.zeros_like(A_orig)
    grad_B = torch.zeros_like(B_orig)

    for k in range(K):
        # Random perturbation directions
        v_A = torch.randn_like(A_orig)
        v_B = torch.randn_like(B_orig)

        # Forward with +epsilon perturbation
        adapted.adapter.A.data = A_orig + epsilon * v_A
        adapted.adapter.B.data = B_orig + epsilon * v_B
        loss_plus = compute_loss_no_grad(model, input_ids, vocab_size)

        # Forward with -epsilon perturbation
        adapted.adapter.A.data = A_orig - epsilon * v_A
        adapted.adapter.B.data = B_orig - epsilon * v_B
        loss_minus = compute_loss_no_grad(model, input_ids, vocab_size)

        # Accumulate gradient estimate
        scale = (loss_plus - loss_minus) / (2.0 * epsilon)
        grad_A += scale * v_A
        grad_B += scale * v_B

    grad_A /= K
    grad_B /= K

    # Restore original params
    adapted.adapter.A.data = A_orig
    adapted.adapter.B.data = B_orig

    # Compute base loss for logging
    base_loss = compute_loss_no_grad(model, input_ids, vocab_size)

    return grad_A, grad_B, base_loss


def train_zeroth_order(
    model, adapted, train_loader, eval_loader,
    vocab_size, device, num_steps, lr, K, epsilon,
    log_every=50,
):
    """Train adapter with zeroth-order gradient estimation."""
    step = 0
    losses = []

    for batch in train_loader:
        if step >= num_steps:
            break
        input_ids = batch["input_ids"].to(device)

        grad_A, grad_B, base_loss = zeroth_order_step(
            model, adapted, input_ids, vocab_size, K, epsilon,
        )

        # SGD update (Adam doesn't help much with noisy ZO gradients)
        adapted.adapter.A.data -= lr * grad_A
        adapted.adapter.B.data -= lr * grad_B

        losses.append(base_loss)
        step += 1
        if step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"    step {step}/{num_steps}, K={K}, loss={avg:.4f}")

    eval_loss = evaluate(model, eval_loader, vocab_size, device)
    return eval_loss, losses


def train_global_baseline(
    model, layer_idx, train_loader, eval_loader,
    vocab_size, device, num_steps, lr, rank, target, log_every=100,
):
    adapted = attach_adapter(model, layer_idx, target=target, rank=rank)
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
            print(f"    step {step}/{num_steps}, loss={loss.item():.4f}")

    eval_loss = evaluate(model, eval_loader, vocab_size, device)
    remove_adapter(model, layer_idx, target)
    return eval_loss


def main():
    parser = argparse.ArgumentParser(description="Experiment 1g: zeroth-order gradient estimation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--target", default="mlp_out")
    parser.add_argument("--Ks", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4, help="Global backprop lr")
    parser.add_argument("--zo-lr", type=float, default=1e-2, help="Zeroth-order lr")
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp1_zeroth_order")
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

    # Adapter param count
    layers = get_layers(model)
    mlp_down = layers[args.layer].mlp.down_proj
    n_params = args.rank * (mlp_down.in_features + mlp_down.out_features)
    print(f"  Adapter: rank {args.rank}, {n_params:,} parameters")

    print("Loading dataset...")
    train_ds = TokenizedDataset(tokenizer, seq_len=args.seq_len, max_tokens=5_000_000)
    eval_ds = TokenizedDataset(tokenizer, split="validation", seq_len=args.seq_len, max_tokens=500_000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    baseline_loss = evaluate(model, eval_loader, vocab_size, device)
    print(f"  Baseline: {baseline_loss:.4f}")

    results = {"baseline": baseline_loss, "n_adapter_params": n_params}

    # Global backprop reference
    print(f"\n{'='*60}")
    print("GLOBAL BACKPROP BASELINE")
    print(f"{'='*60}")
    global_eval = train_global_baseline(
        model, args.layer, train_loader, eval_loader,
        vocab_size, device,
        num_steps=args.train_steps, lr=args.lr, rank=args.rank, target=args.target,
    )
    global_improvement = baseline_loss - global_eval
    print(f"  Global eval: {global_eval:.4f} (improvement: {global_improvement:+.4f})")
    results["global"] = {"eval_loss": global_eval, "improvement": global_improvement}

    # Zeroth-order at each K
    for K in args.Ks:
        print(f"\n{'='*60}")
        print(f"ZEROTH-ORDER K={K} (cost: {2*K} forward passes/step)")
        print(f"{'='*60}")

        adapted = attach_adapter(model, args.layer, target=args.target, rank=args.rank)

        eval_loss, train_losses = train_zeroth_order(
            model, adapted, train_loader, eval_loader,
            vocab_size, device,
            num_steps=args.train_steps, lr=args.zo_lr,
            K=K, epsilon=args.epsilon,
        )
        improvement = baseline_loss - eval_loss
        ratio = improvement / global_improvement if global_improvement > 0 else 0
        print(f"  Eval: {eval_loss:.4f} (improvement: {improvement:+.4f}, ratio: {ratio:.1%})")

        results[f"K={K}"] = {
            "eval_loss": eval_loss,
            "improvement": improvement,
            "ratio": ratio,
            "cost_forward_passes_per_step": 2 * K,
        }

        remove_adapter(model, args.layer, args.target)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_loss:.4f}")
    print(f"{'Method':>20} | {'Eval':>8} | {'Improve':>10} | {'Ratio':>8} | {'Cost':>12}")
    print("-" * 72)
    print(f"{'global backprop':>20} | {global_eval:>8.4f} | {global_improvement:>+10.4f} | {'100.0%':>8} | {'2 passes':>12}")
    for K in args.Ks:
        r = results[f"K={K}"]
        cost = f"{2*K} passes"
        print(f"{'ZO K='+str(K):>20} | {r['eval_loss']:>8.4f} | {r['improvement']:>+10.4f} | {r['ratio']:>7.1%} | {cost:>12}")

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
