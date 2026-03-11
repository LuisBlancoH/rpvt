"""Experiment 1d: Truncated backprop — how many downstream layers are needed?

We've established that truly local backprop (0 downstream layers) doesn't work.
Global backprop (all downstream layers) works but is expensive.

This experiment finds the minimum number of downstream layers needed for
useful gradient signal. If it's small (e.g., 3-5 layers), inference-time
learning is still feasible — much cheaper than full backprop.

For adapter at layer L in an N-layer model:
  - depth=0: local loss at layer L (broken, we know this)
  - depth=3: loss at layer L+3, backprop through 3 frozen layers
  - depth=N-L: full global loss (works, we know this)

We compute the loss at the final model output but only allow gradients
to flow backward from a certain depth. Everything beyond that depth
is detached.
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
    get_layers,
    get_num_layers,
    get_vocab_size,
)
from rpvt.model.adapter import attach_adapter
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


def remove_adapter(model, layer_idx, target, original_linear):
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


class _DetachHook:
    """Pre-forward hook that detaches hidden states to truncate gradient flow."""

    def __init__(self):
        self.handle = None

    def __call__(self, module, args, kwargs=None):
        # args[0] is hidden_states for Qwen2 decoder layers
        detached = args[0].detach().requires_grad_(True)
        return (detached,) + args[1:], kwargs

    def register(self, layer):
        self.handle = layer.register_forward_pre_hook(self, with_kwargs=True)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def forward_truncated(model, input_ids, adapter_layer_idx, grad_depth):
    """Forward pass with truncated gradient flow.

    Uses a pre-hook to detach the residual stream at the right layer,
    so gradients only flow through `grad_depth` layers after the adapter.
    """
    if grad_depth is None:
        return model(input_ids).logits

    layers = get_layers(model)
    num_layers = len(layers)
    detach_at = adapter_layer_idx + grad_depth

    hook = _DetachHook()
    if detach_at < num_layers:
        hook.register(layers[detach_at])

    try:
        logits = model(input_ids).logits
    finally:
        hook.remove()

    return logits


def train_and_eval(
    model, layer_idx, train_loader, eval_loader,
    vocab_size, device, num_steps, lr, target, rank,
    grad_depth=None, log_every=100,
):
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

        logits = forward_truncated(model, input_ids, layer_idx, grad_depth)
        logits = logits[:, :-1]
        loss = global_loss(logits, labels, vocab_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % log_every == 0:
            depth_str = "full" if grad_depth is None else str(grad_depth)
            print(f"    step {step}/{num_steps}, depth={depth_str}, loss={loss.item():.4f}")

    eval_loss = evaluate(model, eval_loader, vocab_size, device)
    remove_adapter(model, layer_idx, target, original_linear)
    return eval_loss


def main():
    parser = argparse.ArgumentParser(description="Experiment 1d: truncated backprop depth")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 25, 30])
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--target", default="mlp_out")
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp1_truncated")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    dtype = torch.bfloat16
    num_layers_model = None

    print(f"Loading {args.model}...")
    model, tokenizer = load_base_model(args.model, device=device, dtype=dtype)
    hidden_size = get_hidden_size(model)
    vocab_size = get_vocab_size(model)
    num_layers_model = get_num_layers(model)
    print(f"  {num_layers_model} layers, hidden_size={hidden_size}, vocab_size={vocab_size}")

    print("Loading dataset...")
    train_ds = TokenizedDataset(tokenizer, seq_len=args.seq_len, max_tokens=5_000_000)
    eval_ds = TokenizedDataset(tokenizer, split="validation", seq_len=args.seq_len, max_tokens=500_000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    baseline_loss = evaluate(model, eval_loader, vocab_size, device)
    print(f"  Baseline: {baseline_loss:.4f}")

    results = {"baseline": baseline_loss, "num_layers": num_layers_model, "layers": {}}

    for layer_idx in args.layers:
        max_possible_depth = num_layers_model - layer_idx
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx} (max depth = {max_possible_depth})")
        print(f"{'='*60}")

        # Full global backprop baseline
        print(f"  Full global (depth=all)...")
        global_eval = train_and_eval(
            model, layer_idx, train_loader, eval_loader,
            vocab_size, device,
            num_steps=args.train_steps, lr=args.lr,
            target=args.target, rank=args.rank,
            grad_depth=None,
        )
        global_improvement = baseline_loss - global_eval
        print(f"    eval: {global_eval:.4f} (improvement: {global_improvement:+.4f})")

        layer_results = {
            "global_eval": global_eval,
            "global_improvement": global_improvement,
            "depths": {},
        }

        # Truncated at each depth
        for depth in args.depths:
            if depth > max_possible_depth:
                continue
            print(f"  Depth={depth} (backprop through {depth} downstream layers)...")
            eval_loss = train_and_eval(
                model, layer_idx, train_loader, eval_loader,
                vocab_size, device,
                num_steps=args.train_steps, lr=args.lr,
                target=args.target, rank=args.rank,
                grad_depth=depth,
            )
            improvement = baseline_loss - eval_loss
            ratio = improvement / global_improvement if global_improvement > 0 else 0
            print(f"    eval: {eval_loss:.4f} (improvement: {improvement:+.4f}, ratio: {ratio:.1%})")
            layer_results["depths"][str(depth)] = {
                "eval_loss": eval_loss,
                "improvement": improvement,
                "ratio": ratio,
            }

        results["layers"][str(layer_idx)] = layer_results
        with open(Path(args.output_dir) / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Ratio of global improvement by gradient depth")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_loss:.4f}\n")

    all_depths = ["full"] + [str(d) for d in args.depths]
    print(f"{'Layer':>6} | ", end="")
    for d in all_depths:
        print(f"{'depth='+d:>12} | ", end="")
    print()
    print("-" * (10 + 15 * len(all_depths)))

    for layer_idx in args.layers:
        lr_key = str(layer_idx)
        if lr_key not in results["layers"]:
            continue
        r = results["layers"][lr_key]
        print(f"{layer_idx:>6} | {'100.0%':>12} | ", end="")
        for depth in args.depths:
            depth_r = r["depths"].get(str(depth), {})
            ratio = depth_r.get("ratio", None)
            if ratio is not None:
                print(f"{ratio:>11.1%} | ", end="")
            else:
                print(f"{'n/a':>12} | ", end="")
        print()

    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
