"""Experiment 4: Multi-timescale adapters with continual learning.

The core architecture validation: can multi-timescale LoRA adapters
learn continuously across domains without catastrophic forgetting?

Setup:
  - Frozen Qwen2.5-3B with multi-timescale adapters at every layer
  - Three domains: A (Wikipedia), B (Code), C (held-out wiki)
  - Train sequentially: A → B → C
  - After each domain, evaluate on ALL domains

Compare:
  1. Single-timescale LoRA (standard, one lr)
  2. Multi-timescale LoRA (fast/med/slow with different lr and decay)
  3. No adapter (frozen baseline)

Key metrics:
  - Forward transfer: does learning A help with B?
  - Backward forgetting: after learning B, how much of A is lost?
  - Does multi-timescale reduce forgetting vs single-timescale?
"""

import argparse
import json
import math
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rpvt.model.base import (
    load_base_model,
    get_hidden_size,
    get_num_layers,
    get_vocab_size,
)
from rpvt.model.multiscale_adapter import (
    attach_multiscale_adapters,
    remove_multiscale_adapters,
    AdaptedLayerMultiScale,
)
from rpvt.model.adapter import attach_adapter, AdaptedLayer
from rpvt.model.base import get_layers
from rpvt.training.continual import load_domain_datasets, evaluate_on_domain
from rpvt.training.losses import global_loss


def remove_single_adapters(model, layer_indices, targets):
    """Remove single-scale adapters."""
    layers = get_layers(model)
    target_map = {
        "mlp_out": "mlp.down_proj",
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
    }
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for target in targets:
            attr_path = target_map[target].split(".")
            parent = layer
            for attr in attr_path[:-1]:
                parent = getattr(parent, attr)
            adapted = getattr(parent, attr_path[-1])
            if isinstance(adapted, AdaptedLayer):
                setattr(parent, attr_path[-1], adapted.frozen)


def attach_single_adapters(model, layer_indices, targets, rank):
    """Attach standard single-timescale LoRA adapters at all specified layers."""
    adapted_modules = []
    for layer_idx in layer_indices:
        for target in targets:
            adapted = attach_adapter(model, layer_idx, target=target, rank=rank)
            adapted_modules.append(adapted)
    return adapted_modules


def train_on_domain(
    model, adapted_modules, dataset, vocab_size, device,
    num_steps, optimizer, batch_size=4, log_every=200,
    decay_fn=None,
):
    """Train on a single domain for num_steps."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.train()
    step = 0
    losses = []

    for batch in loader:
        if step >= num_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = input_ids[:, 1:]

        logits = model(input_ids).logits[:, :-1]
        loss = global_loss(logits, labels, vocab_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Apply decay to fast/medium timescales
        if decay_fn is not None:
            decay_fn()

        losses.append(loss.item())
        step += 1
        if step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"      step {step}/{num_steps}, loss={avg:.4f}")

    return losses


def evaluate_all_domains(model, domains, vocab_size, device, batch_size=4):
    """Evaluate on all domains, return dict of losses."""
    results = {}
    for name, ds in domains.items():
        loss = evaluate_on_domain(model, ds, vocab_size, device, batch_size=batch_size)
        results[name] = loss
    return results


def run_continual_experiment(
    model, domains, vocab_size, device, domain_order,
    adapted_modules, optimizer, num_steps_per_domain,
    method_name, decay_fn=None, batch_size=4,
):
    """Run continual learning: train on domains sequentially, eval after each."""
    print(f"\n  Evaluating before training...")
    pre_eval = evaluate_all_domains(model, domains, vocab_size, device, batch_size)
    for name, loss in pre_eval.items():
        print(f"    {name}: {loss:.4f}")

    history = {"pre": pre_eval, "after": {}}

    for domain_key in domain_order:
        ds = domains[domain_key]
        print(f"\n  Training on Domain {domain_key} ({ds.name}, {num_steps_per_domain} steps)...")

        train_on_domain(
            model, adapted_modules, ds, vocab_size, device,
            num_steps=num_steps_per_domain, optimizer=optimizer,
            batch_size=batch_size, decay_fn=decay_fn,
        )

        print(f"  Evaluating after Domain {domain_key}...")
        post_eval = evaluate_all_domains(model, domains, vocab_size, device, batch_size)
        for name, loss in post_eval.items():
            print(f"    {name}: {loss:.4f}")

        history["after"][domain_key] = post_eval

        # Report timescale norms for multi-scale
        if adapted_modules and hasattr(adapted_modules[0], 'adapter') and hasattr(adapted_modules[0].adapter, 'get_timescale_norms'):
            # Sample a few layers
            for i in [0, len(adapted_modules)//2, -1]:
                norms = adapted_modules[i].adapter.get_timescale_norms()
                print(f"    Adapter {i} norms: fast={norms['fast']:.4f}, med={norms['medium']:.4f}, slow={norms['slow']:.4f}")

    return history


def compute_forgetting(history, domain_order):
    """Compute forgetting metrics from evaluation history.

    Returns dict with:
      - per-domain improvement when trained on it
      - per-domain forgetting after training on subsequent domains
    """
    metrics = {}

    for i, domain in enumerate(domain_order):
        # Improvement on this domain after training on it
        pre = history["pre"][domain]
        post = history["after"][domain][domain]
        improvement = pre - post
        metrics[f"{domain}_improvement"] = improvement

        # Forgetting: how much was lost after subsequent domains
        if i < len(domain_order) - 1:
            final_domain = domain_order[-1]
            final_loss = history["after"][final_domain][domain]
            forgetting = final_loss - post  # positive = forgot
            retention = 1.0 - (forgetting / max(improvement, 1e-6)) if improvement > 0 else 1.0
            metrics[f"{domain}_forgetting"] = forgetting
            metrics[f"{domain}_retention"] = retention

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: multi-timescale continual learning")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--targets", nargs="+", default=["mlp_out"])
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--fast-lr-mult", type=float, default=10.0)
    parser.add_argument("--slow-lr-mult", type=float, default=0.1)
    parser.add_argument("--fast-decay", type=float, default=0.99)
    parser.add_argument("--med-decay", type=float, default=0.9999)
    parser.add_argument("--steps-per-domain", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp4")
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

    # Use all layers
    layer_indices = list(range(num_layers))

    print("\nLoading domain datasets...")
    domains = load_domain_datasets(tokenizer, seq_len=args.seq_len)
    domain_order = ["A", "B", "C"]

    results = {}

    # ── Frozen baseline ──
    print(f"\n{'='*60}")
    print("FROZEN BASELINE (no adapters)")
    print(f"{'='*60}")
    baseline = evaluate_all_domains(model, domains, vocab_size, device, args.batch_size)
    for name, loss in baseline.items():
        print(f"  {name}: {loss:.4f}")
    results["baseline"] = baseline

    # ── Method 1: Single-timescale LoRA ──
    print(f"\n{'='*60}")
    print("METHOD 1: Single-timescale LoRA (all layers)")
    print(f"{'='*60}")

    adapted_single = attach_single_adapters(model, layer_indices, args.targets, args.rank)
    all_adapter_params = []
    for m in adapted_single:
        all_adapter_params.extend([m.adapter.A, m.adapter.B])
    optimizer_single = torch.optim.Adam(all_adapter_params, lr=args.base_lr)

    n_params_single = sum(p.numel() for p in all_adapter_params)
    print(f"  {len(adapted_single)} adapters, {n_params_single:,} total parameters")

    history_single = run_continual_experiment(
        model, domains, vocab_size, device, domain_order,
        adapted_single, optimizer_single, args.steps_per_domain,
        method_name="single",
        batch_size=args.batch_size,
    )
    metrics_single = compute_forgetting(history_single, domain_order)
    results["single"] = {"history": history_single, "metrics": metrics_single}

    # Cleanup
    remove_single_adapters(model, layer_indices, args.targets)

    # ── Method 2: Multi-timescale LoRA ──
    print(f"\n{'='*60}")
    print("METHOD 2: Multi-timescale LoRA (all layers)")
    print(f"{'='*60}")

    adapted_multi = attach_multiscale_adapters(model, layer_indices, args.targets, args.rank)

    # Build optimizer with per-timescale learning rates
    param_groups = []
    for m in adapted_multi:
        param_groups.extend(m.adapter.get_param_groups(
            args.base_lr,
            fast_lr_mult=args.fast_lr_mult,
            slow_lr_mult=args.slow_lr_mult,
        ))
    optimizer_multi = torch.optim.Adam(param_groups)

    n_params_multi = sum(
        p.numel()
        for group in param_groups
        for p in group["params"]
    )
    print(f"  {len(adapted_multi)} adapters, {n_params_multi:,} total parameters (3x single)")

    # Decay function
    def decay_fn():
        for m in adapted_multi:
            m.adapter.decay_fast(args.fast_decay)
            m.adapter.decay_medium(args.med_decay)

    history_multi = run_continual_experiment(
        model, domains, vocab_size, device, domain_order,
        adapted_multi, optimizer_multi, args.steps_per_domain,
        method_name="multi",
        decay_fn=decay_fn,
        batch_size=args.batch_size,
    )
    metrics_multi = compute_forgetting(history_multi, domain_order)
    results["multi"] = {"history": history_multi, "metrics": metrics_multi}

    remove_multiscale_adapters(model, layer_indices, args.targets)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY: Continual Learning Results")
    print(f"{'='*60}")

    print(f"\nBaseline losses:")
    for name, loss in baseline.items():
        print(f"  {name}: {loss:.4f}")

    for method_name, method_key in [("Single-timescale", "single"), ("Multi-timescale", "multi")]:
        print(f"\n{method_name}:")
        m = results[method_key]["metrics"]

        for domain in domain_order:
            imp = m.get(f"{domain}_improvement", 0)
            print(f"  Domain {domain}: improvement={imp:+.4f}", end="")
            if f"{domain}_forgetting" in m:
                forg = m[f"{domain}_forgetting"]
                ret = m[f"{domain}_retention"]
                print(f", forgetting={forg:+.4f}, retention={ret:.1%}", end="")
            print()

    # Final comparison
    print(f"\n{'='*60}")
    print("KEY COMPARISON: Retention of Domain A after learning B and C")
    print(f"{'='*60}")
    ret_single = results["single"]["metrics"].get("A_retention", None)
    ret_multi = results["multi"]["metrics"].get("A_retention", None)
    if ret_single is not None and ret_multi is not None:
        print(f"  Single-timescale: {ret_single:.1%}")
        print(f"  Multi-timescale:  {ret_multi:.1%}")
        if ret_multi > ret_single:
            print(f"  Multi-timescale retains {ret_multi - ret_single:.1%} more knowledge")
        else:
            print(f"  No advantage for multi-timescale")

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
