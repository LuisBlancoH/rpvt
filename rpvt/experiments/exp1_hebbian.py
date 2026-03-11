"""Experiment 1e: Modulated Hebbian learning (no backprop for main adapter).

Instead of backprop through a projection (which the adapter games),
use a Hebbian update rule where:
- Direction comes from local input-output activity (can't game a proxy)
- Magnitude is modulated by surprise (prediction error) × reward (global scalar)

Two components at the target layer:
1. Prediction adapter: learns to predict neighboring layer's contribution (backprop, local)
2. Main adapter: Hebbian update modulated by surprise × reward (no backprop)

Variants:
- base: modulation = surprise × reward (reward from final loss vs running average)
- B3: modulation = surprise × local_reward (reward from prediction error reduction)
- B4: modulation = surprise × random_reward (sanity check)
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rpvt.model.base import (
    load_base_model,
    get_hidden_size,
    get_layers,
    get_num_layers,
    get_vocab_size,
)
from rpvt.model.hooks import capture_residual_stream
from rpvt.training.data import TokenizedDataset
from rpvt.training.losses import global_loss


class PredictionAdapter(nn.Module):
    """Predicts what the layer below contributed to the residual stream."""

    def __init__(self, hidden_size, rank=32):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.normal_(self.up.weight, std=0.01)

    def forward(self, x):
        return self.up(self.down(x))


class HebbianLoRA(nn.Module):
    """LoRA adapter updated via modulated Hebbian learning, not backprop.

    The forward pass computes: output = (x @ B.T) @ A.T
    Updates use outer products of local activity modulated by a scalar signal.
    """

    def __init__(self, in_features, out_features, rank=32):
        super().__init__()
        self.rank = rank
        self.scale = 1.0 / rank

        # Not nn.Parameters — updated manually
        self.A = torch.randn(out_features, rank, device="cuda") * 0.01
        self.B = torch.randn(rank, in_features, device="cuda") * 0.01

    def forward(self, x):
        return ((x @ self.B.T) @ self.A.T) * self.scale

    def hebbian_update(self, x, output, modulation, lr=1e-4):
        """Update A and B using modulated Hebbian learning.

        Args:
            x: (batch, seq, in_features) — input to adapter
            output: (batch, seq, out_features) — output of adapter
            modulation: (batch, seq, 1) — surprise × reward signal
            lr: learning rate
        """
        # Weight by modulation and average over batch and sequence
        x_mod = (x * modulation).mean(dim=(0, 1))  # (in_features,)
        rank_act = (x @ self.B.T * modulation).mean(dim=(0, 1))  # (rank,)
        out_mod = (output * modulation).mean(dim=(0, 1))  # (out_features,)

        # Outer product updates
        delta_B = torch.outer(rank_act, x_mod)  # (rank, in_features)
        delta_A = torch.outer(out_mod, rank_act)  # (out_features, rank)

        # Normalize to prevent explosion
        delta_A_norm = delta_A.norm()
        delta_B_norm = delta_B.norm()
        if delta_A_norm > 0:
            delta_A = delta_A / delta_A_norm
        if delta_B_norm > 0:
            delta_B = delta_B / delta_B_norm

        self.A += lr * delta_A
        self.B += lr * delta_B

    def to(self, device=None, dtype=None):
        if device is not None:
            self.A = self.A.to(device)
            self.B = self.B.to(device)
        if dtype is not None:
            self.A = self.A.to(dtype)
            self.B = self.B.to(dtype)
        return self


def inject_hebbian_adapter(model, layer_idx, adapter):
    """Register a forward hook that adds the Hebbian adapter's output to the layer.

    We hook the MLP's down_proj (which in Qwen is actually mlp.down_proj)
    and add the adapter's contribution.
    """
    layers = get_layers(model)
    layer = layers[layer_idx]

    # Store adapter input/output for Hebbian update
    adapter._last_input = None
    adapter._last_output = None

    def hook_fn(module, input, output):
        x = input[0] if isinstance(input, tuple) else input
        adapter_out = adapter(x.detach().to(torch.float32)).to(output.dtype)
        adapter._last_input = x.detach()
        adapter._last_output = adapter_out.detach()
        return output + adapter_out

    handle = layer.mlp.down_proj.register_forward_hook(hook_fn)
    return handle


def evaluate(model, dataloader, vocab_size, device, max_batches=50):
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


def train_hebbian(
    model, layer_idx, train_loader, eval_loader,
    vocab_size, hidden_size, device, num_steps, lr,
    rank=32, variant="base", log_every=100,
):
    """Train with modulated Hebbian learning.

    variant: "base" (surprise × global reward), "B3" (surprise × local reward),
             "B4" (surprise × random reward)
    """
    num_layers = get_num_layers(model)
    layers = get_layers(model)

    # Get MLP dimensions
    mlp_down = layers[layer_idx].mlp.down_proj
    in_features = mlp_down.in_features
    out_features = mlp_down.out_features

    # Create adapters
    hebbian = HebbianLoRA(in_features, out_features, rank=rank)
    hebbian.to(device=device, dtype=torch.float32)

    prediction = PredictionAdapter(hidden_size, rank=rank).to(device)
    pred_optimizer = torch.optim.Adam(prediction.parameters(), lr=1e-3)

    # Inject Hebbian adapter
    hook_handle = inject_hebbian_adapter(model, layer_idx, hebbian)

    # Running stats
    running_avg_loss = None
    prev_surprise = None

    step = 0
    pred_losses = []
    eval_losses = []

    model.eval()  # Keep model in eval mode — we're not doing backprop through it

    # Layers to capture: layer before target, target, and layer after
    capture_indices = []
    if layer_idx > 0:
        capture_indices.append(layer_idx - 1)
    capture_indices.append(layer_idx)

    for batch in train_loader:
        if step >= num_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = input_ids[:, 1:]

        # Forward pass — captures residual streams, adapter hooks add contribution
        with torch.no_grad():
            residuals = capture_residual_stream(model, input_ids, capture_indices)

            # Final model loss (for reward)
            logits = model(input_ids).logits[:, :-1]
            final_loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), labels.reshape(-1)
            ).item()

        # Layer contributions
        if layer_idx > 0:
            # What the layer below contributed
            hidden_at_target = residuals[layer_idx]
            hidden_before = residuals[layer_idx - 1]
            actual_contribution = (hidden_at_target - hidden_before).detach()

            # Prediction adapter: predict what layer below contributed
            predicted = prediction(hidden_at_target.detach().to(torch.float32))
            pred_loss = F.mse_loss(predicted, actual_contribution.to(torch.float32))

            pred_optimizer.zero_grad()
            pred_loss.backward()
            pred_optimizer.step()
            pred_losses.append(pred_loss.item())

            # Surprise signal
            with torch.no_grad():
                pred_error = (actual_contribution.to(torch.float32) - predicted.detach())
                surprise = pred_error.norm(dim=-1, keepdim=True)  # (batch, seq, 1)
        else:
            surprise = torch.ones(input_ids.shape[0], input_ids.shape[1], 1, device=device)
            pred_losses.append(0.0)

        # Reward signal
        with torch.no_grad():
            if variant == "base":
                if running_avg_loss is None:
                    running_avg_loss = final_loss
                reward = -(final_loss - running_avg_loss)
                running_avg_loss = 0.99 * running_avg_loss + 0.01 * final_loss
                modulation = surprise * reward

            elif variant == "B3":
                # Local reward: is prediction error decreasing?
                current_surprise_mean = surprise.mean().item()
                if prev_surprise is None:
                    prev_surprise = current_surprise_mean
                local_reward = prev_surprise - current_surprise_mean
                prev_surprise = 0.99 * prev_surprise + 0.01 * current_surprise_mean
                modulation = surprise * local_reward

            elif variant == "B4":
                # Random reward (sanity check)
                random_reward = torch.randn(1, device=device).item()
                modulation = surprise * random_reward

            else:
                raise ValueError(f"Unknown variant: {variant}")

        # Hebbian update
        if hebbian._last_input is not None and hebbian._last_output is not None:
            # Trim modulation to match adapter dimensions
            mod = modulation[:, :hebbian._last_input.shape[1]].to(torch.float32)
            hebbian.hebbian_update(
                x=hebbian._last_input.to(torch.float32),
                output=hebbian._last_output.to(torch.float32),
                modulation=mod,
                lr=lr,
            )

        step += 1
        if step % log_every == 0:
            avg_pred = sum(pred_losses[-log_every:]) / log_every
            print(
                f"    step {step}/{num_steps}, "
                f"model_loss={final_loss:.4f}, "
                f"pred_loss={avg_pred:.4f}, "
                f"surprise={surprise.mean().item():.4f}"
            )

        # Periodic eval
        if step % (log_every * 5) == 0:
            el = evaluate(model, eval_loader, vocab_size, device, max_batches=20)
            eval_losses.append((step, el))
            print(f"    [eval] step {step}, loss={el:.4f}")

    # Final eval
    final_eval = evaluate(model, eval_loader, vocab_size, device)

    # Cleanup
    hook_handle.remove()

    return final_eval, pred_losses, eval_losses


def train_global_baseline(
    model, layer_idx, train_loader, eval_loader,
    vocab_size, device, num_steps, lr, rank, target="mlp_out", log_every=100,
):
    """Standard global backprop baseline for comparison."""
    from rpvt.model.adapter import attach_adapter

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
            print(f"    step {step}/{num_steps}, loss={loss.item():.4f}")

    eval_loss = evaluate(model, eval_loader, vocab_size, device)

    # Cleanup
    from rpvt.model.base import get_layers
    layers = get_layers(model)
    layer = layers[layer_idx]
    target_map = {"mlp_out": "mlp.down_proj"}
    attr_path = target_map[target].split(".")
    parent = layer
    for attr in attr_path[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, attr_path[-1], original_linear)

    return eval_loss


def main():
    parser = argparse.ArgumentParser(description="Experiment 1e: Hebbian learning")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Hebbian learning rate")
    parser.add_argument("--global-lr", type=float, default=1e-4, help="Global backprop lr")
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp1_hebbian")
    parser.add_argument("--variants", nargs="+", default=["base", "B3", "B4"])
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

    results = {"baseline": baseline_loss, "methods": {}}

    # Global backprop reference
    print(f"\n{'='*60}")
    print("GLOBAL BACKPROP BASELINE")
    print(f"{'='*60}")
    global_eval = train_global_baseline(
        model, args.layer, train_loader, eval_loader,
        vocab_size, device,
        num_steps=args.train_steps, lr=args.global_lr, rank=args.rank,
    )
    global_improvement = baseline_loss - global_eval
    print(f"  Global eval: {global_eval:.4f} (improvement: {global_improvement:+.4f})")
    results["methods"]["global"] = {
        "eval_loss": global_eval,
        "improvement": global_improvement,
    }

    # Hebbian variants
    for variant in args.variants:
        print(f"\n{'='*60}")
        print(f"HEBBIAN VARIANT: {variant}")
        print(f"{'='*60}")

        eval_loss, pred_losses, eval_curve = train_hebbian(
            model, args.layer, train_loader, eval_loader,
            vocab_size, hidden_size, device,
            num_steps=args.train_steps, lr=args.lr, rank=args.rank,
            variant=variant,
        )
        improvement = baseline_loss - eval_loss
        ratio = improvement / global_improvement if global_improvement > 0 else 0
        print(f"  Eval: {eval_loss:.4f} (improvement: {improvement:+.4f}, ratio: {ratio:.1%})")

        results["methods"][f"hebbian_{variant}"] = {
            "eval_loss": eval_loss,
            "improvement": improvement,
            "ratio": ratio,
            "eval_curve": eval_curve,
            "final_pred_loss": pred_losses[-1] if pred_losses else None,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline:        {baseline_loss:.4f}")
    for name, r in results["methods"].items():
        ratio = r.get("ratio", 1.0)
        print(f"{name:>16}: {r['eval_loss']:.4f} (improvement: {r['improvement']:+.4f}, ratio: {ratio:.1%})")

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
