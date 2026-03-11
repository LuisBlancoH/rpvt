"""Experiment 1: Local backprop at one layer.

Validates that a LoRA adapter trained with local backprop through a logit lens
can match ~80% of the improvement from global backprop.

Two methods trained and compared:
  A) Global backprop — standard loss through full network, gradients to adapter only
  B) Local backprop — logit lens loss at layer 20 only, no gradients through other layers

Both compared against the unmodified base model.
"""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rpvt.model.base import (
    load_base_model,
    get_hidden_size,
    get_layers,
    get_num_layers,
    get_vocab_size,
)
from rpvt.model.adapter import attach_adapter
from rpvt.model.hooks import ResidualStreamCapture
from rpvt.model.logit_lens import LogitLens, train_logit_lens
from rpvt.training.data import TokenizedDataset
from rpvt.training.losses import global_loss, local_logit_lens_loss


def evaluate(model, dataloader, vocab_size, device, max_batches=50):
    """Evaluate next-token prediction loss on held-out data."""
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


def train_global(model, adapter, dataloader, vocab_size, device, num_steps, lr, log_every):
    """Train adapter with global backprop (standard loss through full model)."""
    optimizer = torch.optim.Adam(adapter.adapter.parameters(), lr=lr)
    model.train()
    step = 0
    losses = []

    for batch in dataloader:
        if step >= num_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = input_ids[:, 1:]

        logits = model(input_ids).logits[:, :-1]
        loss = global_loss(logits, labels, vocab_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        step += 1
        if step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"  [global] step {step}/{num_steps}, loss={avg:.4f}")

    return losses


def train_local(
    model, adapter, logit_lens, layer_idx, dataloader, vocab_size, device, num_steps, lr, log_every
):
    """Train adapter with local backprop (logit lens loss at target layer only)."""
    optimizer = torch.optim.Adam(adapter.adapter.parameters(), lr=lr)
    model.train()
    step = 0
    losses = []

    for batch in dataloader:
        if step >= num_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = input_ids[:, 1:]

        # Forward pass, capturing residual stream at target layer with gradients
        with ResidualStreamCapture(model, [layer_idx]) as cap:
            model(input_ids)
            hidden = cap.captured[layer_idx][:, :-1]

        loss = local_logit_lens_loss(hidden, labels, logit_lens, vocab_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        step += 1
        if step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"  [local] step {step}/{num_steps}, loss={avg:.4f}")

    return losses


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Local backprop at one layer")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B", help="Base model name")
    parser.add_argument("--layer", type=int, default=20, help="Layer index to adapt")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--target", default="mlp_out", help="Which weight matrix to adapt")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for adapter training")
    parser.add_argument("--lens-lr", type=float, default=1e-3, help="Learning rate for logit lens")
    parser.add_argument("--lens-steps", type=int, default=1000, help="Logit lens training steps")
    parser.add_argument("--train-steps", type=int, default=2000, help="Adapter training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--output-dir", default="results/exp1", help="Output directory")
    parser.add_argument("--log-every", type=int, default=50, help="Log interval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    dtype = torch.bfloat16

    # ── Load model ──
    print(f"Loading {args.model}...")
    model, tokenizer = load_base_model(args.model, device=device, dtype=dtype)
    hidden_size = get_hidden_size(model)
    vocab_size = get_vocab_size(model)
    num_layers = get_num_layers(model)
    print(f"  {num_layers} layers, hidden_size={hidden_size}, vocab_size={vocab_size}")

    # ── Load data ──
    print("Loading dataset...")
    train_ds = TokenizedDataset(tokenizer, seq_len=args.seq_len)
    eval_ds = TokenizedDataset(tokenizer, split="validation", seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # ── Baseline evaluation ──
    print("Evaluating baseline model...")
    baseline_loss = evaluate(model, eval_loader, vocab_size, device)
    print(f"  Baseline eval loss: {baseline_loss:.4f}")

    # ── Stage 1: Train logit lens ──
    print(f"\nTraining logit lens at layer {args.layer}...")
    logit_lens = train_logit_lens(
        model, tokenizer, args.layer, train_ds,
        hidden_size, vocab_size,
        device=device, dtype=dtype,
        lr=args.lens_lr, num_steps=args.lens_steps,
        batch_size=args.batch_size, seq_len=args.seq_len,
        log_every=args.log_every,
    )
    torch.save(logit_lens.state_dict(), Path(args.output_dir) / "logit_lens.pt")
    print("  Logit lens trained and saved.")

    # ── Method A: Global backprop ──
    print(f"\n{'='*60}")
    print("METHOD A: Global backprop")
    print(f"{'='*60}")

    adapted_global = attach_adapter(model, args.layer, target=args.target, rank=args.rank)
    global_losses = train_global(
        model, adapted_global, train_loader, vocab_size, device,
        num_steps=args.train_steps, lr=args.lr, log_every=args.log_every,
    )
    global_eval_loss = evaluate(model, eval_loader, vocab_size, device)
    print(f"  Global eval loss: {global_eval_loss:.4f} (baseline: {baseline_loss:.4f})")
    global_improvement = baseline_loss - global_eval_loss

    # Save global adapter weights and reset
    torch.save(adapted_global.adapter.state_dict(), Path(args.output_dir) / "adapter_global.pt")

    # Reset adapter to zero (B is initialized to zeros, so re-init both)
    adapted_global.adapter.A.data = torch.zeros_like(adapted_global.adapter.A.data)
    adapted_global.adapter.B.data = torch.zeros_like(adapted_global.adapter.B.data)

    # Verify reset
    reset_loss = evaluate(model, eval_loader, vocab_size, device)
    print(f"  After reset: {reset_loss:.4f} (should match baseline {baseline_loss:.4f})")

    # ── Method B: Local backprop ──
    print(f"\n{'='*60}")
    print("METHOD B: Local backprop")
    print(f"{'='*60}")

    # Re-init adapter properly
    import math
    torch.nn.init.kaiming_uniform_(adapted_global.adapter.A, a=math.sqrt(5))
    adapted_global.adapter.B.data.zero_()

    # Freeze logit lens during adapter training
    for p in logit_lens.parameters():
        p.requires_grad = False

    local_losses = train_local(
        model, adapted_global, logit_lens, args.layer, train_loader,
        vocab_size, device,
        num_steps=args.train_steps, lr=args.lr, log_every=args.log_every,
    )
    local_eval_loss = evaluate(model, eval_loader, vocab_size, device)
    print(f"  Local eval loss: {local_eval_loss:.4f} (baseline: {baseline_loss:.4f})")
    local_improvement = baseline_loss - local_eval_loss

    torch.save(adapted_global.adapter.state_dict(), Path(args.output_dir) / "adapter_local.pt")

    # ── Results ──
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Baseline loss:         {baseline_loss:.4f}")
    print(f"Global backprop loss:  {global_eval_loss:.4f}  (improvement: {global_improvement:+.4f})")
    print(f"Local backprop loss:   {local_eval_loss:.4f}  (improvement: {local_improvement:+.4f})")

    if global_improvement > 0:
        ratio = local_improvement / global_improvement
        print(f"\nLocal / Global ratio:  {ratio:.1%}")
        print(f"Target: >= 80%")
        print(f"{'PASS' if ratio >= 0.8 else 'FAIL'}: Local is {ratio:.1%} of global improvement")
    else:
        print("\nGlobal backprop didn't improve over baseline — check training setup.")

    # Save results
    results = {
        "args": vars(args),
        "baseline_loss": baseline_loss,
        "global_eval_loss": global_eval_loss,
        "local_eval_loss": local_eval_loss,
        "global_improvement": global_improvement,
        "local_improvement": local_improvement,
        "global_train_losses": global_losses,
        "local_train_losses": local_losses,
    }
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
