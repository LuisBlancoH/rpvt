"""Experiment 1f: Synthetic gradients.

Phase 1: Collect real gradients via global backprop.
Phase 2: Train gradient predictors to match them.
Phase 3: Train adapter using only synthetic gradients (no global backprop).
Phase 4: Compare adapter trained with synthetic grads vs global backprop.

The gradient predictor at each layer takes the hidden state and predicts
what gradient backprop would produce. If this works, every layer can
learn independently using only its local hidden state + predicted gradient.
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
from rpvt.model.adapter import attach_adapter, AdaptedLayer
from rpvt.model.synthetic_gradient import GradientPredictor
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
    """Remove adapter and restore original frozen linear."""
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


def collect_real_gradients(
    model, layer_idx, adapted, train_loader, vocab_size, device,
    num_steps=500, log_every=100,
):
    """Phase 1: Run global backprop and collect real gradients at the adapter layer.

    Returns list of (hidden_state, real_gradient) pairs for training the predictor.
    """
    layers = get_layers(model)
    layer = layers[layer_idx]

    gradient_data = []  # list of (hidden_state, gradient) tensors
    model.train()
    step = 0

    # Hook to capture hidden state entering the target layer AND its gradient
    captured_hidden = {}
    captured_grad = {}

    def fwd_hook(module, input, output):
        # Capture the layer's output (residual stream after this layer)
        out = output[0] if isinstance(output, tuple) else output
        out.retain_grad()
        captured_hidden["h"] = out

    def bwd_hook(module, grad_input, grad_output):
        # grad_output[0] is the gradient w.r.t. the layer's output
        captured_grad["g"] = grad_output[0].detach()

    fwd_handle = layer.register_forward_hook(fwd_hook)
    bwd_handle = layer.register_full_backward_hook(bwd_hook)

    optimizer = torch.optim.Adam(adapted.adapter.parameters(), lr=1e-4)

    try:
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

            if "h" in captured_hidden and "g" in captured_grad:
                # Store a subset to save memory (take first item in batch, subsample seq)
                h = captured_hidden["h"][0, ::4].detach().float().cpu()
                g = captured_grad["g"][0, ::4].detach().float().cpu()
                gradient_data.append((h, g))

            captured_hidden.clear()
            captured_grad.clear()

            step += 1
            if step % log_every == 0:
                print(f"    [collect] step {step}/{num_steps}, loss={loss.item():.4f}")
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    return gradient_data


def train_gradient_predictor(
    predictor, gradient_data, device, num_epochs=20, lr=1e-3, log_every=5,
):
    """Phase 2: Train the gradient predictor to match real gradients."""
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0
        count = 0

        # Shuffle data
        import random
        random.shuffle(gradient_data)

        for h, g in gradient_data:
            h = h.to(device)
            g = g.to(device)

            pred_g = predictor(h)
            loss = F.mse_loss(pred_g, g)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        if (epoch + 1) % log_every == 0:
            # Also compute cosine similarity
            with torch.no_grad():
                cos_sims = []
                for h, g in gradient_data[:50]:
                    h, g = h.to(device), g.to(device)
                    pred = predictor(h)
                    cos = F.cosine_similarity(pred.reshape(-1), g.reshape(-1), dim=0)
                    cos_sims.append(cos.item())
                avg_cos = sum(cos_sims) / len(cos_sims)
            print(f"    [predictor] epoch {epoch+1}/{num_epochs}, mse={avg_loss:.6f}, cos_sim={avg_cos:.4f}")

    return avg_loss


def train_with_synthetic_gradients(
    model, layer_idx, adapted, predictor, train_loader, vocab_size, device,
    num_steps=500, lr=1e-4, log_every=100,
):
    """Phase 3: Train the adapter using synthetic gradients only.

    Forward pass through the model, but instead of backprop from the loss,
    use the gradient predictor to get the gradient at the adapter layer,
    then backprop only through the adapter using that synthetic gradient.
    """
    layers = get_layers(model)
    layer = layers[layer_idx]

    optimizer = torch.optim.Adam(adapted.adapter.parameters(), lr=lr)
    model.train()
    step = 0

    for batch in train_loader:
        if step >= num_steps:
            break
        input_ids = batch["input_ids"].to(device)

        # Forward pass — capture hidden state at target layer WITH gradient tracking
        captured = {}

        def fwd_hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            captured["h"] = out

        handle = layer.register_forward_hook(fwd_hook)
        with torch.no_grad():
            model(input_ids)
        handle.remove()

        hidden = captured["h"]  # (batch, seq, hidden_size)

        # Get synthetic gradient from predictor (no grad through predictor)
        with torch.no_grad():
            synthetic_grad = predictor(hidden.float()).to(hidden.dtype)

        # Now: we need to backprop the synthetic gradient through just the adapter.
        # Re-run ONLY the adapter's contribution with gradients enabled.
        # The adapter's input is the input to the adapted linear layer.

        # Get the input to the adapted linear (the MLP intermediate activation)
        mlp_input = {}

        def capture_mlp_input(module, input, output):
            mlp_input["x"] = input[0] if isinstance(input, tuple) else input

        mlp_handle = adapted.register_forward_hook(capture_mlp_input)

        # Re-run forward to get adapter input with grad
        # We need to re-capture because the no_grad pass above didn't track
        captured2 = {}

        def fwd_hook2(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            captured2["h"] = out

        handle2 = layer.register_forward_hook(fwd_hook2)
        model(input_ids)
        handle2.remove()
        mlp_handle.remove()

        hidden_with_grad = captured2["h"]

        # Apply synthetic gradient: hidden.backward(synthetic_grad)
        # This sends the synthetic gradient backward through the layer,
        # which flows through the adapter and updates its parameters.
        optimizer.zero_grad()
        hidden_with_grad.backward(gradient=synthetic_grad)
        optimizer.step()

        step += 1
        if step % log_every == 0:
            # Quick eval
            with torch.no_grad():
                logits = model(input_ids).logits[:, :-1]
                labels = input_ids[:, 1:]
                loss = global_loss(logits, labels, vocab_size)
            print(f"    [synth] step {step}/{num_steps}, model_loss={loss.item():.4f}")

        # Staleness check: periodically compare synthetic vs real gradient
        if step % (log_every * 5) == 0:
            # Run one step of real backprop to get the true gradient
            real_captured = {}
            real_grad = {}

            def real_fwd(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                out.retain_grad()
                real_captured["h"] = out

            def real_bwd(module, grad_input, grad_output):
                real_grad["g"] = grad_output[0].detach()

            fh = layer.register_forward_hook(real_fwd)
            bh = layer.register_full_backward_hook(real_bwd)

            logits = model(input_ids).logits[:, :-1]
            labels = input_ids[:, 1:]
            real_loss = global_loss(logits, labels, vocab_size)
            real_loss.backward()

            fh.remove()
            bh.remove()

            if "h" in real_captured and "g" in real_grad:
                with torch.no_grad():
                    h_check = real_captured["h"].float()
                    g_real = real_grad["g"].float()
                    g_synth = predictor(h_check)
                    cos = F.cosine_similarity(
                        g_synth.reshape(-1), g_real.reshape(-1), dim=0
                    ).item()
                print(f"    [staleness] step {step}, cos_sim(synth, real)={cos:.4f}")

            # Zero grads from the staleness check
            optimizer.zero_grad()

    return


def main():
    parser = argparse.ArgumentParser(description="Experiment 1f: synthetic gradients")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--target", default="mlp_out")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--collect-steps", type=int, default=500, help="Steps to collect real gradients")
    parser.add_argument("--predictor-epochs", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=1000, help="Steps to train with synthetic grads")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp1_synthetic")
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
    baseline_loss = evaluate(model, eval_loader, vocab_size, device)
    print(f"  Baseline: {baseline_loss:.4f}")

    results = {"baseline": baseline_loss}

    # ── Phase 0: Global backprop reference ──
    print(f"\n{'='*60}")
    print("PHASE 0: Global backprop reference")
    print(f"{'='*60}")
    adapted = attach_adapter(model, args.layer, target=args.target, rank=args.rank)
    opt = torch.optim.Adam(adapted.adapter.parameters(), lr=args.lr)
    model.train()
    step = 0
    for batch in train_loader:
        if step >= args.train_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = input_ids[:, 1:]
        logits = model(input_ids).logits[:, :-1]
        loss = global_loss(logits, labels, vocab_size)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"    step {step}/{args.train_steps}, loss={loss.item():.4f}")

    global_eval = evaluate(model, eval_loader, vocab_size, device)
    global_improvement = baseline_loss - global_eval
    print(f"  Global eval: {global_eval:.4f} (improvement: {global_improvement:+.4f})")
    results["global"] = {"eval_loss": global_eval, "improvement": global_improvement}

    # Reset adapter
    remove_adapter(model, args.layer, args.target)

    # ── Phase 1: Collect real gradients ──
    print(f"\n{'='*60}")
    print("PHASE 1: Collecting real gradients via global backprop")
    print(f"{'='*60}")
    adapted = attach_adapter(model, args.layer, target=args.target, rank=args.rank)
    gradient_data = collect_real_gradients(
        model, args.layer, adapted, train_loader, vocab_size, device,
        num_steps=args.collect_steps,
    )
    print(f"  Collected {len(gradient_data)} gradient samples")

    # Save the adapter state after collection (it was trained during collection)
    collection_eval = evaluate(model, eval_loader, vocab_size, device)
    print(f"  Eval after collection phase: {collection_eval:.4f}")

    # Reset adapter for synthetic training
    nn.init.kaiming_uniform_(adapted.adapter.A, a=math.sqrt(5))
    adapted.adapter.B.data.zero_()
    reset_eval = evaluate(model, eval_loader, vocab_size, device)
    print(f"  Eval after adapter reset: {reset_eval:.4f}")

    # ── Phase 2: Train gradient predictor ──
    print(f"\n{'='*60}")
    print("PHASE 2: Training gradient predictor")
    print(f"{'='*60}")
    predictor = GradientPredictor(hidden_size, bottleneck=args.bottleneck).to(device)
    pred_loss = train_gradient_predictor(
        predictor, gradient_data, device,
        num_epochs=args.predictor_epochs,
    )
    results["predictor_final_loss"] = pred_loss

    # Check gradient prediction quality — full distribution
    print("\n  Gradient prediction quality check:")
    with torch.no_grad():
        norms_real = []
        norms_pred = []
        cos_sims = []
        # Per-token cosine sims (not just per-sample)
        cos_sims_per_token = []
        for h, g in gradient_data:
            h, g = h.to(device), g.to(device)
            pred = predictor(h)
            # Per-sample cosine sim
            cos = F.cosine_similarity(pred.reshape(-1), g.reshape(-1), dim=0)
            cos_sims.append(cos.item())
            # Per-token cosine sims
            for t in range(h.shape[0]):
                cos_t = F.cosine_similarity(pred[t:t+1], g[t:t+1], dim=-1)
                cos_sims_per_token.append(cos_t.item())
            norms_real.append(g.norm().item())
            norms_pred.append(pred.norm().item())

        cos_tensor = torch.tensor(cos_sims)
        cos_token_tensor = torch.tensor(cos_sims_per_token)
        pct_negative = (cos_tensor < 0).float().mean().item() * 100
        pct_negative_token = (cos_token_tensor < 0).float().mean().item() * 100

        print(f"  Per-sample cosine similarity (n={len(cos_sims)}):")
        print(f"    mean:  {cos_tensor.mean().item():.4f}")
        print(f"    std:   {cos_tensor.std().item():.4f}")
        print(f"    min:   {cos_tensor.min().item():.4f}")
        print(f"    max:   {cos_tensor.max().item():.4f}")
        print(f"    % negative: {pct_negative:.1f}%")
        print(f"  Per-token cosine similarity (n={len(cos_sims_per_token)}):")
        print(f"    mean:  {cos_token_tensor.mean().item():.4f}")
        print(f"    std:   {cos_token_tensor.std().item():.4f}")
        print(f"    min:   {cos_token_tensor.min().item():.4f}")
        print(f"    max:   {cos_token_tensor.max().item():.4f}")
        print(f"    % negative: {pct_negative_token:.1f}%")
        print(f"  Grad norms — real: {sum(norms_real)/len(norms_real):.6f}, pred: {sum(norms_pred)/len(norms_pred):.6f}")

        results["cos_sim_distribution"] = {
            "per_sample_mean": cos_tensor.mean().item(),
            "per_sample_std": cos_tensor.std().item(),
            "per_sample_min": cos_tensor.min().item(),
            "per_sample_max": cos_tensor.max().item(),
            "per_sample_pct_negative": pct_negative,
            "per_token_mean": cos_token_tensor.mean().item(),
            "per_token_std": cos_token_tensor.std().item(),
            "per_token_min": cos_token_tensor.min().item(),
            "per_token_max": cos_token_tensor.max().item(),
            "per_token_pct_negative": pct_negative_token,
        }

    torch.save(predictor.state_dict(), Path(args.output_dir) / "predictor.pt")

    # ── Phase 3: Train adapter with synthetic gradients ──
    print(f"\n{'='*60}")
    print("PHASE 3: Training adapter with synthetic gradients")
    print(f"{'='*60}")
    train_with_synthetic_gradients(
        model, args.layer, adapted, predictor, train_loader, vocab_size, device,
        num_steps=args.train_steps, lr=args.lr,
    )

    synth_eval = evaluate(model, eval_loader, vocab_size, device)
    synth_improvement = baseline_loss - synth_eval
    ratio = synth_improvement / global_improvement if global_improvement > 0 else 0
    print(f"  Synthetic eval: {synth_eval:.4f} (improvement: {synth_improvement:+.4f}, ratio: {ratio:.1%})")
    results["synthetic"] = {
        "eval_loss": synth_eval,
        "improvement": synth_improvement,
        "ratio": ratio,
    }

    # Cleanup
    remove_adapter(model, args.layer, args.target)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline:           {baseline_loss:.4f}")
    print(f"Global backprop:    {global_eval:.4f} (improvement: {global_improvement:+.4f})")
    print(f"Synthetic gradient: {synth_eval:.4f} (improvement: {synth_improvement:+.4f}, ratio: {ratio:.1%})")

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
