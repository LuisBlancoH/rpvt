"""Experiment v2.1: Fast weight memory pretraining.

The fundamental question: does a transformer learn to USE fast weight memory
when it's available during pretraining?

Setup:
    - 125M parameter transformer (GPT-2 small config: 12 layers, 768 hidden, 12 heads)
    - Train from scratch on wikitext-103
    - Compare: baseline (no memory) vs memory (fast weight at every layer)
    - Both models have identical transformer weights — memory model has extra
      projections (W_query, W_key, W_value, W_out, gate) per layer

Measurements:
    1. Final validation loss — does memory model achieve lower perplexity?
    2. Gate values per layer — does the model learn to open the gates?
       (If gates stay ~0, model ignores memory = experiment failed)
    3. Memory norm over time — is the model actually writing to memory?
    4. Ablation: train with memory, then eval with memory zeroed out.
       If loss increases, the model is using the memory.

Cost estimate:
    - 125M params + ~3M memory params = 128M total
    - wikitext-103: ~100M tokens
    - ~2-4 hours on A100 for a reasonable training run
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
)

from rpvt.model.fast_weight import (
    attach_fast_weight_memory,
    TransformerLayerWithMemory,
)


def load_data(tokenizer, seq_len=512, max_tokens=50_000_000):
    """Load and tokenize wikitext-103."""
    from datasets import load_dataset

    print("Loading wikitext-103...")
    raw_train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    raw_val = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

    def tokenize_texts(texts, max_tok):
        all_tokens = []
        batch_texts = []
        batch_chars = 0
        for text in texts:
            if len(text.strip()) < 50:
                continue
            batch_texts.append(text)
            batch_chars += len(text)
            if batch_chars >= 500_000:
                encoded = tokenizer(batch_texts, return_attention_mask=False, truncation=False)
                for ids in encoded["input_ids"]:
                    all_tokens.extend(ids)
                batch_texts = []
                batch_chars = 0
                if len(all_tokens) >= max_tok:
                    break
        if batch_texts and len(all_tokens) < max_tok:
            encoded = tokenizer(batch_texts, return_attention_mask=False, truncation=False)
            for ids in encoded["input_ids"]:
                all_tokens.extend(ids)
        all_tokens = all_tokens[:max_tok]
        tokens = torch.tensor(all_tokens, dtype=torch.long)
        n_chunks = len(tokens) // seq_len
        return tokens[:n_chunks * seq_len].reshape(n_chunks, seq_len)

    train_chunks = tokenize_texts(raw_train["text"], max_tokens)
    val_chunks = tokenize_texts(raw_val["text"], max_tokens // 10)
    print(f"  Train: {train_chunks.shape[0]:,} chunks of {seq_len}")
    print(f"  Val:   {val_chunks.shape[0]:,} chunks of {seq_len}")
    return train_chunks, val_chunks


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"input_ids": self.chunks[idx]}


def create_model(vocab_size, device, dtype, with_memory=False, memory_size=256, decay=0.99,
                  pretrained=False, freeze_base=False):
    """Create a GPT-2 model, optionally with fast weight memory.

    Args:
        pretrained: If True, load pretrained GPT-2 weights instead of random init.
        freeze_base: If True, freeze all non-memory parameters (only train memory projections).
    """
    if pretrained:
        print("  Loading pretrained GPT-2...")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device, dtype=dtype)
    else:
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=3072,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
        model = GPT2LMHeadModel(config).to(device=device, dtype=dtype)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    memory_modules = []
    if with_memory:
        memory_modules = attach_fast_weight_memory(
            model.transformer.h,
            hidden_size=model.config.n_embd,
            layer_indices=list(range(model.config.n_layer)),
            memory_size=memory_size,
            decay=decay,
        )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_memory = sum(p.numel() for m in memory_modules for p in m.parameters())
    print(f"  Total params: {n_params:,} ({n_trainable:,} trainable)")
    if memory_modules:
        print(f"  Memory params: {n_memory:,} ({n_memory/n_params:.1%} of total)")

    return model, memory_modules


def evaluate(model, val_loader, device, max_batches=100):
    """Evaluate validation loss."""
    model.eval()
    total_loss = 0.0
    count = 0

    # Reset memories before eval
    for layer in (model.transformer.h if hasattr(model.transformer, 'h') else []):
        if isinstance(layer, TransformerLayerWithMemory):
            layer.reset_memory()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item()
            count += 1

    return total_loss / max(count, 1)


def get_gate_values(model):
    """Get gate values from all memory layers."""
    gates = {}
    for i, layer in enumerate(model.transformer.h):
        if isinstance(layer, TransformerLayerWithMemory):
            gate_val = torch.sigmoid(layer.memory.gate).item()
            gates[f"layer_{i}"] = gate_val
    return gates


def get_memory_norms(model):
    """Get memory matrix norms from all memory layers."""
    norms = {}
    for i, layer in enumerate(model.transformer.h):
        if isinstance(layer, TransformerLayerWithMemory):
            norm = layer.memory.M.norm().item()
            norms[f"layer_{i}"] = norm
    return norms


def train_model(
    model, train_loader, val_loader, device,
    num_epochs=3, lr=3e-4, warmup_steps=500,
    log_every=200, eval_every=2000, model_name="model",
):
    """Train with AdamW and linear warmup + cosine decay."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    total_steps = len(train_loader) * num_epochs
    warmup = warmup_steps

    def lr_schedule(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"\nTraining {model_name}: {total_steps} steps, lr={lr}")
    model.train()
    global_step = 0
    best_val = float("inf")
    train_losses = []
    log_data = []
    start_time = time.time()

    for epoch in range(num_epochs):
        # No memory reset between sequences — M accumulates across the epoch.
        # The decay naturally manages capacity (old writes fade).
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = sum(train_losses[-log_every:]) / log_every
                elapsed = time.time() - start_time
                tokens_per_sec = (global_step * input_ids.shape[0] * input_ids.shape[1]) / elapsed

                log_entry = {
                    "step": global_step,
                    "train_loss": avg_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_per_sec": tokens_per_sec,
                }

                # Log gate values and memory norms if applicable
                gates = get_gate_values(model)
                if gates:
                    log_entry["gates"] = gates
                    mean_gate = sum(gates.values()) / len(gates)
                    log_entry["mean_gate"] = mean_gate

                norms = get_memory_norms(model)
                if norms:
                    log_entry["memory_norms"] = norms
                    mean_norm = sum(norms.values()) / len(norms)
                    log_entry["mean_memory_norm"] = mean_norm

                log_data.append(log_entry)

                gate_str = f", gate={log_entry.get('mean_gate', 0):.4f}" if gates else ""
                norm_str = f", M_norm={log_entry.get('mean_memory_norm', 0):.1f}" if norms else ""
                print(
                    f"  [{model_name}] step {global_step}/{total_steps}, "
                    f"loss={avg_loss:.4f}, lr={log_entry['lr']:.2e}, "
                    f"{tokens_per_sec:.0f} tok/s{gate_str}{norm_str}"
                )

            if global_step % eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"  [{model_name}] step {global_step} val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                model.train()

    final_val = evaluate(model, val_loader, device)
    print(f"  [{model_name}] Final val_loss={final_val:.4f}")

    return final_val, log_data


def memory_ablation(model, val_loader, device):
    """Evaluate with memory active vs zeroed out.

    If loss increases when memory is zeroed, the model is using the memory.
    """
    # Eval with memory (normal)
    loss_with = evaluate(model, val_loader, device)

    # Zero out all memory gates
    original_gates = {}
    for i, layer in enumerate(model.transformer.h):
        if isinstance(layer, TransformerLayerWithMemory):
            original_gates[i] = layer.memory.gate.data.clone()
            layer.memory.gate.data.fill_(-100.0)  # sigmoid(-100) ~ 0

    loss_without = evaluate(model, val_loader, device)

    # Restore gates
    for i, gate_val in original_gates.items():
        model.transformer.h[i].memory.gate.data = gate_val

    return loss_with, loss_without


def run_ablation_and_summary(model, val_loader, device, results, output_dir):
    """Run memory ablation, print gate values and summary, save results."""

    # ── Memory ablation ──
    print(f"\n{'='*60}")
    print("ABLATION: Memory active vs zeroed")
    print(f"{'='*60}")

    loss_with, loss_without = memory_ablation(model, val_loader, device)
    memory_contribution = loss_without - loss_with
    print(f"  With memory:    {loss_with:.4f}")
    print(f"  Without memory: {loss_without:.4f}")
    print(f"  Memory contribution: {memory_contribution:+.4f}")
    if memory_contribution > 0.01:
        print(f"  Model IS using the memory.")
    elif memory_contribution > 0:
        print(f"  Model is using memory minimally.")
    else:
        print(f"  Model is NOT using the memory.")

    results["ablation"] = {
        "loss_with_memory": loss_with,
        "loss_without_memory": loss_without,
        "memory_contribution": memory_contribution,
    }

    # ── Final gate values ──
    print(f"\n{'='*60}")
    print("GATE VALUES (sigmoid of learned gate parameter)")
    print(f"{'='*60}")
    gates = get_gate_values(model)
    for layer, val in gates.items():
        bar = "#" * int(val * 50)
        print(f"  {layer}: {val:.4f} |{bar}")
    results["final_gates"] = gates

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if "baseline" in results:
        print(f"  Baseline val loss:  {results['baseline']['val_loss']:.4f}")
    if "memory" in results:
        print(f"  Memory val loss:    {results['memory']['val_loss']:.4f}")
    if "baseline" in results and "memory" in results:
        diff = results["baseline"]["val_loss"] - results["memory"]["val_loss"]
        print(f"  Difference:         {diff:+.4f} ({'memory better' if diff > 0 else 'baseline better'})")
    print(f"  Memory contribution (ablation): {memory_contribution:+.4f}")

    # Save
    torch.save(model.state_dict(), Path(output_dir) / "memory_model.pt")
    with open(Path(output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="v2.1: fast weight memory pretraining")
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=20_000_000,
                        help="Token budget. 20M ~ 2-3h on A100 with memory model")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v2_1")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline training (if already have results)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained GPT-2 instead of training from scratch")
    parser.add_argument("--freeze-base", action="store_true",
                        help="Freeze transformer weights, only train memory projections")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    dtype = torch.bfloat16

    # Use GPT-2 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Load data
    train_chunks, val_chunks = load_data(tokenizer, seq_len=args.seq_len, max_tokens=args.max_tokens)
    train_loader = DataLoader(
        ChunkDataset(train_chunks), batch_size=args.batch_size,
        shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        ChunkDataset(val_chunks), batch_size=args.batch_size,
        shuffle=False, drop_last=True,
    )

    results = {
        "config": vars(args),
        "vocab_size": vocab_size,
        "train_chunks": len(train_chunks),
        "val_chunks": len(val_chunks),
    }

    if args.pretrained:
        # ── Pretrained path: bolt memory onto existing GPT-2 ──

        # First: eval pretrained GPT-2 without memory (our baseline)
        print(f"\n{'='*60}")
        print("BASELINE: Pretrained GPT-2 (no memory, no training)")
        print(f"{'='*60}")
        model_base = create_model(vocab_size, device, dtype, pretrained=True)[0]
        val_base = evaluate(model_base, val_loader, device)
        print(f"  Val loss: {val_base:.4f}")
        results["baseline"] = {"val_loss": val_base}
        del model_base
        torch.cuda.empty_cache()

        # Now: pretrained GPT-2 + memory, continue training
        mode = "frozen base + memory only" if args.freeze_base else "full fine-tune + memory"
        print(f"\n{'='*60}")
        print(f"PRETRAINED + MEMORY ({mode})")
        print(f"{'='*60}")

        model_mem, memory_modules = create_model(
            vocab_size, device, dtype,
            with_memory=True, memory_size=args.memory_size, decay=args.decay,
            pretrained=True, freeze_base=args.freeze_base,
        )

        lr = args.lr if not args.freeze_base else args.lr * 3  # higher lr for memory-only
        val_mem, log_mem = train_model(
            model_mem, train_loader, val_loader, device,
            num_epochs=args.epochs, lr=lr,
            warmup_steps=args.warmup_steps,
            log_every=args.log_every, eval_every=args.eval_every,
            model_name="pretrained+memory",
        )
        results["memory"] = {"val_loss": val_mem, "log": log_mem}

        run_ablation_and_summary(model_mem, val_loader, device, results, args.output_dir)

    else:
        # ── From-scratch path (original experiment) ──

        if not args.skip_baseline:
            print(f"\n{'='*60}")
            print("BASELINE: GPT-2 125M (no memory)")
            print(f"{'='*60}")

            model_base = create_model(vocab_size, device, dtype, with_memory=False)[0]
            val_base, log_base = train_model(
                model_base, train_loader, val_loader, device,
                num_epochs=args.epochs, lr=args.lr,
                warmup_steps=args.warmup_steps,
                log_every=args.log_every, eval_every=args.eval_every,
                model_name="baseline",
            )
            results["baseline"] = {"val_loss": val_base, "log": log_base}

            torch.save(model_base.state_dict(), Path(args.output_dir) / "baseline.pt")
            del model_base
            torch.cuda.empty_cache()
        else:
            print("Skipping baseline (--skip-baseline)")

        # Memory model from scratch
        print(f"\n{'='*60}")
        print(f"MEMORY: GPT-2 125M + fast weight memory (size={args.memory_size}, decay={args.decay})")
        print(f"{'='*60}")

        model_mem, memory_modules = create_model(
            vocab_size, device, dtype,
            with_memory=True, memory_size=args.memory_size, decay=args.decay,
        )

        val_mem, log_mem = train_model(
            model_mem, train_loader, val_loader, device,
            num_epochs=args.epochs, lr=args.lr,
            warmup_steps=args.warmup_steps,
            log_every=args.log_every, eval_every=args.eval_every,
            model_name="memory",
        )
        results["memory"] = {"val_loss": val_mem, "log": log_mem}

        run_ablation_and_summary(model_mem, val_loader, device, results, args.output_dir)


if __name__ == "__main__":
    main()
