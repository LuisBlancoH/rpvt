"""v3.26: Train the Predictive Transformer from scratch.

Train on WikiText-2 with next-token prediction + prediction error loss.
Evaluate whether settling (multi-pass) reduces perplexity compared to
single-pass, and whether memory/state improve over time.

Key metrics:
- Perplexity (language modeling quality)
- Prediction error per layer (are top-down predictions improving?)
- Settling gain: does n_settle=2 beat n_settle=1?
- Memory usage: does the model learn to read/write memory?

Usage:
    python -m rpvt.experiments.exp_v3_26_train_predictive
    python -m rpvt.experiments.exp_v3_26_train_predictive --epochs 20 --hidden 512
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from rpvt.model.predictive_transformer import PredictiveTransformer


class TextDataset(Dataset):
    """Simple dataset: chunks of tokenized text."""

    def __init__(self, token_ids, seq_len=128):
        self.seq_len = seq_len
        # Reshape into chunks
        n_chunks = len(token_ids) // (seq_len + 1)
        self.data = token_ids[:n_chunks * (seq_len + 1)].reshape(n_chunks, seq_len + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]  # input, target


def load_wikitext(tokenizer, seq_len=128):
    """Load and tokenize WikiText-2."""
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_split(split):
        texts = [t for t in ds[split]["text"] if len(t.strip()) > 0]
        full_text = "\n".join(texts)
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        return torch.tensor(tokens, dtype=torch.long)

    train_tokens = tokenize_split("train")
    val_tokens = tokenize_split("validation")

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)

    print(f"  Train: {len(train_ds)} chunks ({len(train_tokens):,} tokens)")
    print(f"  Val:   {len(val_ds)} chunks ({len(val_tokens):,} tokens)")

    return train_ds, val_ds


def evaluate(model, dataloader, device, n_settle=1, max_batches=None):
    """Evaluate perplexity and prediction errors."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_errors = [0.0] * model.n_layers
    n_error_samples = 0

    with torch.no_grad():
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            model.reset_state()
            result = model(
                input_ids, labels=targets,
                n_settle=n_settle, return_errors=True,
            )
            logits, loss, errors = result[0], result[1], result[2]

            total_loss += loss.item() * input_ids.shape[0]
            total_tokens += input_ids.shape[0]

            # Track errors from last settling step
            last_errors = errors[-1]
            for i, e in enumerate(last_errors):
                total_errors[i] += e
            n_error_samples += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
    avg_errors = [e / max(n_error_samples, 1) for e in total_errors]

    return perplexity, avg_errors


def main():
    parser = argparse.ArgumentParser(description="v3.26: Train Predictive Transformer")
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-self-heads", type=int, default=8)
    parser.add_argument("--n-mem-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--n-memory-slots", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--output-dir", type=str,
                        default="results/predictive_transformer")
    parser.add_argument("--save-every", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer (use Qwen's tokenizer for vocab)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B", trust_remote_code=True
    )
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Data
    train_ds, val_ds = load_wikitext(tokenizer, seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, drop_last=True)

    # Model
    model = PredictiveTransformer(
        vocab_size=vocab_size,
        hidden_size=args.hidden,
        n_layers=args.n_layers,
        n_self_heads=args.n_self_heads,
        n_mem_heads=args.n_mem_heads,
        state_dim=args.state_dim,
        n_memory_slots=args.n_memory_slots,
        max_seq_len=args.seq_len,
    ).to(device)

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=0.01, betas=(0.9, 0.95),
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Initial eval
    print("\n--- Initial evaluation ---")
    ppl_1, errors_1 = evaluate(model, val_loader, device, n_settle=1, max_batches=20)
    ppl_2, errors_2 = evaluate(model, val_loader, device, n_settle=2, max_batches=20)
    print(f"  1-pass perplexity: {ppl_1:.1f}")
    print(f"  2-pass perplexity: {ppl_2:.1f}")
    print(f"  Errors (1-pass): {['%.2f' % e for e in errors_1]}")

    # Training
    print(f"\n--- Training ({args.epochs} epochs) ---")
    global_step = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_pred_err = 0
        n_batches = 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            model.reset_state()

            # Alternate between 1-pass and 2-pass training
            # (so the model learns to use settling)
            n_settle = 1 if batch_idx % 3 != 0 else 2

            result = model(
                input_ids, labels=targets,
                n_settle=n_settle, return_errors=True,
            )
            logits, loss, errors = result[0], result[1], result[2]

            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * args.grad_accum
            epoch_pred_err += sum(errors[-1]) / len(errors[-1])
            n_batches += 1

        # Flush remaining gradients
        if (batch_idx + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_pred_err = epoch_pred_err / max(n_batches, 1)

        # Evaluate
        ppl_1, err_1 = evaluate(model, val_loader, device, n_settle=1, max_batches=20)
        ppl_2, err_2 = evaluate(model, val_loader, device, n_settle=2, max_batches=20)

        settling_gain = ppl_1 - ppl_2  # positive = settling helps

        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, "
              f"pred_err={avg_pred_err:.2f}, "
              f"ppl_1={ppl_1:.1f}, ppl_2={ppl_2:.1f}, "
              f"settle_gain={settling_gain:+.1f}, "
              f"lr={scheduler.get_last_lr()[0]:.6f}, ({elapsed:.0f}s)")

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "pred_error": avg_pred_err,
            "perplexity_1pass": ppl_1,
            "perplexity_2pass": ppl_2,
            "settling_gain": settling_gain,
            "errors_1pass": err_1,
            "errors_2pass": err_2,
        })

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "history": history,
            }, ckpt_path)
            print(f"    Saved {ckpt_path}")

    # Final eval with more settling steps
    print(f"\n--- Final evaluation ---")
    for n_s in [1, 2, 3, 5]:
        ppl, errs = evaluate(model, val_loader, device, n_settle=n_s, max_batches=50)
        print(f"  n_settle={n_s}: perplexity={ppl:.1f}, "
              f"avg_error={sum(errs)/len(errs):.2f}")

    # Test generation
    print(f"\n--- Generation samples ---")
    prompt = "The quick brown fox"
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)

    for n_s in [1, 2]:
        model.reset_state()
        gen = model.generate(
            prompt_ids["input_ids"], max_new_tokens=50, n_settle=n_s,
        )
        text = tokenizer.decode(gen, skip_special_tokens=True)
        print(f"  n_settle={n_s}: {prompt}{text[:80]}")

    # Memory usage stats
    mem = model.memory
    print(f"\n--- Memory stats ---")
    print(f"  Slots used: {(mem.strength > 1e-8).sum().item()}/{mem.n_slots}")
    print(f"  Write pointer: {mem.write_ptr}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "history": history,
            "final_perplexities": {
                str(n_s): ppl for n_s in [1, 2, 3, 5]
                for ppl, _ in [evaluate(model, val_loader, device, n_s, 50)]
            } if False else {},
        }, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
