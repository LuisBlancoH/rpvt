"""v3.28: Train Predictive Transformer (Qwen 0.5B wrapped).

Qwen layers frozen. Only new mechanisms train (201M params):
memory attention, memory FFN, GRU state, prediction heads, write gates,
halt net, value head, reward net, goal state.

Multi-chunk training: process N consecutive chunks without resetting memory.
TD learning: value head + reward network trained via temporal difference.

Usage:
    python -m rpvt.experiments.exp_v3_28_qwen_wrapped
    python -m rpvt.experiments.exp_v3_28_qwen_wrapped --epochs 20
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.predictive_transformer import PredictiveTransformer


class MultiChunkDataset(Dataset):
    """Groups consecutive chunks for multi-chunk training."""

    def __init__(self, token_ids, seq_len=128, n_chunks=4):
        self.seq_len = seq_len
        self.n_chunks = n_chunks
        chunk_len = seq_len + 1
        group_len = chunk_len * n_chunks
        n_groups = len(token_ids) // group_len
        self.data = token_ids[:n_groups * group_len].reshape(
            n_groups, n_chunks, chunk_len
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        group = self.data[idx]
        return group[:, :-1], group[:, 1:]


def evaluate(model, dataloader, device, n_settle=1, max_batches=50):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            model.reset_state()
            group_loss = 0
            n_valid = 0
            for c in range(inputs.shape[1]):
                ids = inputs[:, c, :]
                tgt = targets[:, c, :]
                result = model(ids, labels=tgt, n_settle=n_settle,
                              return_errors=True)
                loss = result[1]
                if loss is not None and not torch.isnan(loss):
                    group_loss += loss.item()
                    n_valid += 1
            if n_valid > 0:
                total_loss += group_loss / n_valid
                n += 1
    avg = total_loss / max(n, 1)
    return math.exp(min(avg, 20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-chunks", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print debug info every N groups")
    parser.add_argument("--output-dir", type=str,
                        default="results/qwen_wrapped_pt")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Qwen
    print("Loading Qwen2.5-0.5B...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, trust_remote_code=True,
    )

    model = PredictiveTransformer(qwen, n_mem_heads=2, state_dim=224)
    model.freeze_base()
    model = model.to(device)

    del qwen
    import gc; gc.collect(); torch.cuda.empty_cache()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.1f}GB")

    # Data
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(t for t in ds["train"]["text"] if t.strip())
    val_text = "\n".join(t for t in ds["validation"]["text"] if t.strip())

    train_tokens = torch.tensor(
        tokenizer.encode(train_text, add_special_tokens=False), dtype=torch.long
    )
    val_tokens = torch.tensor(
        tokenizer.encode(val_text, add_special_tokens=False), dtype=torch.long
    )
    train_ds = MultiChunkDataset(train_tokens, args.seq_len, args.n_chunks)
    val_ds = MultiChunkDataset(val_tokens, args.seq_len, args.n_chunks)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, drop_last=True)
    print(f"  Train: {len(train_ds)} groups ({args.n_chunks} chunks each), "
          f"Val: {len(val_ds)} groups")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Curriculum
    settle_schedule = {0: 1, 3: 2, 6: 3, 8: 5}

    # Initial eval
    print("\n--- Initial (Qwen, no training) ---")
    ppl = evaluate(model, val_loader, device, n_settle=1, max_batches=30)
    print(f"  PPL: {ppl:.1f}")

    prompt = "The capital of France is"
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
    model.reset_state()
    gen = model.generate(prompt_ids["input_ids"], max_new_tokens=30, n_settle=1)
    print(f"  Gen: {prompt}{tokenizer.decode(gen)}")

    # Train
    print(f"\n--- Training ({args.epochs} epochs, {args.n_chunks} chunks/group) ---")
    history = []

    for epoch in range(args.epochs):
        max_settle = 1
        for start, val in settle_schedule.items():
            if epoch >= start:
                max_settle = val
        model.max_settle = max_settle

        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()
        optimizer.zero_grad()

        # Debug accumulators
        debug_values = []
        debug_td = []
        debug_rewards = []
        debug_mem = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            model.reset_state()
            chunk_losses = []

            for c in range(inputs.shape[1]):
                ids = inputs[:, c, :]
                tgt = targets[:, c, :]

                result = model(ids, labels=tgt, n_settle=None,
                              return_errors=True)
                loss = result[1]
                info = result[4]  # value, td_error, intrinsic_reward, memory_used

                if loss is not None and not torch.isnan(loss):
                    (loss / (args.grad_accum * inputs.shape[1])).backward()
                    chunk_losses.append(loss.item())

                    # Collect debug info
                    debug_values.append(info["value"])
                    debug_td.append(info["td_error"])
                    debug_rewards.append(info["intrinsic_reward"])
                    debug_mem.append(info["memory_used"])

                model.detach_state()

            if not chunk_losses:
                continue

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += sum(chunk_losses) / len(chunk_losses)
            n_batches += 1

            # Debug logging
            if (batch_idx + 1) % args.log_every == 0:
                avg_loss = total_loss / n_batches
                avg_val = sum(debug_values[-args.log_every*args.n_chunks:]) / max(len(debug_values[-args.log_every*args.n_chunks:]), 1)
                avg_td = sum(debug_td[-args.log_every*args.n_chunks:]) / max(len(debug_td[-args.log_every*args.n_chunks:]), 1)
                avg_rew = sum(debug_rewards[-args.log_every*args.n_chunks:]) / max(len(debug_rewards[-args.log_every*args.n_chunks:]), 1)
                last_mem = debug_mem[-1] if debug_mem else 0
                elapsed = time.time() - t0
                speed = n_batches / elapsed
                eta = (len(train_loader) - batch_idx) / speed if speed > 0 else 0
                print(f"    [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={avg_loss:.4f} val={avg_val:.3f} "
                      f"td={avg_td:.4f} rew={avg_rew:.4f} "
                      f"mem={last_mem}/{model.memory.n_slots} "
                      f"({speed:.1f} grp/s, ETA {eta:.0f}s)")

        # Flush remaining gradients
        if (batch_idx + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)

        # Eval
        ppl_1 = evaluate(model, val_loader, device, n_settle=1, max_batches=30)
        model.max_settle = max_settle
        ppl_s = evaluate(model, val_loader, device, n_settle=None, max_batches=30)
        gain = ppl_1 - ppl_s

        # Summary stats
        avg_val = sum(debug_values) / max(len(debug_values), 1)
        avg_td = sum(debug_td) / max(len(debug_td), 1)
        avg_rew = sum(debug_rewards) / max(len(debug_rewards), 1)

        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, "
              f"ppl_1={ppl_1:.1f}, ppl_settle={ppl_s:.1f}, "
              f"gain={gain:+.1f}, max_settle={max_settle}, "
              f"val={avg_val:.3f}, td={avg_td:.4f}, rew={avg_rew:.4f}, "
              f"({elapsed:.0f}s)")

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "ppl_1": ppl_1,
            "ppl_settle": ppl_s,
            "gain": gain,
            "max_settle": max_settle,
            "avg_value": avg_val,
            "avg_td_error": avg_td,
            "avg_intrinsic_reward": avg_rew,
        })

        # Save checkpoint (include new components)
        ckpt = output_dir / f"checkpoint_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": {
                k: v.cpu() for k, v in model.state_dict().items()
                if any(x in k for x in ["mem_", "state_", "predictor",
                       "write_gate", "halt_", "ln_mem", "memory",
                       "value_", "reward_", "goal_", "strength_"])
            },
            "optimizer": optimizer.state_dict(),
            "history": history,
        }, ckpt)

    # Final
    print(f"\n--- Final ---")
    for ns in [1, 2, 3]:
        model.max_settle = ns
        ppl = evaluate(model, val_loader, device, n_settle=None, max_batches=50)
        print(f"  max_settle={ns}: PPL={ppl:.1f}")

    print(f"\n--- Generation ---")
    for ns in [1, 2]:
        model.reset_state()
        model.max_settle = ns
        gen = model.generate(prompt_ids["input_ids"], max_new_tokens=40,
                            n_settle=None)
        print(f"  settle={ns}: {prompt}{tokenizer.decode(gen)}")

    mem = model.memory
    print(f"\n  Memory: {mem.n_stored}/{mem.n_slots} slots")

    with open(output_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
