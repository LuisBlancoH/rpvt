"""Experiment v3.17: Predictive coding alignment (no memory).

Train ONLY the predictive coding system — no memory, no recall task.

Goal: align the forward model (instruct + LoRA) with the inverse
transformer so that prediction errors become meaningful signals.

Two losses only:
  1. Prediction loss: inverse predicts forward hidden states
  2. Consistency loss: LoRA makes forward predictable by inverse

No answer loss. No KV cache. No memory. Just alignment.

After alignment:
  - The inverse can predict what the forward model does
  - The error signals are meaningful (not noise)
  - Generation should survive (consistency is gentle, no recall pressure)
  - THEN add memory on top of the aligned system
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from rpvt.model.dual_network import (
    InverseTransformer, DualModulationWrapper, DualNetworkSystem,
)


def build_text_dataset(tokenizer, n_docs=5000, chunk_size=128, seed=42):
    """Just raw text chunks. No QA, no memory, no filler distinction."""
    rng = random.Random(seed)

    print(f"  Loading WikiText...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t for t in wiki["text"] if len(t.strip()) > 100]
    rng.shuffle(texts)

    docs = []
    for text in texts[:n_docs * 2]:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 20:
            continue

        if len(tokens) >= chunk_size:
            start = rng.randint(0, len(tokens) - chunk_size)
            ct = tokens[start:start + chunk_size]
        else:
            ct = tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(tokens))

        docs.append(torch.tensor(ct, dtype=torch.long))

        if len(docs) >= n_docs:
            break

    print(f"  Dataset: {len(docs)} text chunks")
    return docs


def train(system, tokenizer, train_chunks, eval_chunks, device,
          num_epochs=10, lr_inverse=1e-3, lr_lora=1e-5,
          alpha_consist=1.0, log_every=100, checkpoint_dir=None):
    """Align forward and inverse — no task loss, just prediction + consistency."""

    inverse_params = list(system.inverse.parameters())
    lora_params = [p for n, p in system.forward_model.named_parameters()
                   if p.requires_grad and "lora" in n.lower()]
    scale_params = [w.scale for w in system.wrappers.values()]

    optimizer = torch.optim.AdamW([
        {"params": inverse_params, "lr": lr_inverse},
        {"params": lora_params, "lr": lr_lora},
        {"params": scale_params, "lr": lr_inverse},
    ], weight_decay=0.01)

    total_steps = len(train_chunks) * num_epochs
    print(f"\nAlignment training:")
    print(f"  Inverse: {sum(p.numel() for p in inverse_params):,} params (lr={lr_inverse})")
    print(f"  LoRA: {sum(p.numel() for p in lora_params):,} params (lr={lr_lora})")
    print(f"  {total_steps} steps, {num_epochs} epochs")
    print(f"  Losses: prediction + {alpha_consist}× consistency (NO answer loss)")

    def lr_schedule(step):
        warmup = 200
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    system.forward_model.train()
    system.inverse.train()
    global_step = 0
    losses_pred = []
    losses_consist = []
    start_time = time.time()

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        order = list(range(len(train_chunks)))
        random.shuffle(order)

        for idx in order:
            chunk_ids = train_chunks[idx].unsqueeze(0).to(device)

            # Forward pass — no modulation during alignment
            for w in system.wrappers.values():
                w.disable_modulation()

            output = system.forward_model(chunk_ids)
            hidden_dict = system.get_captured_hidden_states()

            # Prediction loss: inverse learns to predict forward
            pred_loss = system.prediction_loss(hidden_dict)

            # Consistency loss: forward learns to be predictable
            consist_loss = system.consistency_loss(hidden_dict)

            total_loss = pred_loss + alpha_consist * consist_loss

            if not torch.isnan(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    inverse_params + lora_params + scale_params, 1.0
                )
                optimizer.step()
                scheduler.step()

            losses_pred.append(pred_loss.item())
            losses_consist.append(consist_loss.item())
            global_step += 1

            if global_step % log_every == 0:
                p_avg = sum(losses_pred[-log_every:]) / log_every
                c_avg = sum(losses_consist[-log_every:]) / log_every
                elapsed = time.time() - start_time
                eta = elapsed / global_step * (total_steps - global_step)
                print(f"  step {global_step}/{total_steps}, "
                      f"pred={p_avg:.4f}, consist={c_avg:.4f}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, "
                      f"{elapsed:.0f}s (ETA {eta/3600:.1f}h)")

        # Eval: check generation quality
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        test_generation(system, tokenizer, device)

        if checkpoint_dir:
            state = {
                "inverse": {n: p.data.clone() for n, p in system.inverse.named_parameters()},
                "lora": {n: p.data.clone() for n, p in system.forward_model.named_parameters()
                         if p.requires_grad},
                "scales": {li: w.scale.data.clone() for li, w in system.wrappers.items()},
                "epoch": epoch, "step": global_step,
            }
            torch.save(state, os.path.join(checkpoint_dir, "latest.pt"))

        system.inverse.train()
        system.forward_model.train()


def test_generation(system, tokenizer, device):
    """Test that generation still works after alignment."""
    system.forward_model.eval()

    def generate(question, max_new=80):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            out = system.forward_model.generate(
                input_ids, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][len(tokens):], skip_special_tokens=True)

    questions = [
        "What is the capital of France?",
        "Write a haiku about the ocean.",
        "Explain photosynthesis in one sentence.",
    ]

    for q in questions:
        resp = generate(q)
        print(f"  Q: {q}")
        print(f"  A: {resp[:150]}")


def main():
    parser = argparse.ArgumentParser(description="v3.17: Predictive coding alignment")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--lr-inverse", type=float, default=1e-3)
    parser.add_argument("--lr-lora", type=float, default=1e-5)
    parser.add_argument("--alpha-consist", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_17_alignment")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"\nLoading: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device)

    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
        lora_alpha=args.lora_rank * 2, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    hidden_size = model.config.hidden_size
    target_layers = [7, 14, 21]

    base = model.base_model.model
    layers = base.model.layers if hasattr(base, 'model') else base.layers

    wrappers = []
    for li in target_layers:
        w = DualModulationWrapper(layers[li], li).to(args.device, dtype=torch.bfloat16)
        layers[li] = w
        wrappers.append(w)

    inverse = InverseTransformer(
        hidden_size, n_layers=len(target_layers),
        target_layers=target_layers, n_heads=8,
    ).to(args.device, dtype=torch.bfloat16)

    system = DualNetworkSystem(model, inverse, target_layers, wrappers)

    print(f"  Target layers: {target_layers}")
    print(f"  Inverse: {sum(p.numel() for p in inverse.parameters()):,} params")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    print("\nBuilding data...")
    train_chunks = build_text_dataset(tokenizer, n_docs=args.n_train, seed=args.seed)
    eval_chunks = build_text_dataset(tokenizer, n_docs=args.n_eval, seed=args.seed + 1000)

    print("\nPre-alignment generation test:")
    test_generation(system, tokenizer, args.device)

    train(
        system, tokenizer, train_chunks, eval_chunks, args.device,
        num_epochs=args.epochs, lr_inverse=args.lr_inverse,
        lr_lora=args.lr_lora, alpha_consist=args.alpha_consist,
        log_every=args.log_every, checkpoint_dir=checkpoint_dir,
    )

    print("\nPost-alignment generation test:")
    test_generation(system, tokenizer, args.device)

    results = {"config": vars(args)}
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
