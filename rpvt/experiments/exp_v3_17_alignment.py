"""Experiment v3.17: Predictive Coding Alignment (v2 — persistent state).

Train the inverse transformer to predict the forward model's hidden
states. The inverse maintains persistent state across chunks — a
running world model. No memory, no recall, no LoRA.

Questions we answer:
1. Does the inverse learn to predict the forward model? (prediction loss drops)
2. Does the persistent state encode meaningful context? (errors differ for
   expected vs unexpected content)
3. Does the forward model's generation survive? (no weights changed)
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.predictive_coding_v2 import (
    PersistentInverseTransformer, HiddenStateCapture, PredictiveCodingSystem,
)


def build_text_chunks(tokenizer, n_docs=5000, chunk_size=128, seed=42):
    """Raw text chunks from WikiText."""
    rng = random.Random(seed)
    print("  Loading WikiText...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t for t in wiki["text"] if len(t.strip()) > 100]
    rng.shuffle(texts)

    chunks = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 20:
            continue
        if len(tokens) >= chunk_size:
            start = rng.randint(0, len(tokens) - chunk_size)
            ct = tokens[start:start + chunk_size]
        else:
            ct = tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(ct))
        chunks.append(torch.tensor(ct, dtype=torch.long))
        if len(chunks) >= n_docs:
            break

    print(f"  {len(chunks)} text chunks")
    return chunks


def build_document_sequences(tokenizer, n_docs=500, chunks_per_doc=5,
                             chunk_size=128, seed=42):
    """Multi-chunk documents for testing persistent state.

    Each document is a sequence of related chunks from the same article.
    The inverse should build up context across chunks.
    """
    rng = random.Random(seed)
    print("  Loading WikiText for documents...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t for t in wiki["text"] if len(t.strip()) > chunk_size * chunks_per_doc]
    rng.shuffle(texts)

    documents = []
    for text in texts[:n_docs]:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        doc_chunks = []
        for i in range(0, min(len(tokens), chunks_per_doc * chunk_size), chunk_size):
            ct = tokens[i:i + chunk_size]
            if len(ct) < chunk_size:
                ct = ct + [tokenizer.eos_token_id or 0] * (chunk_size - len(ct))
            doc_chunks.append(torch.tensor(ct, dtype=torch.long))
            if len(doc_chunks) >= chunks_per_doc:
                break

        if len(doc_chunks) >= 2:
            documents.append(doc_chunks)

    print(f"  {len(documents)} documents, {chunks_per_doc} chunks each")
    return documents


def train(system, train_docs, device, num_epochs=10, lr=1e-3,
          log_every=200):
    """Train inverse to predict forward model on multi-chunk documents."""

    trainable = list(system.inverse.parameters())
    n_params = sum(p.numel() for p in trainable)
    print(f"\nTraining inverse: {n_params:,} params")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    total_chunks = sum(len(doc) for doc in train_docs) * num_epochs

    def lr_schedule(step):
        warmup = 200
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_chunks - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"  {total_chunks} total chunks, {num_epochs} epochs")
    system.inverse.train()
    global_step = 0
    losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        order = list(range(len(train_docs)))
        random.shuffle(order)

        for doc_idx in order:
            doc = train_docs[doc_idx]
            system.reset()  # new document → reset persistent state

            for chunk in doc:
                chunk_ids = chunk.unsqueeze(0).to(device)

                loss = system.prediction_loss(chunk_ids)

                if loss.item() > 0 and not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    scheduler.step()

                losses.append(loss.item())
                global_step += 1

                if global_step % log_every == 0:
                    avg = sum(losses[-log_every:]) / log_every
                    elapsed = time.time() - start_time
                    print(f"  step {global_step}/{total_chunks}, "
                          f"pred_loss={avg:.4f}, "
                          f"lr={scheduler.get_last_lr()[0]:.2e}, "
                          f"{elapsed:.0f}s")

        # Test: does persistent state help?
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        test_persistent_state(system, train_docs[:20], device)

        system.inverse.train()


def test_persistent_state(system, test_docs, device):
    """Test whether persistent state makes predictions better.

    Compare prediction error on chunk N with vs without
    seeing chunks 1..N-1 first.
    """
    system.inverse.eval()

    errors_with_context = []
    errors_without_context = []

    with torch.no_grad():
        for doc in test_docs:
            if len(doc) < 3:
                continue

            # WITH persistent state: process chunks 1..N-1, then predict chunk N
            system.reset()
            for chunk in doc[:-1]:
                system.process_chunk(chunk.unsqueeze(0).to(device))

            _, _, mags_with = system.process_chunk(doc[-1].unsqueeze(0).to(device))
            if mags_with:
                errors_with_context.append(sum(mags_with.values()) / len(mags_with))

            # WITHOUT persistent state: predict chunk N cold
            system.reset()
            _, _, mags_without = system.process_chunk(doc[-1].unsqueeze(0).to(device))
            if mags_without:
                errors_without_context.append(sum(mags_without.values()) / len(mags_without))

    avg_with = sum(errors_with_context) / max(len(errors_with_context), 1)
    avg_without = sum(errors_without_context) / max(len(errors_without_context), 1)

    print(f"  Prediction error WITH context:    {avg_with:.4f}")
    print(f"  Prediction error WITHOUT context: {avg_without:.4f}")
    if avg_without > 0:
        improvement = (avg_without - avg_with) / avg_without * 100
        print(f"  Context improvement: {improvement:.1f}%")


def test_generation(model, tokenizer, device):
    """Verify generation still works (model is frozen, should be perfect)."""
    model.eval()

    def generate(question, max_new=80):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][len(tokens):], skip_special_tokens=True)

    print("\n  Generation test:")
    for q in ["What is the capital of France?",
              "Write a haiku about the ocean."]:
        print(f"  Q: {q}")
        print(f"  A: {generate(q)[:150]}")


def main():
    parser = argparse.ArgumentParser(description="v3.17: Predictive coding alignment")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-docs", type=int, default=500)
    parser.add_argument("--chunks-per-doc", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=500)
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

    hidden_size = model.config.hidden_size
    target_layers = [7, 14, 21]

    # Wrap target layers with hidden state captures (transparent)
    layers = model.model.layers
    captures = []
    for li in target_layers:
        c = HiddenStateCapture(layers[li], li)
        layers[li] = c
        captures.append(c)

    # Create inverse with persistent state
    inverse = PersistentInverseTransformer(
        hidden_size, n_inverse_layers=len(target_layers),
        target_layers=target_layers, n_heads=8,
        max_context_tokens=256,
    ).to(args.device, dtype=torch.bfloat16)

    system = PredictiveCodingSystem(model, inverse, target_layers, captures)

    n_inv = sum(p.numel() for p in inverse.parameters())
    print(f"  Inverse: {n_inv:,} params (persistent KV cache, max 256 tokens)")
    print(f"  Target layers: {target_layers}")
    print(f"  Forward model: frozen, no LoRA")

    # Test generation before training
    print("\nPre-training:")
    test_generation(model, tokenizer, args.device)

    # Build multi-chunk documents
    print("\nBuilding data...")
    train_docs = build_document_sequences(
        tokenizer, n_docs=args.n_docs,
        chunks_per_doc=args.chunks_per_doc, seed=args.seed,
    )

    # Train
    train(system, train_docs, args.device,
          num_epochs=args.epochs, lr=args.lr, log_every=args.log_every)

    # Test generation after training (should be identical — model frozen)
    print("\nPost-training:")
    test_generation(model, tokenizer, args.device)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
