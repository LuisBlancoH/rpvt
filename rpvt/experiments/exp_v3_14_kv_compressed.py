"""Experiment v3.14: Compressed KV Memory — hippocampal compression + reconstruction.

Like the brain: compress the full KV cache into a sparse representation,
store it, then reconstruct at recall time. The model sees reconstructed
KV pairs — native format, but compressed storage.

    128 tokens → KV cache → compress to 8 vectors → store
    At recall: 8 vectors → decompress to 16 KV positions → model attends

Compression: 128 tokens → 8 vectors (~16× compression)
Decompression: 8 vectors → 16 KV positions (model attends natively)
Training: reconstruction loss + answer loss (end-to-end)

This gives BOTH compression AND compatibility.
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

from rpvt.model.kv_compressor import KVMemorySystem
from rpvt.experiments.exp_v3_2_nlp_recall import _generate_natural_facts


def _make_qa_chunk(tokenizer, question, answer, chunk_size):
    messages = [{"role": "user", "content": question}]
    chat_prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = chat_prefix + answer + "<|im_end|>"
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prefix_len = len(tokenizer.encode(chat_prefix, add_special_tokens=False))

    answer_mask = torch.zeros(chunk_size, dtype=torch.float32)
    for pos in range(max(0, prefix_len - 1), min(len(full_tokens) - 1, chunk_size)):
        answer_mask[pos] = 1.0

    if len(full_tokens) >= chunk_size:
        full_tokens = full_tokens[:chunk_size]
    else:
        full_tokens = full_tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(full_tokens))

    return torch.tensor(full_tokens, dtype=torch.long), answer_mask


def build_dataset(tokenizer, n_memory=500, chunk_size=128,
                  gap_range=(2, 6), max_qa_pairs=3, seed=42):
    rng = random.Random(seed)
    print(f"  Generating {n_memory} memory docs...")
    recall_docs = _generate_natural_facts(rng, n_memory, max_qa_pairs)

    print("  Loading WikiText for filler...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    filler_texts = [t for t in wiki["text"] if len(t.strip()) > 100]

    docs = []
    for passage, qa_pairs in recall_docs:
        gap = rng.randint(gap_range[0], gap_range[1])
        passage_tokens = tokenizer.encode(passage, add_special_tokens=False)
        passage_chunks = []
        for i in range(0, len(passage_tokens), chunk_size):
            ct = passage_tokens[i:i + chunk_size]
            if len(ct) < chunk_size:
                ct = ct + [tokenizer.eos_token_id or 0] * (chunk_size - len(ct))
            passage_chunks.append(torch.tensor(ct, dtype=torch.long))

        filler_chunks = []
        for _ in range(gap):
            ft = rng.choice(filler_texts)
            ft_tok = tokenizer.encode(ft, add_special_tokens=False)
            if len(ft_tok) >= chunk_size:
                start = rng.randint(0, len(ft_tok) - chunk_size)
                ct = ft_tok[start:start + chunk_size]
            else:
                ct = ft_tok + [tokenizer.eos_token_id or 0] * (chunk_size - len(ft_tok))
            filler_chunks.append(torch.tensor(ct, dtype=torch.long))

        for qa in qa_pairs:
            qa_chunk, answer_mask = _make_qa_chunk(
                tokenizer, qa["question"], qa["answer"], chunk_size
            )
            chunk_types = (["passage"] * len(passage_chunks) +
                          ["filler"] * len(filler_chunks) + ["qa"])
            docs.append({
                "chunks": passage_chunks + filler_chunks + [qa_chunk],
                "chunk_types": chunk_types,
                "answer_mask": answer_mask,
            })

    rng.shuffle(docs)
    print(f"  Dataset: {len(docs)} docs")
    return docs


def train(model, kv_system, tokenizer, train_docs, eval_docs, device,
          num_epochs=15, lr=1e-3, log_every=50, checkpoint_dir=None):
    """Train compressor + decompressor end-to-end."""

    trainable = list(kv_system.parameters())
    n_params = sum(p.numel() for p in trainable)
    print(f"\nTraining {n_params:,} parameters (compressor + decompressor)")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = len(train_docs) * num_epochs

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"  {total_steps} steps, {num_epochs} epochs")
    model.eval()
    kv_system.train()
    global_step = 0
    losses = []
    start_time = time.time()

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        order = list(range(len(train_docs)))
        random.shuffle(order)

        for doc_idx in order:
            doc = train_docs[doc_idx]
            chunks = doc["chunks"]
            chunk_types = doc["chunk_types"]
            answer_mask = doc["answer_mask"].to(device)

            kv_system.reset()

            # Process context chunks
            with torch.no_grad():
                for chunk_idx, chunk in enumerate(chunks[:-1]):
                    chunk_ids = chunk.unsqueeze(0).to(device)
                    output = model(chunk_ids, use_cache=True)
                    if chunk_types[chunk_idx] == "passage":
                        kv_system.store(output.past_key_values,
                                        seq_len=chunk_ids.shape[1])
                    else:
                        kv_system.skip(chunk_ids.shape[1])

            # Reconstruct KV cache from compressed memory
            reconstructed = kv_system.reconstruct(device, torch.bfloat16)

            if reconstructed is not None:
                qa_chunk = chunks[-1].unsqueeze(0).to(device)
                n_decomp = kv_system.get_n_decompressed()
                n_total_seen = kv_system.total_tokens_seen.item()
                seq_len = qa_chunk.shape[1]

                position_ids = torch.arange(
                    n_total_seen, n_total_seen + seq_len, device=device
                ).unsqueeze(0)
                attn_mask = torch.ones(
                    1, n_decomp + seq_len, device=device, dtype=torch.long
                )

                output = model(
                    qa_chunk,
                    past_key_values=reconstructed,
                    position_ids=position_ids,
                    attention_mask=attn_mask,
                )
            else:
                qa_chunk = chunks[-1].unsqueeze(0).to(device)
                output = model(qa_chunk)

            # Answer loss
            logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
            targets = qa_chunk[:, 1:].reshape(-1)
            per_token = F.cross_entropy(logits, targets, reduction='none')
            mask = answer_mask[:-1]
            n_tokens = mask.sum().clamp(min=1)
            loss = (per_token * mask).sum() / n_tokens

            if loss.item() > 0 and not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()

            losses.append(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = sum(losses[-log_every:]) / min(len(losses), log_every)
                elapsed = time.time() - start_time
                eta = elapsed / global_step * (total_steps - global_step)
                print(f"  step {global_step}/{total_steps}, "
                      f"loss={avg_loss:.3f}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, "
                      f"{elapsed:.0f}s (ETA {eta/3600:.1f}h)")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        eval_results = evaluate(model, kv_system, tokenizer, eval_docs, device)
        print(f"  Memory recall: {eval_results['token_acc']:.1%} "
              f"({eval_results['correct']}/{eval_results['total']})")

        if checkpoint_dir:
            state = {
                "kv_system": {n: p.data.clone() for n, p in kv_system.named_parameters()},
                "epoch": epoch, "global_step": global_step,
            }
            torch.save(state, os.path.join(checkpoint_dir, "latest.pt"))

        kv_system.train()

    return eval_results


def evaluate(model, kv_system, tokenizer, eval_docs, device):
    model.eval()
    kv_system.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            chunk_types = doc["chunk_types"]
            answer_mask = doc["answer_mask"].to(device)

            kv_system.reset()

            for chunk_idx, chunk in enumerate(chunks[:-1]):
                chunk_ids = chunk.unsqueeze(0).to(device)
                output = model(chunk_ids, use_cache=True)
                if chunk_types[chunk_idx] == "passage":
                    kv_system.store(output.past_key_values, chunk_ids.shape[1])
                else:
                    kv_system.skip(chunk_ids.shape[1])

            reconstructed = kv_system.reconstruct(device, torch.bfloat16)

            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            if reconstructed is not None:
                n_decomp = kv_system.get_n_decompressed()
                n_total_seen = kv_system.total_tokens_seen.item()
                seq_len = qa_chunk.shape[1]
                position_ids = torch.arange(
                    n_total_seen, n_total_seen + seq_len, device=device
                ).unsqueeze(0)
                attn_mask = torch.ones(
                    1, n_decomp + seq_len, device=device, dtype=torch.long
                )
                output = model(qa_chunk, past_key_values=reconstructed,
                               position_ids=position_ids, attention_mask=attn_mask)
            else:
                output = model(qa_chunk)

            predictions = output.logits[0, :-1].argmax(dim=-1)
            targets = qa_chunk[0, 1:]
            mask = answer_mask[:-1]
            positions = mask.nonzero(as_tuple=True)[0]
            for p in positions:
                total += 1
                if predictions[p].item() == targets[p].item():
                    correct += 1

    return {"token_acc": correct / max(total, 1), "correct": correct, "total": total}


def test_generation(model, kv_system, tokenizer, device):
    model.eval()
    kv_system.eval()
    filler = "Modern computing has revolutionized information processing."

    def process_chunk(text, chunk_size=128, store=True):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:chunk_size]
        if len(tokens) < chunk_size:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(tokens))
        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            output = model(input_ids, use_cache=True)
            if store:
                kv_system.store(output.past_key_values, len(tokens))
            else:
                kv_system.skip(len(tokens))

    def generate_with_memory(question, max_new=100):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            reconstructed = kv_system.reconstruct(device, torch.bfloat16)
            if reconstructed is not None:
                n_decomp = kv_system.get_n_decompressed()
                n_total = kv_system.total_tokens_seen.item()
                seq_len = input_ids.shape[1]
                position_ids = torch.arange(
                    n_total, n_total + seq_len, device=device
                ).unsqueeze(0)
                attn_mask = torch.ones(
                    1, n_decomp + seq_len, device=device, dtype=torch.long
                )
                out = model.generate(
                    input_ids, past_key_values=reconstructed,
                    position_ids=position_ids, attention_mask=attn_mask,
                    max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                out = model.generate(
                    input_ids, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        return tokenizer.decode(out[0][len(tokens):], skip_special_tokens=True)

    print("\n=== GENERATION TESTS ===\n")

    print("1. Basic instruct (no memory):")
    kv_system.reset()
    for q in ["What is the capital of France?", "Write a haiku about programming."]:
        resp = generate_with_memory(q)
        print(f"  Q: {q}")
        print(f"  A: {resp[:200]}")
        print()

    print("2. Memory recall + generation:")
    tests = [
        ("Classified briefing: The operation code is VIPER-371. Agent Blackwood is stationed at Nordheim.",
         "What is the operation code from the briefing?"),
        ("The secret password is THUNDERBOLT. Remember this password.",
         "What is the secret password?"),
    ]
    for passage, q in tests:
        kv_system.reset()
        process_chunk(passage, store=True)
        for _ in range(3):
            process_chunk(filler, store=False)
        resp = generate_with_memory(q)
        print(f"  Stored: \"{passage[:60]}...\"")
        print(f"  Q: {q}")
        print(f"  A: {resp[:200]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="v3.14: Compressed KV Memory")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-memory", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--n-compressed", type=int, default=8)
    parser.add_argument("--n-decompressed", type=int, default=16)
    parser.add_argument("--compress-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_14_compressed")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"\nLoading: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device)

    for param in model.parameters():
        param.requires_grad = False

    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    kv_system = KVMemorySystem(
        n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
        n_slots=64, n_compressed=args.n_compressed,
        n_decompressed=args.n_decompressed,
        compress_dim=args.compress_dim,
    ).to(device=args.device, dtype=torch.bfloat16)

    n_params = sum(p.numel() for p in kv_system.parameters())
    storage_per_slot = args.n_compressed * args.compress_dim * 2  # bytes in bf16
    print(f"  Compressor+Decompressor: {n_params:,} params")
    print(f"  Storage per chunk: {storage_per_slot / 1024:.1f} KB "
          f"(vs 3.5 MB raw KV = {3.5 * 1024 * 1024 / storage_per_slot:.0f}× compression)")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    print("\nBuilding training data...")
    train_docs = build_dataset(tokenizer, n_memory=args.n_memory, seed=args.seed)
    print("Building eval data...")
    eval_docs = build_dataset(tokenizer, n_memory=args.n_eval, seed=args.seed + 1000)

    results = train(
        model, kv_system, tokenizer, train_docs, eval_docs, args.device,
        num_epochs=args.epochs, lr=args.lr, log_every=args.log_every,
        checkpoint_dir=checkpoint_dir,
    )

    test_generation(model, kv_system, tokenizer, args.device)

    results["config"] = vars(args)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
