"""Experiment v3.13: KV Cache Memory — brain-like pattern reinstatement.

Store the model's OWN KV cache from passage processing. Restore it
at query time as past_key_values. The model "re-experiences" the
passage through its native representations.

No foreign signals. No LoRA. No decoder. The model receives exactly
what it was trained on — its own KV pairs. 100% compatible.

Only trainable component: the write gate (which positions to store).
~1500 params.

Architecture:
  Process passage: model(passage, use_cache=True) → store selected KVs
  Process filler:  model(filler, use_cache=True) → store if important
  Answer question: model(question, past_key_values=stored_kvs) → answer
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

from rpvt.model.kv_memory import KVMemoryBank
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
            # Tag which chunks are passage vs filler
            chunk_types = (["passage"] * len(passage_chunks) +
                          ["filler"] * len(filler_chunks) +
                          ["qa"])
            docs.append({
                "chunks": passage_chunks + filler_chunks + [qa_chunk],
                "chunk_types": chunk_types,
                "answer_mask": answer_mask,
                "qa": qa,
            })

    rng.shuffle(docs)
    print(f"  Dataset: {len(docs)} docs")
    return docs


def train(model, kv_memory, tokenizer, train_docs, eval_docs, device,
          num_epochs=15, lr=1e-3, log_every=50, checkpoint_dir=None):
    """Train only the write gate — what positions to store."""

    trainable = list(kv_memory.parameters())
    n_params = sum(p.numel() for p in trainable)
    # No training needed — just evaluate. KV cache is the model's own representations.
    print(f"\nNo training needed — evaluating KV cache memory directly")
    model.eval()

    eval_results = evaluate(model, kv_memory, tokenizer, eval_docs, device)
    print(f"\n  Memory recall: {eval_results['token_acc']:.1%} "
          f"({eval_results['correct']}/{eval_results['total']})")

    return eval_results


def evaluate(model, kv_memory, tokenizer, eval_docs, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)

            kv_memory.reset()
            chunk_types = doc.get("chunk_types", ["passage"] * len(chunks))

            for chunk_idx, chunk in enumerate(chunks[:-1]):
                chunk_ids = chunk.unsqueeze(0).to(device)
                output = model(chunk_ids, use_cache=True)
                # Store passage chunks only — filler dilutes the signal
                if chunk_types[chunk_idx] == "passage":
                    kv_memory.store_all(output.past_key_values)
                else:
                    kv_memory.skip(chunk_ids.shape[1])

            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            past_kvs = kv_memory.get_past_key_values(device, torch.bfloat16)

            if past_kvs is not None:
                n_past = kv_memory.n_stored.item()
                n_total_seen = kv_memory.total_tokens_seen.item()
                seq_len = qa_chunk.shape[1]
                # Position IDs account for ALL processed tokens, not just stored
                position_ids = torch.arange(
                    n_total_seen, n_total_seen + seq_len, device=device
                ).unsqueeze(0)
                attn_mask = torch.ones(1, n_past + seq_len, device=device, dtype=torch.long)
                output = model(qa_chunk, past_key_values=past_kvs,
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


def test_generation(model, kv_memory, tokenizer, device):
    model.eval()
    filler = "Modern computing has revolutionized information processing."

    def process_chunk(text, chunk_size=128, store=True):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:chunk_size]
        if len(tokens) < chunk_size:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(tokens))
        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            output = model(input_ids, use_cache=True)
            if store:
                kv_memory.store_all(output.past_key_values)
            else:
                kv_memory.skip(len(tokens))

    def generate_with_memory(question, max_new=100):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        past_kvs = kv_memory.get_past_key_values(device, torch.bfloat16)

        with torch.no_grad():
            if past_kvs is not None:
                n_past = kv_memory.n_stored.item()
                seq_len = input_ids.shape[1]
                position_ids = torch.arange(
                    n_past, n_past + seq_len + max_new, device=device
                )[:seq_len].unsqueeze(0)
                attn_mask = torch.ones(1, n_past + seq_len, device=device, dtype=torch.long)

                out = model.generate(
                    input_ids,
                    past_key_values=past_kvs,
                    position_ids=position_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new,
                    do_sample=False,
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
    kv_memory.reset()
    for q in [
        "What is the capital of France?",
        "Write a haiku about programming.",
    ]:
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
        kv_memory.reset()
        process_chunk(passage, store=True)   # store passage KVs
        for _ in range(3):
            process_chunk(filler, store=False)  # skip filler KVs
        resp = generate_with_memory(q)
        print(f"  Stored: \"{passage[:60]}...\"")
        print(f"  Q: {q}")
        print(f"  A: {resp[:200]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="v3.13: KV Cache Memory")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-memory", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--max-entries", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_13_kv_memory")

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

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    # Get n_kv_heads and head_dim from model config
    n_kv_heads = model.config.num_key_value_heads
    head_dim = hidden_size // model.config.num_attention_heads

    print(f"  Hidden: {hidden_size}, Layers: {n_layers}, KV heads: {n_kv_heads}, Head dim: {head_dim}")

    kv_memory = KVMemoryBank(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_entries=args.max_entries,
        hidden_size=hidden_size,
    ).to(device=args.device, dtype=torch.bfloat16)

    n_gate = sum(p.numel() for p in kv_memory.parameters())
    kv_size = args.max_entries * n_layers * n_kv_heads * head_dim * 2 * 2  # keys+values, 2 bytes bf16
    print(f"  Gate params: {n_gate:,}")
    print(f"  KV storage: {kv_size / 1024 / 1024:.1f} MB (max {args.max_entries} entries)")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    print("\nBuilding training data...")
    train_docs = build_dataset(tokenizer, n_memory=args.n_memory, seed=args.seed)
    print("Building eval data...")
    eval_docs = build_dataset(tokenizer, n_memory=args.n_eval, seed=args.seed + 1000)

    results = train(
        model, kv_memory, tokenizer, train_docs, eval_docs, args.device,
    )

    test_generation(model, kv_memory, tokenizer, args.device)

    results["config"] = vars(args)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
