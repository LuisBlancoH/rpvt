"""Experiment v3.18: Predictive coding + KV cache memory combined.

Two systems, separate roles:
  Predictive coding (inverse + GRU): decides what's surprising → what to store
  KV cache memory: stores passage KVs → model attends natively

The inverse watches the forward model process each chunk and computes
prediction errors. High error = surprising = store this chunk's KVs.
Low error = expected = skip.

No modulation. No LoRA. No weight changes. The predictive coding
system is a SELECTOR, not a modifier.

Compare:
  - Store ALL passage KVs (baseline, 74.2%)
  - Store only HIGH-ERROR passage KVs (predictive selection)
  - Store NO passage KVs (no memory baseline)
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


# Use the GRU version of the inverse (proven 26% context improvement)
from rpvt.model.predictive_coding_v2 import (
    PersistentInverseTransformer, HiddenStateCapture, PredictiveCodingSystem,
)


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


def evaluate_combined(model, pc_system, kv_memory, eval_docs, device,
                      mode="all", error_threshold=None):
    """Evaluate memory recall with different storage strategies.

    Modes:
      "all": store all passage KVs (baseline)
      "predictive": store only chunks with high prediction error
      "none": no memory (baseline)
    """
    model.eval()
    pc_system.inverse.eval()
    correct = total = 0
    chunks_stored = 0
    chunks_skipped = 0

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            chunk_types = doc["chunk_types"]
            answer_mask = doc["answer_mask"].to(device)

            kv_memory.reset()
            pc_system.reset()

            for ci, chunk in enumerate(chunks[:-1]):
                chunk_ids = chunk.unsqueeze(0).to(device)

                # Forward pass
                output = model(chunk_ids, use_cache=True)

                # Predictive coding: compute surprise
                _, errors, magnitudes = pc_system.process_chunk(chunk_ids)
                avg_error = sum(magnitudes.values()) / max(len(magnitudes), 1) if magnitudes else 0

                if chunk_types[ci] == "passage":
                    if mode == "all":
                        kv_memory.store_all(output.past_key_values)
                        chunks_stored += 1
                    elif mode == "predictive":
                        if error_threshold is None or avg_error > error_threshold:
                            kv_memory.store_all(output.past_key_values)
                            chunks_stored += 1
                        else:
                            kv_memory.skip(chunk_ids.shape[1])
                            chunks_skipped += 1
                    elif mode == "none":
                        kv_memory.skip(chunk_ids.shape[1])
                else:
                    # Filler: check if predictive coding flags it as surprising
                    if mode == "predictive" and error_threshold is not None:
                        if avg_error > error_threshold * 1.5:
                            # Very surprising filler — might be worth storing
                            kv_memory.store_all(output.past_key_values)
                            chunks_stored += 1
                        else:
                            kv_memory.skip(chunk_ids.shape[1])
                    else:
                        kv_memory.skip(chunk_ids.shape[1])

            # QA with stored KVs
            qa = chunks[-1].unsqueeze(0).to(device)
            past = kv_memory.get_past_key_values(device, torch.bfloat16)

            if past is not None:
                n_past = kv_memory.n_stored.item()
                n_total = kv_memory.total_tokens_seen.item()
                sl = qa.shape[1]
                output = model(
                    qa, past_key_values=past,
                    position_ids=torch.arange(n_total, n_total + sl, device=device).unsqueeze(0),
                    attention_mask=torch.ones(1, n_past + sl, device=device, dtype=torch.long),
                )
            else:
                output = model(qa)

            preds = output.logits[0, :-1].argmax(dim=-1)
            targets = qa[0, 1:]
            for p in answer_mask[:-1].nonzero(as_tuple=True)[0]:
                total += 1
                if preds[p].item() == targets[p].item():
                    correct += 1

    acc = correct / max(total, 1)
    return {
        "token_acc": acc, "correct": correct, "total": total,
        "chunks_stored": chunks_stored, "chunks_skipped": chunks_skipped,
    }


def main():
    parser = argparse.ArgumentParser(description="v3.18: Predictive coding + KV memory")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-memory", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--inverse-checkpoint", type=str, default=None,
                        help="Load pre-trained inverse from alignment experiment")
    parser.add_argument("--train-epochs", type=int, default=5,
                        help="Epochs to train inverse if no checkpoint")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_18_combined")

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

    # Set up hidden state captures
    layers = model.model.layers
    captures = []
    for li in target_layers:
        c = HiddenStateCapture(layers[li], li)
        layers[li] = c
        captures.append(c)

    # GRU-based inverse (proven 26% context improvement)
    # Need to use the GRU version — reimport with GRU
    inverse = PersistentInverseTransformer(
        hidden_size, n_inverse_layers=len(target_layers),
        target_layers=target_layers, n_heads=8,
        max_context_tokens=256,
    ).to(args.device, dtype=torch.bfloat16)

    pc_system = PredictiveCodingSystem(model, inverse, target_layers, captures)

    # KV memory
    n_kv_heads = model.config.num_key_value_heads
    head_dim = hidden_size // model.config.num_attention_heads
    kv_memory = KVMemoryBank(
        model.config.num_hidden_layers, n_kv_heads, head_dim,
        max_entries=512, hidden_size=hidden_size,
    ).to(args.device, dtype=torch.bfloat16)

    print(f"  Inverse: {sum(p.numel() for p in inverse.parameters()):,} params")
    print(f"  KV memory: max 512 entries")

    # Build data
    print("\nBuilding data...")
    train_docs = build_dataset(tokenizer, n_memory=args.n_memory, seed=args.seed)
    eval_docs = build_dataset(tokenizer, n_memory=args.n_eval, seed=args.seed + 1000)

    # Train inverse on the memory docs (learn to predict forward model)
    print(f"\nTraining inverse on memory docs ({args.train_epochs} epochs)...")
    trainable = list(inverse.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    inverse.train()

    for epoch in range(args.train_epochs):
        random.shuffle(train_docs)
        total_loss = 0
        n_steps = 0

        for doc in train_docs:
            pc_system.reset()
            for chunk in doc["chunks"][:-1]:
                chunk_ids = chunk.unsqueeze(0).to(args.device)
                loss = pc_system.prediction_loss(chunk_ids)
                if loss.item() > 0 and not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        print(f"  Epoch {epoch + 1}: pred_loss={avg_loss:.4f}")

    # Now evaluate with different storage strategies
    print("\n=== Evaluation ===\n")

    # Baseline: no memory
    result_none = evaluate_combined(
        model, pc_system, kv_memory, eval_docs, args.device, mode="none"
    )
    print(f"No memory:     {result_none['token_acc']:.1%} "
          f"({result_none['correct']}/{result_none['total']})")

    # Baseline: store all passage KVs
    result_all = evaluate_combined(
        model, pc_system, kv_memory, eval_docs, args.device, mode="all"
    )
    print(f"All KVs:       {result_all['token_acc']:.1%} "
          f"({result_all['correct']}/{result_all['total']}) "
          f"— {result_all['chunks_stored']} chunks stored")

    # Predictive: compute error distribution to set threshold
    print("\nComputing error distribution...")
    all_errors = []
    with torch.no_grad():
        for doc in eval_docs[:20]:
            pc_system.reset()
            for ci, chunk in enumerate(doc["chunks"][:-1]):
                chunk_ids = chunk.unsqueeze(0).to(args.device)
                model(chunk_ids, use_cache=True)
                _, _, mags = pc_system.process_chunk(chunk_ids)
                if mags:
                    avg_e = sum(mags.values()) / len(mags)
                    all_errors.append((doc["chunk_types"][ci], avg_e))

    passage_errors = [e for t, e in all_errors if t == "passage"]
    filler_errors = [e for t, e in all_errors if t == "filler"]
    print(f"  Passage avg error: {sum(passage_errors)/max(len(passage_errors),1):.4f}")
    print(f"  Filler avg error:  {sum(filler_errors)/max(len(filler_errors),1):.4f}")

    # Test predictive selection at different thresholds
    if passage_errors:
        median_error = sorted(passage_errors)[len(passage_errors) // 2]
        for threshold_mult in [0.0, 0.5, 1.0, 1.5]:
            threshold = median_error * threshold_mult if threshold_mult > 0 else 0.0
            result = evaluate_combined(
                model, pc_system, kv_memory, eval_docs, args.device,
                mode="predictive", error_threshold=threshold,
            )
            print(f"Predictive (t={threshold:.4f}): {result['token_acc']:.1%} "
                  f"— stored={result['chunks_stored']}, skipped={result['chunks_skipped']}")

    # Test generation
    print("\n=== Generation Tests ===\n")
    filler = "Modern computing has revolutionized information processing."

    def process_and_store(text, chunk_size=128):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:chunk_size]
        if len(tokens) < chunk_size:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(tokens))
        input_ids = torch.tensor([tokens], device=args.device)
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
            kv_memory.store_all(out.past_key_values)

    def generate(question, max_new=100):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=args.device)
        past = kv_memory.get_past_key_values(args.device, torch.bfloat16)
        kwargs = {}
        if past is not None:
            n_past = kv_memory.n_stored.item()
            n_total = kv_memory.total_tokens_seen.item()
            sl = input_ids.shape[1]
            kwargs["past_key_values"] = past
            kwargs["position_ids"] = torch.arange(
                n_total, n_total + sl, device=args.device
            ).unsqueeze(0)
            kwargs["attention_mask"] = torch.ones(
                1, n_past + sl, device=args.device, dtype=torch.long
            )
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, **kwargs
            )
        return tokenizer.decode(out[0][len(tokens):], skip_special_tokens=True)

    # No memory
    kv_memory.reset()
    print("No memory:")
    print(f"  Q: What is the capital of France?")
    print(f"  A: {generate('What is the capital of France?')[:150]}\n")

    # With memory
    kv_memory.reset()
    process_and_store("Classified briefing: The operation code is VIPER-371. Agent Blackwood is stationed at Nordheim.")
    for _ in range(3):
        tokens = tokenizer.encode(filler, add_special_tokens=False)[:128]
        tokens = tokens + [tokenizer.eos_token_id] * (128 - len(tokens))
        with torch.no_grad():
            model(torch.tensor([tokens], device=args.device), use_cache=True)
        kv_memory.skip(128)
    print("With memory:")
    print(f"  Q: What is the operation code?")
    print(f"  A: {generate('What is the operation code from the briefing?')[:150]}\n")

    results = {
        "no_memory": result_none,
        "all_kvs": result_all,
        "config": vars(args),
    }
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
