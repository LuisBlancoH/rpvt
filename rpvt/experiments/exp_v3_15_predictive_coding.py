"""Experiment v3.15: Recurrent Predictive Coding for instruct+memory.

The architecture:
  1. Frozen instruct transformer (bottom-up, runs once to observe)
  2. Trainable inverse transformer (top-down, predicts layer states)
  3. Prediction errors modulate activations on second forward pass
  4. KV cache stores passage representations for memory

Zero modulation = exact instruct model (generation preserved).
Error modulation = memory-augmented processing.

Training: answer loss backprops through modulated forward pass
→ inverse transformer learns what to predict
→ modulation learns the right scale
→ end-to-end, no LoRA, weights frozen.
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

from rpvt.model.predictive_coding import (
    InverseTransformer, ModulationWrapper, RecurrentPredictiveCoding,
)
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


def build_system(model, device, target_layers, hidden_size):
    """Attach modulation wrappers and create inverse transformer."""

    # Wrap target layers with modulation
    layers = model.model.layers
    wrappers = []
    for layer_idx in target_layers:
        wrapper = ModulationWrapper(layers[layer_idx], layer_idx, scale=0.1)
        wrapper = wrapper.to(device=device, dtype=torch.bfloat16)
        layers[layer_idx] = wrapper
        wrappers.append(wrapper)

    # Create inverse transformer
    inverse = InverseTransformer(
        hidden_size=hidden_size,
        n_inverse_layers=len(target_layers),
        target_layers=target_layers,
        n_heads=8,
    ).to(device=device, dtype=torch.bfloat16)

    # Create KV memory bank
    n_kv_heads = model.config.num_key_value_heads
    head_dim = hidden_size // model.config.num_attention_heads
    kv_memory = KVMemoryBank(
        n_layers=model.config.num_hidden_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_entries=512,
        hidden_size=hidden_size,
    ).to(device=device, dtype=torch.bfloat16)

    # Create the full system
    system = RecurrentPredictiveCoding(
        model=model,
        inverse_transformer=inverse,
        target_layers=target_layers,
        modulation_wrappers=wrappers,
        kv_memory=kv_memory,
    )

    return system, inverse, kv_memory, wrappers


def train(system, inverse, kv_memory, wrappers, tokenizer,
          train_docs, eval_docs, device,
          num_epochs=15, lr=1e-3, n_cycles=2, log_every=100,
          checkpoint_dir=None):
    """Train inverse transformer + modulation scales."""

    # Collect trainable params
    trainable = list(inverse.parameters())
    for w in wrappers:
        trainable.append(w.scale)
    trainable.extend(list(kv_memory.parameters()))

    n_params = sum(p.numel() for p in trainable)
    print(f"\nTraining {n_params:,} params (inverse={sum(p.numel() for p in inverse.parameters()):,})")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = len(train_docs) * num_epochs

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"  {total_steps} steps, {num_epochs} epochs, {n_cycles} cycles per step")
    inverse.train()
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

            kv_memory.reset()

            # Process context chunks — store passage KVs
            with torch.no_grad():
                for chunk_idx, chunk in enumerate(chunks[:-1]):
                    chunk_ids = chunk.unsqueeze(0).to(device)
                    output = system.model(chunk_ids, use_cache=True)
                    if chunk_types[chunk_idx] == "passage":
                        kv_memory.store_all(output.past_key_values)
                    else:
                        kv_memory.skip(chunk_ids.shape[1])

            # Get stored KVs for memory
            past_kvs = kv_memory.get_past_key_values(device, torch.bfloat16)

            # Prepare QA
            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            n_past = kv_memory.n_stored.item()
            n_total = kv_memory.total_tokens_seen.item()
            seq_len = qa_chunk.shape[1]

            kwargs = {}
            if past_kvs is not None:
                kwargs["past_key_values"] = past_kvs
                kwargs["position_ids"] = torch.arange(
                    n_total, n_total + seq_len, device=device
                ).unsqueeze(0)
                kwargs["attention_mask"] = torch.ones(
                    1, n_past + seq_len, device=device, dtype=torch.long
                )

            # Predictive coding cycle: observe → predict → modulate
            # Find answer start from mask to prevent leakage
            ans_positions = answer_mask.nonzero(as_tuple=True)[0]
            answer_start = ans_positions[0].item() + 1 if len(ans_positions) > 0 else None

            output, cycle_errors = system(
                qa_chunk, n_cycles=n_cycles, answer_start=answer_start, **kwargs
            )

            # Answer loss
            logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
            targets = qa_chunk[:, 1:].reshape(-1)
            per_token = F.cross_entropy(logits, targets, reduction='none')
            mask = answer_mask[:-1]
            n_tokens = mask.sum().clamp(min=1)
            loss = (per_token * mask).sum() / n_tokens

            # Also add prediction loss (self-supervised)
            if cycle_errors:
                pred_loss = sum(
                    sum(v for v in errs.values())
                    for errs in cycle_errors
                ) / len(cycle_errors)
                loss = loss + 0.01 * pred_loss

            if loss.item() > 0 and not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()

            losses.append(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                avg = sum(losses[-log_every:]) / min(len(losses), log_every)
                elapsed = time.time() - start_time
                eta = elapsed / global_step * (total_steps - global_step)
                scales = [f"{w.scale.item():.3f}" for w in wrappers]
                print(f"  step {global_step}/{total_steps}, loss={avg:.3f}, "
                      f"scales={scales}, lr={scheduler.get_last_lr()[0]:.2e}, "
                      f"{elapsed:.0f}s (ETA {eta/3600:.1f}h)")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        eval_results = evaluate(system, kv_memory, tokenizer, eval_docs,
                                device, n_cycles)
        print(f"  Memory recall: {eval_results['token_acc']:.1%} "
              f"({eval_results['correct']}/{eval_results['total']})")

        if checkpoint_dir:
            state = {
                "inverse": {n: p.data.clone() for n, p in inverse.named_parameters()},
                "scales": [w.scale.data.clone() for w in wrappers],
                "epoch": epoch, "global_step": global_step,
            }
            torch.save(state, os.path.join(checkpoint_dir, "latest.pt"))

        inverse.train()

    return eval_results


def evaluate(system, kv_memory, tokenizer, eval_docs, device, n_cycles=2):
    system.model.eval()
    system.inverse.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            chunk_types = doc["chunk_types"]
            answer_mask = doc["answer_mask"].to(device)

            kv_memory.reset()

            for chunk_idx, chunk in enumerate(chunks[:-1]):
                chunk_ids = chunk.unsqueeze(0).to(device)
                output = system.model(chunk_ids, use_cache=True)
                if chunk_types[chunk_idx] == "passage":
                    kv_memory.store_all(output.past_key_values)
                else:
                    kv_memory.skip(chunk_ids.shape[1])

            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            past_kvs = kv_memory.get_past_key_values(device, torch.bfloat16)

            kwargs = {}
            if past_kvs is not None:
                n_past = kv_memory.n_stored.item()
                n_total = kv_memory.total_tokens_seen.item()
                seq_len = qa_chunk.shape[1]
                kwargs["past_key_values"] = past_kvs
                kwargs["position_ids"] = torch.arange(
                    n_total, n_total + seq_len, device=device
                ).unsqueeze(0)
                kwargs["attention_mask"] = torch.ones(
                    1, n_past + seq_len, device=device, dtype=torch.long
                )

            ans_positions = answer_mask[:-1].nonzero(as_tuple=True)[0]
            answer_start = ans_positions[0].item() + 1 if len(ans_positions) > 0 else None

            output, _ = system(
                qa_chunk, n_cycles=n_cycles, answer_start=answer_start, **kwargs
            )

            predictions = output.logits[0, :-1].argmax(dim=-1)
            targets = qa_chunk[0, 1:]
            mask = answer_mask[:-1]
            for p in mask.nonzero(as_tuple=True)[0]:
                total += 1
                if predictions[p].item() == targets[p].item():
                    correct += 1

    return {"token_acc": correct / max(total, 1), "correct": correct, "total": total}


def test_generation(system, kv_memory, tokenizer, device):
    system.model.eval()
    system.inverse.eval()
    filler = "Modern computing has revolutionized information processing."

    def process_chunk(text, chunk_size=128, store=True):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:chunk_size]
        if len(tokens) < chunk_size:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(tokens))
        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            output = system.model(input_ids, use_cache=True)
            if store:
                kv_memory.store_all(output.past_key_values)
            else:
                kv_memory.skip(len(tokens))

    def generate(question, max_new=100):
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
                n_total = kv_memory.total_tokens_seen.item()
                seq_len = input_ids.shape[1]
                # For generation, use the modulated forward directly
                # (system.forward runs observe+predict+modulate cycles)
                output, _ = system(
                    input_ids, n_cycles=2,
                    past_key_values=past_kvs,
                    position_ids=torch.arange(
                        n_total, n_total + seq_len, device=device
                    ).unsqueeze(0),
                    attention_mask=torch.ones(
                        1, n_past + seq_len, device=device, dtype=torch.long
                    ),
                )
                # Greedy decode from modulated output
                generated = []
                for _ in range(max_new):
                    next_token = output.logits[0, -1].argmax()
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    generated.append(next_token.item())
                    # Simple greedy — just use the first modulated output
                    # For full generation, would need to iterate
                    break  # just get first token for now
                # Fall back to normal generate for the rest
                if generated:
                    out = system.model.generate(
                        input_ids, max_new_tokens=max_new, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        past_key_values=past_kvs,
                        position_ids=torch.arange(
                            n_total, n_total + seq_len, device=device
                        ).unsqueeze(0),
                        attention_mask=torch.ones(
                            1, n_past + seq_len, device=device, dtype=torch.long
                        ),
                    )
                else:
                    out = system.model.generate(
                        input_ids, max_new_tokens=max_new, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            else:
                out = system.model.generate(
                    input_ids, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        return tokenizer.decode(out[0][len(tokens):], skip_special_tokens=True)

    print("\n=== GENERATION TESTS ===\n")

    print("1. Basic instruct (no memory, no modulation):")
    kv_memory.reset()
    for q in ["What is the capital of France?", "Write a haiku about programming."]:
        resp = generate(q)
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
        process_chunk(passage, store=True)
        for _ in range(3):
            process_chunk(filler, store=False)
        resp = generate(q)
        print(f"  Stored: \"{passage[:60]}...\"")
        print(f"  Q: {q}")
        print(f"  A: {resp[:200]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="v3.15: Recurrent Predictive Coding")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-memory", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--n-cycles", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_15_predictive_coding")

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
    target_layers = [0, 7, 14, 21]

    print(f"  Target layers for predictive coding: {target_layers}")

    system, inverse, kv_memory, wrappers = build_system(
        model, args.device, target_layers, hidden_size
    )

    n_inv = sum(p.numel() for p in inverse.parameters())
    print(f"  Inverse transformer: {n_inv:,} params")
    print(f"  Modulation scales: {len(wrappers)} layers")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    print("\nBuilding training data...")
    train_docs = build_dataset(tokenizer, n_memory=args.n_memory, seed=args.seed)
    print("Building eval data...")
    eval_docs = build_dataset(tokenizer, n_memory=args.n_eval, seed=args.seed + 1000)

    results = train(
        system, inverse, kv_memory, wrappers, tokenizer,
        train_docs, eval_docs, args.device,
        num_epochs=args.epochs, lr=args.lr, n_cycles=args.n_cycles,
        log_every=args.log_every, checkpoint_dir=checkpoint_dir,
    )

    test_generation(system, kv_memory, tokenizer, args.device)

    results["config"] = vars(args)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
