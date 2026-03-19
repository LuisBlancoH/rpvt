"""Experiment v3.6: Parallel cross-attention for instruct models.

The key insight: decouple memory reading from the model's generation distribution.

Previous approaches (v3.4, v3.5) all used LoRA on q_proj/v_proj, which shifts
the output distribution and breaks coherent generation. This experiment uses
a completely SEPARATE cross-attention module:

  - Instruct model weights: 100% frozen (no LoRA)
  - Memory bank: write gate only (1,537 params)
  - Parallel cross-attention: own Q/K/V/O projections (~1.6M params)
  - Output gate starts at 0 → model is pure instruct initially

The parallel cross-attention reads from memory and adds a gated signal to the
residual stream. Because the model's self-attention is untouched, generation
quality is preserved by construction.

Architecture:
  Layer 14 (WriteWrapper): normal forward + write gated hidden states to bank
  Layer 15 (ParallelCrossAttentionWrapper):
    → normal self_attn + MLP (unchanged)
    → + gate * cross_attn(layer_output, memory)  ← NEW, separate weights
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

from rpvt.experiments.exp_v3_1_pretrained_recall import (
    build_model,
    reset_memories,
    set_persistent_grad,
    detach_memory_state,
)
from rpvt.experiments.exp_v3_2_nlp_recall import (
    _get_memory_module,
    _generate_natural_facts,
    _generate_instruction_following,
)


def get_parallel_params(model):
    """Get all trainable params: memory bank + parallel cross-attention."""
    params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[name] = param
    return params


def build_instruct_dataset(tokenizer, n_memory=500, n_instruct=500,
                           chunk_size=128, gap_range=(2, 6), max_qa_pairs=3,
                           seed=42):
    """Build mixed dataset: memory recall + instruct examples.

    Memory docs: passage → filler → QA (chat format, answer-only loss)
    Instruct docs: single chunk, chat format, response-only loss
    """
    rng = random.Random(seed)

    # Memory docs
    print(f"  Generating {n_memory} memory docs...")
    recall_docs = _generate_natural_facts(rng, n_memory, max_qa_pairs)

    # Load filler
    print("  Loading WikiText for filler...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    filler_texts = [t for t in wiki["text"] if len(t.strip()) > 100]

    memory_docs = []
    for passage, qa_pairs in recall_docs:
        gap = rng.randint(gap_range[0], gap_range[1])

        # Passage chunks
        passage_tokens = tokenizer.encode(passage, add_special_tokens=False)
        passage_chunks = []
        for i in range(0, len(passage_tokens), chunk_size):
            ct = passage_tokens[i:i + chunk_size]
            if len(ct) < chunk_size:
                ct = ct + [tokenizer.eos_token_id or 0] * (chunk_size - len(ct))
            passage_chunks.append(torch.tensor(ct, dtype=torch.long))

        # Filler chunks
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

        # QA chunks with chat template
        for qa in qa_pairs:
            messages = [{"role": "user", "content": qa["question"]}]
            chat_prefix = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = chat_prefix + qa["answer"] + "<|im_end|>"
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            prefix_len = len(tokenizer.encode(chat_prefix, add_special_tokens=False))

            answer_mask = torch.zeros(chunk_size, dtype=torch.float32)
            for pos in range(max(0, prefix_len - 1), min(len(full_tokens) - 1, chunk_size)):
                answer_mask[pos] = 1.0

            if len(full_tokens) >= chunk_size:
                full_tokens = full_tokens[:chunk_size]
            else:
                full_tokens = full_tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(full_tokens))

            qa_chunk = torch.tensor(full_tokens, dtype=torch.long)
            all_chunks = passage_chunks + filler_chunks + [qa_chunk]

            memory_docs.append({
                "type": "memory",
                "chunks": all_chunks,
                "answer_mask": answer_mask,
            })

    # Instruct docs from Alpaca
    print(f"  Loading {n_instruct} instruct examples...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    alpaca_list = list(alpaca)
    rng.shuffle(alpaca_list)

    instruct_docs = []
    for ex in alpaca_list[:n_instruct * 3]:
        if ex.get("input") and ex["input"].strip():
            content = f"{ex['instruction']}\n\nInput: {ex['input']}"
        else:
            content = ex["instruction"]

        messages = [{"role": "user", "content": content}]
        chat_prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = ex["output"]
        full_text = chat_prefix + response + "<|im_end|>"
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
        prefix_len = len(tokenizer.encode(chat_prefix, add_special_tokens=False))

        if len(full_tokens) > chunk_size * 2:
            continue
        if prefix_len >= chunk_size - 5:
            continue

        response_mask = torch.zeros(chunk_size, dtype=torch.float32)
        for pos in range(max(0, prefix_len - 1), min(len(full_tokens) - 1, chunk_size)):
            response_mask[pos] = 1.0

        if len(full_tokens) >= chunk_size:
            full_tokens = full_tokens[:chunk_size]
        else:
            full_tokens = full_tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(full_tokens))

        instruct_docs.append({
            "type": "instruct",
            "chunks": [torch.tensor(full_tokens, dtype=torch.long)],
            "answer_mask": response_mask,
        })

        if len(instruct_docs) >= n_instruct:
            break

    all_docs = memory_docs + instruct_docs
    rng.shuffle(all_docs)

    n_mem = sum(1 for d in all_docs if d["type"] == "memory")
    n_inst = sum(1 for d in all_docs if d["type"] == "instruct")
    print(f"  Dataset: {len(all_docs)} docs ({n_mem} memory, {n_inst} instruct)")
    return all_docs


def train(model, tokenizer, train_docs, eval_docs, device,
          num_epochs=15, lr=1e-3, lr_lora=2e-4, log_every=50):
    """Train parallel cross-attention + optional LoRA on downstream layers."""
    # Separate param groups: memory/cross-attn get higher LR, LoRA gets lower
    memory_params = []
    lora_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in name.lower():
            lora_params.append(p)
        else:
            memory_params.append(p)

    param_groups = [{"params": memory_params, "lr": lr}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora})

    n_mem = sum(p.numel() for p in memory_params)
    n_lora = sum(p.numel() for p in lora_params)
    print(f"\nTraining {n_mem + n_lora:,} parameters ({n_mem:,} memory/cross-attn, {n_lora:,} LoRA), {num_epochs} epochs")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    total_steps = len(train_docs) * num_epochs

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    model.train()
    global_step = 0
    losses = {"memory": [], "instruct": []}
    start_time = time.time()

    for epoch in range(num_epochs):
        order = list(range(len(train_docs)))
        random.shuffle(order)

        for doc_idx in order:
            doc = train_docs[doc_idx]
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)
            is_memory = doc["type"] == "memory"

            if is_memory:
                reset_memories(model)
                set_persistent_grad(model, True)

            doc_loss = torch.tensor(0.0, device=device)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                is_last = (chunk_idx == len(chunks) - 1)

                output = model(chunk_ids)

                if is_last:
                    logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
                    targets = chunk_ids[:, 1:].reshape(-1)
                    per_token = F.cross_entropy(logits, targets, reduction='none')
                    mask = answer_mask[:-1]
                    n_tokens = mask.sum().clamp(min=1)
                    doc_loss = (per_token * mask).sum() / n_tokens

            if doc_loss.item() > 0:
                optimizer.zero_grad()
                doc_loss.backward()
                all_trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
                optimizer.step()
                scheduler.step()

            if is_memory:
                detach_memory_state(model)
                set_persistent_grad(model, False)

            losses[doc["type"]].append(doc_loss.item())
            global_step += 1

            if global_step % log_every == 0:
                mem_recent = losses["memory"][-log_every:]
                inst_recent = losses["instruct"][-log_every:]
                mem_avg = sum(mem_recent) / max(len(mem_recent), 1)
                inst_avg = sum(inst_recent) / max(len(inst_recent), 1)
                elapsed = time.time() - start_time

                # Log o_proj norm as proxy for cross-attn strength
                o_norm = None
                for name, param in model.named_parameters():
                    if "cross_attn.o_proj" in name:
                        o_norm = param.data.norm().item()
                        break

                o_str = f", o_norm={o_norm:.4f}" if o_norm is not None else ""
                print(f"  step {global_step}/{total_steps}, "
                      f"mem_loss={mem_avg:.3f}, inst_loss={inst_avg:.3f}"
                      f"{o_str}, lr={scheduler.get_last_lr()[0]:.2e}, {elapsed:.0f}s")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        eval_results = evaluate(model, tokenizer, eval_docs, device)
        print(f"  Memory recall: {eval_results['memory_token_acc']:.1%} "
              f"({eval_results['memory_correct']}/{eval_results['memory_total']})")
        print(f"  Instruct loss: {eval_results['instruct_loss']:.3f}")
        model.train()

    return evaluate(model, tokenizer, eval_docs, device, verbose=True)


def evaluate(model, tokenizer, eval_docs, device, verbose=False):
    """Evaluate memory recall accuracy and instruct loss."""
    model.eval()
    mem_correct = 0
    mem_total = 0
    inst_losses = []

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)
            is_memory = doc["type"] == "memory"

            if is_memory:
                reset_memories(model)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                output = model(chunk_ids)

                if chunk_idx == len(chunks) - 1:
                    if is_memory:
                        predictions = output.logits[0, :-1].argmax(dim=-1)
                        targets = chunk_ids[0, 1:]
                        mask = answer_mask[:-1]
                        positions = mask.nonzero(as_tuple=True)[0]
                        for p in positions:
                            mem_total += 1
                            if predictions[p].item() == targets[p].item():
                                mem_correct += 1
                    else:
                        logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
                        targets = chunk_ids[:, 1:].reshape(-1)
                        per_token = F.cross_entropy(logits, targets, reduction='none')
                        mask = answer_mask[:-1]
                        if mask.sum() > 0:
                            loss = (per_token * mask).sum() / mask.sum()
                            inst_losses.append(loss.item())

    return {
        "memory_token_acc": mem_correct / max(mem_total, 1),
        "memory_correct": mem_correct,
        "memory_total": mem_total,
        "instruct_loss": sum(inst_losses) / max(len(inst_losses), 1),
    }


def test_generation(model, tokenizer, device):
    """Test both memory recall and free generation quality."""
    model.eval()
    filler = "Modern computing has revolutionized information processing. Algorithms handle billions of operations per second."

    def process_chunk(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:128]
        if len(tokens) < 128:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (128 - len(tokens))
        with torch.no_grad():
            model(torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device))

    def generate_chat(question, max_new=80):
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

    print("\n=== GENERATION TESTS ===\n")

    # Test 1: basic generation (no memory) — should be perfect since no LoRA
    print("1. Basic instruct (no memory, should be perfect):")
    reset_memories(model)
    for q in [
        "What is the capital of France?",
        "Write a haiku about programming.",
        "Explain what a neural network is in one sentence.",
    ]:
        resp = generate_chat(q)
        print(f"  Q: {q}")
        print(f"  A: {resp[:150]}")
        print()

    # Test 2: memory recall
    print("2. Memory recall + generation:")
    memory_tests = [
        ("Classified briefing: The operation code is VIPER-371. Agent Blackwood is stationed at Nordheim.",
         "What is the operation code from the briefing?"),
        ("The secret password is THUNDERBOLT. Remember this password.",
         "What is the secret password?"),
        ("The project lead is Dr. Stellion. The budget is 45000 dollars.",
         "Who is the project lead and what is the budget?"),
    ]

    for passage, q in memory_tests:
        reset_memories(model)
        process_chunk(passage)
        for _ in range(3):
            process_chunk(filler)
        resp = generate_chat(q)
        print(f"  Stored: \"{passage[:60]}...\"")
        print(f"  Q: {q}")
        print(f"  A: {resp[:150]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="v3.6: Parallel cross-attention for instruct models")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-memory", type=int, default=500)
    parser.add_argument("--n-instruct", type=int, default=0)
    parser.add_argument("--n-eval-memory", type=int, default=50)
    parser.add_argument("--n-eval-instruct", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-layers", type=str, default=None,
                        help="Comma-separated layer indices for LoRA, e.g. '16,17,18,...,27'")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_6_parallel")
    parser.add_argument("--save-checkpoint", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Parse lora_layers
    if args.lora_layers:
        args.lora_layers = [int(x) for x in args.lora_layers.split(",")]

    # Build instruct model with parallel cross-attention
    # + small LoRA on downstream layers only (to help process memory signal)
    model, tokenizer = build_model(
        model_name=args.model_name, device=args.device,
        memory_layer=-1, memory_size=256, n_slots=64,
        decay=0.999, gate_bias=-2.0,
        lora_rank=args.lora_rank, lora_targets="q_proj,v_proj",
        no_memory=False, no_lora=(args.lora_rank == 0),
        init_qk_shared=False, n_extract=1,
        memory_mode="parallel_cross_attn",
        lora_layers=args.lora_layers,
    )

    # Verify: only memory/cross-attn params are trainable
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\nTrainable parameters ({len(trainable_names)}):")
    for name in trainable_names:
        p = dict(model.named_parameters())[name]
        print(f"  {name}: {p.shape}")

    print("\nBuilding training data...")
    train_docs = build_instruct_dataset(
        tokenizer, n_memory=args.n_memory, n_instruct=args.n_instruct,
        seed=args.seed,
    )
    print("Building eval data...")
    eval_docs = build_instruct_dataset(
        tokenizer, n_memory=args.n_eval_memory, n_instruct=args.n_eval_instruct,
        seed=args.seed + 1000,
    )

    results = train(
        model, tokenizer, train_docs, eval_docs, args.device,
        num_epochs=args.epochs, lr=args.lr, log_every=args.log_every,
    )

    if args.save_checkpoint:
        os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
        state = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(state, args.save_checkpoint)
        print(f"\nCheckpoint saved to {args.save_checkpoint}")

    test_generation(model, tokenizer, args.device)

    results["config"] = vars(args)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
