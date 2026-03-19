"""Experiment v3.9: Train memory + instruct from scratch on base model.

The instruct+memory problem cannot be solved post-hoc: any weight changes
on an instruct model destroy its generation distribution, and frozen models
can only achieve ~33% recall.

Solution: train BOTH skills from scratch on the base model.

Key differences from v3.5 (which failed at generation):
  - 10k+ instruct examples (vs 1000)
  - 512-token sequences for instruct (vs 128)
  - Full response loss on instruct (not truncated)
  - Memory uses 128-token chunks as before

The LoRA learns generation AND memory reading simultaneously.
Neither skill is pretrained, so there's no distribution to destroy.
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
    get_memory_params,
    get_lora_params,
)
from rpvt.experiments.exp_v3_2_nlp_recall import (
    _get_memory_module,
    _generate_natural_facts,
)


def _make_qa_chunk(tokenizer, question, answer, chunk_size):
    """Helper: create a chat-formatted QA chunk with answer mask."""
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


def build_dataset(tokenizer, n_memory=500, n_instruct=10000,
                  mem_chunk_size=128, inst_max_len=512,
                  gap_range=(2, 6), max_qa_pairs=3, seed=42):
    """Build mixed dataset: memory recall + instruct + abstention + adversarial."""
    rng = random.Random(seed)

    # === Memory docs (128-token chunks, same as always) ===
    print(f"  Generating {n_memory} memory docs...")
    recall_docs = _generate_natural_facts(rng, n_memory, max_qa_pairs)

    print("  Loading WikiText for filler...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    filler_texts = [t for t in wiki["text"] if len(t.strip()) > 100]

    memory_docs = []
    for passage, qa_pairs in recall_docs:
        gap = rng.randint(gap_range[0], gap_range[1])

        # Passage chunks (128 tokens each)
        passage_tokens = tokenizer.encode(passage, add_special_tokens=False)
        passage_chunks = []
        for i in range(0, len(passage_tokens), mem_chunk_size):
            ct = passage_tokens[i:i + mem_chunk_size]
            if len(ct) < mem_chunk_size:
                ct = ct + [tokenizer.eos_token_id or 0] * (mem_chunk_size - len(ct))
            passage_chunks.append(torch.tensor(ct, dtype=torch.long))

        # Filler chunks
        filler_chunks = []
        for _ in range(gap):
            ft = rng.choice(filler_texts)
            ft_tok = tokenizer.encode(ft, add_special_tokens=False)
            if len(ft_tok) >= mem_chunk_size:
                start = rng.randint(0, len(ft_tok) - mem_chunk_size)
                ct = ft_tok[start:start + mem_chunk_size]
            else:
                ct = ft_tok + [tokenizer.eos_token_id or 0] * (mem_chunk_size - len(ft_tok))
            filler_chunks.append(torch.tensor(ct, dtype=torch.long))

        # QA chunk with chat template
        for qa in qa_pairs:
            qa_chunk, answer_mask = _make_qa_chunk(
                tokenizer, qa["question"], qa["answer"], mem_chunk_size
            )
            memory_docs.append({
                "type": "memory",
                "chunks": passage_chunks + filler_chunks + [qa_chunk],
                "answer_mask": answer_mask,
                "seq_len": mem_chunk_size,
            })

    # === Abstention docs: question WITHOUT passage in memory ===
    # Teaches model to say "I don't know" when memory has no evidence
    n_abstention = int(n_memory * 0.3)
    print(f"  Generating {n_abstention} abstention docs...")
    abstention_facts = _generate_natural_facts(rng, n_abstention, max_qa_pairs=2)

    abstention_responses = [
        "I don't have that information stored in my memory.",
        "I don't have information about that.",
        "I'm not sure, I don't have that in my memory.",
        "I don't have enough information to answer that.",
        "That information isn't in my memory.",
    ]

    abstention_docs = []
    for passage, qa_pairs in abstention_facts:
        gap = rng.randint(gap_range[0], gap_range[1])
        # ONLY filler chunks — NO passage. Memory will be empty of relevant facts.
        only_filler = []
        for _ in range(gap + 2):
            ft = rng.choice(filler_texts)
            ft_tok = tokenizer.encode(ft, add_special_tokens=False)
            if len(ft_tok) >= mem_chunk_size:
                start = rng.randint(0, len(ft_tok) - mem_chunk_size)
                ct = ft_tok[start:start + mem_chunk_size]
            else:
                ct = ft_tok + [tokenizer.eos_token_id or 0] * (mem_chunk_size - len(ft_tok))
            only_filler.append(torch.tensor(ct, dtype=torch.long))

        for qa in qa_pairs:
            # Same question, but answer is abstention
            abstention_answer = rng.choice(abstention_responses)
            qa_chunk, answer_mask = _make_qa_chunk(
                tokenizer, qa["question"], abstention_answer, mem_chunk_size
            )
            abstention_docs.append({
                "type": "abstention",
                "chunks": only_filler + [qa_chunk],
                "answer_mask": answer_mask,
                "seq_len": mem_chunk_size,
            })

    # === Adversarial docs: related but WRONG information in memory ===
    # Teaches model not to confuse similar entities
    n_adversarial = int(n_memory * 0.15)
    print(f"  Generating {n_adversarial} adversarial docs...")
    adversarial_facts_a = _generate_natural_facts(rng, n_adversarial, max_qa_pairs=2)
    adversarial_facts_b = _generate_natural_facts(rng, n_adversarial, max_qa_pairs=2)

    adversarial_docs = []
    for (passage_a, qa_a), (passage_b, qa_b) in zip(adversarial_facts_a, adversarial_facts_b):
        gap = rng.randint(gap_range[0], gap_range[1])
        # Store passage A in memory
        pa_tokens = tokenizer.encode(passage_a, add_special_tokens=False)
        pa_chunks = []
        for i in range(0, len(pa_tokens), mem_chunk_size):
            ct = pa_tokens[i:i + mem_chunk_size]
            if len(ct) < mem_chunk_size:
                ct = ct + [tokenizer.eos_token_id or 0] * (mem_chunk_size - len(ct))
            pa_chunks.append(torch.tensor(ct, dtype=torch.long))

        adv_filler = []
        for _ in range(gap):
            ft = rng.choice(filler_texts)
            ft_tok = tokenizer.encode(ft, add_special_tokens=False)
            if len(ft_tok) >= mem_chunk_size:
                start = rng.randint(0, len(ft_tok) - mem_chunk_size)
                ct = ft_tok[start:start + mem_chunk_size]
            else:
                ct = ft_tok + [tokenizer.eos_token_id or 0] * (mem_chunk_size - len(ft_tok))
            adv_filler.append(torch.tensor(ct, dtype=torch.long))

        # Ask about person B (not stored!) — model should abstain
        for qa in qa_b[:1]:
            abstention_answer = rng.choice(abstention_responses)
            qa_chunk, answer_mask = _make_qa_chunk(
                tokenizer, qa["question"], abstention_answer, mem_chunk_size
            )
            adversarial_docs.append({
                "type": "adversarial",
                "chunks": pa_chunks + adv_filler + [qa_chunk],
                "answer_mask": answer_mask,
                "seq_len": mem_chunk_size,
            })

    # === Instruct docs (up to 512 tokens, full response) ===
    print(f"  Loading {n_instruct} instruct examples...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    alpaca_list = list(alpaca)
    rng.shuffle(alpaca_list)

    instruct_docs = []
    for ex in alpaca_list:
        if len(instruct_docs) >= n_instruct:
            break

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

        # Skip if too long or too short
        if len(full_tokens) > inst_max_len:
            continue
        if prefix_len >= len(full_tokens) - 5:
            continue
        if len(full_tokens) < 20:
            continue

        # Response mask — loss on response tokens only
        seq_len = len(full_tokens)
        response_mask = torch.zeros(seq_len, dtype=torch.float32)
        for pos in range(max(0, prefix_len - 1), seq_len - 1):
            response_mask[pos] = 1.0

        # Pad to seq_len (no fixed size — variable length)
        instruct_docs.append({
            "type": "instruct",
            "chunks": [torch.tensor(full_tokens, dtype=torch.long)],
            "answer_mask": response_mask,
            "seq_len": seq_len,
        })

    all_docs = memory_docs + abstention_docs + adversarial_docs + instruct_docs
    rng.shuffle(all_docs)

    type_counts = {}
    for d in all_docs:
        type_counts[d["type"]] = type_counts.get(d["type"], 0) + 1
    avg_inst_len = sum(d["seq_len"] for d in all_docs if d["type"] == "instruct") / max(type_counts.get("instruct", 1), 1)
    print(f"  Dataset: {len(all_docs)} docs — {type_counts}, "
          f"avg instruct len={avg_inst_len:.0f} tokens")
    return all_docs


def train(model, tokenizer, train_docs, eval_docs, device,
          num_epochs=15, lr_memory=1e-3, lr_lora=2e-4, log_every=50):
    """Train memory + instruct jointly from scratch."""
    memory_params = get_memory_params(model)
    lora_params = get_lora_params(model)

    param_groups = []
    seen_ids = set()
    mem_p = [p for n, p in model.named_parameters()
             if p.requires_grad and n in memory_params and id(p) not in seen_ids
             and not seen_ids.add(id(p))]
    lora_p = [p for n, p in model.named_parameters()
              if p.requires_grad and n in lora_params and id(p) not in seen_ids
              and not seen_ids.add(id(p))]

    if mem_p:
        param_groups.append({"params": mem_p, "lr": lr_memory})
    if lora_p:
        param_groups.append({"params": lora_p, "lr": lr_lora})

    n_mem = sum(p.numel() for p in mem_p)
    n_lora = sum(p.numel() for p in lora_p)
    print(f"\nTraining {n_mem + n_lora:,} params (memory={n_mem:,}, lora={n_lora:,})")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    total_steps = len(train_docs) * num_epochs

    def lr_schedule(step):
        warmup = 200
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"  {total_steps} total steps, {num_epochs} epochs")
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
            is_memory = doc["type"] in ("memory", "abstention", "adversarial")

            if is_memory:
                reset_memories(model)
                set_persistent_grad(model, True)

            # Process chunks
            doc_loss = torch.tensor(0.0, device=device)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                is_last = (chunk_idx == len(chunks) - 1)

                output = model(chunk_ids)

                if is_last:
                    logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
                    targets = chunk_ids[:, 1:].reshape(-1)
                    per_token = F.cross_entropy(logits, targets, reduction='none')

                    # Answer/response mask
                    mask = doc["answer_mask"].to(device)
                    if mask.shape[0] > per_token.shape[0]:
                        mask = mask[:per_token.shape[0]]
                    elif mask.shape[0] < per_token.shape[0]:
                        mask = F.pad(mask, (0, per_token.shape[0] - mask.shape[0]))

                    n_tokens = mask.sum().clamp(min=1)
                    doc_loss = (per_token * mask).sum() / n_tokens

            if doc_loss.item() > 0:
                optimizer.zero_grad()
                doc_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()

            if is_memory:
                detach_memory_state(model)
                set_persistent_grad(model, False)

            loss_type = "memory" if doc["type"] in ("memory", "abstention", "adversarial") else "instruct"
            if loss_type not in losses:
                losses[loss_type] = []
            losses[loss_type].append(doc_loss.item())
            global_step += 1

            if global_step % log_every == 0:
                mem_recent = losses["memory"][-log_every:]
                inst_recent = losses["instruct"][-log_every:]
                mem_avg = sum(mem_recent) / max(len(mem_recent), 1)
                inst_avg = sum(inst_recent) / max(len(inst_recent), 1)
                elapsed = time.time() - start_time
                eta = elapsed / global_step * (total_steps - global_step)
                print(f"  step {global_step}/{total_steps}, "
                      f"mem={mem_avg:.3f}, inst={inst_avg:.3f}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, "
                      f"{elapsed:.0f}s (ETA {eta/3600:.1f}h)")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        eval_results = evaluate(model, tokenizer, eval_docs, device)
        print(f"  Memory recall: {eval_results['memory_token_acc']:.1%} "
              f"({eval_results['memory_correct']}/{eval_results['memory_total']})")
        print(f"  Instruct loss: {eval_results['instruct_loss']:.3f}")
        model.train()

    return evaluate(model, tokenizer, eval_docs, device)


def evaluate(model, tokenizer, eval_docs, device):
    """Evaluate memory recall and instruct quality."""
    model.eval()
    mem_correct = 0
    mem_total = 0
    inst_losses = []

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            is_memory = doc["type"] in ("memory", "abstention", "adversarial")

            if is_memory:
                reset_memories(model)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                output = model(chunk_ids)

                if chunk_idx == len(chunks) - 1:
                    if is_memory:
                        predictions = output.logits[0, :-1].argmax(dim=-1)
                        targets = chunk_ids[0, 1:]
                        mask = doc["answer_mask"]
                        if mask.shape[0] > predictions.shape[0]:
                            mask = mask[:predictions.shape[0]]
                        positions = mask.nonzero(as_tuple=True)[0]
                        for p in positions:
                            mem_total += 1
                            if predictions[p].item() == targets[p].item():
                                mem_correct += 1
                    else:
                        logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
                        targets = chunk_ids[:, 1:].reshape(-1)
                        per_token = F.cross_entropy(logits, targets, reduction='none')
                        mask = doc["answer_mask"].to(device)
                        if mask.shape[0] > per_token.shape[0]:
                            mask = mask[:per_token.shape[0]]
                        elif mask.shape[0] < per_token.shape[0]:
                            mask = F.pad(mask, (0, per_token.shape[0] - mask.shape[0]))
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
    """Test both memory recall generation and instruct generation."""
    model.eval()
    filler = "Modern computing has revolutionized information processing. Algorithms handle billions of operations per second."

    def process_chunk(text, chunk_size=128):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:chunk_size]
        if len(tokens) < chunk_size:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(tokens))
        with torch.no_grad():
            model(torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device))

    def generate_chat(question, max_new=100):
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

    # Test 1: Basic instruct (no memory)
    print("1. Basic instruct (no memory):")
    reset_memories(model)
    for q in [
        "What is the capital of France?",
        "Write a haiku about programming.",
        "Explain what a neural network is in one sentence.",
        "List three benefits of exercise.",
    ]:
        resp = generate_chat(q)
        print(f"  Q: {q}")
        print(f"  A: {resp[:200]}")
        print()

    # Test 2: Memory recall + generation
    print("2. Memory recall + generation:")
    tests = [
        ("Classified briefing: The operation code is VIPER-371. Agent Blackwood is stationed at Nordheim.",
         "What is the operation code from the briefing?"),
        ("The secret password is THUNDERBOLT. Remember this password.",
         "What is the secret password?"),
        ("The project lead is Dr. Stellion. The budget is 45000 dollars.",
         "Who is the project lead and what is the budget?"),
    ]

    for passage, q in tests:
        reset_memories(model)
        process_chunk(passage)
        for _ in range(3):
            process_chunk(filler)
        resp = generate_chat(q)
        print(f"  Stored: \"{passage[:60]}...\"")
        print(f"  Q: {q}")
        print(f"  A: {resp[:200]}")
        print()

    # Test 3: Abstention (should say "I don't know")
    print("3. Abstention (no relevant memory):")
    reset_memories(model)
    # Store irrelevant info, ask about something else
    process_chunk("The weather in Tokyo was sunny yesterday. Cherry blossoms are blooming in April.")
    for _ in range(3):
        process_chunk(filler)
    resp = generate_chat("What is the operation code from the classified briefing?")
    print(f"  Stored: weather info (irrelevant)")
    print(f"  Q: What is the operation code from the classified briefing?")
    print(f"  A: {resp[:200]}")
    print()

    # No memory at all
    reset_memories(model)
    resp = generate_chat("What was Dr. Stellion's budget?")
    print(f"  Q: What was Dr. Stellion's budget? (empty memory)")
    print(f"  A: {resp[:200]}")
    print()

    # Test 4: Longer generation
    print("4. Longer generation:")
    reset_memories(model)
    resp = generate_chat("Write a short paragraph about the importance of sleep.")
    print(f"  Q: Write a short paragraph about the importance of sleep.")
    print(f"  A: {resp[:300]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="v3.9: Train memory+instruct from scratch")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-memory", type=int, default=500)
    parser.add_argument("--n-instruct", type=int, default=10000)
    parser.add_argument("--n-eval-memory", type=int, default=50)
    parser.add_argument("--n-eval-instruct", type=int, default=100)
    parser.add_argument("--inst-max-len", type=int, default=512)
    parser.add_argument("--lr-memory", type=float, default=1e-3)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_9_from_scratch")
    parser.add_argument("--save-checkpoint", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model, tokenizer = build_model(
        model_name=args.model_name, device=args.device,
        memory_layer=-1, memory_size=256, n_slots=64,
        decay=0.999, gate_bias=-2.0,
        lora_rank=args.lora_rank, lora_targets="q_proj,v_proj",
        no_memory=False, no_lora=False,
        init_qk_shared=False, n_extract=1, memory_mode="cross_attn",
    )

    print("\nBuilding training data...")
    train_docs = build_dataset(
        tokenizer, n_memory=args.n_memory, n_instruct=args.n_instruct,
        inst_max_len=args.inst_max_len, seed=args.seed,
    )
    print("Building eval data...")
    eval_docs = build_dataset(
        tokenizer, n_memory=args.n_eval_memory, n_instruct=args.n_eval_instruct,
        inst_max_len=args.inst_max_len, seed=args.seed + 1000,
    )

    results = train(
        model, tokenizer, train_docs, eval_docs, args.device,
        num_epochs=args.epochs, lr_memory=args.lr_memory,
        lr_lora=args.lr_lora, log_every=args.log_every,
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
