"""Experiment v3.11: KL-divergence regularized memory on instruct model.

The most principled approach to instruct+memory: explicitly penalize
the model for diverging from the original instruct distribution.

  loss = memory_loss + beta * KL(adapted || original)

The instruct model already knows how to generate. The KL term protects
that ability. The memory loss teaches recall. Beta controls the tradeoff.

Key insight: PEFT's disable_adapter_layers() gives us the original
instruct output for free — no extra VRAM, just a second forward pass.
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


def build_dataset(tokenizer, n_memory=500, chunk_size=128,
                  gap_range=(2, 6), max_qa_pairs=3, seed=42):
    """Build memory recall dataset. No instruct data needed —
    KL penalty preserves instruct distribution automatically."""
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
            docs.append({
                "chunks": passage_chunks + filler_chunks + [qa_chunk],
                "answer_mask": answer_mask,
                "qa": qa,
            })

    rng.shuffle(docs)
    print(f"  Dataset: {len(docs)} docs")
    return docs


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, path):
    """Save full training state for resume."""
    state = {
        "model": {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(state, path)
    print(f"  Checkpoint saved: epoch {epoch + 1}, step {global_step} → {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load training state to resume."""
    state = torch.load(path, map_location=device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in state["model"]:
            param.data.copy_(state["model"][name])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"  Resumed from epoch {state['epoch'] + 1}, step {state['global_step']}")
    return state["epoch"], state["global_step"]


def compute_kl_loss(model, chunk_ids, answer_logits):
    """Compute KL divergence between adapted and original instruct model.

    Disables LoRA adapters to get original distribution, then computes
    KL(adapted || original) on all tokens.
    """
    # Get original instruct distribution (LoRA disabled)
    model.disable_adapter_layers()
    with torch.no_grad():
        original_output = model(chunk_ids)
        original_logits = original_output.logits[:, :-1]
    model.enable_adapter_layers()

    # KL divergence: adapted vs original
    adapted_log_probs = F.log_softmax(answer_logits, dim=-1)
    original_probs = F.softmax(original_logits, dim=-1)

    # KL(adapted || original) = sum(adapted * log(adapted/original))
    kl = F.kl_div(
        F.log_softmax(original_logits, dim=-1),  # target in log space
        adapted_log_probs,                         # input in log space
        log_target=True,
        reduction='batchmean',
    )
    return kl


def train(model, tokenizer, train_docs, eval_docs, device,
          num_epochs=15, lr_memory=1e-3, lr_lora=2e-4, beta=1.0,
          log_every=50, checkpoint_dir=None, resume_from=None):
    """Train with KL-regularized memory loss."""
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
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"  {total_steps} steps, {num_epochs} epochs, beta={beta}")
    model.train()
    global_step = 0
    start_epoch = 0
    losses = {"memory": [], "kl": []}
    start_time = time.time()

    if resume_from and os.path.exists(resume_from):
        start_epoch, global_step = load_checkpoint(
            model, optimizer, scheduler, resume_from, device
        )
        start_epoch += 1

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        order = list(range(len(train_docs)))
        random.shuffle(order)

        for doc_idx in order:
            doc = train_docs[doc_idx]
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)

            reset_memories(model)
            set_persistent_grad(model, True)

            # Process context chunks (accumulate memory)
            for chunk in chunks[:-1]:
                chunk_ids = chunk.unsqueeze(0).to(device)
                model(chunk_ids)

            detach_memory_state(model)

            # Forward QA chunk with LoRA (adapted model)
            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            set_persistent_grad(model, True)
            output = model(qa_chunk)

            # Memory loss (answer tokens only)
            logits = output.logits[:, :-1]
            flat_logits = logits.reshape(-1, logits.size(-1))
            targets = qa_chunk[:, 1:].reshape(-1)
            per_token = F.cross_entropy(flat_logits, targets, reduction='none')
            mask = answer_mask[:-1]
            n_tokens = mask.sum().clamp(min=1)
            memory_loss = (per_token * mask).sum() / n_tokens

            # KL loss (all tokens — preserve full distribution)
            kl_loss = compute_kl_loss(model, qa_chunk, logits)

            # Combined loss
            total_loss = memory_loss + beta * kl_loss

            if total_loss.item() > 0 and not torch.isnan(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()

            detach_memory_state(model)
            set_persistent_grad(model, False)

            losses["memory"].append(memory_loss.item())
            losses["kl"].append(kl_loss.item())
            global_step += 1

            if global_step % log_every == 0:
                mem_avg = sum(losses["memory"][-log_every:]) / min(len(losses["memory"]), log_every)
                kl_avg = sum(losses["kl"][-log_every:]) / min(len(losses["kl"]), log_every)
                elapsed = time.time() - start_time
                eta = elapsed / global_step * (total_steps - global_step)
                print(f"  step {global_step}/{total_steps}, "
                      f"mem={mem_avg:.3f}, kl={kl_avg:.4f}, "
                      f"total={mem_avg + beta * kl_avg:.3f}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, "
                      f"{elapsed:.0f}s (ETA {eta/3600:.1f}h)")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        eval_results = evaluate(model, tokenizer, eval_docs, device)
        print(f"  Memory recall: {eval_results['token_acc']:.1%} "
              f"({eval_results['correct']}/{eval_results['total']})")

        if checkpoint_dir:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                os.path.join(checkpoint_dir, "latest.pt"),
            )
            if not hasattr(train, '_best'):
                train._best = 0
            if eval_results['token_acc'] > train._best:
                train._best = eval_results['token_acc']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    os.path.join(checkpoint_dir, "best.pt"),
                )

        model.train()

    return evaluate(model, tokenizer, eval_docs, device)


def evaluate(model, tokenizer, eval_docs, device):
    """Evaluate memory recall."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)

            reset_memories(model)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                output = model(chunk_ids)

                if chunk_idx == len(chunks) - 1:
                    predictions = output.logits[0, :-1].argmax(dim=-1)
                    targets = chunk_ids[0, 1:]
                    mask = answer_mask[:-1]
                    positions = mask.nonzero(as_tuple=True)[0]
                    for p in positions:
                        total += 1
                        if predictions[p].item() == targets[p].item():
                            correct += 1

    return {"token_acc": correct / max(total, 1), "correct": correct, "total": total}


def test_generation(model, tokenizer, device):
    """Test instruct generation + memory recall."""
    model.eval()
    filler = "Modern computing has revolutionized information processing."

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

    print("3. Abstention (no relevant memory):")
    reset_memories(model)
    resp = generate_chat("What was Dr. Stellion's budget?")
    print(f"  Q: What was Dr. Stellion's budget? (empty memory)")
    print(f"  A: {resp[:200]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="v3.11: KL-regularized instruct+memory")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-memory", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL penalty weight. Higher = more instruct preservation")
    parser.add_argument("--lr-memory", type=float, default=1e-3)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_11_kl")
    parser.add_argument("--save-checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

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

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    print("\nBuilding training data...")
    train_docs = build_dataset(tokenizer, n_memory=args.n_memory, seed=args.seed)
    print("Building eval data...")
    eval_docs = build_dataset(tokenizer, n_memory=args.n_eval, seed=args.seed + 1000)

    results = train(
        model, tokenizer, train_docs, eval_docs, args.device,
        num_epochs=args.epochs, lr_memory=args.lr_memory,
        lr_lora=args.lr_lora, beta=args.beta,
        log_every=args.log_every, checkpoint_dir=checkpoint_dir,
        resume_from=args.resume,
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
