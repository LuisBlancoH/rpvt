"""Experiment v3.4: Memory for instruct models.

Trains memory on an instruct-tuned model using chat template formatting.
Passage chunks are plain text (processed through memory), QA chunks use
the model's chat template so generation ability is preserved.

Key differences from v3.2:
  - QA chunk uses chat template (<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...)
  - Loss only on assistant response tokens (qa_only_loss by default)
  - Supports both recall and instruction-following training data
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    _random_name,
    _random_word,
    _generate_natural_facts,
    _generate_instruction_following,
)


class ChatMemoryDataset:
    """Dataset with chat-formatted QA chunks for instruct models.

    Passage/filler chunks are plain text (for memory write).
    QA chunks use the model's chat template for proper generation.
    """

    def __init__(self, tokenizer, n_docs=500, chunk_size=128,
                 gap_range=(2, 6), max_qa_pairs=3, seed=42,
                 recall_ratio=0.6):
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer

        rng = random.Random(seed)

        # Generate mixed data
        n_recall = int(n_docs * recall_ratio)
        n_instruct = n_docs - n_recall

        recall_docs = _generate_natural_facts(rng, n_recall, max_qa_pairs)
        instruct_docs = _generate_instruction_following(rng, n_instruct)

        all_docs = recall_docs + instruct_docs
        rng.shuffle(all_docs)

        # Load filler
        from datasets import load_dataset
        print("  Loading WikiText for filler...")
        wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        filler_texts = [t for t in wiki["text"] if len(t.strip()) > 100]

        self.documents = []
        for passage, qa_pairs in all_docs:
            gap = rng.randint(gap_range[0], gap_range[1])

            # Passage chunks (plain text)
            passage_tokens = tokenizer.encode(passage, add_special_tokens=False)
            passage_chunks = []
            for i in range(0, len(passage_tokens), chunk_size):
                ct = passage_tokens[i:i + chunk_size]
                if len(ct) < chunk_size:
                    ct = ct + [tokenizer.eos_token_id or 0] * (chunk_size - len(ct))
                passage_chunks.append(torch.tensor(ct, dtype=torch.long))

            # Filler chunks (plain text)
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

            # QA chunk in CHAT FORMAT
            for qa in qa_pairs:
                question = qa["question"]
                answer = qa["answer"]

                # Format as chat
                messages = [
                    {"role": "user", "content": question},
                ]
                chat_prefix = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # Full text = chat_prefix + answer + end token
                full_text = chat_prefix + answer + "<|im_end|>"

                full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
                prefix_tokens = tokenizer.encode(chat_prefix, add_special_tokens=False)
                answer_start = len(prefix_tokens)

                # Build answer mask
                answer_mask = torch.zeros(chunk_size, dtype=torch.float32)
                for pos in range(max(0, answer_start - 1), min(len(full_tokens) - 1, chunk_size)):
                    answer_mask[pos] = 1.0

                if len(full_tokens) >= chunk_size:
                    full_tokens = full_tokens[:chunk_size]
                else:
                    full_tokens = full_tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(full_tokens))

                qa_chunk = torch.tensor(full_tokens, dtype=torch.long)

                all_chunks = passage_chunks + filler_chunks + [qa_chunk]

                self.documents.append({
                    "chunks": all_chunks,
                    "answer_mask": answer_mask,
                    "qa_pairs": [qa],
                    "gap": gap,
                    "n_passage_chunks": len(passage_chunks),
                    "context": passage,
                })

        print(f"  ChatMemoryDataset: {len(self.documents)} docs, "
              f"gap range {gap_range}, chunk_size={chunk_size}")


def train_and_eval_chat(
    model, tokenizer, train_dataset, eval_dataset, device,
    num_epochs=10, lr_memory=1e-3, lr_lora=2e-4,
    log_every=50, output_dir="results",
):
    """Train with answer-only loss on chat-formatted QA chunks."""
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

    if not param_groups:
        print("  WARNING: No trainable parameters!")
        return eval_chat(model, tokenizer, eval_dataset, device)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    total_steps = len(train_dataset.documents) * num_epochs

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"\nTraining: {total_steps} steps, {num_epochs} epochs")
    model.train()
    global_step = 0
    train_losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_order = list(range(len(train_dataset.documents)))
        random.shuffle(epoch_order)

        for doc_idx in epoch_order:
            doc = train_dataset.documents[doc_idx]
            chunks = doc["chunks"]
            n_chunks = len(chunks)
            answer_mask = doc["answer_mask"].to(device)

            reset_memories(model)
            set_persistent_grad(model, True)

            # Process all chunks
            doc_loss = torch.tensor(0.0, device=device)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                is_qa_chunk = (chunk_idx == n_chunks - 1)

                output = model(chunk_ids)

                if is_qa_chunk:
                    # Answer-only loss
                    logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
                    targets = chunk_ids[:, 1:].reshape(-1)
                    per_token_loss = F.cross_entropy(logits, targets, reduction='none')
                    mask = answer_mask[:-1]
                    n_answer = mask.sum().clamp(min=1)
                    doc_loss = (per_token_loss * mask).sum() / n_answer
                # No loss on passage/filler — preserves instruct behavior

            if doc_loss.item() > 0:
                optimizer.zero_grad()
                doc_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()

            detach_memory_state(model)
            set_persistent_grad(model, False)

            train_losses.append(doc_loss.item())
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = sum(train_losses[-log_every:]) / log_every
                elapsed = time.time() - start_time
                print(f"  step {global_step}/{total_steps}, "
                      f"loss={avg_loss:.4f}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, {elapsed:.0f}s")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} eval ===")
        eval_results = eval_chat(model, tokenizer, eval_dataset, device, verbose=False)
        print(f"  Token accuracy: {eval_results['token_accuracy']:.1%}, "
              f"Exact match: {eval_results['exact_match']:.1%}")
        model.train()

    return eval_chat(model, tokenizer, eval_dataset, device, verbose=True)


def eval_chat(model, tokenizer, dataset, device, verbose=True, n_debug=10):
    """Evaluate recall accuracy AND generation quality."""
    model.eval()

    total_correct = 0
    total_tokens = 0
    total_pairs = 0
    exact_matches = 0

    with torch.no_grad():
        for doc_idx, doc in enumerate(dataset.documents):
            chunks = doc["chunks"]
            n_chunks = len(chunks)
            answer_mask = doc["answer_mask"]

            reset_memories(model)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                output = model(chunk_ids)

                if chunk_idx == n_chunks - 1:
                    logits = output.logits[0]
                    predictions = logits[:-1].argmax(dim=-1)
                    targets = chunk_ids[0, 1:]
                    mask = answer_mask[:-1]
                    positions = mask.nonzero(as_tuple=True)[0]

                    correct = sum(1 for p in positions
                                  if predictions[p].item() == targets[p].item())
                    total_correct += correct
                    total_tokens += len(positions)
                    total_pairs += 1
                    if len(positions) > 0 and correct == len(positions):
                        exact_matches += 1

                    if verbose and doc_idx < n_debug:
                        qa = doc["qa_pairs"][0]
                        pred_text = tokenizer.decode(
                            [predictions[p].item() for p in positions[:10]])
                        exp_text = tokenizer.decode(
                            [targets[p].item() for p in positions[:10]])
                        print(f"\n  [{doc_idx}] Q: {qa['question'][:60]}")
                        print(f"    Expected: {exp_text[:60]}")
                        print(f"    Predicted: {pred_text[:60]}")

    token_acc = total_correct / max(total_tokens, 1)
    exact_match = exact_matches / max(total_pairs, 1)

    if verbose:
        print(f"\n  Token accuracy: {total_correct}/{total_tokens} = {token_acc:.1%}")
        print(f"  Exact match: {exact_matches}/{total_pairs} = {exact_match:.1%}")

    return {
        "token_accuracy": token_acc,
        "exact_match": exact_match,
        "correct_tokens": total_correct,
        "total_tokens": total_tokens,
    }


def test_generation(model, tokenizer, device, checkpoint_path=None):
    """Test free-form generation with memory."""
    model.eval()

    filler = ("Modern computing has revolutionized how we process information. "
              "Algorithms handle billions of operations per second.")

    def process_chunk(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:128]
        if len(tokens) < 128:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (128 - len(tokens))
        with torch.no_grad():
            model(torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device))

    def generate(question, max_new=60):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_ids)
            generated = list(tokens)
            for _ in range(max_new):
                next_token = output.logits[0, -1].argmax().item()
                generated.append(next_token)
                if next_token == tokenizer.eos_token_id:
                    break
                # Check for im_end token
                if tokenizer.decode([next_token]) == "<|im_end|>":
                    break
                output = model(torch.tensor([[next_token]], dtype=torch.long).to(device))
        return tokenizer.decode(generated[len(tokens):], skip_special_tokens=True)

    print("\n=== GENERATION TESTS ===\n")

    tests = [
        ("Classified briefing: The operation code is VIPER-371. Agent Blackwood is stationed at Nordheim.",
         "What is the operation code from the briefing?"),
        ("The secret password is THUNDERBOLT. Remember this password.",
         "What is the secret password?"),
        ("Meeting notes: The project lead is Dr. Stellion. The budget is 45000 dollars.",
         "Who is the project lead?"),
        ("You are Atlas, a navigator from the city of Zephyria.",
         "What is your name?"),
    ]

    for instruction, question in tests:
        reset_memories(model)
        process_chunk(instruction)
        for _ in range(3):
            process_chunk(filler)
        resp = generate(question)
        print(f"Inst: \"{instruction[:55]}...\"")
        print(f"Q: {question}")
        print(f"A: {resp[:100]}")
        print()

    # Control: basic generation without memory
    print("=== CONTROL: generation without memory ===\n")
    reset_memories(model)
    for _ in range(4):
        process_chunk(filler)
    resp = generate("What is the capital of France?")
    print(f"Q: What is the capital of France?")
    print(f"A: {resp[:100]}")


def main():
    parser = argparse.ArgumentParser(description="v3.4: Memory for instruct models")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--memory-mode", type=str, default="cross_attn")
    parser.add_argument("--n-slots", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-targets", type=str, default="q_proj,v_proj")
    parser.add_argument("--lora-layers", type=str, default=None)
    parser.add_argument("--n-extract", type=int, default=1)
    parser.add_argument("--mem-proj", action="store_true",
                        help="Add dedicated memory projection layer (no LoRA needed)")
    parser.add_argument("--no-lora", action="store_true")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr-memory", type=float, default=1e-3)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--max-qa-pairs", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_4_instruct")

    parser.add_argument("--save-checkpoint", type=str, default=None)
    parser.add_argument("--load-checkpoint", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model, tokenizer = build_model(
        model_name=args.model_name,
        device=args.device,
        memory_layer=-1,
        memory_size=256,
        n_slots=args.n_slots,
        decay=0.999,
        gate_bias=-2.0,
        lora_rank=args.lora_rank,
        lora_targets=args.lora_targets,
        no_memory=False,
        no_lora=args.no_lora,
        init_qk_shared=False,
        n_extract=args.n_extract,
        memory_mode=args.memory_mode,
        lora_layers=[int(x) for x in args.lora_layers.split(",")] if args.lora_layers else None,
        mem_proj=args.mem_proj,
    )

    if args.load_checkpoint:
        print(f"\nLoading checkpoint: {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, map_location=args.device, weights_only=True)
        state = model.state_dict()
        loaded = sum(1 for n, p in ckpt.items() if n in state and not state[n].copy_(p) is None)
        print(f"  Loaded {loaded} tensors")
        test_generation(model, tokenizer, args.device)
        return

    print(f"\nCreating datasets (seed={args.seed})...")
    train_data = ChatMemoryDataset(
        tokenizer, n_docs=args.n_train, chunk_size=128,
        gap_range=(2, 6), max_qa_pairs=args.max_qa_pairs,
        seed=args.seed,
    )
    eval_data = ChatMemoryDataset(
        tokenizer, n_docs=args.n_eval, chunk_size=128,
        gap_range=(2, 6), max_qa_pairs=args.max_qa_pairs,
        seed=args.seed + 1000,
    )

    results = train_and_eval_chat(
        model, tokenizer, train_data, eval_data, args.device,
        num_epochs=args.epochs,
        lr_memory=args.lr_memory,
        lr_lora=args.lr_lora,
        log_every=args.log_every,
        output_dir=args.output_dir,
    )

    if args.save_checkpoint:
        state = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(state, args.save_checkpoint)
        print(f"\nCheckpoint saved to {args.save_checkpoint}")

    # Test generation
    test_generation(model, tokenizer, args.device)

    results["config"] = vars(args)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
