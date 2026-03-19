"""Experiment v3.7: Thinking module — recurrent deliberation over memory.

The model processes chunks and accumulates memory as before. But before
generating a response, it runs N "thinking steps" that cross-attend to
memory and update a recurrent thought state. This enables multi-step
reasoning: combining facts, following chains, making comparisons.

Architecture:
  1. Process passage chunks → memory accumulates (same as v3.2)
  2. Process filler chunks (same)
  3. Encode question → get hidden states at layer 15 → mean pool → thought seed
  4. ThinkingModule: N steps of cross-attend + GRU over memory
  5. ThoughtInjector: inject thought state at layer 16
  6. Full forward pass on QA chunk with thought injection → answer loss

Training data mix:
  - 40% simple retrieval (maintain memory skill)
  - 60% inference tasks (multi-hop, comparison, constraint, temporal, aggregation)
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
)
from rpvt.data.inference_tasks import generate_inference_tasks
from rpvt.model.thinking import ThinkingModule, ThoughtInjector


def get_layers(model):
    """Get the layers list from the model, handling peft wrapping."""
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        base = model.base_model.model
        return base.model.layers if hasattr(base, 'model') else base.layers
    elif hasattr(model, 'model'):
        return model.model.layers
    return model.layers


def attach_thinking(model, memory_bank, device, hidden_size,
                    n_think_steps=4, inner_dim=512, inject_layer=16):
    """Attach ThinkingModule and ThoughtInjector to the model.

    Returns:
        thinking_module: the ThinkingModule instance
        thought_injector: the ThoughtInjector wrapping inject_layer
    """
    thinking = ThinkingModule(
        hidden_size=hidden_size,
        n_think_steps=n_think_steps,
        inner_dim=inner_dim,
        n_heads=8,
        consolidate=False,  # start simple
    ).to(device=device, dtype=torch.bfloat16)

    layers = get_layers(model)
    injector = ThoughtInjector(
        layers[inject_layer], hidden_size
    ).to(device=device, dtype=torch.bfloat16)
    layers[inject_layer] = injector

    return thinking, injector


def get_hidden_at_layer(model, input_ids, target_layer=15):
    """Run forward pass and capture hidden states at a specific layer.

    Uses a forward hook to capture the output of the target layer.
    Returns hidden states without running the full model forward.
    """
    captured = {}

    layers = get_layers(model)
    target = layers[target_layer]

    # Get the actual layer (unwrap WriteWrapper etc)
    hook_target = target
    if hasattr(target, 'layer'):
        hook_target = target

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured['hidden'] = hidden.detach()

    handle = hook_target.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(input_ids)

    handle.remove()
    return captured.get('hidden')


def build_thinking_dataset(tokenizer, n_retrieval=200, n_inference=300,
                           chunk_size=128, gap_range=(2, 4), seed=42):
    """Build mixed dataset: simple retrieval + inference tasks.

    All tasks use the same format:
        passage chunks → filler chunks → QA chunk (chat format, answer-only loss)

    But inference tasks have MULTIPLE passage chunks that must be combined.
    """
    rng = random.Random(seed)

    # Load filler
    print("  Loading WikiText for filler...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    filler_texts = [t for t in wiki["text"] if len(t.strip()) > 100]

    def make_filler_chunk():
        ft = rng.choice(filler_texts)
        ft_tok = tokenizer.encode(ft, add_special_tokens=False)
        if len(ft_tok) >= chunk_size:
            start = rng.randint(0, len(ft_tok) - chunk_size)
            ct = ft_tok[start:start + chunk_size]
        else:
            ct = ft_tok + [tokenizer.eos_token_id or 0] * (chunk_size - len(ft_tok))
        return torch.tensor(ct, dtype=torch.long)

    def passage_to_chunks(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            ct = tokens[i:i + chunk_size]
            if len(ct) < chunk_size:
                ct = ct + [tokenizer.eos_token_id or 0] * (chunk_size - len(ct))
            chunks.append(torch.tensor(ct, dtype=torch.long))
        return chunks

    def make_qa_chunk(question, answer):
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

    all_docs = []

    # Simple retrieval tasks
    print(f"  Generating {n_retrieval} retrieval tasks...")
    retrieval_facts = _generate_natural_facts(rng, n_retrieval, max_qa_pairs=3)
    for passage, qa_pairs in retrieval_facts:
        p_chunks = passage_to_chunks(passage)
        gap = rng.randint(gap_range[0], gap_range[1])
        f_chunks = [make_filler_chunk() for _ in range(gap)]

        for qa in qa_pairs:
            qa_chunk, answer_mask = make_qa_chunk(qa["question"], qa["answer"])
            all_docs.append({
                "type": "retrieval",
                "chunks": p_chunks + f_chunks + [qa_chunk],
                "answer_mask": answer_mask,
                "qa": qa,
            })

    # Inference tasks
    print(f"  Generating {n_inference} inference tasks...")
    inference_data = generate_inference_tasks(rng, n_inference)
    for passages, qa_pairs in inference_data:
        # Each passage becomes its own chunk(s), with filler between
        all_chunks = []
        for passage in passages:
            all_chunks.extend(passage_to_chunks(passage))
            # Add 1-2 filler chunks between passages
            for _ in range(rng.randint(1, 2)):
                all_chunks.append(make_filler_chunk())

        for qa in qa_pairs:
            qa_chunk, answer_mask = make_qa_chunk(qa["question"], qa["answer"])
            all_docs.append({
                "type": qa.get("type", "inference"),
                "chunks": all_chunks + [qa_chunk],
                "answer_mask": answer_mask,
                "qa": qa,
            })

    rng.shuffle(all_docs)

    type_counts = {}
    for d in all_docs:
        type_counts[d["type"]] = type_counts.get(d["type"], 0) + 1
    print(f"  Dataset: {len(all_docs)} docs — {type_counts}")
    return all_docs


def train(model, thinking_module, thought_injector, tokenizer,
          train_docs, eval_docs, device, memory_bank,
          num_epochs=15, lr=1e-3, lr_lora=2e-4, log_every=50,
          think_layer=15):
    """Train memory + thinking module end-to-end."""

    # Separate param groups
    thinking_params = list(thinking_module.parameters()) + list(thought_injector.parameters())
    memory_params = list(memory_bank.parameters())
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora" in n.lower()]

    param_groups = [
        {"params": thinking_params + memory_params, "lr": lr},
    ]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora})

    n_think = sum(p.numel() for p in thinking_params)
    n_mem = sum(p.numel() for p in memory_params)
    n_lora = sum(p.numel() for p in lora_params)
    total = n_think + n_mem + n_lora
    print(f"\nTraining {total:,} params (thinking={n_think:,}, memory={n_mem:,}, lora={n_lora:,})")

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
    thinking_module.train()
    global_step = 0
    losses_by_type = {}
    start_time = time.time()

    for epoch in range(num_epochs):
        order = list(range(len(train_docs)))
        random.shuffle(order)

        for doc_idx in order:
            doc = train_docs[doc_idx]
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)
            doc_type = doc["type"]

            reset_memories(model)
            set_persistent_grad(model, True)

            # Process all chunks except the last (QA chunk)
            for chunk in chunks[:-1]:
                chunk_ids = chunk.unsqueeze(0).to(device)
                with torch.no_grad():
                    model(chunk_ids)

            detach_memory_state(model)

            # Get thought seed: hidden states at think_layer for QA chunk
            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            hidden = get_hidden_at_layer(model, qa_chunk, target_layer=think_layer)

            if hidden is not None:
                # Mean pool as thought seed
                thought_seed = hidden.mean(dim=1).squeeze(0)  # (hidden_size,)

                # Run thinking
                thought, think_states = thinking_module(thought_seed, memory_bank)

                # Inject thought for generation
                thought_injector.set_thought(thought)

            # Forward pass with thought injection
            set_persistent_grad(model, True)
            output = model(qa_chunk)

            # Answer-only loss
            logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
            targets = qa_chunk[:, 1:].reshape(-1)
            per_token = F.cross_entropy(logits, targets, reduction='none')
            mask = answer_mask[:-1]
            n_tokens = mask.sum().clamp(min=1)
            loss = (per_token * mask).sum() / n_tokens

            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                all_trainable = thinking_params + memory_params + lora_params
                torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
                optimizer.step()
                scheduler.step()

            thought_injector.clear_thought()
            detach_memory_state(model)
            set_persistent_grad(model, False)

            # Track losses by type
            if doc_type not in losses_by_type:
                losses_by_type[doc_type] = []
            losses_by_type[doc_type].append(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                elapsed = time.time() - start_time
                parts = []
                for t, vals in sorted(losses_by_type.items()):
                    recent = vals[-log_every:]
                    avg = sum(recent) / max(len(recent), 1)
                    parts.append(f"{t}={avg:.3f}")
                loss_str = ", ".join(parts)
                print(f"  step {global_step}/{total_steps}, {loss_str}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, {elapsed:.0f}s")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        eval_results = evaluate(model, thinking_module, thought_injector,
                                tokenizer, eval_docs, device, memory_bank,
                                think_layer=think_layer)
        for task_type, metrics in sorted(eval_results.items()):
            if task_type == "overall":
                print(f"  Overall: {metrics['token_acc']:.1%} "
                      f"({metrics['correct']}/{metrics['total']})")
            else:
                print(f"  {task_type}: {metrics['token_acc']:.1%} "
                      f"({metrics['correct']}/{metrics['total']})")
        model.train()
        thinking_module.train()

    return eval_results


def evaluate(model, thinking_module, thought_injector, tokenizer,
             eval_docs, device, memory_bank, think_layer=15):
    """Evaluate by task type."""
    model.eval()
    thinking_module.eval()

    results_by_type = {}

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)
            doc_type = doc["type"]

            reset_memories(model)

            # Process context chunks
            for chunk in chunks[:-1]:
                chunk_ids = chunk.unsqueeze(0).to(device)
                model(chunk_ids)

            # Think
            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            hidden = get_hidden_at_layer(model, qa_chunk, target_layer=think_layer)

            if hidden is not None:
                thought_seed = hidden.mean(dim=1).squeeze(0)
                thought, _ = thinking_module(thought_seed, memory_bank)
                thought_injector.set_thought(thought)

            # Generate
            output = model(qa_chunk)
            thought_injector.clear_thought()

            # Score
            predictions = output.logits[0, :-1].argmax(dim=-1)
            targets = qa_chunk[0, 1:]
            mask = answer_mask[:-1]
            positions = mask.nonzero(as_tuple=True)[0]

            correct = 0
            total = 0
            for p in positions:
                total += 1
                if predictions[p].item() == targets[p].item():
                    correct += 1

            if doc_type not in results_by_type:
                results_by_type[doc_type] = {"correct": 0, "total": 0}
            results_by_type[doc_type]["correct"] += correct
            results_by_type[doc_type]["total"] += total

    # Compute accuracies
    overall_correct = 0
    overall_total = 0
    for task_type in results_by_type:
        r = results_by_type[task_type]
        r["token_acc"] = r["correct"] / max(r["total"], 1)
        overall_correct += r["correct"]
        overall_total += r["total"]

    results_by_type["overall"] = {
        "correct": overall_correct,
        "total": overall_total,
        "token_acc": overall_correct / max(overall_total, 1),
    }

    return results_by_type


def test_generation(model, thinking_module, thought_injector, tokenizer,
                    device, memory_bank, think_layer=15):
    """Test thinking + generation on hand-crafted examples."""
    model.eval()
    thinking_module.eval()

    def process_chunk(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:128]
        if len(tokens) < 128:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (128 - len(tokens))
        with torch.no_grad():
            model(torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device))

    def think_and_generate(question, max_new=80):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Get thought seed
        with torch.no_grad():
            hidden = get_hidden_at_layer(model, input_ids, target_layer=think_layer)
            if hidden is not None:
                thought_seed = hidden.mean(dim=1).squeeze(0)
                thought, think_states = thinking_module(thought_seed, memory_bank)
                thought_injector.set_thought(thought)

            out = model.generate(
                input_ids, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        thought_injector.clear_thought()
        return tokenizer.decode(out[0][len(tokens):], skip_special_tokens=True)

    filler = "Modern computing has revolutionized information processing."

    print("\n=== THINKING GENERATION TESTS ===\n")

    # Test 1: Simple retrieval (should still work)
    print("1. Simple retrieval:")
    reset_memories(model)
    process_chunk("Agent Blackwood's operation code is VIPER-371. He is stationed at Nordheim.")
    for _ in range(3):
        process_chunk(filler)
    resp = think_and_generate("What is Agent Blackwood's operation code?")
    print(f"  A: {resp[:150]}\n")

    # Test 2: Multi-hop reasoning
    print("2. Multi-hop (A→B→C):")
    reset_memories(model)
    process_chunk("Dr. Elena Vasquez works at Helios Labs as the lead researcher.")
    process_chunk(filler)
    process_chunk("Helios Labs is headquartered in Tokyo, Japan.")
    for _ in range(2):
        process_chunk(filler)
    resp = think_and_generate("What city does Dr. Vasquez work in?")
    print(f"  A: {resp[:150]}\n")

    # Test 3: Comparison
    print("3. Comparison:")
    reset_memories(model)
    process_chunk("Project Alpha has a budget of 45000 dollars and 8 team members.")
    process_chunk(filler)
    process_chunk("Project Beta has a budget of 72000 dollars and 5 team members.")
    for _ in range(2):
        process_chunk(filler)
    resp = think_and_generate("Which project has a larger budget?")
    print(f"  A: {resp[:150]}\n")

    # Test 4: Constraint
    print("4. Constraint satisfaction:")
    reset_memories(model)
    process_chunk("Alice has a severe allergy to peanuts. She must avoid all peanut products.")
    process_chunk(filler)
    process_chunk("The Thai restaurant's signature dish is pad thai with peanut sauce.")
    for _ in range(2):
        process_chunk(filler)
    resp = think_and_generate("Can Alice safely eat at the Thai restaurant?")
    print(f"  A: {resp[:150]}\n")

    # Test 5: Basic instruct (no memory, should be perfect)
    print("5. Basic instruct (no memory):")
    reset_memories(model)
    resp = think_and_generate("What is the capital of France?")
    print(f"  A: {resp[:150]}\n")


def main():
    parser = argparse.ArgumentParser(description="v3.7: Thinking module")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-retrieval", type=int, default=200)
    parser.add_argument("--n-inference", type=int, default=300)
    parser.add_argument("--n-eval-retrieval", type=int, default=30)
    parser.add_argument("--n-eval-inference", type=int, default=50)
    parser.add_argument("--n-think-steps", type=int, default=4)
    parser.add_argument("--think-inner-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_7_thinking")
    parser.add_argument("--save-checkpoint", type=str, default=None)
    parser.add_argument("--load-checkpoint", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Build model with cross-attention memory (standard approach)
    model, tokenizer = build_model(
        model_name=args.model_name, device=args.device,
        memory_layer=-1, memory_size=256, n_slots=64,
        decay=0.999, gate_bias=-2.0,
        lora_rank=args.lora_rank, lora_targets="q_proj,v_proj",
        no_memory=False, no_lora=False,
        init_qk_shared=False, n_extract=1,
        memory_mode="cross_attn",
    )

    # Get memory bank reference
    memory_bank = _get_memory_module(model)
    hidden_size = model.config.hidden_size if hasattr(model, 'config') else 1536

    # Attach thinking module
    # Think layer = 15 (same as memory read layer — thought seed from memory-augmented representations)
    # Inject layer = 16 (layer after memory read)
    thinking_module, thought_injector = attach_thinking(
        model, memory_bank, args.device, hidden_size,
        n_think_steps=args.n_think_steps,
        inner_dim=args.think_inner_dim,
        inject_layer=16,
    )

    n_think = sum(p.numel() for p in thinking_module.parameters())
    n_inject = sum(p.numel() for p in thought_injector.parameters())
    print(f"  Thinking module: {n_think:,} params")
    print(f"  Thought injector: {n_inject:,} params")

    # Optionally load pre-trained memory checkpoint
    if args.load_checkpoint:
        print(f"\nLoading checkpoint: {args.load_checkpoint}")
        state = torch.load(args.load_checkpoint, map_location=args.device)
        missing = []
        for name, param in model.named_parameters():
            if name in state:
                param.data.copy_(state[name])
            elif param.requires_grad:
                missing.append(name)
        if missing:
            print(f"  {len(missing)} params not in checkpoint (new modules)")

    print("\nBuilding training data...")
    train_docs = build_thinking_dataset(
        tokenizer, n_retrieval=args.n_retrieval, n_inference=args.n_inference,
        seed=args.seed,
    )
    print("Building eval data...")
    eval_docs = build_thinking_dataset(
        tokenizer, n_retrieval=args.n_eval_retrieval, n_inference=args.n_eval_inference,
        seed=args.seed + 1000,
    )

    results = train(
        model, thinking_module, thought_injector, tokenizer,
        train_docs, eval_docs, args.device, memory_bank,
        num_epochs=args.epochs, lr=args.lr, lr_lora=args.lr_lora,
        log_every=args.log_every,
    )

    if args.save_checkpoint:
        os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
        state = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                state[n] = p.data.clone()
        for n, p in thinking_module.named_parameters():
            state[f"thinking.{n}"] = p.data.clone()
        for n, p in thought_injector.named_parameters():
            if "layer" not in n:  # don't save the wrapped layer
                state[f"injector.{n}"] = p.data.clone()
        torch.save(state, args.save_checkpoint)
        print(f"\nCheckpoint saved to {args.save_checkpoint}")

    test_generation(model, thinking_module, thought_injector,
                    tokenizer, args.device, memory_bank)

    # Save results
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: vv for kk, vv in v.items()}
    serializable["config"] = vars(args)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
