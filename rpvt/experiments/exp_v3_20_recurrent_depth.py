"""v3.20: Recurrent depth — loop upper layers for deeper reasoning.

Tests whether running the upper layers of a transformer multiple times
(with a learned projection back) gives the model more reasoning depth.

Comparison:
- n_loops=1: normal forward pass (baseline)
- n_loops=2: 2× depth through upper layers
- n_loops=3: 3× depth

Evaluated on:
1. Simple reasoning tasks (reverse string, arithmetic, etc.)
2. Data exploration tasks (parse structured text)

Usage:
    python -m rpvt.experiments.exp_v3_20_recurrent_depth
    python -m rpvt.experiments.exp_v3_20_recurrent_depth --n-loops 3
    python -m rpvt.experiments.exp_v3_20_recurrent_depth --eval-only --load-checkpoint results/recurrent_depth/checkpoint_epoch5.pt
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.recurrent_depth import RecurrentDepthWrapper


# ── Task generators ────────────────────────────────────────────

def generate_reasoning_tasks(n, tokenizer, seed=42):
    """Generate simple reasoning tasks that benefit from deeper thinking."""
    rng = random.Random(seed)
    tasks = []

    for i in range(n):
        task_type = rng.choice([
            "reverse", "count", "extract", "compare", "multi_step",
        ])

        if task_type == "reverse":
            word = ''.join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(4, 8)))
            prompt = f"Reverse the string '{word}'. Answer with just the reversed string."
            answer = word[::-1]

        elif task_type == "count":
            letter = rng.choice("aeiou")
            word = ''.join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(8, 15)))
            prompt = f"How many times does '{letter}' appear in '{word}'? Answer with just the number."
            answer = str(word.count(letter))

        elif task_type == "extract":
            # Extract specific info from structured text
            names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank",
                     "Grace", "Henry", "Ivy", "Jack"]
            name = rng.choice(names)
            age = rng.randint(20, 60)
            city = rng.choice(["NYC", "LA", "Chicago", "Houston", "Phoenix"])
            job = rng.choice(["engineer", "teacher", "doctor", "artist", "chef"])
            text = f"Name: {name}, Age: {age}, City: {city}, Job: {job}"
            question = rng.choice(["age", "city", "job"])
            if question == "age":
                prompt = f"Given: {text}\nWhat is {name}'s age? Answer with just the number."
                answer = str(age)
            elif question == "city":
                prompt = f"Given: {text}\nWhat city does {name} live in? Answer with just the city."
                answer = city
            else:
                prompt = f"Given: {text}\nWhat is {name}'s job? Answer with just the job."
                answer = job

        elif task_type == "compare":
            a, b = rng.randint(10, 999), rng.randint(10, 999)
            prompt = f"Which is larger, {a} or {b}? Answer with just the number."
            answer = str(max(a, b))

        elif task_type == "multi_step":
            # Multi-step: read data, filter, answer
            items = []
            for _ in range(rng.randint(4, 7)):
                item_name = ''.join(rng.choices("ABCDEFGH", k=3))
                item_year = rng.randint(1990, 2025)
                item_type = rng.choice(["book", "movie", "game"])
                items.append((item_name, item_year, item_type))

            target_type = rng.choice(["book", "movie", "game"])
            # Find oldest of target type
            matching = [(n, y) for n, y, t in items if t == target_type]
            if matching:
                oldest = min(matching, key=lambda x: x[1])
                answer = oldest[0]
            else:
                answer = "none"

            items_text = "\n".join(f"- {n} ({y}, {t})" for n, y, t in items)
            prompt = (f"Items:\n{items_text}\n\n"
                      f"What is the name of the oldest {target_type}? "
                      f"Answer with just the name.")

        tasks.append({"prompt": prompt, "answer": answer, "type": task_type})

    return tasks


# ── Training ────────────────────────────────────────────────────

def train_epoch(model, wrapper, tokenizer, tasks, optimizer, device,
                max_length=256):
    """Train one epoch on reasoning tasks."""
    wrapper.train()
    total_loss = 0
    correct = 0
    total = 0

    indices = list(range(len(tasks)))
    random.shuffle(indices)

    for idx in indices:
        task = tasks[idx]
        # Format as chat
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task["prompt"]},
                {"role": "assistant", "content": task["answer"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
        input_ids = tokens["input_ids"]

        # Build labels: mask everything except the answer
        labels = input_ids.clone()
        # Find the assistant answer tokens
        answer_tokens = tokenizer(task["answer"], add_special_tokens=False)["input_ids"]
        answer_len = len(answer_tokens)
        # Mask everything except the last answer_len tokens (+ eos)
        labels[:, :-(answer_len + 1)] = -100

        out = wrapper(input_ids, labels=labels)
        loss = out.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in wrapper.parameters() if p.requires_grad] +
            [p for p in model.parameters() if p.requires_grad],
            1.0,
        )
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total += 1

        # Quick accuracy check: does greedy generation match?
        if total <= 50 or total % 20 == 0:
            with torch.no_grad():
                eval_tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": task["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                eval_ids = tokenizer(eval_tokens, return_tensors="pt").to(device)
                gen = wrapper.generate_with_loops(
                    eval_ids["input_ids"],
                    attention_mask=eval_ids["attention_mask"],
                    max_new_tokens=30,
                )
                pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
                if pred.lower() == task["answer"].lower():
                    correct += 1

    return total_loss / max(total, 1), correct, total


def evaluate(wrapper, tokenizer, tasks, device, n_loops_list=None):
    """Evaluate on reasoning tasks, optionally comparing different n_loops."""
    wrapper.eval()

    if n_loops_list is None:
        n_loops_list = [1, wrapper.n_loops]

    results = {}
    for n_loops in n_loops_list:
        correct = 0
        by_type = {}

        for task in tasks:
            eval_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            eval_ids = tokenizer(eval_tokens, return_tensors="pt").to(device)

            with torch.no_grad():
                gen = wrapper.generate_with_loops(
                    eval_ids["input_ids"],
                    attention_mask=eval_ids["attention_mask"],
                    max_new_tokens=30,
                    n_loops=n_loops,
                )
            pred = tokenizer.decode(gen, skip_special_tokens=True).strip()

            is_correct = pred.lower().strip() == task["answer"].lower().strip()
            if is_correct:
                correct += 1

            task_type = task["type"]
            if task_type not in by_type:
                by_type[task_type] = {"correct": 0, "total": 0}
            by_type[task_type]["total"] += 1
            if is_correct:
                by_type[task_type]["correct"] += 1

        results[n_loops] = {
            "accuracy": correct / max(len(tasks), 1),
            "correct": correct,
            "total": len(tasks),
            "by_type": by_type,
        }

    return results


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="v3.20: Recurrent Depth")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--split-layer", type=int, default=14)
    parser.add_argument("--n-loops", type=int, default=2)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-targets", type=str, default="q_proj,v_proj")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--output-dir", type=str,
                        default="results/recurrent_depth")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--save-checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Freeze base
    for p in model.parameters():
        p.requires_grad = False

    # Apply LoRA to upper layers only (the looped layers)
    n_layers = model.config.num_hidden_layers
    lora_layers = list(range(args.split_layer, n_layers))
    targets = [t.strip() for t in args.lora_targets.split(",")]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=targets,
        layers_to_transform=lora_layers,
    )
    model = get_peft_model(model, lora_config)
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA: rank={args.lora_rank}, layers={lora_layers}, params={n_lora:,}")

    # Wrap with recurrent depth
    wrapper = RecurrentDepthWrapper(
        model,
        split_layer=args.split_layer,
        n_loops=args.n_loops,
        residual_scale=args.residual_scale,
    )

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        # Load recurrence params
        wrapper.project_back.load_state_dict(ckpt["project_back"])
        wrapper.loop_embed.load_state_dict(ckpt["loop_embed"])
        # Load LoRA params
        if "lora_state_dict" in ckpt:
            for name, param in model.named_parameters():
                if name in ckpt["lora_state_dict"]:
                    param.data.copy_(ckpt["lora_state_dict"][name])

    # Generate tasks
    print(f"\nGenerating tasks: {args.n_train} train, {args.n_eval} eval")
    train_tasks = generate_reasoning_tasks(args.n_train, tokenizer, seed=42)
    eval_tasks = generate_reasoning_tasks(args.n_eval, tokenizer, seed=123)

    # Count by type
    from collections import Counter
    type_counts = Counter(t["type"] for t in train_tasks)
    print(f"  Types: {dict(type_counts)}")

    # Baseline evaluation (before training)
    print("\n--- Baseline (before training) ---")
    baseline = evaluate(wrapper, tokenizer, eval_tasks[:20], device,
                        n_loops_list=[1, args.n_loops])
    for nl, res in baseline.items():
        print(f"  n_loops={nl}: {res['correct']}/{res['total']} "
              f"({100*res['accuracy']:.1f}%)")

    if args.eval_only:
        print("\n--- Full evaluation ---")
        results = evaluate(wrapper, tokenizer, eval_tasks, device,
                           n_loops_list=[1, args.n_loops])
        for nl, res in results.items():
            print(f"\n  n_loops={nl}: {res['correct']}/{res['total']} "
                  f"({100*res['accuracy']:.1f}%)")
            for t, r in res["by_type"].items():
                print(f"    {t}: {r['correct']}/{r['total']} "
                      f"({100*r['correct']/max(r['total'],1):.0f}%)")
        return

    # Optimizer: LoRA + recurrence params
    all_params = [
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": args.lr},
        {"params": wrapper.get_trainable_params(),
         "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(all_params, weight_decay=0.01)

    # Training loop
    print(f"\n--- Training: {args.epochs} epochs ---")
    for epoch in range(args.epochs):
        t0 = time.time()
        loss, correct, total = train_epoch(
            model, wrapper, tokenizer, train_tasks, optimizer, device,
        )
        elapsed = time.time() - t0

        # Evaluate
        eval_results = evaluate(
            wrapper, tokenizer, eval_tasks[:20], device,
            n_loops_list=[1, args.n_loops],
        )

        loop1 = eval_results[1]
        loopN = eval_results[args.n_loops]
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={loss:.4f}, "
              f"1-loop={100*loop1['accuracy']:.1f}%, "
              f"{args.n_loops}-loop={100*loopN['accuracy']:.1f}%, "
              f"({elapsed:.0f}s)")

        # Save checkpoint
        ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
        lora_sd = {
            k: v.cpu() for k, v in model.named_parameters() if v.requires_grad
        }
        torch.save({
            "epoch": epoch + 1,
            "project_back": wrapper.project_back.state_dict(),
            "loop_embed": wrapper.loop_embed.state_dict(),
            "lora_state_dict": lora_sd,
            "args": vars(args),
            "eval_results": eval_results,
        }, ckpt_path)

    # Final evaluation
    print(f"\n--- Final evaluation ({args.n_eval} tasks) ---")
    final = evaluate(wrapper, tokenizer, eval_tasks, device,
                     n_loops_list=[1, args.n_loops])
    for nl, res in final.items():
        print(f"\n  n_loops={nl}: {res['correct']}/{res['total']} "
              f"({100*res['accuracy']:.1f}%)")
        for t, r in res["by_type"].items():
            print(f"    {t}: {r['correct']}/{r['total']} "
                  f"({100*r['correct']/max(r['total'],1):.0f}%)")

    # Save final results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "baseline": {str(k): v for k, v in baseline.items()},
            "final": {str(k): v for k, v in final.items()},
        }, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
