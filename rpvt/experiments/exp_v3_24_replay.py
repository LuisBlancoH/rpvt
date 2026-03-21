"""v3.24: LoRA training with replay to prevent catastrophic forgetting.

The problem: LoRA training on reasoning tasks drops accuracy from 48% to 10%.
The hypothesis: catastrophic forgetting — gradients overwrite existing capabilities.
The fix: mix reasoning tasks with general text replay (like hippocampal replay).

Tests:
1. LoRA only, NO replay (control — expect collapse)
2. LoRA only, WITH replay (50/50 mix — expect preservation)
3. Different replay ratios (25%, 50%, 75%)
4. Lower learning rates

Usage:
    python -m rpvt.experiments.exp_v3_24_replay
    python -m rpvt.experiments.exp_v3_24_replay --replay-ratio 0.75
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_tasks(n, seed=42):
    rng = random.Random(seed)
    tasks = []
    for _ in range(n):
        t = rng.choice(["extract", "compare", "count", "pattern", "multi_fact"])

        if t == "extract":
            name = rng.choice(["Alice", "Bob", "Carol", "David", "Eve"])
            age = rng.randint(20, 60)
            city = rng.choice(["NYC", "LA", "Chicago", "Boston", "Denver"])
            role = rng.choice(["engineer", "designer", "manager", "analyst"])
            field = rng.choice(["age", "city", "role"])
            answer = {"age": str(age), "city": city, "role": role}[field]
            prompt = f"{name} is a {age}-year-old {role} based in {city}. What is {name}'s {field}?"

        elif t == "compare":
            items = [(''.join(rng.choices("ABCDEF", k=3)), rng.randint(10, 100))
                     for _ in range(rng.randint(3, 5))]
            table = ", ".join(f"{n_}={v}" for n_, v in items)
            best = max(items, key=lambda x: x[1])
            prompt = f"Values: {table}. Which has the highest value?"
            answer = best[0]

        elif t == "count":
            colors = ["red", "blue", "green", "yellow"]
            items = [rng.choice(colors) for _ in range(rng.randint(6, 12))]
            target = rng.choice(colors)
            prompt = f"Items: {', '.join(items)}. How many are {target}?"
            answer = str(items.count(target))

        elif t == "pattern":
            start = rng.randint(1, 10)
            step = rng.randint(2, 5)
            seq = [start + step * i for i in range(5)]
            prompt = f"Sequence: {', '.join(map(str, seq[:-1]))}. What comes next?"
            answer = str(seq[-1])

        elif t == "multi_fact":
            people = rng.sample(["Alice", "Bob", "Carol", "David", "Eve"], 3)
            hobbies = rng.sample(["painting", "chess", "running", "cooking", "singing"], 3)
            lines = [f"{p} enjoys {h}." for p, h in zip(people, hobbies)]
            rng.shuffle(lines)
            target = rng.choice(people)
            hobby = hobbies[people.index(target)]
            prompt = f"{' '.join(lines)} What does {target} enjoy?"
            answer = hobby

        tasks.append({"prompt": prompt, "answer": answer, "type": t})
    return tasks


def load_replay_data(tokenizer, n=500, max_length=128):
    """Load general text for replay (WikiText)."""
    print("  Loading replay data (wikitext)...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 50][:n]
    except Exception:
        # Fallback: generate simple text
        print("  WikiText unavailable, generating synthetic replay data...")
        rng = random.Random(42)
        templates = [
            "The {} is a {} that {} in the {}.",
            "In {}, {} {} the {} and {} the {}.",
            "{} {} {} {} {} {}.",
        ]
        words = ["cat", "dog", "bird", "fish", "tree", "river", "mountain",
                 "city", "house", "book", "runs", "jumps", "flies", "swims",
                 "big", "small", "old", "new", "red", "blue", "green",
                 "forest", "ocean", "desert", "garden", "park", "school"]
        texts = []
        for _ in range(n):
            template = rng.choice(templates)
            n_blanks = template.count("{}")
            fillers = [rng.choice(words) for _ in range(n_blanks)]
            texts.append(template.format(*fillers))

    print(f"  Replay texts: {len(texts)}")
    return texts


def train_epoch_with_replay(model, tokenizer, tasks, replay_texts,
                            optimizer, device, replay_ratio=0.5,
                            max_length=256):
    """Train one epoch mixing reasoning tasks with replay data."""
    model.train()

    # Build training batch: interleave tasks and replay
    n_tasks = len(tasks)
    n_replay = int(n_tasks * replay_ratio / (1 - replay_ratio + 1e-8))
    n_replay = min(n_replay, len(replay_texts))

    # Prepare task examples
    task_examples = []
    for task in tasks:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task["prompt"]},
                {"role": "assistant", "content": task["answer"]},
            ],
            tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        task_examples.append(("task", text, task))

    # Prepare replay examples (plain next-token prediction)
    replay_examples = []
    rng = random.Random()
    for text in rng.sample(replay_texts, min(n_replay, len(replay_texts))):
        replay_examples.append(("replay", text, None))

    # Interleave
    all_examples = task_examples + replay_examples
    random.shuffle(all_examples)

    total_task_loss = 0
    total_replay_loss = 0
    n_task = 0
    n_replay_count = 0

    for example_type, text, task in all_examples:
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
        input_ids = tokens["input_ids"]

        if example_type == "task":
            # Task: mask everything except answer
            labels = input_ids.clone()
            answer_tokens = tokenizer(
                task["answer"], add_special_tokens=False
            )["input_ids"]
            answer_len = len(answer_tokens)
            labels[:, :-(answer_len + 1)] = -100

            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_task_loss += loss.item()
            n_task += 1

        else:
            # Replay: full next-token prediction (preserve generation)
            labels = input_ids.clone()
            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_replay_loss += loss.item()
            n_replay_count += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

    return (total_task_loss / max(n_task, 1),
            total_replay_loss / max(n_replay_count, 1))


def evaluate(model, tokenizer, tasks, device):
    model.eval()
    correct = 0
    by_type = {}

    for task in tasks:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer(prompt_text, return_tensors="pt",
                       truncation=True, max_length=256).to(device)

        gen_ids = ids["input_ids"].clone()
        gen_mask = ids["attention_mask"].clone()
        tokens = []
        for _ in range(30):
            with torch.no_grad():
                out = model(gen_ids, attention_mask=gen_mask, use_cache=False)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_tok.item() == tokenizer.eos_token_id:
                break
            tokens.append(next_tok.item())
            gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=device,
                                  dtype=gen_mask.dtype)], dim=1)

        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        gold = task["answer"].lower().strip()
        is_correct = gold in answer.lower()

        if is_correct:
            correct += 1

        t = task["type"]
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0}
        by_type[t]["total"] += 1
        if is_correct:
            by_type[t]["correct"] += 1

    return correct, len(tasks), by_type


def run_experiment(model_name, tokenizer, train_tasks, eval_tasks,
                   replay_texts, device, replay_ratio, lr, epochs,
                   label):
    """Run one training configuration."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  replay_ratio={replay_ratio}, lr={lr}, epochs={epochs}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False

    n_layers = model.config.num_hidden_layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(14, n_layers)),
    )
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    # Baseline
    c, t, bt = evaluate(model, tokenizer, eval_tasks[:20], device)
    print(f"  Baseline: {c}/{t} ({100*c/t:.1f}%)")

    best_acc = 0
    for epoch in range(epochs):
        t0 = time.time()
        task_loss, replay_loss = train_epoch_with_replay(
            model, tokenizer, train_tasks, replay_texts,
            optimizer, device, replay_ratio=replay_ratio,
        )
        elapsed = time.time() - t0

        c, t, bt = evaluate(model, tokenizer, eval_tasks[:20], device)
        acc = 100 * c / t
        best_acc = max(best_acc, acc)

        replay_str = f"replay={replay_loss:.3f}" if replay_ratio > 0 else ""
        print(f"  Epoch {epoch+1}/{epochs}: task_loss={task_loss:.4f} "
              f"{replay_str} acc={acc:.1f}% ({elapsed:.0f}s)")

    # Final
    c, t, bt = evaluate(model, tokenizer, eval_tasks, device)
    final_acc = 100 * c / t
    print(f"  Final: {c}/{t} ({final_acc:.1f}%)")
    for typ, r in sorted(bt.items()):
        print(f"    {typ:12s}: {r['correct']}/{r['total']}")

    del model
    torch.cuda.empty_cache()

    return {"final_acc": final_acc, "best_acc": best_acc, "label": label}


def main():
    parser = argparse.ArgumentParser(description="v3.24: Replay")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="results/replay")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    train_tasks = generate_tasks(args.n_train, seed=42)
    eval_tasks = generate_tasks(args.n_eval, seed=123)
    replay_texts = load_replay_data(tokenizer, n=500)

    print(f"Tasks: {args.n_train} train, {args.n_eval} eval")
    print(f"Types: {dict(Counter(t['type'] for t in train_tasks))}")

    all_results = {}

    # Experiment 1: No replay (control — expect collapse)
    r = run_experiment(
        args.model_name, tokenizer, train_tasks, eval_tasks,
        replay_texts, device,
        replay_ratio=0.0, lr=1e-4, epochs=args.epochs,
        label="No replay (lr=1e-4)",
    )
    all_results["no_replay"] = r

    # Experiment 2: 50% replay
    r = run_experiment(
        args.model_name, tokenizer, train_tasks, eval_tasks,
        replay_texts, device,
        replay_ratio=0.5, lr=1e-4, epochs=args.epochs,
        label="50% replay (lr=1e-4)",
    )
    all_results["replay_50"] = r

    # Experiment 3: 75% replay
    r = run_experiment(
        args.model_name, tokenizer, train_tasks, eval_tasks,
        replay_texts, device,
        replay_ratio=0.75, lr=1e-4, epochs=args.epochs,
        label="75% replay (lr=1e-4)",
    )
    all_results["replay_75"] = r

    # Experiment 4: No replay but lower LR
    r = run_experiment(
        args.model_name, tokenizer, train_tasks, eval_tasks,
        replay_texts, device,
        replay_ratio=0.0, lr=1e-5, epochs=args.epochs,
        label="No replay (lr=1e-5)",
    )
    all_results["no_replay_low_lr"] = r

    # Experiment 5: 50% replay + lower LR
    r = run_experiment(
        args.model_name, tokenizer, train_tasks, eval_tasks,
        replay_texts, device,
        replay_ratio=0.5, lr=1e-5, epochs=args.epochs,
        label="50% replay (lr=1e-5)",
    )
    all_results["replay_50_low_lr"] = r

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline (no training): 48.0%")
    for k, v in all_results.items():
        delta = v['final_acc'] - 48.0
        marker = "✓" if delta >= 0 else "✗"
        print(f"  {marker} {v['label']:30s}: {v['final_acc']:.1f}% "
              f"({'+' if delta >= 0 else ''}{delta:.1f}%)")

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
