"""v3.22: Active inference settling with cross-attention.

Tests whether the forward model's output changes when it can attend
to the inverse model's predictions, and whether settling improves
the representations.

Key questions:
1. Does prediction error decrease during settling? (v2 with cross-attn)
2. Do the model's logits change when predictions are available?
3. Does settling produce different (hopefully better) generations?

Usage:
    python -m rpvt.experiments.exp_v3_22_settling
    python -m rpvt.experiments.exp_v3_22_settling --n-settle 5
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.active_inference_v2 import ActiveInferenceSettler


def generate_tasks(n, seed=42):
    """Tasks that require understanding before answering."""
    import random
    rng = random.Random(seed)
    tasks = []

    for _ in range(n):
        task_type = rng.choice([
            "extract", "compare", "count", "pattern", "multi_fact",
        ])

        if task_type == "extract":
            names = ["Alice", "Bob", "Carol", "David", "Eve"]
            name = rng.choice(names)
            age = rng.randint(20, 60)
            city = rng.choice(["NYC", "LA", "Chicago", "Boston", "Denver"])
            role = rng.choice(["engineer", "designer", "manager", "analyst"])
            text = f"{name} is a {age}-year-old {role} based in {city}."
            field = rng.choice(["age", "city", "role"])
            answer = {"age": str(age), "city": city, "role": role}[field]
            prompt = f"{text} What is {name}'s {field}?"

        elif task_type == "compare":
            items = []
            for _ in range(rng.randint(3, 5)):
                n_ = ''.join(rng.choices("ABCDEF", k=3))
                v = rng.randint(10, 100)
                items.append((n_, v))
            table = ", ".join(f"{n_}={v}" for n_, v in items)
            best = max(items, key=lambda x: x[1])
            prompt = f"Values: {table}. Which has the highest value?"
            answer = best[0]

        elif task_type == "count":
            colors = ["red", "blue", "green", "yellow"]
            items = [rng.choice(colors) for _ in range(rng.randint(6, 12))]
            target = rng.choice(colors)
            prompt = f"Items: {', '.join(items)}. How many are {target}?"
            answer = str(items.count(target))

        elif task_type == "pattern":
            start = rng.randint(1, 10)
            step = rng.randint(2, 5)
            seq = [start + step * i for i in range(5)]
            prompt = (f"Sequence: {', '.join(map(str, seq[:-1]))}. "
                      f"What comes next?")
            answer = str(seq[-1])

        elif task_type == "multi_fact":
            people = rng.sample(["Alice", "Bob", "Carol", "David", "Eve"], 3)
            hobbies = rng.sample(["painting", "chess", "running",
                                   "cooking", "singing"], 3)
            lines = [f"{p} enjoys {h}." for p, h in zip(people, hobbies)]
            rng.shuffle(lines)
            target = rng.choice(people)
            hobby = hobbies[people.index(target)]
            prompt = f"{' '.join(lines)} What does {target} enjoy?"
            answer = hobby

        tasks.append({"prompt": prompt, "answer": answer, "type": task_type})

    return tasks


def main():
    parser = argparse.ArgumentParser(description="v3.22: Settling")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--n-settle", type=int, default=3)
    parser.add_argument("--n-tasks", type=int, default=30)
    parser.add_argument("--inverse-lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str,
                        default="results/settling")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False

    settler = ActiveInferenceSettler(
        model,
        hidden_size=model.config.hidden_size,
        inverse_lr=args.inverse_lr,
    )

    tasks = generate_tasks(args.n_tasks)
    print(f"Tasks: {args.n_tasks}, types: {dict(Counter(t['type'] for t in tasks))}")

    results = []
    total_error_reduced = 0
    total_logit_changed = 0
    total_answer_match_before = 0
    total_answer_match_after = 0

    print(f"\n--- Settling test ({args.n_settle} steps) ---")

    for i, task in enumerate(tasks):
        settler.reset()

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer(prompt_text, return_tensors="pt",
                        truncation=True, max_length=256).to(device)

        # 1. Get logits WITHOUT settling (baseline)
        with torch.no_grad():
            baseline_out = model(
                ids["input_ids"],
                attention_mask=ids["attention_mask"],
                use_cache=False,
            )
        baseline_logits = baseline_out.logits[:, -1, :].clone()

        # Generate baseline answer
        baseline_gen = []
        gen_ids = ids["input_ids"].clone()
        gen_mask = ids["attention_mask"].clone()
        for _ in range(20):
            with torch.no_grad():
                out = model(gen_ids, attention_mask=gen_mask, use_cache=False)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_tok.item() == tokenizer.eos_token_id:
                break
            baseline_gen.append(next_tok.item())
            gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=device, dtype=gen_mask.dtype)], dim=1)
        baseline_answer = tokenizer.decode(baseline_gen, skip_special_tokens=True).strip()

        # 2. Settle
        error_history, info = settler.settle(
            ids["input_ids"],
            attention_mask=ids["attention_mask"],
            n_steps=args.n_settle,
        )

        # 3. Get logits AFTER settling
        settled_logits = settler.get_logits(
            ids["input_ids"],
            attention_mask=ids["attention_mask"],
        )[:, -1, :]

        # 4. Generate after settling
        settler.reset()  # Reset inverse state
        settled_gen_tokens = settler.generate(
            ids["input_ids"],
            attention_mask=ids["attention_mask"],
            max_new_tokens=20,
            n_settle=args.n_settle,
        )
        settled_answer = tokenizer.decode(settled_gen_tokens, skip_special_tokens=True).strip()

        # Metrics
        error_reduced = error_history[-1] < error_history[0]
        logit_diff = (baseline_logits - settled_logits).abs().mean().item()
        logit_changed = logit_diff > 0.01

        baseline_correct = task["answer"].lower() in baseline_answer.lower()
        settled_correct = task["answer"].lower() in settled_answer.lower()

        if error_reduced:
            total_error_reduced += 1
        if logit_changed:
            total_logit_changed += 1
        if baseline_correct:
            total_answer_match_before += 1
        if settled_correct:
            total_answer_match_after += 1

        results.append({
            "type": task["type"],
            "error_history": error_history,
            "error_reduced": error_reduced,
            "logit_diff": logit_diff,
            "baseline_answer": baseline_answer[:100],
            "settled_answer": settled_answer[:100],
            "gold": task["answer"],
            "baseline_correct": baseline_correct,
            "settled_correct": settled_correct,
        })

        if i < 5 or (i + 1) % 10 == 0:
            arrow = "↓" if error_reduced else "↑"
            print(f"  [{i+1}] {task['type']:12s} "
                  f"err: {error_history[0]:.2f}→{error_history[-1]:.2f}{arrow} "
                  f"logit_diff={logit_diff:.4f} "
                  f"base='{baseline_answer[:30]}' "
                  f"settled='{settled_answer[:30]}' "
                  f"gold='{task['answer']}'")

    n = len(results)
    print(f"\n--- Results ---")
    print(f"  Error decreased:    {total_error_reduced}/{n} ({100*total_error_reduced/n:.0f}%)")
    print(f"  Logits changed:     {total_logit_changed}/{n} ({100*total_logit_changed/n:.0f}%)")
    print(f"  Baseline correct:   {total_answer_match_before}/{n} ({100*total_answer_match_before/n:.0f}%)")
    print(f"  Settled correct:    {total_answer_match_after}/{n} ({100*total_answer_match_after/n:.0f}%)")

    delta = total_answer_match_after - total_answer_match_before
    print(f"  Delta:              {'+' if delta >= 0 else ''}{delta}")

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "summary": {
                "error_reduced": total_error_reduced,
                "logits_changed": total_logit_changed,
                "baseline_correct": total_answer_match_before,
                "settled_correct": total_answer_match_after,
                "total": n,
            },
            "tasks": results,
        }, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
