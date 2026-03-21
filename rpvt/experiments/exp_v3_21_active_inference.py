"""v3.21: Active Inference — settling loop driven by prediction error.

Tests whether iterative settling between forward and inverse models
reduces prediction error and improves understanding.

Key question: does prediction error decrease during settling?
If yes → the models are converging on shared understanding.
If no → the architecture needs adjustment.

Usage:
    python -m rpvt.experiments.exp_v3_21_active_inference
    python -m rpvt.experiments.exp_v3_21_active_inference --n-settle 5
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.active_inference import ActiveInferenceEngine


def generate_understanding_tasks(n, tokenizer, seed=42):
    """Generate tasks that require building understanding of data.

    These tasks need the model to process structured information
    before answering — exactly where settling should help.
    """
    rng = random.Random(seed)
    tasks = []

    for i in range(n):
        task_type = rng.choice([
            "structured_extract", "multi_fact", "hidden_pattern",
            "compare_entries", "count_filter",
        ])

        if task_type == "structured_extract":
            # Parse structured text and extract a specific field
            names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank"]
            name = rng.choice(names)
            fields = {
                "age": rng.randint(20, 60),
                "city": rng.choice(["NYC", "LA", "Chicago", "Boston"]),
                "role": rng.choice(["engineer", "designer", "manager"]),
                "team": rng.choice(["Alpha", "Beta", "Gamma"]),
            }
            # Present in random order with noise
            filler = rng.choice([
                "Employee record updated.",
                "See HR for details.",
                "Annual review pending.",
            ])
            lines = [filler]
            for k, v in fields.items():
                lines.append(f"  {k}: {v}")
            rng.shuffle(lines)
            text = f"Record for {name}:\n" + "\n".join(lines)
            q_field = rng.choice(list(fields.keys()))
            prompt = f"{text}\n\nWhat is {name}'s {q_field}?"
            answer = str(fields[q_field])

        elif task_type == "multi_fact":
            # Multiple people, ask about a specific one
            people = rng.sample(["Alice", "Bob", "Carol", "David", "Eve"], 3)
            facts = {}
            lines = []
            for p in people:
                fact = rng.choice(["likes cats", "plays guitar", "runs marathons",
                                   "speaks French", "writes poetry"])
                facts[p] = fact
                lines.append(f"{p} {fact}.")
            rng.shuffle(lines)
            text = " ".join(lines)
            target = rng.choice(people)
            prompt = f"{text}\n\nWhat does {target} do?"
            answer = facts[target]

        elif task_type == "hidden_pattern":
            # Numbers with a pattern
            start = rng.randint(1, 10)
            step = rng.randint(2, 5)
            seq = [start + step * i for i in range(5)]
            prompt = f"Sequence: {', '.join(map(str, seq[:-1]))}, ?\n\nWhat is the next number?"
            answer = str(seq[-1])

        elif task_type == "compare_entries":
            # Table of items, ask which has max/min of a field
            items = []
            for _ in range(rng.randint(3, 5)):
                name = ''.join(rng.choices("ABCDEF", k=3))
                year = rng.randint(1990, 2025)
                score = rng.randint(1, 100)
                items.append({"name": name, "year": year, "score": score})
            table = "\n".join(
                f"  {it['name']}: year={it['year']}, score={it['score']}"
                for it in items
            )
            query = rng.choice(["highest score", "lowest score",
                                "oldest", "newest"])
            if query == "highest score":
                answer = max(items, key=lambda x: x["score"])["name"]
            elif query == "lowest score":
                answer = min(items, key=lambda x: x["score"])["name"]
            elif query == "oldest":
                answer = min(items, key=lambda x: x["year"])["name"]
            else:
                answer = max(items, key=lambda x: x["year"])["name"]
            prompt = f"Items:\n{table}\n\nWhich item has the {query}?"

        elif task_type == "count_filter":
            # Count items matching a criterion
            categories = ["red", "blue", "green"]
            items = [rng.choice(categories) for _ in range(rng.randint(5, 10))]
            target = rng.choice(categories)
            items_str = ", ".join(items)
            prompt = f"Items: {items_str}\n\nHow many are {target}?"
            answer = str(items.count(target))

        tasks.append({
            "prompt": prompt,
            "answer": answer,
            "type": task_type,
        })

    return tasks


def evaluate_settling(engine, tokenizer, tasks, device, n_settle=3):
    """Evaluate whether settling reduces prediction error.

    For each task:
    1. Measure initial prediction error (step 0)
    2. Run settling loop
    3. Measure final prediction error
    4. Check if model generates correct answer
    """
    results = []

    for i, task in enumerate(tasks):
        engine.reset()

        # Tokenize
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=512).to(device)

        # Settle
        final_error, history = engine.settle(
            ids["input_ids"],
            attention_mask=ids["attention_mask"],
            n_steps=n_settle,
        )

        # Check error reduction
        error_reduced = history[-1] < history[0] if len(history) > 1 else False
        reduction_pct = (
            100 * (history[0] - history[-1]) / max(history[0], 1e-8)
            if len(history) > 1 else 0
        )

        results.append({
            "type": task["type"],
            "history": history,
            "initial_error": history[0],
            "final_error": history[-1],
            "error_reduced": error_reduced,
            "reduction_pct": reduction_pct,
        })

        if (i + 1) % 10 == 0 or i < 3:
            print(f"  [{i+1}] {task['type']:20s} "
                  f"error: {history[0]:.4f} → {history[-1]:.4f} "
                  f"({'↓' if error_reduced else '↑'} {abs(reduction_pct):.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="v3.21: Active Inference")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--source-layer", type=int, default=27,
                        help="High layer (source for inverse)")
    parser.add_argument("--target-layer", type=int, default=14,
                        help="Low layer (inverse predicts this)")
    parser.add_argument("--inject-layer", type=int, default=15,
                        help="Layer where errors are injected")
    parser.add_argument("--n-settle", type=int, default=3,
                        help="Settling steps per task")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--inverse-lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str,
                        default="results/active_inference")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (no LoRA for now — just testing the settling loop)
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

    # Create engine
    engine = ActiveInferenceEngine(
        forward_model=model,
        hidden_size=model.config.hidden_size,
        source_layer=args.source_layer,
        target_layer=args.target_layer,
        inject_layer=args.inject_layer,
        inverse_lr=args.inverse_lr,
    )

    # Generate tasks
    print(f"\nGenerating {args.n_tasks} tasks...")
    tasks = generate_understanding_tasks(args.n_tasks, tokenizer)
    from collections import Counter
    type_counts = Counter(t["type"] for t in tasks)
    print(f"  Types: {dict(type_counts)}")

    # Test settling
    print(f"\n--- Settling test ({args.n_settle} steps per task) ---")
    results = evaluate_settling(
        engine, tokenizer, tasks, device, n_settle=args.n_settle
    )

    # Analyze
    n_reduced = sum(1 for r in results if r["error_reduced"])
    avg_initial = sum(r["initial_error"] for r in results) / len(results)
    avg_final = sum(r["final_error"] for r in results) / len(results)
    avg_reduction = sum(r["reduction_pct"] for r in results) / len(results)

    print(f"\n--- Results ---")
    print(f"  Tasks where error decreased: {n_reduced}/{len(results)} "
          f"({100*n_reduced/len(results):.0f}%)")
    print(f"  Avg initial error: {avg_initial:.4f}")
    print(f"  Avg final error:   {avg_final:.4f}")
    print(f"  Avg reduction:     {avg_reduction:.1f}%")

    # By type
    print(f"\n  By type:")
    for task_type in sorted(set(r["type"] for r in results)):
        type_results = [r for r in results if r["type"] == task_type]
        n_red = sum(1 for r in type_results if r["error_reduced"])
        avg_red = sum(r["reduction_pct"] for r in type_results) / len(type_results)
        print(f"    {task_type:20s}: {n_red}/{len(type_results)} reduced, "
              f"avg {avg_red:+.1f}%")

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "summary": {
                "n_reduced": n_reduced,
                "total": len(results),
                "avg_initial_error": avg_initial,
                "avg_final_error": avg_final,
                "avg_reduction_pct": avg_reduction,
            },
            "tasks": results,
        }, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
