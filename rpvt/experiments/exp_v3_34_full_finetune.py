"""v3.34: Full fine-tune 1.5B + RMT with mechanism testing.

No LoRA — entire transformer fine-tuned with 8-bit Adam.
Tests ALL mechanisms: value, confidence, memory, management, settling, temporal.

Usage:
    python -m rpvt.experiments.exp_v3_34_full_finetune
"""

import argparse
import json
import random
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.recurrent_memory import RecurrentMemoryTransformer
from rpvt.experiments.exp_v3_32_agent import (
    gen_l1, gen_l2, gen_person, InteractiveEnv,
    eval_qa, eval_interactive, eval_code, run_eval,
)


# ─── Mechanism-specific evaluations ─────────────────────────────────────

def eval_memory_retrieval(model, tokenizer, device, n=50, seed=42):
    """Test: does memory retrieval work? Passage → memory → QA."""
    model.eval()
    rng = random.Random(seed)
    correct_mem = 0
    correct_nomem = 0
    total = 0

    for _ in range(n):
        sample = gen_l1(rng, n_passages=1)
        passage = sample["passages"][0]
        q, a = sample["question"], sample["answer"]

        # WITH memory
        model.reset_memory()
        p_ids = tokenizer(passage, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.forward(p_ids, n_passes=1)
        q_ids = tokenizer(f"Q: {q} A:", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(q_ids, max_new_tokens=15, n_passes=1)
        if a.lower() in tokenizer.decode(gen, skip_special_tokens=True).lower():
            correct_mem += 1

        # WITHOUT memory
        model.reset_memory()
        with torch.no_grad():
            gen = model.generate(q_ids, max_new_tokens=15, n_passes=1)
        if a.lower() in tokenizer.decode(gen, skip_special_tokens=True).lower():
            correct_nomem += 1

        total += 1

    return {
        "with_memory": correct_mem / total,
        "without_memory": correct_nomem / total,
        "delta": (correct_mem - correct_nomem) / total,
    }


def eval_settling(model, tokenizer, device, n=30, seed=42):
    """Test: does multi-pass beat single-pass?"""
    model.eval()
    rng = random.Random(seed)
    results = {}

    for n_passes in [1, 2, 3]:
        correct = 0
        total = 0
        for _ in range(n):
            sample = gen_l2(rng, n_hops=2)
            model.reset_memory()
            for p in sample["passages"]:
                p_ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    model.forward(p_ids, n_passes=1)
            q_ids = tokenizer(f"Q: {sample['question']} A:",
                              return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                gen = model.generate(q_ids, max_new_tokens=15, n_passes=n_passes)
            if sample["answer"].lower() in tokenizer.decode(gen, skip_special_tokens=True).lower():
                correct += 1
            total += 1
        results[f"pass_{n_passes}"] = correct / total

    return results


def eval_value_accuracy(model, tokenizer, device, n=30, seed=42):
    """Test: does value predict success? High value → correct answers?"""
    model.eval()
    rng = random.Random(seed)
    values_correct = []
    values_wrong = []

    for _ in range(n):
        sample = gen_l1(rng, n_passages=1)
        model.reset_memory()
        p_ids = tokenizer(sample["passages"][0], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.forward(p_ids, n_passes=1)

        q_ids = tokenizer(f"Q: {sample['question']} A:",
                          return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            _, _, info = model.forward(q_ids, return_info=True, n_passes=1)
            gen = model.generate(q_ids, max_new_tokens=15, n_passes=1)

        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
        is_correct = sample["answer"].lower() in answer.lower()
        value = info["value"]

        if is_correct:
            values_correct.append(value)
        else:
            values_wrong.append(value)

    avg_correct = sum(values_correct) / max(len(values_correct), 1)
    avg_wrong = sum(values_wrong) / max(len(values_wrong), 1)

    return {
        "avg_value_correct": avg_correct,
        "avg_value_wrong": avg_wrong,
        "value_separates": avg_correct > avg_wrong,
        "n_correct": len(values_correct),
        "n_wrong": len(values_wrong),
    }


def eval_memory_management(model, tokenizer, device, n=20, seed=42):
    """Test: does manager keep relevant entries and evict irrelevant ones?"""
    model.eval()
    rng = random.Random(seed)

    relevant_priority = []
    irrelevant_priority = []

    for _ in range(n):
        model.reset_memory()

        # Store relevant passage
        name = rng.choice(["Alice", "Bob", "Carol"])
        company = rng.choice(["Acme", "Globex", "Initech"])
        relevant = f"{name} works at {company}."
        r_ids = tokenizer(relevant, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.forward(r_ids, n_passes=1)

        # Store irrelevant filler
        for _ in range(3):
            filler = f"The weather is {rng.choice(['sunny', 'rainy', 'cloudy'])} today."
            f_ids = tokenizer(filler, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                model.forward(f_ids, n_passes=1)

        # Ask question (triggers manager re-evaluation)
        q = f"Q: What company does {name} work for? A:"
        q_ids = tokenizer(q, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.forward(q_ids, n_passes=1)

        # Check priorities: first 16 entries (relevant passage) vs rest (filler)
        pris = model.memory_buffer.priorities[:model.memory_buffer.n_stored]
        if len(pris) >= 32:
            relevant_priority.append(pris[:16].mean().item())
            irrelevant_priority.append(pris[16:32].mean().item())

    avg_rel = sum(relevant_priority) / max(len(relevant_priority), 1)
    avg_irr = sum(irrelevant_priority) / max(len(irrelevant_priority), 1)

    return {
        "relevant_priority": avg_rel,
        "irrelevant_priority": avg_irr,
        "manager_discriminates": avg_rel > avg_irr,
    }


def eval_temporal(model, tokenizer, device, n=30, seed=42):
    """Test: can the model distinguish ordering?"""
    model.eval()
    rng = random.Random(seed)
    correct = 0
    total = 0

    for _ in range(n):
        model.reset_memory()
        name = rng.choice(["Alice", "Bob", "Carol", "David"])
        city1 = rng.choice(["Tokyo", "Paris", "London"])
        city2 = rng.choice(["Berlin", "Sydney", "Toronto"])

        # Store: moved to city1 first, then city2
        s1 = f"{name} moved to {city1}."
        s2 = f"Later, {name} relocated to {city2}."
        for s in [s1, s2]:
            ids = tokenizer(s, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                model.forward(ids, n_passes=1)

        # Ask current location (should be city2)
        q = f"Q: Where does {name} currently live? A:"
        q_ids = tokenizer(q, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(q_ids, max_new_tokens=10, n_passes=1)
        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()

        if city2.lower() in answer.lower():
            correct += 1
        total += 1

    return {"accuracy": correct / total, "correct": correct, "total": total}


def run_mechanism_eval(model, tokenizer, device, label=""):
    """Run all mechanism-specific evaluations."""
    print(f"\n{'='*60}")
    print(f"  MECHANISM EVALUATION {label}")
    print(f"{'='*60}")

    r = eval_memory_retrieval(model, tokenizer, device)
    print(f"  Memory: with={r['with_memory']:.1%}, without={r['without_memory']:.1%}, "
          f"delta={r['delta']:+.1%}")

    r2 = eval_settling(model, tokenizer, device)
    print(f"  Settling: 1-pass={r2['pass_1']:.1%}, 2-pass={r2['pass_2']:.1%}, "
          f"3-pass={r2['pass_3']:.1%}")

    r3 = eval_value_accuracy(model, tokenizer, device)
    print(f"  Value: correct={r3['avg_value_correct']:.3f}, "
          f"wrong={r3['avg_value_wrong']:.3f}, "
          f"separates={r3['value_separates']}")

    r4 = eval_memory_management(model, tokenizer, device)
    print(f"  Manager: relevant={r4['relevant_priority']:.3f}, "
          f"irrelevant={r4['irrelevant_priority']:.3f}, "
          f"discriminates={r4['manager_discriminates']}")

    r5 = eval_temporal(model, tokenizer, device)
    print(f"  Temporal: {r5['accuracy']:.1%} ({r5['correct']}/{r5['total']})")

    print(f"{'='*60}")
    return {"memory": r, "settling": r2, "value": r3, "manager": r4, "temporal": r5}


# ─── Training ───────────────────────────────────────────────────────────

def train_qa_batch(model, tokenizer, device, rng, level):
    """Train on QA sample."""
    model.reset_memory()

    if level == "L1":
        sample = gen_l1(rng, rng.choice([1, 2, 3, 5]))
        passages = sample["passages"]
    else:
        sample = gen_l2(rng, rng.choice([2, 3]))
        passages = sample["passages"]

    for p in passages:
        p_ids = tokenizer(p, return_tensors="pt", truncation=True,
                          max_length=128).input_ids.to(device)
        model.forward(p_ids, n_passes=1)

    qa_text = f"Q: {sample['question']} A: {sample['answer']}"
    qa_ids = tokenizer(qa_text, return_tensors="pt", truncation=True,
                       max_length=128).input_ids.to(device)
    result = model.forward(qa_ids[:, :-1], labels=qa_ids[:, 1:], n_passes=1)
    return result[1]


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=str, default="results/rmt_1.5b_full")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True)

    # No LoRA — full fine-tune
    model = RecurrentMemoryTransformer(
        qwen, n_memory_tokens=16, lora_rank=0, max_passes=3)
    model = model.to(device)
    del qwen
    import gc; gc.collect(); torch.cuda.empty_cache()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    print(f"  Trainable: {trainable/1e6:.1f}M (full fine-tune)")

    # 8-bit Adam (fits in VRAM)
    optimizer = bnb.optim.Adam8bit(
        model.parameters(), lr=args.lr, weight_decay=0.01)

    # Initial evals
    run_eval(model, tokenizer, device, "Initial (L1-L4)")
    mech_results = run_mechanism_eval(model, tokenizer, device, "Initial (mechanisms)")

    history = []

    print(f"\n--- Training ({args.epochs} epochs, full fine-tune) ---")
    for epoch in range(args.epochs):
        model.train()
        rng = random.Random(epoch + 42)
        total_loss = 0
        n = 0
        t0 = time.time()
        optimizer.zero_grad()

        for step in range(args.n_train):
            level = "L1" if rng.random() < 0.6 else "L2"
            loss = train_qa_batch(model, tokenizer, device, rng, level)

            if loss is not None and not torch.isnan(loss):
                (loss / args.grad_accum).backward()
                total_loss += loss.item()
                n += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % 200 == 0:
                elapsed = time.time() - t0
                speed = (step + 1) / elapsed
                print(f"    [{step+1}/{args.n_train}] "
                      f"loss={total_loss/max(n,1):.4f} "
                      f"({speed:.1f} step/s, ETA {(args.n_train-step)/speed:.0f}s)")

        if (step + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n, 1)

        # Level evals
        level_results = run_eval(model, tokenizer, device, f"Epoch {epoch+1}")

        # Mechanism evals (every 3 epochs to save time)
        mech = {}
        if (epoch + 1) % 3 == 0 or epoch == 0:
            mech = run_mechanism_eval(model, tokenizer, device, f"Epoch {epoch+1}")

        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} ({elapsed:.0f}s)")

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "levels": level_results,
            "mechanisms": mech,
        })

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "history": history,
        }, output_dir / f"checkpoint_epoch{epoch+1}.pt")

    # Final
    final_levels = run_eval(model, tokenizer, device, "Final (levels)")
    final_mechs = run_mechanism_eval(model, tokenizer, device, "Final (mechanisms)")

    with open(output_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "history": history,
                   "final_levels": final_levels, "final_mechanisms": final_mechs},
                  f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
