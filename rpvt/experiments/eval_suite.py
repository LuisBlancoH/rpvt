"""Evaluation suite for Predictive Transformer.

Tests memory, retrieval, long-range dependencies across multiple datasets.
Compares with-memory vs without-memory (baseline).

Datasets:
  1. Synthetic QA — controlled memory retrieval
  2. bAbI tasks 1-3 — structured reasoning at increasing difficulty
  3. LAMBADA — last-word prediction requiring long context
  4. SQuAD v2 — real-world reading comprehension
  5. WikiText long-doc PPL — perplexity over 20+ consecutive chunks

Usage:
    python -m rpvt.experiments.eval_suite
    python -m rpvt.experiments.eval_suite --checkpoint results/qwen_wrapped_pt_v6/checkpoint_epoch1.pt
    python -m rpvt.experiments.eval_suite --tasks synthetic babi lambada
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.predictive_transformer import PredictiveTransformer


# ─── Synthetic QA ───────────────────────────────────────────────────────

NAMES = ["Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry",
         "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul"]
COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella", "Cyberdyne",
             "Stark Industries", "Wayne Enterprises", "Oscorp"]
CITIES = ["Tokyo", "Paris", "London", "Berlin", "Sydney", "Toronto",
          "Dubai", "Singapore", "Mumbai", "Seoul"]
ROLES = ["engineer", "designer", "manager", "analyst", "researcher",
         "consultant", "director", "scientist"]


def generate_synthetic_qa(n=100, seed=42):
    """Generate synthetic passage + QA pairs for memory testing."""
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        name = rng.choice(NAMES)
        company = rng.choice(COMPANIES)
        city = rng.choice(CITIES)
        role = rng.choice(ROLES)

        passage = f"{name} works at {company} in {city} as a {role}."

        qas = [
            (f"What company does {name} work for?", company),
            (f"Where does {name} work?", city),
            (f"What is {name}'s role?", role),
        ]
        q, a = rng.choice(qas)
        samples.append({"passage": passage, "question": q, "answer": a})
    return samples


def eval_synthetic_qa(model, tokenizer, device, n=100, use_memory=True):
    """Evaluate on synthetic QA. Passage in chunk 1, question in chunk 2."""
    samples = generate_synthetic_qa(n)
    correct = 0
    total = 0

    for sample in samples:
        model.reset_state()

        # Chunk 1: passage
        passage_text = sample["passage"]
        passage_ids = tokenizer(passage_text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            if use_memory:
                model.forward(passage_ids, n_settle=1)
            # If not using memory, skip — model has no context

        # Chunk 2: question
        question_text = f"Q: {sample['question']} A:"
        question_ids = tokenizer(question_text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            if use_memory:
                generated = model.generate(question_ids, max_new_tokens=20, n_settle=1)
            else:
                model.reset_state()  # no memory
                generated = model.generate(question_ids, max_new_tokens=20, n_settle=1)

        answer_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        expected = sample["answer"]

        if expected.lower() in answer_text.lower():
            correct += 1
        total += 1

    return {"accuracy": correct / total, "correct": correct, "total": total}


# ─── bAbI Tasks ─────────────────────────────────────────────────────────

def load_babi_task(task_id, split="test", max_samples=200):
    """Load bAbI task samples."""
    try:
        ds = load_dataset("Muennighoff/babi", split=split)
    except Exception:
        # Fallback
        ds = load_dataset("facebook/babi_qa", f"en-10k-qa{task_id}",
                          split=split, trust_remote_code=True)
        return [{"passage": s["story"]["text"],
                 "question": s["story"]["text"][-1] if s["story"]["text"] else "",
                 "answer": s["story"]["answer"][-1] if s["story"]["answer"] else ""}
                for s in ds[:max_samples]]

    # Filter to specific task
    task_samples = [s for s in ds if s.get("task") == task_id]
    if not task_samples:
        task_samples = list(ds)  # use all if no task field
    return task_samples[:max_samples]


def eval_babi(model, tokenizer, device, task_id=1, max_samples=100,
              use_memory=True):
    """Evaluate on bAbI task. Passage → memory, question → generate answer."""
    samples = load_babi_task(task_id, max_samples=max_samples)
    if not samples:
        return {"accuracy": 0, "error": "Could not load bAbI data"}

    correct = 0
    total = 0

    for sample in samples:
        model.reset_state()
        passage = sample.get("passage", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        if not passage or not answer:
            continue

        # Chunk 1: passage (store in memory)
        passage_ids = tokenizer(passage, return_tensors="pt",
                                truncation=True, max_length=256).input_ids.to(device)

        with torch.no_grad():
            if use_memory:
                model.forward(passage_ids, n_settle=1)

        # Chunk 2: question
        q_text = f"Q: {question} A:"
        q_ids = tokenizer(q_text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            if use_memory:
                generated = model.generate(q_ids, max_new_tokens=10, n_settle=1)
            else:
                model.reset_state()
                generated = model.generate(q_ids, max_new_tokens=10, n_settle=1)

        gen_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        if answer.lower() in gen_text.lower():
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


# ─── LAMBADA ────────────────────────────────────────────────────────────

def eval_lambada(model, tokenizer, device, max_samples=200, use_memory=True,
                 chunk_size=64):
    """Evaluate on LAMBADA. Process context in chunks, predict last word."""
    ds = load_dataset("lambada", split="test")
    samples = list(ds)[:max_samples]

    correct = 0
    total = 0

    for sample in samples:
        text = sample["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) < 10:
            continue

        # Last token is the target
        target_token = tokens[-1]
        context_tokens = tokens[:-1]

        model.reset_state()

        if use_memory and len(context_tokens) > chunk_size:
            # Process in chunks (memory persists)
            for i in range(0, len(context_tokens) - chunk_size, chunk_size):
                chunk = context_tokens[i:i + chunk_size]
                chunk_ids = torch.tensor([chunk], device=device)
                with torch.no_grad():
                    model.forward(chunk_ids, n_settle=1)

            # Last chunk
            remaining = context_tokens[-(len(context_tokens) % chunk_size or chunk_size):]
            if remaining:
                last_ids = torch.tensor([remaining], device=device)
                with torch.no_grad():
                    logits, _ = model.forward(last_ids, n_settle=1)
        else:
            # No memory: just use last chunk_size tokens
            if not use_memory:
                context_tokens = context_tokens[-chunk_size:]
            ctx_ids = torch.tensor([context_tokens[-chunk_size:]], device=device)
            with torch.no_grad():
                logits, _ = model.forward(ctx_ids, n_settle=1)

        # Check prediction
        predicted = logits[0, -1, :].argmax().item()
        if predicted == target_token:
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


# ─── SQuAD v2 ──────────────────────────────────────────────────────────

def eval_squad(model, tokenizer, device, max_samples=100, use_memory=True,
               chunk_size=128):
    """Evaluate on SQuAD v2. Context in chunks → memory, then answer question."""
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    # Filter to answerable questions
    samples = [s for s in ds if s["answers"]["text"]][:max_samples]

    correct = 0
    total = 0

    for sample in samples:
        model.reset_state()
        context = sample["context"]
        question = sample["question"]
        answers = sample["answers"]["text"]

        # Chunk context into memory
        ctx_tokens = tokenizer.encode(context, add_special_tokens=False)

        if use_memory:
            for i in range(0, len(ctx_tokens), chunk_size):
                chunk = ctx_tokens[i:i + chunk_size]
                if chunk:
                    chunk_ids = torch.tensor([chunk], device=device)
                    with torch.no_grad():
                        model.forward(chunk_ids, n_settle=1)

        # Ask question
        q_text = f"Q: {question} A:"
        q_ids = tokenizer(q_text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            if not use_memory:
                model.reset_state()
            generated = model.generate(q_ids, max_new_tokens=30, n_settle=1)

        gen_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Check if any answer appears in generation
        if any(a.lower() in gen_text.lower() for a in answers):
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


# ─── WikiText Long-Doc PPL ──────────────────────────────────────────────

def eval_longdoc_ppl(model, tokenizer, device, n_chunks=20,
                     chunk_size=128, n_docs=20, use_memory=True):
    """Evaluate perplexity over long sequences (n_chunks consecutive chunks).

    Measures whether memory helps with long-range dependencies.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = "\n".join(t for t in ds["test"]["text"] if t.strip())
    tokens = tokenizer.encode(text, add_special_tokens=False)

    doc_len = (chunk_size + 1) * n_chunks
    n_possible = len(tokens) // doc_len
    n_docs = min(n_docs, n_possible)

    # Per-chunk PPL (to see if later chunks benefit from memory)
    chunk_losses = [[] for _ in range(n_chunks)]

    for doc_idx in range(n_docs):
        start = doc_idx * doc_len
        doc_tokens = tokens[start:start + doc_len]

        model.reset_state()

        for c in range(n_chunks):
            chunk_start = c * (chunk_size + 1)
            input_ids = torch.tensor(
                [doc_tokens[chunk_start:chunk_start + chunk_size]], device=device
            )
            target_ids = torch.tensor(
                [doc_tokens[chunk_start + 1:chunk_start + chunk_size + 1]], device=device
            )

            with torch.no_grad():
                if not use_memory:
                    model.reset_state()
                result = model.forward(input_ids, labels=target_ids,
                                       n_settle=1, return_errors=True)
                loss = result[1]

                if loss is not None and not torch.isnan(loss):
                    chunk_losses[c].append(loss.item())

    # Compute per-chunk PPL
    results = {}
    for c in range(n_chunks):
        if chunk_losses[c]:
            avg_loss = sum(chunk_losses[c]) / len(chunk_losses[c])
            results[f"chunk_{c+1}_ppl"] = math.exp(min(avg_loss, 20))

    # Overall
    all_losses = [l for chunk in chunk_losses for l in chunk]
    if all_losses:
        results["overall_ppl"] = math.exp(min(sum(all_losses) / len(all_losses), 20))

    # Memory benefit: compare first chunk vs last chunk
    if chunk_losses[0] and chunk_losses[-1]:
        first_ppl = math.exp(min(sum(chunk_losses[0]) / len(chunk_losses[0]), 20))
        last_ppl = math.exp(min(sum(chunk_losses[-1]) / len(chunk_losses[-1]), 20))
        results["first_chunk_ppl"] = first_ppl
        results["last_chunk_ppl"] = last_ppl
        results["memory_benefit"] = first_ppl - last_ppl

    return results


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predictive Transformer eval suite")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (if None, uses untrained model)")
    parser.add_argument("--tasks", nargs="+",
                        default=["synthetic", "babi", "lambada", "squad", "longdoc"],
                        help="Which tasks to run")
    parser.add_argument("--output-dir", type=str, default="results/eval_suite")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Max samples per task")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading Qwen2.5-0.5B...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PredictiveTransformer(qwen, n_mem_heads=2, state_dim=224)
    model.freeze_base()
    model = model.to(device)

    del qwen
    import gc; gc.collect(); torch.cuda.empty_cache()

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        print(f"  Loaded ({len(missing)} missing, {len(unexpected)} unexpected)")

    model.eval()
    results = {"checkpoint": args.checkpoint, "tasks": {}}

    # ─── Run tasks ───

    if "synthetic" in args.tasks:
        print("\n=== Synthetic QA ===")
        for use_mem in [True, False]:
            label = "with_memory" if use_mem else "no_memory"
            print(f"  {label}...")
            r = eval_synthetic_qa(model, tokenizer, device,
                                  n=args.n_samples, use_memory=use_mem)
            print(f"    accuracy: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
            results["tasks"][f"synthetic_{label}"] = r

    if "babi" in args.tasks:
        print("\n=== bAbI Tasks ===")
        for task_id in [1, 2, 3]:
            for use_mem in [True, False]:
                label = f"task{task_id}_{'memory' if use_mem else 'no_memory'}"
                print(f"  {label}...")
                r = eval_babi(model, tokenizer, device, task_id=task_id,
                             max_samples=args.n_samples, use_memory=use_mem)
                print(f"    accuracy: {r['accuracy']:.1%} ({r.get('correct',0)}/{r.get('total',0)})")
                results["tasks"][f"babi_{label}"] = r

    if "lambada" in args.tasks:
        print("\n=== LAMBADA ===")
        for use_mem in [True, False]:
            label = "with_memory" if use_mem else "no_memory"
            print(f"  {label}...")
            r = eval_lambada(model, tokenizer, device,
                            max_samples=args.n_samples, use_memory=use_mem)
            print(f"    accuracy: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
            results["tasks"][f"lambada_{label}"] = r

    if "squad" in args.tasks:
        print("\n=== SQuAD v2 ===")
        for use_mem in [True, False]:
            label = "with_memory" if use_mem else "no_memory"
            print(f"  {label}...")
            r = eval_squad(model, tokenizer, device,
                          max_samples=args.n_samples, use_memory=use_mem)
            print(f"    accuracy: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
            results["tasks"][f"squad_{label}"] = r

    if "longdoc" in args.tasks:
        print("\n=== WikiText Long-Doc PPL ===")
        for use_mem in [True, False]:
            label = "with_memory" if use_mem else "no_memory"
            print(f"  {label}...")
            r = eval_longdoc_ppl(model, tokenizer, device,
                                n_chunks=20, n_docs=min(args.n_samples, 20),
                                use_memory=use_mem)
            print(f"    overall PPL: {r.get('overall_ppl', 0):.1f}")
            print(f"    chunk 1 PPL: {r.get('first_chunk_ppl', 0):.1f}")
            print(f"    chunk 20 PPL: {r.get('last_chunk_ppl', 0):.1f}")
            print(f"    memory benefit: {r.get('memory_benefit', 0):+.1f}")
            results["tasks"][f"longdoc_{label}"] = r

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Task':<30} {'With Memory':>12} {'No Memory':>12} {'Delta':>8}")
    print("-" * 65)
    for task_base in ["synthetic", "babi_task1", "babi_task2", "babi_task3",
                      "lambada", "squad", "longdoc"]:
        mem_key = f"{task_base}_with_memory" if "babi" not in task_base else f"{task_base}_memory"
        nomem_key = f"{task_base}_no_memory"
        if mem_key in results["tasks"] and nomem_key in results["tasks"]:
            mem_r = results["tasks"][mem_key]
            nomem_r = results["tasks"][nomem_key]
            if "accuracy" in mem_r:
                m = mem_r["accuracy"]
                n = nomem_r["accuracy"]
                print(f"{task_base:<30} {m:>11.1%} {n:>11.1%} {m-n:>+7.1%}")
            elif "overall_ppl" in mem_r:
                m = mem_r["overall_ppl"]
                n = nomem_r["overall_ppl"]
                print(f"{task_base:<30} {m:>11.1f} {n:>11.1f} {m-n:>+7.1f}")


if __name__ == "__main__":
    main()
