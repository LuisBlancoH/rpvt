"""v3.30: Train Recurrent Memory Transformer on Instruct + QA.

Qwen2.5-0.5B-Instruct + LoRA (all linear) + memory extraction.
Train on mix of WikiText (general LM) + synthetic QA (memory retrieval).

Usage:
    python -m rpvt.experiments.exp_v3_30_rmt_qa
    python -m rpvt.experiments.exp_v3_30_rmt_qa --epochs 15
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.recurrent_memory import RecurrentMemoryTransformer


# ─── Data ───────────────────────────────────────────────────────────────

NAMES = [
    "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
]
COMPANIES = [
    "Acme Corp", "Globex", "Initech", "Umbrella", "Cyberdyne",
    "Stark Industries", "Wayne Corp", "Oscorp", "Aperture",
    "Weyland", "Soylent", "Tyrell", "Massive Dynamic", "Dharma",
]
CITIES = [
    "Tokyo", "Paris", "London", "Berlin", "Sydney", "Toronto",
    "Dubai", "Singapore", "Mumbai", "Seoul", "Cairo", "Rome",
    "Vienna", "Oslo", "Prague", "Lima", "Nairobi", "Bangkok",
]
ROLES = [
    "engineer", "designer", "manager", "analyst", "researcher",
    "consultant", "director", "scientist", "architect", "coordinator",
]
FIELDS = [
    "artificial intelligence", "renewable energy", "biotechnology",
    "cybersecurity", "quantum computing", "data science",
    "robotics", "aerospace", "finance", "healthcare",
]

PASSAGE_TEMPLATES = [
    "{name} works at {company} in {city} as a {role}.",
    "{name} is a {role} at {company}, based in {city}.",
    "{name} joined {company} in {city}. Their role is {role}.",
    "In {city}, {name} serves as a {role} for {company}.",
    "{name}, a {role} specializing in {field}, works at {company} in {city}.",
]

QA_TEMPLATES = [
    ("What company does {name} work for?", "{company}"),
    ("Where does {name} work?", "{city}"),
    ("What is {name}'s role?", "{role}"),
    ("Who works at {company}?", "{name}"),
    ("Which city is {name} based in?", "{city}"),
]


def generate_qa_sample(rng):
    name = rng.choice(NAMES)
    company = rng.choice(COMPANIES)
    city = rng.choice(CITIES)
    role = rng.choice(ROLES)
    field = rng.choice(FIELDS)
    passage = rng.choice(PASSAGE_TEMPLATES).format(
        name=name, company=company, city=city, role=role, field=field
    )
    q_template, a_template = rng.choice(QA_TEMPLATES)
    question = q_template.format(name=name, company=company)
    answer = a_template.format(name=name, company=company, city=city, role=role)
    return {"passage": passage, "question": question, "answer": answer}


class QADataset(Dataset):
    def __init__(self, tokenizer, n_samples=2000, seed=42):
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.samples = [generate_qa_sample(self.rng) for _ in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        passage_ids = self.tokenizer.encode(sample["passage"], add_special_tokens=False)
        qa_text = f"Q: {sample['question']} A: {sample['answer']}"
        qa_ids = self.tokenizer.encode(qa_text, add_special_tokens=False)
        return {
            "passage_ids": torch.tensor(passage_ids, dtype=torch.long),
            "qa_ids": torch.tensor(qa_ids, dtype=torch.long),
            "answer": sample["answer"],
            "question": sample["question"],
        }


class WikiTextDataset(Dataset):
    def __init__(self, token_ids, seq_len=128):
        n = len(token_ids) // (seq_len + 1)
        self.data = token_ids[:n * (seq_len + 1)].reshape(n, seq_len + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][:-1], self.data[idx][1:]


# ─── Training ───────────────────────────────────────────────────────────

def train_qa_batch(model, sample, device):
    model.reset_memory()
    # Chunk 1: passage (store in memory)
    p_ids = sample["passage_ids"].unsqueeze(0).to(device)
    model.forward(p_ids, n_passes=1)
    # Chunk 2: QA (use memory)
    qa_ids = sample["qa_ids"].unsqueeze(0).to(device)
    input_ids = qa_ids[:, :-1]
    labels = qa_ids[:, 1:]
    result = model.forward(input_ids, labels=labels, n_passes=1)
    return result[1]  # loss


def train_wiki_batch(model, input_ids, targets, device):
    model.reset_memory()
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    result = model.forward(input_ids, labels=targets, n_passes=1)
    return result[1]


def evaluate_qa(model, tokenizer, device, n=50, seed=99):
    model.eval()
    rng = random.Random(seed)
    correct = 0
    total = 0
    for _ in range(n):
        sample = generate_qa_sample(rng)
        model.reset_memory()
        # Store passage
        p_ids = tokenizer(sample["passage"], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.forward(p_ids, n_passes=1)
        # Ask question
        q_text = f"Q: {sample['question']} A:"
        q_ids = tokenizer(q_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(q_ids, max_new_tokens=15, n_passes=1)
        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
        if sample["answer"].lower() in answer.lower():
            correct += 1
        total += 1
    return correct / total, correct, total


def evaluate_qa_no_memory(model, tokenizer, device, n=50, seed=99):
    model.eval()
    rng = random.Random(seed)
    correct = 0
    total = 0
    for _ in range(n):
        sample = generate_qa_sample(rng)
        model.reset_memory()
        # Skip passage
        q_text = f"Q: {sample['question']} A:"
        q_ids = tokenizer(q_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(q_ids, max_new_tokens=15, n_passes=1)
        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
        if sample["answer"].lower() in answer.lower():
            correct += 1
        total += 1
    return correct / total, correct, total


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-qa", type=int, default=2000)
    parser.add_argument("--wiki-ratio", type=float, default=0.3)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--n-memory-tokens", type=int, default=16)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="results/rmt_qa")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading Qwen2.5-0.5B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
    )
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = RecurrentMemoryTransformer(
        qwen,
        n_memory_tokens=args.n_memory_tokens,
        lora_rank=args.lora_rank,
        max_passes=args.max_passes,
    )
    model = model.to(device)
    del qwen
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    # Data
    print("Loading data...")
    qa_dataset = QADataset(tokenizer, n_samples=args.n_qa, seed=42)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    wiki_text = "\n".join(t for t in ds["train"]["text"] if t.strip())
    wiki_tokens = torch.tensor(
        tokenizer.encode(wiki_text, add_special_tokens=False), dtype=torch.long
    )
    wiki_dataset = WikiTextDataset(wiki_tokens, seq_len=128)
    wiki_loader = DataLoader(wiki_dataset, batch_size=1, shuffle=True, drop_last=True)
    print(f"  QA: {len(qa_dataset)} samples, Wiki: {len(wiki_dataset)} chunks")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Initial eval
    print("\n--- Initial (no training) ---")
    acc, c, t = evaluate_qa(model, tokenizer, device, n=30)
    acc_no, c_no, t_no = evaluate_qa_no_memory(model, tokenizer, device, n=30)
    print(f"  QA with memory:    {acc:.1%} ({c}/{t})")
    print(f"  QA without memory: {acc_no:.1%} ({c_no}/{t_no})")

    model.reset_memory()
    p_ids = tokenizer("Alice works at Acme Corp in Tokyo.", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        model.forward(p_ids, n_passes=1)
    q_ids = tokenizer("Q: What company does Alice work for? A:", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        gen = model.generate(q_ids, max_new_tokens=15, n_passes=1)
    print(f"  Gen: Q: What company does Alice work for? A: {tokenizer.decode(gen, skip_special_tokens=True)}")

    # Train
    print(f"\n--- Training ({args.epochs} epochs) ---")
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_qa_loss = 0
        total_wiki_loss = 0
        n_qa = 0
        n_wiki = 0
        t0 = time.time()
        optimizer.zero_grad()

        qa_indices = list(range(len(qa_dataset)))
        random.shuffle(qa_indices)
        wiki_iter = iter(wiki_loader)

        step = 0
        for qa_idx in qa_indices:
            sample = qa_dataset[qa_idx]
            qa_loss = train_qa_batch(model, sample, device)

            if qa_loss is not None and not torch.isnan(qa_loss):
                (qa_loss / args.grad_accum).backward()
                total_qa_loss += qa_loss.item()
                n_qa += 1

            if random.random() < args.wiki_ratio:
                try:
                    wiki_input, wiki_target = next(wiki_iter)
                except StopIteration:
                    wiki_iter = iter(wiki_loader)
                    wiki_input, wiki_target = next(wiki_iter)
                wiki_loss = train_wiki_batch(model, wiki_input, wiki_target, device)
                if wiki_loss is not None and not torch.isnan(wiki_loss):
                    (wiki_loss / args.grad_accum).backward()
                    total_wiki_loss += wiki_loss.item()
                    n_wiki += 1

            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                avg_qa = total_qa_loss / max(n_qa, 1)
                avg_wiki = total_wiki_loss / max(n_wiki, 1)
                elapsed = time.time() - t0
                speed = step / elapsed
                eta = (len(qa_indices) - step) / speed if speed > 0 else 0
                print(f"    [{step}/{len(qa_indices)}] "
                      f"qa_loss={avg_qa:.4f} wiki_loss={avg_wiki:.4f} "
                      f"mem={model.memory_buffer.n_stored}/{model.memory_buffer.max_entries} "
                      f"({speed:.1f} step/s, ETA {eta:.0f}s)")

        # Flush
        if step % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - t0
        avg_qa = total_qa_loss / max(n_qa, 1)
        avg_wiki = total_wiki_loss / max(n_wiki, 1)

        # Eval
        acc, c, t = evaluate_qa(model, tokenizer, device, n=50)
        acc_no, c_no, t_no = evaluate_qa_no_memory(model, tokenizer, device, n=50)

        print(f"  Epoch {epoch+1}/{args.epochs}: "
              f"qa_loss={avg_qa:.4f}, wiki_loss={avg_wiki:.4f}, "
              f"QA_mem={acc:.1%}, QA_no_mem={acc_no:.1%}, "
              f"delta={acc-acc_no:+.1%}, ({elapsed:.0f}s)")

        history.append({
            "epoch": epoch + 1,
            "qa_loss": avg_qa,
            "wiki_loss": avg_wiki,
            "qa_accuracy_memory": acc,
            "qa_accuracy_no_memory": acc_no,
            "qa_delta": acc - acc_no,
        })

        # Save checkpoint
        ckpt = output_dir / f"checkpoint_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": {
                k: v.cpu() for k, v in model.state_dict().items()
                if v.requires_grad or "lora" in k or "extractor" in k
                or "mem_attn" in k or "memory_gate" in k
            },
            "history": history,
        }, ckpt)

        # Sample generations every 3 epochs
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"\n  --- Generations (epoch {epoch+1}) ---")
            for i in range(3):
                rng = random.Random(i + 100)
                s = generate_qa_sample(rng)
                model.reset_memory()
                p_ids = tokenizer(s["passage"], return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    model.forward(p_ids, n_passes=1)
                q_text = f"Q: {s['question']} A:"
                q_ids = tokenizer(q_text, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    gen = model.generate(q_ids, max_new_tokens=15, n_passes=1)
                gen_text = tokenizer.decode(gen, skip_special_tokens=True)
                ok = "OK" if s["answer"].lower() in gen_text.lower() else "WRONG"
                print(f"    P: {s['passage']}")
                print(f"    Q: {s['question']}")
                print(f"    A: {gen_text[:60]}  [{ok}, expected: {s['answer']}]")
                print()

    # Final
    print("\n--- Final ---")
    acc, c, t = evaluate_qa(model, tokenizer, device, n=100)
    acc_no, c_no, t_no = evaluate_qa_no_memory(model, tokenizer, device, n=100)
    print(f"  QA with memory:    {acc:.1%} ({c}/{t})")
    print(f"  QA without memory: {acc_no:.1%} ({c_no}/{t_no})")
    print(f"  Memory delta:      {acc-acc_no:+.1%}")

    with open(output_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
