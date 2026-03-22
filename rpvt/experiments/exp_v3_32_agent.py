"""v3.32: Agent training with Qwen2.5-3B-Instruct + RMT.

Trains on 4 levels:
  L1: Multi-passage QA (memory retrieval)
  L2: Multi-hop QA (reasoning chains)
  L3: Interactive tasks (search → answer)
  L4: Code execution (generate → run → debug)

Uses 3B instruct model for much stronger base capabilities.

Usage:
    python -m rpvt.experiments.exp_v3_32_agent
    python -m rpvt.experiments.exp_v3_32_agent --epochs 10
"""

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.recurrent_memory import RecurrentMemoryTransformer


# ─── Shared Data Utils ──────────────────────────────────────────────────

NAMES = ["Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry",
         "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
         "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander"]
COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella", "Cyberdyne",
             "Stark Industries", "Wayne Corp", "Oscorp", "Aperture",
             "Weyland", "Soylent", "Tyrell", "Massive Dynamic", "Dharma"]
CITIES = ["Tokyo", "Paris", "London", "Berlin", "Sydney", "Toronto",
          "Dubai", "Singapore", "Mumbai", "Seoul", "Cairo", "Rome"]
ROLES = ["engineer", "designer", "manager", "analyst", "researcher",
         "consultant", "director", "scientist", "architect", "coordinator"]
COUNTRY_MAP = {"Tokyo": "Japan", "Paris": "France", "London": "UK",
               "Berlin": "Germany", "Sydney": "Australia", "Toronto": "Canada",
               "Dubai": "UAE", "Singapore": "Singapore", "Mumbai": "India",
               "Seoul": "South Korea", "Cairo": "Egypt", "Rome": "Italy"}


def gen_person(rng):
    return {"name": rng.choice(NAMES), "company": rng.choice(COMPANIES),
            "city": rng.choice(CITIES), "role": rng.choice(ROLES)}


# ─── L1: Multi-Passage QA ───────────────────────────────────────────────

def gen_l1(rng, n_passages=3):
    used = set()
    people = []
    for _ in range(n_passages):
        p = gen_person(rng)
        while p["name"] in used:
            p = gen_person(rng)
        used.add(p["name"])
        people.append(p)

    passages = [f"{p['name']} works at {p['company']} in {p['city']} as a {p['role']}."
                for p in people]
    person = rng.choice(people)
    q_type = rng.choice(["company", "city", "role"])
    q_map = {"company": (f"What company does {person['name']} work for?", person["company"]),
             "city": (f"Where does {person['name']} work?", person["city"]),
             "role": (f"What is {person['name']}'s role?", person["role"])}
    q, a = q_map[q_type]
    return {"passages": passages, "question": q, "answer": a}


# ─── L2: Multi-Hop QA ───────────────────────────────────────────────────

def gen_l2(rng, n_hops=2):
    name = rng.choice(NAMES)
    company = rng.choice(COMPANIES)
    city = rng.choice(CITIES)
    country = COUNTRY_MAP.get(city, "Unknown")

    if n_hops == 2:
        passages = [f"{name} works at {company}.", f"{company} is headquartered in {city}."]
        q = f"In which city is {name}'s company headquartered?"
        a = city
    else:
        passages = [f"{name} works at {company}.", f"{company} is headquartered in {city}.",
                    f"{city} is located in {country}."]
        q = f"In which country does {name}'s company operate?"
        a = country

    # Distractors
    for _ in range(2):
        d = gen_person(rng)
        passages.insert(rng.randint(0, len(passages)),
                       f"{d['name']} works at {d['company']} in {d['city']}.")
    rng.shuffle(passages)
    return {"passages": passages, "question": q, "answer": a}


# ─── L3: Interactive Tasks ──────────────────────────────────────────────

def gen_l3_trajectory(rng):
    """Generate a demonstration trajectory for interactive search task."""
    people = [gen_person(rng) for _ in range(3)]
    target = rng.choice(people)
    fields = rng.sample(["company", "city", "role"], k=2)
    field_names = {"company": "company", "city": "location", "role": "role"}
    fields_str = " and ".join(field_names[f] for f in fields)

    task = f"Find out the {fields_str} of {target['name']}."

    # Build demonstration trajectory
    turns = []
    turns.append(f"Task: {task}")
    turns.append(f"Action: search {target['name']}")
    turns.append(f"Result: {target['name']} works at {target['company']} "
                 f"in {target['city']} as a {target['role']}.")

    answer_parts = [f"{field_names[f]}: {target[f]}" for f in fields]
    turns.append(f"Action: answer {', '.join(answer_parts)}")
    turns.append(f"Result: Correct! Reward: 1.0")

    return {"turns": turns, "target": target, "fields": fields,
            "task": task, "people": people}


def gen_l3_training_sample(rng):
    """Generate L3 training data: task + search + answer as text."""
    traj = gen_l3_trajectory(rng)
    # Passage chunks: task description + search result
    passage = f"{traj['turns'][0]}\n{traj['turns'][1]}\n{traj['turns'][2]}"

    # QA: the action to take
    target = traj["target"]
    fields = traj["fields"]
    field_names = {"company": "company", "city": "location", "role": "role"}
    answer_parts = [f"{field_names[f]}: {target[f]}" for f in fields]

    question = f"Based on the search results, provide the answer."
    answer = ", ".join(answer_parts)

    return {"passage": passage, "question": question, "answer": answer}


class InteractiveEnv:
    """Interactive search environment for evaluation."""

    def __init__(self, rng):
        people = [gen_person(rng) for _ in range(3)]
        self.facts = {p["name"]: p for p in people}
        target = rng.choice(people)
        self.target = target
        fields = rng.sample(["company", "city", "role"], k=2)
        self.fields = fields
        field_names = {"company": "company", "city": "location", "role": "role"}
        fields_str = " and ".join(field_names[f] for f in fields)
        self.task = f"Find out the {fields_str} of {target['name']}."
        self.steps = 0

    def reset(self):
        self.steps = 0
        return (f"Task: {self.task}\n"
                f"Available people: {', '.join(self.facts.keys())}\n"
                f"Actions: 'search <name>' or 'answer <response>'")

    def step(self, action):
        self.steps += 1
        action = action.strip()

        if "search" in action.lower():
            for name, info in self.facts.items():
                if name.lower() in action.lower():
                    return (f"Found: {name} works at {info['company']} "
                            f"in {info['city']} as a {info['role']}."), 0, False
            return "No results found.", 0, False

        elif "answer" in action.lower():
            correct = 0
            for f in self.fields:
                if self.target[f].lower() in action.lower():
                    correct += 1
            reward = correct / len(self.fields)
            return f"Reward: {reward:.1f}", reward, True

        if self.steps >= 5:
            return "Max steps.", 0, True
        return "Use 'search <name>' or 'answer <response>'.", 0, False


# ─── L4: Code Execution ─────────────────────────────────────────────────

CODE_TASKS = [
    {"prompt": "Write a Python function `solution(numbers)` that returns the sum of a list.",
     "test": "assert solution([1,2,3]) == 6 and solution([]) == 0",
     "solution": "def solution(numbers):\n    return sum(numbers)"},
    {"prompt": "Write a Python function `solution(numbers)` that returns the largest number.",
     "test": "assert solution([1,5,3]) == 5 and solution([-1,-5]) == -1",
     "solution": "def solution(numbers):\n    return max(numbers)"},
    {"prompt": "Write a Python function `solution(s)` that reverses a string.",
     "test": "assert solution('hello') == 'olleh' and solution('') == ''",
     "solution": "def solution(s):\n    return s[::-1]"},
    {"prompt": "Write a Python function `solution(s)` that counts vowels.",
     "test": "assert solution('hello') == 2 and solution('aeiou') == 5",
     "solution": "def solution(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"},
    {"prompt": "Write a Python function `solution(n)` that checks if n is prime.",
     "test": "assert solution(7) == True and solution(4) == False and solution(2) == True",
     "solution": "def solution(n):\n    if n < 2: return False\n    return all(n % i != 0 for i in range(2, int(n**0.5)+1))"},
    {"prompt": "Write a Python function `solution(n)` that returns factorial of n.",
     "test": "assert solution(5) == 120 and solution(0) == 1",
     "solution": "def solution(n):\n    if n <= 1: return 1\n    return n * solution(n-1)"},
    {"prompt": "Write a Python function `solution(lst)` that removes duplicates preserving order.",
     "test": "assert solution([1,2,2,3,1]) == [1,2,3]",
     "solution": "def solution(lst):\n    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]"},
    {"prompt": "Write a Python function `solution(n)` that returns the nth Fibonacci number.",
     "test": "assert solution(0) == 0 and solution(1) == 1 and solution(6) == 8",
     "solution": "def solution(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a+b\n    return a"},
]


def execute_code(code, test, timeout=5):
    try:
        result = subprocess.run(["python", "-c", code + "\n" + test],
                                capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stderr.strip().split("\n")[-1] if result.stderr else ""
    except:
        return False, "Error"


# ─── Training ───────────────────────────────────────────────────────────

def train_qa_batch(model, tokenizer, device, rng, level, n_passes=1):
    """Train on one QA sample (L1, L2, or L3)."""
    model.reset_memory()

    if level == "L1":
        n_p = rng.choice([1, 2, 3, 5])
        sample = gen_l1(rng, n_p)
        passages = sample["passages"]
    elif level == "L2":
        n_h = rng.choice([2, 3])
        sample = gen_l2(rng, n_h)
        passages = sample["passages"]
    elif level == "L3":
        sample = gen_l3_training_sample(rng)
        passages = [sample["passage"]]
    else:
        return None

    # Store passages in memory
    for p in passages:
        p_ids = tokenizer(p, return_tensors="pt", truncation=True,
                          max_length=128).input_ids.to(device)
        model.forward(p_ids, n_passes=1)

    # QA loss
    qa_text = f"Q: {sample['question']} A: {sample['answer']}"
    qa_ids = tokenizer(qa_text, return_tensors="pt", truncation=True,
                       max_length=128).input_ids.to(device)
    input_ids = qa_ids[:, :-1]
    labels = qa_ids[:, 1:]
    result = model.forward(input_ids, labels=labels, n_passes=n_passes)
    return result[1]


def train_code_batch(model, tokenizer, device, rng):
    """Train on code: show task + solution as supervised example."""
    model.reset_memory()
    task = rng.choice(CODE_TASKS)

    # Store task in memory
    task_text = f"Task: {task['prompt']}"
    t_ids = tokenizer(task_text, return_tensors="pt").input_ids.to(device)
    model.forward(t_ids, n_passes=1)

    # Train on solution
    sol_text = task["solution"]
    sol_ids = tokenizer(sol_text, return_tensors="pt", truncation=True,
                        max_length=128).input_ids.to(device)
    input_ids = sol_ids[:, :-1]
    labels = sol_ids[:, 1:]
    result = model.forward(input_ids, labels=labels, n_passes=1)
    return result[1]


# ─── Evaluation ─────────────────────────────────────────────────────────

def eval_qa(model, tokenizer, device, level, n_eval=30, seed=42, **kwargs):
    model.eval()
    rng = random.Random(seed)
    correct = total = 0

    for _ in range(n_eval):
        if level == "L1":
            sample = gen_l1(rng, kwargs.get("n_passages", 3))
            passages = sample["passages"]
        elif level == "L2":
            sample = gen_l2(rng, kwargs.get("n_hops", 2))
            passages = sample["passages"]
        else:
            break

        model.reset_memory()
        for p in passages:
            p_ids = tokenizer(p, return_tensors="pt", truncation=True,
                              max_length=128).input_ids.to(device)
            with torch.no_grad():
                model.forward(p_ids, n_passes=1)

        q_text = f"Q: {sample['question']} A:"
        q_ids = tokenizer(q_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(q_ids, max_new_tokens=20, n_passes=1)
        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
        if sample["answer"].lower() in answer.lower():
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def eval_interactive(model, tokenizer, device, n_eval=15, seed=42):
    model.eval()
    rng = random.Random(seed)
    total_reward = 0
    completed = 0

    for _ in range(n_eval):
        env = InteractiveEnv(rng)
        obs = env.reset()
        model.reset_memory()

        # Store task
        obs_ids = tokenizer(obs, return_tensors="pt", truncation=True,
                            max_length=256).input_ids.to(device)
        with torch.no_grad():
            model.forward(obs_ids, n_passes=1)

        done = False
        ep_reward = 0
        for step in range(5):
            prompt = "Based on the task, what action should I take? Action:"
            p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                gen = model.generate(p_ids, max_new_tokens=30, n_passes=1)
            action = tokenizer.decode(gen, skip_special_tokens=True).strip()
            action = action.split("\n")[0].strip()

            obs, reward, done = env.step(action)
            ep_reward = max(ep_reward, reward)

            # Store result
            result_text = f"Action: {action}\nResult: {obs}"
            r_ids = tokenizer(result_text, return_tensors="pt", truncation=True,
                              max_length=128).input_ids.to(device)
            with torch.no_grad():
                model.forward(r_ids, n_passes=1)

            if done:
                break

        total_reward += ep_reward
        if ep_reward > 0.5:
            completed += 1

    return {"avg_reward": total_reward / n_eval, "completion_rate": completed / n_eval,
            "completed": completed, "total": n_eval}


def eval_code(model, tokenizer, device, max_attempts=2):
    model.eval()
    correct = 0

    for task in CODE_TASKS:
        model.reset_memory()
        t_ids = tokenizer(f"Task: {task['prompt']}", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.forward(t_ids, n_passes=1)

        success = False
        for attempt in range(max_attempts):
            prompt = "def solution("
            p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                gen = model.generate(p_ids, max_new_tokens=80, n_passes=1)
            code = "def solution(" + tokenizer.decode(gen, skip_special_tokens=True)
            lines = []
            for line in code.split("\n"):
                lines.append(line)
                if line.strip().startswith("return"):
                    break
            code = "\n".join(lines)

            passed, error = execute_code(code, task["test"])
            if passed:
                success = True
                break
            else:
                err_text = f"Error: {error}\nCode: {code}"
                e_ids = tokenizer(err_text, return_tensors="pt", truncation=True,
                                  max_length=128).input_ids.to(device)
                with torch.no_grad():
                    model.forward(e_ids, n_passes=1)

        if success:
            correct += 1

    return {"accuracy": correct / len(CODE_TASKS), "correct": correct, "total": len(CODE_TASKS)}


def run_eval(model, tokenizer, device, label=""):
    print(f"\n{'='*60}")
    print(f"  EVALUATION {label}")
    print(f"{'='*60}")
    results = {}

    for n_p in [1, 3, 5]:
        r = eval_qa(model, tokenizer, device, "L1", n_eval=30, n_passages=n_p)
        print(f"  L1 ({n_p}p): {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
        results[f"L1_{n_p}p"] = r

    for n_h in [2, 3]:
        r = eval_qa(model, tokenizer, device, "L2", n_eval=30, n_hops=n_h)
        print(f"  L2 ({n_h}-hop): {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
        results[f"L2_{n_h}hop"] = r

    r = eval_interactive(model, tokenizer, device, n_eval=15)
    print(f"  L3 (interactive): reward={r['avg_reward']:.2f}, "
          f"completed={r['completion_rate']:.1%}")
    results["L3"] = r

    r = eval_code(model, tokenizer, device)
    print(f"  L4 (code): {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
    results["L4"] = r

    print(f"{'='*60}")
    return results


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/agent_3b")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True)

    model = RecurrentMemoryTransformer(
        qwen, n_memory_tokens=16, lora_rank=args.lora_rank, max_passes=3)
    model = model.to(device)
    del qwen
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    # Initial eval
    initial = run_eval(model, tokenizer, device, "Initial (untrained)")

    if args.eval_only:
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(initial, f, indent=2)
        return

    # Train
    print(f"\n--- Training ({args.epochs} epochs, all 4 levels) ---")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    history = []

    # Training mix: 40% L1, 25% L2, 20% L3, 15% L4
    level_weights = [("L1", 0.40), ("L2", 0.25), ("L3", 0.20), ("L4", 0.15)]

    for epoch in range(args.epochs):
        model.train()
        rng = random.Random(epoch + 42)
        losses = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}
        counts = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}
        t0 = time.time()
        optimizer.zero_grad()

        for step in range(args.n_train):
            # Pick level based on weights
            r = rng.random()
            cumulative = 0
            level = "L1"
            for lv, w in level_weights:
                cumulative += w
                if r < cumulative:
                    level = lv
                    break

            if level == "L4":
                loss = train_code_batch(model, tokenizer, device, rng)
            else:
                loss = train_qa_batch(model, tokenizer, device, rng, level)

            if loss is not None and not torch.isnan(loss):
                (loss / args.grad_accum).backward()
                losses[level] += loss.item()
                counts[level] += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % 150 == 0:
                elapsed = time.time() - t0
                speed = (step + 1) / elapsed
                eta = (args.n_train - step) / speed
                loss_str = " ".join(
                    f"{k}={losses[k]/max(counts[k],1):.3f}"
                    for k in ["L1", "L2", "L3", "L4"]
                )
                print(f"    [{step+1}/{args.n_train}] {loss_str} "
                      f"({speed:.1f} step/s, ETA {eta:.0f}s)")

        if (step + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - t0
        loss_str = " ".join(
            f"{k}={losses[k]/max(counts[k],1):.3f}"
            for k in ["L1", "L2", "L3", "L4"]
        )

        # Eval
        eval_results = run_eval(model, tokenizer, device, f"Epoch {epoch+1}")

        print(f"  Epoch {epoch+1}/{args.epochs}: {loss_str} ({elapsed:.0f}s)")
        history.append({
            "epoch": epoch + 1,
            "losses": {k: losses[k]/max(counts[k],1) for k in losses},
            "eval": eval_results,
        })

        # Save
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "history": history,
        }, output_dir / f"checkpoint_epoch{epoch+1}.pt")

    # Final
    final = run_eval(model, tokenizer, device, "Final")
    with open(output_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "history": history, "final": final}, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
