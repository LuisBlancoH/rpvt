"""v3.31: Agent capability levels — graduated difficulty testing.

Level 1: Multi-passage QA (memory capacity + selective retrieval)
Level 2: Multi-hop QA (reasoning through memory chains)
Level 3: Interactive tasks (actions with consequences)
Level 4: Code execution (generate → run → debug)

Trains on levels 1+2, evaluates on all 4.

Usage:
    python -m rpvt.experiments.exp_v3_31_agent_levels
    python -m rpvt.experiments.exp_v3_31_agent_levels --checkpoint results/rmt_qa/checkpoint_epoch15.pt
"""

import argparse
import json
import math
import random
import re
import subprocess
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.recurrent_memory import RecurrentMemoryTransformer


# ─── Level 1: Multi-Passage QA ─────────────────────────────────────────

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


def generate_person(rng):
    return {
        "name": rng.choice(NAMES),
        "company": rng.choice(COMPANIES),
        "city": rng.choice(CITIES),
        "role": rng.choice(ROLES),
    }


def generate_multi_passage_qa(n_passages, n_questions, rng):
    """Generate N passages about different people, then K questions."""
    people = []
    used_names = set()
    for _ in range(n_passages):
        p = generate_person(rng)
        while p["name"] in used_names:
            p = generate_person(rng)
        used_names.add(p["name"])
        people.append(p)

    passages = [
        f"{p['name']} works at {p['company']} in {p['city']} as a {p['role']}."
        for p in people
    ]

    questions = []
    for _ in range(n_questions):
        person = rng.choice(people)
        q_type = rng.choice(["company", "city", "role"])
        if q_type == "company":
            q = f"What company does {person['name']} work for?"
            a = person["company"]
        elif q_type == "city":
            q = f"Where does {person['name']} work?"
            a = person["city"]
        else:
            q = f"What is {person['name']}'s role?"
            a = person["role"]
        questions.append({"question": q, "answer": a})

    return {"passages": passages, "questions": questions, "people": people}


def eval_level1(model, tokenizer, device, n_eval=30, n_passages=3,
                n_questions=3, seed=42):
    """Evaluate multi-passage QA."""
    model.eval()
    rng = random.Random(seed)
    correct = 0
    total = 0

    for _ in range(n_eval):
        sample = generate_multi_passage_qa(n_passages, n_questions, rng)
        model.reset_memory()

        # Store all passages
        for passage in sample["passages"]:
            p_ids = tokenizer(passage, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                model.forward(p_ids, n_passes=1)

        # Answer questions
        for qa in sample["questions"]:
            q_text = f"Q: {qa['question']} A:"
            q_ids = tokenizer(q_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                gen = model.generate(q_ids, max_new_tokens=15, n_passes=1)
            answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
            if qa["answer"].lower() in answer.lower():
                correct += 1
            total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


# ─── Level 2: Multi-Hop QA ─────────────────────────────────────────────

def generate_multi_hop_qa(n_hops, rng):
    """Generate chain of facts requiring multi-hop reasoning."""
    if n_hops == 2:
        name = rng.choice(NAMES)
        company = rng.choice(COMPANIES)
        city = rng.choice(CITIES)

        passages = [
            f"{name} works at {company}.",
            f"{company} is headquartered in {city}.",
        ]
        question = f"In which city is {name}'s company headquartered?"
        answer = city

    elif n_hops == 3:
        name = rng.choice(NAMES)
        company = rng.choice(COMPANIES)
        city = rng.choice(CITIES)
        country_map = {"Tokyo": "Japan", "Paris": "France", "London": "UK",
                       "Berlin": "Germany", "Sydney": "Australia",
                       "Toronto": "Canada", "Dubai": "UAE",
                       "Singapore": "Singapore", "Mumbai": "India",
                       "Seoul": "South Korea", "Cairo": "Egypt", "Rome": "Italy"}
        country = country_map.get(city, "Unknown")

        passages = [
            f"{name} works at {company}.",
            f"{company} is headquartered in {city}.",
            f"{city} is located in {country}.",
        ]
        question = f"In which country does {name}'s company operate?"
        answer = country
    else:
        return generate_multi_hop_qa(2, rng)

    # Add distractor passages
    for _ in range(2):
        d = generate_person(rng)
        passages.insert(rng.randint(0, len(passages)),
                       f"{d['name']} works at {d['company']} in {d['city']}.")

    rng.shuffle(passages)
    return {"passages": passages, "question": question, "answer": answer}


def eval_level2(model, tokenizer, device, n_eval=30, n_hops=2, seed=42):
    """Evaluate multi-hop QA."""
    model.eval()
    rng = random.Random(seed)
    correct = 0
    total = 0

    for _ in range(n_eval):
        sample = generate_multi_hop_qa(n_hops, rng)
        model.reset_memory()

        # Store passages
        for passage in sample["passages"]:
            p_ids = tokenizer(passage, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                model.forward(p_ids, n_passes=1)

        # Answer (with settling for multi-hop)
        q_text = f"Q: {sample['question']} A:"
        q_ids = tokenizer(q_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(q_ids, max_new_tokens=15,
                                n_passes=min(n_hops, 3))
        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
        if sample["answer"].lower() in answer.lower():
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


# ─── Level 3: Interactive Tasks ─────────────────────────────────────────

class SimpleTaskEnvironment:
    """Simple text-based task environment."""

    def __init__(self, rng):
        self.rng = rng
        self.task = None
        self.facts = {}
        self.found = set()
        self.steps = 0
        self.max_steps = 5

    def reset(self):
        """Generate a new task."""
        self.steps = 0
        self.found = set()

        # Create a research task
        people = [generate_person(self.rng) for _ in range(3)]
        self.facts = {p["name"]: p for p in people}
        target = self.rng.choice(people)
        self.target_name = target["name"]
        self.target_info = target

        fields = self.rng.sample(["company", "city", "role"], k=2)
        self.required_fields = fields
        field_names = {"company": "company", "city": "location", "role": "role"}
        fields_str = " and ".join(field_names[f] for f in fields)

        self.task = f"Find out the {fields_str} of {self.target_name}."
        return f"Task: {self.task}\nAvailable actions: search <name>, answer <response>"

    def step(self, action):
        """Process an action, return (observation, reward, done)."""
        self.steps += 1
        action = action.strip().lower()

        if action.startswith("search"):
            name = action.replace("search", "").strip()
            # Find matching person
            for pname, info in self.facts.items():
                if pname.lower() in name.lower() or name.lower() in pname.lower():
                    return (f"Found: {pname} works at {info['company']} "
                            f"in {info['city']} as a {info['role']}."), 0, False
            return f"No results found for '{name}'.", 0, False

        elif action.startswith("answer"):
            answer = action.replace("answer", "").strip()
            # Check if answer contains required info
            correct_count = 0
            for field in self.required_fields:
                if self.target_info[field].lower() in answer.lower():
                    correct_count += 1
            reward = correct_count / len(self.required_fields)
            return f"Reward: {reward:.1f}", reward, True

        else:
            return "Unknown action. Use 'search <name>' or 'answer <response>'.", 0, False

        if self.steps >= self.max_steps:
            return "Max steps reached.", 0, True


def eval_level3(model, tokenizer, device, n_eval=20, seed=42):
    """Evaluate interactive task completion."""
    model.eval()
    rng = random.Random(seed)
    total_reward = 0
    completed = 0
    total = 0

    for _ in range(n_eval):
        env = SimpleTaskEnvironment(rng)
        obs = env.reset()
        model.reset_memory()

        # Store initial observation
        obs_ids = tokenizer(obs, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.forward(obs_ids, n_passes=1)

        done = False
        episode_reward = 0
        for step in range(env.max_steps):
            # Generate action
            prompt = f"Action:"
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                gen = model.generate(prompt_ids, max_new_tokens=30, n_passes=1)
            action = tokenizer.decode(gen, skip_special_tokens=True).strip()

            # Take first line as action
            action = action.split("\n")[0].strip()

            # Execute
            obs, reward, done = env.step(action)
            episode_reward = max(episode_reward, reward)

            # Store observation in memory
            obs_text = f"Action: {action}\nResult: {obs}"
            obs_ids = tokenizer(obs_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                model.forward(obs_ids, n_passes=1)

            if done:
                break

        total_reward += episode_reward
        if episode_reward > 0.5:
            completed += 1
        total += 1

    return {
        "avg_reward": total_reward / max(total, 1),
        "completion_rate": completed / max(total, 1),
        "completed": completed,
        "total": total,
    }


# ─── Level 4: Code Execution ───────────────────────────────────────────

CODE_TASKS = [
    {
        "prompt": "Write a Python function that returns the sum of a list of numbers.",
        "test": "assert solution([1,2,3]) == 6 and solution([]) == 0 and solution([10]) == 10",
        "hint": "def solution(numbers):",
    },
    {
        "prompt": "Write a Python function that returns the largest number in a list.",
        "test": "assert solution([1,5,3]) == 5 and solution([-1,-5,-3]) == -1",
        "hint": "def solution(numbers):",
    },
    {
        "prompt": "Write a Python function that reverses a string.",
        "test": "assert solution('hello') == 'olleh' and solution('') == ''",
        "hint": "def solution(s):",
    },
    {
        "prompt": "Write a Python function that counts vowels in a string.",
        "test": "assert solution('hello') == 2 and solution('aeiou') == 5 and solution('xyz') == 0",
        "hint": "def solution(s):",
    },
    {
        "prompt": "Write a Python function that checks if a number is prime.",
        "test": "assert solution(7) == True and solution(4) == False and solution(2) == True",
        "hint": "def solution(n):",
    },
    {
        "prompt": "Write a Python function that returns the factorial of n.",
        "test": "assert solution(5) == 120 and solution(0) == 1 and solution(1) == 1",
        "hint": "def solution(n):",
    },
    {
        "prompt": "Write a Python function that removes duplicates from a list while preserving order.",
        "test": "assert solution([1,2,2,3,1]) == [1,2,3] and solution([]) == []",
        "hint": "def solution(lst):",
    },
    {
        "prompt": "Write a Python function that returns the nth Fibonacci number (0-indexed).",
        "test": "assert solution(0) == 0 and solution(1) == 1 and solution(6) == 8",
        "hint": "def solution(n):",
    },
]


def execute_code(code, test, timeout=5):
    """Execute code + test, return (success, output)."""
    full_code = code + "\n" + test
    try:
        result = subprocess.run(
            ["python", "-c", full_code],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return True, "All tests passed!"
        else:
            error = result.stderr.strip().split("\n")[-1] if result.stderr else "Unknown error"
            return False, error
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def eval_level4(model, tokenizer, device, n_eval=None, max_attempts=3, seed=42):
    """Evaluate code generation with execution feedback."""
    model.eval()
    rng = random.Random(seed)

    tasks = CODE_TASKS if n_eval is None else CODE_TASKS[:n_eval]
    correct = 0
    total = 0

    for task in tasks:
        model.reset_memory()

        # Store task description
        task_text = f"Task: {task['prompt']}\nStart with: {task['hint']}"
        task_ids = tokenizer(task_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.forward(task_ids, n_passes=1)

        success = False
        for attempt in range(max_attempts):
            # Generate code
            prompt = f"{task['hint']}"
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                gen = model.generate(prompt_ids, max_new_tokens=100, n_passes=1)
            code = task["hint"] + tokenizer.decode(gen, skip_special_tokens=True)

            # Clean up code (take first function definition)
            lines = code.split("\n")
            clean_lines = []
            for line in lines:
                clean_lines.append(line)
                if line.strip().startswith("return"):
                    break
            code = "\n".join(clean_lines)

            # Execute
            passed, output = execute_code(code, task["test"])

            if passed:
                success = True
                break
            else:
                # Store error in memory for next attempt
                error_text = f"Attempt {attempt+1} failed: {output}\nCode: {code}"
                err_ids = tokenizer(error_text, return_tensors="pt",
                                    truncation=True, max_length=128).input_ids.to(device)
                with torch.no_grad():
                    model.forward(err_ids, n_passes=1)

        if success:
            correct += 1
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


# ─── Training: Multi-Passage + Multi-Hop QA ─────────────────────────────

def train_level1_batch(model, tokenizer, device, rng, n_passages=3):
    """Train on one multi-passage QA sample."""
    sample = generate_multi_passage_qa(n_passages, 1, rng)
    model.reset_memory()

    # Store passages
    for passage in sample["passages"]:
        p_ids = tokenizer(passage, return_tensors="pt").input_ids.to(device)
        model.forward(p_ids, n_passes=1)

    # QA loss
    qa = sample["questions"][0]
    qa_text = f"Q: {qa['question']} A: {qa['answer']}"
    qa_ids = tokenizer(qa_text, return_tensors="pt").input_ids.to(device)
    input_ids = qa_ids[:, :-1]
    labels = qa_ids[:, 1:]
    result = model.forward(input_ids, labels=labels, n_passes=1)
    return result[1]


def train_level2_batch(model, tokenizer, device, rng, n_hops=2):
    """Train on one multi-hop QA sample."""
    sample = generate_multi_hop_qa(n_hops, rng)
    model.reset_memory()

    for passage in sample["passages"]:
        p_ids = tokenizer(passage, return_tensors="pt").input_ids.to(device)
        model.forward(p_ids, n_passes=1)

    qa_text = f"Q: {sample['question']} A: {sample['answer']}"
    qa_ids = tokenizer(qa_text, return_tensors="pt").input_ids.to(device)
    input_ids = qa_ids[:, :-1]
    labels = qa_ids[:, 1:]
    result = model.forward(input_ids, labels=labels, n_passes=min(n_hops, 3))
    return result[1]


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Start from checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=1500,
                        help="Training samples per epoch")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--output-dir", type=str,
                        default="results/agent_levels")
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
        qwen, n_memory_tokens=16, lora_rank=args.lora_rank, max_passes=3,
    )
    model = model.to(device)
    del qwen
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("  Loaded")

    # ─── Evaluate all levels ───
    def run_eval(label=""):
        print(f"\n{'='*60}")
        print(f"  EVALUATION {label}")
        print(f"{'='*60}")

        results = {}

        # Level 1: Multi-passage QA
        for n_p in [1, 3, 5]:
            r = eval_level1(model, tokenizer, device, n_eval=30,
                           n_passages=n_p, n_questions=2)
            print(f"  Level 1 ({n_p} passages): {r['accuracy']:.1%} "
                  f"({r['correct']}/{r['total']})")
            results[f"L1_{n_p}p"] = r

        # Level 2: Multi-hop QA
        for n_h in [2, 3]:
            r = eval_level2(model, tokenizer, device, n_eval=30, n_hops=n_h)
            print(f"  Level 2 ({n_h}-hop): {r['accuracy']:.1%} "
                  f"({r['correct']}/{r['total']})")
            results[f"L2_{n_h}hop"] = r

        # Level 3: Interactive tasks
        r = eval_level3(model, tokenizer, device, n_eval=15)
        print(f"  Level 3 (interactive): reward={r['avg_reward']:.2f}, "
              f"completed={r['completion_rate']:.1%} ({r['completed']}/{r['total']})")
        results["L3_interactive"] = r

        # Level 4: Code execution
        r = eval_level4(model, tokenizer, device)
        print(f"  Level 4 (code): {r['accuracy']:.1%} "
              f"({r['correct']}/{r['total']})")
        results["L4_code"] = r

        print(f"{'='*60}\n")
        return results

    # Initial evaluation
    eval_results = run_eval("Initial (before training)")

    if args.eval_only:
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Saved to {output_dir}")
        return

    # ─── Train on Level 1 + Level 2 ───
    print(f"\n--- Training ({args.epochs} epochs) ---")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    history = []

    for epoch in range(args.epochs):
        model.train()
        rng = random.Random(epoch + 42)
        total_l1_loss = 0
        total_l2_loss = 0
        n_l1 = 0
        n_l2 = 0
        t0 = time.time()
        optimizer.zero_grad()

        for step in range(args.n_train):
            # Alternate between levels
            if step % 3 == 0:
                # Level 2: multi-hop (harder, 1/3 of training)
                n_hops = random.choice([2, 3])
                loss = train_level2_batch(model, tokenizer, device, rng, n_hops)
                if loss is not None and not torch.isnan(loss):
                    (loss / args.grad_accum).backward()
                    total_l2_loss += loss.item()
                    n_l2 += 1
            else:
                # Level 1: multi-passage (easier, 2/3 of training)
                n_passages = random.choice([1, 2, 3, 5])
                loss = train_level1_batch(model, tokenizer, device, rng, n_passages)
                if loss is not None and not torch.isnan(loss):
                    (loss / args.grad_accum).backward()
                    total_l1_loss += loss.item()
                    n_l1 += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % 200 == 0:
                avg_l1 = total_l1_loss / max(n_l1, 1)
                avg_l2 = total_l2_loss / max(n_l2, 1)
                elapsed = time.time() - t0
                speed = (step + 1) / elapsed
                eta = (args.n_train - step) / speed
                print(f"    [{step+1}/{args.n_train}] "
                      f"L1_loss={avg_l1:.4f} L2_loss={avg_l2:.4f} "
                      f"({speed:.1f} step/s, ETA {eta:.0f}s)")

        # Flush
        if (step + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - t0
        avg_l1 = total_l1_loss / max(n_l1, 1)
        avg_l2 = total_l2_loss / max(n_l2, 1)

        # Eval
        eval_results = run_eval(f"Epoch {epoch+1}")

        print(f"  Epoch {epoch+1}/{args.epochs}: "
              f"L1_loss={avg_l1:.4f}, L2_loss={avg_l2:.4f}, "
              f"L1_1p={eval_results['L1_1p']['accuracy']:.1%}, "
              f"L1_3p={eval_results['L1_3p']['accuracy']:.1%}, "
              f"L1_5p={eval_results['L1_5p']['accuracy']:.1%}, "
              f"L2_2hop={eval_results['L2_2hop']['accuracy']:.1%}, "
              f"L2_3hop={eval_results['L2_3hop']['accuracy']:.1%}, "
              f"({elapsed:.0f}s)")

        history.append({
            "epoch": epoch + 1,
            "l1_loss": avg_l1,
            "l2_loss": avg_l2,
            "eval": {k: v for k, v in eval_results.items()},
        })

        # Save checkpoint
        ckpt = output_dir / f"checkpoint_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "history": history,
        }, ckpt)

    # Final eval
    final_results = run_eval("Final")

    with open(output_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "history": history,
                   "final": final_results}, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
