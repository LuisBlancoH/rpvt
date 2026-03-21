"""Overnight experiment suite: explore all paths systematically.

Runs sequentially on single GPU:
1. Fix recurrent depth eval — check what model actually generates
2. Pre-train inverse model on diverse forward passes
3. Test pre-trained inverse predictions via cross-attention
4. Test settling with pre-trained inverse
5. Compare baseline vs settled accuracy

Results saved to results/overnight/
"""

import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("results/overnight")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


# ── Task generation ──────────────────────────────────────────

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
            table = ", ".join(f"{n}={v}" for n, v in items)
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


def tokenize_task(task, tokenizer, device):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": task["prompt"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=256).to(device)
    return ids


def generate_answer(model, ids, tokenizer, max_tokens=30):
    """Generate and return text."""
    gen_ids = ids["input_ids"].clone()
    gen_mask = ids["attention_mask"].clone()
    tokens = []
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(gen_ids, attention_mask=gen_mask, use_cache=False)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_tok.item() == tokenizer.eos_token_id:
            break
        tokens.append(next_tok.item())
        gen_ids = torch.cat([gen_ids, next_tok], dim=1)
        gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=gen_mask.device,
                              dtype=gen_mask.dtype)], dim=1)
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def check_answer(generated, gold):
    gen_lower = generated.lower().strip()
    gold_lower = gold.lower().strip()
    if gold_lower in gen_lower:
        return True
    # Check if gold appears as a word
    for word in gen_lower.split():
        if word.strip(".,!?;:") == gold_lower:
            return True
    return False


# ── Phase 1: Baseline + check what model generates ──────────

def phase1_baseline(model, tokenizer, tasks, device):
    log("=== PHASE 1: Baseline evaluation ===")
    correct = 0
    results = []

    for i, task in enumerate(tasks):
        ids = tokenize_task(task, tokenizer, device)
        answer = generate_answer(model, ids, tokenizer)
        is_correct = check_answer(answer, task["answer"])
        if is_correct:
            correct += 1
        results.append({
            "type": task["type"],
            "gold": task["answer"],
            "generated": answer[:100],
            "correct": is_correct,
        })
        if i < 5:
            log(f"  [{i+1}] {task['type']:12s} gold='{task['answer']}' "
                f"gen='{answer[:50]}' {'✓' if is_correct else '✗'}")

    acc = 100 * correct / len(tasks)
    log(f"  Baseline: {correct}/{len(tasks)} ({acc:.1f}%)")

    # By type
    for t in sorted(set(r["type"] for r in results)):
        tr = [r for r in results if r["type"] == t]
        tc = sum(1 for r in tr if r["correct"])
        log(f"    {t:12s}: {tc}/{len(tr)}")

    return results, correct


# ── Phase 2: Pre-train inverse model ────────────────────────

def phase2_pretrain_inverse(model, tokenizer, device, n_train=200,
                            n_epochs=10):
    log("=== PHASE 2: Pre-train inverse model ===")
    from rpvt.model.active_inference import InverseModel

    hidden_size = model.config.hidden_size

    # Create inverse
    inverse = InverseModel(
        hidden_size=hidden_size, n_layers=2,
    ).to(dtype=torch.bfloat16, device=device)

    n_params = sum(p.numel() for p in inverse.parameters())
    log(f"  Inverse params: {n_params:,}")

    optimizer = torch.optim.Adam(inverse.parameters(), lr=1e-3)

    # Capture hooks for layers 14 and 27
    captured = {}

    def get_layers():
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        return model.model.model.layers

    layers = get_layers()
    hooks = []
    for idx in [14, 27]:
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                h = output[0]
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                captured[layer_idx] = h.detach()
            return hook_fn
        hooks.append(layers[idx].register_forward_hook(make_hook(idx)))

    # Generate diverse training texts
    log("  Generating training data...")
    train_texts = []
    rng = random.Random(42)
    # Mix of prompts
    templates = [
        "The capital of {} is",
        "{} works as a {} in {}.",
        "In the year {}, {} happened.",
        "The sequence {}, {}, {} continues with",
        "If {} equals {} and {} equals {}, then",
    ]
    names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank",
             "Grace", "Henry", "Ivy", "Jack", "Karen", "Leo"]
    cities = ["NYC", "London", "Tokyo", "Paris", "Berlin", "Sydney"]
    jobs = ["engineer", "teacher", "doctor", "artist", "chef", "pilot"]

    for _ in range(n_train):
        template = rng.choice(templates)
        if "{}" in template:
            n_blanks = template.count("{}")
            fillers = [rng.choice(names + cities + jobs + [str(rng.randint(1, 1000))])
                       for _ in range(n_blanks)]
            text = template.format(*fillers)
        else:
            text = template
        train_texts.append(text)

    # Also add the actual task prompts
    tasks = generate_tasks(50, seed=99)
    for task in tasks:
        train_texts.append(task["prompt"])

    log(f"  Training texts: {len(train_texts)}")

    # Train
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        random.shuffle(train_texts)

        for text in train_texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=128).to(device)

            # Forward pass (frozen)
            captured.clear()
            with torch.no_grad():
                model(ids["input_ids"], use_cache=False)

            if 14 not in captured or 27 not in captured:
                continue

            source_h = captured[27]
            target_h = captured[14]

            # Inverse prediction
            predicted, _ = inverse.predict(source_h)
            loss = F.mse_loss(predicted, target_h)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inverse.parameters(), 1.0)
            optimizer.step()

            # Reset GRU state every few samples to prevent accumulation issues
            inverse.reset_state()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        log(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Save inverse
    save_path = OUTPUT_DIR / "pretrained_inverse.pt"
    torch.save(inverse.state_dict(), save_path)
    log(f"  Saved inverse to {save_path}")

    return inverse


# ── Phase 3: Test pre-trained inverse via cross-attention ───

def phase3_test_cross_attention(model, tokenizer, inverse, tasks, device):
    log("=== PHASE 3: Cross-attention with pre-trained inverse ===")
    from rpvt.model.active_inference_v2 import ActiveInferenceSettler

    settler = ActiveInferenceSettler(
        model,
        hidden_size=model.config.hidden_size,
        inverse_lr=1e-4,  # slower LR since already pretrained
    )

    # Load pre-trained weights
    settler.inverse.load_state_dict(inverse.state_dict())
    log("  Loaded pre-trained inverse weights")

    correct = 0
    error_decreased = 0
    results = []

    for i, task in enumerate(tasks):
        settler.reset()
        ids = tokenize_task(task, tokenizer, device)

        # Settle (3 steps)
        error_history, info = settler.settle(
            ids["input_ids"],
            attention_mask=ids["attention_mask"],
            n_steps=3,
        )

        # Generate after settling
        settler.reset()
        settler.inverse.load_state_dict(inverse.state_dict())
        gen_tokens = settler.generate(
            ids["input_ids"],
            attention_mask=ids["attention_mask"],
            max_new_tokens=30,
            n_settle=3,
        )
        answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        is_correct = check_answer(answer, task["answer"])
        if is_correct:
            correct += 1
        if error_history[-1] < error_history[0]:
            error_decreased += 1

        results.append({
            "type": task["type"],
            "gold": task["answer"],
            "generated": answer[:100],
            "correct": is_correct,
            "error_history": error_history,
            "error_decreased": error_history[-1] < error_history[0],
        })

        if i < 5:
            arrow = "↓" if error_history[-1] < error_history[0] else "↑"
            log(f"  [{i+1}] {task['type']:12s} gold='{task['answer']}' "
                f"gen='{answer[:40]}' err:{error_history[0]:.1f}→{error_history[-1]:.1f}{arrow} "
                f"{'✓' if is_correct else '✗'}")

    acc = 100 * correct / len(tasks)
    log(f"  Pre-trained settled: {correct}/{len(tasks)} ({acc:.1f}%)")
    log(f"  Error decreased: {error_decreased}/{len(tasks)}")

    settler.remove_hooks()
    return results, correct


# ── Phase 4: Ablation — more settling steps ─────────────────

def phase4_settling_ablation(model, tokenizer, inverse, tasks, device):
    log("=== PHASE 4: Settling steps ablation (1, 3, 5, 10) ===")
    from rpvt.model.active_inference_v2 import ActiveInferenceSettler

    results_by_steps = {}

    for n_steps in [1, 3, 5, 10]:
        settler = ActiveInferenceSettler(
            model,
            hidden_size=model.config.hidden_size,
            inverse_lr=1e-4,
        )
        settler.inverse.load_state_dict(inverse.state_dict())

        correct = 0
        for task in tasks[:20]:  # subset for speed
            settler.reset()
            settler.inverse.load_state_dict(inverse.state_dict())
            ids = tokenize_task(task, tokenizer, device)

            gen_tokens = settler.generate(
                ids["input_ids"],
                attention_mask=ids["attention_mask"],
                max_new_tokens=30,
                n_settle=n_steps,
            )
            answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            if check_answer(answer, task["answer"]):
                correct += 1

        acc = 100 * correct / 20
        results_by_steps[n_steps] = {"correct": correct, "total": 20, "accuracy": acc}
        log(f"  n_settle={n_steps}: {correct}/20 ({acc:.1f}%)")

        settler.remove_hooks()

    return results_by_steps


# ── Phase 5: Inverse prediction quality analysis ────────────

def phase5_prediction_quality(model, tokenizer, inverse, tasks, device):
    log("=== PHASE 5: Inverse prediction quality ===")

    captured = {}
    def get_layers():
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        return model.model.model.layers

    layers = get_layers()
    hooks = []
    for idx in [14, 27]:
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                h = output[0]
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                captured[layer_idx] = h.detach()
            return hook_fn
        hooks.append(layers[idx].register_forward_hook(make_hook(idx)))

    errors = []
    cosine_sims = []

    for task in tasks[:30]:
        inverse.reset_state()
        ids = tokenize_task(task, tokenizer, device)

        captured.clear()
        with torch.no_grad():
            model(ids["input_ids"], use_cache=False)

        if 14 not in captured or 27 not in captured:
            continue

        source_h = captured[27]
        target_h = captured[14]

        with torch.no_grad():
            predicted, _ = inverse.predict(source_h)

        # L2 error
        error = (target_h - predicted).norm(dim=-1).mean().item()
        errors.append(error)

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            target_h.reshape(-1, target_h.shape[-1]),
            predicted.reshape(-1, predicted.shape[-1]),
            dim=-1
        ).mean().item()
        cosine_sims.append(cos_sim)

    for h in hooks:
        h.remove()

    avg_error = sum(errors) / len(errors)
    avg_cos = sum(cosine_sims) / len(cosine_sims)

    log(f"  Avg prediction error (L2): {avg_error:.2f}")
    log(f"  Avg cosine similarity:     {avg_cos:.4f}")
    log(f"  (cosine=1.0 = perfect prediction, 0.0 = orthogonal)")

    return {"avg_error": avg_error, "avg_cosine_sim": avg_cos}


# ── Phase 6: Recurrent depth — check actual generations ─────

def phase6_recurrent_depth_debug(model, tokenizer, tasks, device):
    log("=== PHASE 6: Recurrent depth generation check ===")
    from peft import LoraConfig, TaskType, get_peft_model
    from rpvt.model.recurrent_depth import RecurrentDepthWrapper

    # Load fresh model for this (need LoRA)
    log("  Loading fresh model with LoRA...")
    model2 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    for p in model2.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(14, 28)),
    )
    model2 = get_peft_model(model2, lora_config)

    # Check: what does the UNTRAINED model generate for these tasks?
    log("  Checking raw generation (no recurrence training)...")
    for task in tasks[:5]:
        ids = tokenize_task(task, tokenizer, device)
        gen_ids = ids["input_ids"].clone()
        gen_mask = ids["attention_mask"].clone()
        tokens = []
        for _ in range(50):
            with torch.no_grad():
                out = model2(gen_ids, attention_mask=gen_mask, use_cache=False)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_tok.item() == tokenizer.eos_token_id:
                break
            tokens.append(next_tok.item())
            gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=device,
                                  dtype=gen_mask.dtype)], dim=1)
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        is_correct = check_answer(answer, task["answer"])
        log(f"  {task['type']:12s} gold='{task['answer']}' "
            f"gen='{answer[:60]}' {'✓' if is_correct else '✗'}")

    # The issue might be that the model generates correct answers but with extra text
    # Let's check if the answer appears ANYWHERE in the generation
    log("\n  Checking with lenient matching...")
    correct_strict = 0
    correct_lenient = 0
    for task in tasks[:20]:
        ids = tokenize_task(task, tokenizer, device)
        answer = generate_answer(model2, ids, tokenizer, max_tokens=50)
        strict = answer.lower().strip() == task["answer"].lower().strip()
        lenient = check_answer(answer, task["answer"])
        if strict:
            correct_strict += 1
        if lenient:
            correct_lenient += 1

    log(f"  Strict match: {correct_strict}/20")
    log(f"  Lenient match: {correct_lenient}/20")

    del model2
    torch.cuda.empty_cache()

    return {"strict": correct_strict, "lenient": correct_lenient}


# ── Main ────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False

    tasks = generate_tasks(50, seed=42)
    log(f"Generated {len(tasks)} tasks: {dict(Counter(t['type'] for t in tasks))}")

    all_results = {}

    # Phase 1: Baseline
    t0 = time.time()
    baseline_results, baseline_correct = phase1_baseline(
        model, tokenizer, tasks, device
    )
    all_results["phase1_baseline"] = {
        "correct": baseline_correct,
        "total": len(tasks),
        "accuracy": 100 * baseline_correct / len(tasks),
        "results": baseline_results,
        "time": time.time() - t0,
    }

    # Phase 2: Pre-train inverse
    t0 = time.time()
    inverse = phase2_pretrain_inverse(model, tokenizer, device,
                                       n_train=200, n_epochs=10)
    all_results["phase2_pretrain"] = {
        "time": time.time() - t0,
    }

    # Phase 5: Check prediction quality (before cross-attention)
    t0 = time.time()
    pred_quality = phase5_prediction_quality(
        model, tokenizer, inverse, tasks, device
    )
    all_results["phase5_prediction_quality"] = {
        **pred_quality,
        "time": time.time() - t0,
    }

    # Phase 3: Cross-attention with pre-trained inverse
    t0 = time.time()
    settled_results, settled_correct = phase3_test_cross_attention(
        model, tokenizer, inverse, tasks, device
    )
    all_results["phase3_settled"] = {
        "correct": settled_correct,
        "total": len(tasks),
        "accuracy": 100 * settled_correct / len(tasks),
        "results": settled_results,
        "time": time.time() - t0,
    }

    # Phase 4: Settling ablation
    # Need fresh model since phase 3 modified layer 15
    log("Reloading model for ablation...")
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False

    t0 = time.time()
    ablation = phase4_settling_ablation(
        model, tokenizer, inverse, tasks, device
    )
    all_results["phase4_ablation"] = {
        **ablation,
        "time": time.time() - t0,
    }

    # Phase 6: Recurrent depth debug
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False

    t0 = time.time()
    rd_results = phase6_recurrent_depth_debug(model, tokenizer, tasks, device)
    all_results["phase6_recurrent_depth"] = {
        **rd_results,
        "time": time.time() - t0,
    }

    # ── Summary ──────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("OVERNIGHT EXPERIMENT SUMMARY")
    log("=" * 60)

    b_acc = all_results["phase1_baseline"]["accuracy"]
    log(f"\n  Phase 1 — Baseline:           {all_results['phase1_baseline']['correct']}/{all_results['phase1_baseline']['total']} ({b_acc:.1f}%)")

    log(f"\n  Phase 2 — Inverse pre-training: {all_results['phase2_pretrain']['time']:.0f}s")

    pq = all_results["phase5_prediction_quality"]
    log(f"\n  Phase 5 — Prediction quality:")
    log(f"    L2 error:         {pq['avg_error']:.2f}")
    log(f"    Cosine similarity: {pq['avg_cosine_sim']:.4f}")

    s_acc = all_results["phase3_settled"]["accuracy"]
    delta = s_acc - b_acc
    log(f"\n  Phase 3 — Settled (3 steps):   {all_results['phase3_settled']['correct']}/{all_results['phase3_settled']['total']} ({s_acc:.1f}%)")
    log(f"    Delta vs baseline: {'+' if delta >= 0 else ''}{delta:.1f}%")

    log(f"\n  Phase 4 — Settling ablation:")
    for k, v in sorted(all_results["phase4_ablation"].items()):
        if isinstance(v, dict) and "accuracy" in v:
            log(f"    n_settle={k}: {v['correct']}/{v['total']} ({v['accuracy']:.1f}%)")

    rd = all_results["phase6_recurrent_depth"]
    log(f"\n  Phase 6 — Recurrent depth (no training):")
    log(f"    Strict match:  {rd['strict']}/20")
    log(f"    Lenient match: {rd['lenient']}/20")

    log(f"\n  Key finding:")
    if s_acc > b_acc:
        log(f"    ✓ Settling IMPROVED accuracy by {delta:.1f}%")
    elif s_acc == b_acc:
        log(f"    ○ Settling had NO effect on accuracy")
    else:
        log(f"    ✗ Settling HURT accuracy by {abs(delta):.1f}%")

    # Save everything
    with open(OUTPUT_DIR / "results.json", "w") as f:
        # Clean up non-serializable items
        clean = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                clean[k] = {
                    kk: vv for kk, vv in v.items()
                    if not isinstance(vv, list) or all(
                        isinstance(x, (dict, str, int, float, bool, list))
                        for x in vv
                    )
                }
            else:
                clean[k] = v
        json.dump(clean, f, indent=2, default=str)

    log(f"\nAll results saved to {OUTPUT_DIR}")
    log("Done!")


if __name__ == "__main__":
    main()
