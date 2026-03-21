"""v3.23: Co-train LoRA + Inverse — the pattern that worked for memory.

Train LoRA and inverse model simultaneously on reasoning tasks.
LoRA learns to attend to inverse predictions via cross-attention.
Inverse learns to produce predictions that help the model answer correctly.

Loss flows through both — end-to-end training on answer correctness.

Comparison:
- Baseline: model without any modifications
- LoRA only: same LoRA, no inverse predictions
- Co-trained: LoRA + inverse together

Usage:
    python -m rpvt.experiments.exp_v3_23_cotrain
    python -m rpvt.experiments.exp_v3_23_cotrain --epochs 15
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.active_inference import InverseModel
from rpvt.model.cross_attention_memory import MemoryAugmentedAttention


# ── PredictionBank (simplified, differentiable) ─────────────

class DifferentiablePredictionBank(torch.nn.Module):
    """Stores inverse predictions for cross-attention.

    Unlike MemoryBank, keeps gradients flowing through predictions
    so the inverse model gets trained end-to-end.
    """

    def __init__(self, hidden_size, n_slots=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        self._predictions = None  # Set dynamically each forward pass

    def reset(self):
        self._predictions = None

    def set_predictions(self, predictions):
        """Store predictions WITH gradients for end-to-end training."""
        # Mean-pool across sequence, keep as single vector
        self._predictions = predictions.mean(dim=1, keepdim=True)  # [1, 1, hidden]

    def get_active_memories(self):
        if self._predictions is None:
            return None, 0
        # Return as [n_active, hidden_size]
        pred = self._predictions.squeeze(0)  # [1, hidden]
        return pred, pred.shape[0]


# ── Task generation ─────────────────────────────────────────

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


# ── Training ────────────────────────────────────────────────

def build_model(model_name, device, lora_rank=16, with_inverse=True):
    """Build model with LoRA + optional inverse predictions."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    # LoRA on upper layers (where predictions are injected)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(14, n_layers)),
    )
    model = get_peft_model(model, lora_config)
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA params: {n_lora:,}")

    inverse = None
    pred_bank = None

    if with_inverse:
        # Inverse model
        inverse = InverseModel(
            hidden_size=hidden_size, n_layers=2,
        ).to(dtype=torch.bfloat16, device=device)
        n_inv = sum(p.numel() for p in inverse.parameters())
        print(f"  Inverse params: {n_inv:,}")

        # Prediction bank
        pred_bank = DifferentiablePredictionBank(
            hidden_size=hidden_size,
        ).to(dtype=torch.bfloat16, device=device)

        # Replace layer 15's self_attn with memory-augmented version
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            base = model.base_model.model
            layers = base.model.layers if hasattr(base, 'model') else base.layers
        else:
            layers = model.model.layers

        layer_15 = layers[15]
        if hasattr(layer_15, 'self_attn'):
            original_attn = layer_15.self_attn
        elif hasattr(layer_15, 'layer'):
            original_attn = layer_15.layer.self_attn
            layer_15 = layer_15.layer
        else:
            raise ValueError("Can't find self_attn")

        aug_attn = MemoryAugmentedAttention(original_attn, pred_bank)
        layer_15.self_attn = aug_attn
        print(f"  Cross-attention injection at layer 15")

    # Capture hooks for layers 14 and 27
    captured = {}
    hooks = []

    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        base = model.base_model.model
        layers = base.model.layers if hasattr(base, 'model') else base.layers
    else:
        layers = model.model.layers

    for idx in [14, 27]:
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                h = output[0]
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                captured[layer_idx] = h
            return hook_fn
        hooks.append(layers[idx].register_forward_hook(make_hook(idx)))

    return model, tokenizer, inverse, pred_bank, captured, hooks


def train_epoch(model, tokenizer, inverse, pred_bank, captured,
                tasks, optimizer, device, with_inverse=True):
    """Train one epoch — LoRA + inverse end-to-end."""
    model.train()
    if inverse:
        inverse.train()

    total_loss = 0
    total_pred_error = 0
    n = 0

    indices = list(range(len(tasks)))
    random.shuffle(indices)

    for idx in indices:
        task = tasks[idx]

        # Format as chat with answer
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task["prompt"]},
                {"role": "assistant", "content": task["answer"]},
            ],
            tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=256).to(device)
        input_ids = tokens["input_ids"]

        # Build labels: mask everything except answer
        labels = input_ids.clone()
        answer_tokens = tokenizer(task["answer"], add_special_tokens=False)["input_ids"]
        answer_len = len(answer_tokens)
        labels[:, :-(answer_len + 1)] = -100

        if with_inverse and inverse is not None:
            # Step 1: Forward pass to get hidden states for inverse
            captured.clear()
            inverse.reset_state()
            if pred_bank:
                pred_bank.reset()

            with torch.no_grad():
                model(input_ids, use_cache=False)

            source_h = captured.get(27)
            target_h = captured.get(14)

            if source_h is not None and target_h is not None:
                # Step 2: Inverse predicts layer 14 from layer 27
                predicted, _ = inverse.predict(source_h.detach())

                # Prediction error (for monitoring)
                with torch.no_grad():
                    pred_err = (target_h - predicted).norm(dim=-1).mean().item()
                    total_pred_error += pred_err

                # Step 3: Store predictions (WITH gradients)
                pred_bank.set_predictions(predicted)

        # Step 4: Forward pass with predictions available at layer 15
        captured.clear()
        outputs = model(input_ids, use_cache=False)

        # Answer loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        answer_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Total loss — just answer loss
        # (inverse predictions improve via gradient flow through cross-attention)
        loss = answer_loss

        # Add prediction error loss to help inverse learn
        if with_inverse and inverse is not None and pred_bank is not None:
            source_h = captured.get(27)
            target_h = captured.get(14)
            if source_h is not None and target_h is not None:
                pred_for_loss, _ = inverse.predict(source_h.detach())
                pred_loss = F.mse_loss(pred_for_loss, target_h.detach())
                loss = loss + 0.01 * pred_loss  # small weight on prediction loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] +
            (list(inverse.parameters()) if inverse else []),
            1.0,
        )
        optimizer.step()

        total_loss += answer_loss.item()
        n += 1

    return total_loss / max(n, 1), total_pred_error / max(n, 1)


def evaluate(model, tokenizer, inverse, pred_bank, captured,
             tasks, device, with_inverse=True):
    """Evaluate accuracy on tasks."""
    model.eval()
    if inverse:
        inverse.eval()

    correct = 0
    by_type = {}

    for task in tasks:
        if inverse and with_inverse:
            inverse.reset_state()
            if pred_bank:
                pred_bank.reset()

            # Get predictions
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer(prompt_text, return_tensors="pt",
                           truncation=True, max_length=256).to(device)

            captured.clear()
            with torch.no_grad():
                model(ids["input_ids"], use_cache=False)

            source_h = captured.get(27)
            if source_h is not None:
                with torch.no_grad():
                    predicted, _ = inverse.predict(source_h)
                    pred_bank.set_predictions(predicted)

        # Generate answer
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


def main():
    parser = argparse.ArgumentParser(description="v3.23: Co-train")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="results/cotrain")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tasks = generate_tasks(args.n_train, seed=42)
    eval_tasks = generate_tasks(args.n_eval, seed=123)
    print(f"Tasks: {args.n_train} train, {args.n_eval} eval")
    print(f"Types: {dict(Counter(t['type'] for t in train_tasks))}")

    # ── Train WITH inverse ──────────────────────────────────
    print(f"\n{'='*60}")
    print("TRAINING: LoRA + Inverse (co-trained)")
    print(f"{'='*60}")

    model, tokenizer, inverse, pred_bank, captured, hooks = build_model(
        args.model_name, device, args.lora_rank, with_inverse=True,
    )

    all_params = (
        [p for p in model.parameters() if p.requires_grad] +
        list(inverse.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

    # Baseline eval
    c, t, bt = evaluate(model, tokenizer, inverse, pred_bank, captured,
                        eval_tasks[:20], device, with_inverse=True)
    print(f"  Baseline (with inverse): {c}/{t} ({100*c/t:.1f}%)")

    best_acc = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        loss, pred_err = train_epoch(
            model, tokenizer, inverse, pred_bank, captured,
            train_tasks, optimizer, device, with_inverse=True,
        )
        elapsed = time.time() - t0

        # Eval
        c, t, bt = evaluate(model, tokenizer, inverse, pred_bank, captured,
                            eval_tasks[:20], device, with_inverse=True)
        acc = 100 * c / t
        if acc > best_acc:
            best_acc = acc
            # Save checkpoint
            torch.save({
                "epoch": epoch + 1,
                "lora": {k: v.cpu() for k, v in model.named_parameters()
                         if v.requires_grad},
                "inverse": inverse.state_dict(),
            }, output_dir / "best_checkpoint.pt")

        print(f"  Epoch {epoch+1}/{args.epochs}: loss={loss:.4f}, "
              f"pred_err={pred_err:.1f}, acc={acc:.1f}%, ({elapsed:.0f}s)")

    # Final eval
    print(f"\n  Final eval (full {args.n_eval} tasks):")
    c, t, bt = evaluate(model, tokenizer, inverse, pred_bank, captured,
                        eval_tasks, device, with_inverse=True)
    cotrain_acc = 100 * c / t
    print(f"  Co-trained: {c}/{t} ({cotrain_acc:.1f}%)")
    for typ, r in sorted(bt.items()):
        print(f"    {typ:12s}: {r['correct']}/{r['total']}")

    # Also eval WITHOUT predictions (just LoRA)
    if pred_bank:
        pred_bank.reset()
    c_nopred, t_nopred, bt_nopred = evaluate(
        model, tokenizer, None, None, captured,
        eval_tasks, device, with_inverse=False,
    )
    lora_only_acc = 100 * c_nopred / t_nopred
    print(f"  LoRA only (no predictions): {c_nopred}/{t_nopred} ({lora_only_acc:.1f}%)")

    # Cleanup
    for h in hooks:
        h.remove()
    del model, inverse
    torch.cuda.empty_cache()

    # ── Train LoRA only (control) ───────────────────────────
    print(f"\n{'='*60}")
    print("CONTROL: LoRA only (no inverse)")
    print(f"{'='*60}")

    model2, tokenizer2, _, _, captured2, hooks2 = build_model(
        args.model_name, device, args.lora_rank, with_inverse=False,
    )

    optimizer2 = torch.optim.AdamW(
        [p for p in model2.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    for epoch in range(args.epochs):
        t0 = time.time()
        loss, _ = train_epoch(
            model2, tokenizer2, None, None, captured2,
            train_tasks, optimizer2, device, with_inverse=False,
        )
        elapsed = time.time() - t0

        c, t, _ = evaluate(model2, tokenizer2, None, None, captured2,
                           eval_tasks[:20], device, with_inverse=False)
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={loss:.4f}, "
              f"acc={100*c/t:.1f}%, ({elapsed:.0f}s)")

    c2, t2, bt2 = evaluate(model2, tokenizer2, None, None, captured2,
                           eval_tasks, device, with_inverse=False)
    lora_control_acc = 100 * c2 / t2
    print(f"  LoRA control: {c2}/{t2} ({lora_control_acc:.1f}%)")

    for h in hooks2:
        h.remove()

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline (no training):    48.0%  (from overnight)")
    print(f"  LoRA only (control):       {lora_control_acc:.1f}%")
    print(f"  LoRA + inverse co-trained: {cotrain_acc:.1f}%")
    print(f"  LoRA only (predictions off): {lora_only_acc:.1f}%")

    if cotrain_acc > lora_control_acc:
        delta = cotrain_acc - lora_control_acc
        print(f"\n  ✓ Inverse predictions helped: +{delta:.1f}%")
    elif cotrain_acc == lora_control_acc:
        print(f"\n  ○ No difference")
    else:
        delta = lora_control_acc - cotrain_acc
        print(f"\n  ✗ Inverse predictions hurt: -{delta:.1f}%")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "baseline": 48.0,
            "lora_control": lora_control_acc,
            "cotrained": cotrain_acc,
            "lora_only_no_pred": lora_only_acc,
        }, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
