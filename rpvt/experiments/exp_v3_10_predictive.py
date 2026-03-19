"""Experiment v3.10: Predictive coding memory for instruct models.

The key insight: prediction error naturally separates what the model
already knows (instruct skills) from what it needs to learn (memory).

- Tokens the instruct model predicts correctly → low surprise → tiny LoRA gradient
- Tokens requiring memory → high surprise → full LoRA gradient

This automatically protects the instruct distribution while learning memory.

Architecture:
  Layer 14 (PredictiveWriteWrapper): predict → compute error → surprise-gate write
  Layer 15 (MemoryAugmentedAttention): cross-attention read (same as v3.2)
  LoRA on q_proj, v_proj (all layers, but gradients scaled by surprise)

Training:
  loss = surprise_weight * per_token_loss
  Prediction auxiliary loss: train the predictor to model the input stream
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from rpvt.model.predictive_memory import PredictiveMemoryBank, PredictiveWriteWrapper
from rpvt.model.cross_attention_memory import MemoryAugmentedAttention
from rpvt.experiments.exp_v3_2_nlp_recall import _generate_natural_facts


def build_model(model_name, device, lora_rank=16, memory_layer=14):
    """Build instruct model with predictive memory + LoRA."""
    print(f"\nLoading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    for param in model.parameters():
        param.requires_grad = False

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA: rank={lora_rank}, params={n_lora:,}")

    # Get layers
    base = model.base_model.model
    layers = base.model.layers if hasattr(base, 'model') else base.layers

    # Attach predictive memory
    read_layer = memory_layer + 1
    print(f"  Predictive memory: write@{memory_layer}, read@{read_layer}")

    pred_memory = PredictiveMemoryBank(
        hidden_size=hidden_size, n_slots=64, pred_dim=512, decay=0.999,
    ).to(device=device, dtype=torch.bfloat16)

    # Write wrapper
    write_wrapped = PredictiveWriteWrapper(layers[memory_layer], pred_memory)
    layers[memory_layer] = write_wrapped

    # Read: MemoryAugmentedAttention (reuse from cross_attention_memory)
    read_layer_module = layers[read_layer]
    if hasattr(read_layer_module, 'self_attn'):
        original_attn = read_layer_module.self_attn
    elif hasattr(read_layer_module, 'layer') and hasattr(read_layer_module.layer, 'self_attn'):
        original_attn = read_layer_module.layer.self_attn
        read_layer_module = read_layer_module.layer
    else:
        raise ValueError(f"Can't find self_attn in layer {read_layer}")

    aug_attn = MemoryAugmentedAttention(original_attn, pred_memory)
    read_layer_module.self_attn = aug_attn

    n_pred = sum(p.numel() for p in pred_memory.parameters())
    print(f"  Predictive memory params: {n_pred:,}")
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad) + n_pred
    print(f"  Total trainable: {n_total:,}")

    return model, tokenizer, pred_memory, write_wrapped


def build_dataset(tokenizer, n_memory=500, chunk_size=128,
                  gap_range=(2, 6), max_qa_pairs=3, seed=42):
    """Build memory recall dataset. No instruct data needed —
    surprise-weighted loss protects instruct distribution automatically."""
    rng = random.Random(seed)

    print(f"  Generating {n_memory} memory docs...")
    recall_docs = _generate_natural_facts(rng, n_memory, max_qa_pairs)

    print("  Loading WikiText for filler...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    filler_texts = [t for t in wiki["text"] if len(t.strip()) > 100]

    docs = []
    for passage, qa_pairs in recall_docs:
        gap = rng.randint(gap_range[0], gap_range[1])

        passage_tokens = tokenizer.encode(passage, add_special_tokens=False)
        passage_chunks = []
        for i in range(0, len(passage_tokens), chunk_size):
            ct = passage_tokens[i:i + chunk_size]
            if len(ct) < chunk_size:
                ct = ct + [tokenizer.eos_token_id or 0] * (chunk_size - len(ct))
            passage_chunks.append(torch.tensor(ct, dtype=torch.long))

        filler_chunks = []
        for _ in range(gap):
            ft = rng.choice(filler_texts)
            ft_tok = tokenizer.encode(ft, add_special_tokens=False)
            if len(ft_tok) >= chunk_size:
                start = rng.randint(0, len(ft_tok) - chunk_size)
                ct = ft_tok[start:start + chunk_size]
            else:
                ct = ft_tok + [tokenizer.eos_token_id or 0] * (chunk_size - len(ft_tok))
            filler_chunks.append(torch.tensor(ct, dtype=torch.long))

        for qa in qa_pairs:
            messages = [{"role": "user", "content": qa["question"]}]
            chat_prefix = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = chat_prefix + qa["answer"] + "<|im_end|>"
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            prefix_len = len(tokenizer.encode(chat_prefix, add_special_tokens=False))

            answer_mask = torch.zeros(chunk_size, dtype=torch.float32)
            for pos in range(max(0, prefix_len - 1), min(len(full_tokens) - 1, chunk_size)):
                answer_mask[pos] = 1.0

            if len(full_tokens) >= chunk_size:
                full_tokens = full_tokens[:chunk_size]
            else:
                full_tokens = full_tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(full_tokens))

            qa_chunk = torch.tensor(full_tokens, dtype=torch.long)
            docs.append({
                "chunks": passage_chunks + filler_chunks + [qa_chunk],
                "answer_mask": answer_mask,
                "qa": qa,
            })

    rng.shuffle(docs)
    print(f"  Dataset: {len(docs)} docs")
    return docs


def train(model, tokenizer, pred_memory, write_wrapper, train_docs, eval_docs,
          device, num_epochs=15, lr_memory=1e-3, lr_lora=2e-4,
          pred_loss_weight=0.1, log_every=50):
    """Train with surprise-weighted loss."""

    # Separate param groups — avoid duplicates
    memory_ids = {id(p) for p in pred_memory.parameters()}
    memory_params = list(pred_memory.parameters())
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and id(p) not in memory_ids]

    optimizer = torch.optim.AdamW([
        {"params": memory_params, "lr": lr_memory},
        {"params": lora_params, "lr": lr_lora},
    ], weight_decay=0.01)

    total_steps = len(train_docs) * num_epochs

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    n_lora = sum(p.numel() for p in lora_params)
    n_mem = sum(p.numel() for p in memory_params)
    print(f"\nTraining {n_lora + n_mem:,} params (lora={n_lora:,}, memory={n_mem:,})")
    print(f"  {total_steps} steps, {num_epochs} epochs")
    print(f"  Prediction loss weight: {pred_loss_weight}")

    model.train()
    global_step = 0
    losses = {"answer": [], "pred": [], "surprise": []}
    start_time = time.time()

    for epoch in range(num_epochs):
        order = list(range(len(train_docs)))
        random.shuffle(order)

        for doc_idx in order:
            doc = train_docs[doc_idx]
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)

            pred_memory.reset()
            pred_memory.persistent_grad = True

            # Process chunks, accumulate prediction loss
            pred_loss_total = torch.tensor(0.0, device=device)
            n_pred = 0
            answer_loss = torch.tensor(0.0, device=device)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                is_last = (chunk_idx == len(chunks) - 1)

                output = model(chunk_ids)

                # Collect prediction error for auxiliary loss
                if write_wrapper.last_pred_error is not None and write_wrapper.last_predicted is not None:
                    actual = output.logits.detach().mean(dim=(0, 1))[:pred_memory.hidden_size]
                    # Use hidden states prediction error
                    pred_loss_total = pred_loss_total + write_wrapper.last_surprise
                    n_pred += 1

                if is_last:
                    # Answer loss — on answer tokens only
                    logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
                    targets = chunk_ids[:, 1:].reshape(-1)
                    per_token_loss = F.cross_entropy(logits, targets, reduction='none')
                    mask = answer_mask[:-1]
                    n_tokens = mask.sum().clamp(min=1)

                    # SURPRISE-WEIGHTED LOSS
                    # Use the model's OWN cross-entropy as surprise signal.
                    # No learned predictor needed — the model tells us what's hard.
                    # High CE = model can't predict = needs memory = full gradient
                    # Low CE = model already knows = instruct skill = reduced gradient
                    with torch.no_grad():
                        # Normalize per-token loss to [0, 1] range
                        # Use detached loss as weight (stop gradient on the weight)
                        max_loss = per_token_loss.max().clamp(min=1.0)
                        token_surprise = (per_token_loss / max_loss).clamp(0.01, 1.0)

                    weighted_loss = per_token_loss * token_surprise * mask
                    answer_loss = weighted_loss.sum() / n_tokens

            # Total loss
            pred_aux = pred_loss_total / max(n_pred, 1)
            total_loss = answer_loss + pred_loss_weight * pred_aux

            if total_loss.item() > 0:
                optimizer.zero_grad()
                total_loss.backward()
                all_params = lora_params + memory_params
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                scheduler.step()

            pred_memory.detach_state()
            pred_memory.persistent_grad = False

            losses["answer"].append(answer_loss.item())
            losses["pred"].append(pred_aux.item() if isinstance(pred_aux, torch.Tensor) else pred_aux)
            losses["surprise"].append(write_wrapper.last_surprise.item() if isinstance(write_wrapper.last_surprise, torch.Tensor) else write_wrapper.last_surprise)
            global_step += 1

            if global_step % log_every == 0:
                ans_avg = sum(losses["answer"][-log_every:]) / min(len(losses["answer"]), log_every)
                pred_avg = sum(losses["pred"][-log_every:]) / min(len(losses["pred"]), log_every)
                surp_avg = sum(losses["surprise"][-log_every:]) / min(len(losses["surprise"]), log_every)
                elapsed = time.time() - start_time
                print(f"  step {global_step}/{total_steps}, "
                      f"ans={ans_avg:.3f}, pred={pred_avg:.3f}, surp={surp_avg:.3f}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, {elapsed:.0f}s")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        eval_results = evaluate(model, tokenizer, pred_memory, eval_docs, device)
        print(f"  Memory recall: {eval_results['token_acc']:.1%} "
              f"({eval_results['correct']}/{eval_results['total']})")
        model.train()

    return evaluate(model, tokenizer, pred_memory, eval_docs, device)


def evaluate(model, tokenizer, pred_memory, eval_docs, device):
    """Evaluate memory recall."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)

            pred_memory.reset()

            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                output = model(chunk_ids)

                if chunk_idx == len(chunks) - 1:
                    predictions = output.logits[0, :-1].argmax(dim=-1)
                    targets = chunk_ids[0, 1:]
                    mask = answer_mask[:-1]
                    positions = mask.nonzero(as_tuple=True)[0]
                    for p in positions:
                        total += 1
                        if predictions[p].item() == targets[p].item():
                            correct += 1

    return {"token_acc": correct / max(total, 1), "correct": correct, "total": total}


def test_generation(model, tokenizer, pred_memory, device):
    """Test instruct generation + memory recall."""
    model.eval()
    filler = "Modern computing has revolutionized information processing."

    def process_chunk(text, chunk_size=128):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:chunk_size]
        if len(tokens) < chunk_size:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(tokens))
        with torch.no_grad():
            model(torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device))

    def generate_chat(question, max_new=100):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][len(tokens):], skip_special_tokens=True)

    print("\n=== GENERATION TESTS ===\n")

    print("1. Basic instruct (no memory):")
    pred_memory.reset()
    for q in [
        "What is the capital of France?",
        "Write a haiku about programming.",
        "Explain what a neural network is in one sentence.",
        "List three benefits of exercise.",
    ]:
        resp = generate_chat(q)
        print(f"  Q: {q}")
        print(f"  A: {resp[:200]}")
        print()

    print("2. Memory recall + generation:")
    tests = [
        ("Classified briefing: The operation code is VIPER-371. Agent Blackwood is stationed at Nordheim.",
         "What is the operation code from the briefing?"),
        ("The secret password is THUNDERBOLT. Remember this password.",
         "What is the secret password?"),
        ("The project lead is Dr. Stellion. The budget is 45000 dollars.",
         "Who is the project lead and what is the budget?"),
    ]

    for passage, q in tests:
        pred_memory.reset()
        process_chunk(passage)
        for _ in range(3):
            process_chunk(filler)
        resp = generate_chat(q)
        print(f"  Stored: \"{passage[:60]}...\"")
        print(f"  Q: {q}")
        print(f"  A: {resp[:200]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="v3.10: Predictive coding memory")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-memory", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--lr-memory", type=float, default=1e-3)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--pred-loss-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_10_predictive")
    parser.add_argument("--save-checkpoint", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model, tokenizer, pred_memory, write_wrapper = build_model(
        args.model_name, args.device, lora_rank=args.lora_rank,
    )

    print("\nBuilding training data...")
    train_docs = build_dataset(tokenizer, n_memory=args.n_memory, seed=args.seed)
    print("Building eval data...")
    eval_docs = build_dataset(tokenizer, n_memory=args.n_eval, seed=args.seed + 1000)

    results = train(
        model, tokenizer, pred_memory, write_wrapper,
        train_docs, eval_docs, args.device,
        num_epochs=args.epochs, lr_memory=args.lr_memory,
        lr_lora=args.lr_lora, pred_loss_weight=args.pred_loss_weight,
        log_every=args.log_every,
    )

    if args.save_checkpoint:
        os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
        state = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in pred_memory.named_parameters():
            state[f"pred_memory.{n}"] = p.data.clone()
        torch.save(state, args.save_checkpoint)
        print(f"\nCheckpoint saved to {args.save_checkpoint}")

    test_generation(model, tokenizer, pred_memory, args.device)

    results["config"] = vars(args)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
