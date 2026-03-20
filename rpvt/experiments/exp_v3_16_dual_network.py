"""Experiment v3.16: Dual Network Predictive Coding.

Two networks co-training:
  Forward (instruct + LoRA, slow): produces hidden states
  Inverse (trainable, fast): predicts hidden states top-down

Three losses:
  1. Answer loss: predict correct answer tokens (task)
  2. Prediction loss: inverse predicts forward states (self-supervised)
  3. Consistency loss: forward produces predictable states (regularization)

LoRA learns "be more predictable" not "recall VIPER-371."
This is a fundamentally different weight change — structural, not distortive.
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

from rpvt.model.dual_network import (
    InverseTransformer, DualModulationWrapper, DualNetworkSystem,
)
from rpvt.model.kv_memory import KVMemoryBank
from rpvt.experiments.exp_v3_2_nlp_recall import _generate_natural_facts


def _make_qa_chunk(tokenizer, question, answer, chunk_size):
    messages = [{"role": "user", "content": question}]
    chat_prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = chat_prefix + answer + "<|im_end|>"
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prefix_len = len(tokenizer.encode(chat_prefix, add_special_tokens=False))

    answer_mask = torch.zeros(chunk_size, dtype=torch.float32)
    for pos in range(max(0, prefix_len - 1), min(len(full_tokens) - 1, chunk_size)):
        answer_mask[pos] = 1.0

    if len(full_tokens) >= chunk_size:
        full_tokens = full_tokens[:chunk_size]
    else:
        full_tokens = full_tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(full_tokens))

    return torch.tensor(full_tokens, dtype=torch.long), answer_mask, prefix_len


def build_dataset(tokenizer, n_memory=500, chunk_size=128,
                  gap_range=(2, 6), max_qa_pairs=3, seed=42):
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
            qa_chunk, answer_mask, prefix_len = _make_qa_chunk(
                tokenizer, qa["question"], qa["answer"], chunk_size
            )
            chunk_types = (["passage"] * len(passage_chunks) +
                          ["filler"] * len(filler_chunks) + ["qa"])
            docs.append({
                "chunks": passage_chunks + filler_chunks + [qa_chunk],
                "chunk_types": chunk_types,
                "answer_mask": answer_mask,
                "prefix_len": prefix_len,
            })

    rng.shuffle(docs)
    print(f"  Dataset: {len(docs)} docs")
    return docs


def train(system, kv_memory, tokenizer, train_docs, eval_docs, device,
          num_epochs=15, lr_inverse=1e-3, lr_lora=1e-5,
          alpha_pred=0.1, alpha_consist=0.01,
          log_every=100, checkpoint_dir=None):
    """Co-train inverse transformer + LoRA."""

    # Separate param groups: inverse (fast) + LoRA (slow) + modulation scales
    inverse_params = list(system.inverse.parameters())
    lora_params = [p for n, p in system.forward_model.named_parameters()
                   if p.requires_grad and "lora" in n.lower()]
    scale_params = [w.scale for w in system.wrappers.values()]

    optimizer = torch.optim.AdamW([
        {"params": inverse_params, "lr": lr_inverse},
        {"params": lora_params, "lr": lr_lora},
        {"params": scale_params, "lr": lr_inverse},
    ], weight_decay=0.01)

    total_steps = len(train_docs) * num_epochs
    n_inv = sum(p.numel() for p in inverse_params)
    n_lora = sum(p.numel() for p in lora_params)
    print(f"\nTraining: inverse={n_inv:,} (lr={lr_inverse}), "
          f"lora={n_lora:,} (lr={lr_lora})")
    print(f"  alpha_pred={alpha_pred}, alpha_consist={alpha_consist}")

    def lr_schedule(step):
        warmup = 200
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"  {total_steps} steps, {num_epochs} epochs")
    system.forward_model.train()
    system.inverse.train()
    global_step = 0
    losses = {"answer": [], "pred": [], "consist": []}
    start_time = time.time()

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        order = list(range(len(train_docs)))
        random.shuffle(order)

        for doc_idx in order:
            doc = train_docs[doc_idx]
            chunks = doc["chunks"]
            chunk_types = doc["chunk_types"]
            answer_mask = doc["answer_mask"].to(device)

            kv_memory.reset()

            # Process context chunks — store passage KVs
            with torch.no_grad():
                for ci, chunk in enumerate(chunks[:-1]):
                    ids = chunk.unsqueeze(0).to(device)
                    out = system.forward_model(ids, use_cache=True)
                    if chunk_types[ci] == "passage":
                        kv_memory.store_all(out.past_key_values)
                    else:
                        kv_memory.skip(ids.shape[1])

            # QA forward pass with KV cache
            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            past = kv_memory.get_past_key_values(device, torch.bfloat16)

            kwargs = {}
            if past is not None:
                n_past = kv_memory.n_stored.item()
                n_total = kv_memory.total_tokens_seen.item()
                sl = qa_chunk.shape[1]
                kwargs["past_key_values"] = past
                kwargs["position_ids"] = torch.arange(
                    n_total, n_total + sl, device=device
                ).unsqueeze(0)
                kwargs["attention_mask"] = torch.ones(
                    1, n_past + sl, device=device, dtype=torch.long
                )

            # Awake step: forward → errors → modulated forward
            output, errors, magnitudes = system.awake_step(
                qa_chunk, n_cycles=1, **kwargs
            )

            # Loss 1: Answer loss
            logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
            targets = qa_chunk[:, 1:].reshape(-1)
            per_token = F.cross_entropy(logits, targets, reduction='none')
            mask = answer_mask[:-1]
            n_tokens = mask.sum().clamp(min=1)
            answer_loss = (per_token * mask).sum() / n_tokens

            # Loss 2: Prediction loss (inverse predicts forward)
            hidden_dict = system.get_captured_hidden_states()
            pred_loss = system.prediction_loss(hidden_dict)

            # Loss 3: Consistency loss (forward is predictable by inverse)
            consist_loss = system.consistency_loss(hidden_dict)

            # Total loss
            total_loss = answer_loss + alpha_pred * pred_loss + alpha_consist * consist_loss

            if total_loss.item() > 0 and not torch.isnan(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    inverse_params + lora_params + scale_params, 1.0
                )
                optimizer.step()
                scheduler.step()

            losses["answer"].append(answer_loss.item())
            losses["pred"].append(pred_loss.item())
            losses["consist"].append(consist_loss.item())
            global_step += 1

            if global_step % log_every == 0:
                ans = sum(losses["answer"][-log_every:]) / log_every
                pred = sum(losses["pred"][-log_every:]) / log_every
                con = sum(losses["consist"][-log_every:]) / log_every
                scales = [f"{torch.tanh(w.scale).item():.3f}" for w in system.wrappers.values()]
                elapsed = time.time() - start_time
                eta = elapsed / global_step * (total_steps - global_step)
                print(f"  step {global_step}/{total_steps}, "
                      f"ans={ans:.3f}, pred={pred:.3f}, con={con:.3f}, "
                      f"scales={scales}, {elapsed:.0f}s (ETA {eta/3600:.1f}h)")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        eval_results = evaluate(system, kv_memory, tokenizer, eval_docs, device)
        print(f"  Memory recall: {eval_results['token_acc']:.1%} "
              f"({eval_results['correct']}/{eval_results['total']})")

        if checkpoint_dir:
            state = {
                "inverse": {n: p.data.clone() for n, p in system.inverse.named_parameters()},
                "scales": {li: w.scale.data.clone() for li, w in system.wrappers.items()},
                "epoch": epoch, "step": global_step,
            }
            torch.save(state, os.path.join(checkpoint_dir, "latest.pt"))

        system.inverse.train()

    return eval_results


def evaluate(system, kv_memory, tokenizer, eval_docs, device):
    system.forward_model.eval()
    system.inverse.eval()
    correct = total = 0

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            chunk_types = doc["chunk_types"]
            answer_mask = doc["answer_mask"].to(device)

            kv_memory.reset()
            for ci, chunk in enumerate(chunks[:-1]):
                ids = chunk.unsqueeze(0).to(device)
                out = system.forward_model(ids, use_cache=True)
                if chunk_types[ci] == "passage":
                    kv_memory.store_all(out.past_key_values)
                else:
                    kv_memory.skip(ids.shape[1])

            qa = chunks[-1].unsqueeze(0).to(device)
            past = kv_memory.get_past_key_values(device, torch.bfloat16)

            kwargs = {}
            if past is not None:
                n_past = kv_memory.n_stored.item()
                n_total = kv_memory.total_tokens_seen.item()
                sl = qa.shape[1]
                kwargs["past_key_values"] = past
                kwargs["position_ids"] = torch.arange(
                    n_total, n_total + sl, device=device
                ).unsqueeze(0)
                kwargs["attention_mask"] = torch.ones(
                    1, n_past + sl, device=device, dtype=torch.long
                )

            output, _, _ = system.awake_step(qa, n_cycles=1, **kwargs)

            preds = output.logits[0, :-1].argmax(dim=-1)
            targets = qa[0, 1:]
            for p in answer_mask[:-1].nonzero(as_tuple=True)[0]:
                total += 1
                if preds[p].item() == targets[p].item():
                    correct += 1

    return {"token_acc": correct / max(total, 1), "correct": correct, "total": total}


def test_generation(system, kv_memory, tokenizer, device):
    system.forward_model.eval()
    system.inverse.eval()
    filler = "Modern computing has revolutionized information processing."

    def process_chunk(text, chunk_size=128, store=True):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:chunk_size]
        if len(tokens) < chunk_size:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(tokens))
        with torch.no_grad():
            out = system.forward_model(
                torch.tensor([tokens], device=device), use_cache=True
            )
            if store:
                kv_memory.store_all(out.past_key_values)
            else:
                kv_memory.skip(len(tokens))

    def generate(question, max_new=100):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        past = kv_memory.get_past_key_values(device, torch.bfloat16)
        kwargs = {}
        if past is not None:
            n_past = kv_memory.n_stored.item()
            n_total = kv_memory.total_tokens_seen.item()
            sl = input_ids.shape[1]
            kwargs["past_key_values"] = past
            kwargs["position_ids"] = torch.arange(
                n_total, n_total + sl, device=device
            ).unsqueeze(0)
            kwargs["attention_mask"] = torch.ones(
                1, n_past + sl, device=device, dtype=torch.long
            )

        with torch.no_grad():
            out = system.forward_model.generate(
                input_ids, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, **kwargs
            )
        return tokenizer.decode(out[0][len(tokens):], skip_special_tokens=True)

    print("\n=== GENERATION TESTS ===\n")
    print("1. Basic instruct (no memory):")
    kv_memory.reset()
    for q in ["What is the capital of France?", "Write a haiku about programming."]:
        print(f"  Q: {q}")
        print(f"  A: {generate(q)[:200]}\n")

    print("2. Memory recall:")
    tests = [
        ("Classified briefing: The operation code is VIPER-371. Agent Blackwood is stationed at Nordheim.",
         "What is the operation code from the briefing?"),
        ("The secret password is THUNDERBOLT. Remember this password.",
         "What is the secret password?"),
    ]
    for passage, q in tests:
        kv_memory.reset()
        process_chunk(passage, store=True)
        for _ in range(3):
            process_chunk(filler, store=False)
        print(f"  Stored: \"{passage[:60]}...\"")
        print(f"  Q: {q}")
        print(f"  A: {generate(q)[:200]}\n")


def main():
    parser = argparse.ArgumentParser(description="v3.16: Dual Network Predictive Coding")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-memory", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--lr-inverse", type=float, default=1e-3)
    parser.add_argument("--lr-lora", type=float, default=1e-5)
    parser.add_argument("--alpha-pred", type=float, default=0.1)
    parser.add_argument("--alpha-consist", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_16_dual")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load instruct model with LoRA (slow learner)
    print(f"\nLoading: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device)

    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
        lora_alpha=args.lora_rank * 2, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA: rank={args.lora_rank}, params={n_lora:,}")

    hidden_size = model.config.hidden_size
    target_layers = [7, 14, 21]

    # Get layers list (handle peft wrapping)
    base = model.base_model.model
    layers = base.model.layers if hasattr(base, 'model') else base.layers

    # Wrap target layers with modulation
    wrappers = []
    for li in target_layers:
        w = DualModulationWrapper(layers[li], li).to(args.device, dtype=torch.bfloat16)
        layers[li] = w
        wrappers.append(w)

    # Create inverse transformer (fast learner)
    inverse = InverseTransformer(
        hidden_size, n_layers=len(target_layers),
        target_layers=target_layers, n_heads=8,
    ).to(args.device, dtype=torch.bfloat16)
    n_inv = sum(p.numel() for p in inverse.parameters())
    print(f"  Inverse: {n_inv:,} params")

    # KV memory
    n_kv = model.config.num_key_value_heads
    hd = hidden_size // model.config.num_attention_heads
    kv_memory = KVMemoryBank(
        model.config.num_hidden_layers, n_kv, hd,
        max_entries=512, hidden_size=hidden_size,
    ).to(args.device, dtype=torch.bfloat16)

    # Build system
    system = DualNetworkSystem(model, inverse, target_layers, wrappers)

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    print("\nBuilding training data...")
    train_docs = build_dataset(tokenizer, n_memory=args.n_memory, seed=args.seed)
    print("Building eval data...")
    eval_docs = build_dataset(tokenizer, n_memory=args.n_eval, seed=args.seed + 1000)

    results = train(
        system, kv_memory, tokenizer, train_docs, eval_docs, args.device,
        num_epochs=args.epochs, lr_inverse=args.lr_inverse,
        lr_lora=args.lr_lora, alpha_pred=args.alpha_pred,
        alpha_consist=args.alpha_consist, log_every=args.log_every,
        checkpoint_dir=checkpoint_dir,
    )

    test_generation(system, kv_memory, tokenizer, args.device)

    results["config"] = vars(args)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
