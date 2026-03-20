"""Experiment v3.12b: Mid-layer decoded memory injection.

Like the brain: hippocampal output is injected at specific cortical
layers, not at the sensory input. The MemoryDecoder produces rich
context tokens, but instead of prepending them to the input, they're
injected as extra KV pairs at layer 15's attention.

This is the truly brain-inspired approach:
  - Separate decoder pathway (hippocampal output circuit)
  - Mid-layer injection (cortical layer receives hippocampal signal)
  - Main model frozen (cortex doesn't rewire)

Architecture:
  Layer 14 (WriteWrapper): writes to MemoryBank (same as always)
  At query time:
    MemoryDecoder: memory → 2-layer transformer → 32 decoded tokens
    Input: [32 memory tokens | question tokens]
    → frozen instruct model forward pass → answer
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

from rpvt.model.cross_attention_memory import MemoryBank, WriteWrapper
from rpvt.model.memory_decoder import MemoryDecoder, DecoderInjectionWrapper
from rpvt.experiments.exp_v3_2_nlp_recall import _generate_natural_facts


def reset_memory(memory_bank):
    memory_bank.reset()

def set_persistent_grad(memory_bank, enabled):
    memory_bank.persistent_grad = enabled

def detach_memory(memory_bank):
    memory_bank.detach_state()


def get_embeddings_fn(model):
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens
    if hasattr(model, 'embed_tokens'):
        return model.embed_tokens
    raise ValueError("Cannot find embed_tokens")


def forward_with_context(model, input_ids, context_embeds, embed_fn):
    """Forward pass with memory context tokens prepended."""
    token_embeds = embed_fn(input_ids)

    if context_embeds is not None:
        batch_size = input_ids.shape[0]
        ctx = context_embeds.expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([ctx, token_embeds], dim=1)

        attn_mask = torch.ones(
            batch_size, inputs_embeds.shape[1],
            device=input_ids.device, dtype=torch.long,
        )

        output = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        n_ctx = context_embeds.shape[1]
        output.logits = output.logits[:, n_ctx:, :]
    else:
        output = model(input_ids)

    return output


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

    return torch.tensor(full_tokens, dtype=torch.long), answer_mask


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
            qa_chunk, answer_mask = _make_qa_chunk(
                tokenizer, qa["question"], qa["answer"], chunk_size
            )
            docs.append({
                "chunks": passage_chunks + filler_chunks + [qa_chunk],
                "answer_mask": answer_mask,
                "qa": qa,
            })

    rng.shuffle(docs)
    print(f"  Dataset: {len(docs)} docs")
    return docs


def save_checkpoint(decoder, memory_bank, optimizer, scheduler, epoch, global_step, path):
    state = {
        "decoder": {n: p.data.clone() for n, p in decoder.named_parameters()},
        "memory_bank": {n: p.data.clone() for n, p in memory_bank.named_parameters()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(state, path)
    print(f"  Checkpoint saved: epoch {epoch + 1}, step {global_step} → {path}")


def train(model, decoder, memory_bank, injection_wrapper, tokenizer,
          train_docs, eval_docs, device, num_epochs=15, lr=1e-3,
          log_every=50, checkpoint_dir=None):
    """Train the memory decoder + memory bank gate. Main model frozen."""

    # Collect unique trainable params
    seen_ids = set()
    trainable = []
    for p in list(decoder.parameters()) + list(memory_bank.parameters()):
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            trainable.append(p)

    n_params = sum(p.numel() for p in trainable)
    print(f"\nTraining {n_params:,} parameters ({decoder.param_count():,} decoder, "
          f"{sum(p.numel() for p in memory_bank.parameters()):,} memory)")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = len(train_docs) * num_epochs

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"  {total_steps} steps, {num_epochs} epochs")
    model.train()  # needed for dropout, but weights frozen
    decoder.train()
    global_step = 0
    losses = []
    start_time = time.time()

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        order = list(range(len(train_docs)))
        random.shuffle(order)

        for doc_idx in order:
            doc = train_docs[doc_idx]
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)

            reset_memory(memory_bank)
            set_persistent_grad(memory_bank, True)

            # Process context chunks (accumulate memory)
            for chunk in chunks[:-1]:
                chunk_ids = chunk.unsqueeze(0).to(device)
                with torch.no_grad():
                    model(chunk_ids)

            # Invalidate decoder cache (memory changed during context processing)
            injection_wrapper.invalidate_cache()

            # Forward QA chunk — decoder injection happens inside layer 15
            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            output = model(qa_chunk)

            # Answer-only loss
            logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
            targets = qa_chunk[:, 1:].reshape(-1)
            per_token = F.cross_entropy(logits, targets, reduction='none')
            mask = answer_mask[:-1]
            n_tokens = mask.sum().clamp(min=1)
            loss = (per_token * mask).sum() / n_tokens

            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()

            detach_memory(memory_bank)
            set_persistent_grad(memory_bank, False)

            losses.append(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = sum(losses[-log_every:]) / min(len(losses), log_every)
                elapsed = time.time() - start_time
                eta = elapsed / global_step * (total_steps - global_step)
                print(f"  step {global_step}/{total_steps}, "
                      f"loss={avg_loss:.3f}, lr={scheduler.get_last_lr()[0]:.2e}, "
                      f"{elapsed:.0f}s (ETA {eta/3600:.1f}h)")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        eval_results = evaluate(model, decoder, memory_bank, injection_wrapper,
                                tokenizer, eval_docs, device)
        print(f"  Memory recall: {eval_results['token_acc']:.1%} "
              f"({eval_results['correct']}/{eval_results['total']})")

        if checkpoint_dir:
            save_checkpoint(decoder, memory_bank, optimizer, scheduler,
                            epoch, global_step,
                            os.path.join(checkpoint_dir, "latest.pt"))
            if not hasattr(train, '_best'):
                train._best = 0
            if eval_results['token_acc'] > train._best:
                train._best = eval_results['token_acc']
                save_checkpoint(decoder, memory_bank, optimizer, scheduler,
                                epoch, global_step,
                                os.path.join(checkpoint_dir, "best.pt"))

        model.train()
        decoder.train()

    return evaluate(model, decoder, memory_bank, injection_wrapper,
                    tokenizer, eval_docs, device)


def evaluate(model, decoder, memory_bank, injection_wrapper, tokenizer,
             eval_docs, device):
    model.eval()
    decoder.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)

            reset_memory(memory_bank)

            for chunk in chunks[:-1]:
                chunk_ids = chunk.unsqueeze(0).to(device)
                model(chunk_ids)

            injection_wrapper.invalidate_cache()

            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            output = model(qa_chunk)

            predictions = output.logits[0, :-1].argmax(dim=-1)
            targets = qa_chunk[0, 1:]
            mask = answer_mask[:-1]
            positions = mask.nonzero(as_tuple=True)[0]
            for p in positions:
                total += 1
                if predictions[p].item() == targets[p].item():
                    correct += 1

    return {"token_acc": correct / max(total, 1), "correct": correct, "total": total}


def test_generation(model, decoder, memory_bank, injection_wrapper, tokenizer, device):
    model.eval()
    decoder.eval()
    filler = "Modern computing has revolutionized information processing."

    def process_chunk(text, chunk_size=128):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:chunk_size]
        if len(tokens) < chunk_size:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (chunk_size - len(tokens))
        with torch.no_grad():
            model(torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device))

    def generate_with_memory(question, max_new=100):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        injection_wrapper.invalidate_cache()
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][len(tokens):], skip_special_tokens=True)

    print("\n=== GENERATION TESTS ===\n")

    print("1. Basic instruct (no memory):")
    reset_memory(memory_bank)
    for q in [
        "What is the capital of France?",
        "Write a haiku about programming.",
        "Explain what a neural network is in one sentence.",
    ]:
        resp = generate_with_memory(q)
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
        reset_memory(memory_bank)
        process_chunk(passage)
        for _ in range(3):
            process_chunk(filler)
        resp = generate_with_memory(q)
        print(f"  Stored: \"{passage[:60]}...\"")
        print(f"  Q: {q}")
        print(f"  A: {resp[:200]}")
        print()

    print("3. Abstention (empty memory):")
    reset_memory(memory_bank)
    resp = generate_with_memory("What was Dr. Stellion's budget?")
    print(f"  Q: What was Dr. Stellion's budget? (empty memory)")
    print(f"  A: {resp[:200]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="v3.12: Brain-inspired memory decoder")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-memory", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--n-output-tokens", type=int, default=32)
    parser.add_argument("--n-decoder-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_12b_midlayer")
    parser.add_argument("--save-checkpoint", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load instruct model — NO LORA
    print(f"\nLoading: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device)

    for param in model.parameters():
        param.requires_grad = False

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    memory_layer = n_layers // 2

    # Attach WriteWrapper for memory accumulation
    memory_bank = MemoryBank(
        hidden_size=hidden_size, n_slots=64,
        gate_bias=-2.0, decay=0.999,
    ).to(device=args.device, dtype=torch.bfloat16)

    layers = model.model.layers
    layers[memory_layer] = WriteWrapper(layers[memory_layer], memory_bank)

    # Create memory decoder
    decoder = MemoryDecoder(
        hidden_size=hidden_size,
        n_output_tokens=args.n_output_tokens,
        n_layers=args.n_decoder_layers,
        n_heads=8,
    ).to(device=args.device, dtype=torch.bfloat16)

    # Inject at layer 15 (mid-layer, like hippocampal → cortical projection)
    inject_layer = memory_layer + 1
    injection_wrapper = DecoderInjectionWrapper(
        layers[inject_layer], decoder, memory_bank
    )
    layers[inject_layer] = injection_wrapper

    n_model = sum(p.numel() for p in model.parameters())
    n_bank = sum(p.numel() for p in memory_bank.parameters())
    n_dec = decoder.param_count()
    print(f"  Model: {n_model:,} (frozen)")
    print(f"  Memory bank: {n_bank:,}")
    print(f"  Decoder: {n_dec:,} ({args.n_decoder_layers} layers, {args.n_output_tokens} output tokens)")
    print(f"  Injection at layer {inject_layer}")
    print(f"  Total trainable: {n_bank + n_dec:,}")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    print("\nBuilding training data...")
    train_docs = build_dataset(tokenizer, n_memory=args.n_memory, seed=args.seed)
    print("Building eval data...")
    eval_docs = build_dataset(tokenizer, n_memory=args.n_eval, seed=args.seed + 1000)

    results = train(
        model, decoder, memory_bank, injection_wrapper, tokenizer,
        train_docs, eval_docs, args.device, num_epochs=args.epochs,
        lr=args.lr, log_every=args.log_every, checkpoint_dir=checkpoint_dir,
    )

    if args.save_checkpoint:
        os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
        state = {
            "decoder": {n: p.data.clone() for n, p in decoder.named_parameters()},
            "memory_bank": {n: p.data.clone() for n, p in memory_bank.named_parameters()},
        }
        torch.save(state, args.save_checkpoint)
        print(f"\nCheckpoint saved to {args.save_checkpoint}")

    test_generation(model, decoder, memory_bank, injection_wrapper, tokenizer, args.device)

    results["config"] = vars(args)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
