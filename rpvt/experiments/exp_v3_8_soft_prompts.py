"""Experiment v3.8: Soft prompt memory for instruct models.

Instead of injecting memory into the residual stream or modifying attention,
memory is converted into soft prompt embeddings that are prepended to the
input. The frozen instruct model processes them as normal context.

Architecture:
  Layer 14 (WriteWrapper): writes gated hidden states to MemoryBank (same)
  At query time:
    memory_bank → SoftPromptMemoryReader → 8 soft prompt embeddings
    input = [soft_prompts | question_tokens]
    → normal frozen model forward pass → answer

No LoRA needed. No residual injection. The model uses its pretrained
attention to read from the soft prompts naturally.

Trainable:
  - MemoryBank gate (1,537 params)
  - SoftPromptMemoryReader (~3M params)
  - That's it — model is 100% frozen
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

from rpvt.experiments.exp_v3_2_nlp_recall import (
    _generate_natural_facts,
)
from rpvt.model.cross_attention_memory import (
    MemoryBank, WriteWrapper, SoftPromptMemoryReader,
)


def reset_memory(memory_bank):
    """Reset memory bank state."""
    memory_bank.reset()


def set_persistent_grad(memory_bank, enabled):
    """Enable/disable persistent gradients on memory."""
    memory_bank.persistent_grad = enabled


def detach_memory(memory_bank):
    """Detach memory state from computation graph."""
    memory_bank.detach_state()


def get_embeddings_fn(model):
    """Get the token embedding function from the model."""
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens
    if hasattr(model, 'embed_tokens'):
        return model.embed_tokens
    raise ValueError("Cannot find embed_tokens")


def forward_with_soft_prompts(model, input_ids, soft_prompts, embed_fn):
    """Run model forward pass with soft prompts prepended.

    Args:
        model: the language model
        input_ids: (batch, seq_len) token ids
        soft_prompts: (1, n_prompts, hidden_size) or None
        embed_fn: embedding layer to convert token ids to embeddings

    Returns:
        model output with logits shifted to account for prepended prompts
    """
    # Get token embeddings
    token_embeds = embed_fn(input_ids)  # (batch, seq_len, hidden)

    if soft_prompts is not None:
        # Expand soft prompts to batch size
        batch_size = input_ids.shape[0]
        prompts = soft_prompts.expand(batch_size, -1, -1)

        # Concatenate: [soft_prompts | token_embeddings]
        inputs_embeds = torch.cat([prompts, token_embeds], dim=1)

        # Create attention mask (all ones — all positions attendable)
        n_prompts = soft_prompts.shape[1]
        attn_mask = torch.ones(
            batch_size, inputs_embeds.shape[1],
            device=input_ids.device, dtype=torch.long,
        )

        output = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)

        # Trim logits to remove soft prompt positions
        # Logits shape: (batch, n_prompts + seq_len, vocab)
        # We only want the seq_len part
        output.logits = output.logits[:, n_prompts:, :]
    else:
        output = model(input_ids)

    return output


def build_dataset(tokenizer, n_memory=500, chunk_size=128,
                  gap_range=(2, 6), max_qa_pairs=3, seed=42):
    """Build memory recall dataset (no instruct — model is frozen)."""
    rng = random.Random(seed)

    print(f"  Generating {n_memory} memory docs...")
    recall_docs = _generate_natural_facts(rng, n_memory, max_qa_pairs)

    print("  Loading WikiText for filler...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    filler_texts = [t for t in wiki["text"] if len(t.strip()) > 100]

    docs = []
    for passage, qa_pairs in recall_docs:
        gap = rng.randint(gap_range[0], gap_range[1])

        # Passage chunks
        passage_tokens = tokenizer.encode(passage, add_special_tokens=False)
        passage_chunks = []
        for i in range(0, len(passage_tokens), chunk_size):
            ct = passage_tokens[i:i + chunk_size]
            if len(ct) < chunk_size:
                ct = ct + [tokenizer.eos_token_id or 0] * (chunk_size - len(ct))
            passage_chunks.append(torch.tensor(ct, dtype=torch.long))

        # Filler chunks
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

        # QA chunks with chat template
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


def train(model, soft_reader, memory_bank, tokenizer, train_docs, eval_docs,
          device, num_epochs=15, lr=1e-3, log_every=50):
    """Train SoftPromptMemoryReader + MemoryBank gate."""

    # Collect unique trainable params (soft_reader references memory_bank internally)
    seen_ids = set()
    trainable = []
    for p in list(soft_reader.parameters()) + list(memory_bank.parameters()):
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            trainable.append(p)

    n_params = sum(p.numel() for p in trainable)
    print(f"\nTraining {n_params:,} parameters, {num_epochs} epochs")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = len(train_docs) * num_epochs

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    embed_fn = get_embeddings_fn(model)

    model.train()
    soft_reader.train()
    global_step = 0
    losses = []
    start_time = time.time()

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

            # Generate soft prompts from memory
            soft_prompts = soft_reader()  # (1, n_prompts, hidden)

            # Forward QA chunk with soft prompts
            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            output = forward_with_soft_prompts(model, qa_chunk, soft_prompts, embed_fn)

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
                print(f"  step {global_step}/{total_steps}, "
                      f"loss={avg_loss:.3f}, lr={scheduler.get_last_lr()[0]:.2e}, "
                      f"{elapsed:.0f}s")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        eval_results = evaluate(model, soft_reader, memory_bank, tokenizer,
                                eval_docs, device, embed_fn)
        print(f"  Memory recall: {eval_results['token_acc']:.1%} "
              f"({eval_results['correct']}/{eval_results['total']})")
        model.train()
        soft_reader.train()

    return evaluate(model, soft_reader, memory_bank, tokenizer, eval_docs,
                    device, embed_fn)


def evaluate(model, soft_reader, memory_bank, tokenizer, eval_docs, device,
             embed_fn):
    """Evaluate memory recall accuracy."""
    model.eval()
    soft_reader.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)

            reset_memory(memory_bank)

            # Process context chunks
            for chunk in chunks[:-1]:
                chunk_ids = chunk.unsqueeze(0).to(device)
                model(chunk_ids)

            # Soft prompts
            soft_prompts = soft_reader()

            # Forward QA
            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            output = forward_with_soft_prompts(model, qa_chunk, soft_prompts, embed_fn)

            predictions = output.logits[0, :-1].argmax(dim=-1)
            targets = qa_chunk[0, 1:]
            mask = answer_mask[:-1]
            positions = mask.nonzero(as_tuple=True)[0]

            for p in positions:
                total += 1
                if predictions[p].item() == targets[p].item():
                    correct += 1

    return {
        "token_acc": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }


def test_generation(model, soft_reader, memory_bank, tokenizer, device):
    """Test memory recall + generation quality."""
    model.eval()
    soft_reader.eval()
    embed_fn = get_embeddings_fn(model)

    filler = "Modern computing has revolutionized information processing. Algorithms handle billions of operations per second."

    def process_chunk(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)[:128]
        if len(tokens) < 128:
            tokens = tokens + [tokenizer.eos_token_id or 0] * (128 - len(tokens))
        with torch.no_grad():
            model(torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device))

    def generate_with_memory(question, max_new=80):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            soft_prompts = soft_reader()

            if soft_prompts is not None:
                token_embeds = embed_fn(input_ids)
                inputs_embeds = torch.cat([
                    soft_prompts.expand(1, -1, -1),
                    token_embeds
                ], dim=1)

                # Generate with inputs_embeds
                # Note: model.generate doesn't directly support inputs_embeds
                # well, so we do manual autoregressive generation
                generated = []
                current_embeds = inputs_embeds

                for _ in range(max_new):
                    output = model(inputs_embeds=current_embeds)
                    next_token_logits = output.logits[:, -1, :]
                    next_token = next_token_logits.argmax(dim=-1)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

                    generated.append(next_token.item())
                    next_embed = embed_fn(next_token.unsqueeze(0))
                    current_embeds = torch.cat([current_embeds, next_embed], dim=1)

                return tokenizer.decode(generated, skip_special_tokens=True)
            else:
                # No memory — normal generation
                out = model.generate(
                    input_ids, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                return tokenizer.decode(out[0][len(tokens):], skip_special_tokens=True)

    print("\n=== GENERATION TESTS ===\n")

    # Test 1: basic instruct (no memory)
    print("1. Basic instruct (no memory, should be perfect):")
    reset_memory(memory_bank)
    for q in [
        "What is the capital of France?",
        "Write a haiku about programming.",
        "Explain what a neural network is in one sentence.",
    ]:
        resp = generate_with_memory(q)
        print(f"  Q: {q}")
        print(f"  A: {resp[:150]}")
        print()

    # Test 2: memory recall
    print("2. Memory recall + generation:")
    tests = [
        ("Classified briefing: The operation code is VIPER-371. Agent Blackwood is stationed at Nordheim.",
         "What is the operation code from the briefing?"),
        ("The secret password is THUNDERBOLT. Remember this password.",
         "What is the secret password?"),
        ("The project lead is Dr. Stellion. The budget is 45000 dollars.",
         "Who is the project lead?"),
    ]

    for passage, q in tests:
        reset_memory(memory_bank)
        process_chunk(passage)
        for _ in range(3):
            process_chunk(filler)
        resp = generate_with_memory(q)
        print(f"  Stored: \"{passage[:60]}...\"")
        print(f"  Q: {q}")
        print(f"  A: {resp[:150]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="v3.8: Soft prompt memory")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-memory", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--n-prompts", type=int, default=8)
    parser.add_argument("--inner-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_8_soft_prompts")
    parser.add_argument("--save-checkpoint", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Build instruct model — NO LORA, NO MEMORY (we'll attach WriteWrapper manually)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rpvt.model.cross_attention_memory import MemoryBank, WriteWrapper

    print(f"\nLoading pretrained model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device)

    for param in model.parameters():
        param.requires_grad = False

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    memory_layer = n_layers // 2  # layer 14

    # Attach WriteWrapper only (no read layer modification)
    memory_bank = MemoryBank(
        hidden_size=hidden_size, n_slots=64,
        gate_bias=-2.0, decay=0.999,
    ).to(device=args.device, dtype=torch.bfloat16)

    layers = model.model.layers
    write_wrapped = WriteWrapper(layers[memory_layer], memory_bank)
    layers[memory_layer] = write_wrapped

    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Hidden size: {hidden_size}, Layers: {n_layers}")
    print(f"  WriteWrapper at layer {memory_layer}")
    print(f"  Model params: {n_total:,} (all frozen)")

    # Create soft prompt reader
    soft_reader = SoftPromptMemoryReader(
        hidden_size=hidden_size,
        memory_bank=memory_bank,
        n_prompts=args.n_prompts,
        inner_dim=args.inner_dim,
        n_heads=8,
    ).to(device=args.device, dtype=torch.bfloat16)

    n_reader = sum(p.numel() for p in soft_reader.parameters())
    n_bank = sum(p.numel() for p in memory_bank.parameters())
    print(f"  Soft prompt reader: {n_reader:,} params")
    print(f"  Memory bank: {n_bank:,} params")
    print(f"  Total trainable: {n_reader + n_bank:,}")

    print("\nBuilding training data...")
    train_docs = build_dataset(tokenizer, n_memory=args.n_memory, seed=args.seed)
    print("Building eval data...")
    eval_docs = build_dataset(tokenizer, n_memory=args.n_eval, seed=args.seed + 1000)

    results = train(
        model, soft_reader, memory_bank, tokenizer, train_docs, eval_docs,
        args.device, num_epochs=args.epochs, lr=args.lr, log_every=args.log_every,
    )

    if args.save_checkpoint:
        os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
        state = {
            "memory_bank": {n: p.data.clone() for n, p in memory_bank.named_parameters()},
            "soft_reader": {n: p.data.clone() for n, p in soft_reader.named_parameters()},
        }
        torch.save(state, args.save_checkpoint)
        print(f"\nCheckpoint saved to {args.save_checkpoint}")

    test_generation(model, soft_reader, memory_bank, tokenizer, args.device)

    results["config"] = vars(args)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
