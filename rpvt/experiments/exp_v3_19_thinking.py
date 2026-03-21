"""Experiment v3.19: Unified Thinking Module.

Test one recurrent mechanism on three task types:
  1. Simple retrieval: "What is the code?" → attend to memory → answer
  2. Multi-hop: "Where does Alice work?" → chain through memory → answer
  3. Computation: "What is the security code (ID + offset)?" → compute → answer

The thinking module should learn different attention patterns for
different task types — without being told which is which.

Architecture:
  Forward model (frozen) processes chunks → KV cache stored
  Thinking module: recurrent attention over [memory + working memory + thought history]
  Result injected into model for answer generation

Training: answer loss end-to-end through thinking module.
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

from rpvt.model.thinking_v2 import UnifiedThinkingModule
from rpvt.model.cross_attention_memory import MemoryBank, WriteWrapper
from rpvt.experiments.exp_v3_2_nlp_recall import _generate_natural_facts
from rpvt.data.inference_tasks import generate_inference_tasks


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


def build_mixed_dataset(tokenizer, n_retrieval=200, n_inference=200,
                        chunk_size=128, gap_range=(2, 4), seed=42):
    """Mix of retrieval + inference tasks."""
    rng = random.Random(seed)

    print("  Loading WikiText for filler...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    filler_texts = [t for t in wiki["text"] if len(t.strip()) > 100]

    def make_filler():
        ft = rng.choice(filler_texts)
        ft_tok = tokenizer.encode(ft, add_special_tokens=False)
        if len(ft_tok) >= chunk_size:
            start = rng.randint(0, len(ft_tok) - chunk_size)
            ct = ft_tok[start:start + chunk_size]
        else:
            ct = ft_tok + [tokenizer.eos_token_id or 0] * (chunk_size - len(ft_tok))
        return torch.tensor(ct, dtype=torch.long)

    def passage_to_chunks(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            ct = tokens[i:i + chunk_size]
            if len(ct) < chunk_size:
                ct = ct + [tokenizer.eos_token_id or 0] * (chunk_size - len(ct))
            chunks.append(torch.tensor(ct, dtype=torch.long))
        return chunks

    docs = []

    # Simple retrieval
    print(f"  Generating {n_retrieval} retrieval tasks...")
    retrieval_facts = _generate_natural_facts(rng, n_retrieval, max_qa_pairs=2)
    for passage, qa_pairs in retrieval_facts:
        p_chunks = passage_to_chunks(passage)
        gap = rng.randint(gap_range[0], gap_range[1])
        f_chunks = [make_filler() for _ in range(gap)]
        for qa in qa_pairs:
            qa_chunk, answer_mask = _make_qa_chunk(
                tokenizer, qa["question"], qa["answer"], chunk_size
            )
            docs.append({
                "type": "retrieval",
                "chunks": p_chunks + f_chunks + [qa_chunk],
                "answer_mask": answer_mask,
            })

    # Inference tasks (multi-hop, comparison, derivation)
    print(f"  Generating {n_inference} inference tasks...")
    inference_data = generate_inference_tasks(rng, n_inference)
    for passages, qa_pairs in inference_data:
        all_chunks = []
        for passage in passages:
            all_chunks.extend(passage_to_chunks(passage))
            all_chunks.append(make_filler())
        for qa in qa_pairs:
            qa_chunk, answer_mask = _make_qa_chunk(
                tokenizer, qa["question"], qa["answer"], chunk_size
            )
            docs.append({
                "type": qa.get("type", "inference"),
                "chunks": all_chunks + [qa_chunk],
                "answer_mask": answer_mask,
            })

    rng.shuffle(docs)
    type_counts = {}
    for d in docs:
        type_counts[d["type"]] = type_counts.get(d["type"], 0) + 1
    print(f"  Dataset: {len(docs)} docs — {type_counts}")
    return docs


def train(model, thinker, memory_bank, tokenizer, train_docs, eval_docs,
          device, num_epochs=15, lr=1e-3, log_every=100, checkpoint_dir=None):
    """Train the thinking module end-to-end from answer loss."""

    trainable = list(thinker.parameters()) + list(memory_bank.parameters())
    n_params = sum(p.numel() for p in trainable)
    print(f"\nTraining {n_params:,} params (thinker + memory gate)")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = len(train_docs) * num_epochs

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    embed_fn = model.model.embed_tokens

    print(f"  {total_steps} steps, {num_epochs} epochs")
    model.eval()
    thinker.train()
    global_step = 0
    losses_by_type = {}
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
            doc_type = doc["type"]

            memory_bank.reset()
            memory_bank.persistent_grad = True

            # Process context chunks → accumulate memory
            for chunk in chunks[:-1]:
                chunk_ids = chunk.unsqueeze(0).to(device)
                with torch.no_grad():
                    model(chunk_ids)

            memory_bank.detach_state()

            # Get memory states for thinking
            mem_states, n_active = memory_bank.get_active_memories()

            # Get question seed from QA chunk
            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            with torch.no_grad():
                qa_output = model(qa_chunk, output_hidden_states=True)
            query_seed = qa_output.hidden_states[14].mean(dim=1).squeeze(0)

            # THINK: recurrent attention over memory + working memory
            final_thought, _, _ = thinker(
                query_seed, memory_states=mem_states,
            )

            # Inject thought into model: prepend as context embedding
            thought_embed = final_thought.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden)
            token_embeds = embed_fn(qa_chunk)
            inputs_embeds = torch.cat([thought_embed, token_embeds], dim=1)

            output = model(inputs_embeds=inputs_embeds)
            # Trim thought position from logits
            output_logits = output.logits[:, 1:, :]

            # Answer loss
            logits = output_logits[:, :-1].reshape(-1, output_logits.size(-1))
            targets = qa_chunk[:, 1:].reshape(-1)
            per_token = F.cross_entropy(logits, targets, reduction='none')
            mask = answer_mask[:-1]
            n_tokens = mask.sum().clamp(min=1)
            loss = (per_token * mask).sum() / n_tokens

            if loss.item() > 0 and not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()

            memory_bank.persistent_grad = False

            if doc_type not in losses_by_type:
                losses_by_type[doc_type] = []
            losses_by_type[doc_type].append(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                elapsed = time.time() - start_time
                parts = []
                for t in sorted(losses_by_type.keys()):
                    recent = losses_by_type[t][-log_every:]
                    parts.append(f"{t}={sum(recent)/len(recent):.3f}")
                print(f"  step {global_step}/{total_steps}, "
                      f"{', '.join(parts)}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, {elapsed:.0f}s")

        # Eval
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} ===")
        eval_results = evaluate(model, thinker, memory_bank, tokenizer,
                                eval_docs, device)
        for task_type, metrics in sorted(eval_results.items()):
            print(f"  {task_type}: {metrics['acc']:.1%} "
                  f"({metrics['correct']}/{metrics['total']})")

        if checkpoint_dir:
            state = {
                "thinker": {n: p.data.clone() for n, p in thinker.named_parameters()},
                "memory": {n: p.data.clone() for n, p in memory_bank.named_parameters()},
                "epoch": epoch,
            }
            torch.save(state, os.path.join(checkpoint_dir, "latest.pt"))

        thinker.train()

    return eval_results


def evaluate(model, thinker, memory_bank, tokenizer, eval_docs, device):
    model.eval()
    thinker.eval()
    embed_fn = model.model.embed_tokens

    results_by_type = {}

    with torch.no_grad():
        for doc in eval_docs:
            chunks = doc["chunks"]
            answer_mask = doc["answer_mask"].to(device)
            doc_type = doc["type"]

            memory_bank.reset()

            for chunk in chunks[:-1]:
                model(chunk.unsqueeze(0).to(device))

            mem_states, n_active = memory_bank.get_active_memories()

            qa_chunk = chunks[-1].unsqueeze(0).to(device)
            qa_output = model(qa_chunk, output_hidden_states=True)
            query_seed = qa_output.hidden_states[14].mean(dim=1).squeeze(0)

            final_thought, _, _ = thinker(query_seed, memory_states=mem_states)

            thought_embed = final_thought.unsqueeze(0).unsqueeze(0)
            token_embeds = embed_fn(qa_chunk)
            inputs_embeds = torch.cat([thought_embed, token_embeds], dim=1)
            output = model(inputs_embeds=inputs_embeds)
            output_logits = output.logits[:, 1:, :]

            predictions = output_logits[0, :-1].argmax(dim=-1)
            targets = qa_chunk[0, 1:]
            mask = answer_mask[:-1]

            if doc_type not in results_by_type:
                results_by_type[doc_type] = {"correct": 0, "total": 0}

            for p in mask.nonzero(as_tuple=True)[0]:
                results_by_type[doc_type]["total"] += 1
                if predictions[p].item() == targets[p].item():
                    results_by_type[doc_type]["correct"] += 1

    for t in results_by_type:
        r = results_by_type[t]
        r["acc"] = r["correct"] / max(r["total"], 1)

    return results_by_type


def main():
    parser = argparse.ArgumentParser(description="v3.19: Unified Thinking")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-retrieval", type=int, default=300)
    parser.add_argument("--n-inference", type=int, default=200)
    parser.add_argument("--n-eval-retrieval", type=int, default=30)
    parser.add_argument("--n-eval-inference", type=int, default=30)
    parser.add_argument("--n-think-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_19_thinking")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Use base model + LoRA + cross-attention memory (proven 97%)
    from rpvt.experiments.exp_v3_1_pretrained_recall import build_model
    model, tokenizer = build_model(
        model_name=args.model_name, device=args.device,
        memory_layer=-1, memory_size=256, n_slots=64,
        decay=0.999, gate_bias=-2.0,
        lora_rank=args.lora_rank, lora_targets="q_proj,v_proj",
        no_memory=False, no_lora=False,
        init_qk_shared=False, n_extract=1, memory_mode="cross_attn",
    )

    from rpvt.experiments.exp_v3_2_nlp_recall import _get_memory_module
    memory_bank = _get_memory_module(model)
    hidden_size = 1536

    # Thinking module
    thinker = UnifiedThinkingModule(
        hidden_size=hidden_size, n_heads=8,
        n_work_slots=4, max_think_steps=args.n_think_steps,
    ).to(device=args.device, dtype=torch.bfloat16)

    n_think = sum(p.numel() for p in thinker.parameters())
    print(f"  Thinking module: {n_think:,} params, {args.n_think_steps} steps")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    print("\nBuilding data...")
    train_docs = build_mixed_dataset(
        tokenizer, n_retrieval=args.n_retrieval,
        n_inference=args.n_inference, seed=args.seed,
    )
    eval_docs = build_mixed_dataset(
        tokenizer, n_retrieval=args.n_eval_retrieval,
        n_inference=args.n_eval_inference, seed=args.seed + 1000,
    )

    results = train(
        model, thinker, memory_bank, tokenizer, train_docs, eval_docs,
        args.device, num_epochs=args.epochs, lr=args.lr,
        log_every=args.log_every, checkpoint_dir=checkpoint_dir,
    )

    results_ser = {k: v for k, v in results.items()}
    results_ser["config"] = vars(args)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results_ser, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
