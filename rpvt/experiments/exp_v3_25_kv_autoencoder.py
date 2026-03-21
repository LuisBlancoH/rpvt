"""v3.25: KV Cache Autoencoder — compress memory, preserve recall.

Train an autoencoder to compress KV cache (~3.5MB) into a compact
representation that still lets the model answer correctly.

End-to-end training:
1. Process passage → get KV cache (full)
2. Encode KV cache → latent (compressed)
3. Decode latent → reconstructed KV pairs
4. Model answers question using reconstructed KVs as past_key_values
5. Loss = answer correctness (not reconstruction MSE)

The encoder learns WHAT matters, the decoder learns to reconstruct
just enough for the model to answer.

Comparison:
- Full KV cache (uncompressed): ~74% recall baseline
- Autoencoded KV cache: target >50% recall at 10-75x compression

Usage:
    python -m rpvt.experiments.exp_v3_25_kv_autoencoder
    python -m rpvt.experiments.exp_v3_25_kv_autoencoder --n-latent 32 --n-output 64
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.kv_autoencoder import KVAutoencoder


def generate_memory_tasks(n, seed=42):
    """Generate passage + QA tasks that REQUIRE cross-chunk memory.

    The passage is processed in one forward pass, then the question
    is asked in a separate forward pass with only the KV cache as context.
    """
    rng = random.Random(seed)
    tasks = []

    names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank",
             "Grace", "Henry", "Ivy", "Jack", "Karen", "Leo",
             "Mary", "Nick", "Olivia", "Paul", "Quinn", "Rosa"]
    cities = ["NYC", "London", "Tokyo", "Paris", "Berlin", "Sydney",
              "Toronto", "Mumbai", "Cairo", "Seoul", "Rome", "Lima"]
    orgs = ["Acme Corp", "Globex", "Initech", "Umbrella", "Wayne Enterprises",
            "Stark Industries", "Oscorp", "LexCorp", "Cyberdyne", "Weyland"]
    roles = ["engineer", "researcher", "director", "analyst", "designer",
             "consultant", "manager", "scientist", "architect", "lead"]

    for _ in range(n):
        name = rng.choice(names)
        city = rng.choice(cities)
        org = rng.choice(orgs)
        role = rng.choice(roles)
        year = rng.randint(2010, 2024)
        code = ''.join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=6))

        passage = (
            f"{name} works at {org} as a {role}. "
            f"Based in {city} since {year}. "
            f"Employee code: {code}."
        )

        q_type = rng.choice(["org", "role", "city", "year", "code"])
        if q_type == "org":
            question = f"Where does {name} work?"
            answer = org
        elif q_type == "role":
            question = f"What is {name}'s role?"
            answer = role
        elif q_type == "city":
            question = f"What city is {name} based in?"
            answer = city
        elif q_type == "year":
            question = f"Since what year has {name} been there?"
            answer = str(year)
        else:
            question = f"What is {name}'s employee code?"
            answer = code

        tasks.append({
            "passage": passage,
            "question": question,
            "answer": answer,
            "type": q_type,
        })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="v3.25: KV Autoencoder")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--n-latent", type=int, default=16,
                        help="Number of latent vectors")
    parser.add_argument("--n-output", type=int, default=32,
                        help="Number of output KV positions")
    parser.add_argument("--latent-dim", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--output-dir", type=str,
                        default="results/kv_autoencoder")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False

    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    print(f"  Layers: {n_layers}, KV heads: {n_kv_heads}, head_dim: {head_dim}")

    # Create autoencoder
    autoencoder = KVAutoencoder(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        latent_dim=args.latent_dim,
        n_latent=args.n_latent,
        n_output_tokens=args.n_output,
    ).to(dtype=torch.bfloat16, device=device)

    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.lr)

    # Generate tasks
    train_tasks = generate_memory_tasks(args.n_train, seed=42)
    eval_tasks = generate_memory_tasks(args.n_eval, seed=123)
    print(f"Tasks: {args.n_train} train, {args.n_eval} eval")

    # ── Baseline: full KV cache ────────────────────────────
    print("\n--- Baseline: full KV cache ---")
    correct_full = 0
    for task in eval_tasks[:50]:
        # Process passage
        passage_ids = tokenizer(task["passage"], return_tensors="pt").to(device)
        with torch.no_grad():
            passage_out = model(passage_ids["input_ids"], use_cache=True)
        full_kvs = passage_out.past_key_values

        # Ask question with full KVs
        qa_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["question"]}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        qa_ids = tokenizer(qa_text, return_tensors="pt").to(device)

        # Build position_ids and attention_mask for KV cache
        n_past = passage_ids["input_ids"].shape[1]
        n_new = qa_ids["input_ids"].shape[1]
        position_ids = torch.arange(
            n_past, n_past + n_new, device=device
        ).unsqueeze(0)
        attn_mask = torch.ones(1, n_past + n_new, device=device, dtype=torch.long)

        with torch.no_grad():
            qa_out = model(
                qa_ids["input_ids"],
                past_key_values=full_kvs,
                position_ids=position_ids,
                attention_mask=attn_mask,
            )

        # Generate
        gen_tokens = []
        gen_ids = qa_ids["input_ids"]
        past_kvs = qa_out.past_key_values
        cur_pos = n_past + n_new
        gen_mask = attn_mask.clone()

        for _ in range(20):
            logits = qa_out.logits if len(gen_tokens) == 0 else out.logits
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_tok.item() == tokenizer.eos_token_id:
                break
            gen_tokens.append(next_tok.item())

            pos = torch.tensor([[cur_pos]], device=device)
            gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1)
            with torch.no_grad():
                out = model(next_tok, past_key_values=past_kvs,
                           position_ids=pos, attention_mask=gen_mask)
            past_kvs = out.past_key_values
            cur_pos += 1

        answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if task["answer"].lower() in answer.lower():
            correct_full += 1

    print(f"  Full KV baseline: {correct_full}/50 ({100*correct_full/50:.1f}%)")

    # ── Training: autoencoder ──────────────────────────────
    print(f"\n--- Training autoencoder ({args.epochs} epochs) ---")

    for epoch in range(args.epochs):
        autoencoder.train()
        total_loss = 0
        n = 0
        t0 = time.time()

        indices = list(range(len(train_tasks)))
        random.shuffle(indices)

        for idx in indices:
            task = train_tasks[idx]

            # Process passage → get KV cache
            passage_ids = tokenizer(task["passage"], return_tensors="pt").to(device)
            with torch.no_grad():
                passage_out = model(passage_ids["input_ids"], use_cache=True)
            full_kvs = passage_out.past_key_values

            # Extract KV pairs into list format
            kv_list = []
            for layer_idx in range(n_layers):
                k = full_kvs.layers[layer_idx].keys.squeeze(0)  # (n_heads, seq, head_dim)
                v = full_kvs.layers[layer_idx].values.squeeze(0)
                # Transpose to (seq, n_heads, head_dim)
                kv_list.append((k.transpose(0, 1), v.transpose(0, 1)))

            # Encode → decode
            reconstructed, latent = autoencoder(kv_list)

            # Use reconstructed KVs for question answering
            qa_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": task["question"]},
                    {"role": "assistant", "content": task["answer"]},
                ],
                tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            qa_ids = tokenizer(qa_text, return_tensors="pt").to(device)

            # Build past_key_values from reconstructed
            past_kvs = autoencoder.to_past_key_values(reconstructed)

            n_past = args.n_output  # reconstructed has n_output positions
            n_new = qa_ids["input_ids"].shape[1]
            position_ids = torch.arange(
                n_past, n_past + n_new, device=device
            ).unsqueeze(0)
            attn_mask = torch.ones(
                1, n_past + n_new, device=device, dtype=torch.long
            )

            # Forward with reconstructed KVs
            qa_out = model(
                qa_ids["input_ids"],
                past_key_values=past_kvs,
                position_ids=position_ids,
                attention_mask=attn_mask,
            )

            # Answer loss
            labels = qa_ids["input_ids"].clone()
            answer_tokens = tokenizer(
                task["answer"], add_special_tokens=False
            )["input_ids"]
            answer_len = len(answer_tokens)
            labels[:, :-(answer_len + 1)] = -100

            logits = qa_out.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n, 1)

        # Quick eval
        autoencoder.eval()
        correct = 0
        for task in eval_tasks[:20]:
            passage_ids = tokenizer(task["passage"], return_tensors="pt").to(device)
            with torch.no_grad():
                passage_out = model(passage_ids["input_ids"], use_cache=True)
            full_kvs = passage_out.past_key_values

            kv_list = []
            for li in range(n_layers):
                k = full_kvs.layers[li].keys.squeeze(0).transpose(0, 1)
                v = full_kvs.layers[li].values.squeeze(0).transpose(0, 1)
                kv_list.append((k, v))

            with torch.no_grad():
                recon, _ = autoencoder(kv_list)
                past_kvs = autoencoder.to_past_key_values(recon)

            qa_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["question"]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            qa_ids_eval = tokenizer(qa_text, return_tensors="pt").to(device)

            n_past = args.n_output
            n_new = qa_ids_eval["input_ids"].shape[1]
            pos_ids = torch.arange(n_past, n_past + n_new, device=device).unsqueeze(0)
            a_mask = torch.ones(1, n_past + n_new, device=device, dtype=torch.long)

            # Generate
            with torch.no_grad():
                out = model(qa_ids_eval["input_ids"], past_key_values=past_kvs,
                           position_ids=pos_ids, attention_mask=a_mask)

            gen_tokens = []
            cur_pos = n_past + n_new
            gen_past = out.past_key_values
            gen_mask = a_mask.clone()

            for _ in range(20):
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if next_tok.item() == tokenizer.eos_token_id:
                    break
                gen_tokens.append(next_tok.item())
                pos = torch.tensor([[cur_pos]], device=device)
                gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1)
                with torch.no_grad():
                    out = model(next_tok, past_key_values=gen_past,
                               position_ids=pos, attention_mask=gen_mask)
                gen_past = out.past_key_values
                cur_pos += 1

            answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            if task["answer"].lower() in answer.lower():
                correct += 1

        acc = 100 * correct / 20
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, "
              f"recall={acc:.1f}%, ({elapsed:.0f}s)")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "autoencoder": autoencoder.state_dict(),
                "args": vars(args),
            }, output_dir / f"checkpoint_epoch{epoch+1}.pt")

    # ── Final evaluation ───────────────────────────────────
    print(f"\n--- Final evaluation ---")
    autoencoder.eval()
    correct_ae = 0

    for task in eval_tasks[:50]:
        passage_ids = tokenizer(task["passage"], return_tensors="pt").to(device)
        with torch.no_grad():
            passage_out = model(passage_ids["input_ids"], use_cache=True)
        full_kvs = passage_out.past_key_values

        kv_list = []
        for li in range(n_layers):
            k = full_kvs.layers[li].keys.squeeze(0).transpose(0, 1)
            v = full_kvs.layers[li].values.squeeze(0).transpose(0, 1)
            kv_list.append((k, v))

        with torch.no_grad():
            recon, _ = autoencoder(kv_list)
            past_kvs = autoencoder.to_past_key_values(recon)

        qa_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["question"]}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        qa_ids_eval = tokenizer(qa_text, return_tensors="pt").to(device)

        n_past = args.n_output
        n_new = qa_ids_eval["input_ids"].shape[1]
        pos_ids = torch.arange(n_past, n_past + n_new, device=device).unsqueeze(0)
        a_mask = torch.ones(1, n_past + n_new, device=device, dtype=torch.long)

        with torch.no_grad():
            out = model(qa_ids_eval["input_ids"], past_key_values=past_kvs,
                       position_ids=pos_ids, attention_mask=a_mask)

        gen_tokens = []
        cur_pos = n_past + n_new
        gen_past = out.past_key_values
        gen_mask = a_mask.clone()

        for _ in range(20):
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_tok.item() == tokenizer.eos_token_id:
                break
            gen_tokens.append(next_tok.item())
            pos = torch.tensor([[cur_pos]], device=device)
            gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1)
            with torch.no_grad():
                out = model(next_tok, past_key_values=gen_past,
                           position_ids=pos, attention_mask=gen_mask)
            gen_past = out.past_key_values
            cur_pos += 1

        answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if task["answer"].lower() in answer.lower():
            correct_ae += 1

    ae_acc = 100 * correct_ae / 50

    # Size comparison
    original_size = n_layers * 2 * n_kv_heads * head_dim * 2  # per token, bytes (bf16)
    # Full KV: passage_len tokens × original_size
    # We don't know exact passage_len, estimate ~30 tokens
    full_size_est = 30 * original_size
    compressed_size = args.n_latent * args.latent_dim * 2  # bf16
    ratio = full_size_est / max(compressed_size, 1)

    print(f"\n--- Results ---")
    print(f"  Full KV cache:     {correct_full}/50 ({100*correct_full/50:.1f}%)")
    print(f"  Autoencoded KV:    {correct_ae}/50 ({ae_acc:.1f}%)")
    print(f"  Compression ratio: ~{ratio:.0f}x")
    print(f"  Full size (est):   ~{full_size_est/1024:.0f}KB per passage")
    print(f"  Compressed size:   ~{compressed_size/1024:.0f}KB per passage")

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "full_kv_accuracy": 100 * correct_full / 50,
            "autoencoded_accuracy": ae_acc,
            "compression_ratio": ratio,
        }, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
