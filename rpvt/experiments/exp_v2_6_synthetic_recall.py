"""Experiment v2.6: Synthetic recall task — can M do memory at all?

Tests whether M can store and retrieve information across chunk boundaries
when the task *requires* it. Natural language modeling doesn't force memory
(the adapter basin is too deep). This task does.

Setup:
    - Synthetic sequences with a "key-value" pair in chunk 0
    - Filler tokens in chunks 1..K-1
    - A "query" in chunk K that requires recalling the value from chunk 0
    - The transformer alone can't solve this (value is outside context window)
    - M is the only path for cross-chunk retrieval

Success criteria:
    - M-equipped model achieves near-perfect recall accuracy
    - Accuracy stays high as gap K increases (tests memory horizon)
    - Without M, accuracy is at chance level
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rpvt.model.fast_weight import (
    attach_fast_weight_memory,
    TransformerLayerWithMemory,
)


class SyntheticRecallDataset(Dataset):
    """Synthetic task: store a value, recall it after a gap.

    Each document is a sequence of chunks:
        Chunk 0: "STORE <key_id> <value_id> PAD PAD ..."
        Chunks 1..gap-1: random filler tokens
        Chunk gap: "RECALL <key_id> PAD ... <value_id>"

    The model must predict <value_id> at the end of the recall chunk.
    This is impossible without cross-chunk memory since chunk gap
    has no information about the value stored in chunk 0.

    We use a small vocabulary (special tokens + filler vocab) to keep
    the task clean and fast.
    """

    def __init__(
        self,
        n_docs=1000,
        gap_range=(2, 10),
        chunk_size=64,
        n_keys=32,
        n_values=64,
        filler_vocab_size=256,
        seed=42,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_keys = n_keys
        self.n_values = n_values

        # Special tokens (offset above filler vocab)
        self.STORE = filler_vocab_size
        self.RECALL = filler_vocab_size + 1
        self.PAD = filler_vocab_size + 2
        self.key_offset = filler_vocab_size + 3
        self.value_offset = self.key_offset + n_keys
        self.vocab_size = self.value_offset + n_values

        rng = torch.Generator().manual_seed(seed)

        self.documents = []

        for _ in range(n_docs):
            gap = torch.randint(gap_range[0], gap_range[1] + 1, (1,), generator=rng).item()
            key_id = torch.randint(0, n_keys, (1,), generator=rng).item()
            value_id = torch.randint(0, n_values, (1,), generator=rng).item()

            chunks = []

            # Chunk 0: STORE key value PAD...
            store_chunk = torch.full((chunk_size,), self.PAD, dtype=torch.long)
            store_chunk[0] = self.STORE
            store_chunk[1] = self.key_offset + key_id
            store_chunk[2] = self.value_offset + value_id
            chunks.append(store_chunk)

            # Filler chunks
            for _ in range(gap - 1):
                filler = torch.randint(0, filler_vocab_size, (chunk_size,), generator=rng)
                chunks.append(filler)

            # Recall chunk: RECALL key PAD... value (value is the LAST token)
            recall_chunk = torch.full((chunk_size,), self.PAD, dtype=torch.long)
            recall_chunk[0] = self.RECALL
            recall_chunk[1] = self.key_offset + key_id
            recall_chunk[-1] = self.value_offset + value_id
            chunks.append(recall_chunk)

            self.documents.append({
                "chunks": chunks,
                "key_id": key_id,
                "value_id": value_id,
                "gap": gap,
            })

        # Flatten into sequential chunks with doc boundaries
        self.all_chunks = []
        self.doc_ids = []
        self.is_first = []
        self.is_recall = []
        self.recall_value_id = []

        for doc_idx, doc in enumerate(self.documents):
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                self.all_chunks.append(chunk)
                self.doc_ids.append(doc_idx)
                self.is_first.append(chunk_idx == 0)
                is_recall_chunk = (chunk_idx == len(doc["chunks"]) - 1)
                self.is_recall.append(is_recall_chunk)
                self.recall_value_id.append(
                    doc["value_id"] if is_recall_chunk else -1
                )

        # Also build whole-document sequences (concatenated chunks)
        self.doc_sequences = []  # list of (input_ids_tensor, value_id, gap)
        for doc in self.documents:
            seq = torch.cat(doc["chunks"])  # (n_chunks * chunk_size,)
            self.doc_sequences.append({
                "input_ids": seq,
                "value_id": doc["value_id"],
                "gap": doc["gap"],
            })

        print(f"  SyntheticRecall: {n_docs} docs, {len(self.all_chunks)} chunks, "
              f"gap range {gap_range}, {n_keys} keys, {n_values} values, "
              f"vocab size {self.vocab_size}")

    def __len__(self):
        return len(self.all_chunks)

    def __getitem__(self, idx):
        return {
            "input_ids": self.all_chunks[idx],
            "doc_id": self.doc_ids[idx],
            "is_first": self.is_first[idx],
            "is_recall": self.is_recall[idx],
            "recall_value_id": self.recall_value_id[idx],
        }


def recall_collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "doc_ids": [b["doc_id"] for b in batch],
        "is_first": [b["is_first"] for b in batch],
        "is_recall": [b["is_recall"] for b in batch],
        "recall_value_id": [b["recall_value_id"] for b in batch],
    }


class SmallTransformerLM(nn.Module):
    """Small transformer for the synthetic task.

    We don't use GPT-2 here because:
    1. The vocab is tiny (< 500 tokens)
    2. We want to train from scratch so M can't cheat via pretrained knowledge
    3. Faster iteration
    """

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embed.weight

        # For compatibility with memory attachment
        self.h = nn.ModuleList([
            self.transformer.layers[i] for i in range(n_layers)
        ])

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1
        )

        # Run through layers (use self.h so memory-wrapped layers work)
        for layer in self.h:
            x = layer(x, src_mask=causal_mask, is_causal=True)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )

        return type('Output', (), {'loss': loss, 'logits': logits})()


def reset_memories(model):
    for layer in model.h:
        if isinstance(layer, TransformerLayerWithMemory):
            layer.reset_memory()


def train_and_eval(
    model, train_dataset, eval_dataset, device,
    num_epochs=10, lr=1e-3, log_every=50, model_name="model",
    whole_doc=False,
):
    """Train and evaluate.

    If whole_doc=True, feed entire documents as single sequences so gradient
    flows through M across all chunks. This lets M learn what to store.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    if whole_doc:
        total_steps = len(train_dataset.doc_sequences) * num_epochs
    else:
        total_steps = len(train_dataset.all_chunks) * num_epochs

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"\nTraining {model_name}: {total_steps} steps, whole_doc={whole_doc}")
    model.train()
    global_step = 0
    train_losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        if whole_doc:
            # Feed entire documents as single sequences
            for doc in train_dataset.doc_sequences:
                input_ids = doc["input_ids"].unsqueeze(0).to(device)  # (1, n_chunks*chunk_size)
                reset_memories(model)

                output = model(input_ids, labels=input_ids)
                loss = output.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = sum(train_losses[-log_every:]) / log_every
                    elapsed = time.time() - start_time
                    print(f"  [{model_name}] step {global_step}/{total_steps}, "
                          f"loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}, "
                          f"{elapsed:.0f}s")
        else:
            # Feed individual chunks (original behavior)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                                      collate_fn=recall_collate_fn)
            prev_doc_id = None
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                doc_ids = batch["doc_ids"]
                is_first = batch["is_first"]

                if any(is_first) or (prev_doc_id is not None and doc_ids[0] != prev_doc_id):
                    reset_memories(model)
                prev_doc_id = doc_ids[0]

                output = model(input_ids, labels=input_ids)
                loss = output.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = sum(train_losses[-log_every:]) / log_every
                    elapsed = time.time() - start_time
                    print(f"  [{model_name}] step {global_step}/{total_steps}, "
                          f"loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}, "
                          f"{elapsed:.0f}s")

    if whole_doc:
        return evaluate_recall_whole_doc(model, eval_dataset, device, model_name)
    else:
        return evaluate_recall(model, eval_dataset, device, model_name)


def evaluate_recall(model, dataset, device, model_name="model"):
    """Evaluate recall accuracy: can the model predict the stored value?"""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=recall_collate_fn)

    correct = 0
    total = 0
    correct_by_gap = {}
    total_by_gap = {}
    prev_doc_id = None
    reset_memories(model)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            doc_ids = batch["doc_ids"]
            is_first = batch["is_first"]
            is_recall = batch["is_recall"]
            recall_value_id = batch["recall_value_id"]

            if any(is_first) or (prev_doc_id is not None and doc_ids[0] != prev_doc_id):
                reset_memories(model)
            prev_doc_id = doc_ids[0]

            output = model(input_ids, labels=input_ids)

            if any(is_recall):
                # Logits at position -2 predict token at position -1
                logits = output.logits[0, -2, :]
                value_id = recall_value_id[0]
                value_token = dataset.value_offset + value_id
                predicted = logits.argmax().item()

                if predicted == value_token:
                    correct += 1
                total += 1

                doc = dataset.documents[doc_ids[0]]
                gap = doc["gap"]
                correct_by_gap[gap] = correct_by_gap.get(gap, 0) + (1 if predicted == value_token else 0)
                total_by_gap[gap] = total_by_gap.get(gap, 0) + 1

    accuracy = correct / max(total, 1)
    print(f"\n  [{model_name}] Recall accuracy: {correct}/{total} = {accuracy:.1%}")

    if total_by_gap:
        print(f"\n  {'Gap':>5} | {'Correct':>8} | {'Total':>6} | {'Accuracy':>8}")
        print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")
        for gap in sorted(total_by_gap):
            acc = correct_by_gap.get(gap, 0) / total_by_gap[gap]
            print(f"  {gap:>5} | {correct_by_gap.get(gap, 0):>8} | {total_by_gap[gap]:>6} | {acc:>7.1%}")

    accuracy_by_gap = {
        gap: correct_by_gap.get(gap, 0) / total_by_gap[gap]
        for gap in sorted(total_by_gap)
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "accuracy_by_gap": accuracy_by_gap,
    }


def evaluate_recall_whole_doc(model, dataset, device, model_name="model"):
    """Evaluate recall by feeding whole documents."""
    model.eval()

    correct = 0
    total = 0
    correct_by_gap = {}
    total_by_gap = {}

    with torch.no_grad():
        for doc in dataset.doc_sequences:
            input_ids = doc["input_ids"].unsqueeze(0).to(device)
            reset_memories(model)

            output = model(input_ids, labels=input_ids)

            # The value token is at position -1, so logits at -2 predict it
            logits = output.logits[0, -2, :]
            value_token = dataset.value_offset + doc["value_id"]
            predicted = logits.argmax().item()

            if predicted == value_token:
                correct += 1
            total += 1

            gap = doc["gap"]
            correct_by_gap[gap] = correct_by_gap.get(gap, 0) + (1 if predicted == value_token else 0)
            total_by_gap[gap] = total_by_gap.get(gap, 0) + 1

    accuracy = correct / max(total, 1)
    print(f"\n  [{model_name}] Recall accuracy: {correct}/{total} = {accuracy:.1%}")

    if total_by_gap:
        print(f"\n  {'Gap':>5} | {'Correct':>8} | {'Total':>6} | {'Accuracy':>8}")
        print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")
        for gap in sorted(total_by_gap):
            acc = correct_by_gap.get(gap, 0) / total_by_gap[gap]
            print(f"  {gap:>5} | {correct_by_gap.get(gap, 0):>8} | {total_by_gap[gap]:>6} | {acc:>7.1%}")

    accuracy_by_gap = {
        gap: correct_by_gap.get(gap, 0) / total_by_gap[gap]
        for gap in sorted(total_by_gap)
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "accuracy_by_gap": accuracy_by_gap,
    }


def evaluate_recall_no_memory(model, dataset, device):
    """Evaluate with M reset before every chunk."""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=recall_collate_fn)

    correct = 0
    total = 0
    correct_by_gap = {}
    total_by_gap = {}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            is_recall = batch["is_recall"]
            doc_ids = batch["doc_ids"]

            reset_memories(model)

            output = model(input_ids, labels=input_ids)

            if any(is_recall):
                logits = output.logits[0, -2, :]
                value_id = batch["recall_value_id"][0]
                value_token = dataset.value_offset + value_id
                predicted = logits.argmax().item()

                if predicted == value_token:
                    correct += 1
                total += 1

                doc = dataset.documents[doc_ids[0]]
                gap = doc["gap"]
                correct_by_gap[gap] = correct_by_gap.get(gap, 0) + (1 if predicted == value_token else 0)
                total_by_gap[gap] = total_by_gap.get(gap, 0) + 1

    accuracy = correct / max(total, 1)
    accuracy_by_gap = {
        gap: correct_by_gap.get(gap, 0) / total_by_gap[gap]
        for gap in sorted(total_by_gap)
    }
    return {"accuracy": accuracy, "accuracy_by_gap": accuracy_by_gap}


def main():
    parser = argparse.ArgumentParser(description="v2.6: synthetic recall task")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--memory-size", type=int, default=128)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--write-mode", type=str, default="uniform",
                        choices=["uniform", "gate", "surprise", "surprise-fwd",
                                 "surprise-fwd-store", "predictive"])
    parser.add_argument("--max-m-norm", type=float, default=0,
                        help="Cap on M norm (0 = no cap)")
    parser.add_argument("--bptt-steps", type=int, default=0,
                        help="BPTT steps for M (0 = always detach)")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--n-keys", type=int, default=32)
    parser.add_argument("--n-values", type=int, default=64)
    parser.add_argument("--gap-min", type=int, default=2)
    parser.add_argument("--gap-max", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v2_6_recall")
    parser.add_argument("--no-memory", action="store_true",
                        help="Run without memory (baseline)")
    parser.add_argument("--whole-doc", action="store_true",
                        help="Feed whole documents as single sequences (enables gradient flow through M)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    print("Creating synthetic recall datasets...")
    train_data = SyntheticRecallDataset(
        n_docs=args.n_train, gap_range=(args.gap_min, args.gap_max),
        chunk_size=args.chunk_size, n_keys=args.n_keys, n_values=args.n_values,
        seed=42,
    )
    eval_data = SyntheticRecallDataset(
        n_docs=args.n_eval, gap_range=(args.gap_min, args.gap_max),
        chunk_size=args.chunk_size, n_keys=args.n_keys, n_values=args.n_values,
        seed=1337,
    )

    # max_len must fit the longest document when using whole-doc mode
    if args.whole_doc:
        max_len = (args.gap_max + 1) * args.chunk_size
        print(f"\n  Whole-doc mode: max_len = {max_len} "
              f"({args.gap_max + 1} chunks × {args.chunk_size} tokens)")
    else:
        max_len = args.chunk_size

    print(f"\nBuilding model (d={args.d_model}, L={args.n_layers}, H={args.n_heads})...")
    model = SmallTransformerLM(
        vocab_size=train_data.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=max_len,
    ).to(device)

    if not args.no_memory:
        print(f"  Attaching memory (size={args.memory_size}, decay={args.decay}, "
              f"mode={args.write_mode}, bptt={args.bptt_steps})")
        memory_modules = attach_fast_weight_memory(
            model.h,
            hidden_size=args.d_model,
            memory_size=args.memory_size,
            decay=args.decay,
            write_mode=args.write_mode,
            max_m_norm=args.max_m_norm,
            bptt_steps=args.bptt_steps,
        )
    else:
        print("  No memory (baseline)")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params:,}, trainable: {n_trainable:,}")

    model_name = "no_memory" if args.no_memory else f"memory_d={args.decay}_{args.write_mode}"
    if args.whole_doc:
        model_name += "_wholedoc"
    results = train_and_eval(
        model, train_data, eval_data, device,
        num_epochs=args.epochs, lr=args.lr,
        log_every=args.log_every, model_name=model_name,
        whole_doc=args.whole_doc,
    )

    results["config"] = vars(args)

    if not args.no_memory and not args.whole_doc:
        # Chunk-based ablation: reset M before every chunk
        print(f"\n  Testing with memory reset before every chunk...")
        reset_results = evaluate_recall_no_memory(model, eval_data, device)
        results["no_memory_accuracy"] = reset_results["accuracy"]
        results["no_memory_accuracy_by_gap"] = reset_results["accuracy_by_gap"]
        print(f"  Without M: {reset_results['accuracy']:.1%} "
              f"(vs with M: {results['accuracy']:.1%})")

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
