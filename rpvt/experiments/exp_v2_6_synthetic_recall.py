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
        n_pairs=1,
        filler_vocab_size=256,
        seed=42,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_keys = n_keys
        self.n_values = n_values
        self.n_pairs = n_pairs

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

            # Generate n_pairs distinct key-value pairs per document
            pairs = []
            used_keys = set()
            for _ in range(n_pairs):
                # Ensure unique keys within a document
                key_id = torch.randint(0, n_keys, (1,), generator=rng).item()
                while key_id in used_keys and len(used_keys) < n_keys:
                    key_id = torch.randint(0, n_keys, (1,), generator=rng).item()
                used_keys.add(key_id)
                value_id = torch.randint(0, n_values, (1,), generator=rng).item()
                pairs.append((key_id, value_id))

            chunks = []

            # Store chunks (one per pair)
            for key_id, value_id in pairs:
                store_chunk = torch.full((chunk_size,), self.PAD, dtype=torch.long)
                store_chunk[0] = self.STORE
                store_chunk[1] = self.key_offset + key_id
                store_chunk[2] = self.value_offset + value_id
                chunks.append(store_chunk)

            # Filler chunks (gap-1 to maintain backward compat: gap = chunks from store to recall)
            for _ in range(gap - 1):
                filler = torch.randint(0, filler_vocab_size, (chunk_size,), generator=rng)
                chunks.append(filler)

            # Recall chunks (one per pair, same order as stores)
            # Key is placed at position -2 (right before value) so the model
            # can easily form key-dependent queries at the prediction position.
            for key_id, value_id in pairs:
                recall_chunk = torch.full((chunk_size,), self.PAD, dtype=torch.long)
                recall_chunk[0] = self.RECALL
                recall_chunk[-2] = self.key_offset + key_id
                recall_chunk[-1] = self.value_offset + value_id
                chunks.append(recall_chunk)

            self.documents.append({
                "chunks": chunks,
                "pairs": pairs,
                "gap": gap,
                # Keep single-pair compat
                "key_id": pairs[0][0],
                "value_id": pairs[0][1],
            })

        # Flatten into sequential chunks with doc boundaries
        self.all_chunks = []
        self.doc_ids = []
        self.is_first = []
        self.is_recall = []
        self.recall_value_id = []

        for doc_idx, doc in enumerate(self.documents):
            n_chunks = len(doc["chunks"])
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                self.all_chunks.append(chunk)
                self.doc_ids.append(doc_idx)
                self.is_first.append(chunk_idx == 0)
                # Recall chunks are the last n_pairs chunks
                recall_pair_idx = chunk_idx - (n_chunks - n_pairs)
                is_recall_chunk = recall_pair_idx >= 0
                self.is_recall.append(is_recall_chunk)
                self.recall_value_id.append(
                    doc["pairs"][recall_pair_idx][1] if is_recall_chunk else -1
                )

        # Also build whole-document sequences (concatenated chunks)
        self.doc_sequences = []
        for doc in self.documents:
            seq = torch.cat(doc["chunks"])  # (n_chunks * chunk_size,)
            # Recall value positions: last token of each recall chunk
            # Recall chunks start at index (n_pairs + gap - 1)
            recall_positions = []
            for i in range(n_pairs):
                recall_chunk_idx = n_pairs + (doc["gap"] - 1) + i
                pos = (recall_chunk_idx + 1) * chunk_size - 1  # last token of chunk
                recall_positions.append(pos)
            self.doc_sequences.append({
                "input_ids": seq,
                "pairs": doc["pairs"],
                "value_id": doc["value_id"],
                "gap": doc["gap"],
                "recall_positions": recall_positions,
            })

        print(f"  SyntheticRecall: {n_docs} docs, {len(self.all_chunks)} chunks, "
              f"gap range {gap_range}, {n_keys} keys, {n_values} values, "
              f"{n_pairs} pairs/doc, vocab size {self.vocab_size}")

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
    whole_doc=False, recall_loss_weight=1.0, recall_only_loss=False,
    curriculum_train_data=None,
):
    """Train and evaluate.

    If whole_doc=True, feed entire documents as single sequences so gradient
    flows through M across all chunks. This lets M learn what to store.

    recall_loss_weight: multiply the loss at the recall position by this factor.
    Default 1.0 = uniform weighting. Higher values (e.g. 100) amplify the
    recall signal so it doesn't get buried by filler token predictions.

    curriculum_train_data: if provided, train on this (1-pair) dataset for the
    first half of epochs, then switch to train_dataset (N-pair) for the rest.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    if whole_doc:
        total_steps = len(train_dataset.doc_sequences) * num_epochs
    else:
        total_steps = len(train_dataset.all_chunks) * num_epochs

    curriculum_switch_epoch = num_epochs // 2 if curriculum_train_data is not None else 0

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"\nTraining {model_name}: {total_steps} steps, whole_doc={whole_doc}")
    if curriculum_train_data is not None:
        print(f"  Curriculum: epochs 1-{curriculum_switch_epoch} = 1 pair, "
              f"epochs {curriculum_switch_epoch+1}-{num_epochs} = {train_dataset.n_pairs} pairs")
    model.train()
    global_step = 0
    train_losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        # Curriculum: use 1-pair data for first half, N-pair for second half
        if curriculum_train_data is not None and epoch < curriculum_switch_epoch:
            active_dataset = curriculum_train_data
            phase = "phase1"
        else:
            active_dataset = train_dataset
            phase = "phase2" if curriculum_train_data is not None else ""

        if curriculum_train_data is not None and epoch == curriculum_switch_epoch:
            print(f"  [{model_name}] === CURRICULUM SWITCH: 1 pair → {train_dataset.n_pairs} pairs ===")

        if whole_doc:
            # Feed entire documents as single sequences
            for doc in active_dataset.doc_sequences:
                input_ids = doc["input_ids"].unsqueeze(0).to(device)  # (1, n_chunks*chunk_size)
                reset_memories(model)

                output = model(input_ids, labels=input_ids)

                if recall_only_loss or recall_loss_weight > 1.0:
                    # Recompute loss with custom weighting on recall positions
                    logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
                    targets = input_ids[:, 1:].reshape(-1)
                    per_token_loss = F.cross_entropy(logits, targets, reduction='none')
                    if recall_only_loss:
                        # Zero weight on everything except recall positions
                        weights = torch.zeros_like(per_token_loss)
                    else:
                        weights = torch.ones_like(per_token_loss)
                    # Weight all recall positions (last token of each recall chunk)
                    for rpos in doc["recall_positions"]:
                        # logit at rpos-1 predicts token at rpos
                        weights[rpos - 1] = recall_loss_weight if not recall_only_loss else 1.0
                    loss = (per_token_loss * weights).sum() / weights.sum()
                else:
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


def _analyze_gate_values(model, dataset, device, n_docs=50):
    """Analyze gate values by chunk type (store, filler, recall).

    Returns dict with mean/std gate values for each chunk type, or None if no gates.
    """
    import types as _types

    # Find memory modules with gates
    memory_modules = []
    for module in model.modules():
        if isinstance(module, FastWeightMemory) and module.write_mode == "gate":
            memory_modules.append(module)

    if not memory_modules:
        return None

    chunk_size = dataset.chunk_size
    gate_by_type = {"store": [], "filler": [], "recall": []}

    docs = dataset.doc_sequences[:n_docs]
    mem = memory_modules[0]  # analyze first layer

    for doc in docs:
        input_ids = doc["input_ids"].unsqueeze(0).to(device)
        seq_len = input_ids.shape[1]
        n_chunks = seq_len // chunk_size
        gap = doc["gap"]
        n_pairs = dataset.n_pairs

        reset_memories(model)

        # Capture gate values by patching forward
        mem._captured_gates = None
        orig_forward = mem.forward

        def make_capture_forward(m, orig):
            def capture_forward(x, chunk_size=64):
                m._captured_gates = torch.sigmoid(m.W_gate(x)).detach().cpu().squeeze(-1).squeeze(0)
                return orig(x, chunk_size)
            return capture_forward

        mem.forward = make_capture_forward(mem, orig_forward)

        with torch.no_grad():
            model(input_ids)

        mem.forward = orig_forward

        if mem._captured_gates is not None:
            gates = mem._captured_gates  # (seq_len,)
            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = start + chunk_size
                chunk_mean = gates[start:end].mean().item()

                if chunk_idx < n_pairs:
                    gate_by_type["store"].append(chunk_mean)
                elif chunk_idx < n_pairs + (gap - 1):
                    gate_by_type["filler"].append(chunk_mean)
                else:
                    gate_by_type["recall"].append(chunk_mean)

    result = {}
    for ctype in ["store", "filler", "recall"]:
        vals = gate_by_type[ctype]
        if vals:
            t = torch.tensor(vals)
            result[ctype] = {"mean": t.mean().item(), "std": t.std().item()}
        else:
            result[ctype] = {"mean": 0.0, "std": 0.0}
    return result


def evaluate_recall_whole_doc(model, dataset, device, model_name="model"):
    """Evaluate recall by feeding whole documents."""
    model.eval()

    correct = 0
    total = 0
    correct_by_gap = {}
    total_by_gap = {}

    debug_count = 0
    with torch.no_grad():
        for doc in dataset.doc_sequences:
            input_ids = doc["input_ids"].unsqueeze(0).to(device)
            reset_memories(model)

            output = model(input_ids, labels=input_ids)

            gap = doc["gap"]
            # Check each recall position
            for i, (key_id, value_id) in enumerate(doc["pairs"]):
                rpos = doc["recall_positions"][i]
                # Logit at rpos-1 predicts token at rpos
                logits = output.logits[0, rpos - 1, :]
                value_token = dataset.value_offset + value_id
                predicted = logits.argmax().item()

                if predicted == value_token:
                    correct += 1
                total += 1

                correct_by_gap[gap] = correct_by_gap.get(gap, 0) + (1 if predicted == value_token else 0)
                total_by_gap[gap] = total_by_gap.get(gap, 0) + 1

                # Debug: print top predictions for first 5 docs
                if debug_count < 10:
                    top5 = logits.topk(5)
                    top5_tokens = top5.indices.tolist()
                    top5_vals = [f"{v:.2f}" for v in top5.values.tolist()]
                    is_value = "Y" if predicted == value_token else "N"
                    print(f"  [debug] pair {i}: key={key_id} expected_val={value_id} "
                          f"predicted={predicted} correct={is_value} "
                          f"top5={list(zip(top5_tokens, top5_vals))}")
                    debug_count += 1

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

    # Gate value analysis: measure gate strengths by chunk type
    gate_analysis = _analyze_gate_values(model, dataset, device)
    if gate_analysis:
        print(f"\n  Gate values by chunk type:")
        for ctype in ["store", "filler", "recall"]:
            g = gate_analysis[ctype]
            print(f"    {ctype.upper():>8s}: mean={g['mean']:.6f}, std={g['std']:.6f}")
        print(f"    Store/Filler ratio:  {gate_analysis['store']['mean'] / max(gate_analysis['filler']['mean'], 1e-10):.1f}x")
        print(f"    Recall/Filler ratio: {gate_analysis['recall']['mean'] / max(gate_analysis['filler']['mean'], 1e-10):.1f}x")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "accuracy_by_gap": accuracy_by_gap,
        "gate_analysis": gate_analysis,
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
                                 "surprise-fwd-store", "predictive",
                                 "future-only", "subtract-only"])
    parser.add_argument("--max-m-norm", type=float, default=0,
                        help="Cap on M norm (0 = no cap)")
    parser.add_argument("--bptt-steps", type=int, default=0,
                        help="BPTT steps for M (0 = always detach)")
    parser.add_argument("--w-out-std", type=float, default=0.0,
                        help="W_out init std (0 = zero init, >0 = random init for gradient bootstrap)")
    parser.add_argument("--tie-qk", action="store_true",
                        help="Tie W_query = W_key for guaranteed key-query matching")
    parser.add_argument("--delta-rule", action="store_true",
                        help="Use delta update rule: M += k⊗(v - Mk) to reduce interference")
    parser.add_argument("--gate-bias", type=float, default=-2.0,
                        help="Gate bias init (sigmoid(-2)=0.12, sigmoid(-5)=0.007)")
    parser.add_argument("--recall-loss-weight", type=float, default=1.0,
                        help="Extra weight on recall token loss (1.0 = uniform, 100 = 100x recall)")
    parser.add_argument("--recall-only-loss", action="store_true",
                        help="Compute loss ONLY on recall positions (zero weight on all other tokens)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Curriculum: train first half with 1 pair, second half with n-pairs")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--n-keys", type=int, default=32)
    parser.add_argument("--n-values", type=int, default=64)
    parser.add_argument("--n-pairs", type=int, default=1,
                        help="Number of key-value pairs per document (capacity test)")
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
        n_pairs=args.n_pairs, seed=42,
    )
    eval_data = SyntheticRecallDataset(
        n_docs=args.n_eval, gap_range=(args.gap_min, args.gap_max),
        chunk_size=args.chunk_size, n_keys=args.n_keys, n_values=args.n_values,
        n_pairs=args.n_pairs, seed=1337,
    )

    # Curriculum: also create 1-pair dataset for phase 1
    curriculum_train_data = None
    if args.curriculum and args.n_pairs > 1:
        print("  Creating curriculum phase 1 dataset (1 pair)...")
        curriculum_train_data = SyntheticRecallDataset(
            n_docs=args.n_train, gap_range=(args.gap_min, args.gap_max),
            chunk_size=args.chunk_size, n_keys=args.n_keys, n_values=args.n_values,
            n_pairs=1, seed=42,
        )

    # max_len must fit the longest document when using whole-doc mode
    if args.whole_doc:
        # Total chunks = 2*n_pairs (store + recall) + (gap_max - 1) fillers
        max_chunks = 2 * args.n_pairs + args.gap_max - 1
        max_len = max_chunks * args.chunk_size
        print(f"\n  Whole-doc mode: max_len = {max_len} "
              f"({max_chunks} chunks × {args.chunk_size} tokens, "
              f"{args.n_pairs} pairs)")
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
        extras = []
        if args.tie_qk: extras.append("tie_qk")
        if args.delta_rule: extras.append("delta")
        if args.gate_bias != -2.0: extras.append(f"gate_bias={args.gate_bias}")
        print(f"  Attaching memory (size={args.memory_size}, decay={args.decay}, "
              f"mode={args.write_mode}, bptt={args.bptt_steps}, w_out_std={args.w_out_std}"
              f"{', ' + ', '.join(extras) if extras else ''})")
        memory_modules = attach_fast_weight_memory(
            model.h,
            hidden_size=args.d_model,
            memory_size=args.memory_size,
            decay=args.decay,
            write_mode=args.write_mode,
            max_m_norm=args.max_m_norm,
            bptt_steps=args.bptt_steps,
            w_out_std=args.w_out_std,
            tie_qk=args.tie_qk,
            delta_rule=args.delta_rule,
            gate_bias=args.gate_bias,
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
        recall_loss_weight=args.recall_loss_weight,
        recall_only_loss=args.recall_only_loss,
        curriculum_train_data=curriculum_train_data,
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
