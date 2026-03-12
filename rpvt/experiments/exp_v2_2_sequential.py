"""Experiment v2.2: Fast weight memory on sequential data.

Exp v2.1 showed 34% of improvement comes from cross-sequence M on shuffled
wikitext — data where there's nothing useful to remember between chunks.

This experiment tests M on sequential data where cross-sequence memory
should actually help: long documents kept in reading order.

Setup:
    - Pretrained GPT-2 + frozen base + memory projections (same as v2.1)
    - Train on long documents (PG19 books), sequences kept in document order
    - M persists within a document, resets between documents
    - Compare cross-sequence contribution on sequential vs shuffled

Key question: does M contribute MORE when there's actually something
useful to remember from previous sequences?
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, AutoTokenizer

from rpvt.model.fast_weight import (
    attach_fast_weight_memory,
    TransformerLayerWithMemory,
)


class SequentialDocDataset(Dataset):
    """Dataset of sequential chunks from long documents.

    Each item is a dict with:
        input_ids: (seq_len,) token ids
        doc_id: int, which document this chunk belongs to
        is_first: bool, True if this is the first chunk of a document
    """

    def __init__(self, tokenizer, seq_len=512, max_docs=50, max_tokens_per_doc=100_000):
        from datasets import load_dataset

        print("  Loading PG19 (books)...")
        try:
            pg19 = load_dataset("pg19", split="train", streaming=True)
        except Exception:
            # Fallback to wikitext but keep articles together
            print("  PG19 unavailable, falling back to long wikitext articles...")
            self._load_wikitext_sequential(tokenizer, seq_len, max_docs, max_tokens_per_doc)
            return

        self.chunks = []
        self.doc_ids = []
        self.is_first = []
        doc_count = 0

        for item in pg19:
            text = item["text"] if isinstance(item["text"], str) else str(item["text"])
            if len(text.strip()) < 10000:  # skip short books
                continue

            # Tokenize this document
            tokens = tokenizer(text, return_attention_mask=False, truncation=False)["input_ids"]
            tokens = tokens[:max_tokens_per_doc]

            if len(tokens) < seq_len * 2:
                continue

            # Split into sequential chunks
            n_chunks = len(tokens) // seq_len
            for i in range(n_chunks):
                chunk = tokens[i * seq_len : (i + 1) * seq_len]
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))
                self.doc_ids.append(doc_count)
                self.is_first.append(i == 0)

            doc_count += 1
            if doc_count >= max_docs:
                break
            if doc_count % 10 == 0:
                print(f"    {doc_count} documents, {len(self.chunks)} chunks")

        print(f"  Total: {doc_count} documents, {len(self.chunks)} chunks")
        self.name = "sequential"

    def _load_wikitext_sequential(self, tokenizer, seq_len, max_docs, max_tokens_per_doc):
        """Fallback: use wikitext articles as sequential documents."""
        from datasets import load_dataset

        raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

        self.chunks = []
        self.doc_ids = []
        self.is_first = []

        # Group text by articles (separated by headers starting with ' = ')
        current_article = []
        doc_count = 0

        for text in raw["text"]:
            if text.startswith(" = ") and not text.startswith(" = = "):
                # New article header — process previous article
                if current_article and len("".join(current_article)) > 5000:
                    full_text = " ".join(current_article)
                    tokens = tokenizer(full_text, return_attention_mask=False, truncation=False)["input_ids"]
                    tokens = tokens[:max_tokens_per_doc]

                    n_chunks = len(tokens) // seq_len
                    for i in range(n_chunks):
                        chunk = tokens[i * seq_len : (i + 1) * seq_len]
                        self.chunks.append(torch.tensor(chunk, dtype=torch.long))
                        self.doc_ids.append(doc_count)
                        self.is_first.append(i == 0)

                    doc_count += 1
                    if doc_count >= max_docs:
                        break

                current_article = [text]
            else:
                current_article.append(text)

        print(f"  Total: {doc_count} documents, {len(self.chunks)} chunks")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {
            "input_ids": self.chunks[idx],
            "doc_id": self.doc_ids[idx],
            "is_first": self.is_first[idx],
        }


class ShuffledWrapper(Dataset):
    """Wraps a SequentialDocDataset but shuffles the order."""

    def __init__(self, sequential_dataset):
        self.dataset = sequential_dataset
        self.perm = torch.randperm(len(sequential_dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[self.perm[idx].item()]
        # Mark all as first (since order is random, every chunk is "first")
        item = dict(item)
        item["is_first"] = True
        return item


def sequential_collate_fn(batch):
    """Collate that preserves doc_id and is_first metadata."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "doc_ids": [b["doc_id"] for b in batch],
        "is_first": [b["is_first"] for b in batch],
    }


def create_model(device, dtype, memory_size=256, decay=0.99):
    """Load pretrained GPT-2 with frozen base + memory."""
    print("  Loading pretrained GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device=device, dtype=dtype)

    for p in model.parameters():
        p.requires_grad = False

    memory_modules = attach_fast_weight_memory(
        model.transformer.h,
        hidden_size=model.config.n_embd,
        memory_size=memory_size,
        decay=decay,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_memory = sum(p.numel() for m in memory_modules for p in m.parameters())
    print(f"  Trainable params: {n_trainable:,} (memory only)")

    return model, memory_modules


def reset_memories(model):
    for layer in model.transformer.h:
        if isinstance(layer, TransformerLayerWithMemory):
            layer.reset_memory()


def get_memory_norms(model):
    norms = []
    for layer in model.transformer.h:
        if isinstance(layer, TransformerLayerWithMemory):
            norms.append(layer.memory.M.norm().item())
    return sum(norms) / len(norms) if norms else 0


def get_wout_norms(model):
    norms = []
    for layer in model.transformer.h:
        if isinstance(layer, TransformerLayerWithMemory):
            norms.append(layer.memory.W_out.weight.norm().item())
    return sum(norms) / len(norms) if norms else 0


def get_write_mode_info(model):
    """Get write mode info for logging. Returns (mode, value) or (None, None)."""
    for layer in model.transformer.h:
        if isinstance(layer, TransformerLayerWithMemory):
            mem = layer.memory
            if mem.write_mode == "gate":
                values = []
                for l in model.transformer.h:
                    if isinstance(l, TransformerLayerWithMemory) and l.memory.write_mode == "gate":
                        values.append(torch.sigmoid(l.memory.W_gate.bias).item())
                return "gate", sum(values) / len(values)
            elif mem.write_mode in ("surprise", "surprise-fwd", "surprise-fwd-store"):
                return mem.write_mode, (mem.surprise_scale, mem.surprise_bias)
            else:
                return "uniform", None
    return None, None


def collect_aux_losses(model):
    """Collect auxiliary losses from all memory layers."""
    total = {}
    count = 0
    for layer in model.transformer.h:
        if isinstance(layer, TransformerLayerWithMemory) and layer._aux_losses:
            for key, val in layer._aux_losses.items():
                if key not in total:
                    total[key] = val
                else:
                    total[key] = total[key] + val
            count += 1
    if count > 0:
        total = {k: v / count for k, v in total.items()}
    return total


def train_sequential(
    model, dataset, device,
    num_epochs=3, lr=9e-4, warmup_steps=200,
    log_every=100, batch_size=1, model_name="model",
    aux_weight=0.0, contrastive_weight=0.0,
):
    """Train on sequential data with document-aware memory management.

    batch_size=1 is important: we process one document at a time in order,
    resetting M between documents. This lets M accumulate within a document.
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    # Use DataLoader with batch_size=1 to maintain document order
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=sequential_collate_fn)

    total_steps = len(loader) * num_epochs
    warmup = warmup_steps

    def lr_schedule(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"\nTraining {model_name}: {total_steps} steps, lr={lr}")
    model.train()
    global_step = 0
    train_losses = []
    log_data = []
    start_time = time.time()

    for epoch in range(num_epochs):
        prev_doc_id = None

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            doc_ids = batch["doc_ids"]
            is_first_flags = batch["is_first"]

            # Reset memory when starting a new document
            if any(is_first_flags) or (prev_doc_id is not None and doc_ids[0] != prev_doc_id):
                reset_memories(model)
            prev_doc_id = doc_ids[0]

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            # ── Collect and add auxiliary losses ──
            aux_losses = collect_aux_losses(model)
            aux_loss_total = 0.0
            if "aux_predict" in aux_losses and aux_weight > 0:
                loss = loss + aux_weight * aux_losses["aux_predict"]
                aux_loss_total += aux_losses["aux_predict"].item()
            if "contrastive" in aux_losses and contrastive_weight > 0:
                loss = loss + contrastive_weight * aux_losses["contrastive"]
                aux_loss_total += aux_losses["contrastive"].item()

            # ── NaN detection: enable debug mode and re-run to trace origin ──
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  [NaN DETECTED] step {global_step + 1}, loss={loss.item()}")
                print(f"    Enabling NaN debug mode and re-running this forward pass...")
                # Enable debug on all memory modules
                for layer in model.transformer.h:
                    if isinstance(layer, TransformerLayerWithMemory):
                        layer.memory._nan_debug = True
                # Re-run forward to get debug prints
                with torch.no_grad():
                    debug_out = model(input_ids, labels=input_ids)
                    print(f"    Debug forward loss: {debug_out.loss.item()}")
                # Check gradients from previous step
                nan_grad_count = sum(
                    1 for p in model.parameters()
                    if p.requires_grad and p.grad is not None and torch.isnan(p.grad).any()
                )
                print(f"    Params with NaN grads (from prev step): {nan_grad_count}")
                # Disable debug
                for layer in model.transformer.h:
                    if isinstance(layer, TransformerLayerWithMemory):
                        layer.memory._nan_debug = False

            optimizer.zero_grad()
            loss.backward()

            # ── NaN check on gradients ──
            nan_grad_params = [
                (name, p.grad.norm().item())
                for name, p in model.named_parameters()
                if p.requires_grad and p.grad is not None and torch.isnan(p.grad).any()
            ]
            if nan_grad_params:
                print(f"\n  [NaN GRADS] step {global_step + 1}: {len(nan_grad_params)} params")
                for name, norm in nan_grad_params[:5]:
                    print(f"    {name}: grad_norm={norm}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = sum(train_losses[-log_every:]) / log_every
                elapsed = time.time() - start_time
                tokens_per_sec = (global_step * input_ids.shape[0] * input_ids.shape[1]) / elapsed
                m_norm = get_memory_norms(model)
                w_norm = get_wout_norms(model)
                wm_mode, wm_val = get_write_mode_info(model)

                log_entry = {
                    "step": global_step,
                    "train_loss": avg_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_per_sec": tokens_per_sec,
                    "M_norm": m_norm,
                    "W_out_norm": w_norm,
                    "write_mode": wm_mode,
                    "write_mode_val": wm_val,
                }
                log_data.append(log_entry)

                if wm_mode == "gate":
                    extra = f", gate={wm_val:.4f}"
                elif wm_mode in ("surprise", "surprise-fwd", "surprise-fwd-store"):
                    extra = f", s={wm_val[0]:.1f}, b={wm_val[1]:.1f}"
                else:
                    extra = ""
                print(
                    f"  [{model_name}] step {global_step}/{total_steps}, "
                    f"loss={avg_loss:.4f}, lr={log_entry['lr']:.2e}, "
                    f"{tokens_per_sec:.0f} tok/s, W_out={w_norm:.2f}, M_norm={m_norm:.2f}"
                    f"{extra}"
                )

    return train_losses, log_data


def evaluate_sequential(model, dataset, device, reset_between_docs=True):
    """Evaluate on sequential data.

    If reset_between_docs=True: M resets at each new document (normal).
    If False: M carries across documents (tests cross-doc memory).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=sequential_collate_fn)
    total_loss, count = 0.0, 0
    prev_doc_id = None
    reset_memories(model)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            doc_ids = batch["doc_ids"]
            is_first_flags = batch["is_first"]

            if reset_between_docs and (any(is_first_flags) or
                    (prev_doc_id is not None and doc_ids[0] != prev_doc_id)):
                reset_memories(model)
            prev_doc_id = doc_ids[0]

            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item()
            count += 1

    return total_loss / max(count, 1)


def evaluate_by_position(model, dataset, device):
    """Evaluate loss at different positions within documents.

    If M is useful, later chunks in a document should have lower loss
    than earlier ones (M has accumulated useful context).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=sequential_collate_fn)

    # Track loss by chunk position within document
    position_losses = {}  # position -> list of losses
    chunk_in_doc = 0
    prev_doc_id = None
    reset_memories(model)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            doc_ids = batch["doc_ids"]
            is_first_flags = batch["is_first"]

            if any(is_first_flags) or (prev_doc_id is not None and doc_ids[0] != prev_doc_id):
                reset_memories(model)
                chunk_in_doc = 0
            prev_doc_id = doc_ids[0]

            outputs = model(input_ids, labels=input_ids)
            pos_key = min(chunk_in_doc, 20)  # bucket positions >= 20
            if pos_key not in position_losses:
                position_losses[pos_key] = []
            position_losses[pos_key].append(outputs.loss.item())
            chunk_in_doc += 1

    # Average by position
    return {pos: sum(losses) / len(losses) for pos, losses in sorted(position_losses.items())}


def evaluate_by_position_no_memory(model, dataset, device):
    """Same as evaluate_by_position but reset M before every forward pass."""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=sequential_collate_fn)

    position_losses = {}
    chunk_in_doc = 0
    prev_doc_id = None

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            doc_ids = batch["doc_ids"]
            is_first_flags = batch["is_first"]

            if any(is_first_flags) or (prev_doc_id is not None and doc_ids[0] != prev_doc_id):
                chunk_in_doc = 0
            prev_doc_id = doc_ids[0]

            # Reset M before every forward pass
            reset_memories(model)

            outputs = model(input_ids, labels=input_ids)
            pos_key = min(chunk_in_doc, 20)
            if pos_key not in position_losses:
                position_losses[pos_key] = []
            position_losses[pos_key].append(outputs.loss.item())
            chunk_in_doc += 1

    return {pos: sum(losses) / len(losses) for pos, losses in sorted(position_losses.items())}


def evaluate_memory_reset_ablation(model, dataset, device, reset_at_positions=(3, 5, 8)):
    """Test whether M stores real document-specific information.

    For each reset_at position K:
        - Run the document normally up to chunk K
        - Reset M at chunk K (erasing all accumulated context)
        - Measure loss at chunk K+1

    Compare against:
        - Normal M (no reset): loss at K+1 with full history
        - No M (always reset): loss at K+1 with no history

    If M stores real document info, resetting at K should hurt more as K
    increases (more information lost). If M is just an adapter, resetting
    won't matter because it rebuilds the same state from any single chunk.
    """
    model.eval()

    # We need to run the same documents multiple times with different reset points
    # First, collect documents (groups of sequential chunks)
    documents = {}  # doc_id -> list of (chunk_idx_in_doc, input_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=sequential_collate_fn)
    chunk_in_doc = 0
    prev_doc_id = None

    for batch in loader:
        doc_id = batch["doc_ids"][0]
        is_first = batch["is_first"][0]
        if is_first or (prev_doc_id is not None and doc_id != prev_doc_id):
            chunk_in_doc = 0
        if doc_id not in documents:
            documents[doc_id] = []
        documents[doc_id].append(batch["input_ids"])
        chunk_in_doc += 1
        prev_doc_id = doc_id

    # Only use documents long enough for the largest reset position
    max_reset = max(reset_at_positions)
    long_docs = {did: chunks for did, chunks in documents.items()
                 if len(chunks) > max_reset + 1}

    results = {}  # reset_pos -> {"normal": loss, "reset": loss, "no_memory": loss}

    for reset_pos in reset_at_positions:
        normal_losses = []
        reset_losses = []
        no_memory_losses = []

        with torch.no_grad():
            for doc_id, chunks in long_docs.items():
                # ── Run 1: Normal (M accumulates) ──
                reset_memories(model)
                for i in range(reset_pos + 2):  # run through chunk reset_pos+1
                    input_ids = chunks[i].to(device)
                    outputs = model(input_ids, labels=input_ids)
                    if i == reset_pos + 1:
                        normal_losses.append(outputs.loss.item())

                # ── Run 2: Reset M at position K, then measure K+1 ──
                reset_memories(model)
                for i in range(reset_pos + 2):
                    input_ids = chunks[i].to(device)
                    if i == reset_pos:
                        reset_memories(model)  # erase accumulated context
                    outputs = model(input_ids, labels=input_ids)
                    if i == reset_pos + 1:
                        reset_losses.append(outputs.loss.item())

                # ── Run 3: No memory (reset before every chunk) ──
                reset_memories(model)
                for i in range(reset_pos + 2):
                    input_ids = chunks[i].to(device)
                    reset_memories(model)
                    outputs = model(input_ids, labels=input_ids)
                    if i == reset_pos + 1:
                        no_memory_losses.append(outputs.loss.item())

        results[reset_pos] = {
            "normal": sum(normal_losses) / len(normal_losses),
            "reset": sum(reset_losses) / len(reset_losses),
            "no_memory": sum(no_memory_losses) / len(no_memory_losses),
            "reset_cost": (sum(reset_losses) - sum(normal_losses)) / len(normal_losses),
            "n_docs": len(long_docs),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="v2.2: fast weight memory on sequential data")
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v2_2")
    parser.add_argument("--checkpoint", default=None,
                        help="Load from exp v2.1 checkpoint instead of training fresh")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load sequential data
    print("Loading sequential document data...")
    seq_dataset = SequentialDocDataset(
        tokenizer, seq_len=args.seq_len, max_docs=args.max_docs,
    )

    # Build model
    print("\nBuilding model...")
    model, memory_modules = create_model(device, dtype, args.memory_size, args.decay)

    results = {"config": vars(args)}

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
    else:
        # Train on sequential data
        print(f"\n{'='*60}")
        print("TRAINING on sequential documents")
        print(f"{'='*60}")

        train_losses, log_data = train_sequential(
            model, seq_dataset, device,
            num_epochs=args.epochs, lr=args.lr,
            warmup_steps=200, log_every=args.log_every,
            model_name="sequential",
        )
        results["training"] = {"log": log_data}
        torch.save(model.state_dict(), Path(args.output_dir) / "model.pt")

    # ── Evaluation: loss by position in document ──
    print(f"\n{'='*60}")
    print("EVAL: Loss by position in document (with M accumulating)")
    print(f"{'='*60}")

    pos_losses_with_M = evaluate_by_position(model, seq_dataset, device)
    print(f"  {'Position':>8} | {'Loss':>8}")
    print(f"  {'-'*8}-+-{'-'*8}")
    for pos, loss in pos_losses_with_M.items():
        label = f"chunk {pos}" if pos < 20 else "chunk 20+"
        print(f"  {label:>8} | {loss:.4f}")
    results["position_losses_with_M"] = pos_losses_with_M

    print(f"\n{'='*60}")
    print("EVAL: Loss by position (WITHOUT M — reset each forward pass)")
    print(f"{'='*60}")

    pos_losses_no_M = evaluate_by_position_no_memory(model, seq_dataset, device)
    print(f"  {'Position':>8} | {'Loss':>8}")
    print(f"  {'-'*8}-+-{'-'*8}")
    for pos, loss in pos_losses_no_M.items():
        label = f"chunk {pos}" if pos < 20 else "chunk 20+"
        print(f"  {label:>8} | {loss:.4f}")
    results["position_losses_no_M"] = pos_losses_no_M

    # ── Position comparison ──
    print(f"\n{'='*60}")
    print("M BENEFIT BY POSITION (positive = M helps)")
    print(f"{'='*60}")
    print(f"  {'Position':>8} | {'With M':>8} | {'No M':>8} | {'Benefit':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for pos in sorted(set(pos_losses_with_M) & set(pos_losses_no_M)):
        with_m = pos_losses_with_M[pos]
        no_m = pos_losses_no_M[pos]
        benefit = no_m - with_m
        label = f"chunk {pos}" if pos < 20 else "chunk 20+"
        print(f"  {label:>8} | {with_m:.4f} | {no_m:.4f} | {benefit:+.4f}")
    results["position_comparison"] = {
        pos: {
            "with_M": pos_losses_with_M.get(pos, 0),
            "no_M": pos_losses_no_M.get(pos, 0),
            "benefit": pos_losses_no_M.get(pos, 0) - pos_losses_with_M.get(pos, 0),
        }
        for pos in sorted(set(pos_losses_with_M) & set(pos_losses_no_M))
    }

    # ── Overall ablation ──
    print(f"\n{'='*60}")
    print("OVERALL ABLATION")
    print(f"{'='*60}")

    loss_with = evaluate_sequential(model, seq_dataset, device, reset_between_docs=True)
    reset_memories(model)

    # No memory output
    saved_wout = {}
    for i, layer in enumerate(model.transformer.h):
        if isinstance(layer, TransformerLayerWithMemory):
            saved_wout[i] = layer.memory.W_out.weight.data.clone()
            layer.memory.W_out.weight.data.zero_()
    loss_no_memory = evaluate_sequential(model, seq_dataset, device)
    for i, w in saved_wout.items():
        model.transformer.h[i].memory.W_out.weight.data = w

    print(f"  With memory:    {loss_with:.4f}")
    print(f"  Without memory: {loss_no_memory:.4f}")
    print(f"  Total benefit:  {loss_no_memory - loss_with:+.4f}")
    results["overall_ablation"] = {
        "with_memory": loss_with,
        "without_memory": loss_no_memory,
        "benefit": loss_no_memory - loss_with,
    }

    # Save
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
