"""Analyze gate values per chunk type (store, filler, recall).

Trains a model with hard gate briefly, then logs per-chunk-type gate values.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rpvt.experiments.exp_v2_6_synthetic_recall import (
    SyntheticRecallDataset,
    SmallTransformerLM,
    reset_memories,
)
from rpvt.model.fast_weight import (
    attach_fast_weight_memory,
    FastWeightMemory,
    TransformerLayerWithMemory,
)


def analyze_gate_values(model, dataset, device, n_docs=50):
    """Run docs through model and collect gate values per chunk type."""
    model.eval()

    # Find memory modules with gates
    memory_modules = []
    for module in model.modules():
        if isinstance(module, FastWeightMemory) and module.write_mode == "gate":
            memory_modules.append(module)

    if not memory_modules:
        print("No gate memory modules found!")
        return

    print(f"Found {len(memory_modules)} gate memory modules across {len([l for l in model.h if isinstance(l, TransformerLayerWithMemory)])} layers")

    chunk_size = dataset.chunk_size

    # We'll collect per-token gate values, labeled by chunk type
    gate_by_type = {"store": [], "filler": [], "recall": []}
    # Also per-token within chunks
    gate_tokens_store = []
    gate_tokens_filler = []
    gate_tokens_recall = []

    docs = dataset.doc_sequences
    for doc_idx in range(min(n_docs, len(docs))):
        doc = docs[doc_idx]
        input_ids = doc["input_ids"].unsqueeze(0).to(device)
        seq_len = input_ids.shape[1]
        n_chunks = seq_len // chunk_size
        gap = doc["gap"]
        n_pairs = dataset.n_pairs

        reset_memories(model)

        # Patch memory modules to capture gate values
        for mem in memory_modules:
            mem._captured_gates = []

        original_forwards = {}
        for i, mem in enumerate(memory_modules):
            original_forward = mem.__class__.forward

            def make_patched(mem_ref, orig_fwd):
                def patched(self, x, chunk_size=64):
                    gate_strengths = torch.sigmoid(self.W_gate(x))
                    mem_ref._captured_gates.append(gate_strengths.detach().cpu().squeeze(-1))  # (1, seq_len)
                    return orig_fwd(self, x, chunk_size)
                return patched

            # Use method patching on instance
            import types
            mem._orig_forward = mem.forward
            mem.forward = types.MethodType(make_patched(mem, original_forward), mem)

        with torch.no_grad():
            model(input_ids)

        # Restore
        for mem in memory_modules:
            mem.forward = mem._orig_forward
            del mem._orig_forward

        # Use first layer's gate values
        if memory_modules[0]._captured_gates:
            gates = memory_modules[0]._captured_gates[0].squeeze(0)  # (seq_len,)

            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = start + chunk_size
                chunk_gate_vals = gates[start:end]
                chunk_mean = chunk_gate_vals.mean().item()

                if chunk_idx < n_pairs:
                    gate_by_type["store"].append(chunk_mean)
                    gate_tokens_store.append(chunk_gate_vals)
                elif chunk_idx < n_pairs + (gap - 1):
                    gate_by_type["filler"].append(chunk_mean)
                    gate_tokens_filler.append(chunk_gate_vals)
                else:
                    gate_by_type["recall"].append(chunk_mean)
                    gate_tokens_recall.append(chunk_gate_vals)

    # Report
    print("\n" + "=" * 60)
    print("GATE VALUES BY CHUNK TYPE (mean over all chunks)")
    print("=" * 60)
    for chunk_type in ["store", "filler", "recall"]:
        vals = gate_by_type[chunk_type]
        if vals:
            t = torch.tensor(vals)
            print(f"  {chunk_type.upper():>8s}: mean={t.mean():.6f}, std={t.std():.6f}, "
                  f"min={t.min():.6f}, max={t.max():.6f}, n={len(vals)}")

    store_mean = torch.tensor(gate_by_type["store"]).mean()
    filler_mean = torch.tensor(gate_by_type["filler"]).mean()
    recall_mean = torch.tensor(gate_by_type["recall"]).mean()
    print(f"\n  Store/Filler ratio:  {store_mean / filler_mean:.1f}x")
    print(f"  Recall/Filler ratio: {recall_mean / filler_mean:.1f}x")

    # Per-token position analysis within chunks
    print("\n" + "=" * 60)
    print("GATE VALUES BY TOKEN POSITION WITHIN CHUNK")
    print("=" * 60)
    for name, token_list in [("store", gate_tokens_store), ("filler", gate_tokens_filler), ("recall", gate_tokens_recall)]:
        if token_list:
            stacked = torch.stack(token_list)  # (n_chunks, chunk_size)
            pos_means = stacked.mean(dim=0)    # (chunk_size,)
            # Show first 5 and last 5 positions
            print(f"\n  {name.upper()} (avg over {len(token_list)} chunks):")
            positions = list(range(min(5, len(pos_means)))) + list(range(max(5, len(pos_means)-5), len(pos_means)))
            for p in positions:
                marker = ""
                if name == "store" and p in (0, 1, 2):
                    marker = " <- STORE/KEY/VALUE tokens"
                if name == "recall" and p == len(pos_means) - 1:
                    marker = " <- VALUE (target)"
                if name == "recall" and p == len(pos_means) - 2:
                    marker = " <- KEY (query)"
                print(f"    pos {p:3d}: {pos_means[p]:.6f}{marker}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-bias", type=float, default=-5.0)
    parser.add_argument("--n-pairs", type=int, default=4)
    parser.add_argument("--n-keys", type=int, default=64)
    parser.add_argument("--n-values", type=int, default=128)
    parser.add_argument("--gap-min", type=int, default=5)
    parser.add_argument("--gap-max", type=int, default=10)
    parser.add_argument("--decay", type=float, default=0.999)
    parser.add_argument("--recall-loss-weight", type=float, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-docs", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chunk_size = 64
    max_chunks = 2 * args.n_pairs + args.gap_max - 1
    max_len = (max_chunks + 1) * chunk_size

    vocab_size = 1 + 1000 + args.n_keys + args.n_values + 3

    print("Building datasets...")
    train_data = SyntheticRecallDataset(
        n_docs=2000, chunk_size=chunk_size, n_keys=args.n_keys, n_values=args.n_values,
        gap_range=(args.gap_min, args.gap_max),
        filler_vocab_size=1000, n_pairs=args.n_pairs, seed=42,
    )
    eval_data = SyntheticRecallDataset(
        n_docs=500, chunk_size=chunk_size, n_keys=args.n_keys, n_values=args.n_values,
        gap_range=(args.gap_min, args.gap_max),
        filler_vocab_size=1000, n_pairs=args.n_pairs, seed=99,
    )

    print("Building model...")
    model = SmallTransformerLM(vocab_size, d_model=256, n_heads=4, n_layers=4, max_len=max_len)
    attach_fast_weight_memory(
        model.h, hidden_size=256, memory_size=128, decay=args.decay,
        write_mode="gate", max_m_norm=0, gate_bias=args.gate_bias,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Use whole-doc sequences for training (same as main experiment)
    train_docs = train_data.doc_sequences
    eval_docs = eval_data.doc_sequences

    print(f"Training for {args.epochs} epochs on {len(train_docs)} docs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for i, doc in enumerate(train_docs):
            input_ids = doc["input_ids"].unsqueeze(0).to(device)

            reset_memories(model)

            output = model(input_ids)
            logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
            targets = input_ids[:, 1:].reshape(-1)

            per_token_loss = F.cross_entropy(logits, targets, reduction='none')
            weights = torch.ones_like(per_token_loss)
            for rpos in doc["recall_positions"]:
                weights[rpos - 1] = args.recall_loss_weight
            loss = (per_token_loss * weights).sum() / weights.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (i + 1) % 500 == 0:
                print(f"  epoch {epoch+1}, step {i+1}/{len(train_docs)}, loss={epoch_loss/n_batches:.4f}")

        # Quick eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for doc in eval_docs[:200]:
                input_ids = doc["input_ids"].unsqueeze(0).to(device)
                reset_memories(model)
                output = model(input_ids)
                for rpos in doc["recall_positions"]:
                    pred = output.logits[0, rpos - 1].argmax().item()
                    target = input_ids[0, rpos].item()
                    if pred == target:
                        correct += 1
                    total += 1
        print(f"Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}, recall={correct}/{total} ({100*correct/total:.1f}%)")

    # Analyze
    print("\n\nAnalyzing gate values on eval data...")
    analyze_gate_values(model, eval_data, device, n_docs=args.n_docs)


if __name__ == "__main__":
    main()
