"""Experiment v3.1: LoRA + Hopfield Memory on a pretrained model.

Stage 1 of the pretrained-model memory integration roadmap.
Tests whether Hopfield memory works as a plug-in module on a frozen
pretrained transformer (Qwen3-0.6B), trained with LoRA adapters.

Key differences from v2.6 (from-scratch):
  - Pretrained model (frozen) instead of SmallTransformerLM
  - LoRA adapters for fine-tuning (peft)
  - Memory attached to a single target layer (not all layers)
  - Synthetic token IDs mapped into pretrained vocab space
  - Two-phase: Phase 1 = train memory only, Phase 2 = unfreeze LoRA + memory

The hypothesis: with a pretrained model, the task of recalling stored values
is genuinely impossible without memory (values are outside the attention window
between chunks). This creates natural gradient pressure toward using M,
unlike from-scratch training where the model learns to ignore M.
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
from torch.utils.data import Dataset

from rpvt.model.hopfield_memory import HopfieldMemory


class PretrainedRecallDataset(Dataset):
    """Synthetic recall task mapped into a pretrained model's vocab space.

    Same structure as SyntheticRecallDataset but token IDs are drawn from
    the pretrained model's vocabulary. We pick arbitrary token IDs for
    STORE, RECALL, PAD, keys, values, and filler — the pretrained embeddings
    give these tokens pre-existing representations that memory can work with.
    """

    def __init__(
        self,
        tokenizer,
        n_docs=1000,
        gap_range=(2, 10),
        chunk_size=64,
        n_keys=64,
        n_values=128,
        n_pairs=1,
        filler_vocab_size=256,
        seed=42,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_keys = n_keys
        self.n_values = n_values
        self.n_pairs = n_pairs

        vocab_size = tokenizer.vocab_size

        # Map synthetic tokens to pretrained vocab IDs.
        # Avoid special tokens (first 100) and pick from mid-range.
        # We need: 3 special + n_keys + n_values + filler_vocab_size tokens
        total_needed = 3 + n_keys + n_values + filler_vocab_size
        assert total_needed < vocab_size - 1000, \
            f"Need {total_needed} tokens but vocab only has {vocab_size}"

        # Use a deterministic mapping starting from token 1000
        base = 1000
        self.STORE = base
        self.RECALL = base + 1
        self.PAD = base + 2
        self.key_offset = base + 3
        self.value_offset = self.key_offset + n_keys
        self.filler_offset = self.value_offset + n_values
        self.filler_vocab_size = filler_vocab_size
        self.pretrained_vocab_size = vocab_size

        # Verify all mapped IDs are valid
        max_id = self.filler_offset + filler_vocab_size
        assert max_id < vocab_size, \
            f"Mapped token IDs go up to {max_id} but vocab size is {vocab_size}"

        rng = torch.Generator().manual_seed(seed)

        self.documents = []
        for _ in range(n_docs):
            gap = torch.randint(gap_range[0], gap_range[1] + 1, (1,), generator=rng).item()

            pairs = []
            used_keys = set()
            for _ in range(n_pairs):
                key_id = torch.randint(0, n_keys, (1,), generator=rng).item()
                while key_id in used_keys and len(used_keys) < n_keys:
                    key_id = torch.randint(0, n_keys, (1,), generator=rng).item()
                used_keys.add(key_id)
                value_id = torch.randint(0, n_values, (1,), generator=rng).item()
                pairs.append((key_id, value_id))

            chunks = []

            # Store chunks
            for key_id, value_id in pairs:
                store_chunk = torch.full((chunk_size,), self.PAD, dtype=torch.long)
                store_chunk[0] = self.STORE
                store_chunk[1] = self.key_offset + key_id
                store_chunk[2] = self.value_offset + value_id
                chunks.append(store_chunk)

            # Filler chunks
            for _ in range(gap - 1):
                filler = torch.randint(0, filler_vocab_size, (chunk_size,), generator=rng)
                filler = filler + self.filler_offset  # shift into pretrained vocab range
                chunks.append(filler)

            # Recall chunks
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
                "key_id": pairs[0][0],
                "value_id": pairs[0][1],
            })

        # Build whole-document sequences
        self.doc_sequences = []
        for doc in self.documents:
            seq = torch.cat(doc["chunks"])
            recall_positions = []
            for i in range(n_pairs):
                recall_chunk_idx = n_pairs + (doc["gap"] - 1) + i
                pos = (recall_chunk_idx + 1) * chunk_size - 1
                recall_positions.append(pos)
            self.doc_sequences.append({
                "input_ids": seq,
                "pairs": doc["pairs"],
                "value_id": doc["value_id"],
                "gap": doc["gap"],
                "recall_positions": recall_positions,
            })

        print(f"  PretrainedRecall: {n_docs} docs, gap range {gap_range}, "
              f"{n_keys} keys, {n_values} values, {n_pairs} pairs/doc, "
              f"mapped to pretrained vocab [{base}..{max_id}]")


class MemoryWrapper(nn.Module):
    """Wraps a single transformer layer with Hopfield memory."""

    def __init__(self, original_layer, memory_module):
        super().__init__()
        self.layer = original_layer
        self.memory = memory_module

    def __getattr__(self, name):
        """Proxy attributes from the original layer (e.g. attention_type)."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        outputs = self.layer(*args, **kwargs)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        memory_output, gate_value, aux_losses = self.memory(hidden_states)
        modified = hidden_states + memory_output
        if isinstance(outputs, tuple):
            return (modified,) + outputs[1:]
        return modified

    def reset_memory(self):
        self.memory.reset_memory()


def reset_memories(model):
    """Reset memory in all wrapped layers."""
    for module in model.modules():
        if isinstance(module, MemoryWrapper):
            module.reset_memory()


def set_persistent_grad(model, enabled):
    """Enable/disable gradient persistence in memory modules."""
    for module in model.modules():
        if isinstance(module, HopfieldMemory):
            module.persistent_grad = enabled


def detach_memory_state(model):
    """Detach memory state after backward pass to free computation graph."""
    for module in model.modules():
        if isinstance(module, HopfieldMemory):
            module.K_mem = module.K_mem.detach()
            module.V_mem = module.V_mem.detach()
            module.mem_strength = module.mem_strength.detach()


def make_chunk_local_mask(seq_len, chunk_size, device):
    """Create a causal attention mask that only allows within-chunk attention.

    Returns a 4D mask (1, 1, seq_len, seq_len) compatible with transformers.
    Tokens can only attend to tokens in the same chunk (and earlier positions
    within that chunk for causality). This makes the transformer chunk-local,
    so memory is the only cross-chunk information channel.
    """
    # Assign each position to its chunk
    positions = torch.arange(seq_len, device=device)
    chunk_ids = positions // chunk_size  # (seq_len,)

    # Same-chunk mask: (seq_len, seq_len)
    same_chunk = chunk_ids.unsqueeze(0) == chunk_ids.unsqueeze(1)

    # Causal mask: can only attend to earlier or same position
    causal = positions.unsqueeze(0) >= positions.unsqueeze(1)

    # Combined: same chunk AND causal
    mask = same_chunk & causal  # True = attend

    # transformers expects: 0 = attend, large negative = don't attend
    float_mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=torch.bfloat16)
    float_mask[:, :, ~mask] = torch.finfo(torch.bfloat16).min
    return float_mask


def build_model(model_name, device, memory_layer, memory_size, n_slots,
                decay, gate_bias, lora_rank, lora_targets, no_memory=False,
                no_lora=False, init_qk_shared=False, n_extract=1):
    """Load pretrained model, attach LoRA and Hopfield memory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    print(f"\nLoading pretrained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  Hidden size: {hidden_size}, Layers: {n_layers}, "
          f"Vocab: {tokenizer.vocab_size}")

    # Apply LoRA
    if not no_lora:
        targets = [t.strip() for t in lora_targets.split(",")]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=targets,
        )
        model = get_peft_model(model, lora_config)
        n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  LoRA: rank={lora_rank}, targets={targets}, params={n_lora:,}")
    else:
        print("  No LoRA (memory only)")

    # Attach Hopfield memory to target layer
    if not no_memory:
        if memory_layer < 0:
            memory_layer = n_layers // 2  # default: middle layer
        print(f"  Attaching Hopfield memory to layer {memory_layer}/{n_layers} "
              f"(size={memory_size}, slots={n_slots}, decay={decay}, bias={gate_bias})")

        # Get the layers list — handle peft wrapping
        if hasattr(model, 'base_model'):
            layers = model.base_model.model.model.layers
        else:
            layers = model.model.layers

        mem = HopfieldMemory(
            hidden_size=hidden_size,
            memory_size=memory_size,
            n_slots=n_slots,
            decay=decay,
            write_mode="gate",
            gate_bias=gate_bias,
            w_out_std=0.0,
            init_qk_shared=init_qk_shared,
            n_extract=n_extract,
        ).to(device=device, dtype=torch.bfloat16)

        original_layer = layers[memory_layer]
        wrapped = MemoryWrapper(original_layer, mem)
        layers[memory_layer] = wrapped

        n_mem = sum(p.numel() for p in mem.parameters())
        print(f"  Memory params: {n_mem:,}")
    else:
        print("  No memory (baseline)")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Total: {n_total:,}, Trainable: {n_trainable:,} "
          f"({100 * n_trainable / n_total:.2f}%)")

    return model, tokenizer


def get_memory_params(model):
    """Get parameter names belonging to memory modules (NOT the wrapped layer)."""
    memory_param_names = set()
    for name, module in model.named_modules():
        if isinstance(module, HopfieldMemory):
            for pname, _ in module.named_parameters():
                full_name = f"{name}.{pname}"
                memory_param_names.add(full_name)
    return memory_param_names


def get_lora_params(model):
    """Get parameter names belonging to LoRA adapters."""
    lora_param_names = set()
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_param_names.add(name)
    return lora_param_names


def train_and_eval(
    model, train_dataset, eval_dataset, device,
    num_epochs=20, lr_memory=1e-3, lr_lora=2e-4,
    log_every=50, recall_loss_weight=100.0,
    two_phase=False, output_dir="results",
):
    """Train with optional two-phase schedule.

    Phase 1: train only memory (LoRA frozen).
    Phase 2: train both memory + LoRA.
    """
    memory_params = get_memory_params(model)
    lora_params = get_lora_params(model)

    if two_phase:
        # Phase 1: freeze LoRA, train only memory
        for name, param in model.named_parameters():
            if name in memory_params:
                param.requires_grad = True
            else:
                param.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Two-phase: Phase 1 — {n_train:,} memory params trainable")

    # Build optimizer with separate param groups (use param id to avoid duplicates)
    param_groups = []
    seen_ids = set()

    mem_p = []
    for n, p in model.named_parameters():
        if p.requires_grad and n in memory_params and id(p) not in seen_ids:
            mem_p.append(p)
            seen_ids.add(id(p))

    lora_p = []
    for n, p in model.named_parameters():
        if p.requires_grad and n in lora_params and id(p) not in seen_ids:
            lora_p.append(p)
            seen_ids.add(id(p))

    other_p = []
    for n, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen_ids:
            other_p.append(p)
            seen_ids.add(id(p))

    if mem_p:
        param_groups.append({"params": mem_p, "lr": lr_memory})
    if lora_p:
        param_groups.append({"params": lora_p, "lr": lr_lora})
    if other_p:
        param_groups.append({"params": other_p, "lr": lr_lora})

    if not param_groups:
        print("  WARNING: No trainable parameters!")
        return evaluate_recall(model, eval_dataset, device)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    total_steps = len(train_dataset.doc_sequences) * num_epochs

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    phase_1_end = num_epochs // 2 if two_phase else 0
    phase_1_done = False

    print(f"\nTraining: {total_steps} steps, {num_epochs} epochs, "
          f"recall_weight={recall_loss_weight}x")
    model.train()
    global_step = 0
    train_losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        # Phase transition
        if two_phase and epoch == phase_1_end and not phase_1_done:
            print(f"\n  === PHASE 2: Unfreezing LoRA ===")
            for name, param in model.named_parameters():
                if name in memory_params or name in lora_params:
                    param.requires_grad = True

            # Rebuild optimizer with both param groups
            seen_ids_p2 = set()
            mem_p = []
            for n, p in model.named_parameters():
                if p.requires_grad and n in memory_params and id(p) not in seen_ids_p2:
                    mem_p.append(p)
                    seen_ids_p2.add(id(p))
            lora_p = []
            for n, p in model.named_parameters():
                if p.requires_grad and n in lora_params and id(p) not in seen_ids_p2:
                    lora_p.append(p)
                    seen_ids_p2.add(id(p))
            param_groups = []
            if mem_p:
                param_groups.append({"params": mem_p, "lr": lr_memory * 0.1})
            if lora_p:
                param_groups.append({"params": lora_p, "lr": lr_lora})

            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
            remaining_steps = total_steps - global_step
            def lr_schedule_p2(step, _rem=remaining_steps):
                warmup = 50
                if step < warmup:
                    return step / warmup
                progress = (step - warmup) / max(_rem - warmup, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_p2)

            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  {n_train:,} params trainable (memory + LoRA)")
            phase_1_done = True

        for doc in train_dataset.documents:
            chunks = doc["chunks"]
            n_chunks = len(chunks)
            chunk_size = train_dataset.chunk_size
            n_pairs = train_dataset.n_pairs
            gap = doc["gap"]

            # Reset memory and enable gradient persistence
            reset_memories(model)
            set_persistent_grad(model, True)

            # Process each chunk independently — memory is the only cross-chunk channel
            doc_loss = torch.tensor(0.0, device=device)
            n_tokens = 0
            recall_loss_total = torch.tensor(0.0, device=device)
            n_recall = 0

            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)  # (1, chunk_size)
                output = model(chunk_ids, labels=chunk_ids)

                # Is this a recall chunk?
                recall_pair_idx = chunk_idx - (n_chunks - n_pairs)
                is_recall = recall_pair_idx >= 0

                if is_recall and recall_loss_weight > 1.0:
                    # Weighted loss: high weight on recall position (last token)
                    logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
                    targets = chunk_ids[:, 1:].reshape(-1)
                    per_token_loss = F.cross_entropy(logits, targets, reduction='none')
                    weights = torch.ones_like(per_token_loss)
                    weights[-1] = recall_loss_weight  # last position predicts value
                    chunk_loss = (per_token_loss * weights).sum() / weights.sum()
                    recall_loss_total = recall_loss_total + per_token_loss[-1]
                    n_recall += 1
                else:
                    chunk_loss = output.loss

                doc_loss = doc_loss + chunk_loss
                n_tokens += chunk_size

            # Average loss over chunks
            doc_loss = doc_loss / n_chunks

            optimizer.zero_grad()
            doc_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            # Detach memory state after backward to free graph
            detach_memory_state(model)
            set_persistent_grad(model, False)

            train_losses.append(doc_loss.item())
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = sum(train_losses[-log_every:]) / log_every
                elapsed = time.time() - start_time
                recall_info = ""
                if n_recall > 0:
                    recall_info = f", recall_loss={recall_loss_total.item()/n_recall:.2f}"
                print(f"  step {global_step}/{total_steps}, "
                      f"loss={avg_loss:.4f}{recall_info}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, {elapsed:.0f}s")

    return evaluate_recall(model, eval_dataset, device)


def evaluate_recall(model, dataset, device):
    """Evaluate recall accuracy using per-chunk processing."""
    model.eval()

    correct = 0
    total = 0
    correct_by_gap = {}
    total_by_gap = {}
    debug_count = 0

    with torch.no_grad():
        for doc_idx, doc_info in enumerate(dataset.documents):
            chunks = doc_info["chunks"]
            n_chunks = len(chunks)
            n_pairs = dataset.n_pairs
            gap = doc_info["gap"]

            reset_memories(model)

            # Process each chunk independently
            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                output = model(chunk_ids)

                # Check recall chunks
                recall_pair_idx = chunk_idx - (n_chunks - n_pairs)
                if recall_pair_idx >= 0:
                    key_id, value_id = doc_info["pairs"][recall_pair_idx]
                    # Prediction at position -2 for token at position -1
                    logits = output.logits[0, -2, :]
                    value_token = dataset.value_offset + value_id
                    predicted = logits.argmax().item()

                    if predicted == value_token:
                        correct += 1
                    total += 1

                    correct_by_gap[gap] = correct_by_gap.get(gap, 0) + (1 if predicted == value_token else 0)
                    total_by_gap[gap] = total_by_gap.get(gap, 0) + 1

                    if debug_count < 10:
                        top5 = logits.topk(5)
                        top5_tokens = top5.indices.tolist()
                        top5_vals = [f"{v:.2f}" for v in top5.values.tolist()]
                        is_value = "Y" if predicted == value_token else "N"
                        print(f"  [debug] pair {recall_pair_idx}: key={key_id} expected_val={value_id} "
                              f"(token {value_token}) predicted={predicted} correct={is_value} "
                              f"top5={list(zip(top5_tokens, top5_vals))}")
                        debug_count += 1

    accuracy = correct / max(total, 1)
    print(f"\n  Recall accuracy: {correct}/{total} = {accuracy:.1%}")

    if total_by_gap:
        print(f"\n  {'Gap':>5} | {'Correct':>8} | {'Total':>6} | {'Accuracy':>8}")
        print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")
        for gap in sorted(total_by_gap):
            acc = correct_by_gap.get(gap, 0) / total_by_gap[gap]
            print(f"  {gap:>5} | {correct_by_gap.get(gap, 0):>8} | {total_by_gap[gap]:>6} | {acc:>7.1%}")

    # Gate analysis
    gate_analysis = _analyze_gate_values(model, dataset, device)
    if gate_analysis:
        print(f"\n  Gate values by chunk type:")
        for ctype in ["store", "filler", "recall"]:
            g = gate_analysis[ctype]
            print(f"    {ctype.upper():>8s}: mean={g['mean']:.6f}, std={g['std']:.6f}")
        ratio = gate_analysis['store']['mean'] / max(gate_analysis['filler']['mean'], 1e-10)
        print(f"    Store/Filler ratio: {ratio:.1f}x")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "accuracy_by_gap": {
            gap: correct_by_gap.get(gap, 0) / total_by_gap[gap]
            for gap in sorted(total_by_gap)
        },
        "gate_analysis": gate_analysis,
    }


def _analyze_gate_values(model, dataset, device, n_docs=50):
    """Analyze gate values by chunk type using per-chunk processing."""
    memory_modules = []
    for module in model.modules():
        if isinstance(module, HopfieldMemory) and module.write_mode == "gate":
            memory_modules.append(module)

    if not memory_modules:
        return None

    gate_by_type = {"store": [], "filler": [], "recall": []}
    mem = memory_modules[0]

    for doc_info in dataset.documents[:n_docs]:
        chunks = doc_info["chunks"]
        n_chunks = len(chunks)
        gap = doc_info["gap"]
        n_pairs = dataset.n_pairs

        reset_memories(model)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_ids = chunk.unsqueeze(0).to(device)

            # Capture gate values
            mem._captured_gates = None
            orig_forward = mem.forward

            def make_capture_forward(m, orig):
                def capture_forward(x, chunk_size=64):
                    param_dtype = m.W_gate.weight.dtype
                    x_cast = x.to(dtype=param_dtype)
                    m._captured_gates = torch.sigmoid(m.W_gate(x_cast)).detach().cpu().squeeze(-1).squeeze(0)
                    return orig(x, chunk_size)
                return capture_forward

            mem.forward = make_capture_forward(mem, orig_forward)

            with torch.no_grad():
                model(chunk_ids)

            mem.forward = orig_forward

            if mem._captured_gates is not None:
                chunk_mean = mem._captured_gates.mean().item()
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


def main():
    parser = argparse.ArgumentParser(description="v3.1: pretrained model + LoRA + Hopfield memory")

    # Model
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B",
                        help="Pretrained model name/path")
    parser.add_argument("--memory-layer", type=int, default=-1,
                        help="Layer to attach memory (-1 = middle layer)")
    parser.add_argument("--memory-size", type=int, default=256,
                        help="Memory projection dimension")
    parser.add_argument("--n-slots", type=int, default=64,
                        help="Number of Hopfield memory slots")
    parser.add_argument("--decay", type=float, default=0.999)
    parser.add_argument("--gate-bias", type=float, default=-2.0)
    parser.add_argument("--init-qk-shared", action="store_true",
                        help="Initialize W_query = W_key for query-key alignment")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-targets", type=str, default="q_proj,v_proj",
                        help="Comma-separated LoRA target modules")

    # Ablations
    parser.add_argument("--no-memory", action="store_true",
                        help="No memory (LoRA-only baseline)")
    parser.add_argument("--no-lora", action="store_true",
                        help="No LoRA (memory-only baseline)")

    # Task
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--n-keys", type=int, default=64)
    parser.add_argument("--n-values", type=int, default=128)
    parser.add_argument("--n-pairs", type=int, default=1)
    parser.add_argument("--gap-min", type=int, default=5)
    parser.add_argument("--gap-max", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=500)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr-memory", type=float, default=1e-3)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--recall-loss-weight", type=float, default=100.0)
    parser.add_argument("--two-phase", action="store_true",
                        help="Phase 1: memory only. Phase 2: memory + LoRA.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_1")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Resolve memory layer
    if args.memory_layer == -1:
        # Will be set to middle layer after model loads
        pass

    # Build model
    model, tokenizer = build_model(
        model_name=args.model_name,
        device=args.device,
        memory_layer=args.memory_layer,
        memory_size=args.memory_size,
        n_slots=args.n_slots,
        decay=args.decay,
        gate_bias=args.gate_bias,
        lora_rank=args.lora_rank,
        lora_targets=args.lora_targets,
        no_memory=args.no_memory,
        no_lora=args.no_lora,
        init_qk_shared=args.init_qk_shared,
    )

    # Create datasets
    print(f"\nCreating datasets (seed={args.seed})...")
    train_data = PretrainedRecallDataset(
        tokenizer=tokenizer,
        n_docs=args.n_train, gap_range=(args.gap_min, args.gap_max),
        chunk_size=args.chunk_size, n_keys=args.n_keys, n_values=args.n_values,
        n_pairs=args.n_pairs, seed=args.seed,
    )
    eval_data = PretrainedRecallDataset(
        tokenizer=tokenizer,
        n_docs=args.n_eval, gap_range=(args.gap_min, args.gap_max),
        chunk_size=args.chunk_size, n_keys=args.n_keys, n_values=args.n_values,
        n_pairs=args.n_pairs, seed=args.seed + 1000,
    )

    # Train and evaluate
    results = train_and_eval(
        model, train_data, eval_data, args.device,
        num_epochs=args.epochs,
        lr_memory=args.lr_memory,
        lr_lora=args.lr_lora,
        log_every=args.log_every,
        recall_loss_weight=args.recall_loss_weight,
        two_phase=args.two_phase,
        output_dir=args.output_dir,
    )

    results["config"] = vars(args)

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
