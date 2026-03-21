"""DPO training on agent trajectories with LoRA.

Takes (chosen, rejected) trajectory pairs and trains the model to prefer
successful trajectories over failed ones.

Usage:
    python -m rpvt.agent.train_dpo \
        --trajectories results/trajectories/dpo_pairs.json \
        --output-dir results/agent_dpo
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class DPODataset(Dataset):
    """Dataset of (chosen, rejected) trajectory pairs."""

    def __init__(self, pairs, tokenizer, max_length=4096):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def _tokenize_messages(self, messages):
        """Convert messages to token ids, return (input_ids, labels).

        Labels mask everything except assistant responses.
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        tokens = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)

        # Build labels: mask non-assistant tokens with -100
        labels = input_ids.clone()

        # Find assistant response boundaries
        # We'll use a simpler approach: tokenize up to each assistant turn,
        # mark only the assistant tokens as trainable
        full_text = text
        labels.fill_(-100)

        # Find assistant content in the tokenized text
        # Split by assistant markers
        assistant_marker = "<|im_start|>assistant\n"
        end_marker = "<|im_end|>"

        pos = 0
        while True:
            start = full_text.find(assistant_marker, pos)
            if start == -1:
                break
            content_start = start + len(assistant_marker)
            end = full_text.find(end_marker, content_start)
            if end == -1:
                end = len(full_text)

            # Find token positions for this span
            prefix = full_text[:content_start]
            prefix_tokens = self.tokenizer(
                prefix, add_special_tokens=False,
            )["input_ids"]
            content = full_text[:end]
            content_tokens = self.tokenizer(
                content, add_special_tokens=False,
            )["input_ids"]

            start_idx = min(len(prefix_tokens), len(input_ids) - 1)
            end_idx = min(len(content_tokens), len(input_ids))

            # Unmask assistant tokens
            labels[start_idx:end_idx] = input_ids[start_idx:end_idx]

            pos = end + len(end_marker)

        return input_ids, labels

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        chosen_ids, chosen_labels = self._tokenize_messages(pair["chosen"])
        rejected_ids, rejected_labels = self._tokenize_messages(pair["rejected"])
        return {
            "chosen_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }


def pad_and_stack(batch, pad_id):
    """Collate function that pads sequences to same length."""
    result = {}
    for key in ["chosen_ids", "chosen_labels", "rejected_ids", "rejected_labels"]:
        tensors = [item[key] for item in batch]
        max_len = max(t.shape[0] for t in tensors)
        fill_val = pad_id if "ids" in key else -100
        padded = torch.full((len(tensors), max_len), fill_val, dtype=torch.long)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
        result[key] = padded
    return result


def dpo_loss(model, chosen_ids, chosen_labels, rejected_ids, rejected_labels,
             ref_model=None, beta=0.1):
    """Compute DPO loss.

    L = -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected)
                              - log_ref(chosen) + log_ref(rejected))))
    """
    def get_log_probs(m, input_ids, labels):
        outputs = m(input_ids=input_ids, attention_mask=(input_ids != m.config.pad_token_id).long())
        logits = outputs.logits[:, :-1, :]
        target = labels[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, target.unsqueeze(2)).squeeze(2)
        # Mask padding
        mask = (target != -100).float()
        return (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # Policy log probs
    pi_chosen = get_log_probs(model, chosen_ids, chosen_labels)
    pi_rejected = get_log_probs(model, rejected_ids, rejected_labels)

    # Reference log probs (frozen model)
    if ref_model is not None:
        with torch.no_grad():
            ref_chosen = get_log_probs(ref_model, chosen_ids, chosen_labels)
            ref_rejected = get_log_probs(ref_model, rejected_ids, rejected_labels)
    else:
        ref_chosen = torch.zeros_like(pi_chosen)
        ref_rejected = torch.zeros_like(pi_rejected)

    # DPO objective
    logits = beta * (
        (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    )
    loss = -F.logsigmoid(logits).mean()

    # Metrics
    with torch.no_grad():
        reward_chosen = beta * (pi_chosen - ref_chosen)
        reward_rejected = beta * (pi_rejected - ref_rejected)
        accuracy = (reward_chosen > reward_rejected).float().mean()

    return loss, {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "reward_chosen": reward_chosen.mean().item(),
        "reward_rejected": reward_rejected.mean().item(),
        "reward_margin": (reward_chosen - reward_rejected).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="DPO training with LoRA")
    parser.add_argument("--trajectories", type=str, required=True,
                        help="Path to dpo_pairs.json")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (temperature)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output-dir", type=str, default="results/agent_dpo")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading trajectories from {args.trajectories}...")
    with open(args.trajectories) as f:
        pairs = json.load(f)
    print(f"  {len(pairs)} DPO pairs")

    if len(pairs) == 0:
        print("No DPO pairs found. Need at least one task with both "
              "successful and failed trajectories.")
        return

    # Load model
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load in 4-bit for training
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = DPODataset(pairs, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_and_stack(b, tokenizer.pad_token_id),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Training loop
    print(f"\nTraining: {args.epochs} epochs, lr={args.lr}, beta={args.beta}")
    print(f"  Batch size: {args.batch_size}, grad accum: {args.grad_accum}")
    print(f"  LoRA rank: {args.lora_rank}")

    device = next(model.parameters()).device
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_metrics = {"loss": 0, "accuracy": 0, "reward_margin": 0, "n": 0}
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            loss, metrics = dpo_loss(
                model, chosen_ids, chosen_labels,
                rejected_ids, rejected_labels,
                ref_model=None,  # No ref model — implicit DPO
                beta=args.beta,
            )

            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_metrics["loss"] += metrics["loss"]
            epoch_metrics["accuracy"] += metrics["accuracy"]
            epoch_metrics["reward_margin"] += metrics["reward_margin"]
            epoch_metrics["n"] += 1

        # Flush remaining gradients
        if (batch_idx + 1) % args.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        n = max(epoch_metrics["n"], 1)
        print(f"  Epoch {epoch+1}/{args.epochs}: "
              f"loss={epoch_metrics['loss']/n:.4f}, "
              f"acc={epoch_metrics['accuracy']/n:.3f}, "
              f"margin={epoch_metrics['reward_margin']/n:.3f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "lora_state_dict": {
                    k: v.cpu() for k, v in model.named_parameters()
                    if v.requires_grad
                },
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

    # Save final
    final_path = output_dir / "lora_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved final LoRA to {final_path}")


if __name__ == "__main__":
    main()
