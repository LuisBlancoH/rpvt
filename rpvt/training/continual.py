"""Continual learning utilities.

Handles sequential training on multiple domains and measuring retention.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset

from rpvt.training.losses import global_loss


class DomainDataset(torch.utils.data.Dataset):
    """Tokenized dataset from a specific domain/topic."""

    def __init__(self, tokenizer, texts, seq_len=512, max_tokens=2_000_000):
        all_tokens = []
        batch_texts = []
        batch_char_count = 0

        for text in texts:
            if len(text.strip()) == 0:
                continue
            batch_texts.append(text)
            batch_char_count += len(text)

            if batch_char_count >= 200_000:
                encoded = tokenizer(
                    batch_texts, return_attention_mask=False, truncation=False
                )
                for ids in encoded["input_ids"]:
                    all_tokens.extend(ids)
                batch_texts = []
                batch_char_count = 0
                if len(all_tokens) >= max_tokens:
                    break

        if batch_texts and len(all_tokens) < max_tokens:
            encoded = tokenizer(
                batch_texts, return_attention_mask=False, truncation=False
            )
            for ids in encoded["input_ids"]:
                all_tokens.extend(ids)

        all_tokens = all_tokens[:max_tokens]
        tokens = torch.tensor(all_tokens, dtype=torch.long)
        n_chunks = len(tokens) // seq_len
        self.chunks = tokens[: n_chunks * seq_len].reshape(n_chunks, seq_len)
        self.name = "unknown"

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"input_ids": self.chunks[idx]}


def load_domain_datasets(tokenizer, seq_len=512, max_tokens=2_000_000):
    """Load three distinct domain datasets for continual learning.

    Domain A: Wikipedia (general knowledge)
    Domain B: Code (programming)
    Domain C: Scientific papers (arxiv abstracts)

    Returns dict of {name: DomainDataset}.
    """
    domains = {}

    # Domain A: Wikipedia
    print("  Loading Domain A (Wikipedia)...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts_a = [t for t in wiki["text"] if len(t.strip()) > 50][:50000]
    ds_a = DomainDataset(tokenizer, texts_a, seq_len=seq_len, max_tokens=max_tokens)
    ds_a.name = "wikipedia"
    domains["A"] = ds_a
    print(f"    {len(ds_a)} chunks")

    # Domain B: Code
    print("  Loading Domain B (Code)...")
    try:
        code = load_dataset("codeparrot/codeparrot-clean-valid", split="train")
        texts_b = [t for t in code["content"] if len(t.strip()) > 50][:50000]
    except Exception:
        # Fallback: use a different code dataset
        code = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
        texts_b = []
        for item in code:
            if len(item["content"].strip()) > 50:
                texts_b.append(item["content"])
            if len(texts_b) >= 10000:
                break
    ds_b = DomainDataset(tokenizer, texts_b, seq_len=seq_len, max_tokens=max_tokens)
    ds_b.name = "code"
    domains["B"] = ds_b
    print(f"    {len(ds_b)} chunks")

    # Domain C: Use wikitext validation as a proxy for "different domain"
    # (In practice you'd use arxiv or another distinct domain)
    print("  Loading Domain C (Wiki-validation as distinct domain)...")
    wiki_val = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    texts_c = [t for t in wiki_val["text"] if len(t.strip()) > 50]
    # Also grab test split to get enough data
    wiki_test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts_c.extend([t for t in wiki_test["text"] if len(t.strip()) > 50])
    ds_c = DomainDataset(tokenizer, texts_c, seq_len=seq_len, max_tokens=max_tokens)
    ds_c.name = "wiki_held_out"
    domains["C"] = ds_c
    print(f"    {len(ds_c)} chunks")

    return domains


def evaluate_on_domain(model, dataset, vocab_size, device, max_batches=50, batch_size=4):
    """Evaluate model loss on a specific domain."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = input_ids[:, 1:]
            logits = model(input_ids).logits[:, :-1]
            loss = global_loss(logits, labels, vocab_size)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)
