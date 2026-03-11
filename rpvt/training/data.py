"""Data loading utilities for training."""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


class TokenizedDataset(Dataset):
    """Pre-tokenized dataset that returns fixed-length sequences."""

    def __init__(
        self,
        tokenizer,
        dataset_name="wikitext",
        split="train",
        seq_len=512,
        max_tokens=5_000_000,
    ):
        if dataset_name == "wikitext":
            raw = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        else:
            raw = load_dataset(dataset_name, split=split)

        # Tokenize in batches to avoid memory issues
        all_tokens = []
        batch_texts = []
        batch_char_count = 0
        target_chars_per_batch = 500_000

        for text in raw["text"]:
            if len(text.strip()) == 0:
                continue
            batch_texts.append(text)
            batch_char_count += len(text)

            if batch_char_count >= target_chars_per_batch:
                encoded = tokenizer(
                    batch_texts, return_attention_mask=False, truncation=False
                )
                for ids in encoded["input_ids"]:
                    all_tokens.extend(ids)
                batch_texts = []
                batch_char_count = 0

                if len(all_tokens) >= max_tokens:
                    break

        # Flush remaining
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
        print(f"  [{split}] {len(tokens):,} tokens -> {n_chunks:,} chunks of {seq_len}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"input_ids": self.chunks[idx]}
