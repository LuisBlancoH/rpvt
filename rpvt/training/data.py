"""Data loading utilities for training."""

from datasets import load_dataset
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    """Pre-tokenized dataset that returns fixed-length sequences."""

    def __init__(self, tokenizer, dataset_name="wikitext", split="train", seq_len=512, max_samples=None):
        if dataset_name == "wikitext":
            raw = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        else:
            raw = load_dataset(dataset_name, split=split)

        # Concatenate all text and tokenize
        texts = [t for t in raw["text"] if len(t.strip()) > 0]
        if max_samples:
            texts = texts[:max_samples]

        full_text = "\n".join(texts)
        tokens = tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"][0]

        # Split into fixed-length chunks
        n_chunks = len(tokens) // seq_len
        self.chunks = tokens[: n_chunks * seq_len].reshape(n_chunks, seq_len)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"input_ids": self.chunks[idx]}
