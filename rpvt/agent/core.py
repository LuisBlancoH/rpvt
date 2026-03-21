"""AgentCore: document QA agent with KV cache memory.

Loads a frozen instruct model, processes documents into KV memory,
and answers questions using stored KV pairs as past_key_values.
No training, no LoRA, no foreign signals — just the model's own
representations replayed.
"""

import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.kv_memory import KVMemoryBank


SYSTEM_PROMPT = (
    "Answer based on the documents you have read. "
    "If you don't have enough information from the documents, say so."
)


class AgentCore:
    """Document QA agent with KV cache memory.

    Args:
        model_name: HuggingFace model name
        device: torch device
        max_entries: maximum KV cache entries to store
        chunk_size: tokens per processing chunk
    """

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cuda",
        max_entries=2048,
        chunk_size=128,
    ):
        self.device = device
        self.chunk_size = chunk_size
        self.model_name = model_name
        self.documents = {}  # doc_id -> metadata

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # Extract model dimensions for KVMemoryBank
        config = self.model.config
        n_layers = config.num_hidden_layers
        n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = config.hidden_size // config.num_attention_heads
        hidden_size = config.hidden_size

        self.kv_memory = KVMemoryBank(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_entries=max_entries,
            hidden_size=hidden_size,
        ).to(device)

        print(f"  Model: {config.hidden_size}d, {n_layers}L, {n_kv_heads} KV heads")
        print(f"  Memory: {max_entries} max entries ({chunk_size} tokens/chunk)")
        print(f"  Ready.")

    def ingest_text(self, text, doc_id=None):
        """Process text into KV memory, chunk by chunk.

        Returns metadata dict with doc_id, n_chunks, n_tokens.
        """
        if doc_id is None:
            doc_id = f"doc_{len(self.documents)}"

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)

        # Check capacity
        space_left = self.kv_memory.max_entries - self.kv_memory.write_ptr.item()
        if space_left <= 0:
            return {"error": "Memory full. Use /clear to reset or increase --max-entries."}

        # Chunk and process
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            ct = tokens[i:i + self.chunk_size]
            if len(ct) < self.chunk_size:
                pad_id = self.tokenizer.eos_token_id or 0
                ct = ct + [pad_id] * (self.chunk_size - len(ct))
            chunks.append(ct)

        n_stored_before = self.kv_memory.n_stored.item()
        t0 = time.time()

        with torch.no_grad():
            for chunk_tokens in chunks:
                if self.kv_memory.write_ptr.item() >= self.kv_memory.max_entries:
                    break
                input_ids = torch.tensor([chunk_tokens], device=self.device)
                output = self.model(input_ids, use_cache=True)
                self.kv_memory.store_all(output.past_key_values)

        n_stored_after = self.kv_memory.n_stored.item()
        elapsed = time.time() - t0

        metadata = {
            "doc_id": doc_id,
            "n_chunks": len(chunks),
            "n_tokens": n_tokens,
            "n_stored": n_stored_after - n_stored_before,
            "time": elapsed,
        }
        self.documents[doc_id] = metadata
        return metadata

    def ingest_file(self, path):
        """Read a file and ingest its text content."""
        from rpvt.agent.file_readers import read_file as read_any_file

        path = Path(path).expanduser()
        if not path.exists():
            return {"error": f"File not found: {path}"}

        text, err = read_any_file(path)
        if err:
            return {"error": err}
        if not text or not text.strip():
            return {"error": f"File is empty: {path}"}

        doc_id = path.stem
        # Avoid duplicate doc_ids
        if doc_id in self.documents:
            i = 1
            while f"{doc_id}_{i}" in self.documents:
                i += 1
            doc_id = f"{doc_id}_{i}"

        return self.ingest_text(text, doc_id=doc_id)

    def generate(self, question, max_new_tokens=200, use_memory=True):
        """Generate a response, optionally using stored KV memory."""
        messages = []
        if use_memory and self.kv_memory.n_stored.item() > 0:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": question})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        past_kvs = None
        if use_memory:
            past_kvs = self.kv_memory.get_past_key_values(
                self.device, torch.bfloat16
            )

        with torch.no_grad():
            if past_kvs is not None:
                n_past = self.kv_memory.n_stored.item()
                seq_len = input_ids.shape[1]
                position_ids = torch.arange(
                    n_past, n_past + seq_len, device=self.device
                ).unsqueeze(0)
                attn_mask = torch.ones(
                    1, n_past + seq_len, device=self.device, dtype=torch.long
                )
                out = self.model.generate(
                    input_ids,
                    past_key_values=past_kvs,
                    position_ids=position_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        return self.tokenizer.decode(
            out[0][len(tokens):], skip_special_tokens=True
        )

    def reset_memory(self):
        """Clear all stored KV pairs and document metadata."""
        self.kv_memory.reset()
        self.documents.clear()

    def memory_status(self):
        """Return memory utilization info."""
        n_stored = self.kv_memory.n_stored.item()
        max_entries = self.kv_memory.max_entries
        return {
            "n_stored": n_stored,
            "max_entries": max_entries,
            "utilization": n_stored / max_entries if max_entries > 0 else 0,
            "n_documents": len(self.documents),
            "documents": dict(self.documents),
        }
