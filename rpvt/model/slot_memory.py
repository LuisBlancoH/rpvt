"""Slot-based memory module.

N independent memory slots with content-based addressing.
Each slot stores one (key, value) pair. Writing uses attention-based
addressing to find the best slot. Reading uses softmax attention over keys.

Key difference from FastWeightMemory:
- FastWeightMemory: single matrix M, linear superposition of all stored pairs
- SlotMemory: N separate slots, attention-based read/write, no interference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotMemory(nn.Module):
    """Slot-based memory with content-based addressing.

    Write: soft-attention over slots to decide where to write.
    Read: soft-attention over slot keys to retrieve values.
    No interference between slots — each is independent.
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 128,
        n_slots: int = 32,
        decay: float = 0.999,
        write_mode: str = "gate",
        gate_bias: float = -2.0,
        w_out_std: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.n_slots = n_slots
        self.decay = decay
        self.write_mode = write_mode

        # Projections
        self.W_key = nn.Linear(hidden_size, memory_size, bias=False)
        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
        self.W_value = nn.Linear(hidden_size, memory_size, bias=False)

        # Write address: project hidden to slot-addressing key
        self.W_write_key = nn.Linear(hidden_size, memory_size, bias=False)

        # Output projection
        self.W_out = nn.Linear(memory_size, hidden_size, bias=False)
        if w_out_std > 0:
            nn.init.normal_(self.W_out.weight, std=w_out_std)
        else:
            nn.init.zeros_(self.W_out.weight)

        # Write gate
        if write_mode == "gate":
            self.W_gate = nn.Linear(hidden_size, 1, bias=True)
            nn.init.zeros_(self.W_gate.weight)
            nn.init.constant_(self.W_gate.bias, gate_bias)

        # Slot storage
        self.register_buffer("slot_keys", torch.zeros(n_slots, memory_size))
        self.register_buffer("slot_values", torch.zeros(n_slots, memory_size))
        self.register_buffer("slot_usage", torch.zeros(n_slots))  # how recently used

        self._nan_debug = False

    def reset_memory(self):
        self.slot_keys.zero_()
        self.slot_values.zero_()
        self.slot_usage.zero_()

    def forward(self, x: torch.Tensor, chunk_size: int = 64) -> tuple[torch.Tensor, float | None, dict]:
        """Process sequence in chunks with slot-based memory.

        Args:
            x: (batch, seq_len, hidden_size)
            chunk_size: tokens per chunk

        Returns:
            output, mean_write_strength, aux_losses
        """
        batch, seq_len, _ = x.shape

        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)
        write_key = self.W_write_key(x)

        key = F.normalize(key, dim=-1)
        value = F.normalize(value, dim=-1)
        write_key = F.normalize(write_key, dim=-1)

        if self.write_mode == "gate":
            gate_strengths = torch.sigmoid(self.W_gate(x))

        retrieved_chunks = []
        write_strength_log = []

        slot_keys = self.slot_keys.clone()      # (n_slots, memory_size)
        slot_values = self.slot_values.clone()   # (n_slots, memory_size)
        slot_usage = self.slot_usage.clone()     # (n_slots,)

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            c_len = chunk_end - chunk_start

            q_chunk = query[:, chunk_start:chunk_end, :]  # (batch, c_len, memory_size)

            # ── READ: attention over slot keys ──
            if slot_usage.sum() < 1e-8:
                r_chunk = torch.zeros_like(q_chunk)
            else:
                slot_mask = (slot_usage > 1e-8).float()
                attn_scores = torch.matmul(q_chunk, slot_keys.T)  # (batch, c_len, n_slots)
                attn_scores = attn_scores / (self.memory_size ** 0.5)  # scale
                attn_scores = attn_scores + (1 - slot_mask).unsqueeze(0).unsqueeze(0) * (-1e9)
                attn_weights = F.softmax(attn_scores, dim=-1)
                r_chunk = torch.matmul(attn_weights, slot_values)  # (batch, c_len, memory_size)

            retrieved_chunks.append(r_chunk)

            # ── WRITE: content-based slot addressing ──
            k_chunk = key[:, chunk_start:chunk_end, :]
            v_chunk = value[:, chunk_start:chunk_end, :]
            wk_chunk = write_key[:, chunk_start:chunk_end, :]

            if self.write_mode == "gate":
                ws_chunk = gate_strengths[:, chunk_start:chunk_end, :]
                write_strength_log.append(ws_chunk.mean().item())
                weights = ws_chunk.squeeze(-1)
                w_sum = weights.sum(dim=(0, 1)).clamp(min=1e-8)
                k_agg = (k_chunk * weights.unsqueeze(-1)).sum(dim=(0, 1)) / w_sum
                v_agg = (v_chunk * weights.unsqueeze(-1)).sum(dim=(0, 1)) / w_sum
                wk_agg = (wk_chunk * weights.unsqueeze(-1)).sum(dim=(0, 1)) / w_sum
                write_str = weights.mean().item()
            else:
                k_agg = k_chunk.mean(dim=(0, 1))
                v_agg = v_chunk.mean(dim=(0, 1))
                wk_agg = wk_chunk.mean(dim=(0, 1))
                write_str = 1.0

            k_agg = F.normalize(k_agg, dim=-1)
            v_agg = F.normalize(v_agg, dim=-1)
            wk_agg = F.normalize(wk_agg, dim=-1)

            # Decay slot usage
            slot_usage = slot_usage * (self.decay ** c_len)

            # Soft write addressing: attention over slots decides where to write
            # Prefer empty/least-used slots (inverse usage) blended with content similarity
            if slot_usage.sum() < 1e-8:
                # Empty memory: uniform write to all slots (first write)
                write_weights = torch.ones(self.n_slots, device=slot_keys.device) / self.n_slots
            else:
                content_sim = torch.matmul(slot_keys, wk_agg)  # (n_slots,)
                # Blend: content similarity for matching slots + inverse usage for new slots
                emptiness = 1 - slot_usage.clamp(0, 1)
                write_logits = content_sim + emptiness * 2.0  # bias toward empty slots
                write_weights = F.softmax(write_logits * 5.0, dim=0)  # (n_slots,) — sharp addressing

            # Soft update: each slot gets a weighted blend
            alpha = write_str * write_weights  # (n_slots,)
            alpha_2d = alpha.unsqueeze(1)  # (n_slots, 1)
            new_keys = alpha_2d * k_agg.unsqueeze(0) + (1 - alpha_2d) * slot_keys
            new_values = alpha_2d * v_agg.unsqueeze(0) + (1 - alpha_2d) * slot_values
            slot_keys = F.normalize(new_keys, dim=-1)
            slot_values = F.normalize(new_values, dim=-1)
            slot_usage = slot_usage + write_weights * write_str

        # Store updated memory
        self.slot_keys = slot_keys.detach()
        self.slot_values = slot_values.detach()
        self.slot_usage = slot_usage.detach()

        retrieved = torch.cat(retrieved_chunks, dim=1)
        output = self.W_out(retrieved)

        # Clamp output norm
        max_output_norm = 10.0
        output_norm = output.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        output = output * (max_output_norm / output_norm).clamp(max=1.0)

        mean_ws = sum(write_strength_log) / len(write_strength_log) if write_strength_log else None
        return output, mean_ws, {}
