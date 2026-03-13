"""Modern Hopfield memory module.

Replaces linear associative memory (M @ q) with softmax-based retrieval
over explicitly stored key-value pairs. Based on Ramsauer et al. (2020):
"Hopfield Networks is All You Need."

Key difference from FastWeightMemory:
- FastWeightMemory: M = Σ v⊗k, retrieval = M @ q (linear, O(d²) storage)
- HopfieldMemory: stores K,V lists, retrieval = softmax(β·K@q) @ V (exponential capacity)

The softmax provides winner-take-all dynamics: the closest key dominates retrieval,
instead of returning a noisy superposition of all stored values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HopfieldMemory(nn.Module):
    """Modern Hopfield associative memory with softmax retrieval.

    Maintains a fixed-size buffer of (key, value) pairs.
    Write: add new key-value pairs to buffer (with decay on old entries).
    Read: softmax attention over stored keys → weighted sum of values.
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 128,
        n_slots: int = 64,
        decay: float = 0.999,
        beta: float = 8.0,
        write_mode: str = "gate",
        gate_bias: float = -2.0,
        w_out_std: float = 0.0,
        init_qk_shared: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.n_slots = n_slots
        self.decay = decay
        self.beta = beta
        self.write_mode = write_mode

        # Projections
        self.W_key = nn.Linear(hidden_size, memory_size, bias=False)
        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
        self.W_value = nn.Linear(hidden_size, memory_size, bias=False)

        # Initialize W_query = W_key so queries match keys from the start
        if init_qk_shared:
            self.W_query.weight.data.copy_(self.W_key.weight.data)

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

        # Learnable inverse temperature
        self.log_beta = nn.Parameter(torch.tensor(float(beta)).log())

        # Memory buffers: fixed-size key and value stores
        # Shape: (n_slots, memory_size) — shared across batch
        self.register_buffer("K_mem", torch.zeros(n_slots, memory_size))
        self.register_buffer("V_mem", torch.zeros(n_slots, memory_size))
        self.register_buffer("mem_strength", torch.zeros(n_slots))  # how "alive" each slot is
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))

        self._nan_debug = False
        self.persistent_grad = False  # if True, don't detach state between calls

    def reset_memory(self):
        self.K_mem.zero_()
        self.V_mem.zero_()
        self.mem_strength.zero_()
        self.write_ptr.zero_()

    def forward(self, x: torch.Tensor, chunk_size: int = 64) -> tuple[torch.Tensor, float | None, dict]:
        """Process sequence in chunks with Hopfield retrieval.

        Args:
            x: (batch, seq_len, hidden_size)
            chunk_size: tokens per chunk

        Returns:
            output, mean_write_strength, aux_losses
        """
        batch, seq_len, _ = x.shape

        # Cast x to match parameter dtype (handles bf16 input with fp32 params)
        param_dtype = self.W_query.weight.dtype
        x_cast = x.to(dtype=param_dtype)

        query = self.W_query(x_cast)
        key = self.W_key(x_cast)
        value = self.W_value(x_cast)

        key = F.normalize(key, dim=-1)
        value = F.normalize(value, dim=-1)

        if self.write_mode == "gate":
            gate_strengths = torch.sigmoid(self.W_gate(x_cast))  # (batch, seq_len, 1)

        beta = self.log_beta.exp()

        retrieved_chunks = []
        write_strength_log = []

        # Cast buffers to match parameter dtype
        K_mem = self.K_mem.clone().to(dtype=param_dtype)
        V_mem = self.V_mem.clone().to(dtype=param_dtype)
        mem_strength = self.mem_strength.clone().to(dtype=param_dtype)
        write_ptr = self.write_ptr.clone()

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            c_len = chunk_end - chunk_start

            q_chunk = query[:, chunk_start:chunk_end, :]  # (batch, c_len, memory_size)

            # ── READ: Hopfield retrieval with softmax ──
            # Attention scores: (batch, c_len, n_slots)
            if mem_strength.sum() < 1e-8:
                # Empty memory → zero output
                r_chunk = torch.zeros_like(q_chunk)
            else:
                # Mask out empty slots
                slot_mask = (mem_strength > 1e-8).to(dtype=param_dtype)  # (n_slots,)
                attn_scores = torch.matmul(q_chunk, K_mem.T) * beta  # (batch, c_len, n_slots)
                attn_scores = attn_scores + (1 - slot_mask).unsqueeze(0).unsqueeze(0) * (-1e9)
                attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, c_len, n_slots)
                r_chunk = torch.matmul(attn_weights, V_mem)  # (batch, c_len, memory_size)

            retrieved_chunks.append(r_chunk)

            # ── WRITE: add chunk's key-values to buffer ──
            k_chunk = key[:, chunk_start:chunk_end, :]
            v_chunk = value[:, chunk_start:chunk_end, :]

            if self.write_mode == "gate":
                ws_chunk = gate_strengths[:, chunk_start:chunk_end, :]  # (batch, c_len, 1)
                write_strength_log.append(ws_chunk.mean().item())
                # Aggregate chunk into single k,v weighted by gate
                weights = ws_chunk.squeeze(-1)  # (batch, c_len)
                w_sum = weights.sum(dim=(0, 1)).clamp(min=1e-8)
                k_agg = (k_chunk * weights.unsqueeze(-1)).sum(dim=(0, 1)) / w_sum  # (memory_size,)
                v_agg = (v_chunk * weights.unsqueeze(-1)).sum(dim=(0, 1)) / w_sum
                write_str = weights.mean().item()
            else:
                # Uniform: mean aggregate
                k_agg = k_chunk.mean(dim=(0, 1))  # (memory_size,)
                v_agg = v_chunk.mean(dim=(0, 1))
                write_str = 1.0

            k_agg = F.normalize(k_agg, dim=-1)
            v_agg = F.normalize(v_agg, dim=-1)

            # Decay existing memories
            mem_strength = mem_strength * (self.decay ** c_len)

            # Write to next slot (circular buffer) — no in-place ops for autograd
            slot_idx = write_ptr % self.n_slots
            mask = F.one_hot(slot_idx, self.n_slots).to(dtype=param_dtype)  # (n_slots,)
            mask_2d = mask.unsqueeze(1)  # (n_slots, 1)
            K_mem = K_mem * (1 - mask_2d) + mask_2d * k_agg.unsqueeze(0)
            V_mem = V_mem * (1 - mask_2d) + mask_2d * v_agg.unsqueeze(0)
            mem_strength = mem_strength * (1 - mask) + mask * write_str
            write_ptr = write_ptr + 1

        # Store updated memory
        if self.persistent_grad:
            # Keep computation graph alive for cross-chunk gradient flow
            self.K_mem = K_mem
            self.V_mem = V_mem
            self.mem_strength = mem_strength
            self.write_ptr = write_ptr.detach()  # ptr is always detached (integer)
        else:
            self.K_mem = K_mem.detach()
            self.V_mem = V_mem.detach()
            self.mem_strength = mem_strength.detach()
            self.write_ptr = write_ptr.detach()

        retrieved = torch.cat(retrieved_chunks, dim=1)
        output = self.W_out(retrieved)

        # Clamp output norm
        max_output_norm = 10.0
        output_norm = output.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        output = output * (max_output_norm / output_norm).clamp(max=1.0)

        # Cast output back to original input dtype
        output = output.to(dtype=x.dtype)

        mean_ws = sum(write_strength_log) / len(write_strength_log) if write_strength_log else None
        return output, mean_ws, {}
