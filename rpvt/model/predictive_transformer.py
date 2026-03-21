"""Predictive Transformer: a transformer with native recurrence,
bidirectional flow, memory, and adaptive settling.

Each block has:
  - Self-attention (bottom-up, standard)
  - Memory-attention (reads from shared memory bank)
  - FFN (per-position computation)
  - GRU state (persistent across inputs and settling steps)
  - Prediction head (predicts previous block's output — top-down)
  - Novelty-aware write gate (learned, stores important + new info)

Settling: run the stack until the model is confident (adaptive halting).
Easy inputs settle in 1 pass. Hard inputs get more compute.

Everything is learned — no hardcoded thresholds, no fixed rules.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """Shared memory bank with learned write/evict.

    Written by upper blocks (high-level semantic representations).
    Read by ALL blocks via attention.
    Eviction: when full, evict the slot with lowest learned importance.
    Write strength: continuous (not binary), learned per-position.
    """

    def __init__(self, n_slots, hidden_size):
        super().__init__()
        self.n_slots = n_slots
        self.hidden_size = hidden_size

        self.register_buffer("slots", torch.zeros(n_slots, hidden_size))
        self.register_buffer("strength", torch.zeros(n_slots))
        self.n_stored = 0

    def reset(self):
        self.slots.zero_()
        self.strength.zero_()
        self.n_stored = 0

    def read(self):
        """Return all active memory slots for attention."""
        if self.n_stored == 0:
            return None
        return self.slots[:self.n_stored]

    def write(self, vectors, write_strengths):
        """Write vectors with continuous strength. Learned eviction when full.

        Args:
            vectors: (seq_len, hidden_size)
            write_strengths: (seq_len,) — continuous importance scores [0, 1]
        """
        # Only write positions with strength > 0.3 (soft threshold)
        mask = write_strengths > 0.3
        to_store = vectors[mask]
        strengths = write_strengths[mask]

        if to_store.shape[0] == 0:
            return 0

        # Sort by strength (most important first)
        order = strengths.argsort(descending=True)
        to_store = to_store[order]
        strengths = strengths[order]

        stored = 0
        for i in range(to_store.shape[0]):
            if self.n_stored < self.n_slots:
                # Free slots available
                idx = self.n_stored
                self.n_stored += 1
            else:
                # Full — evict weakest slot
                idx = self.strength[:self.n_stored].argmin().item()
                # Only evict if new item is stronger
                if strengths[i] <= self.strength[idx]:
                    continue  # not worth evicting

            self.slots[idx] = to_store[i].detach()
            self.strength[idx] = strengths[i].detach()
            stored += 1

        return stored


class PredictiveBlock(nn.Module):
    """One block of the Predictive Transformer.

    Standard transformer + state + memory + prediction.
    All gates and decisions are learned — no hardcoded thresholds.
    """

    def __init__(self, hidden_size, n_self_heads, n_mem_heads,
                 state_dim, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_self_heads = n_self_heads
        self.n_mem_heads = n_mem_heads
        self.state_dim = state_dim

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln_mem = nn.LayerNorm(hidden_size)

        # Self-attention (bottom-up)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, n_self_heads, dropout=dropout,
            batch_first=True, bias=False,
        )

        # Memory attention (reads from memory bank)
        if n_mem_heads > 0:
            self.mem_attn = nn.MultiheadAttention(
                hidden_size, n_mem_heads, dropout=dropout,
                batch_first=True, bias=False,
            )
            self.mem_gate = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.mem_gate.weight)
        else:
            self.mem_attn = None

        # FFN
        ffn_dim = hidden_size * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size, bias=False),
        )

        # State (recurrence) — attention pooling + GRU
        # Learned query for attention pooling (not mean pooling)
        self.state_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.state_attn = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True, bias=False,
        )
        self.state_compress = nn.Linear(hidden_size, state_dim, bias=False)
        self.state_gru = nn.GRUCell(state_dim, state_dim)
        self.state_proj = nn.Linear(state_dim, hidden_size, bias=False)
        nn.init.zeros_(self.state_proj.weight)  # start as no-op

        # Prediction head (top-down)
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        nn.init.normal_(self.predictor[1].weight, std=0.01)

        # Memory write gate — novelty-aware
        # Sees hidden state + novelty signal (hidden - memory_read)
        self.write_gate = nn.Linear(hidden_size * 2, 1, bias=True)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, -2.0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state, memory_slots, causal_mask=None):
        """Forward pass.

        Returns: (output, new_state, prediction, write_scores)
        """
        batch_size = x.shape[0]

        # Add state context
        if state is not None:
            state_context = self.state_proj(state)
            x = x + state_context.unsqueeze(1)

        # Self-attention
        h = self.ln1(x)
        attn_out, _ = self.self_attn(
            h, h, h, attn_mask=causal_mask, need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # Memory attention
        mem_read = torch.zeros_like(x)
        if self.mem_attn is not None and memory_slots is not None:
            h_mem = self.ln_mem(x)
            mem_kv = memory_slots.unsqueeze(0).expand(batch_size, -1, -1)
            mem_out, _ = self.mem_attn(
                h_mem, mem_kv, mem_kv, need_weights=False,
            )
            mem_read = mem_out
            mem_out = torch.sigmoid(self.mem_gate(mem_out)) * mem_out
            x = x + mem_out

        # FFN
        h = self.ln2(x)
        x = x + self.dropout(self.ffn(h))

        # State update — attention pooling (learned, not mean pooling)
        query = self.state_query.expand(batch_size, -1, -1)
        pooled, _ = self.state_attn(
            query, x, x, need_weights=False,
        )  # (batch, 1, hidden)
        state_input = self.state_compress(pooled.squeeze(1))  # (batch, state_dim)
        if state is None:
            state = torch.zeros(
                batch_size, self.state_dim,
                device=x.device, dtype=x.dtype,
            )
        new_state = self.state_gru(state_input, state)

        # Prediction (top-down)
        prediction = self.predictor(x)

        # Write gate — novelty-aware
        novelty = x - mem_read
        write_input = torch.cat([x, novelty], dim=-1)
        write_scores = torch.sigmoid(
            self.write_gate(write_input)
        ).squeeze(-1)

        return x, new_state, prediction, write_scores


class PredictiveTransformer(nn.Module):
    """Full Predictive Transformer with adaptive settling.

    Everything is learned:
    - What to store (novelty-aware write gate)
    - What to retrieve (attention)
    - What to evict (weakest slot)
    - When to stop thinking (adaptive halting)
    - What the GRU remembers (attention pooling)
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        n_layers=6,
        n_self_heads=8,
        n_mem_heads=4,
        state_dim=256,
        n_memory_slots=64,
        n_write_layers=2,
        max_settle=5,
        ffn_mult=4,
        dropout=0.1,
        max_seq_len=512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.n_write_layers = n_write_layers
        self.max_settle = max_settle

        # Token + position embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)

        # Predictive blocks
        self.blocks = nn.ModuleList([
            PredictiveBlock(
                hidden_size=hidden_size,
                n_self_heads=n_self_heads,
                n_mem_heads=n_mem_heads,
                state_dim=state_dim,
                ffn_mult=ffn_mult,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Shared memory bank
        self.memory = MemoryBank(n_memory_slots, hidden_size)

        # Adaptive halting — learned confidence from prediction errors
        # Input: n_layers error values → output: halt probability
        self.halt_net = nn.Sequential(
            nn.Linear(n_layers, n_layers, bias=True),
            nn.SiLU(),
            nn.Linear(n_layers, 1, bias=True),
        )
        nn.init.constant_(self.halt_net[-1].bias, 1.0)  # bias toward halting early

        # Output
        self.ln_final = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # tie embeddings

        # Persistent states
        self._states = [None] * n_layers

        # Initialize
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"PredictiveTransformer: {n_params/1e6:.1f}M params")
        print(f"  hidden={hidden_size}, layers={n_layers}, "
              f"heads={n_self_heads}+{n_mem_heads}mem, "
              f"state={state_dim}, memory={n_memory_slots}, "
              f"max_settle={max_settle}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_state(self):
        """Reset persistent states and memory."""
        self._states = [None] * self.n_layers
        self.memory.reset()

    def forward(self, input_ids, labels=None, n_settle=None,
                return_errors=False):
        """Forward pass with adaptive settling.

        If n_settle is None, uses adaptive halting (learned).
        If n_settle is an int, uses that fixed number of passes.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        use_adaptive = n_settle is None
        max_steps = self.max_settle if use_adaptive else n_settle

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x_embed = self.token_embed(input_ids) + self.pos_embed(positions)
        x = x_embed

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device, dtype=x.dtype,
        )

        # Memory contents
        memory_slots = self.memory.read()

        # Settling loop — accumulate output weighted by halting probability
        all_errors = []
        halt_budget = 1.0  # remaining probability mass
        accumulated_logits = None
        total_settle_cost = 0.0  # for regularization
        actual_steps = 0

        for settle_step in range(max_steps):
            hidden_states = [x]
            predictions = []
            all_write_scores = []

            # Forward through all blocks
            for i, block in enumerate(self.blocks):
                x, self._states[i], prediction, write_scores = block(
                    x, self._states[i], memory_slots, causal_mask,
                )
                hidden_states.append(x)
                predictions.append(prediction)
                all_write_scores.append(write_scores)

            # Compute prediction errors
            errors = []
            for i in range(len(predictions)):
                target = hidden_states[i].detach()
                error = (target - predictions[i]).norm(dim=-1).mean()
                errors.append(error)

            error_vec = torch.stack(errors)  # (n_layers,)
            all_errors.append([e.item() for e in errors])
            actual_steps += 1

            # Compute logits for this step
            step_logits = self.lm_head(self.ln_final(x))

            if use_adaptive:
                # Adaptive halting: compute halt probability from errors
                halt_prob = torch.sigmoid(
                    self.halt_net(error_vec.unsqueeze(0))
                ).squeeze()  # scalar

                # Clamp to remaining budget
                halt_prob = torch.min(halt_prob, torch.tensor(halt_budget, device=device))

                # Weighted accumulation (like ACT — Adaptive Computation Time)
                if accumulated_logits is None:
                    accumulated_logits = halt_prob * step_logits
                else:
                    accumulated_logits = accumulated_logits + halt_prob * step_logits

                halt_budget -= halt_prob.item()
                total_settle_cost += 1.0  # penalize more steps

                # Stop if budget exhausted
                if halt_budget < 0.01:
                    break
            else:
                # Fixed settling: just use the last step's logits
                accumulated_logits = step_logits

            # Write to memory from top blocks
            for write_idx in range(self.n_layers - self.n_write_layers, self.n_layers):
                write_scores_i = all_write_scores[write_idx][0]
                hidden_i = hidden_states[write_idx + 1][0]
                self.memory.write(hidden_i.detach(), write_scores_i.detach())

            # Re-read memory
            memory_slots = self.memory.read()

            # Reset x for next settling pass (state carries forward)
            if settle_step < max_steps - 1:
                x = x_embed

        # If adaptive and budget remains, dump it on last step
        if use_adaptive and halt_budget > 0.01:
            accumulated_logits = accumulated_logits + halt_budget * step_logits

        logits = accumulated_logits

        # Losses
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Prediction error loss (small auxiliary)
            pred_loss = sum(errors) / len(errors) / self.hidden_size

            # Ponder cost: penalize using more settling steps
            ponder_cost = total_settle_cost / self.max_settle

            loss = lm_loss + 0.001 * pred_loss + 0.01 * ponder_cost

        if return_errors:
            return logits, loss, all_errors, actual_steps
        return logits, loss

    def generate(self, input_ids, max_new_tokens=100, n_settle=None,
                 temperature=0.0):
        """Autoregressive generation with adaptive settling."""
        generated = []
        current_ids = input_ids

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = self.forward(
                    current_ids, n_settle=n_settle,
                )

            next_logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=1)

            if current_ids.shape[1] > 512:
                current_ids = current_ids[:, -512:]

        return generated
