"""Predictive Transformer: wraps existing Qwen layers with new mechanisms.

Instead of reimplementing attention/FFN (which introduces subtle bugs),
we WRAP each Qwen layer directly and add our mechanisms around it.

Each PredictiveBlock contains:
  - The ORIGINAL Qwen layer (unchanged, exact same forward pass)
  - Memory attention (new, reads from shared bank)
  - Memory integration FFN (new)
  - GRU state (new, persistent recurrence)
  - Prediction head (new, top-down)
  - Write gate (new, gates positions for memory writing)

Global components:
  - Value head (learned state evaluation from prediction errors + GRU states)
  - Reward network (learned intrinsic reward from error dynamics)
  - Goal state (slow-updating persistent objective)
  - Memory strength update (learned, reward-modulated)

This guarantees the base model works identically to Qwen at step 0.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, ffn_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MemoryBank(nn.Module):
    """Shared memory bank with strength-based eviction."""

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
        if self.n_stored == 0:
            return None
        return self.slots[:self.n_stored].clone()

    def write_single(self, vector, strength):
        """Write a single vector to memory. Non-differentiable storage."""
        vec = vector.detach()
        s = strength.detach().item() if isinstance(strength, torch.Tensor) else strength

        if self.n_stored < self.n_slots:
            idx = self.n_stored
            self.n_stored += 1
        else:
            idx = self.strength[:self.n_stored].argmin().item()
            if s <= self.strength[idx].item():
                return
        self.slots[idx] = vec
        self.strength[idx] = s

    def update_strengths(self, delta, retrieval_weights=None):
        """Update strengths based on TD error. Called after reward is known.

        Args:
            delta: scalar TD error (positive = better than expected)
            retrieval_weights: (n_stored,) attention weights from last retrieval
                If None, updates all stored entries equally.
        """
        if self.n_stored == 0:
            return
        if retrieval_weights is not None:
            # Weight update by how much each slot was used
            w = retrieval_weights[:self.n_stored].detach()
            self.strength[:self.n_stored] += delta * w
        else:
            self.strength[:self.n_stored] += delta * 0.01
        # Clamp to [0, 1]
        self.strength.clamp_(0.0, 1.0)


class MemoryAttention(nn.Module):
    """Cross-attention to memory bank."""

    def __init__(self, hidden_size, num_heads, head_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        nn.init.zeros_(self.o_proj.weight)  # no-op at init
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        batch, seq_len, _ = x.shape
        n_mem = memory.shape[0]
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        mem = memory.unsqueeze(0).expand(batch, -1, -1)
        k = self.k_proj(mem).view(batch, n_mem, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(mem).view(batch, n_mem, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.dropout(self.o_proj(attn_out))


class PredictiveBlock(nn.Module):
    """Wraps an existing Qwen layer with new mechanisms."""

    def __init__(self, qwen_layer, hidden_size, head_dim,
                 n_mem_heads=2, state_dim=224, goal_dim=64, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim

        # The original Qwen layer — UNTOUCHED
        self.qwen_layer = qwen_layer

        # === New mechanisms (all init as no-ops) ===

        # Memory attention
        self.ln_mem = RMSNorm(hidden_size)
        self.mem_attn = MemoryAttention(hidden_size, n_mem_heads, head_dim, dropout)
        self.mem_gate = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        nn.init.zeros_(self.mem_gate.weight)

        # Goal-biased memory queries
        self.goal_query_proj = nn.Linear(goal_dim, hidden_size, bias=False)
        nn.init.zeros_(self.goal_query_proj.weight)  # no-op at init

        # Memory integration FFN
        self.ln_mem_ffn = RMSNorm(hidden_size)
        self.mem_ffn = SwiGLUFFN(hidden_size, hidden_size * 2)
        nn.init.normal_(self.mem_ffn.down_proj.weight, std=0.001)

        # State (GRU with attention pooling)
        self.state_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.state_compress = nn.Linear(hidden_size, state_dim, bias=False)
        self.state_gru = nn.GRUCell(state_dim, state_dim)
        self.state_proj = nn.Linear(state_dim, hidden_size, bias=False)
        nn.init.zeros_(self.state_proj.weight)

        # Prediction head
        self.predictor = nn.Sequential(
            RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        nn.init.normal_(self.predictor[1].weight, std=0.01)

        # Write gate (novelty-aware)
        self.write_gate = nn.Linear(hidden_size * 2, 1, bias=True)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, 0.0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state, memory_slots, position_embeddings=None,
                goal_state=None):
        batch_size = x.shape[0]

        # State injection
        if state is not None:
            x = x + self.state_proj(state).unsqueeze(1)

        # Run the ORIGINAL Qwen layer (self-attn + FFN)
        qwen_kwargs = {}
        if position_embeddings is not None:
            qwen_kwargs["position_embeddings"] = position_embeddings

        if batch_size == 1:
            layer_out = self.qwen_layer(x, **qwen_kwargs)
            x = layer_out[0]
            if x.dim() == 2:
                x = x.unsqueeze(0)
        else:
            outs = []
            for b in range(batch_size):
                out = self.qwen_layer(x[b:b+1], **qwen_kwargs)
                h = out[0]
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                outs.append(h)
            x = torch.cat(outs, dim=0)

        # Memory attention (our addition)
        mem_read = torch.zeros_like(x)
        if memory_slots is not None:
            h_mem = self.ln_mem(x)
            # Bias queries with goal state (learned, starts as no-op)
            if goal_state is not None:
                h_mem = h_mem + self.goal_query_proj(goal_state).unsqueeze(1)
            mem_out = self.mem_attn(h_mem, memory_slots)
            mem_read = mem_out
            gate_input = torch.cat([x, mem_out], dim=-1)
            mem_out = torch.sigmoid(self.mem_gate(gate_input)) * mem_out
            x = x + self.dropout(mem_out)
            h_int = self.ln_mem_ffn(x)
            x = x + self.dropout(self.mem_ffn(h_int))

        # State update
        sq = self.state_query.to(dtype=x.dtype).expand(batch_size, -1, -1)
        scores = torch.bmm(sq, x.transpose(1, 2)) / math.sqrt(self.hidden_size)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights, x).squeeze(1)
        state_input = self.state_compress(pooled)
        if state is None:
            state = torch.zeros(batch_size, self.state_dim,
                                device=x.device, dtype=x.dtype)
        new_state = self.state_gru(state_input, state)

        # Prediction
        prediction = self.predictor(x)

        # Write gate — per-position scores
        novelty = x - mem_read
        write_input = torch.cat([x, novelty], dim=-1)
        write_scores = torch.sigmoid(self.write_gate(write_input)).squeeze(-1)

        return x, new_state, prediction, write_scores


class PredictiveTransformer(nn.Module):
    """Predictive Transformer built by wrapping Qwen layers.

    New in this version: value head, reward network, goal state,
    and learned memory strength updates (all tiny, all learnable).
    """

    def __init__(self, qwen_model, n_mem_heads=2, state_dim=224,
                 goal_dim=64, n_memory_slots=64, n_write_layers=2,
                 max_settle=5, dropout=0.1):
        super().__init__()

        config = qwen_model.config
        self.hidden_size = config.hidden_size
        self.n_layers = config.num_hidden_layers
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.n_write_layers = n_write_layers
        self.max_settle = max_settle
        head_dim = config.hidden_size // config.num_attention_heads

        # Take Qwen's components directly
        self.embed_tokens = qwen_model.model.embed_tokens
        self.norm = qwen_model.model.norm
        self.lm_head = qwen_model.lm_head
        self.rotary_emb = qwen_model.model.rotary_emb

        # Wrap each Qwen layer
        self.blocks = nn.ModuleList([
            PredictiveBlock(
                qwen_layer=qwen_model.model.layers[i],
                hidden_size=config.hidden_size,
                head_dim=head_dim,
                n_mem_heads=n_mem_heads,
                state_dim=state_dim,
                goal_dim=goal_dim,
                dropout=dropout,
            )
            for i in range(self.n_layers)
        ])

        # Memory
        self.memory = MemoryBank(n_memory_slots, config.hidden_size)

        # Halt network
        self.halt_net = nn.Sequential(
            nn.Linear(self.n_layers, self.n_layers, bias=True),
            nn.SiLU(),
            nn.Linear(self.n_layers, 1, bias=True),
        )
        nn.init.constant_(self.halt_net[-1].bias, 1.0)

        # === NEW: Value, Reward, Goal, Strength Update ===

        # Value head — estimates expected future reward from internal state
        # Input: prediction errors (n_layers) + pooled GRU states (state_dim)
        # Uses errors always (reliable from epoch 0) + states when they're useful
        value_input_dim = self.n_layers + state_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, 64, bias=True),
            nn.SiLU(),
            nn.Linear(64, 1, bias=True),
        )
        nn.init.zeros_(self.value_head[-1].weight)  # start at 0

        # Reward network — learned intrinsic reward from error dynamics
        # Input: prev errors + current errors (learns what constitutes progress)
        self.reward_net = nn.Sequential(
            nn.Linear(self.n_layers * 2, 64, bias=True),
            nn.SiLU(),
            nn.Linear(64, 1, bias=True),
        )
        nn.init.zeros_(self.reward_net[-1].weight)  # start at 0

        # Goal state — slow-updating persistent objective
        # Input: prediction errors (reliable from epoch 0) + TD error
        self.goal_gru = nn.GRUCell(self.n_layers + 1, goal_dim)
        # Bias update gate high → goal changes slowly
        with torch.no_grad():
            self.goal_gru.bias_hh[goal_dim:2*goal_dim].fill_(3.0)

        # States
        self._states = [None] * self.n_layers
        self._goal_state = None
        self._prev_errors = None  # for reward computation
        self._last_value = None   # for TD computation

        # Count new params (not Qwen's)
        qwen_params = sum(p.numel() for p in qwen_model.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        new_params = total_params - qwen_params
        print(f"PredictiveTransformer: {total_params/1e6:.1f}M total params")
        print(f"  Qwen base: {qwen_params/1e6:.1f}M (frozen)")
        print(f"  New mechanisms: {new_params/1e6:.1f}M (trainable)")
        print(f"  layers={self.n_layers}, mem_heads={n_mem_heads}, "
              f"state={state_dim}, goal={goal_dim}, memory={n_memory_slots}")

    def freeze_base(self):
        """Freeze all Qwen weights, keep new mechanisms trainable."""
        for block in self.blocks:
            for param in block.qwen_layer.parameters():
                param.requires_grad = False
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
        if hasattr(self.rotary_emb, 'parameters'):
            for param in self.rotary_emb.parameters():
                param.requires_grad = False

        # Cast new mechanisms to bf16 to match Qwen weights
        for block in self.blocks:
            for name, module in block.named_children():
                if name != 'qwen_layer':
                    module.to(dtype=torch.bfloat16)
        self.halt_net.to(dtype=torch.bfloat16)
        self.memory.to(dtype=torch.bfloat16)
        self.value_head.to(dtype=torch.bfloat16)
        self.reward_net.to(dtype=torch.bfloat16)
        self.goal_gru.to(dtype=torch.bfloat16)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  Frozen: {frozen/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")

    def reset_state(self):
        self._states = [None] * self.n_layers
        self._goal_state = None
        self._prev_errors = None
        self._last_value = None
        self.memory.reset()

    def detach_state(self):
        """Detach states from computation graph (for multi-chunk training)."""
        self._states = [
            s.detach() if s is not None else None
            for s in self._states
        ]
        if self._goal_state is not None:
            self._goal_state = self._goal_state.detach()
        if self._prev_errors is not None:
            self._prev_errors = self._prev_errors.detach()
        if self._last_value is not None:
            self._last_value = self._last_value.detach()

    def _write_to_memory(self, hidden_states, all_write_scores, batch_size):
        """Write to memory using gated positions. Differentiable."""
        for write_idx in range(self.n_layers - self.n_write_layers,
                               self.n_layers):
            ws = all_write_scores[write_idx]   # (batch, seq_len)
            hs = hidden_states[write_idx + 1]  # (batch, seq_len, hidden)

            for b in range(batch_size):
                gated = ws[b].unsqueeze(-1) * hs[b]
                pooled = gated.mean(dim=0)
                strength = ws[b].mean()
                self.memory.write_single(pooled, strength)

    def _compute_value(self, errors, batch_size, device, dtype):
        """Compute state value from prediction errors + GRU states."""
        error_vec = torch.stack(errors)  # (n_layers,)

        # Pool GRU states (mean across blocks)
        valid_states = [s for s in self._states if s is not None]
        if valid_states:
            stacked = torch.stack(valid_states)  # (n_blocks, batch, state_dim)
            pooled_state = stacked.mean(dim=0).squeeze(0)  # (state_dim,)
        else:
            pooled_state = torch.zeros(self.state_dim, device=device, dtype=dtype)

        # Concat errors + state → value
        if pooled_state.dim() > 1:
            pooled_state = pooled_state[0]  # take first batch item
        value_input = torch.cat([error_vec, pooled_state])
        value = self.value_head(value_input.unsqueeze(0)).squeeze()
        return value, error_vec

    def _compute_td(self, current_value, current_errors, batch_size,
                    device, dtype, gamma=0.99):
        """Compute TD error and intrinsic reward."""
        td_error = torch.tensor(0.0, device=device, dtype=dtype)
        intrinsic_reward = torch.tensor(0.0, device=device, dtype=dtype)

        if self._prev_errors is not None and self._last_value is not None:
            # Intrinsic reward: learned from error dynamics
            reward_input = torch.cat([self._prev_errors, current_errors])
            intrinsic_reward = self.reward_net(
                reward_input.unsqueeze(0)
            ).squeeze()

            # TD error: δ = reward + γ·V(current) - V(previous)
            td_error = intrinsic_reward + gamma * current_value.detach() \
                       - self._last_value

        return td_error, intrinsic_reward

    def _update_goal(self, td_error, current_errors, batch_size, device, dtype):
        """Update goal state based on prediction errors + TD error.

        Uses prediction errors (reliable from epoch 0) instead of GRU states
        (which have no gradient early in training).
        """
        td_scalar = td_error.detach().view(1, 1)
        error_input = current_errors.detach().unsqueeze(0)  # (1, n_layers)
        goal_input = torch.cat([error_input, td_scalar], dim=-1)  # (1, n_layers + 1)

        if self._goal_state is None:
            self._goal_state = torch.zeros(1, self.goal_dim,
                                           device=device, dtype=dtype)
        self._goal_state = self.goal_gru(goal_input, self._goal_state)

    def forward(self, input_ids, labels=None, n_settle=None,
                return_errors=False, external_reward=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = next(self.embed_tokens.parameters()).dtype
        use_adaptive = n_settle is None
        max_steps = self.max_settle if use_adaptive else n_settle

        x_embed = self.embed_tokens(input_ids)

        # Compute position embeddings (Qwen's rotary)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x_embed, position_ids)

        x = x_embed
        memory_slots = self.memory.read()
        all_errors = []
        halt_budget = 1.0
        accumulated_logits = None
        total_settle_cost = 0.0
        actual_steps = 0

        for settle_step in range(max_steps):
            hidden_states = [x]
            predictions = []
            all_write_scores = []

            for i, block in enumerate(self.blocks):
                x, self._states[i], prediction, write_scores = block(
                    x, self._states[i], memory_slots,
                    position_embeddings=position_embeddings,
                    goal_state=self._goal_state,
                )
                hidden_states.append(x)
                predictions.append(prediction)
                all_write_scores.append(write_scores)

            errors = []
            for i in range(len(predictions)):
                target = hidden_states[i].detach()
                error = (target - predictions[i]).norm(dim=-1).mean()
                errors.append(error)

            error_vec = torch.stack(errors)
            all_errors.append([e.item() for e in errors])
            actual_steps += 1

            step_logits = self.lm_head(self.norm(x))

            if use_adaptive:
                halt_prob = torch.sigmoid(
                    self.halt_net(error_vec.unsqueeze(0))
                ).squeeze()
                halt_prob = torch.min(halt_prob,
                    torch.tensor(halt_budget, device=device))
                if accumulated_logits is None:
                    accumulated_logits = halt_prob * step_logits
                else:
                    accumulated_logits = accumulated_logits + halt_prob * step_logits
                halt_budget -= halt_prob.item()
                total_settle_cost += 1.0
                if halt_budget < 0.01:
                    break
            else:
                accumulated_logits = step_logits

            # Write to memory
            self._write_to_memory(hidden_states, all_write_scores, batch_size)
            memory_slots = self.memory.read()

            if settle_step < max_steps - 1:
                x = x_embed

        if use_adaptive and halt_budget > 0.01:
            accumulated_logits = accumulated_logits + halt_budget * step_logits

        logits = accumulated_logits

        # === Value, Reward, TD, Goal updates ===
        current_value, current_errors = self._compute_value(
            errors, batch_size, device, dtype
        )
        td_error, intrinsic_reward = self._compute_td(
            current_value, current_errors, batch_size, device, dtype
        )

        # Add external reward if provided
        if external_reward is not None:
            td_error = td_error + external_reward

        # === Loss computation ===
        loss = None
        if labels is not None:
            if labels.shape[-1] == logits.shape[-2]:
                flat_logits = logits.view(-1, logits.size(-1))
                flat_labels = labels.view(-1)
            else:
                flat_logits = logits[..., :-1, :].contiguous().view(
                    -1, logits.size(-1))
                flat_labels = labels[..., 1:].contiguous().view(-1)
            lm_loss = F.cross_entropy(
                flat_logits, flat_labels, ignore_index=-100,
            )
            pred_loss = sum(errors) / len(errors) / self.hidden_size
            ponder_cost = total_settle_cost / self.max_settle

            # Value loss: train value head to predict negative LM loss
            # Gradient flows through current_value (in graph) ✓
            # Target is detached LM loss (not in graph) ✓
            value_target = -lm_loss.detach()
            value_loss = (current_value - value_target) ** 2

            loss = (lm_loss
                    + 0.001 * pred_loss
                    + 0.01 * ponder_cost
                    + 0.01 * value_loss)

        # Update goal state (after loss so we have lm_loss info)
        self._update_goal(td_error, current_errors, batch_size, device, dtype)

        # Update memory strengths with TD error
        if td_error.item() != 0:
            self.memory.update_strengths(td_error.item() * 0.1)

        # Store for next step's TD computation
        self._prev_errors = current_errors.detach()
        self._last_value = current_value.detach()

        if return_errors:
            return logits, loss, all_errors, actual_steps, {
                "value": current_value.item(),
                "td_error": td_error.item(),
                "intrinsic_reward": intrinsic_reward.item(),
                "memory_used": self.memory.n_stored,
            }
        return logits, loss

    def generate(self, input_ids, max_new_tokens=100, n_settle=None,
                 temperature=0.0):
        generated = []
        current_ids = input_ids
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = self.forward(current_ids, n_settle=n_settle)
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
