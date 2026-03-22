"""Recurrent Memory Transformer: simple, general, scales.

Frozen Qwen + small LoRA + learned memory extraction.
Two mechanisms: attention (retrieval) + FFN (representation).
Memory carries both facts AND state.
Settling: model decides when to stop via confidence extraction.

Design principles:
  - Minimal structure, maximal learning (bitter lesson)
  - One mechanism type (attention+FFN) applied everywhere
  - Memory IS state (no separate GRU/goal/value)
  - Halt from same mechanism as extraction (not separate network)
  - LoRA gives attention flexibility for memory retrieval
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig


class MemoryFFN(nn.Module):
    """SwiGLU FFN for memory transformation."""

    def __init__(self, hidden_size, ffn_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MemoryExtractor(nn.Module):
    """Extract and transform memory vectors from hidden states.

    K learned queries attend to output hidden states → FFN transforms
    into memory representation. One additional query for halt confidence.
    """

    def __init__(self, hidden_size, n_memory_tokens=16, n_heads=4,
                 ffn_mult=2, dropout=0.1):
        super().__init__()
        self.n_memory_tokens = n_memory_tokens
        self.hidden_size = hidden_size

        # K+2 learned queries: K for memory, 1 for confidence, 1 for value
        self.queries = nn.Parameter(
            torch.randn(n_memory_tokens + 2, hidden_size) * 0.02
        )

        # Cross-attention for extraction
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        # Memory FFN: transform extracted vectors
        self.ln = nn.RMSNorm(hidden_size)
        self.ffn = MemoryFFN(hidden_size, hidden_size * ffn_mult)

        # Priority: per-vector importance score for eviction
        self.priority_proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.constant_(self.priority_proj.bias, 0.5)  # start neutral

        # Halt: project confidence query output to scalar
        self.halt_proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.constant_(self.halt_proj.bias, 1.0)  # start biased toward stopping

        # Value: deep network that sees state (value query) + action context (pooled hidden)
        # Computes value from BOTH what I know AND what I'm about to do
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, bias=True),  # state + action
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 4, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1, bias=True),
        )
        # Zero-init final layer so value starts at 0
        nn.init.zeros_(self.value_net[-1].weight)
        nn.init.zeros_(self.value_net[-1].bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        """Extract memory vectors + confidence from hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            memory: (batch, K, hidden_size) — memory vectors
            confidence: (batch,) — halt confidence
        """
        batch = hidden_states.shape[0]
        n_q = self.n_memory_tokens + 2  # K memory + 1 confidence + 1 value

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)

        # Cross-attention: queries attend to hidden states
        q = self.q_proj(queries).view(batch, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, n_q, -1)
        extracted = self.o_proj(attn_out)

        # Split: K memory + 1 confidence + 1 value
        memory_raw = extracted[:, :self.n_memory_tokens, :]       # (batch, K, hidden)
        confidence_vec = extracted[:, self.n_memory_tokens, :]     # (batch, hidden)
        value_vec = extracted[:, self.n_memory_tokens + 1, :]      # (batch, hidden)

        # Transform memory through FFN
        memory = memory_raw + self.dropout(self.ffn(self.ln(memory_raw)))

        # Priority per memory vector
        priority = torch.sigmoid(self.priority_proj(memory)).squeeze(-1)

        # Confidence (halt)
        confidence = torch.sigmoid(self.halt_proj(confidence_vec)).squeeze(-1)

        # Value: deep computation from state (value query) + action (last position)
        # Last position's hidden state is what determines the next token
        action_context = hidden_states[:, -1, :]  # (batch, hidden) — what model will generate
        value_input = torch.cat([value_vec, action_context], dim=-1)  # (batch, hidden*2)
        value = self.value_net(value_input).squeeze(-1)  # scalar

        return memory, priority, confidence, value, value_vec


class MemoryBuffer:
    """Memory buffer with learned priority-based eviction.

    When space available: append (fast, like FIFO).
    When full: replace lowest-priority entries (learned eviction).
    """

    def __init__(self, max_entries, hidden_size, device='cpu', dtype=torch.bfloat16):
        self.max_entries = max_entries
        self.hidden_size = hidden_size
        self.entries = torch.zeros(max_entries, hidden_size,
                                   device=device, dtype=dtype)
        self.priorities = torch.zeros(max_entries, device=device, dtype=torch.float32)
        self.n_stored = 0

    def reset(self):
        self.entries.zero_()
        self.priorities.zero_()
        self.n_stored = 0

    def read(self):
        if self.n_stored == 0:
            return None
        return self.entries[:self.n_stored].clone()

    def store(self, vectors, priorities=None):
        """Store K vectors with priorities.

        When space available: append.
        When full: replace entries with lowest priority (if new priority is higher).

        Args:
            vectors: (K, hidden_size) or (batch, K, hidden_size)
            priorities: (K,) or (batch, K) — importance scores (0-1)
        """
        if vectors.dim() == 3:
            vectors = vectors[0]
        if priorities is not None and priorities.dim() == 2:
            priorities = priorities[0]

        K = vectors.shape[0]
        vecs = vectors.detach()
        pris = priorities.detach().float() if priorities is not None \
               else torch.full((K,), 0.5, device=vecs.device)

        if self.n_stored + K <= self.max_entries:
            # Space available — just append
            self.entries[self.n_stored:self.n_stored + K] = vecs
            self.priorities[self.n_stored:self.n_stored + K] = pris
            self.n_stored += K
        else:
            # Full — replace lowest-priority entries
            for i in range(K):
                if self.n_stored < self.max_entries:
                    idx = self.n_stored
                    self.n_stored += 1
                else:
                    idx = self.priorities[:self.n_stored].argmin().item()
                    if pris[i] <= self.priorities[idx]:
                        continue  # not important enough to evict
                self.entries[idx] = vecs[i]
                self.priorities[idx] = pris[i]

    def to(self, device=None, dtype=None):
        if device is not None:
            self.entries = self.entries.to(device)
            self.priorities = self.priorities.to(device)
        if dtype is not None:
            self.entries = self.entries.to(dtype)
        return self


class RecurrentMemoryTransformer(nn.Module):
    """Recurrent Memory Transformer.

    Simple architecture: frozen Qwen + LoRA + memory extraction.
    Memory injected as extra KV pairs. Model's own attention retrieves.
    Settling: re-run with updated memory, model decides when to stop.

    Args:
        qwen_model: loaded Qwen model
        n_memory_tokens: K extraction queries (how many vectors per chunk)
        max_memory_entries: total FIFO buffer capacity
        n_extract_heads: heads in extraction attention
        lora_rank: LoRA rank for attention adaptation (0 = no LoRA)
        max_passes: maximum settling iterations
    """

    def __init__(self, qwen_model, n_memory_tokens=16, max_memory_entries=128,
                 n_extract_heads=4, lora_rank=8, max_passes=3, dropout=0.1):
        super().__init__()

        config = qwen_model.config
        self.hidden_size = config.hidden_size
        self.n_layers = config.num_hidden_layers
        self.max_passes = max_passes

        # Apply LoRA if requested
        if lora_rank > 0:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules="all-linear",
                # Entire transformer adapts to process memory-augmented states
                lora_dropout=dropout,
                bias="none",
            )
            qwen_model = get_peft_model(qwen_model, lora_config)
            lora_params = sum(p.numel() for p in qwen_model.parameters()
                            if p.requires_grad)
            print(f"  LoRA: rank={lora_rank}, params={lora_params/1e6:.2f}M")

        # Store Qwen components
        base = qwen_model.model if not hasattr(qwen_model, 'base_model') else qwen_model.base_model.model.model
        self.qwen = qwen_model
        self.embed_tokens = base.embed_tokens
        self.norm = base.norm
        self.rotary_emb = base.rotary_emb
        self.layers = base.layers

        # LM head
        if hasattr(qwen_model, 'base_model'):
            self.lm_head = qwen_model.base_model.model.lm_head
        else:
            self.lm_head = qwen_model.lm_head

        # Memory extraction
        self.extractor = MemoryExtractor(
            hidden_size=config.hidden_size,
            n_memory_tokens=n_memory_tokens,
            n_heads=n_extract_heads,
            ffn_mult=2,
            dropout=dropout,
        )

        self._n_heads = config.num_attention_heads

        # Per-layer memory cross-attention (simple: 2 heads each)
        n_mem_heads = 2
        self.mem_attns = nn.ModuleList([
            nn.ModuleDict({
                'q': nn.Linear(config.hidden_size, n_mem_heads * 64, bias=False),
                'k': nn.Linear(config.hidden_size, n_mem_heads * 64, bias=False),
                'v': nn.Linear(config.hidden_size, n_mem_heads * 64, bias=False),
                'o': nn.Linear(n_mem_heads * 64, config.hidden_size, bias=False),
                'gate': nn.Linear(1, 1, bias=True),  # dummy, using Parameter instead
            })
            for _ in range(self.n_layers)
        ])
        # Zero-init output projections (no-op at start)
        for attn in self.mem_attns:
            nn.init.zeros_(attn['o'].weight)

        self._mem_heads = n_mem_heads
        self._mem_head_dim = 64

        # Per-layer memory gates
        self.memory_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(self.n_layers)
        ])

        # Value token: processed through all 24 layers alongside input
        # Gets deep understanding of state + memory before value estimation
        self.value_token = nn.Parameter(
            torch.randn(1, config.hidden_size) * 0.02
        )
        # Deep value MLP: value_token_hidden (state) + last_pos (action) → scalar
        self.value_mlp = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size // 4, bias=True),
            nn.SiLU(),
            nn.Linear(config.hidden_size // 4, 1, bias=True),
        )
        nn.init.zeros_(self.value_mlp[-1].weight)
        nn.init.zeros_(self.value_mlp[-1].bias)

        # Cast new modules to bf16
        self.extractor.to(dtype=torch.bfloat16)
        self.mem_attns.to(dtype=torch.bfloat16)
        self.value_token.data = self.value_token.data.to(dtype=torch.bfloat16)
        self.value_mlp.to(dtype=torch.bfloat16)

        # Memory buffer
        self.memory_buffer = MemoryBuffer(
            max_memory_entries, config.hidden_size, dtype=torch.bfloat16
        )

        # Count params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"RecurrentMemoryTransformer: {total/1e6:.1f}M total")
        print(f"  Frozen: {frozen/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
        print(f"  memory_tokens={n_memory_tokens}, buffer={max_memory_entries}, "
              f"max_passes={max_passes}")

    def reset_memory(self):
        self.memory_buffer.reset()

    def _forward_with_memory(self, input_ids, memory=None):
        """Run Qwen layers with memory + value token.

        Value token is prepended and processed through all 24 layers.
        It attends to input + memory, building deep state understanding.
        Returns (output_hidden, value_hidden) where value_hidden is
        the value token's final representation.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.embed_tokens(input_ids)

        # Prepend value token
        vt = self.value_token.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 1, hidden)
        x = torch.cat([vt, x], dim=1)  # (batch, 1+seq_len, hidden)

        # Position IDs: value token gets position 0, input starts at 1
        position_ids = torch.arange(seq_len + 1, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)

        for i, layer in enumerate(self.layers):
            # Standard Qwen layer
            layer_out = layer(x, position_embeddings=position_embeddings)
            x = layer_out[0]
            if x.dim() == 2:
                x = x.unsqueeze(0)

            # Cross-attend to memory (gated, o_proj zero-init = no-op at start)
            if memory is not None:
                gate = torch.sigmoid(self.memory_gates[i]).to(dtype=x.dtype)
                attn = self.mem_attns[i]
                n_mem = memory.shape[0]
                nh = self._mem_heads
                hd = self._mem_head_dim
                cur_len = x.shape[1]  # seq_len+1 (includes value token)

                q = attn['q'](x).view(batch_size, cur_len, nh, hd).transpose(1, 2)
                mem = memory.unsqueeze(0).expand(batch_size, -1, -1)
                k = attn['k'](mem).view(batch_size, n_mem, nh, hd).transpose(1, 2)
                v = attn['v'](mem).view(batch_size, n_mem, nh, hd).transpose(1, 2)

                out = F.scaled_dot_product_attention(q, k, v)
                out = out.transpose(1, 2).contiguous().view(batch_size, cur_len, -1)
                out = attn['o'](out)

                x = x + gate * out

        # Split: value token hidden state + rest
        value_hidden = x[:, 0, :]      # (batch, hidden) — deep value state
        output = x[:, 1:, :]           # (batch, seq_len, hidden) — normal output
        return output, value_hidden

    def forward(self, input_ids, labels=None, n_passes=None,
                return_info=False):
        """Forward pass with memory and optional settling.

        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) — targets for loss
            n_passes: number of settling passes (None = adaptive up to max_passes)
            return_info: if True, return extra info dict
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        use_adaptive = n_passes is None
        max_steps = self.max_passes if use_adaptive else n_passes

        # Ensure memory buffer is on correct device
        self.memory_buffer.to(device=device)

        # Read persistent memory (from previous chunks)
        persistent_memory = self.memory_buffer.read()

        # Settling loop
        # During settling: temporary extraction (overwritten each pass)
        # After settling: only final extraction stored persistently
        halt_budget = 1.0
        accumulated_logits = None
        actual_passes = 0
        all_confidences = []
        current_extraction = None  # temporary, refined each pass

        for pass_idx in range(max_steps):
            # Combine persistent memory + current extraction for this pass
            if current_extraction is not None and persistent_memory is not None:
                ext = current_extraction[0].detach()
                if ext.dim() == 3:
                    ext = ext.squeeze(0)
                # Include value token hidden state so model can see its own value
                val_vec = current_extraction[2].detach()  # value token hidden (hidden_size)
                if val_vec.dim() == 1:
                    val_vec = val_vec.unsqueeze(0)  # (1, hidden)
                visible_memory = torch.cat([persistent_memory, ext, val_vec], dim=0)
            elif current_extraction is not None:
                ext = current_extraction[0].detach()
                if ext.dim() == 3:
                    ext = ext.squeeze(0)
                val_vec = current_extraction[2].detach()
                if val_vec.dim() == 1:
                    val_vec = val_vec.unsqueeze(0)
                visible_memory = torch.cat([ext, val_vec], dim=0)
            else:
                visible_memory = persistent_memory  # None or (n, hidden)

            # Forward through Qwen with visible memory (value token processed alongside)
            hidden, value_hidden = self._forward_with_memory(input_ids, visible_memory)

            # Compute value: value token (deep state) + last position (action)
            action_hidden = hidden[:, -1, :]
            value_input = torch.cat([value_hidden, action_hidden], dim=-1)
            value = self.value_mlp(value_input).squeeze(-1)

            # Extract memory + priority + confidence
            extracted_memory, priority, confidence, _, value_vec = self.extractor(hidden)
            # Store value_vec from value token (richer than extraction query)
            current_extraction = (extracted_memory, priority, value_hidden)

            actual_passes += 1

            # Compute logits
            step_logits = self.lm_head(self.norm(hidden))

            if use_adaptive:
                conf = torch.min(confidence.float(),
                    torch.tensor(halt_budget, device=device, dtype=torch.float32))
                all_confidences.append(conf.item())

                if accumulated_logits is None:
                    accumulated_logits = conf * step_logits.float()
                else:
                    accumulated_logits = accumulated_logits + conf * step_logits.float()

                halt_budget -= conf.item()
                if halt_budget < 0.01:
                    break
            else:
                accumulated_logits = step_logits

        # Distribute remaining budget
        if use_adaptive and halt_budget > 0.01:
            accumulated_logits = accumulated_logits + halt_budget * step_logits.float()

        # Store ONLY the final extraction in persistent buffer
        if current_extraction is not None:
            final_memory, final_priority, _ = current_extraction
            self.memory_buffer.store(final_memory, final_priority)

        logits = accumulated_logits

        # Loss
        loss = None
        if labels is not None:
            if labels.shape[-1] == logits.shape[-2]:
                flat_logits = logits.view(-1, logits.size(-1))
                flat_labels = labels.view(-1)
            else:
                flat_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                flat_labels = labels[..., 1:].contiguous().view(-1)

            lm_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
            ponder_cost = actual_passes / self.max_passes
            loss = lm_loss + 0.01 * ponder_cost

        if return_info:
            return logits, loss, {
                "n_passes": actual_passes,
                "confidences": all_confidences,
                "memory_used": self.memory_buffer.n_stored,
                "value": value.item() if isinstance(value, torch.Tensor) else 0.0,
            }
        return logits, loss

    def generate(self, input_ids, max_new_tokens=100, n_passes=1,
                 temperature=0.0):
        """Generate tokens autoregressively."""
        generated = []
        current_ids = input_ids

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = self.forward(current_ids, n_passes=n_passes)
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
