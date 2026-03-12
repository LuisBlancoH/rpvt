"""Fast weight memory module.

A linear associative memory that updates on every forward pass.
No backprop at inference — the write rule is learned during pretraining
and frozen at deployment.

Memory update per token:
    key = W_key @ hidden_state
    value = W_value @ hidden_state
    M = decay * M + write_strength * outer(value, key)

Write modes:
    "uniform":  write_strength = 1 (all tokens write equally)
    "gate":     write_strength = sigmoid(W_gate @ hidden_state) (learned gate)
    "surprise": write_strength = sigmoid(scale * z + bias)
                where z = (||error|| - mean) / std (z-scored prediction error)
                error = hidden_state - W_out(M @ query)
                Surprise-driven: write strongly when M's prediction is wrong.
                Running stats normalize errors so sigmoid inputs are always useful.

Memory retrieval per token:
    query = W_query @ hidden_state
    retrieved = M @ query
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastWeightMemory(nn.Module):
    """Single-timescale linear fast weight memory.

    Maintains a matrix M that accumulates outer products of (value, key) pairs.
    Retrieval is linear: M @ query.
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 256,
        decay: float = 0.99,
        write_mode: str = "uniform",
        max_m_norm: float = 10.0,
        chunk_agg: str = "token",
        aux_predict: bool = False,
        contrastive: bool = False,
        mask_frac: float = 0.0,
        bptt_steps: int = 0,
        # Legacy compat
        use_write_gate: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.decay = decay
        self.max_m_norm = max_m_norm
        self.chunk_agg = chunk_agg  # "token", "mean", "last", "surprise", "learned"
        self.aux_predict = aux_predict
        self.contrastive = contrastive
        self.mask_frac = mask_frac
        self.bptt_steps = bptt_steps  # 0 = always detach (default)
        self._bptt_counter = 0

        # Handle legacy use_write_gate parameter
        if use_write_gate and write_mode == "uniform":
            write_mode = "gate"
        self.write_mode = write_mode

        # Projections: hidden_size -> memory_size
        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
        self.W_key = nn.Linear(hidden_size, memory_size, bias=False)
        self.W_value = nn.Linear(hidden_size, memory_size, bias=False)

        # Output projection: memory_size -> hidden_size
        # Zero-initialized so memory starts with no contribution,
        # and the model learns to use it through W_out weights directly.
        self.W_out = nn.Linear(memory_size, hidden_size, bias=False)
        nn.init.zeros_(self.W_out.weight)

        # Write gate: learns which tokens are worth writing to M
        if write_mode == "gate":
            self.use_write_gate = True  # legacy compat
            self.W_gate = nn.Linear(hidden_size, 1, bias=True)
            nn.init.zeros_(self.W_gate.weight)
            nn.init.constant_(self.W_gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12
        else:
            self.use_write_gate = write_mode == "gate"

        # Predictive/subtract modes: W_nudge projects current state for delta computation
        # output = W_out(retrieved) - W_nudge(x)  → nudge toward predicted state
        # "predictive": future values + subtraction (both changes)
        # "subtract-only": current values + subtraction (isolates subtraction effect)
        if write_mode in ("predictive", "subtract-only"):
            self.W_nudge = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.W_nudge.weight)

        # Surprise-driven write strength (z-score normalized)
        if write_mode in ("surprise", "surprise-fwd", "surprise-fwd-store"):
            # Write strength = sigmoid(scale * z + bias)
            # where z = (||error|| - chunk_mean) / (chunk_std + eps)
            # z is computed per-chunk so tokens within the same chunk
            # get different write strengths based on relative surprise.
            #
            # "surprise": error = current chunk - M's prediction (backward-looking)
            # "surprise-fwd": error = NEXT chunk - M's prediction (forward-looking)
            #   Forward surprise writes strongly when M can't predict what's coming,
            #   forcing M to store forward-predictive content rather than adapter effects.
            # "surprise-fwd-store": like surprise-fwd, but also stores the NEXT chunk's
            #   hidden state as the value (instead of current chunk). This makes M a
            #   transition model: query with current state, retrieve future state.
            #   Stored content doesn't go stale (unlike storing deltas/errors).
            #
            # scale controls sensitivity: higher = more differentiation
            # bias controls baseline: negative = write less on average
            self.surprise_scale = 2.0   # hyperparameter, not learned
            self.surprise_bias = -0.5   # sigmoid(-0.5) ≈ 0.38 at mean error

        # Learned attention pooling for chunk aggregation
        if chunk_agg == "learned":
            self.W_agg_score = nn.Linear(hidden_size, 1, bias=True)
            nn.init.zeros_(self.W_agg_score.weight)
            nn.init.zeros_(self.W_agg_score.bias)

        # Auxiliary cross-chunk prediction head
        if aux_predict:
            self.W_aux_pred = nn.Linear(memory_size, hidden_size, bias=False)
            nn.init.zeros_(self.W_aux_pred.weight)

        # Contrastive projection (maps retrieval to a space for similarity)
        if contrastive:
            self.W_contrast = nn.Linear(memory_size, memory_size, bias=False)
            nn.init.normal_(self.W_contrast.weight, std=0.02)

        # NaN debugging flag — set to True to enable per-chunk NaN tracing
        self._nan_debug = False

        # Memory matrix — not a parameter, not saved, starts at zero
        self.register_buffer("M", torch.zeros(memory_size, memory_size))

    def reset_memory(self):
        """Reset memory to zero."""
        self.M.zero_()

    def forward(self, x: torch.Tensor, chunk_size: int = 64) -> tuple[torch.Tensor, float | None, dict]:
        """Process a sequence in chunks, updating memory between chunks.

        Within each chunk, all tokens read from the same M state (beginning of chunk).
        After the chunk, M is updated with all writes from that chunk.

        Args:
            x: (batch, seq_len, hidden_size)
            chunk_size: Number of tokens to process at once.

        Returns:
            output: (batch, seq_len, hidden_size) — memory output to add to residual
            mean_ws: mean write strength for logging (None if uniform)
            aux_losses: dict of auxiliary losses (empty if none enabled)
        """
        batch, seq_len, _ = x.shape

        query = self.W_query(x)   # (batch, seq_len, memory_size)
        key = self.W_key(x)       # (batch, seq_len, memory_size)
        value = self.W_value(x)   # (batch, seq_len, memory_size)

        # L2-normalize keys and values to prevent M from exploding
        key = F.normalize(key, dim=-1)
        value = F.normalize(value, dim=-1)

        # Pre-compute gate write strengths (only for "gate" mode)
        if self.write_mode == "gate":
            gate_strengths = torch.sigmoid(self.W_gate(x))  # (batch, seq_len, 1)

        retrieved_chunks = []
        write_strength_log = []
        aux_pred_losses = []
        contrastive_anchors = []
        contrastive_targets = []
        M = self.M.clone()  # (memory_size, memory_size)

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            c_len = chunk_end - chunk_start

            # ── READ: all tokens in this chunk read from current M ──
            q_chunk = query[:, chunk_start:chunk_end, :]   # (batch, c_len, memory_size)
            r_chunk = torch.matmul(q_chunk, M.T)           # (batch, c_len, memory_size)
            retrieved_chunks.append(r_chunk)

            # ── NaN check: after read ──
            if self._nan_debug and (torch.isnan(r_chunk).any() or torch.isinf(r_chunk).any()):
                print(f"[NaN DEBUG] r_chunk has nan/inf at chunk_start={chunk_start}, "
                      f"M_norm={M.norm().item():.4f}, q_norm={q_chunk.norm().item():.4f}")

            # ── AUX: cross-chunk prediction loss ──
            if self.aux_predict and chunk_end < seq_len:
                next_start = chunk_end
                next_end = min(next_start + c_len, seq_len)
                if next_end > next_start:
                    r_mean = r_chunk.mean(dim=1)  # (batch, memory_size)
                    pred_next = self.W_aux_pred(r_mean)  # (batch, hidden_size)
                    actual_next = x[:, next_start:next_end, :].mean(dim=1).detach()
                    aux_pred_losses.append(F.mse_loss(pred_next, actual_next))

            # ── AUX: contrastive anchors ──
            if self.contrastive:
                r_proj = self.W_contrast(r_chunk.mean(dim=1))  # (batch, memory_size)
                r_proj = F.normalize(r_proj, dim=-1)
                contrastive_anchors.append(r_proj)
                # Target: next chunk's actual hidden state projected to same space
                if chunk_end < seq_len:
                    next_start = chunk_end
                    next_end = min(next_start + c_len, seq_len)
                    actual_mean = x[:, next_start:next_end, :].mean(dim=1).detach()
                    target_proj = F.normalize(self.W_contrast(self.W_query(actual_mean)), dim=-1)
                    contrastive_targets.append(target_proj)
                else:
                    contrastive_targets.append(None)

            # ── COMPUTE WRITE STRENGTH ──
            k_chunk = key[:, chunk_start:chunk_end, :]     # (batch, c_len, memory_size)
            v_chunk = value[:, chunk_start:chunk_end, :]   # (batch, c_len, memory_size)

            if self.write_mode == "uniform":
                pass  # no scaling

            elif self.write_mode in ("predictive", "future-only"):
                # Store future states: value = W_value(next_chunk_hidden)
                next_start = min(chunk_end, seq_len - c_len)
                next_end = next_start + c_len
                if next_end <= seq_len and next_start != chunk_start:
                    future_x = x[:, next_start:next_end, :]
                    v_chunk = self.W_value(future_x)
                    v_chunk = F.normalize(v_chunk, dim=-1)

            elif self.write_mode == "subtract-only":
                pass  # use current chunk values (same as uniform), subtraction applied at output

            elif self.write_mode == "gate":
                ws_chunk = gate_strengths[:, chunk_start:chunk_end, :]
                k_chunk = k_chunk * ws_chunk
                v_chunk = v_chunk * ws_chunk
                write_strength_log.append(ws_chunk.mean().item())

            elif self.write_mode in ("surprise", "surprise-fwd", "surprise-fwd-store"):
                # Prediction: what does M think the hidden state should be?
                prediction = self.W_out(r_chunk)  # (batch, c_len, hidden_size)

                # Determine the target for surprise comparison
                if self.write_mode == "surprise":
                    # Backward-looking: compare prediction to current chunk
                    actual = x[:, chunk_start:chunk_end, :]  # (batch, c_len, hidden_size)
                else:
                    # Forward-looking: compare prediction to NEXT chunk
                    # M should write strongly when it can't predict what's coming
                    next_start = min(chunk_end, seq_len - c_len)
                    next_end = next_start + c_len
                    if next_end <= seq_len and next_start != chunk_start:
                        actual = x[:, next_start:next_end, :]
                    else:
                        # Last chunk: fall back to current chunk
                        actual = x[:, chunk_start:chunk_end, :]

                # For surprise-fwd-store: also store next chunk as the value
                # This makes M a transition model: query with current → retrieve future
                if self.write_mode == "surprise-fwd-store":
                    next_start_v = min(chunk_end, seq_len - c_len)
                    next_end_v = next_start_v + c_len
                    if next_end_v <= seq_len and next_start_v != chunk_start:
                        v_chunk = self.W_value(actual)  # project next chunk's hidden state
                        v_chunk = F.normalize(v_chunk, dim=-1)
                    # else: keep v_chunk as current chunk (already set above)

                error = actual - prediction

                # ── NaN check: surprise prediction path ──
                if self._nan_debug and (torch.isnan(prediction).any() or torch.isinf(prediction).any()):
                    print(f"[NaN DEBUG] prediction has nan/inf at chunk_start={chunk_start}, "
                          f"r_chunk_norm={r_chunk.norm().item():.4f}, "
                          f"W_out_norm={self.W_out.weight.norm().item():.4f}")
                if self._nan_debug and (torch.isnan(error).any() or torch.isinf(error).any()):
                    print(f"[NaN DEBUG] error has nan/inf at chunk_start={chunk_start}, "
                          f"pred_norm={prediction.norm().item():.4f}, "
                          f"actual_norm={actual.norm().item():.4f}")

                # Compute z-score in float32 to avoid bfloat16 precision issues
                # (std can underflow to 0 in bf16 when error norms are similar)
                error_norm = error.float().norm(dim=-1, keepdim=True)  # (batch, c_len, 1)

                # Z-score within this chunk: tokens compete on relative surprise
                chunk_mean = error_norm.mean().detach()
                chunk_std = error_norm.std().detach().clamp(min=1e-4)
                z = (error_norm - chunk_mean) / chunk_std
                z = z.clamp(-5.0, 5.0)  # prevent extreme z-scores

                # ── NaN check: z-score path ──
                if self._nan_debug and (torch.isnan(z).any() or torch.isinf(z).any()):
                    print(f"[NaN DEBUG] z has nan/inf at chunk_start={chunk_start}, "
                          f"error_norm range=[{error_norm.min().item():.4f}, {error_norm.max().item():.4f}], "
                          f"chunk_mean={chunk_mean.item():.4f}, chunk_std={chunk_std.item():.6f}")

                # Map z-score to write strength
                # z=0 (average surprise) → sigmoid(-0.5) ≈ 0.38
                # z=+1 (1 std above, surprising) → sigmoid(1.5) ≈ 0.82
                # z=-1 (1 std below, expected) → sigmoid(-2.5) ≈ 0.08
                ws_chunk = torch.sigmoid(
                    self.surprise_scale * z + self.surprise_bias
                ).to(x.dtype)  # back to model dtype

                # ── NaN check: write strength ──
                if self._nan_debug and (torch.isnan(ws_chunk).any() or torch.isinf(ws_chunk).any()):
                    print(f"[NaN DEBUG] ws_chunk has nan/inf at chunk_start={chunk_start}, "
                          f"z range=[{z.min().item():.4f}, {z.max().item():.4f}]")

                k_chunk = k_chunk * ws_chunk
                v_chunk = v_chunk * ws_chunk
                write_strength_log.append(ws_chunk.mean().item())

            # ── WRITE ──
            if self.chunk_agg == "token":
                # Token-level: sum of outer products (original behavior)
                chunk_outer = torch.einsum("bci,bcj->ij", v_chunk, k_chunk) / (batch * c_len)
            else:
                # Chunk-level: aggregate tokens into single key-value, one outer product
                if self.chunk_agg == "mean":
                    k_agg = k_chunk.mean(dim=1)  # (batch, memory_size)
                    v_agg = v_chunk.mean(dim=1)
                elif self.chunk_agg == "last":
                    k_agg = k_chunk[:, -1, :]    # (batch, memory_size)
                    v_agg = v_chunk[:, -1, :]
                elif self.chunk_agg == "surprise":
                    # Weight by surprise scores (reuse ws_chunk if available)
                    if write_strength_log:  # surprise mode computed ws_chunk
                        weights = ws_chunk / (ws_chunk.sum(dim=1, keepdim=True) + 1e-8)
                        k_agg = (k_chunk * weights).sum(dim=1)
                        v_agg = (v_chunk * weights).sum(dim=1)
                    else:
                        # Fallback to mean if no surprise scores
                        k_agg = k_chunk.mean(dim=1)
                        v_agg = v_chunk.mean(dim=1)
                elif self.chunk_agg == "learned":
                    x_chunk = x[:, chunk_start:chunk_end, :]
                    scores = self.W_agg_score(x_chunk)  # (batch, c_len, 1)
                    weights = torch.softmax(scores, dim=1)  # (batch, c_len, 1)
                    k_agg = (k_chunk * weights).sum(dim=1)
                    v_agg = (v_chunk * weights).sum(dim=1)
                else:
                    raise ValueError(f"Unknown chunk_agg: {self.chunk_agg}")

                k_agg = F.normalize(k_agg, dim=-1)
                v_agg = F.normalize(v_agg, dim=-1)
                chunk_outer = torch.einsum("bi,bj->ij", v_agg, k_agg) / batch

            M = (self.decay ** c_len) * M + chunk_outer

            # ── NaN check: after write ──
            if self._nan_debug and (torch.isnan(chunk_outer).any() or torch.isinf(chunk_outer).any()):
                print(f"[NaN DEBUG] chunk_outer has nan/inf at chunk_start={chunk_start}, "
                      f"v_norm={v_chunk.norm().item():.4f}, k_norm={k_chunk.norm().item():.4f}")
            if self._nan_debug and (torch.isnan(M).any() or torch.isinf(M).any()):
                print(f"[NaN DEBUG] M has nan/inf after write at chunk_start={chunk_start}, "
                      f"chunk_outer_norm={chunk_outer.norm().item():.4f}")

            # Cap M norm to prevent explosion at slow decay rates
            if self.max_m_norm > 0:
                m_norm = M.norm()
                if m_norm > self.max_m_norm:
                    M = M * (self.max_m_norm / m_norm)

        # Store updated memory — detach unless BPTT is active
        if self.bptt_steps > 0 and self._bptt_counter < self.bptt_steps:
            self.M = M  # keep gradient graph for BPTT
            self._bptt_counter += 1
        else:
            self.M = M.detach()
            self._bptt_counter = 0

        retrieved = torch.cat(retrieved_chunks, dim=1)  # (batch, seq_len, memory_size)

        if self.write_mode in ("predictive", "subtract-only"):
            # Nudge: delta between M's prediction and current state
            prediction = self.W_out(retrieved)     # predicted future state
            current = self.W_nudge(x)              # current state representation
            output = prediction - current          # nudge toward predicted state
        else:
            output = self.W_out(retrieved)  # (batch, seq_len, hidden_size)

        # Clamp memory output to prevent destabilizing the residual stream.
        # At high decay rates, W_out * (M @ query) can produce outputs large
        # enough to blow up the transformer's hidden states.
        max_output_norm = 10.0
        output_norm = output.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        output = output * (max_output_norm / output_norm).clamp(max=1.0)

        # ── Compute auxiliary losses ──
        aux_losses = {}

        if aux_pred_losses:
            aux_losses["aux_predict"] = sum(aux_pred_losses) / len(aux_pred_losses)

        if contrastive_anchors and len(contrastive_anchors) > 1:
            # InfoNCE-style: for each chunk i, retrieval should be similar to
            # next chunk (positive) and dissimilar to other chunks (negatives)
            c_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
            n_pairs = 0
            temperature = 0.1
            for i in range(len(contrastive_anchors) - 1):
                if contrastive_targets[i] is None:
                    continue
                anchor = contrastive_anchors[i]  # (batch, memory_size)
                positive = contrastive_targets[i]  # (batch, memory_size)
                # Negatives: all other anchors
                neg_indices = [j for j in range(len(contrastive_anchors)) if j != i]
                if not neg_indices:
                    continue
                negatives = torch.stack([contrastive_anchors[j] for j in neg_indices], dim=1)
                # (batch, n_neg, memory_size)
                pos_sim = (anchor * positive).sum(dim=-1) / temperature  # (batch,)
                neg_sim = torch.einsum("bd,bnd->bn", anchor, negatives) / temperature
                # (batch, n_neg)
                logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch, 1+n_neg)
                labels = torch.zeros(batch, device=x.device, dtype=torch.long)
                c_loss = c_loss + F.cross_entropy(logits.float(), labels)
                n_pairs += 1
            if n_pairs > 0:
                aux_losses["contrastive"] = c_loss / n_pairs

        mean_ws = sum(write_strength_log) / len(write_strength_log) if write_strength_log else None
        return output, mean_ws, aux_losses


class TransformerLayerWithMemory(nn.Module):
    """Wraps an existing transformer layer, adding fast weight memory.

    The forward pass:
        1. Run the original transformer layer
        2. Read from fast weight memory using the layer's output
        3. Add gated memory output to the residual stream
        4. Write to fast weight memory
    """

    def __init__(self, original_layer: nn.Module, hidden_size: int, memory_size: int = 256,
                 decay: float = 0.99, write_mode: str = "uniform", max_m_norm: float = 10.0,
                 chunk_agg: str = "token", aux_predict: bool = False, contrastive: bool = False,
                 mask_frac: float = 0.0, bptt_steps: int = 0, use_write_gate: bool = False):
        super().__init__()
        self.layer = original_layer
        self.mask_frac = mask_frac
        self.memory = FastWeightMemory(
            hidden_size=hidden_size,
            memory_size=memory_size,
            decay=decay,
            write_mode=write_mode,
            max_m_norm=max_m_norm,
            chunk_agg=chunk_agg,
            aux_predict=aux_predict,
            contrastive=contrastive,
            mask_frac=mask_frac,
            bptt_steps=bptt_steps,
            use_write_gate=use_write_gate,
        )
        # Accumulate aux losses across layers during forward pass
        self._aux_losses = {}

    def forward(self, *args, **kwargs):
        """Run the original layer, then add memory output."""
        # Run original transformer layer
        outputs = self.layer(*args, **kwargs)

        # Handle both tuple outputs (Qwen) and direct tensor outputs (GPT-2)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        # Read from and write to memory
        memory_output, gate_value, aux_losses = self.memory(hidden_states)
        self._aux_losses = aux_losses

        # Masking: for a fraction of tokens, zero out transformer hidden states
        # so that only M's output contributes. Forces M to carry cross-chunk info.
        if self.mask_frac > 0 and self.training:
            batch, seq_len = hidden_states.shape[:2]
            mask = (torch.rand(batch, seq_len, 1, device=hidden_states.device) < self.mask_frac)
            mask = mask.to(hidden_states.dtype)
            modified = hidden_states * (1 - mask) + memory_output
        else:
            modified = hidden_states + memory_output

        # Return in same format as original layer
        if isinstance(outputs, tuple):
            return (modified,) + outputs[1:]
        else:
            return modified

    def reset_memory(self):
        self.memory.reset_memory()


def attach_fast_weight_memory(layers, hidden_size, layer_indices=None, memory_size=256, decay=0.99,
                              write_mode="uniform", max_m_norm=10.0, chunk_agg="token",
                              aux_predict=False, contrastive=False, mask_frac=0.0,
                              bptt_steps=0, use_write_gate=False):
    """Attach fast weight memory modules to specified transformer layers.

    Args:
        layers: nn.ModuleList of transformer layers (e.g., model.transformer.h)
        hidden_size: Hidden dimension of the model.
        layer_indices: Which layers to add memory to. None = all.
        memory_size: Dimension of key/value/query projections.
        decay: Memory decay rate per token.
        write_mode: "uniform", "gate", "surprise", or "predictive".
        max_m_norm: Cap on M's Frobenius norm. 0 = no cap.
        chunk_agg: Chunk aggregation method: "token", "mean", "last", "surprise", "learned".
        aux_predict: Enable auxiliary cross-chunk prediction loss.
        contrastive: Enable contrastive loss for document-specific retrieval.
        mask_frac: Fraction of tokens to mask (0 = disabled).
        bptt_steps: Number of forward calls to keep M in computation graph (0 = always detach).

    Returns:
        List of FastWeightMemory modules (for parameter access).
    """
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    memory_modules = []
    for idx in layer_indices:
        original_layer = layers[idx]
        wrapped = TransformerLayerWithMemory(
            original_layer,
            hidden_size=hidden_size,
            memory_size=memory_size,
            decay=decay,
            write_mode=write_mode,
            max_m_norm=max_m_norm,
            chunk_agg=chunk_agg,
            aux_predict=aux_predict,
            contrastive=contrastive,
            mask_frac=mask_frac,
            bptt_steps=bptt_steps,
            use_write_gate=use_write_gate,
        )
        # Move to same device/dtype as model
        device = next(original_layer.parameters()).device
        dtype = next(original_layer.parameters()).dtype
        wrapped.memory = wrapped.memory.to(device=device, dtype=dtype)
        layers[idx] = wrapped
        memory_modules.append(wrapped.memory)

    return memory_modules


def remove_fast_weight_memory(layers, layer_indices=None):
    """Remove fast weight memory modules, restoring original layers."""
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    for idx in layer_indices:
        layer = layers[idx]
        if isinstance(layer, TransformerLayerWithMemory):
            layers[idx] = layer.layer


def reset_all_memories(layers, layer_indices=None):
    """Reset all fast weight memories to zero."""
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    for idx in layer_indices:
        layer = layers[idx]
        if isinstance(layer, TransformerLayerWithMemory):
            layer.reset_memory()
