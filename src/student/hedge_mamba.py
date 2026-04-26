"""HedgeMamba: paper-faithful implementation of arXiv:2604.14191.

Three listings from the paper appendix, all implemented here:

  Listing 3 — HedgehogProjection
    φ(x) = softmax([Wx, −Wx], dim=-1)
    Doubles the per-head dimension (D_h → 2·D_h).
    Uses softmax on the embedding dimension (NOT sequence dim)
    for numerical stability over vanilla exponentiation.

  Listing 2 — HedgeMambaMixer
    Five key differences from vanilla Mamba:
      (i)   B, C produced by Hedgehog (φ replaces linear Q/K projection)
      (ii)  Separate v_proj recovers the value branch of linear attention
      (iii) Normalization duplication: cat([V, ones]) → divide numerator
            by denominator at the end, mimicking the attention denominator
      (iv)  Input-dependent Δt via dt_proj + softplus (full Mamba ZOH)
      (v)   SiLU gate (not sigmoid)

  Listing 1 — HedgeMambaLayer
    Drop-in replacement for the attention sub-block in a Transformer layer.
    For Whisper we expose WhisperHedgeMambaLayer (in whisper_mamba.py)
    which wraps HedgeMambaMixer and handles the Whisper attention interface.

Note on state size:
    The paper sets state_size = hidden_size so that after Hedgehog doubling
    the effective state N = 2·hidden_size, matching the attention key/value
    space.  This gives a large state (B, 2D, 2D) during the scan but is
    intentional — it lets the SSM fully mimic attention capacity.
    A smaller configurable state_size is also supported for ablations.

Note on RoPE:
    The paper applies rotary position embeddings to B and C.  We omit RoPE
    here because Whisper's decoder already carries position embeddings in its
    own embed_positions layer, so adding another positional encoding on top
    would be double-counting.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Listing 3 — Hedgehog projection
# ─────────────────────────────────────────────────────────────────────────────

class HedgehogProjection(nn.Module):
    """φ(x) = softmax([Wx, −Wx], dim=-1)   (Listing 3).

    Applied per attention head.

    Args:
        head_dim: per-head embedding size D_h.

    Input:  (B, L, H, D_h)
    Output: (B, L, H, 2·D_h)   ← dimension doubled by the negation trick
    """

    def __init__(self, head_dim: int, bias: bool = True):
        super().__init__()
        self.phi = nn.Linear(head_dim, head_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.phi(x)                        # (B, L, H, D_h)
        x = torch.cat([x, -x], dim=-1)         # (B, L, H, 2·D_h)
        return x.softmax(dim=-1)               # softmax on embedding dim


# ─────────────────────────────────────────────────────────────────────────────
#  Listing 2 — HedgeMamba mixer
# ─────────────────────────────────────────────────────────────────────────────

class HedgeMambaMixer(nn.Module):
    """Paper-faithful HedgeMamba sequence mixer (Listing 2).

    Args:
        dim:        model hidden size D.
        num_heads:  number of attention heads H (used for per-head Hedgehog).
        dt_rank:    rank of the Δt projection (default D//16, as in Mamba).
        state_size: SSM state per channel BEFORE Hedgehog doubling.
                    Paper sets this to D so effective state = 2·D after
                    Hedgehog.  Smaller values trade quality for speed.
    """

    def __init__(
        self,
        dim:        int,
        num_heads:  int,
        dt_rank:    int | None = None,
        state_size: int | None = None,
    ):
        super().__init__()
        self.dim        = dim
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.dt_rank    = dt_rank    or max(1, dim // 16)
        # N = state size per channel *before* Hedgehog.
        # After Hedgehog the effective B/C dimension is 2·N,
        # and after normalization duplication the scan runs over 2·D channels.
        self.state_size = state_size or dim   # paper: state_size = dim

        N  = self.state_size          # shorthand
        D  = dim
        Ns = 2 * N                    # effective state after Hedgehog doubling

        # ── (i) Gate — SiLU (paper Listing 2) ────────────────────────────
        self.gate_proj = nn.Linear(D, D, bias=False)

        # ── Causal depthwise conv (same as vanilla Mamba) ─────────────────
        self.conv1d = nn.Conv1d(D, D, kernel_size=4, padding=3,
                                groups=D, bias=False)

        # ── SSM param projections: x → [dt | B_raw | C_raw] ──────────────
        # B_raw and C_raw each have size N (= state_size).
        # They are passed through a single "virtual head" HedgehogProjection
        # (head_dim = N) which doubles them to Ns = 2·N.
        # Paper sets N = D (hidden_size); smaller N is a practical speedup.
        self.x_proj = nn.Linear(D, self.dt_rank + N * 2, bias=False)

        # dt: dt_rank → D, softplus for strict positivity
        self.dt_proj = nn.Linear(self.dt_rank, D, bias=True)
        nn.init.constant_(self.dt_proj.bias, -4.0)  # small initial dt → long memory

        # ── A: diagonal state matrix, log-parameterised ───────────────────
        # Shape (D, Ns) — D channels each with Ns-dim state (Ns = 2·N).
        # log(arange(1, Ns+1)) gives A values from −1 to −Ns after negation,
        # spreading time scales from ~1 step to ~Ns steps.
        A_init = torch.arange(1, Ns + 1, dtype=torch.float32)
        A_init = A_init.unsqueeze(0).expand(D, -1)    # (D, Ns)
        self.A_log = nn.Parameter(torch.log(A_init))   # (D, Ns)
        self.A_log._no_weight_decay = True

        # ── (ii) Value projection ─────────────────────────────────────────
        self.v_proj = nn.Linear(D, D, bias=False)

        # ── (i) Hedgehog projections — one virtual head of size N ─────────
        # head_dim = N so HedgehogProjection doubles N → 2N = Ns.
        # Paper uses head_dim = D/num_heads; equivalent when N = D.
        self.hhog_k = HedgehogProjection(N)   # B ≡ keys
        self.hhog_q = HedgehogProjection(N)   # C ≡ queries

        # ── Output projection ─────────────────────────────────────────────
        self.out_proj = nn.Linear(D, D, bias=False)

    # ── Causal conv (handles training path + Fix-B inference conv cache) ──

    def _apply_conv(
        self,
        x: torch.Tensor,
        conv_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        k = self.conv1d.kernel_size[0]
        if conv_cache is not None:
            x_ctx  = torch.cat([conv_cache, x], dim=1)   # (B, k-1+L, D)
            x_conv = F.conv1d(
                x_ctx.transpose(1, 2),
                self.conv1d.weight,
                bias=self.conv1d.bias,
                groups=D,
            ).transpose(1, 2)                             # (B, L, D)
        else:
            x_ctx  = x
            x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        new_conv_cache = x_ctx[:, -(k - 1):, :].detach()
        return x_conv, new_conv_cache

    # ── Selective scan (ZOH discretization) ──────────────────────────────

    def _selective_scan(
        self,
        u:      torch.Tensor,           # (B, L, 2D) — V_dup
        dt:     torch.Tensor,           # (B, L, 2D) — dt_dup
        A:      torch.Tensor,           # (2D, Ns)   — A_dup, negative
        B:      torch.Tensor,           # (B, L, Ns) — after Hedgehog
        C:      torch.Tensor,           # (B, L, Ns) — after Hedgehog
        h_prev: torch.Tensor | None,    # (B, 2D, Ns) or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (y, h_final).

        y:       (B, L, 2D)
        h_final: (B, 2D, Ns)  — cache for Fix-B single-step generation

        ZOH discretization:
            dA[t]  = exp(Δ[t] · A)
            dBu[t] = (Δ[t] · u[t]) ⊗ B[t]
            h[t]   = dA[t] · h[t-1] + dBu[t]
            y[t]   = C[t] · h[t]

        All dA / dBu tensors are pre-computed in vectorised form (one batched
        exp + two broadcasts) to minimise Python overhead before the loop.
        On CUDA this can be replaced by a fused Triton scan kernel; on MPS
        torch.compile fuses the loop and reduces dispatch overhead.
        """
        Bb, L, D2 = u.shape
        Ns = B.shape[-1]

        # NOTE on pre-computing dA/dBu upfront:
        # Allocating (B, L, D2, Ns) tensors then reading them back L times
        # is slower on MPS than computing per-step (B, D2, Ns) slices that
        # fit in cache.  The proper fix is a fused Metal/Triton kernel.

        h = h_prev if h_prev is not None else u.new_zeros(Bb, D2, Ns)
        outputs = []
        for t in range(L):
            # dA: (B, D2, Ns)
            dA  = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))
            # dBu: (B, D2, Ns)
            dBu = (dt[:, t] * u[:, t]).unsqueeze(-1) * B[:, t].unsqueeze(1)
            h   = dA * h + dBu                                 # (B, D2, Ns)
            outputs.append((h * C[:, t].unsqueeze(1)).sum(-1)) # (B, D2)

        return torch.stack(outputs, dim=1), h   # (B, L, D2), (B, D2, Ns)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x:          torch.Tensor,
        h_prev:     torch.Tensor | None = None,   # (B, 2D, Ns) SSM state cache
        conv_cache: torch.Tensor | None = None,   # (B, k-1, D) conv cache
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            y:              (B, L, D)
            h_new:          (B, 2D, Ns)  — updated SSM state
            new_conv_cache: (B, k-1, D)  — updated conv cache
        """
        B, L, D = x.shape
        N  = self.state_size
        Ns = 2 * N    # effective state after Hedgehog doubling

        # 1. SiLU gate (paper: F.silu)
        gate = F.silu(self.gate_proj(x))                      # (B, L, D)

        # 2. Causal conv (with Fix-B conv cache)
        x_conv, new_conv_cache = self._apply_conv(x, conv_cache)   # (B, L, D)

        # 3. Project → [dt_raw | B_raw | C_raw]
        dt_raw, B_raw, C_raw = torch.split(
            self.x_proj(x_conv),
            [self.dt_rank, N, N], dim=-1,
        )   # (B,L,dt_rank), (B,L,N), (B,L,N)

        # 4. Δt: dt_rank → D, softplus for strict positivity
        dt = F.softplus(self.dt_proj(dt_raw))                 # (B, L, D)

        # 5. Hedgehog on B (≡ K) and C (≡ Q) via one virtual head of size N
        #    B_raw: (B, L, N) → unsqueeze → (B, L, 1, N) → Hedgehog → (B, L, 1, 2N=Ns)
        #    → squeeze → (B, L, Ns)
        B_feat = self.hhog_k(B_raw.unsqueeze(2)).squeeze(2)   # (B, L, Ns)
        C_feat = self.hhog_q(C_raw.unsqueeze(2)).squeeze(2)   # (B, L, Ns)

        # 6. Value projection (separate from SSM params)
        V = self.v_proj(x_conv)                               # (B, L, D)

        # 7. Normalization duplication (paper Listing 2, eq. 10)
        #    Running one scan with doubled channels gives numerator + denominator
        #    in a single pass.  Second half of V is ones → tracks denominator.
        A     = -self.A_log.exp().float()                     # (D, Ns), negative
        A_dup  = torch.cat([A,  A ], dim=0)                   # (2D, Ns)
        dt_dup = torch.cat([dt, dt], dim=-1)                  # (B, L, 2D)
        V_dup  = torch.cat([V, torch.ones_like(V)], dim=-1)   # (B, L, 2D)

        # h_prev covers the full doubled state
        h_prev_dup = h_prev   # (B, 2D, Ns) or None

        # 8. Selective scan (ZOH)
        y_dup, h_new = self._selective_scan(
            V_dup, dt_dup, A_dup, B_feat, C_feat, h_prev_dup
        )   # (B, L, 2D), (B, 2D, Ns)

        # 9. Attention-style normalization: num / den
        y_num = y_dup[..., :D]                                # (B, L, D)
        y_den = y_dup[..., D:]                                # (B, L, D)
        y = gate * y_num / (y_den.abs() + 1e-6)

        # 10. Output projection
        return self.out_proj(y), h_new, new_conv_cache         # (B,L,D), (B,2D,Ns), (B,k-1,D)


# ─────────────────────────────────────────────────────────────────────────────
#  Listing 1 — HedgeMamba block (GPTNeoX / Pythia variant, kept for reference)
# ─────────────────────────────────────────────────────────────────────────────

class HedgeMambaLayer(nn.Module):
    """Drop-in for a GPTNeoX attention sub-block (Listing 1).

    stage='ssm' → HedgeMambaMixer  (paper Stage 2)
    Not used in the Whisper pipeline — see WhisperHedgeMambaLayer instead.
    """

    def __init__(self, dim: int, num_heads: int, state_size: int | None = None):
        super().__init__()
        self.mixer = HedgeMambaMixer(dim, num_heads, state_size=state_size)

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, None]:
        out, _, _ = self.mixer(x)
        return out, None
