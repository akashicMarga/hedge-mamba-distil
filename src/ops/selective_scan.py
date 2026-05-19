"""Optimized selective SSM scan using custom Metal compute kernels.

For MPS (Apple Silicon) training the Python for-loop in the naive scan
dispatches ~4 Metal commands per timestep, ballooning to thousands of GPU
round-trips for L=448. This module replaces that loop with single Metal
kernel dispatches via MLX's fast.metal_kernel facility.

Speedup vs. the naive PyTorch loop:
  - Forward (inference): ~5–6× (includes PT↔MLX copy)
  - Forward+Backward (training): similar speedup expected

Kernels
-------
_fwd_inf_kernel   : forward only, no h_states saved (inference)
_fwd_train_kernel : forward + saves h_states (B, L+1, D2, Ns) for backward
_bwd1_kernel      : backward recurrence — grad_dt, grad_u, g_h_states, grad_A_per_b, grad_h0
_bwd2_kernel      : parallel grad_B, grad_C from g_h_states and h_states

All kernels run in float32. Ns must be a power of 2 and ≤ 256.

Public API
----------
selective_scan(u, dt, A, B, C, h_prev) → (y, h_final)
    Drop-in replacement for HedgeMambaMixer._selective_scan.
    Falls back to the PyTorch loop when MLX is unavailable.
"""

from __future__ import annotations
try:
    import torch
    _AutogradBase = torch.autograd.Function
    _TORCH_AVAILABLE = True
except (ImportError, AttributeError):
    torch = None
    _AutogradBase = object
    _TORCH_AVAILABLE = False

_fwd_inf_kernel   = None   # inference forward (no h_states)
_fwd_train_kernel = None   # training forward (saves h_states)
_bwd1_kernel      = None   # backward recurrence
_bwd2_kernel      = None   # backward grad_B / grad_C
_MLX_AVAILABLE    = False
FORCE_PT_LOOP     = False  # set True to bypass Metal for benchmarking


# ── Kernel build ───────────────────────────────────────────────────────────────

def _try_build_kernels(Ns: int) -> bool:
    global _fwd_inf_kernel, _fwd_train_kernel, _bwd1_kernel, _bwd2_kernel, _MLX_AVAILABLE
    try:
        import mlx.core as mx
        import mlx.core.fast as mxf

        # ── Inference forward (unchanged from original) ───────────────────────
        FWD_INF_SRC = """
    uint n   = thread_position_in_threadgroup.x;
    uint bd  = threadgroup_position_in_grid.x;
    uint Ns  = threads_per_threadgroup.x;
    uint D2  = d2_size[0];
    uint L   = l_size[0];
    uint B   = b_size[0];

    uint b = bd / D2;
    uint d = bd % D2;
    if (b >= B || d >= D2 || n >= Ns) return;

    threadgroup float scratch[256];

    float A_val = A_in[d * Ns + n];
    float h = h0[b * D2 * Ns + d * Ns + n];

    for (uint t = 0; t < L; t++) {
        float dt_val = dt_in[b * L * D2 + t * D2 + d];
        float u_val  = u_in[b * L * D2 + t * D2 + d];
        float B_val  = B_in[b * L * Ns + t * Ns + n];
        float C_val  = C_in[b * L * Ns + t * Ns + n];

        float dA  = metal::exp(dt_val * A_val);
        h = dA * h + dt_val * u_val * B_val;

        scratch[n] = C_val * h;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = Ns / 2; stride > 0; stride >>= 1) {
            if (n < stride) scratch[n] += scratch[n + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (n == 0) y_out[b * L * D2 + t * D2 + d] = scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    h_final[b * D2 * Ns + d * Ns + n] = h;
"""
        _fwd_inf_kernel = mxf.metal_kernel(
            name="selective_scan_fwd_inf",
            input_names=["u_in", "dt_in", "A_in", "B_in", "C_in", "h0",
                         "b_size", "l_size", "d2_size"],
            output_names=["y_out", "h_final"],
            source=FWD_INF_SRC,
        )

        # ── Training forward (saves h_states in shape (B, L+1, D2, Ns)) ──────
        # h_states[:, 0, :, :] = h0
        # h_states[:, t+1, :, :] = h after step t
        FWD_TRAIN_SRC = """
    uint n   = thread_position_in_threadgroup.x;
    uint bd  = threadgroup_position_in_grid.x;
    uint Ns  = threads_per_threadgroup.x;
    uint D2  = d2_size[0];
    uint L   = l_size[0];
    uint B   = b_size[0];

    uint b = bd / D2;
    uint d = bd % D2;
    if (b >= B || d >= D2 || n >= Ns) return;

    threadgroup float scratch[256];

    uint Lp1 = L + 1;
    float A_val = A_in[d * Ns + n];
    float h = h0[b * D2 * Ns + d * Ns + n];

    // store initial state at t=0 slot
    h_states[b * Lp1 * D2 * Ns + 0 * D2 * Ns + d * Ns + n] = h;

    for (uint t = 0; t < L; t++) {
        float dt_val = dt_in[b * L * D2 + t * D2 + d];
        float u_val  = u_in[b * L * D2 + t * D2 + d];
        float B_val  = B_in[b * L * Ns + t * Ns + n];
        float C_val  = C_in[b * L * Ns + t * Ns + n];

        float dA = metal::exp(dt_val * A_val);
        h = dA * h + dt_val * u_val * B_val;

        h_states[b * Lp1 * D2 * Ns + (t + 1) * D2 * Ns + d * Ns + n] = h;

        scratch[n] = C_val * h;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = Ns / 2; stride > 0; stride >>= 1) {
            if (n < stride) scratch[n] += scratch[n + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (n == 0) y_out[b * L * D2 + t * D2 + d] = scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    h_final[b * D2 * Ns + d * Ns + n] = h;
"""
        _fwd_train_kernel = mxf.metal_kernel(
            name="selective_scan_fwd_train",
            input_names=["u_in", "dt_in", "A_in", "B_in", "C_in", "h0",
                         "b_size", "l_size", "d2_size"],
            output_names=["y_out", "h_final", "h_states"],
            source=FWD_TRAIN_SRC,
        )

        # ── BWD1: backward recurrence, one threadgroup per (b, d) ─────────────
        # Outputs (no atomics — each threadgroup owns its slice):
        #   grad_dt_out   (B, L, D2)
        #   grad_u_out    (B, L, D2)
        #   g_h_states_out (B, L, D2, Ns)  — g_h after y-accumulation at each t
        #   grad_A_per_b  (B, D2, Ns)      — Python sums over B
        #   grad_h0_out   (B, D2, Ns)
        BWD1_SRC = """
    uint n   = thread_position_in_threadgroup.x;
    uint bd  = threadgroup_position_in_grid.x;
    uint Ns  = threads_per_threadgroup.x;
    uint D2  = d2_size[0];
    uint L   = l_size[0];
    uint B   = b_size[0];

    uint b = bd / D2;
    uint d = bd % D2;
    if (b >= B || d >= D2 || n >= Ns) return;

    // scratch[0..Ns) = grad_dt partial sums
    // scratch[Ns..2Ns) = grad_u partial sums
    threadgroup float scratch[512];

    uint Lp1 = L + 1;
    float A_val = A_in[d * Ns + n];

    float g_h = grad_h_final[b * D2 * Ns + d * Ns + n];
    float grad_A_acc = 0.0f;

    for (int t = (int)L - 1; t >= 0; t--) {
        uint ut = (uint)t;
        float dt_val = dt_in[b * L * D2 + ut * D2 + d];
        float u_val  = u_in[b * L * D2 + ut * D2 + d];
        float B_val  = B_in[b * L * Ns + ut * Ns + n];
        float C_val  = C_in[b * L * Ns + ut * Ns + n];
        float gy_val = grad_y[b * L * D2 + ut * D2 + d];

        // h_states layout: (B, L+1, D2, Ns)
        float h_prev = h_states[b * Lp1 * D2 * Ns + ut * D2 * Ns + d * Ns + n];

        float dA = metal::exp(dt_val * A_val);

        // Accumulate g_h from y output at time t
        g_h += gy_val * C_val;

        // Store g_h (after y-accumulation) for BWD2 to compute grad_B
        g_h_states_out[b * L * D2 * Ns + ut * D2 * Ns + d * Ns + n] = g_h;

        // grad_A local accumulation (will sum over B in Python)
        float grad_dA_n = g_h * h_prev;
        float dt_A_dA   = grad_dA_n * dA;
        grad_A_acc += dt_A_dA * dt_val;

        // grad_dt[b,t,d] = sum_n( dt_A_dA*A_val + g_h*u*B )
        // grad_u[b,t,d]  = sum_n( g_h*dt*B )
        float grad_dt_n = dt_A_dA * A_val + g_h * u_val * B_val;
        float grad_u_n  = g_h * dt_val * B_val;

        scratch[n]        = grad_dt_n;
        scratch[n + Ns]   = grad_u_n;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = Ns / 2; stride > 0; stride >>= 1) {
            if (n < stride) {
                scratch[n]      += scratch[n + stride];
                scratch[n + Ns] += scratch[n + Ns + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (n == 0) {
            grad_dt_out[b * L * D2 + ut * D2 + d] = scratch[0];
            grad_u_out[b * L * D2 + ut * D2 + d]  = scratch[Ns];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        g_h = g_h * dA;
    }

    grad_A_per_b[b * D2 * Ns + d * Ns + n] = grad_A_acc;
    grad_h0_out[b * D2 * Ns + d * Ns + n]  = g_h;
"""
        _bwd1_kernel = mxf.metal_kernel(
            name="selective_scan_bwd1",
            input_names=["u_in", "dt_in", "A_in", "B_in", "C_in",
                         "h_states", "grad_y", "grad_h_final",
                         "b_size", "l_size", "d2_size"],
            output_names=["grad_dt_out", "grad_u_out",
                          "g_h_states_out", "grad_A_per_b", "grad_h0_out"],
            source=BWD1_SRC,
        )

        # ── BWD2: parallel grad_B / grad_C, one threadgroup per (b, t) ────────
        # Each thread (b, t, n) iterates over d — no atomics.
        # grad_C[b,t,n] = sum_d grad_y[b,t,d]  * h_states[b,t+1,d,n]
        # grad_B[b,t,n] = sum_d g_h_states[b,t,d,n] * dt[b,t,d] * u[b,t,d]
        BWD2_SRC = """
    uint n   = thread_position_in_threadgroup.x;
    uint bt  = threadgroup_position_in_grid.x;
    uint Ns  = threads_per_threadgroup.x;
    uint B   = b_size[0];
    uint L   = l_size[0];
    uint D2  = d2_size[0];

    uint b = bt / L;
    uint t = bt % L;
    if (b >= B || t >= L || n >= Ns) return;

    uint Lp1 = L + 1;
    float gc = 0.0f;
    float gb = 0.0f;

    for (uint d = 0; d < D2; d++) {
        float gy  = grad_y[b * L * D2 + t * D2 + d];
        float h_t = h_states[b * Lp1 * D2 * Ns + (t + 1) * D2 * Ns + d * Ns + n];
        float gh  = g_h_states[b * L * D2 * Ns + t * D2 * Ns + d * Ns + n];
        float dt  = dt_in[b * L * D2 + t * D2 + d];
        float u   = u_in[b * L * D2 + t * D2 + d];

        gc += gy * h_t;
        gb += gh * dt * u;
    }

    grad_C_out[b * L * Ns + t * Ns + n] = gc;
    grad_B_out[b * L * Ns + t * Ns + n] = gb;
"""
        _bwd2_kernel = mxf.metal_kernel(
            name="selective_scan_bwd2",
            input_names=["u_in", "dt_in", "h_states", "g_h_states", "grad_y",
                         "b_size", "l_size", "d2_size"],
            output_names=["grad_C_out", "grad_B_out"],
            source=BWD2_SRC,
        )

        _MLX_AVAILABLE = True
        return True
    except Exception as e:
        import warnings
        warnings.warn(f"selective_scan: Metal kernel build failed ({e}); using PyTorch loop.")
        return False


# ── PT ↔ MLX helpers ──────────────────────────────────────────────────────────

def _pt_to_mlx(t: torch.Tensor):
    import mlx.core as mx
    return mx.array(t.cpu().numpy(), dtype=mx.float32)


def _mlx_to_pt(arr, device: torch.device) -> torch.Tensor:
    import numpy as np
    import mlx.core as mx
    mx.eval(arr)
    return torch.tensor(np.array(arr), device=device)


# ── Metal forward (inference) ─────────────────────────────────────────────────

def _run_metal_scan(u, dt, A, B, C, h_prev):
    import mlx.core as mx
    B_b, L, D2 = u.shape
    Ns = A.shape[-1]
    device = u.device

    h0_m = (_pt_to_mlx(h_prev) if h_prev is not None
            else mx.zeros((B_b, D2, Ns), dtype=mx.float32))

    y_m, hf_m = _fwd_inf_kernel(
        inputs=[_pt_to_mlx(u), _pt_to_mlx(dt), _pt_to_mlx(A),
                _pt_to_mlx(B), _pt_to_mlx(C), h0_m,
                mx.array([B_b], dtype=mx.uint32),
                mx.array([L],   dtype=mx.uint32),
                mx.array([D2],  dtype=mx.uint32)],
        template=[("T", mx.float32)],
        grid=(B_b * D2 * Ns, 1, 1),
        threadgroup=(Ns, 1, 1),
        output_shapes=[(B_b, L, D2), (B_b, D2, Ns)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return _mlx_to_pt(y_m, device), _mlx_to_pt(hf_m, device)


# ── Metal forward (training, saves h_states) ──────────────────────────────────

def _run_metal_scan_train(u, dt, A, B, C, h0):
    import mlx.core as mx
    B_b, L, D2 = u.shape
    Ns = A.shape[-1]
    device = u.device

    h0_m = (_pt_to_mlx(h0) if h0 is not None
            else mx.zeros((B_b, D2, Ns), dtype=mx.float32))

    y_m, hf_m, hs_m = _fwd_train_kernel(
        inputs=[_pt_to_mlx(u), _pt_to_mlx(dt), _pt_to_mlx(A),
                _pt_to_mlx(B), _pt_to_mlx(C), h0_m,
                mx.array([B_b], dtype=mx.uint32),
                mx.array([L],   dtype=mx.uint32),
                mx.array([D2],  dtype=mx.uint32)],
        template=[("T", mx.float32)],
        grid=(B_b * D2 * Ns, 1, 1),
        threadgroup=(Ns, 1, 1),
        output_shapes=[(B_b, L, D2), (B_b, D2, Ns), (B_b, L + 1, D2, Ns)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    mx.eval(y_m, hf_m, hs_m)
    return (_mlx_to_pt(y_m, device),
            _mlx_to_pt(hf_m, device),
            _mlx_to_pt(hs_m, device))


# ── Metal backward ────────────────────────────────────────────────────────────

def _run_metal_scan_bwd(u, dt, A, B, C, h0, h_states, grad_y, grad_h_final):
    """Two-kernel Metal backward: BWD1 (sequential recurrence) + BWD2 (parallel B/C grads)."""
    import mlx.core as mx
    B_b, L, D2 = u.shape
    Ns = A.shape[-1]
    device = u.device

    b_sz  = mx.array([B_b], dtype=mx.uint32)
    l_sz  = mx.array([L],   dtype=mx.uint32)
    d2_sz = mx.array([D2],  dtype=mx.uint32)

    u_m   = _pt_to_mlx(u)
    dt_m  = _pt_to_mlx(dt)
    A_m   = _pt_to_mlx(A)
    B_m   = _pt_to_mlx(B)
    C_m   = _pt_to_mlx(C)
    h0_m  = _pt_to_mlx(h0)
    hs_m  = _pt_to_mlx(h_states)
    gy_m  = _pt_to_mlx(grad_y)
    ghf_m = _pt_to_mlx(grad_h_final)

    # BWD1: backward recurrence, one threadgroup per (b, d)
    (gdt_m, gu_m, ghs_m, gA_per_b_m, gh0_m) = _bwd1_kernel(
        inputs=[u_m, dt_m, A_m, B_m, C_m, hs_m, gy_m, ghf_m,
                b_sz, l_sz, d2_sz],
        template=[("T", mx.float32)],
        grid=(B_b * D2 * Ns, 1, 1),
        threadgroup=(Ns, 1, 1),
        output_shapes=[
            (B_b, L, D2),      # grad_dt
            (B_b, L, D2),      # grad_u
            (B_b, L, D2, Ns),  # g_h_states
            (B_b, D2, Ns),     # grad_A_per_b
            (B_b, D2, Ns),     # grad_h0
        ],
        output_dtypes=[mx.float32] * 5,
    )

    # BWD2: parallel grad_C and grad_B, one threadgroup per (b, t)
    (gC_m, gB_m) = _bwd2_kernel(
        inputs=[u_m, dt_m, hs_m, ghs_m, gy_m, b_sz, l_sz, d2_sz],
        template=[("T", mx.float32)],
        grid=(B_b * L * Ns, 1, 1),
        threadgroup=(Ns, 1, 1),
        output_shapes=[(B_b, L, Ns), (B_b, L, Ns)],
        output_dtypes=[mx.float32, mx.float32],
    )

    mx.eval(gdt_m, gu_m, gA_per_b_m, gh0_m, gC_m, gB_m)

    grad_A_per_b = _mlx_to_pt(gA_per_b_m, device)  # (B, D2, Ns)
    grad_A = grad_A_per_b.sum(0)                     # (D2, Ns) — sum over batch

    return (
        _mlx_to_pt(gu_m,  device),   # grad_u
        _mlx_to_pt(gdt_m, device),   # grad_dt
        grad_A,                       # grad_A
        _mlx_to_pt(gB_m,  device),   # grad_B
        _mlx_to_pt(gC_m,  device),   # grad_C
        _mlx_to_pt(gh0_m, device),   # grad_h0
    )


# ── PyTorch fallback backward ─────────────────────────────────────────────────

def _pt_scan_backward(u, dt, A, B, C, h0, grad_y, grad_h_final):
    B_b, L, D2 = u.shape
    Ns = A.shape[-1]

    h_prev_list = []
    h = h0.clone() if h0 is not None else u.new_zeros(B_b, D2, Ns)
    h_prev_list.append(h.clone())
    with torch.no_grad():
        for t in range(L):
            dA  = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))
            dBu = (dt[:, t] * u[:, t]).unsqueeze(-1) * B[:, t].unsqueeze(1)
            h   = dA * h + dBu
            h_prev_list.append(h.clone())

    grad_u  = torch.zeros_like(u)
    grad_dt = torch.zeros_like(dt)
    grad_A  = torch.zeros_like(A)
    grad_B  = torch.zeros_like(B)
    grad_C  = torch.zeros_like(C)

    g_h = grad_h_final.clone() if grad_h_final is not None else u.new_zeros(B_b, D2, Ns)

    for t in range(L - 1, -1, -1):
        h_t      = h_prev_list[t + 1]
        h_prev_t = h_prev_list[t]

        g_h = g_h + grad_y[:, t].unsqueeze(-1) * C[:, t].unsqueeze(1)
        grad_C[:, t] += (grad_y[:, t].unsqueeze(-1) * h_t).sum(dim=1)

        dt_t = dt[:, t]
        u_t  = u[:, t]
        B_t  = B[:, t]
        dA   = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))

        grad_dA  = g_h * h_prev_t
        dt_A_dA  = grad_dA * dA
        grad_dt[:, t] += (dt_A_dA * A.unsqueeze(0)).sum(-1)
        grad_A         += (dt_A_dA * dt_t.unsqueeze(-1)).sum(0)

        u_B_outer = u_t.unsqueeze(-1) * B_t.unsqueeze(1)
        grad_dt[:, t] += (g_h * u_B_outer).sum(-1)
        grad_u[:, t]  += (g_h * dt_t.unsqueeze(-1) * B_t.unsqueeze(1)).sum(-1)
        grad_B[:, t]  += (g_h * (dt_t * u_t).unsqueeze(-1)).sum(1)

        g_h = g_h * dA

    return grad_u, grad_dt, grad_A, grad_B, grad_C, g_h


# ── autograd.Function ──────────────────────────────────────────────────────────

class _SelectiveScanMetal(_AutogradBase):
    """Forward via Metal kernel (training variant), backward via Metal or PyTorch."""

    @staticmethod
    def forward(ctx, u, dt, A, B, C, h0):
        y, h_final, h_states = _run_metal_scan_train(u, dt, A, B, C, h0)
        ctx.save_for_backward(u, dt, A, B, C, h0, h_states)
        ctx.use_metal_bwd = (_bwd1_kernel is not None and _bwd2_kernel is not None)
        return y, h_final

    @staticmethod
    def backward(ctx, grad_y, grad_h_final):
        u, dt, A, B, C, h0, h_states = ctx.saved_tensors
        gf = (grad_h_final if grad_h_final is not None
              else u.new_zeros(u.shape[0], A.shape[0], A.shape[1]))

        if ctx.use_metal_bwd:
            grads = _run_metal_scan_bwd(u, dt, A, B, C, h0, h_states,
                                        grad_y.float(), gf.float())
        else:
            grads = _pt_scan_backward(u, dt, A, B, C, h0,
                                      grad_y.float(), gf.float())
        return grads  # (grad_u, grad_dt, grad_A, grad_B, grad_C, grad_h0)


# ── PyTorch reference forward ─────────────────────────────────────────────────

def _pt_scan_forward(u, dt, A, B, C, h_prev):
    B_b, L, D2 = u.shape
    Ns = B.shape[-1]
    h = h_prev if h_prev is not None else u.new_zeros(B_b, D2, Ns)
    outputs = []
    for t in range(L):
        dA  = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))
        dBu = (dt[:, t] * u[:, t]).unsqueeze(-1) * B[:, t].unsqueeze(1)
        h   = dA * h + dBu
        outputs.append((h * C[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(outputs, dim=1), h


# ── Public entry point ─────────────────────────────────────────────────────────

# Below this length the PT↔MLX copy overhead exceeds scan savings.
_METAL_MIN_L = 64

def selective_scan(
    u:      torch.Tensor,
    dt:     torch.Tensor,
    A:      torch.Tensor,
    B:      torch.Tensor,
    C:      torch.Tensor,
    h_prev: torch.Tensor | None,
    *,
    use_metal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for HedgeMambaMixer._selective_scan.

    Args:
        u:         (B, L, D2)
        dt:        (B, L, D2)
        A:         (D2, Ns)
        B:         (B, L, Ns)
        C:         (B, L, Ns)
        h_prev:    (B, D2, Ns) | None
        use_metal: Try Metal kernel; fall back to PyTorch loop if unavailable.

    Returns:
        y:       (B, L, D2)
        h_final: (B, D2, Ns)
    """
    global _MLX_AVAILABLE

    if FORCE_PT_LOOP:
        return _pt_scan_forward(u, dt, A, B, C, h_prev)

    B_b, L, D2 = u.shape
    Ns = A.shape[-1]

    if L < _METAL_MIN_L:
        return _pt_scan_forward(u, dt, A, B, C, h_prev)

    ns_ok = (Ns > 0) and (Ns & (Ns - 1) == 0) and (Ns <= 256)

    if use_metal and ns_ok:
        if not _MLX_AVAILABLE:
            _try_build_kernels(Ns)

        if _MLX_AVAILABLE:
            if torch.is_grad_enabled() and any(
                t.requires_grad for t in [u, dt, A, B, C]
            ):
                h0 = h_prev if h_prev is not None else u.new_zeros(B_b, D2, Ns)
                return _SelectiveScanMetal.apply(u, dt, A, B, C, h0)
            else:
                return _run_metal_scan(u, dt, A, B, C, h_prev)

    return _pt_scan_forward(u, dt, A, B, C, h_prev)


# ── MLX-native Metal scan (zero-copy, no PT bridge) ────────────────────────────

FORCE_MLX_LOOP = False       # set True to benchmark against Python loop
_selective_scan_mlx_fn = None  # lazily built by _build_mlx_scan_fn()

# Alias for scripts that import the singular form
_try_build_kernel = _try_build_kernels


def _mx_scan_python_loop(u, dt, A, B, C, h0):
    """Pure-MLX Python-loop fallback. MLX lazy-eval fuses the whole loop."""
    import mlx.core as mx
    Bb, L, D2 = u.shape
    Ns = B.shape[-1]
    h = h0 if h0 is not None else mx.zeros((Bb, D2, Ns))
    outputs = []
    for t in range(L):
        dA  = mx.exp(mx.expand_dims(dt[:, t], -1) * mx.expand_dims(A, 0))
        dBu = mx.expand_dims(dt[:, t] * u[:, t], -1) * mx.expand_dims(B[:, t], 1)
        h   = dA * h + dBu
        outputs.append((h * mx.expand_dims(C[:, t], 1)).sum(axis=-1))
    return mx.stack(outputs, axis=1), h


def _run_metal_fwd_mlx(u_m, dt_m, A_m, B_m, C_m, h0_m):
    """Training forward with all-mx.array I/O. Returns (y, h_final, h_states)."""
    import mlx.core as mx
    B_b, L, D2 = u_m.shape
    Ns = int(A_m.shape[-1])
    b_sz = mx.array([B_b], dtype=mx.uint32)
    l_sz = mx.array([L],   dtype=mx.uint32)
    d_sz = mx.array([D2],  dtype=mx.uint32)
    return _fwd_train_kernel(
        inputs=[u_m, dt_m, A_m, B_m, C_m, h0_m, b_sz, l_sz, d_sz],
        template=[("T", mx.float32)],
        grid=(B_b * D2 * Ns, 1, 1),
        threadgroup=(Ns, 1, 1),
        output_shapes=[(B_b, L, D2), (B_b, D2, Ns), (B_b, L + 1, D2, Ns)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


def _run_metal_bwd_mlx(u_m, dt_m, A_m, B_m, C_m, h0_m, hs_m, gy_m, ghf_m):
    """Metal backward with all-mx.array I/O.
    Returns (grad_u, grad_dt, grad_A, grad_B, grad_C, grad_h0).
    """
    import mlx.core as mx
    B_b, L, D2 = u_m.shape
    Ns = int(A_m.shape[-1])
    b_sz = mx.array([B_b], dtype=mx.uint32)
    l_sz = mx.array([L],   dtype=mx.uint32)
    d_sz = mx.array([D2],  dtype=mx.uint32)

    gdt_m, gu_m, ghs_m, gApb_m, gh0_m = _bwd1_kernel(
        inputs=[u_m, dt_m, A_m, B_m, C_m, hs_m, gy_m, ghf_m,
                b_sz, l_sz, d_sz],
        template=[("T", mx.float32)],
        grid=(B_b * D2 * Ns, 1, 1),
        threadgroup=(Ns, 1, 1),
        output_shapes=[
            (B_b, L, D2), (B_b, L, D2),
            (B_b, L, D2, Ns), (B_b, D2, Ns), (B_b, D2, Ns),
        ],
        output_dtypes=[mx.float32] * 5,
    )
    gC_m, gB_m = _bwd2_kernel(
        inputs=[u_m, dt_m, hs_m, ghs_m, gy_m, b_sz, l_sz, d_sz],
        template=[("T", mx.float32)],
        grid=(B_b * L * Ns, 1, 1),
        threadgroup=(Ns, 1, 1),
        output_shapes=[(B_b, L, Ns), (B_b, L, Ns)],
        output_dtypes=[mx.float32, mx.float32],
    )
    mx.eval(gu_m, gdt_m, gApb_m, gh0_m, gC_m, gB_m)
    return gu_m, gdt_m, gApb_m.sum(axis=0), gB_m, gC_m, gh0_m


def _build_mlx_scan_fn():
    """Build the mx.custom_vjp-wrapped scan once kernels are ready."""
    global _selective_scan_mlx_fn
    import mlx.core as mx

    @mx.custom_vjp
    def _scan_fwd(u, dt, A, B, C, h0):
        # Third output (h_states) is a residual kept only for the VJP.
        return _run_metal_fwd_mlx(u, dt, A, B, C, h0)

    @_scan_fwd.vjp
    def _scan_vjp(primals, cotangents, output):
        u, dt, A, B, C, h0 = primals
        grad_y, grad_hf, _ = cotangents   # _ = cotan for h_states (zeros)
        _, hf, h_states = output
        if grad_hf is None:
            grad_hf = mx.zeros_like(hf)
        gu, gdt, gA, gB, gC, gh0 = _run_metal_bwd_mlx(
            u, dt, A, B, C, h0, h_states,
            grad_y.astype(mx.float32), grad_hf.astype(mx.float32),
        )
        return [gu, gdt, gA, gB, gC, gh0]

    def _fn(u, dt, A, B, C, h0):
        y, hf, _ = _scan_fwd(u, dt, A, B, C, h0)
        return y, hf

    _selective_scan_mlx_fn = _fn


def selective_scan_mlx(u, dt, A, B, C, h0=None):
    """Drop-in selective scan for MLX models.

    Uses Metal kernels directly (no PT bridge) when:
      - FORCE_MLX_LOOP is False
      - L >= 64
      - Ns is a power of 2 and <= 256
    Falls back to the Python loop (still fused by MLX lazy eval) otherwise.
    """
    global _selective_scan_mlx_fn, _MLX_AVAILABLE
    import mlx.core as mx

    B_b, L, D2 = u.shape
    Ns = int(A.shape[-1])

    if h0 is None:
        h0 = mx.zeros((B_b, D2, Ns))

    ns_ok = (Ns & (Ns - 1) == 0) and (0 < Ns <= 256)
    if FORCE_MLX_LOOP or L < _METAL_MIN_L or not ns_ok:
        return _mx_scan_python_loop(u, dt, A, B, C, h0)

    if not _MLX_AVAILABLE:
        _try_build_kernels(Ns)
    if not _MLX_AVAILABLE:
        return _mx_scan_python_loop(u, dt, A, B, C, h0)

    if _selective_scan_mlx_fn is None:
        if not hasattr(mx, 'custom_vjp'):
            # MLX < 0.32 doesn't have custom_vjp — use the Python loop instead.
            # It's still fully differentiable through MLX's standard autograd.
            return _mx_scan_python_loop(u, dt, A, B, C, h0)
        _build_mlx_scan_fn()

    return _selective_scan_mlx_fn(u, dt, A, B, C, h0)
