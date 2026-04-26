"""Stage 1 for Whisper-Mamba: layer-wise cosine distillation.

Goal: warm-initialize each Mamba SSM layer to mimic the corresponding
      Whisper decoder self-attention layer output.

Loss: mean cosine distance averaged over all decoder layers.
  L = mean_over_layers[ 1 - cosine_sim(student_ssm_out, teacher_attn_out, dim=-1) ]

Why this works:
  The teacher's self-attn output at layer i has already blended information
  from all query/key positions. Training the SSM to produce the same hidden
  states forces it to learn the same long-range structure — without ever
  needing autoregressive generation.

Frozen during Stage 1:
  - Teacher: entirely (eval mode, no gradients)
  - Student: encoder, embeddings, cross-attention, FFN, layer norms
Trained: only the SSM layers (B_proj, C_proj, D, log_lambda, in_proj, out_proj)

After Stage 1 the SSM is no longer random — Stage 2 fine-tuning then closes
the remaining gap with cross-entropy + actual transcript supervision.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


# ── Hook utilities ────────────────────────────────────────────────────────────

def _register_hooks(teacher_model, student_model):
    """Attach forward hooks to capture per-layer self-attention outputs.

    Teacher hook: captures layer.self_attn output[0]  → (B, L, dim), detached
    Student hook: captures layer.self_attn output[0]  → (B, L, dim), with grad

    Returns:
        teacher_acts  dict[int, Tensor]   — filled each forward pass
        student_acts  dict[int, Tensor]   — filled each forward pass
        handles       list                — call h.remove() when done
    """
    teacher_acts: dict[int, torch.Tensor] = {}
    student_acts: dict[int, torch.Tensor] = {}
    handles = []

    for i, layer in enumerate(teacher_model.model.decoder.layers):
        def _t_hook(module, inp, out, idx=i):
            # WhisperAttention returns (attn_output, past_key_value, attn_weights)
            # or (attn_output, attn_weights) depending on HF version — index 0 is always attn_output
            teacher_acts[idx] = out[0].detach()  # no grad needed for teacher
        handles.append(layer.self_attn.register_forward_hook(_t_hook))

    for i, layer in enumerate(student_model.backbone.model.decoder.layers):
        def _s_hook(module, inp, out, idx=i):
            # WhisperHedgeMambaLayer returns (ssm_out, None)
            student_acts[idx] = out[0]  # grad must flow through this
        handles.append(layer.self_attn.register_forward_hook(_s_hook))

    return teacher_acts, student_acts, handles


def _cosine_loss(student_acts: dict, teacher_acts: dict) -> torch.Tensor:
    """Mean cosine distance across layers and sequence positions."""
    total = torch.tensor(0.0)
    n = len(teacher_acts)
    for i in teacher_acts:
        s = student_acts[i]   # (B, L, dim)
        t = teacher_acts[i]   # (B, L, dim)
        sim = F.cosine_similarity(s, t, dim=-1)  # (B, L)
        total = total + (1.0 - sim).mean()
    return total / max(n, 1)


def _ssm_params(student_model):
    """SSM-only parameters — the only things we train in Stage 1."""
    params = []
    for layer in student_model.backbone.model.decoder.layers:
        # layer.self_attn is WhisperHedgeMambaLayer; .ssm is SelectiveSSM
        params.extend(p for p in layer.self_attn.parameters() if p.requires_grad)
    return params


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_stage1_val_loss(
    teacher_model,
    student_model,
    val_loader: DataLoader,
    device: str,
    max_batches: int = 30,
) -> float:
    """Cosine distance on the validation set (no gradients needed)."""
    teacher_model.eval()
    student_model.eval()

    teacher_acts, student_acts, handles = _register_hooks(teacher_model, student_model)
    total_loss, count = 0.0, 0

    try:
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            bos = torch.full(
                (labels.shape[0], 1),
                student_model.config.decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )
            decoder_input_ids = torch.cat([bos, labels[:, :-1].clamp(min=0)], dim=1)

            teacher_acts.clear()
            student_acts.clear()

            teacher_model(input_features=input_features, decoder_input_ids=decoder_input_ids)
            student_model(input_features=input_features, decoder_input_ids=decoder_input_ids)

            # Compute scalar cosine distance (no grad under @no_grad context)
            loss_val = 0.0
            n = len(teacher_acts)
            for idx in teacher_acts:
                s = student_acts[idx]
                t = teacher_acts[idx]
                sim = F.cosine_similarity(s, t, dim=-1)
                loss_val += (1.0 - sim).mean().item()
            loss_val /= max(n, 1)

            total_loss += loss_val
            count += 1

    finally:
        for h in handles:
            h.remove()

    student_model.train()
    return total_loss / max(count, 1)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_whisper_stage1(
    teacher_model,
    student_model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    epochs: int = 3,
    checkpoint_dir: str = "./checkpoints/whisper_mamba",
    tb_log_dir: str | None = None,
    log_every: int = 50,
    eval_every: int = 500,
    scheduler=None,
) -> list[float]:
    """Stage 1: train SSM layers to mimic Whisper decoder self-attention.

    Args:
        teacher_model:   original WhisperForConditionalGeneration (not the student copy)
        student_model:   WhisperMambaStudent
        ...

    Returns:
        List of per-step cosine losses.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("whisper_stage1_%Y%m%d_%H%M%S")
    log_dir = Path(tb_log_dir) / run_name if tb_log_dir else Path("runs") / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[Stage 1] TensorBoard → {log_dir}", flush=True)

    # Teacher stays frozen / eval the whole time
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model = teacher_model.to(device)

    student_model.train()

    # Register hooks once; clear dicts each step
    teacher_acts, student_acts, hook_handles = _register_hooks(teacher_model, student_model)

    losses: list[float] = []
    global_step = 0
    last_val_loss = float("nan")

    try:
        for epoch in range(epochs):
            print(f"\n{'='*55}", flush=True)
            print(f"[Stage 1] Epoch {epoch + 1} / {epochs}", flush=True)
            print(f"{'='*55}", flush=True)

            for batch in tqdm(train_loader, desc=f"[S1] Epoch {epoch+1}"):
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)

                bos = torch.full(
                    (labels.shape[0], 1),
                    student_model.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=device,
                )
                decoder_input_ids = torch.cat([bos, labels[:, :-1].clamp(min=0)], dim=1)

                teacher_acts.clear()
                student_acts.clear()

                # Teacher: no grad — just populates teacher_acts via hook
                with torch.no_grad():
                    teacher_model(
                        input_features=input_features,
                        decoder_input_ids=decoder_input_ids,
                    )

                # Student: grad flows through SSM layers → student_acts
                student_model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                )

                loss = _cosine_loss(student_acts, teacher_acts)

                if not torch.isfinite(loss):
                    print(f"  [S1] Non-finite loss at step {global_step} — skipping", flush=True)
                    optimizer.zero_grad()
                    global_step += 1
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_ssm_params(student_model), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                losses.append(loss.item())
                writer.add_scalar("stage1/cosine_loss", loss.item(), global_step)
                writer.add_scalar("stage1/lr", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

                if global_step % log_every == 0:
                    val_str = (
                        f" | val_cos {last_val_loss:.4f}"
                        if math.isfinite(last_val_loss) else ""
                    )
                    print(
                        f"  [S1] Step {global_step} | cos_loss {loss.item():.4f}{val_str}",
                        flush=True,
                    )

                if global_step % eval_every == 0:
                    last_val_loss = compute_stage1_val_loss(
                        teacher_model, student_model, val_loader, device
                    )
                    writer.add_scalar("stage1/val_cosine_loss", last_val_loss, global_step)
                    writer.flush()
                    print(
                        f"  [S1 Val] step {global_step} | val_cos {last_val_loss:.4f}",
                        flush=True,
                    )
                    student_model.train()

            ckpt = checkpoint_dir / f"stage1_epoch_{epoch + 1}.pt"
            torch.save(student_model.state_dict(), ckpt)
            print(f"[Stage 1] Checkpoint → {ckpt}", flush=True)

    finally:
        # Always clean up hooks — even if training crashes
        for h in hook_handles:
            h.remove()

    writer.close()
    return losses
