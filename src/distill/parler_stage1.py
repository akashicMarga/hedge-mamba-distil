"""Stage 1: layer-wise cosine distillation for ParlerTTS-Mamba.

Goal: warm-initialize each Mamba SSM layer to produce the same hidden
states as the corresponding teacher self-attention layer.

Loss:
  L₁ = mean_over_layers[ 1 - cosine_sim(student_ssm_out, teacher_attn_out, dim=-1) ]

Frozen in Stage 1:
  - Teacher: entirely (eval mode)
  - Student: T5 encoder, embed_prompts, audio embeddings, cross-attn, FFN, LN
Trained: SSM layers only (gate_proj, conv1d, x_proj, dt_proj, A_log,
         v_proj, hhog_k, hhog_q, out_proj).

Data:
  Audio description text + codec token sequences from a TTS dataset
  (e.g. parler_tts_mini_v1 or ljspeech processed through the teacher).
  The teacher generates decoder hidden states for each batch; the student
  trains to match them without ever needing to autoregressively generate.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Hook utilities
# ---------------------------------------------------------------------------


def _register_hooks(teacher_model, student_model):
    """Attach forward hooks to capture per-layer self-attention outputs.

    Teacher hook target: layer.self_attn  (ParlerTTSAttention)
      -> out[0] is (B, T, dim), the projected attention output.
    Student hook target: layer.self_attn  (ParlerHedgeMambaLayer)
      -> out[0] is (B, T, dim), the SSM output.

    Returns:
        teacher_acts : dict[int, Tensor]
        student_acts : dict[int, Tensor]
        handles      : list of hook handles
    """
    teacher_acts: dict[int, torch.Tensor] = {}
    student_acts: dict[int, torch.Tensor] = {}
    handles = []

    teacher_layers = teacher_model.decoder.model.decoder.layers
    for i, layer in enumerate(teacher_layers):
        def _t_hook(module, inp, out, idx=i):
            # ParlerTTSAttention returns (attn_output, past_kv, attn_weights)
            teacher_acts[idx] = out[0].detach().float()
        handles.append(layer.self_attn.register_forward_hook(_t_hook))

    student_layers = student_model.backbone.decoder.model.decoder.layers
    for i, layer in enumerate(student_layers):
        def _s_hook(module, inp, out, idx=i):
            # ParlerHedgeMambaLayer returns (ssm_out, None, None)
            student_acts[idx] = out[0].float()   # grad must flow
        handles.append(layer.self_attn.register_forward_hook(_s_hook))

    return teacher_acts, student_acts, handles


def _cosine_loss(
    student_acts: dict[int, torch.Tensor],
    teacher_acts: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Mean cosine distance across layers, batch, and sequence positions."""
    device = next(iter(student_acts.values())).device
    total  = torch.zeros(1, device=device)
    n      = len(teacher_acts)
    for i in teacher_acts:
        s   = student_acts[i]   # (B, T, dim)
        t   = teacher_acts[i]   # (B, T, dim)
        sim = F.cosine_similarity(s, t, dim=-1)  # (B, T)
        total = total + (1.0 - sim).mean()
    return total / max(n, 1)


def _ssm_params(student_model) -> list:
    """Return only the SSM layer parameters (trained in Stage 1)."""
    params = []
    for layer in student_model.backbone.decoder.model.decoder.layers:
        params.extend(
            p for p in layer.self_attn.parameters() if p.requires_grad
        )
    return params


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_stage1_val_loss(
    teacher_model,
    student_model,
    val_loader: DataLoader,
    device: str,
    max_batches: int = 30,
) -> float:
    teacher_model.eval()
    student_model.eval()
    teacher_acts, student_acts, handles = _register_hooks(teacher_model, student_model)
    total, count = 0.0, 0
    try:
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids           = batch["input_ids"].to(device)
            decoder_input_ids   = batch["decoder_input_ids"].to(device)
            attention_mask      = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            teacher_acts.clear()
            student_acts.clear()

            teacher_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )
            student_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )

            loss = 0.0
            n    = len(teacher_acts)
            for idx in teacher_acts:
                s    = student_acts[idx].float()
                t    = teacher_acts[idx].float()
                sim  = F.cosine_similarity(s, t, dim=-1)
                loss += (1.0 - sim).mean().item()
            total += loss / max(n, 1)
            count += 1
    finally:
        for h in handles:
            h.remove()
    student_model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_parler_stage1(
    teacher_model,
    student_model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    epochs: int = 3,
    checkpoint_dir: str = "./checkpoints/parler_mamba",
    tb_log_dir: str | None = None,
    log_every: int = 50,
    eval_every: int = 500,
    scheduler=None,
) -> list[float]:
    """Stage 1: train SSM layers to mimic Parler decoder self-attention.

    Args:
        teacher_model: ParlerTTSForConditionalGeneration (frozen, HF)
        student_model: ParlerMambaStudent
        train_loader:  yields dicts with input_ids / decoder_input_ids
        val_loader:    same format
        ...

    Returns:
        List of per-step cosine losses.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("parler_stage1_%Y%m%d_%H%M%S")
    log_dir  = Path(tb_log_dir) / run_name if tb_log_dir else Path("runs") / run_name
    writer   = SummaryWriter(log_dir=str(log_dir))
    print(f"[Parler Stage 1] TensorBoard → {log_dir}")

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model = teacher_model.to(device)
    student_model.train()

    teacher_acts, student_acts, hook_handles = _register_hooks(
        teacher_model, student_model
    )
    ssm_parameters = _ssm_params(student_model)

    losses: list[float] = []
    global_step  = 0
    last_val_loss = float("nan")

    try:
        for epoch in range(epochs):
            print(f"\n{'='*55}")
            print(f"[Parler S1] Epoch {epoch + 1} / {epochs}")
            print(f"{'='*55}")

            for batch in tqdm(train_loader, desc=f"[S1] Epoch {epoch+1}"):
                input_ids         = batch["input_ids"].to(device)
                decoder_input_ids = batch["decoder_input_ids"].to(device)
                attn_mask         = batch.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                teacher_acts.clear()
                student_acts.clear()

                # Teacher: only fills teacher_acts via hook
                with torch.no_grad():
                    teacher_model(
                        input_ids=input_ids,
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=attn_mask,
                    )

                # Student: grad flows through SSM layers
                student_model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attn_mask,
                )

                loss = _cosine_loss(student_acts, teacher_acts)

                if not torch.isfinite(loss):
                    print(f"  [S1] Non-finite loss step {global_step} — skipping")
                    optimizer.zero_grad()
                    global_step += 1
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ssm_parameters, 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                losses.append(loss.item())
                writer.add_scalar("stage1/cosine_loss", loss.item(), global_step)
                writer.add_scalar("stage1/lr",
                                  optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

                if global_step % log_every == 0:
                    val_str = (
                        f" | val {last_val_loss:.4f}"
                        if math.isfinite(last_val_loss) else ""
                    )
                    print(f"  [S1] step {global_step} | cos {loss.item():.4f}{val_str}")

                if global_step % eval_every == 0:
                    last_val_loss = compute_stage1_val_loss(
                        teacher_model, student_model, val_loader, device
                    )
                    writer.add_scalar(
                        "stage1/val_cosine_loss", last_val_loss, global_step
                    )
                    writer.flush()
                    print(f"  [S1 Val] step {global_step} | val {last_val_loss:.4f}")
                    student_model.train()

            ckpt = ckpt_dir / f"stage1_epoch_{epoch + 1}.pt"
            torch.save(student_model.state_dict(), ckpt)
            print(f"[Parler S1] Checkpoint → {ckpt}")

    finally:
        for h in hook_handles:
            h.remove()

    writer.close()
    return losses
