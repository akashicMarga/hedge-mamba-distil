"""Stage 2: cross-entropy fine-tuning of ParlerTTS-Mamba.

After Stage 1 the SSM layers can mimic the teacher's hidden states.
Stage 2 closes the remaining quality gap with task-supervised training:
the model must predict the next codec token across all 9 codebooks.

Loss:
  L₂ = (1/9) Σ_k  CE(logits_k, labels_k)

where logits and labels are both (B, T) per codebook k.

Scheduled sampling:
  With probability p(step), replace a ground-truth decoder input token
  with the student's own prediction from a no-grad forward pass.
  p(step) rises linearly from 0 to ss_max_p over the first 50% of steps,
  then stays constant.  Default ss_max_p=0.5.
  This forces the student to be robust to its own prediction errors,
  matching the distribution seen at inference time.

Frozen in Stage 2:
  - T5 encoder
  - Audio token embeddings, position embeddings, embed_prompts
Trained: SSM layers, cross-attention, FFN, LayerNorms, LM head.

Delay pattern for labels:
  Parler generates codec token k at step t from delayed input t-k.
  Labels are shifted accordingly: label[k, t] = codec_token[k, t+k].
  We rely on the dataloader to supply pre-shifted decoder_input_ids and
  labels (same convention as the teacher's training data).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _parler_ce_loss(
    logits: torch.Tensor,   # (B, T, num_codebooks * vocab_size)
    labels: torch.Tensor,   # (B, T, num_codebooks)  or (B, T*num_codebooks)
    num_codebooks: int,
    vocab_size: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy over 9 codebook slots, padding-masked."""
    B, T, _ = logits.shape
    # (B, T, 9, V)
    logits_4d = logits.reshape(B, T, num_codebooks, vocab_size)
    # Broadcast labels to (B, T, 9) if needed
    if labels.dim() == 2:
        labels = labels.unsqueeze(-1).expand(B, T, num_codebooks)
    labels = labels[:, :T, :]   # truncate to T if labels are longer

    total = torch.zeros(1, device=logits.device)
    n_valid = 0
    for k in range(num_codebooks):
        lk = logits_4d[:, :, k, :].reshape(B * T, vocab_size)  # (B*T, V)
        yk = labels[:, :, k].reshape(B * T)                    # (B*T,)
        mask = (yk != ignore_index)
        if mask.sum() == 0:
            continue
        safe_yk = yk.clamp(min=0)
        ce = F.cross_entropy(lk, safe_yk, reduction="none")
        total   = total + (ce * mask.float()).sum() / mask.float().sum()
        n_valid += 1
    return total / max(n_valid, 1)


# ---------------------------------------------------------------------------
# Scheduled sampling
# ---------------------------------------------------------------------------


def _scheduled_sample(
    decoder_input_ids: torch.Tensor,  # (B, T)  ground truth
    student_model,
    input_ids: torch.Tensor,
    attention_mask,
    device: str,
    p: float,
) -> torch.Tensor:
    """Replace fraction p of decoder input tokens with student predictions.

    A no-grad forward pass produces logits; argmax gives student tokens.
    For each position independently, swap in the student token with prob p.
    """
    if p <= 0.0:
        return decoder_input_ids

    with torch.no_grad():
        out = student_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
        )
    # logits: (B, T, num_codebooks * vocab_size) -> argmax over vocab per pos/cb
    logits    = out.logits                         # (B, T, 9*V)
    B, T, _   = logits.shape
    cfg       = student_model.config
    num_cb    = cfg.decoder_config.num_codebooks
    vocab     = cfg.decoder_config.vocab_size
    pred_4d   = logits.reshape(B, T, num_cb, vocab).argmax(-1)  # (B, T, 9)
    # Flatten to (B, T) by taking codebook 0 (coarse) — others are shifted
    pred_flat = pred_4d[:, :, 0]                   # (B, T)

    # Bernoulli mask: 1 = use student prediction
    swap_mask = torch.bernoulli(
        torch.full_like(decoder_input_ids, p, dtype=torch.float)
    ).bool()
    return torch.where(swap_mask, pred_flat, decoder_input_ids)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_stage2_val_loss(
    student_model,
    val_loader: DataLoader,
    device: str,
    max_batches: int = 30,
) -> float:
    student_model.eval()
    total, count = 0.0, 0
    cfg   = student_model.config
    num_cb = cfg.decoder_config.num_codebooks
    vocab  = cfg.decoder_config.vocab_size

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids         = batch["input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels            = batch["labels"].to(device)
        attn_mask         = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        out  = student_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attn_mask,
        )
        loss = _parler_ce_loss(out.logits, labels, num_cb, vocab)
        total += loss.item()
        count += 1

    student_model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_parler_stage2(
    student_model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    epochs: int = 5,
    checkpoint_dir: str = "./checkpoints/parler_mamba",
    tb_log_dir: str | None = None,
    log_every: int = 50,
    eval_every: int = 500,
    scheduler=None,
    ss_max_p: float = 0.5,
) -> list[float]:
    """Stage 2: CE fine-tuning with scheduled sampling.

    Args:
        student_model:  ParlerMambaStudent
        train_loader:   yields dicts with input_ids / decoder_input_ids / labels
        ss_max_p:       max scheduled-sampling probability (0 = teacher forcing only)
        ...

    Returns:
        List of per-step CE losses.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("parler_stage2_%Y%m%d_%H%M%S")
    log_dir  = Path(tb_log_dir) / run_name if tb_log_dir else Path("runs") / run_name
    writer   = SummaryWriter(log_dir=str(log_dir))
    print(f"[Parler Stage 2] TensorBoard → {log_dir}")
    print(f"[Parler Stage 2] Scheduled sampling max_p={ss_max_p}")

    # Unfreeze SSM, cross-attn, FFN, LN (embeddings / T5 stay frozen)
    student_model.backbone.decoder.model.decoder.layers.requires_grad_(True)  # type: ignore
    student_model.backbone.decoder.lm_heads.requires_grad_(True)              # type: ignore
    # Keep embeddings frozen
    dec = student_model.backbone.decoder.model.decoder
    for emb in dec.embed_tokens:
        for p in emb.parameters():
            p.requires_grad = False
    for p in dec.embed_positions.parameters():
        p.requires_grad = False

    cfg    = student_model.config
    num_cb = cfg.decoder_config.num_codebooks
    vocab  = cfg.decoder_config.vocab_size

    total_steps = epochs * len(train_loader)
    losses: list[float] = []
    global_step   = 0
    last_val_loss = float("nan")

    student_model.train()

    for epoch in range(epochs):
        print(f"\n{'='*55}")
        print(f"[Parler S2] Epoch {epoch + 1} / {epochs}")
        print(f"{'='*55}")

        for batch in tqdm(train_loader, desc=f"[S2] Epoch {epoch+1}"):
            input_ids         = batch["input_ids"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels            = batch["labels"].to(device)
            attn_mask         = batch.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            # Scheduled sampling: ramp p linearly over first 50% of training
            p_ss = ss_max_p * min(
                1.0, global_step / max(total_steps * 0.5, 1)
            )
            decoder_input_ids = _scheduled_sample(
                decoder_input_ids, student_model,
                input_ids, attn_mask, device, p_ss,
            )

            out  = student_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attn_mask,
            )
            loss = _parler_ce_loss(out.logits, labels, num_cb, vocab)

            if not torch.isfinite(loss):
                print(f"  [S2] Non-finite loss step {global_step} — skipping")
                optimizer.zero_grad()
                global_step += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student_model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            losses.append(loss.item())
            writer.add_scalar("stage2/ce_loss",   loss.item(), global_step)
            writer.add_scalar("stage2/ss_p",      p_ss,        global_step)
            writer.add_scalar("stage2/lr",
                              optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

            if global_step % log_every == 0:
                val_str = (
                    f" | val {last_val_loss:.4f}"
                    if math.isfinite(last_val_loss) else ""
                )
                print(
                    f"  [S2] step {global_step} | ce {loss.item():.4f} | "
                    f"ss_p {p_ss:.3f}{val_str}"
                )

            if global_step % eval_every == 0:
                last_val_loss = compute_stage2_val_loss(
                    student_model, val_loader, device
                )
                writer.add_scalar(
                    "stage2/val_ce_loss", last_val_loss, global_step
                )
                writer.flush()
                print(f"  [S2 Val] step {global_step} | val {last_val_loss:.4f}")
                student_model.train()

        ckpt = ckpt_dir / f"stage2_epoch_{epoch + 1}.pt"
        torch.save(student_model.state_dict(), ckpt)
        print(f"[Parler S2] Checkpoint → {ckpt}")

    writer.close()
    return losses
