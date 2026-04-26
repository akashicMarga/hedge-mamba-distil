"""Stage 2: Fine-tune HedgeMamba student with cross-entropy on next-token prediction.

Everything trains except input/output embeddings (frozen throughout).
Init: apply parameter surgery before calling this (see student/param_init.py).

Logging:
  - TensorBoard: train/loss, val/loss, train/lr, train/tokens  (runs/ directory)
  - stdout: Step N | tokens X | train_loss Y | val_loss Z
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


def freeze_embeddings(student_model: nn.Module) -> None:
    for param in student_model.backbone.gpt_neox.embed_in.parameters():
        param.requires_grad = False
    for param in student_model.backbone.embed_out.parameters():
        param.requires_grad = False


@torch.no_grad()
def compute_val_loss(model, val_loader: DataLoader, device: str, max_batches: int = 50) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        out = model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def train_stage2(
    student_model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    total_tokens: int = 500_000_000,
    checkpoint_every_tokens: int = 50_000_000,
    eval_every_tokens: int = 10_000_000,
    checkpoint_dir: str | Path = "./checkpoints/stage2",
    val_loader: DataLoader | None = None,
    tb_log_dir: str | Path | None = None,
    log_every: int = 100,
) -> list[float]:

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    run_name = datetime.now().strftime("stage2_%Y%m%d_%H%M%S")
    log_dir = Path(tb_log_dir) if tb_log_dir else Path("runs") / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs → {log_dir}  (run: tensorboard --logdir runs/)", flush=True)

    student_model.train()
    freeze_embeddings(student_model)

    losses = []
    tokens_seen = 0
    next_ckpt_at = checkpoint_every_tokens
    next_eval_at = eval_every_tokens
    step = 0
    last_val_loss = float("nan")

    for batch in tqdm(dataloader, desc="Stage 2"):
        if tokens_seen >= total_tokens:
            break

        input_ids = batch["input_ids"].to(device)
        out = student_model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # NaN guard: skip step if loss exploded (don't poison weights)
        if not torch.isfinite(loss):
            print(f"  [WARN] Non-finite loss ({loss.item():.4f}) at step {step+1}, skipping update.", flush=True)
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        optimizer.step()

        batch_tokens = input_ids.numel()
        tokens_seen += batch_tokens
        losses.append(loss.item())
        step += 1

        # TensorBoard: train loss every step
        writer.add_scalar("train/loss", loss.item(), global_step=tokens_seen)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step=tokens_seen)

        # Validation loss
        if val_loader is not None and tokens_seen >= next_eval_at:
            last_val_loss = compute_val_loss(student_model, val_loader, device, max_batches=50)
            writer.add_scalar("val/loss", last_val_loss, global_step=tokens_seen)
            if math.isfinite(last_val_loss):
                writer.add_scalar("val/ppl", math.exp(last_val_loss), global_step=tokens_seen)
            writer.flush()
            next_eval_at += eval_every_tokens

        # Stdout log
        if step % log_every == 0:
            val_str = f" | val_loss {last_val_loss:.4f}" if not math.isnan(last_val_loss) else ""
            print(
                f"Step {step} | tokens {tokens_seen:,} | train_loss {loss.item():.4f}{val_str}",
                flush=True,
            )

        # Checkpoint
        if tokens_seen >= next_ckpt_at:
            ckpt_path = checkpoint_dir / f"ckpt_{tokens_seen // 1_000_000}M.pt"
            torch.save(student_model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}", flush=True)
            next_ckpt_at += checkpoint_every_tokens

    writer.close()
    return losses
