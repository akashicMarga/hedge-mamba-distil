"""Stage 1: Train Hedgehog feature maps via cosine embedding matching loss.

Only the HedgehogFeatureMap MLP weights train — everything else frozen.
Loss: cosine similarity between student layer output and cached teacher layer output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def cosine_distill_loss(student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
    """Cosine embedding loss averaged over batch and sequence positions."""
    B, L, D = student_out.shape
    s = student_out.reshape(B * L, D)
    t = teacher_out.reshape(B * L, D).to(s.dtype)
    # target=1: maximize cosine similarity
    return F.cosine_embedding_loss(s, t, torch.ones(B * L, device=s.device))


def freeze_except_hedgehog(student_model: nn.Module) -> None:
    """Freeze all params except HedgehogFeatureMap MLP weights."""
    from src.student.hedge_mamba import HedgehogFeatureMap
    for name, param in student_model.named_parameters():
        param.requires_grad = False
    for module in student_model.modules():
        if isinstance(module, HedgehogFeatureMap):
            for param in module.parameters():
                param.requires_grad = True


def train_stage1(
    student_model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    total_tokens: int = 100_000_000,
    log_every: int = 100,
) -> list[float]:
    """Train Stage 1. dataloader yields dicts with 'input_ids' and 'layer_outputs'."""
    student_model.train()
    freeze_except_hedgehog(student_model)

    losses = []
    tokens_seen = 0
    step = 0

    # Register hooks on student to capture layer outputs
    student_layer_outputs: list[torch.Tensor] = []
    hooks = []

    def make_hook(idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            student_layer_outputs.append(hidden)
        return hook

    for i, block in enumerate(student_model.backbone.gpt_neox.layers):
        h = block.register_forward_hook(make_hook(i))
        hooks.append(h)

    try:
        for batch in tqdm(dataloader, desc="Stage 1"):
            if tokens_seen >= total_tokens:
                break

            input_ids = batch["input_ids"].to(device)
            teacher_layer_outs = [t.to(device) for t in batch["layer_outputs"]]

            student_layer_outputs.clear()
            _ = student_model(input_ids)

            loss = torch.tensor(0.0, device=device)
            for s_out, t_out in zip(student_layer_outputs, teacher_layer_outs):
                loss = loss + cosine_distill_loss(s_out, t_out)
            loss = loss / len(teacher_layer_outs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_tokens = input_ids.numel()
            tokens_seen += batch_tokens
            losses.append(loss.item())
            step += 1

            if step % log_every == 0:
                print(f"Step {step} | tokens {tokens_seen:,} | loss {loss.item():.4f}", flush=True)
    finally:
        for h in hooks:
            h.remove()

    return losses
