"""Compute perplexity on a token stream."""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def compute_perplexity(
    model,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(tqdm(dataloader, desc="PPL eval")):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    return math.exp(total_loss / total_tokens)
