"""Thin wrapper to run lm-evaluation-harness tasks on HedgeMambaStudent."""
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


def evaluate_tasks(
    model,
    tokenizer,
    tasks: list[str],
    device: str = "cpu",
    batch_size: int = 8,
) -> dict:
    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size)
    results = evaluator.simple_evaluate(model=lm, tasks=tasks, log_samples=False)
    return results["results"]
