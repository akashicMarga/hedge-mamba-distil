def __getattr__(name):
    if name in ("WhisperMambaMLX", "get_teacher_hiddens"):
        from .model import WhisperMambaMLX, get_teacher_hiddens
        return {"WhisperMambaMLX": WhisperMambaMLX, "get_teacher_hiddens": get_teacher_hiddens}[name]
    if name in ("cosine_distill_loss", "ce_loss"):
        from .loss import cosine_distill_loss, ce_loss
        return {"cosine_distill_loss": cosine_distill_loss, "ce_loss": ce_loss}[name]
    raise AttributeError(f"module 'src.mlx' has no attribute {name!r}")
