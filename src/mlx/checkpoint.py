"""Checkpoint save/load for WhisperMambaMLX.

Two formats:
  Native MLX  — model.save_weights("path.npz") / model.load_weights("path.npz")
                + JSON sidecar for metadata (epoch, step, loss, config).
  PT bridge   — load_from_pt_checkpoint(student, pt_path)
                copies the trained weights from the PyTorch checkpoint into the
                MLX model (same mapping as scripts/mlx_inference.py).
"""
import json
import torch
import mlx.core as mx
from pathlib import Path


def save(model, path: str, meta: dict | None = None) -> None:
    """Save model weights (.npz) and optional JSON sidecar."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    npz = p.with_suffix(".npz")
    model.save_weights(str(npz))
    if meta:
        with open(p.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
    print(f"Checkpoint → {npz}", flush=True)


def load(model, path: str) -> dict:
    """Load weights from .npz, return JSON sidecar dict (empty if absent)."""
    p = Path(path)
    npz = p if p.suffix == ".npz" else p.with_suffix(".npz")
    model.load_weights(str(npz))
    mx.eval(model.parameters())
    meta_path = npz.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


# ── PT weight bridge ──────────────────────────────────────────────────────────

def _pt(t) -> mx.array:
    return mx.array(t.detach().cpu().float().numpy())


def load_from_pt_checkpoint(student, pt_ckpt_path: str) -> None:
    """Copy weights from a PyTorch WhisperMambaStudent checkpoint into student.

    Encoder is skipped — it was frozen during PyTorch training so the
    mlx-community model weights are identical.

    Key mapping follows mlx_inference.py exactly.
    """
    pt = torch.load(pt_ckpt_path, weights_only=True, map_location="cpu")
    m = student.model  # underlying mlx-whisper Whisper model

    for i, block in enumerate(m.decoder.blocks):
        sp = f"backbone.model.decoder.layers.{i}.self_attn.mixer."
        hp = f"backbone.model.decoder.layers.{i}."

        # SSM (HedgeMambaMixerMLX)
        # PyTorch conv weight (D, 1, k) → MLX (D, k, 1)
        conv_w = mx.transpose(_pt(pt[sp + "conv1d.weight"]), (0, 2, 1))
        block.attn.update({
            "gate_proj":  {"weight": _pt(pt[sp + "gate_proj.weight"])},
            "conv_weight": conv_w,
            "x_proj":     {"weight": _pt(pt[sp + "x_proj.weight"])},
            "dt_proj":    {"weight": _pt(pt[sp + "dt_proj.weight"]),
                           "bias":   _pt(pt[sp + "dt_proj.bias"])},
            "A_log":      _pt(pt[sp + "A_log"]),
            "v_proj":     {"weight": _pt(pt[sp + "v_proj.weight"])},
            "hhog_k":     {"phi": {"weight": _pt(pt[sp + "hhog_k.phi.weight"]),
                                   "bias":   _pt(pt[sp + "hhog_k.phi.bias"])}},
            "hhog_q":     {"phi": {"weight": _pt(pt[sp + "hhog_q.phi.weight"]),
                                   "bias":   _pt(pt[sp + "hhog_q.phi.bias"])}},
            "out_proj":   {"weight": _pt(pt[sp + "out_proj.weight"])},
        })

        # Self-attn layer norm
        block.attn_ln.update({
            "weight": _pt(pt[hp + "self_attn_layer_norm.weight"]),
            "bias":   _pt(pt[hp + "self_attn_layer_norm.bias"]),
        })

        # Cross-attention
        block.cross_attn.query.update({
            "weight": _pt(pt[hp + "encoder_attn.q_proj.weight"]),
            "bias":   _pt(pt[hp + "encoder_attn.q_proj.bias"]),
        })
        block.cross_attn.key.update({
            "weight": _pt(pt[hp + "encoder_attn.k_proj.weight"]),
        })
        block.cross_attn.value.update({
            "weight": _pt(pt[hp + "encoder_attn.v_proj.weight"]),
            "bias":   _pt(pt[hp + "encoder_attn.v_proj.bias"]),
        })
        block.cross_attn.out.update({
            "weight": _pt(pt[hp + "encoder_attn.out_proj.weight"]),
            "bias":   _pt(pt[hp + "encoder_attn.out_proj.bias"]),
        })
        block.cross_attn_ln.update({
            "weight": _pt(pt[hp + "encoder_attn_layer_norm.weight"]),
            "bias":   _pt(pt[hp + "encoder_attn_layer_norm.bias"]),
        })

        # FFN
        block.mlp1.update({
            "weight": _pt(pt[hp + "fc1.weight"]),
            "bias":   _pt(pt[hp + "fc1.bias"]),
        })
        block.mlp2.update({
            "weight": _pt(pt[hp + "fc2.weight"]),
            "bias":   _pt(pt[hp + "fc2.bias"]),
        })
        block.mlp_ln.update({
            "weight": _pt(pt[hp + "final_layer_norm.weight"]),
            "bias":   _pt(pt[hp + "final_layer_norm.bias"]),
        })

    # Global decoder weights
    m.decoder.token_embedding.update({
        "weight": _pt(pt["backbone.model.decoder.embed_tokens.weight"])
    })
    m.decoder.update({
        "positional_embedding": _pt(
            pt["backbone.model.decoder.embed_positions.weight"]
        )
    })
    m.decoder.ln.update({
        "weight": _pt(pt["backbone.model.decoder.layer_norm.weight"]),
        "bias":   _pt(pt["backbone.model.decoder.layer_norm.bias"]),
    })

    mx.eval(student.parameters())
    print(f"Loaded PT checkpoint: {pt_ckpt_path}", flush=True)
