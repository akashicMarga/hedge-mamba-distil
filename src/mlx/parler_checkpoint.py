"""Checkpoint save/load for ParlerMambaMLX.

Mirrors src/mlx/checkpoint.py (Whisper) with:
  save() / load()  — identical (reuse checkpoint.save/load)
  load_from_pt_checkpoint()  — maps PyTorch WhisperMambaStudent-style key
    paths to the ParlerMambaMLX parameter tree, handling:
      - SSM layer weights (gate_proj, conv, x_proj, dt_proj, A_log,
        v_proj, hhog_k/q, out_proj)
      - Cross-attention (q/k/v/out per layer)
      - FFN (fc1, fc2)
      - LayerNorms (self_attn_ln, cross_attn_ln, ffn_ln)
      - LM heads
      - embed_prompts, embed_positions, embed_tokens (9 codebooks)

Key naming convention from hedge-mamba-distil's PyTorch checkpoints:
    backbone.decoder.model.decoder.layers.{i}.<sub_path>
"""
import json
import torch
import mlx.core as mx
from pathlib import Path

from src.mlx import checkpoint as _base_ckpt


def save(model, path: str, meta: dict | None = None) -> None:
    _base_ckpt.save(model, path, meta)


def load(model, path: str) -> dict:
    return _base_ckpt.load(model, path)


# ---------------------------------------------------------------------------
# PT → MLX weight bridge
# ---------------------------------------------------------------------------


def _pt(t) -> mx.array:
    return mx.array(t.detach().cpu().float().numpy())


def load_from_pt_checkpoint(
    student,          # ParlerMambaMLX
    pt_ckpt_path: str,
) -> None:
    """Copy weights from a ParlerMambaStudent .pt checkpoint into student.

    The PyTorch checkpoint is saved by ParlerMambaStudent.export_mamba_weights()
    or via torch.save(student.state_dict()).  We map each key path to the
    corresponding MLX module path inside student.model.

    T5 encoder weights are skipped — they are frozen and already loaded
    from the teacher (IndicParlerTTS.from_pretrained).
    """
    pt = torch.load(pt_ckpt_path, weights_only=True, map_location="cpu")
    dec = student.model.decoder  # ParlerMambaDecoder (MLX)

    prefix_dec = "backbone.decoder.model.decoder"

    for i, layer in enumerate(dec.layers):
        lp = f"{prefix_dec}.layers.{i}"
        sp = f"{lp}.self_attn.mixer."   # SSM mixer path in PT state dict
        hp = f"{lp}."                  # layer-level path

        # ── SSM (ParlerMambaSelfAttnMLX.mixer = HedgeMambaMixerMLX) ────────
        # PyTorch conv1d weight: (D, 1, k) → MLX (D, k, 1)
        conv_w = mx.transpose(_pt(pt[sp + "conv1d.weight"]), (0, 2, 1))
        layer.self_attn.mixer.update({
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

        # ── Self-attention LayerNorm ────────────────────────────────────
        layer.self_attn_ln.update({
            "weight": _pt(pt[hp + "self_attn_layer_norm.weight"]),
            "bias":   _pt(pt[hp + "self_attn_layer_norm.bias"]),
        })

        # ── Cross-attention ──────────────────────────────────────────
        # HF key: encoder_attn.{q,k,v,out}_proj; MLX: cross_attn.{q,k,v,out}
        layer.cross_attn.q.update({"weight": _pt(pt[hp + "encoder_attn.q_proj.weight"])})
        layer.cross_attn.k.update({"weight": _pt(pt[hp + "encoder_attn.k_proj.weight"])})
        layer.cross_attn.v.update({"weight": _pt(pt[hp + "encoder_attn.v_proj.weight"])})
        layer.cross_attn.out.update({"weight": _pt(pt[hp + "encoder_attn.out_proj.weight"])})
        layer.cross_attn_ln.update({
            "weight": _pt(pt[hp + "encoder_attn_layer_norm.weight"]),
            "bias":   _pt(pt[hp + "encoder_attn_layer_norm.bias"]),
        })

        # ── FFN ─────────────────────────────────────────────────────────
        layer.fc1.update({"weight": _pt(pt[hp + "fc1.weight"])})
        layer.fc2.update({"weight": _pt(pt[hp + "fc2.weight"])})
        layer.ffn_ln.update({
            "weight": _pt(pt[hp + "final_layer_norm.weight"]),
            "bias":   _pt(pt[hp + "final_layer_norm.bias"]),
        })

    # ── Decoder-level weights ───────────────────────────────────────
    # embed_tokens: 9 codebook embeddings
    for k in range(len(dec.embed_tokens)):
        dec.embed_tokens[k].update({
            "weight": _pt(
                pt[f"{prefix_dec}.embed_tokens.{k}.weight"]
            )
        })

    dec.embed_positions.update({
        "weight": _pt(pt[f"{prefix_dec}.embed_positions.weight"])
    })
    dec.final_ln.update({
        "weight": _pt(pt[f"{prefix_dec}.layer_norm.weight"]),
        "bias":   _pt(pt[f"{prefix_dec}.layer_norm.bias"]),
    })
    dec.lm_heads.update({
        "weight": _pt(pt["backbone.decoder.lm_heads.weight"])
    })

    # embed_prompts
    student.model.embed_prompts.update({
        "weight": _pt(pt["backbone.embed_prompts.weight"])
    })

    mx.eval(student.parameters())
    print(f"[parler_checkpoint] Loaded PT checkpoint: {pt_ckpt_path}", flush=True)
