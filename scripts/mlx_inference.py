#!/usr/bin/env python
"""MLX inference for the WhisperMamba student — benchmark vs PyTorch teacher.

Strategy
────────
1. Load mlx-community/whisper-tiny-mlx  (MLX Whisper backbone)
2. Replace each decoder self-attn with HedgeMambaMixerMLX
3. Load ALL trained weights (SSM + cross-attn + FFN + layer-norms) from
   our PyTorch checkpoint.  Encoder weights stay from the mlx-community model
   (they were frozen during training so they're identical).
4. Benchmark latency and WER vs. PyTorch teacher.

MLX advantage: lazy evaluation fuses the scan loop's per-step ops into
fewer Metal kernel dispatches (~2.3× scan speedup, ~3.7× end-to-end).

Usage:
    python scripts/mlx_inference.py
    python scripts/mlx_inference.py --n_samples 30
"""
import sys, time, warnings, logging, argparse
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, numpy as np
import mlx.core as mx
import mlx_whisper.load_models as lm
import mlx_whisper.decoding   as dec
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.student.mlx_hedge_mamba import HedgeMambaMixerMLX
from src.data.librispeech import make_librispeech_loaders, LibriSpeechCollator
from torch.utils.data import Subset, DataLoader

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",       default="checkpoints/whisper_mamba/whisper_mamba_final.pt")
parser.add_argument("--mlx_repo",   default="mlx-community/whisper-tiny-mlx")
parser.add_argument("--hf_model",   default="openai/whisper-tiny")
parser.add_argument("--state_size", type=int, default=64)
parser.add_argument("--n_samples",  type=int, default=20)
parser.add_argument("--warmup",     type=int, default=3)
args = parser.parse_args()

DEVICE = "mps"

# ── Weight helpers ────────────────────────────────────────────────────────────

def pt_to_mx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().float().numpy())


def build_mlx_student(pt_ckpt: dict, mlx_repo: str, state_size: int):
    """Load mlx-whisper backbone, patch SSM, load all trained weights."""
    model = lm.load_model(mlx_repo)
    dims  = model.dims

    for i, block in enumerate(model.decoder.blocks):
        # ── SSM self-attention ────────────────────────────────────────────
        mixer = HedgeMambaMixerMLX(
            dim=dims.n_text_state, num_heads=dims.n_text_head,
            state_size=state_size)
        p = f"backbone.model.decoder.layers.{i}.self_attn.mixer."
        # PyTorch conv weight (D,1,k) → MLX (D,k,1)
        conv_w = mx.transpose(pt_to_mx(pt_ckpt[p+"conv1d.weight"]), (0, 2, 1))
        mixer.update({
            "gate_proj":  {"weight": pt_to_mx(pt_ckpt[p+"gate_proj.weight"])},
            "conv_weight": conv_w,
            "x_proj":     {"weight": pt_to_mx(pt_ckpt[p+"x_proj.weight"])},
            "dt_proj":    {"weight": pt_to_mx(pt_ckpt[p+"dt_proj.weight"]),
                           "bias":   pt_to_mx(pt_ckpt[p+"dt_proj.bias"])},
            "A_log":      pt_to_mx(pt_ckpt[p+"A_log"]),
            "v_proj":     {"weight": pt_to_mx(pt_ckpt[p+"v_proj.weight"])},
            "hhog_k":     {"phi": {"weight": pt_to_mx(pt_ckpt[p+"hhog_k.phi.weight"]),
                                   "bias":   pt_to_mx(pt_ckpt[p+"hhog_k.phi.bias"])}},
            "hhog_q":     {"phi": {"weight": pt_to_mx(pt_ckpt[p+"hhog_q.phi.weight"]),
                                   "bias":   pt_to_mx(pt_ckpt[p+"hhog_q.phi.bias"])}},
            "out_proj":   {"weight": pt_to_mx(pt_ckpt[p+"out_proj.weight"])},
        })
        mx.eval(mixer.parameters())
        block["attn"] = mixer

        # ── Cross-attention (trained weights) ─────────────────────────────
        hp = f"backbone.model.decoder.layers.{i}."
        block.attn_ln.update({
            "weight": pt_to_mx(pt_ckpt[hp+"self_attn_layer_norm.weight"]),
            "bias":   pt_to_mx(pt_ckpt[hp+"self_attn_layer_norm.bias"]),
        })
        block.cross_attn.query.update({
            "weight": pt_to_mx(pt_ckpt[hp+"encoder_attn.q_proj.weight"]),
            "bias":   pt_to_mx(pt_ckpt[hp+"encoder_attn.q_proj.bias"]),
        })
        block.cross_attn.key.update({
            "weight": pt_to_mx(pt_ckpt[hp+"encoder_attn.k_proj.weight"]),
        })
        block.cross_attn.value.update({
            "weight": pt_to_mx(pt_ckpt[hp+"encoder_attn.v_proj.weight"]),
            "bias":   pt_to_mx(pt_ckpt[hp+"encoder_attn.v_proj.bias"]),
        })
        block.cross_attn.out.update({
            "weight": pt_to_mx(pt_ckpt[hp+"encoder_attn.out_proj.weight"]),
            "bias":   pt_to_mx(pt_ckpt[hp+"encoder_attn.out_proj.bias"]),
        })
        block.cross_attn_ln.update({
            "weight": pt_to_mx(pt_ckpt[hp+"encoder_attn_layer_norm.weight"]),
            "bias":   pt_to_mx(pt_ckpt[hp+"encoder_attn_layer_norm.bias"]),
        })
        # ── FFN ──────────────────────────────────────────────────────────
        block.mlp1.update({"weight": pt_to_mx(pt_ckpt[hp+"fc1.weight"]),
                            "bias":   pt_to_mx(pt_ckpt[hp+"fc1.bias"])})
        block.mlp2.update({"weight": pt_to_mx(pt_ckpt[hp+"fc2.weight"]),
                            "bias":   pt_to_mx(pt_ckpt[hp+"fc2.bias"])})
        block.mlp_ln.update({"weight": pt_to_mx(pt_ckpt[hp+"final_layer_norm.weight"]),
                              "bias":   pt_to_mx(pt_ckpt[hp+"final_layer_norm.bias"])})

    # ── Global decoder weights ────────────────────────────────────────────
    model.decoder.token_embedding.update({
        "weight": pt_to_mx(pt_ckpt["backbone.model.decoder.embed_tokens.weight"])
    })
    model.decoder.positional_embedding = pt_to_mx(
        pt_ckpt["backbone.model.decoder.embed_positions.weight"])
    model.decoder.ln.update({
        "weight": pt_to_mx(pt_ckpt["backbone.model.decoder.layer_norm.weight"]),
        "bias":   pt_to_mx(pt_ckpt["backbone.model.decoder.layer_norm.bias"]),
    })
    mx.eval(model.parameters())
    return model


# ── Load models ───────────────────────────────────────────────────────────────
print("Loading MLX student...", flush=True)
pt_ckpt    = torch.load(args.ckpt, weights_only=True, map_location="cpu")
mlx_student = build_mlx_student(pt_ckpt, args.mlx_repo, args.state_size)

print("Loading PyTorch teacher...", flush=True)
teacher   = WhisperForConditionalGeneration.from_pretrained(
    args.hf_model, torch_dtype=torch.float32).to(DEVICE).eval()
processor = WhisperProcessor.from_pretrained(args.hf_model)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
_, val_loader, _ = make_librispeech_loaders(
    model_id=args.hf_model, language="en", task="transcribe",
    train_split="train.100", val_split="validation",
    batch_size=1, max_label_length=448,
)
loader = DataLoader(
    Subset(val_loader.dataset, range(args.warmup + args.n_samples)),
    batch_size=1, shuffle=False, collate_fn=LibriSpeechCollator(), num_workers=0,
)
batches = list(loader)
refs    = [processor.decode(b["labels"][0].clamp(min=0), skip_special_tokens=True)
           for b in batches]

_GEN_OPTS = dec.DecodingOptions(
    language="en", task="transcribe", fp16=False,
    without_timestamps=True, suppress_blank=True)


def run_teacher(batch):
    with torch.no_grad():
        ids = teacher.generate(
            batch["input_features"].to(DEVICE),
            language="en", task="transcribe",
            max_new_tokens=128, repetition_penalty=1.1)
    return processor.decode(ids[0], skip_special_tokens=True)


def run_mlx(batch):
    # input_features: (1, n_mels=80, T=3000) in PyTorch → (1, T, n_mels) in MLX
    feats = mx.array(batch["input_features"].squeeze(0).numpy().T[None])
    return dec.decode(mlx_student, feats, _GEN_OPTS)[0].text.strip()


# ── Benchmark ─────────────────────────────────────────────────────────────────
print(f"\nBenchmarking {args.n_samples} samples  (warmup={args.warmup}, device={DEVICE})\n")

t_lats, s_lats, t_preds, s_preds = [], [], [], []
for i, batch in enumerate(batches):
    tag = "(warmup)" if i < args.warmup else f"[{i - args.warmup + 1}/{args.n_samples}]"

    t0 = time.perf_counter()
    tp = run_teacher(batch)
    torch.mps.synchronize()
    t_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sp = run_mlx(batch)
    s_ms = (time.perf_counter() - t0) * 1000

    print(f"  {tag}  teacher={t_ms:5.0f}ms  mlx={s_ms:5.0f}ms  {sp[:55]!r}", flush=True)

    if i >= args.warmup:
        t_lats.append(t_ms); s_lats.append(s_ms)
        t_preds.append(tp);  s_preds.append(sp)

# ── Results ───────────────────────────────────────────────────────────────────
t_m = np.mean(t_lats)
s_m = np.mean(s_lats)
refs_eval = refs[args.warmup:]

try:
    import jiwer
    t_wer = jiwer.wer(refs_eval, t_preds) * 100
    s_wer = jiwer.wer(refs_eval, s_preds) * 100
    wer_str = f"  WER:  Teacher {t_wer:.1f}%   MLX-SSM {s_wer:.1f}%"
except ImportError:
    wer_str = "  (install jiwer for WER)"

print(f"""
{'='*56}
  Results  ({args.n_samples} samples, device={DEVICE})
{'='*56}
  Teacher (PyTorch MPS, multi-head attn)   {t_m:>6.0f} ms
  Student (MLX SSM, fused Metal kernels)   {s_m:>6.0f} ms
  Speedup                                  {t_m/s_m:>6.2f}×
{wer_str}
{'='*56}

── Sample transcriptions ──────────────────────────────""")

for i in range(min(4, len(refs_eval))):
    print(f"  ref    : {refs_eval[i]!r}")
    print(f"  teacher: {t_preds[i]!r}")
    print(f"  mlx    : {s_preds[i]!r}")
    print()
