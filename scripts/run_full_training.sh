#!/bin/bash
# Full overnight ParlerTTS-Mamba distillation — Hindi subset
# Run with: caffeinate -d -i -m bash scripts/run_full_training.sh
set -e

PYTHON=/Users/akashsingh/miniconda3/envs/p11/bin/python
MLX=/Users/akashsingh/Documents/exps/mlx-audio-train
LOG=./logs/full_training_$(date +%Y%m%d_%H%M%S).log
STATE_SIZE=128      # Ns=256 after Hedgehog doubling — good quality/speed tradeoff
MAX_AUDIO=256       # ~6 seconds of audio
GRAD_ACCUM=4        # effective batch = 4
TRAIN_SAMPLES=3000  # Hindi train samples
TEST_SAMPLES=500    # Hindi test samples
TRAIN_CACHE=./data/parler_distil_en/train
TEST_CACHE=./data/parler_distil_en/test
CKPT_DIR=./checkpoints/parler_mamba_en

mkdir -p logs "$CKPT_DIR"

echo "================================================================" | tee -a "$LOG"
echo "Full overnight training started: $(date)" | tee -a "$LOG"
echo "  state_size=$STATE_SIZE  max_audio=$MAX_AUDIO  grad_accum=$GRAD_ACCUM" | tee -a "$LOG"
echo "  train_samples=$TRAIN_SAMPLES  test_samples=$TEST_SAMPLES" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"

# ── Step 1: Preprocess Hindi train split ─────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] STEP 1/4: Preprocessing Hindi train ($TRAIN_SAMPLES samples)..." | tee -a "$LOG"

if [ -d "$TRAIN_CACHE" ] && [ "$(ls -A $TRAIN_CACHE/*.npz 2>/dev/null | wc -l)" -ge "$TRAIN_SAMPLES" ]; then
    echo "  Already cached — skipping." | tee -a "$LOG"
else
    $PYTHON scripts/run_parler_preprocess.py \
        --mlx_audio_train "$MLX" \
        --split train \
        --max_samples "$TRAIN_SAMPLES" \
        --max_audio_len_s 6.0 \
        --out_dir ./data/parler_distil_en \
        2>&1 | tee -a "$LOG"
fi

# ── Step 2: Preprocess English test split ────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] STEP 2/4: Preprocessing English test ($TEST_SAMPLES samples)..." | tee -a "$LOG"

if [ -d "$TEST_CACHE" ] && [ "$(ls -A $TEST_CACHE/*.npz 2>/dev/null | wc -l)" -ge "$TEST_SAMPLES" ]; then
    echo "  Already cached — skipping." | tee -a "$LOG"
else
    $PYTHON scripts/run_parler_preprocess.py \
        --mlx_audio_train "$MLX" \
        --split test \
        --max_samples "$TEST_SAMPLES" \
        --max_audio_len_s 6.0 \
        --out_dir ./data/parler_distil_en \
        2>&1 | tee -a "$LOG"
fi

echo "[$(date +%H:%M:%S)] Preprocessing done." | tee -a "$LOG"

# ── Step 3: Stage 1 — cosine distillation (3 epochs) ─────────────────────────
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] STEP 3/4: Stage 1 — cosine distillation (1 epoch)..." | tee -a "$LOG"

$PYTHON scripts/run_parler_stage1.py \
    --mlx_audio_train "$MLX" \
    --train_cache "$TRAIN_CACHE" \
    --val_cache   "$TEST_CACHE" \
    --state_size  "$STATE_SIZE" \
    --epochs      1 \
    --batch_size  1 \
    --grad_accumulation "$GRAD_ACCUM" \
    --lr          3e-4 \
    --warmup_steps 200 \
    --max_audio_len "$MAX_AUDIO" \
    --log_every   100 \
    --eval_every  500 \
    --checkpoint_dir "$CKPT_DIR" \
    --tb_log_dir  runs/ \
    2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] Stage 1 done." | tee -a "$LOG"

# ── Step 4: Stage 2 — CE + scheduled sampling (1 epoch) ──────────────────────
echo "" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] STEP 4/4: Stage 2 — CE + scheduled sampling (1 epoch)..." | tee -a "$LOG"

$PYTHON scripts/run_parler_stage2.py \
    --mlx_audio_train "$MLX" \
    --train_cache "$TRAIN_CACHE" \
    --val_cache   "$TEST_CACHE" \
    --stage1_ckpt "$CKPT_DIR/stage1_epoch_1" \
    --state_size  "$STATE_SIZE" \
    --epochs      1 \
    --batch_size  1 \
    --grad_accumulation "$GRAD_ACCUM" \
    --lr          1e-4 \
    --warmup_steps 200 \
    --ss_max_p    0.5 \
    --max_audio_len "$MAX_AUDIO" \
    --log_every   100 \
    --eval_every  500 \
    --checkpoint_dir "$CKPT_DIR" \
    --tb_log_dir  runs/ \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "ALL STAGES COMPLETE: $(date)" | tee -a "$LOG"
echo "Checkpoints: $CKPT_DIR" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
