"""Export a trained ParlerMambaStudent checkpoint to MLX-loadable .npz.

After Stage 2 training, run:
    python scripts/export_parler_mamba.py \\
        --checkpoint ./checkpoints/parler_mamba/stage2_epoch_5.pt \\
        --out        ./checkpoints/parler_mamba_stage2.npz

The .npz can then be loaded in mlx-audio-train:
    from models.indic_parler_tts.mamba_model import IndicParlerTTSMamba
    model = IndicParlerTTSMamba.load_from_pt_checkpoint(
        "./checkpoints/parler_mamba_stage2.npz"
    )
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def export(checkpoint_path: str, out_path: str) -> None:
    import torch
    from src.student.parler_mamba import ParlerMambaStudent
    from parler_tts import ParlerTTSForConditionalGeneration

    print(f"Loading teacher for model config...")
    teacher = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts"
    )
    student = ParlerMambaStudent(teacher)
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    student.load_state_dict(state, strict=False)
    student.eval()

    print(f"Exporting SSM weights to: {out_path}")
    student.export_mamba_weights(out_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.out)
