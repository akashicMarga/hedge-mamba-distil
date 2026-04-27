"""Push WhisperMamba checkpoints and model card to Hugging Face Hub."""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


REPO_ID = "akashicmarga/whisper-tiny-hedgemamba"
BASE = Path(__file__).parent.parent

CHECKPOINT_FILES = [
    # (local path, path-in-repo)
    ("checkpoints/whisper_mamba/whisper_mamba_final.pt", "pytorch/whisper_mamba_final.pt"),
    ("checkpoints/whisper_mamba/stage1_final.pt",        "pytorch/stage1_final.pt"),
    ("checkpoints/mlx/whisper_mamba_mlx_final.npz",      "mlx/whisper_mamba_mlx_final.npz"),
    ("checkpoints/mlx/whisper_mamba_mlx_final.json",     "mlx/whisper_mamba_mlx_final.json"),
]


def main(dry_run: bool = False) -> None:
    api = HfApi()

    if not dry_run:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{REPO_ID}")

    # Upload model card
    card_path = BASE / "hf_model_card.md"
    if not card_path.exists():
        raise FileNotFoundError("hf_model_card.md not found — run after creating it")
    if dry_run:
        print(f"[dry-run] would upload {card_path} → README.md")
    else:
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Add model card",
        )
        print("Uploaded README.md")

    # Upload checkpoints
    for local_rel, repo_path in CHECKPOINT_FILES:
        local = BASE / local_rel
        if not local.exists():
            print(f"SKIP (not found): {local_rel}")
            continue
        size_mb = local.stat().st_size / 1024 / 1024
        if dry_run:
            print(f"[dry-run] would upload {local_rel} ({size_mb:.0f} MB) → {repo_path}")
        else:
            print(f"Uploading {local_rel} ({size_mb:.0f} MB) ...")
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"Add {repo_path}",
            )
            print(f"  done → {repo_path}")

    if not dry_run:
        print(f"\nDone. View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
