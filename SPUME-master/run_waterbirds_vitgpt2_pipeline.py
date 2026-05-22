import argparse
import os
import subprocess
import sys
from pathlib import Path

from extract_concepts import (
    VITGPT2_CAPTIONING,
    get_concept_embeddings,
    get_concepts,
    get_data_folder,
)


def run_cmd(cmd, cwd):
    print(f"RUN: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="waterbirds")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_dir", default=r"d:/SPUME/vit-gpt2-model")
    parser.add_argument(
        "--config", default=r"d:/SPUME/SPUME-master/config/waterbirds_vitgpt2_real.yaml"
    )
    parser.add_argument("--threshold", type=int, default=10)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    img_path, csv_path = get_data_folder(args.dataset)

    caption_model = VITGPT2_CAPTIONING(local_dir=args.model_dir)
    caption_path = caption_model.get_img_captions(
        img_path,
        csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"CAPTIONS_READY: {caption_path}", flush=True)

    concept_path = get_concepts(caption_path)
    get_concept_embeddings(concept_path, threshold=args.threshold)
    print("CONCEPTS_READY", flush=True)

    run_cmd(
        [sys.executable, "train_meta_spurious.py", "--config", args.config],
        cwd=str(repo_dir),
    )

    output_dir = Path(r"d:/SPUME/how_to_run/meta_spurious_exprs")
    latest = max(
        [p for p in output_dir.iterdir() if p.is_dir() and "real_vitgpt2" in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    checkpoint = latest / "pseudo_unbiased_model.pt"
    figures_dir = latest / "figures"
    run_cmd(
        [
            sys.executable,
            "visualize_waterbirds_results.py",
            "--config",
            args.config,
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(figures_dir),
        ],
        cwd=str(repo_dir),
    )
    print(f"PIPELINE_DONE: {latest}", flush=True)


if __name__ == "__main__":
    main()
