import argparse
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file does not exist: {metadata_path}")
    df = pd.read_csv(metadata_path)
    required_columns = {"img_path", "class_name", "y", "split", "env", "filename"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"metadata is missing required columns: {missing_str}")
    return df


def resolve_image_path(data_root, img_path):
    data_root = Path(data_root)
    img_path = Path(img_path)
    candidates = [
        img_path,
        data_root / img_path,
        data_root.parent / img_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"image file not found. tried: {tried}")


def print_basic_stats(df):
    print("== Split counts ==")
    split_counts = df["split"].value_counts().sort_index()
    for split, count in split_counts.items():
        print(f"- {split}: {count}")
    print()

    print("== Class counts ==")
    class_counts = df["class_name"].value_counts().sort_index()
    for class_name, count in class_counts.items():
        print(f"- {class_name}: {count}")
    print()

    print("== Env counts ==")
    env_counts = df["env"].value_counts().sort_index()
    for env_name, count in env_counts.items():
        print(f"- {env_name}: {count}")
    print()

    print("== Split x Class counts ==")
    split_class = (
        df.groupby(["split", "class_name"]).size().reset_index(name="count")
    )
    for split in ["train", "val", "test"]:
        print(f"[{split}]")
        subset = split_class[split_class["split"] == split]
        for _, row in subset.sort_values("class_name").iterrows():
            print(f"  - {row['class_name']}: {int(row['count'])}")
        print()

    print("== Split x Env counts ==")
    split_env = df.groupby(["split", "env"]).size().reset_index(name="count")
    for split in ["train", "val", "test"]:
        print(f"[{split}]")
        subset = split_env[split_env["split"] == split]
        for _, row in subset.sort_values("env").iterrows():
            print(f"  - {row['env']}: {int(row['count'])}")
        print()


def sample_rows(df, sample_per_split, seed):
    rng = random.Random(seed)
    sampled = []
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        if len(split_df) == 0:
            continue
        indices = list(split_df.index)
        rng.shuffle(indices)
        selected = indices[: min(sample_per_split, len(indices))]
        sampled.extend(df.loc[selected].to_dict("records"))
    return sampled


def make_figure(rows, data_root, output_path):
    if not rows:
        print("No rows sampled for visualization.")
        return

    cols = min(4, len(rows))
    rows_n = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 4.2 * rows_n))
    if not isinstance(axes, (list, tuple)):
        axes = axes.reshape(rows_n, cols)

    axes_flat = axes.reshape(-1)
    for ax in axes_flat:
        ax.axis("off")

    for ax, row in zip(axes_flat, rows):
        image_path = resolve_image_path(data_root, row["img_path"])
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        ax.imshow(image)
        ax.set_title(
            f"{row['split']} | y={row['y']}\n{row['class_name']} | {row['env']}",
            fontsize=9,
        )
        ax.axis("off")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved visualization to {output_path}")


def print_sample_rows(rows):
    print("== Sample rows ==")
    for row in rows:
        print(
            f"- split={row['split']}, class={row['class_name']}, y={row['y']}, env={row['env']}, img_path={row['img_path']}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Sanity-check Spawrious dataset.")
    parser.add_argument(
        "--data-root",
        default=r"D:\SPUME\spawrious224__o2o_easy",
        help="Spawrious dataset root",
    )
    parser.add_argument(
        "--metadata-path",
        default=r"D:\SPUME\SPUME-master\data\spawrious_o2o_easy_metadata.csv",
        help="Metadata CSV path",
    )
    parser.add_argument(
        "--sample-per-split",
        type=int,
        default=4,
        help="Number of random samples to visualize per split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output",
        default=r"D:\SPUME\how_to_run\meta_spurious_exprs\figures\spawrious_o2o_easy_sanity_check.png",
        help="Output path for the visualization figure",
    )
    args = parser.parse_args()

    df = load_metadata(args.metadata_path)
    print_basic_stats(df)
    sampled_rows = sample_rows(df, args.sample_per_split, args.seed)
    print_sample_rows(sampled_rows)
    make_figure(sampled_rows, args.data_root, args.output)


if __name__ == "__main__":
    main()
