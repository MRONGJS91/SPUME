"""MetaShift dataset loader for the SPUME framework.

MetaShift (Liang & Zou, NeurIPS 2022) is a dataset for studying distribution
shifts where object classes appear with different co-occurring contexts in
train vs test splits. This creates a natural spurious correlation shift,
making it well-suited for evaluating SPUME.

Typical usage: the "Cat vs Dog" subset is a binary classification task where
train and test have different class-context distributions.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.biased_dataset import get_transform_biased

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _normalize_key(path: str) -> str:
    key = str(path).strip().replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key


def _infer_key_path(embed_path):
    embed_path = Path(embed_path)
    name = embed_path.name
    if name.endswith("_img_embeddings.pickle"):
        key_name = name[: -len("_img_embeddings.pickle")] + "_img_embedding_keys.pickle"
    else:
        key_name = embed_path.stem + "_keys.pickle"
    return embed_path.with_name(key_name)


class MetaShiftDataset(Dataset):
    """Dataset for MetaShift images with spurious context correlations.

    Parameters
    ----------
    data_root : str or Path
        Root directory containing MetaShift image subdirectories.
    metadata_path : str or Path
        Path to metadata CSV with columns:
        img_path, class_name, y, split, env, filename.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    transform : callable, optional
        Image transform to apply.
    concept_embed : str or Path, optional
        Path to concept embedding pickle file.
    return_metadata : bool
        If True, ``__getitem__`` returns ``(image, label, metadata_dict)``.
    """

    def __init__(
        self,
        data_root,
        metadata_path,
        split="train",
        transform=None,
        concept_embed=None,
        return_metadata=True,
    ):
        assert split in ("train", "val", "test"), f"invalid split = {split}"

        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path)
        self.split = split
        self.transform = transform
        self.concept_embed = concept_embed
        self.return_metadata = return_metadata

        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root does not exist: {self.data_root}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"metadata file does not exist: {self.metadata_path}"
            )

        metadata_df = pd.read_csv(self.metadata_path)
        required = {"img_path", "class_name", "y", "split", "env", "filename"}
        missing = required - set(metadata_df.columns)
        if missing:
            raise ValueError(
                f"metadata is missing required columns: {', '.join(sorted(missing))}"
            )

        split_mask = metadata_df["split"] == split
        self.metadata_df = metadata_df[split_mask].reset_index(drop=True)
        if len(self.metadata_df) == 0:
            raise ValueError(f"no samples found for split={split}")

        self.y_array = self.metadata_df["y"].astype(int).values
        self.class_name_array = self.metadata_df["class_name"].values
        self.env_array = self.metadata_df["env"].values
        self.filename_array = self.metadata_df["filename"].values
        self.img_path_array = self.metadata_df["img_path"].values
        self.img_key_array = np.array(
            [_normalize_key(path) for path in self.img_path_array]
        )

        self.n_classes = int(np.unique(self.y_array).size)
        env_names = sorted(self.metadata_df["env"].unique())
        self.env_name_to_idx = {name: idx for idx, name in enumerate(env_names)}
        self.p_array = (
            self.metadata_df["env"].map(self.env_name_to_idx).astype(int).values
        )
        self.confounder_array = self.p_array
        self.n_places = int(np.unique(self.p_array).size)
        self.group_array = (self.y_array * self.n_places + self.p_array).astype(int)
        self.n_groups = self.n_classes * self.n_places

        # Load concept embeddings if provided (skip for test split)
        if concept_embed and Path(concept_embed).exists() and split != "test":
            import pickle

            with open(concept_embed, "rb") as f:
                all_embeddings = pickle.load(f)
            key_path = _infer_key_path(concept_embed)
            if key_path.exists():
                with open(key_path, "rb") as f:
                    embedding_keys = [
                        _normalize_key(k) for k in pickle.load(f)
                    ]
                if len(embedding_keys) != len(all_embeddings):
                    raise ValueError(
                        f"concept key count ({len(embedding_keys)}) does not "
                        f"match embedding rows ({len(all_embeddings)})"
                    )
                key_to_row = {
                    key: row_idx for row_idx, key in enumerate(embedding_keys)
                }
                missing = [
                    key for key in self.img_key_array if key not in key_to_row
                ]
                if missing:
                    raise KeyError(
                        f"{len(missing)} {split} images are missing concept "
                        f"embeddings; first missing: {missing[0]}"
                    )
                row_indices = [key_to_row[key] for key in self.img_key_array]
                self.embeddings = all_embeddings[row_indices]
            else:
                # Assume embedding rows align 1:1 with non-test metadata rows
                non_test_mask = (
                    metadata_df["split"].astype(str).str.lower() != "test"
                ).values
                split_mask_arr = split_mask.values
                self.embeddings = all_embeddings[
                    split_mask_arr[non_test_mask]
                ]
        else:
            self.embeddings = None

    def __len__(self):
        return len(self.metadata_df)

    def _resolve_image_path(self, img_path):
        img_path = Path(img_path)
        candidates = [
            img_path,
            self.data_root / img_path,
            self.data_root.parent / img_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"image file not found. tried: {tried}")

    def __getitem__(self, idx):
        idx = int(idx)
        row = self.metadata_df.iloc[idx]
        image_path = self._resolve_image_path(row["img_path"])
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        y = int(row["y"])
        g = int(self.group_array[idx])
        p = int(self.p_array[idx])

        if self.return_metadata:
            metadata = {
                "img_path": row["img_path"],
                "class_name": row["class_name"],
                "split": row["split"],
                "env": row["env"],
                "filename": row["filename"],
                "group_id": int(row.get("group_id", self.group_array[idx])),
                "context_name": row.get("context_name", row["env"]),
                "y": int(row["y"]),
            }
            return image, y, metadata

        if self.embeddings is None:
            return image, y, g, p, p
        return image, y, g, p, self.embeddings[idx]


def build_default_transform(split):
    return get_transform_biased(
        target_resolution=(224, 224),
        train=(split == "train"),
        augment_data=(split == "train"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Minimal test for MetaShiftDataset."
    )
    parser.add_argument(
        "--data-root",
        default=r"D:\SPUME\datasets\metashift",
        help="MetaShift dataset root",
    )
    parser.add_argument(
        "--metadata-path",
        default=r"D:\SPUME\datasets\metashift\metadata.csv",
        help="Metadata CSV path",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split",
    )
    args = parser.parse_args()

    dataset = MetaShiftDataset(
        data_root=args.data_root,
        metadata_path=args.metadata_path,
        split=args.split,
        transform=build_default_transform(args.split),
        return_metadata=True,
    )
    image, label, metadata = dataset[0]

    print(f"dataset length: {len(dataset)}")
    print(f"first image shape: {tuple(image.shape)}")
    print(f"label type: {type(label).__name__}, value: {label}")
    print(f"metadata: {metadata}")
    print(f"n_classes: {dataset.n_classes}")
    print(f"n_places (contexts): {dataset.n_places}")
    print(f"n_groups: {dataset.n_groups}")
    print(f"classes: {np.unique(dataset.class_name_array).tolist()}")
    print(f"contexts: {sorted(dataset.env_name_to_idx.keys())}")


if __name__ == "__main__":
    main()
