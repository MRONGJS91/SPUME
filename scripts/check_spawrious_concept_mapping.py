from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "spawrious-o2o-easy": {
        "data_root": Path("spawrious224__o2o_easy"),
        "metadata": Path("SPUME-master/data/spawrious_o2o_easy_metadata.csv"),
    },
    "spawrious-o2o-medium": {
        "data_root": Path("spawrious224__o2o_medium"),
        "metadata": Path("SPUME-master/data/spawrious_o2o_medium_metadata.csv"),
    },
}
MODELS = ("blip", "vit-gpt2")


def normalize_key(path) -> str:
    key = str(path).strip().replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key


def key_type(key: str) -> str:
    p = Path(key)
    if p.is_absolute():
        return "absolute_path"
    if "/" in key or "\\" in key:
        return "relative_path"
    return "basename"


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def read_caption_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    keys: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                keys.append(normalize_key(row[0]))
    return keys


def infer_paths(data_root: Path, model: str) -> dict[str, Path]:
    prefix = f"{model}_extracted_concepts_0_0"
    return {
        "captions": data_root / f"{model}_captions.csv",
        "concepts": data_root / f"{prefix}.pickle",
        "embeddings": data_root / f"{prefix}_img_embeddings.pickle",
        "vocab": data_root / f"{prefix}_vocab.pickle",
        "keys": data_root / f"{prefix}_img_embedding_keys.pickle",
        "keys_csv": data_root / f"{prefix}_img_embedding_keys.csv",
    }


def duplicate_examples(keys: list[str], limit: int = 5) -> list[dict[str, object]]:
    counts = Counter(keys)
    examples = []
    for key, count in counts.items():
        if count > 1:
            examples.append({"key": key, "count": count})
        if len(examples) >= limit:
            break
    return examples


def basename_duplicate_count(keys: list[str]) -> int:
    basenames = [Path(key).name for key in keys]
    return len(basenames) - len(set(basenames))


def write_key_files(paths: dict[str, Path], concept_keys: list[str]) -> None:
    with paths["keys"].open("wb") as f:
        pickle.dump(concept_keys, f)
    with paths["keys_csv"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "image_key"])
        for row_index, key in enumerate(concept_keys):
            writer.writerow([row_index, key])


def analyze_one(root: Path, dataset_name: str, model: str, write_keys: bool) -> dict[str, object]:
    info = DATASETS[dataset_name]
    data_root = root / info["data_root"]
    metadata_path = root / info["metadata"]
    paths = infer_paths(data_root, model)

    metadata_df = pd.read_csv(metadata_path)
    metadata_df["split"] = metadata_df["split"].astype(str).str.lower()
    dataset_all_keys = [normalize_key(p) for p in metadata_df["img_path"].tolist()]
    metadata_non_test = metadata_df[metadata_df["split"] != "test"].reset_index(drop=True)
    dataset_non_test_keys = [normalize_key(p) for p in metadata_non_test["img_path"].tolist()]

    caption_keys = read_caption_keys(paths["captions"])
    concept_arr = load_pickle(paths["concepts"])
    concept_keys = [normalize_key(row[1]) for row in concept_arr]
    concept_lists = [row[2] for row in concept_arr]
    embeddings = np.asarray(load_pickle(paths["embeddings"]))
    existing_key_file_keys = []
    if paths["keys"].exists():
        existing_key_file_keys = [normalize_key(k) for k in load_pickle(paths["keys"])]

    if write_keys:
        write_key_files(paths, concept_keys)
        existing_key_file_keys = list(concept_keys)

    dataset_key_set = set(dataset_non_test_keys)
    concept_key_set = set(concept_keys)
    missing_concepts = [key for key in dataset_non_test_keys if key not in concept_key_set]
    extra_concepts = [key for key in concept_keys if key not in dataset_key_set]
    order_mismatches = sum(
        1
        for dataset_key, concept_key in zip(dataset_non_test_keys, concept_keys)
        if dataset_key != concept_key
    )
    key_types = Counter(key_type(key) for key in concept_keys)

    split_rows = {}
    for split in ["train", "val", "test"]:
        split_df = metadata_df[metadata_df["split"] == split]
        split_keys = [normalize_key(p) for p in split_df["img_path"].tolist()]
        split_rows[split] = {
            "samples": int(len(split_df)),
            "classes": int(split_df["class_name"].nunique()),
            "unique_img_path_keys": int(len(set(split_keys))),
            "duplicate_img_path_keys": int(len(split_keys) - len(set(split_keys))),
            "duplicate_basenames": int(basename_duplicate_count(split_keys)),
        }

    return {
        "dataset": dataset_name,
        "model": model,
        "metadata_path": str(metadata_path),
        "caption_path": str(paths["captions"]),
        "concept_path": str(paths["concepts"]),
        "embedding_path": str(paths["embeddings"]),
        "key_file_path": str(paths["keys"]),
        "key_csv_path": str(paths["keys_csv"]),
        "key_file_written": bool(write_keys),
        "concept_key_type_counts": dict(key_types),
        "concept_key_mode": key_types.most_common(1)[0][0] if key_types else "none",
        "dataset_samples_total": int(len(metadata_df)),
        "dataset_non_test_samples": int(len(metadata_non_test)),
        "caption_rows": int(len(caption_keys)),
        "concept_entries": int(len(concept_arr)),
        "concept_mentions_total": int(sum(len(items) for items in concept_lists)),
        "embedding_shape": [int(x) for x in embeddings.shape],
        "concept_key_unique_count": int(len(set(concept_keys))),
        "concept_duplicate_key_count": int(len(concept_keys) - len(set(concept_keys))),
        "concept_duplicate_key_examples": duplicate_examples(concept_keys),
        "concept_basename_duplicate_count": int(basename_duplicate_count(concept_keys)),
        "caption_key_unique_count": int(len(set(caption_keys))),
        "caption_duplicate_key_count": int(len(caption_keys) - len(set(caption_keys))),
        "dataset_missing_concept_count": int(len(missing_concepts)),
        "dataset_missing_concept_examples": missing_concepts[:10],
        "concept_without_dataset_count": int(len(extra_concepts)),
        "concept_without_dataset_examples": extra_concepts[:10],
        "dataset_concept_order_mismatch_count": int(order_mismatches),
        "embedding_rows_match_concepts": bool(embeddings.shape[0] == len(concept_arr)),
        "embedding_rows_match_dataset_non_test": bool(
            embeddings.shape[0] == len(metadata_non_test)
        ),
        "existing_key_file_entries": int(len(existing_key_file_keys)),
        "existing_key_file_unique_count": int(len(set(existing_key_file_keys))),
        "existing_key_file_matches_concepts": bool(existing_key_file_keys == concept_keys),
        "split_summary": split_rows,
    }


def write_markdown(report: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Spawrious Concept Mapping Report",
        "",
        "| dataset | model | key mode | concept entries | dataset non-test | key unique | dup keys | missing concepts | extra concepts | order mismatches | embedding shape |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report:
        lines.append(
            "| {dataset} | {model} | {concept_key_mode} | {concept_entries} | "
            "{dataset_non_test_samples} | {concept_key_unique_count} | "
            "{concept_duplicate_key_count} | {dataset_missing_concept_count} | "
            "{concept_without_dataset_count} | {dataset_concept_order_mismatch_count} | "
            "{embedding_shape} |".format(**row)
        )
    lines.append("")
    lines.append("## Split Summary")
    for row in report:
        lines.append("")
        lines.append(f"### {row['dataset']} / {row['model']}")
        lines.append(
            "| split | samples | classes | unique img_path keys | duplicate img_path keys | duplicate basenames |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for split, split_row in row["split_summary"].items():
            lines.append(
                f"| {split} | {split_row['samples']} | {split_row['classes']} | "
                f"{split_row['unique_img_path_keys']} | {split_row['duplicate_img_path_keys']} | "
                f"{split_row['duplicate_basenames']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--write-key-files",
        action="store_true",
        help="Write *_img_embedding_keys.pickle/csv next to existing embeddings.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("how_to_run/spawrious_concept_mapping_report.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("how_to_run/spawrious_concept_mapping_report.md"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    report = []
    for dataset_name in DATASETS:
        for model in MODELS:
            report.append(analyze_one(root, dataset_name, model, args.write_key_files))

    json_out = args.json_out if args.json_out.is_absolute() else root / args.json_out
    md_out = args.md_out if args.md_out.is_absolute() else root / args.md_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_out)

    for row in report:
        print(
            f"{row['dataset']} / {row['model']}: "
            f"concept_entries={row['concept_entries']}, "
            f"dataset_non_test={row['dataset_non_test_samples']}, "
            f"unique_keys={row['concept_key_unique_count']}, "
            f"dup_keys={row['concept_duplicate_key_count']}, "
            f"missing={row['dataset_missing_concept_count']}, "
            f"extra={row['concept_without_dataset_count']}, "
            f"order_mismatch={row['dataset_concept_order_mismatch_count']}, "
            f"key_mode={row['concept_key_mode']}"
        )
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
