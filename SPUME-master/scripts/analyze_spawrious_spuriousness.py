import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


BACKGROUND_KEYWORDS = [
    "snow",
    "beach",
    "desert",
    "jungle",
    "mountain",
    "forest",
    "sand",
    "sandy",
    "tree",
    "trees",
    "ocean",
    "field",
    "dirt",
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_concept_paths(data_root):
    data_root = Path(data_root)
    vocab_path = data_root / "vit-gpt2_extracted_concepts_0_0_vocab.pickle"
    embeddings_path = data_root / "vit-gpt2_extracted_concepts_0_0_img_embeddings.pickle"
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab file not found: {vocab_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"embedding file not found: {embeddings_path}")
    return vocab_path, embeddings_path


def normalize_vocab_entry(entry):
    text = str(entry)
    if ":" in text:
        text, pos = text.rsplit(":", 1)
        return text.strip(), pos.strip()
    return text.strip(), ""


def make_global_frequency_table(vocab, embeddings):
    counts = embeddings.sum(axis=0).astype(int)
    rows = []
    for idx, (concept, count) in enumerate(zip(vocab, counts)):
        concept_text, pos = normalize_vocab_entry(concept)
        rows.append(
            {
                "concept_index": idx,
                "concept": str(concept),
                "concept_text": concept_text,
                "pos": pos,
                "count": int(count),
            }
        )
    rows.sort(key=lambda row: (-row["count"], row["concept"]))
    return rows


def make_class_tables(metadata_df, vocab, embeddings):
    rows = []
    y_array = metadata_df["y"].astype(int).to_numpy()
    class_names = (
        metadata_df[["y", "class_name"]]
        .drop_duplicates()
        .sort_values("y")
        .to_records(index=False)
    )

    for class_id, class_name in class_names:
        class_mask = y_array == class_id
        class_embeddings = embeddings[class_mask]
        total_class_samples = int(class_mask.sum())
        if total_class_samples == 0:
            continue
        concept_counts = class_embeddings.sum(axis=0).astype(int)
        concept_rates = concept_counts / max(total_class_samples, 1)

        for idx, (count, rate) in enumerate(zip(concept_counts, concept_rates)):
            if count <= 0:
                continue
            concept_text, pos = normalize_vocab_entry(vocab[idx])
            rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                    "concept_index": idx,
                    "concept": str(vocab[idx]),
                    "concept_text": concept_text,
                    "pos": pos,
                    "count_in_class": int(count),
                    "rate_in_class": float(rate),
                    "class_support": total_class_samples,
                }
            )
    rows.sort(key=lambda row: (row["class_id"], -row["rate_in_class"], -row["count_in_class"], row["concept"]))
    return rows


def make_spurious_tables(metadata_df, vocab, embeddings):
    rows = []
    y_array = metadata_df["y"].astype(int).to_numpy()
    env_array = metadata_df["env"].astype(str).to_numpy()
    class_names = (
        metadata_df[["y", "class_name"]]
        .drop_duplicates()
        .sort_values("y")
        .to_records(index=False)
    )

    for class_id, class_name in class_names:
        class_mask = y_array == class_id
        class_embeddings = embeddings[class_mask]
        class_envs = env_array[class_mask]
        total_class_samples = int(class_mask.sum())
        if total_class_samples == 0:
            continue

        for idx in range(embeddings.shape[1]):
            active_mask = class_embeddings[:, idx] > 0
            active_count = int(active_mask.sum())
            if active_count == 0:
                continue

            active_envs = class_envs[active_mask]
            env_counter = Counter(active_envs.tolist())
            dominant_env, dominant_count = env_counter.most_common(1)[0]
            dominance = dominant_count / active_count
            support_rate = active_count / max(total_class_samples, 1)
            score = dominance * support_rate
            concept_text, pos = normalize_vocab_entry(vocab[idx])

            rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                    "concept_index": idx,
                    "concept": str(vocab[idx]),
                    "concept_text": concept_text,
                    "pos": pos,
                    "active_count": active_count,
                    "support_rate": float(support_rate),
                    "dominant_env": dominant_env,
                    "dominant_env_count": int(dominant_count),
                    "dominance": float(dominance),
                    "spurious_score": float(score),
                }
            )
    rows.sort(
        key=lambda row: (
            row["class_id"],
            -row["spurious_score"],
            -row["active_count"],
            row["concept"],
        )
    )
    return rows


def filter_top_rows(rows, key_name, top_k):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[key_name], [])
        if len(grouped[row[key_name]]) < top_k:
            grouped[row[key_name]].append(row)
    return grouped


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(global_rows, class_rows, spurious_rows, top_k):
    top_global = global_rows[:top_k]
    top_class = filter_top_rows(class_rows, "class_name", top_k)
    top_spurious = filter_top_rows(spurious_rows, "class_name", top_k)
    keyword_hits = [
        row
        for row in global_rows
        if row["concept_text"] in BACKGROUND_KEYWORDS
    ]
    return {
        "top_global_concepts": top_global,
        "top_class_concepts": top_class,
        "top_spurious_concepts": top_spurious,
        "background_keyword_hits": keyword_hits,
    }


def print_summary(summary, top_k):
    print("== Global high-frequency concepts ==")
    for row in summary["top_global_concepts"][:top_k]:
        print(f"  {row['concept']:<24} count={row['count']}")
    print()

    print("== Background keyword hits ==")
    if not summary["background_keyword_hits"]:
        print("  no tracked background keywords found")
    else:
        for row in summary["background_keyword_hits"][:top_k]:
            print(f"  {row['concept']:<24} count={row['count']}")
    print()

    print("== Top concepts per class ==")
    for class_name, rows in summary["top_class_concepts"].items():
        print(f"[{class_name}]")
        for row in rows:
            print(
                f"  {row['concept']:<24} rate={row['rate_in_class']:.3f} count={row['count_in_class']}"
            )
        print()

    print("== Most likely spurious concepts per class ==")
    for class_name, rows in summary["top_spurious_concepts"].items():
        print(f"[{class_name}]")
        for row in rows:
            print(
                f"  {row['concept']:<24} score={row['spurious_score']:.3f} "
                f"dominant_env={row['dominant_env']} dominance={row['dominance']:.3f} active={row['active_count']}"
            )
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze concept frequencies and likely spurious concepts for Spawrious."
    )
    parser.add_argument(
        "--data-root",
        default=r"D:\SPUME\spawrious224__o2o_easy",
        help="directory containing extracted concept files",
    )
    parser.add_argument(
        "--metadata-path",
        default=r"D:\SPUME\SPUME-master\data\spawrious_o2o_easy_metadata.csv",
        help="metadata csv path",
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\SPUME\how_to_run\meta_spurious_exprs\spawrious_analysis",
        help="directory to save analysis outputs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="number of top concepts to keep per table",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path, embeddings_path = infer_concept_paths(data_root)
    vocab = load_pickle(vocab_path)
    embeddings = load_pickle(embeddings_path)
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df[metadata_df["split"].astype(str).str.lower() != "test"].reset_index(drop=True)

    if len(metadata_df) != embeddings.shape[0]:
        raise ValueError(
            f"metadata rows ({len(metadata_df)}) and embedding rows ({embeddings.shape[0]}) do not match"
        )

    global_rows = make_global_frequency_table(vocab, embeddings)
    class_rows = make_class_tables(metadata_df, vocab, embeddings)
    spurious_rows = make_spurious_tables(metadata_df, vocab, embeddings)
    summary = build_summary(global_rows, class_rows, spurious_rows, args.top_k)

    write_csv(
        output_dir / "global_top_concepts.csv",
        global_rows[: args.top_k],
        ["concept_index", "concept", "concept_text", "pos", "count"],
    )
    write_csv(
        output_dir / "class_top_concepts.csv",
        class_rows,
        [
            "class_id",
            "class_name",
            "concept_index",
            "concept",
            "concept_text",
            "pos",
            "count_in_class",
            "rate_in_class",
            "class_support",
        ],
    )
    write_csv(
        output_dir / "class_spurious_concepts.csv",
        spurious_rows,
        [
            "class_id",
            "class_name",
            "concept_index",
            "concept",
            "concept_text",
            "pos",
            "active_count",
            "support_rate",
            "dominant_env",
            "dominant_env_count",
            "dominance",
            "spurious_score",
        ],
    )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary, args.top_k)
    print(f"saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
