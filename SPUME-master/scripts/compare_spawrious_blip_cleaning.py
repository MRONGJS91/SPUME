import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_concept(concept):
    text = str(concept)
    if ":" in text:
        concept_text, pos = text.rsplit(":", 1)
        return concept_text.strip().lower(), pos.strip().lower()
    return text.strip().lower(), ""


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_global_top(vocab, embeddings, top_k):
    counts = embeddings.sum(axis=0).astype(int)
    rows = []
    for idx, c in enumerate(vocab):
        concept_text, pos = parse_concept(c)
        rows.append(
            {
                "concept_index": int(idx),
                "concept": str(c),
                "concept_text": concept_text,
                "pos": pos,
                "count": int(counts[idx]),
            }
        )
    rows.sort(key=lambda r: (-r["count"], r["concept"]))
    for i, row in enumerate(rows[:top_k], start=1):
        row["rank"] = i
    return rows[:top_k]


def build_class_top(metadata_df, vocab, embeddings, top_k):
    y = metadata_df["y"].astype(int).to_numpy()
    class_rows = (
        metadata_df[["y", "class_name"]]
        .drop_duplicates()
        .sort_values("y")
        .reset_index(drop=True)
    )
    rows = []
    for _, row in class_rows.iterrows():
        class_id = int(row["y"])
        class_name = str(row["class_name"])
        mask = y == class_id
        support = int(mask.sum())
        class_counts = embeddings[mask].sum(axis=0).astype(int)
        order = np.argsort(-class_counts)
        rank = 1
        for idx in order:
            cnt = int(class_counts[idx])
            if cnt <= 0:
                continue
            concept = str(vocab[idx])
            concept_text, pos = parse_concept(concept)
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "rank": rank,
                    "concept": concept,
                    "concept_text": concept_text,
                    "pos": pos,
                    "count_in_class": cnt,
                    "rate_in_class": float(cnt / max(support, 1)),
                    "class_support": support,
                }
            )
            rank += 1
            if rank > top_k:
                break
    return rows


def to_class_map(rows):
    result = {}
    for row in rows:
        result.setdefault(row["class_name"], [])
        result[row["class_name"]].append(row)
    return result


def compare_class_tops(before_rows, after_rows):
    before_map = to_class_map(before_rows)
    after_map = to_class_map(after_rows)
    class_names = sorted(set(before_map) | set(after_map))

    changes = []
    rank_changes = []
    for class_name in class_names:
        b_rows = before_map.get(class_name, [])
        a_rows = after_map.get(class_name, [])
        b_rank = {r["concept_text"]: r["rank"] for r in b_rows}
        a_rank = {r["concept_text"]: r["rank"] for r in a_rows}
        b_set = set(b_rank)
        a_set = set(a_rank)
        added = sorted(a_set - b_set)
        removed = sorted(b_set - a_set)
        kept = sorted(a_set & b_set, key=lambda x: a_rank[x])

        changes.append(
            {
                "class_name": class_name,
                "added_count": len(added),
                "removed_count": len(removed),
                "added_concepts": "|".join(added),
                "removed_concepts": "|".join(removed),
            }
        )
        for concept in kept:
            rank_changes.append(
                {
                    "class_name": class_name,
                    "concept_text": concept,
                    "before_rank": int(b_rank[concept]),
                    "after_rank": int(a_rank[concept]),
                    "rank_shift": int(b_rank[concept] - a_rank[concept]),
                }
            )
    rank_changes.sort(
        key=lambda r: (r["class_name"], -abs(r["rank_shift"]), r["after_rank"], r["concept_text"])
    )
    return changes, rank_changes


def build_filter_summary(before_global_rows, ignore_terms, optional_terms):
    before_count = {r["concept_text"]: r["count"] for r in before_global_rows}
    rows = []
    for t in ignore_terms:
        rows.append(
            {
                "term": t,
                "term_type": "base_ignore",
                "count_before_clean": int(before_count.get(t, 0)),
            }
        )
    for t in optional_terms:
        rows.append(
            {
                "term": t,
                "term_type": "optional",
                "count_before_clean": int(before_count.get(t, 0)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare BLIP concept cleaning before/after.")
    parser.add_argument(
        "--metadata-path",
        default=r"D:\SPUME\SPUME-master\data\spawrious_o2o_easy_metadata.csv",
    )
    parser.add_argument(
        "--before-vocab",
        default=r"D:\SPUME\spawrious224__o2o_easy\blip_extracted_concepts_0_0_vocab_before_final_clean.pickle",
    )
    parser.add_argument(
        "--before-embed",
        default=r"D:\SPUME\spawrious224__o2o_easy\blip_extracted_concepts_0_0_img_embeddings_before_final_clean.pickle",
    )
    parser.add_argument(
        "--after-vocab",
        default=r"D:\SPUME\spawrious224__o2o_easy\blip_extracted_concepts_0_0_vocab.pickle",
    )
    parser.add_argument(
        "--after-embed",
        default=r"D:\SPUME\spawrious224__o2o_easy\blip_extracted_concepts_0_0_img_embeddings.pickle",
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\SPUME\how_to_run\meta_spurious_exprs\spawrious_blip_finalclean_compare",
    )
    parser.add_argument("--top-global", type=int, default=50)
    parser.add_argument("--top-class", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(args.metadata_path)
    metadata_df = metadata_df[
        metadata_df["split"].astype(str).str.lower() != "test"
    ].reset_index(drop=True)

    before_vocab = np.array(load_pickle(args.before_vocab))
    before_embed = np.array(load_pickle(args.before_embed))
    after_vocab = np.array(load_pickle(args.after_vocab))
    after_embed = np.array(load_pickle(args.after_embed))

    if len(metadata_df) != before_embed.shape[0] or len(metadata_df) != after_embed.shape[0]:
        raise ValueError("metadata rows do not match before/after embedding rows.")

    before_global = build_global_top(before_vocab, before_embed, args.top_global)
    after_global = build_global_top(after_vocab, after_embed, args.top_global)
    before_class = build_class_top(metadata_df, before_vocab, before_embed, args.top_class)
    after_class = build_class_top(metadata_df, after_vocab, after_embed, args.top_class)
    class_changes, rank_changes = compare_class_tops(before_class, after_class)

    ignore_terms = ["dog", "frisbee", "front", "camera"]
    optional_terms = ["black", "white", "brown", "small"]
    filter_summary = build_filter_summary(before_global, ignore_terms, optional_terms)

    write_csv(
        output_dir / "blip_global_top50_before.csv",
        before_global,
        ["rank", "concept_index", "concept", "concept_text", "pos", "count"],
    )
    write_csv(
        output_dir / "blip_global_top50_after.csv",
        after_global,
        ["rank", "concept_index", "concept", "concept_text", "pos", "count"],
    )
    write_csv(
        output_dir / "blip_class_top30_before.csv",
        before_class,
        [
            "class_id",
            "class_name",
            "rank",
            "concept",
            "concept_text",
            "pos",
            "count_in_class",
            "rate_in_class",
            "class_support",
        ],
    )
    write_csv(
        output_dir / "blip_class_top30_after.csv",
        after_class,
        [
            "class_id",
            "class_name",
            "rank",
            "concept",
            "concept_text",
            "pos",
            "count_in_class",
            "rate_in_class",
            "class_support",
        ],
    )
    write_csv(
        output_dir / "blip_class_top30_changes.csv",
        class_changes,
        ["class_name", "added_count", "removed_count", "added_concepts", "removed_concepts"],
    )
    write_csv(
        output_dir / "blip_class_rank_shifts.csv",
        rank_changes,
        ["class_name", "concept_text", "before_rank", "after_rank", "rank_shift"],
    )
    write_csv(
        output_dir / "blip_filter_summary.csv",
        filter_summary,
        ["term", "term_type", "count_before_clean"],
    )

    with open(output_dir / "blip_cleaning_compare_summary.txt", "w", encoding="utf-8") as f:
        f.write("BLIP Final Cleaning Compare (Spawrious o2o_easy)\n")
        f.write(f"Top global: {args.top_global}, top class: {args.top_class}\n\n")

        f.write("Filtered terms:\n")
        for row in filter_summary:
            f.write(
                f"  {row['term']:<10} type={row['term_type']:<12} "
                f"count_before={row['count_before_clean']}\n"
            )
        f.write("\nClass top-30 changes:\n")
        for row in class_changes:
            f.write(
                f"[{row['class_name']}] added={row['added_count']} removed={row['removed_count']}\n"
            )
            if row["added_concepts"]:
                f.write(f"  + {row['added_concepts']}\n")
            if row["removed_concepts"]:
                f.write(f"  - {row['removed_concepts']}\n")
        f.write("\nLargest rank shifts (top 40):\n")
        for row in rank_changes[:40]:
            f.write(
                f"  {row['class_name']:<10} {row['concept_text']:<16} "
                f"{row['before_rank']} -> {row['after_rank']} (shift={row['rank_shift']})\n"
            )

    print(f"Saved reports to {output_dir}")


if __name__ == "__main__":
    main()
