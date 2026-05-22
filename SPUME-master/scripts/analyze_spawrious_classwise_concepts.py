import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


DEFAULT_KEYWORDS = [
    "dog",
    "beach",
    "snow",
    "jungle",
    "desert",
    "dirt",
    "water",
    "cactus",
    "frisbee",
    "black",
    "white",
    "brown",
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_concept(concept):
    text = str(concept)
    if ":" in text:
        text_part, pos = text.rsplit(":", 1)
        return text_part.strip().lower(), pos.strip().lower()
    return text.strip().lower(), ""


def infer_concept_paths(data_root, concept_model):
    prefix = concept_model.strip().lower().replace("_", "-")
    vocab_path = Path(data_root) / f"{prefix}_extracted_concepts_0_0_vocab.pickle"
    embed_path = Path(data_root) / f"{prefix}_extracted_concepts_0_0_img_embeddings.pickle"
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab file not found: {vocab_path}")
    if not embed_path.exists():
        raise FileNotFoundError(f"embedding file not found: {embed_path}")
    return vocab_path, embed_path


def build_class_counts(metadata_df, embeddings):
    class_rows = (
        metadata_df[["y", "class_name"]]
        .drop_duplicates()
        .sort_values("y")
        .reset_index(drop=True)
    )
    class_ids = class_rows["y"].astype(int).to_numpy()
    class_names = class_rows["class_name"].astype(str).to_numpy()
    y_array = metadata_df["y"].astype(int).to_numpy()

    class_count = {}
    class_rate = {}
    class_support = {}
    for class_id, class_name in zip(class_ids, class_names):
        mask = y_array == class_id
        support = int(mask.sum())
        counts = embeddings[mask].sum(axis=0).astype(int)
        rates = counts / max(support, 1)
        class_count[class_name] = counts
        class_rate[class_name] = rates
        class_support[class_name] = support
    return class_names.tolist(), class_count, class_rate, class_support


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Class-wise concept analysis for Spawrious concept embeddings."
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
        "--concept-model",
        default="blip",
        choices=["blip", "vit-gpt2", "vit_gpt2"],
        help="concept embedding source model",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="top-k concepts per class",
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\SPUME\how_to_run\meta_spurious_exprs\spawrious_classwise_concepts",
        help="directory to save reports",
    )
    parser.add_argument(
        "--keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help="comma-separated keyword list",
    )
    args = parser.parse_args()

    concept_model = args.concept_model.replace("_", "-").lower()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path, embed_path = infer_concept_paths(args.data_root, concept_model)
    vocab = np.array(load_pickle(vocab_path))
    embeddings = np.array(load_pickle(embed_path))

    metadata_df = pd.read_csv(args.metadata_path)
    metadata_df = metadata_df[
        metadata_df["split"].astype(str).str.lower() != "test"
    ].reset_index(drop=True)

    if len(metadata_df) != embeddings.shape[0]:
        raise ValueError(
            f"metadata rows ({len(metadata_df)}) and embedding rows ({embeddings.shape[0]}) do not match"
        )

    concept_text = np.array([parse_concept(c)[0] for c in vocab])
    concept_pos = np.array([parse_concept(c)[1] for c in vocab])
    env_array = metadata_df["env"].astype(str).to_numpy()

    class_names, class_count, class_rate, class_support = build_class_counts(
        metadata_df, embeddings
    )
    class_num = len(class_names)

    # 1) Top-K concepts per class (cleaned)
    top_rows = []
    topk_text_sets = {}
    for class_name in class_names:
        counts = class_count[class_name]
        rates = class_rate[class_name]
        order = np.argsort(-counts)
        sel_idx = [idx for idx in order if counts[idx] > 0][: args.top_k]
        text_set = set()
        for rank, idx in enumerate(sel_idx, start=1):
            text_set.add(concept_text[idx])
            top_rows.append(
                {
                    "class_name": class_name,
                    "rank": rank,
                    "concept": str(vocab[idx]),
                    "concept_text": concept_text[idx],
                    "pos": concept_pos[idx],
                    "count_in_class": int(counts[idx]),
                    "rate_in_class": float(rates[idx]),
                    "class_support": int(class_support[class_name]),
                }
            )
        topk_text_sets[class_name] = text_set

    # 2) Keyword frequencies per class
    keyword_list = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    text_to_indices = defaultdict(list)
    for idx, text in enumerate(concept_text):
        text_to_indices[text].append(idx)

    keyword_rows = []
    for class_name in class_names:
        counts = class_count[class_name]
        support = class_support[class_name]
        for kw in keyword_list:
            idxes = text_to_indices.get(kw, [])
            kw_count = int(counts[idxes].sum()) if idxes else 0
            keyword_rows.append(
                {
                    "class_name": class_name,
                    "keyword": kw,
                    "count_in_class": kw_count,
                    "rate_in_class": float(kw_count / max(support, 1)),
                    "class_support": int(support),
                }
            )

    # 3) Concept class coverage
    coverage_rows = []
    for idx in range(len(vocab)):
        per_class_counts = [int(class_count[c][idx]) for c in class_names]
        coverage = sum(1 for v in per_class_counts if v > 0)
        coverage_rows.append(
            {
                "concept": str(vocab[idx]),
                "concept_text": concept_text[idx],
                "pos": concept_pos[idx],
                "classes_present": int(coverage),
                "present_ratio": float(coverage / max(class_num, 1)),
                "total_count": int(embeddings[:, idx].sum()),
            }
        )
    coverage_rows.sort(
        key=lambda r: (-r["classes_present"], -r["total_count"], r["concept"])
    )

    # 4) Generic concepts: appears in every class top-K
    generic_texts = set.intersection(*[topk_text_sets[c] for c in class_names])
    generic_rows = []
    for text in sorted(generic_texts):
        idxes = text_to_indices.get(text, [])
        per_class_rate = []
        per_class_count = []
        for c in class_names:
            cls_count = int(class_count[c][idxes].sum()) if idxes else 0
            cls_rate = float(cls_count / max(class_support[c], 1))
            per_class_count.append(cls_count)
            per_class_rate.append(cls_rate)
        generic_rows.append(
            {
                "concept_text": text,
                "classes_present": class_num,
                "mean_rate": float(np.mean(per_class_rate)),
                "min_rate": float(np.min(per_class_rate)),
                "max_rate": float(np.max(per_class_rate)),
                "per_class_count": "|".join(
                    [f"{c}:{v}" for c, v in zip(class_names, per_class_count)]
                ),
            }
        )
    generic_rows.sort(key=lambda r: (-r["mean_rate"], -r["min_rate"], r["concept_text"]))

    # 5) Class-specific / env-specific analysis at concept_text level
    specificity_rows = []
    for text, idxes in text_to_indices.items():
        per_class_counts = []
        per_class_rates = []
        for c in class_names:
            cnt = int(class_count[c][idxes].sum())
            rate = float(cnt / max(class_support[c], 1))
            per_class_counts.append(cnt)
            per_class_rates.append(rate)

        classes_present = sum(1 for v in per_class_counts if v > 0)
        top_class_i = int(np.argmax(per_class_rates))
        top_rate = float(per_class_rates[top_class_i])
        sorted_rates = sorted(per_class_rates, reverse=True)
        second_rate = float(sorted_rates[1] if len(sorted_rates) > 1 else 0.0)
        class_specific_score = top_rate - second_rate

        active_mask = np.zeros(embeddings.shape[0], dtype=bool)
        for idx in idxes:
            active_mask |= embeddings[:, idx] > 0
        active_total = int(active_mask.sum())
        env_counter = Counter(env_array[active_mask].tolist()) if active_total > 0 else Counter()
        if env_counter:
            dominant_env, dominant_cnt = env_counter.most_common(1)[0]
            env_dominance = float(dominant_cnt / active_total)
        else:
            dominant_env, env_dominance = "", 0.0

        is_class_specific = classes_present <= 2 or class_specific_score >= 0.10
        is_env_specific = active_total >= 50 and env_dominance >= 0.75
        if is_class_specific and is_env_specific:
            tag = "both"
        elif is_class_specific:
            tag = "class-specific"
        elif is_env_specific:
            tag = "env-specific"
        elif classes_present == class_num:
            tag = "generic"
        else:
            tag = "mixed"

        specificity_rows.append(
            {
                "concept_text": text,
                "classes_present": int(classes_present),
                "top_class_name": class_names[top_class_i],
                "top_rate": top_rate,
                "second_rate": second_rate,
                "class_specific_score": class_specific_score,
                "active_total": active_total,
                "dominant_env": dominant_env,
                "env_dominance": env_dominance,
                "specificity_tag": tag,
            }
        )
    specificity_rows.sort(
        key=lambda r: (
            {"both": 0, "class-specific": 1, "env-specific": 2, "generic": 3, "mixed": 4}[r["specificity_tag"]],
            -r["class_specific_score"],
            -r["env_dominance"],
            -r["active_total"],
            r["concept_text"],
        )
    )

    prefix = f"{concept_model}_spawrious_o2o_easy"
    top_path = output_dir / f"{prefix}_class_top{args.top_k}_concepts.csv"
    kw_path = output_dir / f"{prefix}_class_keyword_frequency.csv"
    coverage_path = output_dir / f"{prefix}_concept_class_coverage.csv"
    generic_path = output_dir / f"{prefix}_generic_all_classes.csv"
    spec_path = output_dir / f"{prefix}_concept_specificity.csv"
    report_path = output_dir / f"{prefix}_classwise_report.txt"

    write_csv(
        top_path,
        top_rows,
        [
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
        kw_path,
        keyword_rows,
        ["class_name", "keyword", "count_in_class", "rate_in_class", "class_support"],
    )
    write_csv(
        coverage_path,
        coverage_rows,
        ["concept", "concept_text", "pos", "classes_present", "present_ratio", "total_count"],
    )
    write_csv(
        generic_path,
        generic_rows,
        ["concept_text", "classes_present", "mean_rate", "min_rate", "max_rate", "per_class_count"],
    )
    write_csv(
        spec_path,
        specificity_rows,
        [
            "concept_text",
            "classes_present",
            "top_class_name",
            "top_rate",
            "second_rate",
            "class_specific_score",
            "active_total",
            "dominant_env",
            "env_dominance",
            "specificity_tag",
        ],
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {concept_model}\n")
        f.write(f"Top-K per class: {args.top_k}\n")
        f.write(f"Classes: {', '.join(class_names)}\n")
        f.write(f"Keywords: {', '.join(keyword_list)}\n\n")

        f.write("=== Top concepts per class ===\n")
        for class_name in class_names:
            f.write(f"[{class_name}]\n")
            rows = [r for r in top_rows if r["class_name"] == class_name]
            for row in rows[: args.top_k]:
                f.write(
                    f"  #{row['rank']:>2} {row['concept']:<24} "
                    f"count={row['count_in_class']:<6} rate={row['rate_in_class']:.4f}\n"
                )
            f.write("\n")

        f.write("=== Keyword frequencies per class ===\n")
        for class_name in class_names:
            f.write(f"[{class_name}]\n")
            rows = [r for r in keyword_rows if r["class_name"] == class_name]
            for row in rows:
                f.write(
                    f"  {row['keyword']:<10} count={row['count_in_class']:<6} rate={row['rate_in_class']:.4f}\n"
                )
            f.write("\n")

        f.write("=== Generic concepts (in top-K of all classes) ===\n")
        if generic_rows:
            for row in generic_rows:
                f.write(
                    f"  {row['concept_text']:<16} mean_rate={row['mean_rate']:.4f} "
                    f"min_rate={row['min_rate']:.4f} max_rate={row['max_rate']:.4f}\n"
                )
        else:
            f.write("  (none)\n")
        f.write("\n")

        f.write("=== Class-specific / Env-specific candidates (top 60) ===\n")
        for row in specificity_rows[:60]:
            f.write(
                f"  {row['concept_text']:<16} tag={row['specificity_tag']:<14} "
                f"classes={row['classes_present']} top_class={row['top_class_name']:<10} "
                f"class_gap={row['class_specific_score']:.4f} env={row['dominant_env']:<16} "
                f"env_dom={row['env_dominance']:.4f} active={row['active_total']}\n"
            )

    print(f"Saved: {top_path}")
    print(f"Saved: {kw_path}")
    print(f"Saved: {coverage_path}")
    print(f"Saved: {generic_path}")
    print(f"Saved: {spec_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
