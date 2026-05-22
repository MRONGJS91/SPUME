#!/usr/bin/env python3
"""Analyse class-background distributions in Spawrious train/test splits."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

DATASETS = {
    "spawrious_o2o_easy": ROOT / "SPUME-master" / "data" / "spawrious_o2o_easy_metadata.csv",
    "spawrious_o2o_medium": ROOT / "SPUME-master" / "data" / "spawrious_o2o_medium_metadata.csv",
}


def parse_background(env_str: str) -> str:
    """Extract background name from env string like 'env_0_beach'."""
    parts = env_str.split("_", 2)  # ['env', '0', 'beach']
    if len(parts) >= 3:
        return parts[2]
    return env_str


def compute_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute class x background contingency table (counts)."""
    df = df.copy()
    df["background"] = df["env"].apply(parse_background)
    table = df.groupby(["class_name", "background"]).size().unstack(fill_value=0)
    return table


def compute_proportion(df: pd.DataFrame) -> pd.DataFrame:
    """Compute row-normalised proportions."""
    row_sums = df.sum(axis=1)
    return df.div(row_sums, axis=0)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two discrete distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    # Handle edge cases: 0 * log(0/m) = 0, and 0 * log(0/0) = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log(p / m), 0.0)
        kl_qm = np.where(q > 0, q * np.log(q / m), 0.0)
    return float(0.5 * (np.nansum(kl_pm) + np.nansum(kl_qm)))


def analyse_dataset(name: str, metadata_path: Path) -> dict:
    """Full analysis for one dataset."""
    df = pd.read_csv(metadata_path)
    df["background"] = df["env"].apply(parse_background)

    classes = sorted(df["class_name"].unique())
    backgrounds = sorted(df["background"].unique())

    # --- train / test distributions ---
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_counts = compute_distribution(train_df)
    test_counts = compute_distribution(test_df)
    train_prop = compute_proportion(train_counts)
    test_prop = compute_proportion(test_counts)

    # --- overall counts ---
    train_total = len(train_df)
    test_total = len(test_df)

    # --- per-class JS divergence (train vs test) ---
    js_scores = {}
    for cls in classes:
        p = np.array([train_counts.loc[cls, bg] if bg in train_counts.columns else 0 for bg in backgrounds])
        q = np.array([test_counts.loc[cls, bg] if bg in test_counts.columns else 0 for bg in backgrounds])
        js_scores[cls] = js_divergence(p, q)

    # --- marginal background distribution shift ---
    train_bg_marginal = train_df["background"].value_counts(normalize=True)
    test_bg_marginal = test_df["background"].value_counts(normalize=True)
    p_marg = np.array([train_bg_marginal.get(bg, 0) for bg in backgrounds])
    q_marg = np.array([test_bg_marginal.get(bg, 0) for bg in backgrounds])
    marginal_js = js_divergence(p_marg, q_marg)

    # --- summary stats ---
    summary = {
        "dataset": name,
        "train_samples": train_total,
        "test_samples": test_total,
        "classes": classes,
        "backgrounds": backgrounds,
        "marginal_background_js": round(marginal_js, 6),
        "per_class_js": {cls: round(v, 6) for cls, v in js_scores.items()},
    }

    return {
        "summary": summary,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "train_prop": train_prop,
        "test_prop": test_prop,
        "train_df": train_df,
        "test_df": test_df,
    }


def format_table(counts: pd.DataFrame, prop: pd.DataFrame) -> pd.DataFrame:
    """Merge counts and proportions into a single readable table."""
    result = counts.copy().astype(str)
    for cls in counts.index:
        for bg in counts.columns:
            c = counts.loc[cls, bg]
            p = prop.loc[cls, bg] if bg in prop.columns else 0.0
            result.loc[cls, bg] = f"{c} ({p:.1%})"
    return result


def main():
    all_summaries = []

    for name, path in DATASETS.items():
        if not path.exists():
            print(f"[SKIP] {path} not found")
            continue

        print(f"\n{'=' * 80}")
        print(f"Analysing: {name}")
        print(f"{'=' * 80}")

        result = analyse_dataset(name, path)

        s = result["summary"]
        all_summaries.append(s)

        print(f"\nTrain samples: {s['train_samples']:,}")
        print(f"Test samples:  {s['test_samples']:,}")
        print(f"Classes:      {s['classes']}")
        print(f"Backgrounds:  {s['backgrounds']}")
        print(f"Marginal background JS divergence (train vs test): {s['marginal_background_js']:.4f}")

        print(f"\n--- Train split: class × background counts (proportion) ---")
        train_table = format_table(result["train_counts"], result["train_prop"])
        print(train_table.to_string())

        print(f"\n--- Test split: class × background counts (proportion) ---")
        test_table = format_table(result["test_counts"], result["test_prop"])
        print(test_table.to_string())

        print(f"\n--- Per-class JS divergence (train vs test) ---")
        for cls, js in s["per_class_js"].items():
            print(f"  {cls}: {js:.4f}")

        # --- Save CSVs ---
        out_dir = SCRIPTS / f"{name}_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Raw counts
        result["train_counts"].to_csv(out_dir / "train_class_background_counts.csv")
        result["test_counts"].to_csv(out_dir / "test_class_background_counts.csv")
        # Proportions
        result["train_prop"].to_csv(out_dir / "train_class_background_proportions.csv")
        result["test_prop"].to_csv(out_dir / "test_class_background_proportions.csv")
        # Combined train+test view
        combined = pd.concat(
            {"train_count": result["train_counts"], "test_count": result["test_counts"]},
            axis=0,
        )
        combined.to_csv(out_dir / "combined_class_background_counts.csv")

        train_flat = result["train_counts"].reset_index().melt(
            id_vars="class_name", var_name="background", value_name="train_count"
        )
        test_flat = result["test_counts"].reset_index().melt(
            id_vars="class_name", var_name="background", value_name="test_count"
        )
        merged = train_flat.merge(test_flat, on=["class_name", "background"], how="outer").fillna(0)
        merged["train_count"] = merged["train_count"].astype(int)
        merged["test_count"] = merged["test_count"].astype(int)
        for col in ["train_count", "test_count"]:
            merged[col] = merged[col].astype(int)

        # compute proportions
        train_totals = merged.groupby("class_name")["train_count"].transform("sum")
        test_totals = merged.groupby("class_name")["test_count"].transform("sum")
        merged["train_prop"] = (merged["train_count"] / train_totals.replace(0, 1)).round(4)
        merged["test_prop"] = (merged["test_count"] / test_totals.replace(0, 1)).round(4)
        merged["prop_diff"] = (merged["test_prop"] - merged["train_prop"]).round(4)
        merged.to_csv(out_dir / "class_background_comparison.csv", index=False)

        print(f"\nCSV files saved to: {out_dir}")

    # --- Cross-dataset summary ---
    summary_path = SCRIPTS / "spawrious_distribution_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSummary JSON saved to: {summary_path}")

    # --- Brief report ---
    report_lines = [
        "# Spawrious Dataset: Class-Background Distribution Analysis",
        "",
        "## 1. Overview",
        "",
    ]
    for s in all_summaries:
        report_lines.append(
            f"- **{s['dataset']}**: {s['train_samples']:,} train / {s['test_samples']:,} test, "
            f"classes={s['classes']}, backgrounds={s['backgrounds']}"
        )
        report_lines.append(f"  - Marginal background JS divergence: {s['marginal_background_js']:.4f}")
        report_lines.append(
            f"  - Per-class JS: " + ", ".join(f"{c}={v:.4f}" for c, v in s["per_class_js"].items())
        )
        report_lines.append("")

    report_lines.extend([
        "## 2. Interpretation",
        "",
        "- **JS divergence** measures how different the train and test distributions are.",
        "  - 0 = identical distributions; larger values = more shift.",
        "- **Marginal background JS** tells whether the overall background mix changes between train and test.",
        "- **Per-class JS** tells whether a specific class sees its background distribution shift.",
        "",
        "## 3. Key Findings",
        "",
    ])

    for s in all_summaries:
        name = s["dataset"]
        js_vals = s["per_class_js"]
        max_cls = max(js_vals, key=js_vals.get)
        min_cls = min(js_vals, key=js_vals.get)

        report_lines.append(f"### {name}")
        report_lines.append(f"- Marginal background shift (JS): **{s['marginal_background_js']:.4f}**")
        report_lines.append(
            f"- Largest per-class shift: **{max_cls}** (JS={js_vals[max_cls]:.4f})"
        )
        report_lines.append(
            f"- Smallest per-class shift: **{min_cls}** (JS={js_vals[min_cls]:.4f})"
        )

        # check if shift is "significant"
        if s["marginal_background_js"] < 0.01 and all(v < 0.02 for v in js_vals.values()):
            report_lines.append(
                "- **Conclusion**: Train and test class-background distributions are nearly identical. "
                "The correlation structure does NOT change between splits."
            )
        else:
            report_lines.append(
                "- **Conclusion**: There IS a noticeable distribution shift between train and test. "
                "Spurious correlations may differ across splits."
            )
        report_lines.append("")

    report_path = SCRIPTS / "spawrious_distribution_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
