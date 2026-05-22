from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
CONCEPT_TOP30_CSV = (
    ROOT
    / "how_to_run"
    / "meta_spurious_exprs"
    / "spawrious_classwise_concepts_blip"
    / "blip_spawrious_o2o_easy_class_top30_concepts.csv"
)
CONCEPT_KEYWORD_CSV = (
    ROOT
    / "how_to_run"
    / "meta_spurious_exprs"
    / "spawrious_classwise_concepts_blip"
    / "blip_spawrious_o2o_easy_class_keyword_frequency.csv"
)

CLASSES = ["bulldog", "corgi", "dachshund", "labrador"]
CONCEPTS = ["dirt", "beach", "snow", "jungle", "water", "cactus", "road", "desert", "grass"]
METHODS = ["ERM", "Mixup", "SPUME-ViT-GPT2", "SPUME-BLIP"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#d9dee7",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "axes.axisbelow": True,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename, transparent=False)
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.2f}", pad: float = 2.0, fontsize: int = 8) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#1f2933",
        )


def load_concept_rates() -> dict[str, dict[str, float]]:
    rates = {cls: {concept: 0.0 for concept in CONCEPTS} for cls in CLASSES}

    for csv_path, concept_field in [(CONCEPT_TOP30_CSV, "concept_text"), (CONCEPT_KEYWORD_CSV, "keyword")]:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                class_name = row["class_name"]
                concept = row[concept_field]
                if class_name in rates and concept in rates[class_name]:
                    rates[class_name][concept] = float(row["rate_in_class"]) * 100.0

    return rates


def draw_waterbirds() -> None:
    methods = ["ERM", "SPUME-ViT-GPT2", "SPUME-BLIP"]
    worst = [66.4, 85.9, 85.7]
    gap = [23.8, 6.9, 6.1]
    colors = ["#6b7280", "#2563eb", "#0f766e"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={"wspace": 0.28})
    for ax, values, title, ylabel, ylim in [
        (axes[0], worst, "Worst-group Accuracy", "Accuracy (%)", (0, 100)),
        (axes[1], gap, "Accuracy Gap", "Gap (%)", (0, 30)),
    ]:
        x = np.arange(len(methods))
        bars = ax.bar(x, values, color=colors, width=0.58)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=18, ha="right")
        ax.grid(axis="x", visible=False)
        add_bar_labels(ax, bars, fmt="{:.1f}", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Waterbirds 鲁棒性指标对比", fontsize=14, y=1.03)
    save(fig, "waterbirds_spume_metrics.png")


def draw_concept_rates(rates: dict[str, dict[str, float]]) -> None:
    x = np.arange(len(CONCEPTS))
    width = 0.2
    palette = ["#2563eb", "#f59e0b", "#10b981", "#ef4444"]

    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    for i, class_name in enumerate(CLASSES):
        values = [rates[class_name][concept] for concept in CONCEPTS]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width=width, label=class_name, color=palette[i])
        add_bar_labels(ax, bars, fmt="{:.1f}", pad=1.5, fontsize=7)

    ax.set_title("Spawrious-o2o-easy 清洗后背景概念出现比例", fontsize=14, pad=12)
    ax.set_ylabel("出现比例 (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(CONCEPTS)
    ax.set_ylim(0, 42)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "spawrious_filtered_concept_rates.png")


def draw_concept_heatmap(rates: dict[str, dict[str, float]]) -> None:
    data = np.array([[rates[class_name][concept] for concept in CONCEPTS] for class_name in CLASSES])

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=max(40, data.max()))
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("出现比例 (%)", rotation=270, labelpad=14)

    ax.set_title("Spawrious 类别-背景概念出现比例热力图", fontsize=14, pad=12)
    ax.set_xticks(np.arange(len(CONCEPTS)))
    ax.set_xticklabels(CONCEPTS)
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.grid(False)

    threshold = data.max() * 0.48
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] >= threshold else "#111827"
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", color=color, fontsize=9)

    save(fig, "spawrious_concept_heatmap.png")


def draw_accuracy_comparison(filename: str, title: str, avg: list[float], worst: list[float]) -> None:
    x = np.arange(len(METHODS))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    bars_avg = ax.bar(x - width / 2, avg, width, label="Average Accuracy", color="#2563eb")
    bars_worst = ax.bar(x + width / 2, worst, width, label="Worst-group Accuracy", color="#f97316")
    add_bar_labels(ax, bars_avg, fontsize=8)
    add_bar_labels(ax, bars_worst, fontsize=8)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=12, ha="right")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, filename)


def draw_gap_comparison() -> None:
    datasets = ["Spawrious-o2o-easy", "Spawrious-o2o-medium"]
    values = {
        "ERM": [5.50, 4.18],
        "Mixup": [5.50, 4.59],
        "SPUME-ViT-GPT2": [14.38, 13.83],
        "SPUME-BLIP": [16.59, 13.87],
    }
    palette = ["#6b7280", "#10b981", "#2563eb", "#f97316"]
    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    for i, method in enumerate(METHODS):
        bars = ax.bar(x + (i - 1.5) * width, values[method], width, label=method, color=palette[i])
        add_bar_labels(ax, bars, fontsize=8)

    ax.set_title("Spawrious 不同方法 Accuracy Gap 对比", fontsize=14, pad=12)
    ax.set_ylabel("Accuracy Gap (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 18.5)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "spawrious_accuracy_gap_comparison.png")


def draw_results_heatmap() -> None:
    rows = [
        ("easy / ERM", 98.27, 92.77, 5.50),
        ("easy / Mixup", 98.27, 92.77, 5.50),
        ("easy / SPUME-ViT-GPT2", 90.80, 76.42, 14.38),
        ("easy / SPUME-BLIP", 89.86, 73.27, 16.59),
        ("medium / ERM", 98.83, 94.65, 4.18),
        ("medium / Mixup", 98.61, 94.03, 4.59),
        ("medium / SPUME-ViT-GPT2", 92.45, 78.62, 13.83),
        ("medium / SPUME-BLIP", 91.86, 77.99, 13.87),
    ]
    labels = [row[0] for row in rows]
    data = np.array([row[1:] for row in rows], dtype=float)
    metric_labels = ["Avg.", "Worst", "Gap"]

    normalized = np.zeros_like(data)
    for col in range(data.shape[1]):
        col_values = data[:, col]
        span = col_values.max() - col_values.min()
        if span == 0:
            normalized[:, col] = 0.5
        elif col == 2:
            normalized[:, col] = 1.0 - (col_values - col_values.min()) / span
        else:
            normalized[:, col] = (col_values - col_values.min()) / span

    cmap = LinearSegmentedColormap.from_list("performance", ["#fef3c7", "#7dd3fc", "#0f766e"])

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    im = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("相对表现", rotation=270, labelpad=14)

    ax.set_title("Spawrious 实验结果汇总", fontsize=14, pad=12)
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if normalized[i, j] > 0.62 else "#111827"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)

    save(fig, "spawrious_results_heatmap.png")


def main() -> None:
    setup_style()
    concept_rates = load_concept_rates()

    draw_waterbirds()
    draw_concept_rates(concept_rates)
    draw_concept_heatmap(concept_rates)
    draw_accuracy_comparison(
        "spawrious_o2o_easy_acc_comparison.png",
        "Spawrious-o2o-easy 准确率对比",
        avg=[98.27, 98.27, 90.80, 89.86],
        worst=[92.77, 92.77, 76.42, 73.27],
    )
    draw_accuracy_comparison(
        "spawrious_o2o_medium_acc_comparison.png",
        "Spawrious-o2o-medium 准确率对比",
        avg=[98.83, 98.61, 92.45, 91.86],
        worst=[94.65, 94.03, 78.62, 77.99],
    )
    draw_gap_comparison()
    draw_results_heatmap()


if __name__ == "__main__":
    main()
