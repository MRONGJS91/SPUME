"""Analyze concept quality on MetaShift: PMI, mutual information, spuriousness.

Identifies:
- Strongly class-related concepts (invariant / useful for classification)
- Strongly context-related concepts (spurious / shortcut)
- Top spurious concepts (high MI with context, low MI with class)
- Top invariant concepts (high MI with class, low MI with context)
"""

import json, csv, pickle
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis" / "metashift"

# ---------------------------------------------------------------------------
# Load + prepare
# ---------------------------------------------------------------------------
def load_cleaned_concepts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_cooccurrence(records, min_count=5):
    """Build concept×class and concept×context contingency tables."""
    concept_class = defaultdict(lambda: defaultdict(int))
    concept_ctx = defaultdict(lambda: defaultdict(int))
    concept_total = Counter()
    class_total = Counter()
    ctx_total = Counter()

    for r in records:
        cls = r["class_name"]
        ctx = r.get("context_name", r.get("env", ""))
        if not ctx:
            ctx = "unknown"
        concepts = set(r.get("nouns", []) + r.get("adjectives", []))
        concepts = {c.split(":")[0] if ":" in c else c for c in concepts}

        class_total[cls] += 1
        ctx_total[ctx] += 1
        for c in concepts:
            concept_total[c] += 1
            concept_class[c][cls] += 1
            concept_ctx[c][ctx] += 1

    # Filter rare concepts
    kept = {c for c, cnt in concept_total.items() if cnt >= min_count}
    return {
        "concept_class": {c: dict(concept_class[c]) for c in kept},
        "concept_ctx": {c: dict(concept_ctx[c]) for c in kept},
        "concept_total": {c: concept_total[c] for c in kept},
        "class_total": dict(class_total),
        "ctx_total": dict(ctx_total),
        "total_records": len(records),
        "classes": sorted(class_total.keys()),
        "contexts": sorted(ctx_total.keys()),
    }


# ---------------------------------------------------------------------------
# PMI & MI
# ---------------------------------------------------------------------------
def compute_pmi(data, target_key, target_total_key):
    """PMI(concept, target) = log( P(c,t) / (P(c)*P(t)) )"""
    N = data["total_records"]
    pmi = {}
    targets = list(data[target_total_key].keys())

    for c, c_total in data["concept_total"].items():
        p_c = c_total / N
        target_map = data[target_key].get(c, {})
        for t in targets:
            p_t = data[target_total_key].get(t, 1) / N
            joint = target_map.get(t, 0) / N
            if joint > 0:
                pmi_val = np.log(joint / (p_c * p_t + 1e-12))
            else:
                pmi_val = 0.0
            if target_key not in ("concept_ctx",) or pmi_val != 0:
                pass
            pmi[(c, t)] = pmi_val
    return pmi


def compute_mi(data, target_key, target_total_key):
    """MI(concept; target) = sum_t P(c,t) * PMI(c,t)"""
    N = data["total_records"]
    pmi = compute_pmi(data, target_key, target_total_key)
    mi = defaultdict(float)
    targets = list(data[target_total_key].keys())

    for c in data["concept_total"]:
        target_map = data[target_key].get(c, {})
        for t in targets:
            joint = target_map.get(t, 0) / N
            if joint > 0:
                mi[c] += joint * pmi.get((c, t), 0.0)
    return dict(mi)


# ---------------------------------------------------------------------------
# Spuriousness score
# ---------------------------------------------------------------------------
def compute_spurious_scores(data):
    """Spuriousness = MI(concept, context) / (MI(concept, class) + epsilon)

    High ratio → concept is more associated with context than class (spurious).
    Low ratio → concept is more associated with class (invariant/useful).
    """
    mi_class = compute_mi(data, "concept_class", "class_total")
    mi_ctx = compute_mi(data, "concept_ctx", "ctx_total")
    eps = 1e-8

    scores = []
    for c in data["concept_total"]:
        mic = mi_class.get(c, 0.0)
        mix = mi_ctx.get(c, 0.0)
        spurious_ratio = mix / (mic + eps)
        scores.append({
            "concept": c,
            "mi_class": round(mic, 6),
            "mi_context": round(mix, 6),
            "spurious_ratio": round(spurious_ratio, 4),
            "frequency": data["concept_total"][c],
        })
    return sorted(scores, key=lambda x: -x["spurious_ratio"])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_mi_scatter(scores, model_name):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei","SimHei","Arial"],
                         "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

    top_n = 40
    items = scores[:top_n] + scores[-top_n//2:]
    items = sorted(set((s["concept"], s["mi_class"], s["mi_context"], s["spurious_ratio"])
                       for s in scores), key=lambda x: x[1] + x[2])[-top_n:]

    x_vals = [s[1] for s in items]
    y_vals = [s[2] for s in items]
    labels = [s[0] for s in items]
    ratios = [s[3] for s in items]

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(x_vals, y_vals, c=ratios, cmap="RdYlBu_r", s=60, edgecolors="#333333", linewidth=0.3, alpha=0.85)

    # Label interesting points
    for i, label in enumerate(labels):
        if ratios[i] > 5 or x_vals[i] > 0.005:
            ax.annotate(label, (x_vals[i], y_vals[i]), fontsize=7, alpha=0.9,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("MI(concept, class)", fontsize=12)
    ax.set_ylabel("MI(concept, context)", fontsize=12)
    ax.set_title(f"MetaShift — Concept Spuriousness ({model_name})\n"
                 "Red=Spurious (context-associated)  Blue=Invariant (class-associated)",
                 fontsize=13, fontweight="bold")
    fig.colorbar(sc, ax=ax, label="Spurious Ratio (MI_ctx / MI_class)")
    ax.grid(True, alpha=0.2)
    ax.axline((0, 0), slope=1, color="gray", linestyle="--", alpha=0.3, label="y=x (equal MI)")

    # Shade regions
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.fill_between([0, max(xlim)], [0, 0], [0, 0], alpha=0.03, color="blue", label="Invariant zone")
    ax.fill_between([0, 0], [0, max(ylim)], [0, 0], alpha=0.03, color="red", label="Spurious zone")
    ax.legend(fontsize=7)
    plt.tight_layout()

    p = ANALYSIS / f"concept_spuriousness_{model_name}.png"
    fig.savefig(p); plt.close(fig)
    return p


def plot_top_spurious(scores, model_name, n=15):
    """Horizontal bar chart of top spurious concepts."""
    top = [s for s in scores if s["mi_class"] < 0.001][:n]
    if not top:
        top = scores[:n]

    labels = [s["concept"] for s in top][::-1]
    values = [s["spurious_ratio"] for s in top][::-1]
    freqs = [s["frequency"] for s in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#c62828" if v > 10 else "#ef5350" if v > 5 else "#ffcdd2" for v in values]
    ax.barh(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Spurious Ratio (MI_ctx / MI_class)"); ax.invert_yaxis()
    for i, (v, f) in enumerate(zip(values, freqs)):
        ax.text(v + max(values)*0.01, i, f"n={f}", va="center", fontsize=7, color="#555")
    ax.set_title(f"MetaShift — Top Spurious Concepts ({model_name})", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = ANALYSIS / f"top_spurious_{model_name}.png"
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(scores, model_name, scatter_path, bar_path):
    lines = [
        f"# MetaShift — Concept Quality Report: {model_name}",
        "",
        "## 1. Methodology",
        "",
        "- **PMI(c, t)** = log P(c,t) / (P(c)·P(t))",
        "- **MI(c; class)** = Σ P(c,cls) · PMI(c, cls)",
        "- **MI(c; context)** = Σ P(c,ctx) · PMI(c, ctx)",
        "- **Spurious Ratio** = MI(c; context) / (MI(c; class) + ε)",
        "",
        "High spurious ratio → concept strongly associated with specific contexts (likely spurious).",
        "Low spurious ratio → concept strongly associated with specific classes (invariant / diagnostic).",
        "",
        "## 2. Top Spurious Concepts (High context association)",
        "",
        "| Rank | Concept | MI Class | MI Context | Spurious Ratio | Frequency |",
        "|------|---------|----------|------------|----------------|-----------|",
    ]
    spurious = [s for s in scores if s["mi_context"] > 0.0001]
    for i, s in enumerate(spurious[:20], 1):
        lines.append(f"| {i} | {s['concept']} | {s['mi_class']:.5f} | {s['mi_context']:.5f} | "
                     f"{s['spurious_ratio']:.1f} | {s['frequency']} |")

    lines.extend([
        "",
        "## 3. Top Invariant Concepts (High class association)",
        "",
        "| Rank | Concept | MI Class | MI Context | Frequency |",
        "|------|---------|----------|------------|-----------|",
    ])
    invariant = sorted(scores, key=lambda x: -x["mi_class"])
    for i, s in enumerate(invariant[:20], 1):
        lines.append(f"| {i} | {s['concept']} | {s['mi_class']:.5f} | "
                     f"{s['mi_context']:.5f} | {s['frequency']} |")

    lines.extend([
        "",
        "## 4. Potential Spurious Concepts for SPUME",
        "",
        "These concepts are strongly associated with specific contexts but not with classes:",
        "",
    ])
    candidates = [s for s in scores if s["spurious_ratio"] > 3.0 and s["frequency"] >= 10]
    for s in candidates[:15]:
        lines.append(f"- **{s['concept']}** (ratio={s['spurious_ratio']:.1f}, freq={s['frequency']})")

    lines.extend([
        "",
        "## 5. Diagnostic Concepts (use for classification)",
        "",
    ])
    diagnostic = sorted(scores, key=lambda x: -(x["mi_class"] - x["mi_context"] * 0.5))
    for s in diagnostic[:10]:
        lines.append(f"- **{s['concept']}** (MI_class={s['mi_class']:.5f}, MI_ctx={s['mi_context']:.5f})")

    lines.extend([
        "",
        "## 6. Visualizations",
        "",
        f"![scatter]({Path(scatter_path).name})",
        "",
        f"![top spurious]({Path(bar_path).name})",
    ])

    p = ANALYSIS / f"concept_quality_{model_name}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    models = {
        "vitgpt2": ANALYSIS / "vitgpt2" / "concepts_cleaned_vitgpt2.json",
        "blip": ANALYSIS / "blip" / "concepts_cleaned_blip.json",
    }

    for name, path in models.items():
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing: {name}")
        print(f"{'='*60}")

        records = load_cleaned_concepts(path)
        print(f"  Records: {len(records)}")

        data = build_cooccurrence(records)
        print(f"  Concepts: {len(data['concept_total'])}")
        print(f"  Classes: {data['classes']}")
        print(f"  Contexts: {len(data['contexts'])}")

        scores = compute_spurious_scores(data)

        # Top spurious
        print(f"\n  Top-10 Most Spurious:")
        for s in scores[:10]:
            print(f"    {s['concept']:25s}  ratio={s['spurious_ratio']:8.1f}  "
                  f"MI_cls={s['mi_class']:.5f}  MI_ctx={s['mi_context']:.5f}  freq={s['frequency']}")

        # Top invariant
        print(f"\n  Top-10 Most Invariant (class-associated):")
        for s in sorted(scores, key=lambda x: -x["mi_class"])[:10]:
            print(f"    {s['concept']:25s}  MI_cls={s['mi_class']:.5f}  "
                  f"MI_ctx={s['mi_context']:.5f}  freq={s['frequency']}")

        # Plots
        scatter_path = plot_mi_scatter(scores, name)
        bar_path = plot_top_spurious(scores, name)
        print(f"\n  Scatter: {scatter_path}")
        print(f"  Bar: {bar_path}")

        # Report
        report = build_report(scores, name, scatter_path, bar_path)
        print(f"  Report: {report}")


if __name__ == "__main__":
    main()
