"""Debug SPUME meta-task construction on MetaShift.

Samples 50 tasks, records support/query concepts & distributions,
checks for concept leakage, empty queries, trivial tasks.
"""

import csv, json, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SPUME-master"))

from data.dataloader import get_loader
from data.metashift_data import MetaShiftDataset
from data.biased_dataset import get_transform_biased, IdxDataset
from torch.utils.data import DataLoader
from methods import get_correlated_features, REPModel
from data.sampler import AttributesTaskSampler
import torch, utils

ANALYSIS = ROOT / "analysis" / "metashift"
VOCAB_PATH = ROOT / "datasets/metashift/blip_extracted_concepts_0_0_vocab.pickle"
EMBED_PATH = ROOT / "datasets/metashift/blip_extracted_concepts_0_0_img_embeddings.pickle"
KEYS_PATH = ROOT / "datasets/metashift/blip_extracted_concepts_0_0_img_embedding_keys.pickle"
META_PATH = ROOT / "datasets/metashift/metadata_metashift_catdog.csv"

NUM_TASKS = 50
TOP_K = 10
NUM_SUPP = 10
NUM_QUERY = 10


def main():
    # --- Load vocab ---
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocab: {len(vocab)} concepts")

    # --- Build dataset + loader (use IdxDataset to match SPUME format: 6-tuple) ---
    tf = get_transform_biased(target_resolution=(224, 224), train=False, augment_data=False)
    ds = MetaShiftDataset(
        data_root=str(ROOT / "datasets/metashift"),
        metadata_path=str(META_PATH),
        split="train",
        transform=tf,
        concept_embed=str(EMBED_PATH),
        return_metadata=False,
    )
    ds_idx = IdxDataset(ds)
    loader = DataLoader(ds_idx, batch_size=64, shuffle=False, num_workers=0)

    # --- Load metadata for context lookup ---
    meta_rows = list(csv.DictReader(open(META_PATH)))
    def norm(p): return str(p).strip().replace("\\", "/")

    # Map dataset index → metadata row
    ds_meta = [None] * len(ds)
    for i in range(len(ds)):
        row = ds.metadata_df.iloc[i]
        ds_meta[i] = {
            "class_name": row["class_name"],
            "context_name": row.get("context_name", row["env"]),
            "filename": row["filename"],
        }

    # sampler uses dataset-level indices, so ds_meta[idx] gives metadata directly

    # --- Compute spuriousness ---
    print("Computing spuriousness scores...")
    gpu = ",".join([str(i) for i in utils.get_free_gpu()[0:1]])
    utils.set_gpu(gpu)
    model = REPModel("resnet50", 2, pretrained=True).cuda()
    model.init(loader, use_all=True, batch_size=64)

    class_correlated = get_correlated_features(model, loader, score_func="tanh-abs-log")
    for c, (scores, indices) in class_correlated.items():
        print(f"  Class {c}: {len(indices)} active concepts, top-5 scores: {np.sort(scores)[-5:][::-1].round(3)}")

    # --- Sample tasks ---
    print(f"\nSampling {NUM_TASKS} tasks...")
    sampler = AttributesTaskSampler(
        dataset=ds,
        num_supp=NUM_SUPP,
        num_query=NUM_QUERY,
        num_batches=1,
        task_num=NUM_TASKS,
        topk=TOP_K,
        class_correlated_feas=class_correlated,
    )

    tasks = []
    task_batch = next(iter(sampler))
    # task_batch is (2*NUM_SUPP + 2*NUM_QUERY) * NUM_TASKS total indices
    samples_per_task = 2 * (NUM_SUPP + NUM_QUERY)  # 2 classes × (10 supp + 10 query) = 40
    for t in range(NUM_TASKS):
        start = t * samples_per_task
        end = start + samples_per_task
        indices = task_batch[start:end].numpy()

        supp_idx = indices[: 2 * NUM_SUPP]
        query_idx = indices[2 * NUM_SUPP :]

        # Get metadata for these indices (use ds_meta which indexes into ds directly)
        supp_meta = [ds_meta[idx] for idx in supp_idx if ds_meta[idx] is not None]
        query_meta = [ds_meta[idx] for idx in query_idx if ds_meta[idx] is not None]

        # Get concepts
        supp_concepts = set()
        query_concepts = set()
        for idx in supp_idx:
            emb = ds.embeddings[idx]
            for cid in range(len(vocab)):
                if emb[cid] == 1:
                    supp_concepts.add(vocab[cid].split(":")[0])
        for idx in query_idx:
            emb = ds.embeddings[idx]
            for cid in range(len(vocab)):
                if emb[cid] == 1:
                    query_concepts.add(vocab[cid].split(":")[0])

        tasks.append({
            "task_id": t,
            "supp_indices": supp_idx.tolist(),
            "query_indices": query_idx.tolist(),
            "supp_concepts": sorted(supp_concepts),
            "query_concepts": sorted(query_concepts),
            "supp_classes": [m["class_name"] for m in supp_meta],
            "query_classes": [m["class_name"] for m in query_meta],
            "supp_contexts": [m.get("context_name", m.get("env", "")) for m in supp_meta],
            "query_contexts": [m.get("context_name", m.get("env", "")) for m in query_meta],
        })

    # --- Analysis ---
    print("\n" + "=" * 72)
    print("TASK ANALYSIS")
    print("=" * 72)

    empty_query = 0
    trivial_tasks = 0
    concept_leakage = 0
    genuine_shift = 0
    overlap_rates = []
    ctx_diff_rates = []

    for task in tasks:
        supp_c = set(task["supp_concepts"])
        query_c = set(task["query_concepts"])
        overlap = supp_c & query_c
        only_supp = supp_c - query_c
        only_query = query_c - supp_c

        overlap_rate = len(overlap) / max(1, len(supp_c | query_c))
        overlap_rates.append(overlap_rate)

        supp_ctx = Counter(task["supp_contexts"])
        query_ctx = Counter(task["query_contexts"])
        ctx_overlap = set(supp_ctx.keys()) & set(query_ctx.keys())
        ctx_diff = set(query_ctx.keys()) - set(supp_ctx.keys())
        ctx_diff_rates.append(len(ctx_diff) / max(1, len(query_ctx)))

        if len(query_c) == 0:
            empty_query += 1
        if overlap_rate > 0.8:
            trivial_tasks += 1
        if overlap_rate < 0.5:
            concept_leakage += 1
        if len(only_query) > 0:
            genuine_shift += 1

    print(f"\n  Total tasks sampled: {len(tasks)}")
    print(f"  Empty queries: {empty_query}/{NUM_TASKS}")
    print(f"  Trivial tasks (overlap > 80%): {trivial_tasks}/{NUM_TASKS}")
    print(f"  Genuine concept shift (new concepts in query): {genuine_shift}/{NUM_TASKS}")
    print(f"  Mean concept overlap: {np.mean(overlap_rates):.2%}")
    print(f"  Mean new context rate in query: {np.mean(ctx_diff_rates):.2%}")

    # --- Detailed task samples ---
    print("\n--- Sample Tasks ---")
    for task in tasks[:5]:
        supp_c = set(task["supp_concepts"])
        query_c = set(task["query_concepts"])
        only_query = query_c - supp_c
        print(f"\n  Task {task['task_id']}:")
        print(f"    Support contexts: {Counter(task['supp_contexts']).most_common(3)}")
        print(f"    Query contexts:   {Counter(task['query_contexts']).most_common(3)}")
        print(f"    New concepts in query: {sorted(only_query)[:10]}")
        print(f"    Concept overlap: {len(supp_c & query_c)}/{len(supp_c | query_c)}")

    # --- Concept frequency in support vs query ---
    supp_concept_freq = Counter()
    query_concept_freq = Counter()
    for task in tasks:
        supp_concept_freq.update(task["supp_concepts"])
        query_concept_freq.update(task["query_concepts"])

    print(f"\n--- Top Support vs Query Concepts ---")
    print(f"  Top Support: {supp_concept_freq.most_common(10)}")
    print(f"  Top Query:   {query_concept_freq.most_common(10)}")

    # --- Save report ---
    lines = [
        "# SPUME MetaShift Task Construction Debug Report",
        "",
        f"- **Vocab size**: {len(vocab)}",
        f"- **Tasks sampled**: {NUM_TASKS}",
        f"- **Support size**: {NUM_SUPP} per class",
        f"- **Query size**: {NUM_QUERY} per class",
        f"- **Top-K spurious attributes**: {TOP_K}",
        "",
        "## Task Quality Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Empty queries | {empty_query}/{NUM_TASKS} |",
        f"| Trivial tasks (concept overlap > 80%) | {trivial_tasks}/{NUM_TASKS} |",
        f"| Genuine shift (new concepts in query) | {genuine_shift}/{NUM_TASKS} |",
        f"| Mean concept overlap | {np.mean(overlap_rates):.1%} |",
        f"| Mean new context rate in query | {np.mean(ctx_diff_rates):.1%} |",
        "",
        "## Interpretation",
        "",
        "- **Empty queries**: should be 0. If >0, spuriousness sampling fails.",
        f"- **Genuine shift**: {genuine_shift}/{NUM_TASKS} tasks have concepts in query that don't appear in support. " +
        "This is the intended SPUME mechanism — support and query should differ.",
        f"- **Concept overlap**: {np.mean(overlap_rates):.1%} overlap between support and query concepts. " +
        "Lower is better for SPUME (forces model to not rely on spurious concepts).",
        "",
        "## Verdict",
        "",
    ]
    if empty_query == 0 and genuine_shift > 40:
        lines.append("**PASS** — Task construction is correct. Support and query sets have different concept distributions.")
    elif empty_query > 0:
        lines.append("**FAIL** — Some tasks have empty queries. Spuriousness sampling may be broken.")
    elif genuine_shift < 30:
        lines.append("**WARNING** — Few tasks have genuine concept shift. SPUME may not be effective.")
    else:
        lines.append("**OK** — Tasks have reasonable structure but could be improved.")

    report_path = ANALYSIS / "task_debug_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
