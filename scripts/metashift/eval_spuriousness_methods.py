"""Compare spuriousness score distributions across methods."""
import sys, pickle, csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SPUME-master"))
from data.metashift_data import MetaShiftDataset
from data.biased_dataset import get_transform_biased, IdxDataset
from torch.utils.data import DataLoader
from methods import REPModel, get_correlated_features
import torch, utils

EMBED_PATH = ROOT / "datasets/metashift/blip_extracted_concepts_0_0_img_embeddings.pickle"
KEYS_PATH = ROOT / "datasets/metashift/blip_extracted_concepts_0_0_img_embedding_keys.pickle"
META_PATH = ROOT / "datasets/metashift/metadata_metashift_catdog.csv"
VOCAB_PATH = ROOT / "datasets/metashift/blip_extracted_concepts_0_0_vocab.pickle"


def build_loader(split="train"):
    tf = get_transform_biased(target_resolution=(224, 224), train=False, augment_data=False)
    ds = MetaShiftDataset(
        data_root=str(ROOT / "datasets/metashift"),
        metadata_path=str(META_PATH), split=split, transform=tf,
        concept_embed=str(EMBED_PATH), return_metadata=False,
    )
    return DataLoader(IdxDataset(ds), batch_size=64, shuffle=False, num_workers=0)


def evaluate_scores(loader, score_func, name):
    gpu = ",".join([str(i) for i in utils.get_free_gpu()[0:1]])
    utils.set_gpu(gpu)
    model = REPModel("resnet50", 2, pretrained=True).cuda()
    model.init(loader, use_all=True, batch_size=64)
    scores = get_correlated_features(model, loader, score_func=score_func)

    all_scores = []
    n_saturated = 0
    for c, (s_arr, indices) in scores.items():
        all_scores.extend(s_arr.tolist())
        n_saturated += (s_arr > 0.99).sum()

    all_scores = np.array(all_scores)
    if len(all_scores) == 0:
        return {"name": name, "n_concepts": 0, "saturated": 0, "mean": 0, "std": 0}

    return {
        "name": name,
        "n_concepts": len(all_scores),
        "saturated": int(n_saturated),
        "saturated_pct": float(n_saturated / len(all_scores)),
        "mean": float(all_scores.mean()),
        "std": float(all_scores.std()),
        "min": float(all_scores.min()),
        "p25": float(np.percentile(all_scores, 25)),
        "p50": float(np.percentile(all_scores, 50)),
        "p75": float(np.percentile(all_scores, 75)),
        "max": float(all_scores.max()),
        "unique_vals": len(set(all_scores.round(3))),
    }


def custom_scores_bayes(loader, alpha=1.0, lambda_penalty=0.01):
    """Bayes-smoothed spuriousness (Laplace smoothing + sample-size penalty)."""
    eps = 1e-10
    gpu = ",".join([str(i) for i in utils.get_free_gpu()[0:1]])
    utils.set_gpu(gpu)
    model = REPModel("resnet50", 2, pretrained=True).cuda()
    model.init(loader, use_all=True, batch_size=64)

    # Collect per-class results
    model.eval()
    class_wise_data = {}
    with torch.no_grad():
        for idx, data, y, _, _, _ in loader:
            logits = model(data.cuda()).detach().cpu()
            preds = torch.argmax(logits, dim=1).numpy()
            for i in range(len(y)):
                l = y[i].item()
                if l not in class_wise_data:
                    class_wise_data[l] = []
                class_wise_data[l].append((idx[i].item(), int(preds[i] == l)))

    embeddings = loader.dataset.dataset.embeddings
    scores_dict = {}

    for c in class_wise_data:
        counts_pos_w = np.zeros(embeddings.shape[1])
        counts_neg_w = np.zeros(embeddings.shape[1])
        counts_pos_wo = np.zeros(embeddings.shape[1])
        counts_neg_wo = np.zeros(embeddings.shape[1])

        for idx, pred_res in class_wise_data[c]:
            if pred_res == 1:
                counts_pos_w[embeddings[idx] == 1] += 1
                counts_pos_wo[embeddings[idx] != 1] += 1
            else:
                counts_neg_w[embeddings[idx] == 1] += 1
                counts_neg_wo[embeddings[idx] != 1] += 1

        active_idx = np.arange(embeddings.shape[1])[(counts_pos_w + counts_neg_w) > 0]

        # Bayes smoothing
        p_w = (counts_pos_w[active_idx] + alpha) / (counts_pos_w[active_idx] + counts_neg_w[active_idx] + 2*alpha + eps)
        p_wo = (counts_pos_wo[active_idx] + alpha) / (counts_pos_wo[active_idx] + counts_neg_wo[active_idx] + 2*alpha + eps)

        # Sample-size penalty
        n_w = counts_pos_w[active_idx] + counts_neg_w[active_idx]
        n_wo = counts_pos_wo[active_idx] + counts_neg_wo[active_idx]
        penalty = 1 - np.exp(-lambda_penalty * np.minimum(n_w, n_wo))

        # Score = weighted absolute difference
        raw_scores = np.abs(p_w - p_wo)
        scores = raw_scores * penalty

        # NaN check
        scores = np.nan_to_num(scores, nan=0.0)
        scores_dict[c] = (scores, active_idx)

    return scores_dict


# --- Run evaluation ---
print("Loading data...")
train_loader = build_loader("train")

results = []
for sf in ["tanh-abs-log", "abs-diff", "abs-log", "log"]:
    print(f"\nComputing: {sf} ...")
    r = evaluate_scores(train_loader, sf, sf)
    results.append(r)

print("\nComputing: Bayes-smoothed ...")
bs = custom_scores_bayes(train_loader, alpha=1.0, lambda_penalty=0.005)
all_bs = []
sat_bs = 0
for c, (s_arr, indices) in bs.items():
    all_bs.extend(s_arr.tolist())
    sat_bs += (s_arr > 0.9).sum()
all_bs = np.array(all_bs)
results.append({
    "name": "bayes-smoothed",
    "n_concepts": len(all_bs),
    "saturated": int(sat_bs),
    "saturated_pct": float(sat_bs / len(all_bs)),
    "mean": float(all_bs.mean()),
    "std": float(all_bs.std()),
    "min": float(all_bs.min()),
    "p25": float(np.percentile(all_bs, 25)),
    "p50": float(np.percentile(all_bs, 50)),
    "p75": float(np.percentile(all_bs, 75)),
    "max": float(all_bs.max()),
    "unique_vals": len(set(all_bs.round(3))),
})

# --- Print comparison ---
print("\n" + "=" * 80)
print("SPURIOUSNESS SCORE COMPARISON")
print("=" * 80)
print(f"{'Method':<20s} {'Concepts':>8s} {'Sat%':>8s} {'Mean':>8s} {'Std':>8s} {'p50':>8s} {'Unique':>8s}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<20s} {r['n_concepts']:>8d} {r['saturated_pct']:>7.1%} "
          f"{r['mean']:>8.4f} {r['std']:>8.4f} {r['p50']:>8.4f} {r['unique_vals']:>8d}")

# Best method by lowest saturation + highest std (more discriminative)
best = max(results, key=lambda r: (1 - r['saturated_pct']) * r['std'] * r['unique_vals'])
print(f"\nBest method: {best['name']} ({best['saturated_pct']:.1%} saturated, std={best['std']:.4f})")
