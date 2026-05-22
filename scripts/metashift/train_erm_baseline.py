"""ERM baseline training for MetaShift Cat vs Dog.

Features:
- ResNet18 (ImageNet pretrained), 224x224 input
- SGD with cosine annealing
- Per-epoch: loss, avg acc, worst-group acc, accuracy gap
- Saves latest_model.pt, best_avg.pt, best_worst_group.pt
- Generates training curves (loss, avg acc, worst-group acc)
- Full per-group accuracy evaluation on test set
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SPUME-master"))

from data.metashift_data import MetaShiftDataset
from data.biased_dataset import get_transform_biased
from utils import set_gpu, get_free_gpu, set_seed, AverageMeter

OUT_DIR = ROOT / "analysis" / "metashift" / "erm"  # overridden by --mixup_alpha > 0

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ERMResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        d = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(d, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_loaders(batch_size=64, num_workers=4):
    metadata_path = ROOT / "datasets" / "metashift" / "metadata_metashift_catdog.csv"
    data_root = ROOT / "datasets" / "metashift"

    train_tf = get_transform_biased(target_resolution=(224, 224), train=True, augment_data=True)
    test_tf = get_transform_biased(target_resolution=(224, 224), train=False, augment_data=False)

    train_ds = MetaShiftDataset(data_root=data_root, metadata_path=metadata_path,
                                split="train", transform=train_tf, return_metadata=True)
    val_ds = MetaShiftDataset(data_root=data_root, metadata_path=metadata_path,
                              split="val", transform=test_tf, return_metadata=True)
    test_ds = MetaShiftDataset(data_root=data_root, metadata_path=metadata_path,
                               split="test", transform=test_tf, return_metadata=True)

    kwargs = {"num_workers": num_workers, "pin_memory": True}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    """Return (avg_acc, worst_group_acc, accuracy_gap, per_group_dict)."""
    model.eval()
    all_preds, all_labels, all_groups = [], [], []
    for image, label, meta in loader:
        image = image.cuda()
        logits = model(image)
        preds = torch.argmax(logits, dim=-1).cpu()
        all_preds.append(preds)
        all_labels.append(label)
        all_groups.append(meta["group_id"])

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    groups = torch.cat(all_groups).numpy()

    avg_acc = float((preds == labels).mean())

    # Per-group accuracy
    unique_groups = np.unique(groups)
    per_group = {}
    group_accs = []
    for g in unique_groups:
        mask = groups == g
        if mask.sum() >= 5:
            acc = float((preds[mask] == labels[mask]).mean())
            per_group[int(g)] = {"count": int(mask.sum()), "accuracy": acc}
            group_accs.append(acc)

    worst_group_acc = float(np.min(group_accs)) if group_accs else 0.0
    accuracy_gap = avg_acc - worst_group_acc
    return avg_acc, worst_group_acc, accuracy_gap, per_group


def mixup_batch(x, y, alpha):
    if alpha <= 0:
        return x, y, y, torch.ones(x.size(0), device=x.device)
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    global OUT_DIR
    if args.mixup_alpha > 0:
        OUT_DIR = ROOT / "analysis" / "metashift" / "mixup"
        print(f"Mixup mode: alpha={args.mixup_alpha}, output -> {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # GPU
    free = get_free_gpu()
    gpu_id = ",".join([str(i) for i in free[0:1]])
    set_gpu(gpu_id)
    set_seed(args.seed)

    # Data
    train_loader, val_loader, test_loader = build_loaders(args.batch_size, args.num_workers)
    print(f"Train: {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}  Test: {len(test_loader.dataset):,}")

    # Model
    model = ERMResNet(num_classes=2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = nn.CrossEntropyLoss()

    # Tracking
    log_rows = []
    best_avg = 0.0
    best_worst = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for image, label, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False):
            image, label = image.cuda(), label.cuda()
            optimizer.zero_grad()

            if args.mixup_alpha > 0:
                mixed_x, y_a, y_b, lam = mixup_batch(image, label, args.mixup_alpha)
                logits = model(mixed_x)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                logits = model(image)
                loss = criterion(logits, label)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), image.size(0))
            acc_meter.update((torch.argmax(logits, dim=-1) == label).float().mean().item(), image.size(0))

        scheduler.step()

        # Validation (every epoch for comprehensive logging)
        val_avg, val_worst, val_gap, val_per_group = evaluate(model, val_loader)

        train_loss = loss_meter.avg
        train_acc = acc_meter.avg

        log_rows.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_avg_acc": round(val_avg, 4),
            "val_worst_acc": round(val_worst, 4),
            "val_gap": round(val_gap, 4),
        })

        # Save latest
        torch.save(model.state_dict(), OUT_DIR / "latest_model.pt")

        # Save best avg
        if val_avg > best_avg:
            best_avg = val_avg
            torch.save(model.state_dict(), OUT_DIR / "best_avg.pt")

        # Save best worst-group
        if val_worst > best_worst:
            best_worst = val_worst
            torch.save(model.state_dict(), OUT_DIR / "best_worst_group.pt")

        print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Avg: {val_avg:.4f} | "
              f"Val Worst: {val_worst:.4f} | Gap: {val_gap:.4f}")

    # ---- Final evaluation on test set ----
    print("\n--- Final Test Evaluation ---")
    # Best avg model
    model.load_state_dict(torch.load(OUT_DIR / "best_avg.pt", map_location="cuda"))
    test_avg, test_worst, test_gap, test_per_group = evaluate(model, test_loader)
    print(f"[best_avg] Avg: {test_avg:.4f}  Worst: {test_worst:.4f}  Gap: {test_gap:.4f}")

    # Best worst-group model
    model.load_state_dict(torch.load(OUT_DIR / "best_worst_group.pt", map_location="cuda"))
    _, tw2, tg2, tgp2 = evaluate(model, test_loader)
    print(f"[best_worst] Avg: {_:.4f}  Worst: {tw2:.4f}  Gap: {tg2:.4f}")

    # ---- Save logs ----
    # Training log CSV
    with open(OUT_DIR / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys())); w.writeheader(); w.writerows(log_rows)

    # Per-group accuracy CSV (using best_avg model)
    with open(OUT_DIR / "per_group_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "count", "accuracy"])
        for gid, info in sorted(test_per_group.items()):
            w.writerow([gid, info["count"], round(info["accuracy"], 4)])

    # Summary JSON
    summary = {
        "model": "ResNet18_ERM",
        "dataset": "metashift_catdog",
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "test_best_avg": {"avg_acc": test_avg, "worst_group_acc": test_worst, "accuracy_gap": test_gap},
        "test_best_worst": {"avg_acc": float(_), "worst_group_acc": tw2, "accuracy_gap": tg2},
        "n_groups": len(test_per_group),
        "groups": {str(k): v for k, v in sorted(test_per_group.items())},
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- Training curves ----
    epochs = [r["epoch"] for r in log_rows]
    train_losses = [r["train_loss"] for r in log_rows]
    train_accs = [r["train_acc"] for r in log_rows]
    val_avgs = [r["val_avg_acc"] for r in log_rows]
    val_worsts = [r["val_worst_acc"] for r in log_rows]
    val_gaps = [r["val_gap"] for r in log_rows]

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei","SimHei","Arial"],
                         "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_losses, color="#2171b5", linewidth=1.5)
    axes[0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, label="Train", color="#2171b5", linewidth=1.5)
    axes[1].plot(epochs, val_avgs, label="Val Avg", color="#2ca02c", linewidth=1.5)
    axes[1].plot(epochs, val_worsts, label="Val Worst-Group", color="#c62828", linewidth=1.5)
    axes[1].set_title("Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, val_gaps, color="#e67e22", linewidth=1.5)
    axes[2].set_title("Accuracy Gap (Avg - Worst)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Gap")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("MetaShift ERM Baseline (ResNet18)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png"); plt.close(fig)
    print(f"\nTraining curves saved to: {OUT_DIR / 'training_curves.png'}")

    print(f"\nAll outputs in: {OUT_DIR}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ERM baseline for MetaShift Cat vs Dog")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="Mixup alpha (0 = no mixup, 0.2 = standard mixup)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
