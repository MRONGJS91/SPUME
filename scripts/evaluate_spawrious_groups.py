#!/usr/bin/env python3
"""Evaluate Spawrious checkpoints with explicit per-group accuracies.

This script is intentionally separate from the training code.  It mirrors the
current Spawrious group definition in `data/spawrious_data.py`:

    group = class_id * num_envs_in_split + env_id

where `env_id` is assigned by sorting the metadata `env` values inside the
evaluated split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SPUME_ROOT = REPO_ROOT / "SPUME-master"
sys.path.insert(0, str(SPUME_ROOT))

from data.biased_dataset import get_transform_biased  # noqa: E402
from data.spawrious_data import SpawriousDataset  # noqa: E402
from models.resnet import resnet18, resnet50  # noqa: E402


@dataclass(frozen=True)
class EvalSpec:
    dataset_name: str
    method: str
    checkpoint_type: str
    model_type: str
    checkpoint: Path
    data_root: Path
    metadata_path: Path


class ERMCosineEvalModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.weights = nn.Parameter(
            torch.normal(0, 0.1, size=(num_classes, self.backbone.out_dim))
        )

    def forward(self, x):
        fea = self.backbone(x)
        return torch.matmul(F.normalize(fea, dim=-1), F.normalize(self.weights, dim=-1).T)


class REPEvalModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.n_classes = num_classes
        self.centroids = None

    def forward(self, x):
        if self.centroids is None:
            raise RuntimeError("REP centroids have not been initialized")
        fea = self.backbone(x)
        return torch.matmul(
            F.normalize(fea, dim=-1), F.normalize(self.centroids, dim=-1).T
        )


def build_backbone(backbone: str):
    if backbone == "resnet18":
        return resnet18()
    if backbone == "resnet50":
        return resnet50()
    raise ValueError(f"Unsupported backbone: {backbone}")


def make_test_transform():
    return get_transform_biased(
        target_resolution=(224, 224), train=False, augment_data=False
    )


def default_specs() -> list[EvalSpec]:
    base = REPO_ROOT / "how_to_run" / "meta_spurious_exprs"
    easy_data = REPO_ROOT / "spawrious224__o2o_easy"
    medium_data = REPO_ROOT / "spawrious224__o2o_medium"
    easy_meta = SPUME_ROOT / "data" / "spawrious_o2o_easy_metadata.csv"
    medium_meta = SPUME_ROOT / "data" / "spawrious_o2o_medium_metadata.csv"

    return [
        EvalSpec(
            "spawrious_o2o_easy",
            "ERM",
            "latest_model",
            "erm_cosine",
            base
            / "pretrain_cosine_spawrious_o2o_easy_resnet18_04192026-205116_xinxin的book"
            / "latest_model.pt",
            easy_data,
            easy_meta,
        ),
        EvalSpec(
            "spawrious_o2o_easy",
            "Mixup",
            "latest_model",
            "erm_cosine",
            base
            / "pretrain_cosine_spawrious_o2o_easy_resnet18_04242026-193319_xinxin的book"
            / "latest_model.pt",
            easy_data,
            easy_meta,
        ),
        EvalSpec(
            "spawrious_o2o_easy",
            "SPUME-ViT-GPT2",
            "worst_model",
            "rep",
            base
            / "spawrious_o2o_easy_resnet18_lr_0.001000_train_8B_1topK_10ns_10nq_100Epochs_80Epi_1T_vit-gpt2_5.00temp_1.0alpha_tanh-abs-log_cosine_scheduler_imagenet_weights_vitgpt2_spume_full"
            / "worst_model.pt",
            easy_data,
            easy_meta,
        ),
        EvalSpec(
            "spawrious_o2o_easy",
            "SPUME-BLIP",
            "worst_model",
            "rep",
            base
            / "spawrious_o2o_easy_resnet18_lr_0.001000_train_8B_1topK_10ns_10nq_100Epochs_80Epi_1T_blip_5.00temp_1.0alpha_tanh-abs-log_cosine_scheduler_imagenet_weights_blip_spume_full"
            / "worst_model.pt",
            easy_data,
            easy_meta,
        ),
        EvalSpec(
            "spawrious_o2o_medium",
            "ERM",
            "latest_model",
            "erm_cosine",
            base
            / "pretrain_cosine_spawrious_o2o_medium_resnet18_04272026-003422_xinxin的book"
            / "latest_model.pt",
            medium_data,
            medium_meta,
        ),
        EvalSpec(
            "spawrious_o2o_medium",
            "Mixup",
            "latest_model",
            "erm_cosine",
            base
            / "pretrain_cosine_spawrious_o2o_medium_resnet18_04272026-012117_xinxin的book"
            / "latest_model.pt",
            medium_data,
            medium_meta,
        ),
        EvalSpec(
            "spawrious_o2o_medium",
            "SPUME-ViT-GPT2",
            "avg_model_embed",
            "rep",
            base
            / "spawrious_o2o_medium_resnet18_lr_0.001000_train_8B_1topK_10ns_10nq_100Epochs_80Epi_1T_vit-gpt2_5.00temp_1.0alpha_tanh-abs-log_cosine_scheduler_imagenet_weights_vitgpt2_spume_medium_full"
            / "avg_model_embed.pt",
            medium_data,
            medium_meta,
        ),
        EvalSpec(
            "spawrious_o2o_medium",
            "SPUME-BLIP",
            "avg_model_embed",
            "rep",
            base
            / "spawrious_o2o_medium_resnet18_lr_0.001000_train_8B_1topK_10ns_10nq_100Epochs_80Epi_1T_blip_5.00temp_1.0alpha_tanh-abs-log_cosine_scheduler_imagenet_weights_blip_spume_medium_full"
            / "avg_model_embed.pt",
            medium_data,
            medium_meta,
        ),
    ]


def load_state(model: nn.Module, checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def initialize_rep_centroids(
    model: REPEvalModel,
    train_dataset: SpawriousDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    sums = torch.zeros(model.n_classes, model.backbone.out_dim, device=device)
    counts = torch.zeros(model.n_classes, device=device)
    model.eval()
    with torch.inference_mode():
        for x, y, _, _, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            fea = model.backbone(x)
            for cls in range(model.n_classes):
                mask = y == cls
                if mask.any():
                    sums[cls] += fea[mask].sum(dim=0)
                    counts[cls] += mask.sum()
    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].detach().cpu().tolist()
        raise ValueError(f"Cannot initialize REP centroids; empty classes: {missing}")
    model.centroids = sums / counts.unsqueeze(1)


def evaluate_spec(
    spec: EvalSpec,
    split: str,
    device: torch.device,
    batch_size: int,
    centroid_batch_size: int,
    num_workers: int,
    backbone: str,
):
    if not spec.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {spec.checkpoint}")

    transform = make_test_transform()
    dataset = SpawriousDataset(
        data_root=spec.data_root,
        metadata_path=spec.metadata_path,
        split=split,
        transform=transform,
        concept_embed=None,
        return_metadata=False,
    )

    if spec.model_type == "erm_cosine":
        model = ERMCosineEvalModel(backbone, dataset.n_classes)
        model.to(device)
        missing, unexpected = load_state(model, spec.checkpoint, device)
    elif spec.model_type == "rep":
        model = REPEvalModel(backbone, dataset.n_classes)
        model.to(device)
        missing, unexpected = load_state(model, spec.checkpoint, device)
        train_dataset = SpawriousDataset(
            data_root=spec.data_root,
            metadata_path=spec.metadata_path,
            split="train",
            transform=transform,
            concept_embed=None,
            return_metadata=False,
        )
        initialize_rep_centroids(
            model,
            train_dataset,
            device,
            batch_size=centroid_batch_size,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unsupported model type: {spec.model_type}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    rows = []
    correct_total = 0
    count_total = 0
    group_stats = {}
    with torch.inference_mode():
        for x, y, g, p, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct = (pred == y).detach().cpu()
            g_cpu = g.detach().cpu()
            for group_id, ok in zip(g_cpu.tolist(), correct.tolist()):
                stat = group_stats.setdefault(int(group_id), {"count": 0, "correct": 0})
                stat["count"] += 1
                stat["correct"] += int(ok)
                count_total += 1
                correct_total += int(ok)

    metadata = dataset.metadata_df.copy()
    metadata["_env_id"] = dataset.p_array
    metadata["_group_id"] = dataset.group_array
    group_names = (
        metadata.groupby(["_group_id", "class_name", "y", "env", "_env_id"], dropna=False)
        .size()
        .reset_index(name="metadata_count")
    )

    for _, row in group_names.sort_values("_group_id").iterrows():
        group_id = int(row["_group_id"])
        stat = group_stats.get(group_id, {"count": 0, "correct": 0})
        count = int(stat["count"])
        correct = int(stat["correct"])
        acc = correct / count if count else float("nan")
        rows.append(
            {
                "dataset": spec.dataset_name,
                "split": split,
                "method": spec.method,
                "checkpoint_type": spec.checkpoint_type,
                "group_id": group_id,
                "group_name": f"{row['class_name']} x {row['env']}",
                "class_name": row["class_name"],
                "class_id": int(row["y"]),
                "env": row["env"],
                "env_id": int(row["_env_id"]),
                "sample_count": count,
                "correct": correct,
                "accuracy": acc,
            }
        )

    valid_acc = [r["accuracy"] for r in rows if r["sample_count"] >= 10]
    summary = {
        "dataset": spec.dataset_name,
        "split": split,
        "method": spec.method,
        "checkpoint_type": spec.checkpoint_type,
        "model_type": spec.model_type,
        "checkpoint": str(spec.checkpoint),
        "sample_count": count_total,
        "num_classes": int(dataset.n_classes),
        "num_envs_in_split": int(dataset.n_places),
        "num_observed_groups": len(rows),
        "group_definition": "group_id = class_id * num_envs_in_split + env_id",
        "env_id_definition": "env_id is assigned from sorted metadata env values in the evaluated split",
        "avg_acc": correct_total / count_total,
        "worst_group_acc": min(valid_acc),
        "unbiased_group_acc": sum(valid_acc) / len(valid_acc),
        "missing_state_keys": len(missing),
        "unexpected_state_keys": len(unexpected),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split",
        "method",
        "checkpoint_type",
        "group_id",
        "group_name",
        "class_name",
        "class_id",
        "env",
        "env_id",
        "sample_count",
        "correct",
        "accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summaries: list[dict], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spawrious Per-Group Accuracy Report",
        "",
        "Group definition used by the current loader:",
        "",
        "`group_id = class_id * num_envs_in_split + env_id`",
        "",
        "`env_id` is assigned from sorted metadata `env` values inside the evaluated split.",
        "Therefore current worst-group accuracy is computed over class x env groups, not over class only.",
        "",
        "## Summary",
        "",
        "| dataset | method | checkpoint | samples | classes | envs | groups | avg acc | worst-group acc | unbiased-group acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {dataset} | {method} | {checkpoint_type} | {sample_count} | "
            "{num_classes} | {num_envs_in_split} | {num_observed_groups} | "
            "{avg_acc:.6f} | {worst_group_acc:.6f} | {unbiased_group_acc:.6f} |".format(
                **s
            )
        )

    for dataset_name in sorted({r["dataset"] for r in rows}):
        lines.extend(["", f"## {dataset_name}", ""])
        methods = []
        for r in rows:
            if r["dataset"] == dataset_name and r["method"] not in methods:
                methods.append(r["method"])
        for method in methods:
            method_rows = [
                r
                for r in rows
                if r["dataset"] == dataset_name and r["method"] == method
            ]
            if not method_rows:
                continue
            lines.extend(
                [
                    f"### {method}",
                    "",
                    "| group id | group name | samples | correct | accuracy |",
                    "|---:|---|---:|---:|---:|",
                ]
            )
            for r in method_rows:
                lines.append(
                    "| {group_id} | {group_name} | {sample_count} | {correct} | {accuracy:.6f} |".format(
                        **r
                    )
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate current Spawrious checkpoints per class x env group."
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--centroid-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional substring filter, e.g. easy, medium, spawrious_o2o_easy.",
    )
    parser.add_argument(
        "--method",
        default="",
        help="Optional substring filter, e.g. ERM, Mixup, SPUME-ViT-GPT2.",
    )
    parser.add_argument(
        "--csv-out",
        default=str(REPO_ROOT / "how_to_run" / "spawrious_per_group_accuracy.csv"),
    )
    parser.add_argument(
        "--json-out",
        default=str(REPO_ROOT / "how_to_run" / "spawrious_per_group_accuracy.json"),
    )
    parser.add_argument(
        "--md-out",
        default=str(REPO_ROOT / "how_to_run" / "spawrious_per_group_accuracy.md"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    summaries = []
    all_rows = []
    specs = default_specs()
    if args.dataset:
        specs = [s for s in specs if args.dataset.lower() in s.dataset_name.lower()]
    if args.method:
        specs = [s for s in specs if args.method.lower() in s.method.lower()]
    if not specs:
        raise ValueError("No evaluation specs matched the provided filters.")
    for spec in specs:
        print(
            f"Evaluating {spec.dataset_name} / {spec.method} / {spec.checkpoint_type}",
            flush=True,
        )
        summary, rows = evaluate_spec(
            spec,
            split=args.split,
            device=device,
            batch_size=args.batch_size,
            centroid_batch_size=args.centroid_batch_size,
            num_workers=args.num_workers,
            backbone=args.backbone,
        )
        summaries.append(summary)
        all_rows.extend(rows)
        print(
            f"{summary['dataset']} / {summary['method']}: "
            f"avg={summary['avg_acc']:.6f}, "
            f"worst={summary['worst_group_acc']:.6f}, "
            f"groups={summary['num_observed_groups']}",
            flush=True,
        )

    csv_out = Path(args.csv_out)
    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    write_csv(csv_out, all_rows)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps({"summaries": summaries, "groups": all_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(md_out, summaries, all_rows)
    print(f"Wrote {csv_out}")
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
