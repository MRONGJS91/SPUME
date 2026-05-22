import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

import utils
from data.dataloader import get_loader
from data.sampler import AttributesTaskSampler
from methods import REPModel, get_correlated_features


CLASS_NAMES = {0: "landbird", 1: "waterbird"}


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return utils.get_config_from_yaml(config_path)


def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        return pickle.load(f)


def open_dataset_image(dataset, idx):
    rel_path = dataset.filename_array[idx]
    return Image.open(os.path.join(dataset.basedir, "images", rel_path)).convert("RGB")


def make_task_figure(train_dataset, model, idx_train_loader, vocab, save_path, args):
    class_features = get_correlated_features(model, idx_train_loader, args.score_func)
    sampler = AttributesTaskSampler(
        train_dataset,
        args.num_supp,
        args.num_query,
        num_batches=1,
        task_num=1,
        topk=args.topk,
        class_correlated_feas=class_features,
    )
    batch = next(iter(sampler)).numpy()

    num_classes = args.n_classes
    supp = batch[: args.num_supp * num_classes]
    query = batch[args.num_supp * num_classes :]

    fig, axes = plt.subplots(
        2,
        args.num_supp * num_classes,
        figsize=(2.2 * args.num_supp * num_classes, 5.2),
    )
    for ax in axes.reshape(-1):
        ax.axis("off")

    for i, idx in enumerate(supp):
        img = open_dataset_image(train_dataset, idx)
        label = int(train_dataset.y_array[idx])
        place = int(train_dataset.p_array[idx])
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"S {CLASS_NAMES[label]}\nplace={place}", fontsize=9)

    for i, idx in enumerate(query):
        img = open_dataset_image(train_dataset, idx)
        label = int(train_dataset.y_array[idx])
        place = int(train_dataset.p_array[idx])
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Q {CLASS_NAMES[label]}\nplace={place}", fontsize=9)

    fig.suptitle("SPUME-style spuriousness-aware task on Waterbirds", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_score_figure(train_dataset, idx_train_loader, vocab, checkpoint_path, save_path, args):
    before_model = REPModel(args.backbone, args.n_classes, args.pretrained).cuda()
    before_model.init(idx_train_loader, use_all=True, batch_size=32, num_workers=0)
    before_scores = get_correlated_features(before_model, idx_train_loader, args.score_func)

    after_model = REPModel(args.backbone, args.n_classes, args.pretrained).cuda()
    state_dict = torch.load(checkpoint_path, map_location="cuda")
    after_model.load_state_dict(state_dict)
    after_model.init(idx_train_loader, use_all=True, batch_size=32, num_workers=0)
    after_scores = get_correlated_features(after_model, idx_train_loader, args.score_func)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for class_id in range(args.n_classes):
        scores_before, active_before = before_scores[class_id]
        order = np.argsort(-scores_before)
        sorted_before = scores_before[order]
        sorted_active = active_before[order]

        after_map = {
            int(attr_idx): float(score)
            for score, attr_idx in zip(after_scores[class_id][0], after_scores[class_id][1])
        }
        aligned_after = np.array([after_map.get(int(attr_idx), 0.0) for attr_idx in sorted_active])

        top_labels = [str(vocab[idx]) for idx in sorted_active[:10]]
        x = np.arange(len(sorted_before))

        axes[class_id, 0].plot(x, sorted_before, color="#1f77b4", lw=1.5)
        axes[class_id, 0].set_title(f"{CLASS_NAMES[class_id]} before")
        axes[class_id, 0].set_ylabel("spuriousness score")
        axes[class_id, 0].grid(alpha=0.25)

        axes[class_id, 1].plot(x, aligned_after, color="#d62728", lw=1.5)
        axes[class_id, 1].set_title(f"{CLASS_NAMES[class_id]} after")
        axes[class_id, 1].grid(alpha=0.25)

        if len(top_labels) > 0:
            tick_pos = np.arange(min(10, len(top_labels)))
            axes[class_id, 1].set_xticks(tick_pos)
            axes[class_id, 1].set_xticklabels(top_labels, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Spuriousness scores before and after SPUME training", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    args_cli = parser.parse_args()

    args = load_config(args_cli.config)
    train_loader, idx_train_loader, _, _ = get_loader(args)
    vocab = load_vocab(args.vit_gpt2_vocab_path)

    model = REPModel(args.backbone, args.n_classes, args.pretrained).cuda()
    state_dict = torch.load(args_cli.checkpoint, map_location="cuda")
    model.load_state_dict(state_dict)
    model.init(idx_train_loader, use_all=True, batch_size=32, num_workers=0)

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_task_figure(
        train_loader.dataset,
        model,
        idx_train_loader,
        vocab,
        out_dir / "task_visualization.png",
        args,
    )
    make_score_figure(
        train_loader.dataset,
        idx_train_loader,
        vocab,
        args_cli.checkpoint,
        out_dir / "spuriousness_scores.png",
        args,
    )
    print(f"saved figures to {out_dir}")


if __name__ == "__main__":
    main()
