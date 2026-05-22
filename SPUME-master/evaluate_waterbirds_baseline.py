import argparse
import json
import os

import torch
import yaml

import utils
from data.dataloader import get_loader
from pretrain import prepare_model
from test import test_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = utils.EasyDict(yaml.safe_load(f))
    cfg.config_file = args.config
    cfg.seed = getattr(cfg, "seed", 100)
    cfg.n_gpu = getattr(cfg, "n_gpu", 1)
    gpu = ",".join([str(i) for i in utils.get_free_gpu()[0:1]])
    utils.set_gpu(gpu)

    train_loader, idx_train_loader, val_loader, test_loader = get_loader(cfg)
    model = prepare_model(cfg.dataset, cfg.backbone, pretrained=cfg.pretrained)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict)

    val_avg, val_worst, val_unbiased = test_model(model, val_loader)
    test_avg, test_worst, test_unbiased = test_model(model, test_loader)

    result = {
        "ckpt": args.ckpt,
        "val": {
            "avg_acc": float(val_avg),
            "worst_group_acc": float(val_worst),
            "unbiased_acc": float(val_unbiased),
            "accuracy_gap": float(val_avg - val_worst),
        },
        "test": {
            "avg_acc": float(test_avg),
            "worst_group_acc": float(test_worst),
            "unbiased_acc": float(test_unbiased),
            "accuracy_gap": float(test_avg - test_worst),
        },
    }

    output_path = args.output
    if not output_path:
        output_path = os.path.join(os.path.dirname(args.ckpt), "evaluation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
