import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast, GradScaler

from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
import logging
import os
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data.sampler import GroupTaskSampler, AttributesTaskSampler, RandomTaskSampler
import torch.nn.functional as F
import pickle

# import torch.multiprocessing as mp
import copy
from data.dataloader import get_loader

from torch.utils.tensorboard import SummaryWriter

import utils
from methods import ERMModel, get_correlated_features, REPModel
from test import test_model, test_model_pseudo
import yaml


def forward_in_chunks(model, x, chunk_size=0):
    """Run the model forward pass in smaller chunks to reduce peak GPU memory."""
    if chunk_size is None or chunk_size <= 0 or x.size(0) <= chunk_size:
        return model(x.cuda(non_blocking=True), get_fea=True)

    logits_chunks = []
    embed_chunks = []
    for start in range(0, x.size(0), chunk_size):
        end = start + chunk_size
        x_chunk = x[start:end].cuda(non_blocking=True)
        logits_chunk, embed_chunk = model(x_chunk, get_fea=True)
        logits_chunks.append(logits_chunk)
        embed_chunks.append(embed_chunk)
    return torch.cat(logits_chunks, dim=0), torch.cat(embed_chunks, dim=0)


def load_backbone_from_checkpoint(model, ckpt_path, logger=None):
    """Warm-start the backbone from a checkpoint produced by pretrain.py or a full model ckpt."""
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if any(k.startswith("backbone.") for k in state_dict):
        backbone_state = {
            k[len("backbone.") :]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
    else:
        backbone_state = state_dict
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    if logger is not None:
        logger.info(
            f"Loaded warm-start backbone from {ckpt_path}; missing={len(missing)}, unexpected={len(unexpected)}"
        )


def load_full_model_checkpoint(model, ckpt_path, logger=None):
    """Load a full model checkpoint into the current model."""
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if logger is not None:
        logger.info(
            f"Loaded full-model checkpoint from {ckpt_path}; missing={len(missing)}, unexpected={len(unexpected)}"
        )


def save_training_state(
    path,
    model,
    optimizer,
    lr_scheduler,
    scaler,
    epoch,
    best_metrics,
    args,
):
    model_state = model.state_dict() if args.n_gpu == 1 else model.module.state_dict()
    payload = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_metrics": best_metrics,
    }
    torch.save(payload, path)


def maybe_load_training_state(
    path,
    model,
    optimizer,
    lr_scheduler,
    scaler,
    bootstrap_ckpt="",
    bootstrap_epoch=0,
    bootstrap_best_metrics=None,
    logger=None,
):
    default_best = {
        "best_worst_acc": 0,
        "best_worst_acc_psu": 0,
        "best_avg_acc": 0,
        "best_unbiased_pseudo": 0,
    }
    if bootstrap_best_metrics:
        default_best.update(bootstrap_best_metrics)
    if not path or not os.path.exists(path):
        if bootstrap_ckpt and os.path.exists(bootstrap_ckpt):
            load_full_model_checkpoint(model, bootstrap_ckpt, logger)
            if logger is not None:
                logger.info(
                    f"Bootstrapped training from {bootstrap_ckpt} at epoch {bootstrap_epoch}"
                )
            return int(bootstrap_epoch), default_best
        return 0, default_best

    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    if lr_scheduler and payload.get("scheduler_state") is not None:
        lr_scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    start_epoch = int(payload.get("epoch", -1)) + 1
    best_metrics = payload.get("best_metrics", default_best)
    if logger is not None:
        logger.info(f"Resumed training from {path} at epoch {start_epoch}")
    return start_epoch, best_metrics


def init_rep_model(model, idx_train_loader, args, default_use_all):
    """Initialize REPModel centroids with optional config overrides."""
    init_use_all = getattr(args, "init_use_all", default_use_all)
    init_batch_size = getattr(args, "init_batch_size", 64)
    init_num_workers = getattr(args, "init_num_workers", 0)
    model.init(
        idx_train_loader,
        use_all=init_use_all,
        batch_size=init_batch_size,
        num_workers=init_num_workers,
    )


def meta_train(model, train_loader, idx_train_loader, val_loader, test_loader, args):
    """Train the model using the meta-learning strategy

    Args:
        model (torch.nn.Module): a prediction model.
        train_loader (torch.utils.data.DataLoader): a train dataloader.
        idx_train_loader (torch.utils.data.DataLoader): a train dataloader that also returns the indexes of the data.
        val_loader (torch.utils.data.DataLoader): a validation dataloader.
        test_loader (torch.utils.data.DataLoader): a test dataloader.
        args (argparse.Namespace): arguments.

    Returns:
        None
    """
    timer = utils.Timer()
    logger = logging.getLogger("expr")

    # Build IdxDataset-wrapped val loader for spuriousness computation on val split
    from data.biased_dataset import IdxDataset as _IdxDataset
    _val_idx_ds = _IdxDataset(val_loader.dataset)
    _val_idx_loader = torch.utils.data.DataLoader(
        _val_idx_ds, batch_size=val_loader.batch_size, shuffle=False,
        num_workers=val_loader.num_workers, pin_memory=True,
    )

    criterion = torch.nn.CrossEntropyLoss()
    best_worst_acc = 0
    best_worst_acc_psu = 0
    best_avg_acc = 0
    best_unbiased_pseudo = 0
    get_best = False
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1.0e-4
    )
    use_amp = bool(getattr(args, "amp", False)) and torch.cuda.is_available()
    scaler = GradScaler("cuda", enabled=use_amp)
    # set the learning rate scheduler
    if args.scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma
        )
    elif args.scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.num_epochs, eta_min=1e-6
        )
    else:
        lr_scheduler = None
    start_epoch, best_metrics = maybe_load_training_state(
        getattr(args, "resume_ckpt", ""),
        model,
        optimizer,
        lr_scheduler,
        scaler,
        getattr(args, "resume_model_ckpt", ""),
        getattr(args, "resume_start_epoch", 0),
        {
            "best_worst_acc": getattr(args, "resume_best_worst_acc", 0),
            "best_worst_acc_psu": getattr(args, "resume_best_worst_acc_psu", 0),
            "best_avg_acc": getattr(args, "resume_best_avg_acc", 0),
            "best_unbiased_pseudo": getattr(args, "resume_best_unbiased_pseudo", 0),
        },
        logger,
    )
    best_worst_acc = best_metrics["best_worst_acc"]
    best_worst_acc_psu = best_metrics["best_worst_acc_psu"]
    best_avg_acc = best_metrics["best_avg_acc"]
    best_unbiased_pseudo = best_metrics["best_unbiased_pseudo"]
    meta_num_workers = getattr(args, "meta_num_workers", 4)
    if args.random_sampler:
        sampler = RandomTaskSampler(
            train_loader.dataset,
            args.num_supp,
            args.num_query,
            args.num_episode,
            args.task_num,
        )
        metatrain_loader = DataLoader(
            train_loader.dataset,
            batch_sampler=sampler,
            pin_memory=True,
            num_workers=meta_num_workers,
        )
        loader_func = lambda x, y, z, w: metatrain_loader
    elif args.use_group_label:
        sampler = GroupTaskSampler(
            train_loader.dataset, args.num_supp, args.num_query, args.num_episode
        )
        metatrain_loader = DataLoader(
            train_loader.dataset,
            batch_sampler=sampler,
            pin_memory=True,
            num_workers=meta_num_workers,
        )
        loader_func = lambda x, y, z, w: metatrain_loader
    else:

        def loader_func(model, dataset, idx_loader, score_func):
            class_correlated_feas = get_correlated_features(
                model, idx_loader, score_func, pred_loader=_val_idx_loader
            )
            sampler = AttributesTaskSampler(
                dataset,
                args.num_supp,
                args.num_query,
                args.num_episode,
                args.task_num,
                args.topk,
                class_correlated_feas,
            )
            metatrain_loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                pin_memory=True,
                num_workers=meta_num_workers,
            )
            return metatrain_loader

    records = {}
    probs = None
    tolerance = 0
    writer = SummaryWriter(log_dir=args.tensorboard_path)
    for epoch in range(start_epoch, args.num_epochs):
        weight_epoch = []
        ave_meters = {
            k: utils.AverageMeter()
            for k in ["loss_all", "acc_cls", "loss_cls", "acc_rep", "loss_rep"]
        }
        model.train()
        for all_data in tqdm(
            loader_func(model, train_loader.dataset, idx_train_loader, args.score_func),
            leave=False,
        ):
            data, y, g, p, g_psu = all_data

            y = y.cuda(non_blocking=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits, embeds = forward_in_chunks(
                    model, data, getattr(args, "meta_chunk_size", 0)
                )

                cls_logits = logits * args.temp
                cls_acc = (torch.argmax(cls_logits, dim=-1) == y).sum() / len(y)
                cls_losses = criterion(cls_logits, y)

                embeds = embeds.reshape(
                    args.task_num,
                    (args.num_supp + args.num_query) * args.n_classes,
                    -1,
                )
                support_embeds, query_embeds = (
                    embeds[:, 0 : args.num_supp * args.n_classes],
                    embeds[:, args.num_supp * args.n_classes :],
                )
                centroids_batch = support_embeds.reshape(
                    args.task_num, args.n_classes, args.num_supp, -1
                ).mean(dim=-2)

                query_labels = torch.repeat_interleave(
                    torch.arange(args.n_classes), args.num_query
                ).repeat(args.task_num)
                rep_logits = args.temp * torch.matmul(
                    F.normalize(query_embeds, dim=-1),
                    F.normalize(centroids_batch, dim=-1).permute(0, 2, 1),
                ).reshape(
                    -1, args.n_classes
                )  # (B,Q,D) (B, N, D)
                rep_acc = (
                    torch.argmax(rep_logits, dim=-1) == query_labels.cuda()
                ).sum() / len(query_labels)
                rep_losses = criterion(rep_logits, query_labels.cuda())
                loss = cls_losses + args.alpha * rep_losses

            ave_meters["loss_cls"].update(cls_losses.item())
            ave_meters["acc_cls"].update(cls_acc.item())
            ave_meters["loss_rep"].update(rep_losses.item())
            ave_meters["acc_rep"].update(rep_acc.item())
            ave_meters["loss_all"].update(loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        init_rep_model(model, idx_train_loader, args, default_use_all=False)
        if args.use_group_label:
            avg_acc, worst_acc, unbiased_acc = test_model(model, val_loader)
        else:
            avg_acc, worst_acc_psu, pseudo_unbiased = test_model_pseudo(
                model, val_loader, args.val_threshold_num
            )
            if args.dataset != "nico":
                avg_acc, worst_acc, unbiased_acc = test_model(model, val_loader)
        test_avg_acc, test_worst_acc, test_unbiased_acc = test_model(model, test_loader)

        if best_avg_acc < avg_acc:
            best_avg_acc = avg_acc
            if args.n_gpu == 1:
                torch.save(model.state_dict(), args.avg_model_path)
            else:
                torch.save(model.module.state_dict(), args.avg_model_path)

        if args.use_group_label:
            if best_worst_acc < worst_acc:
                best_worst_acc = worst_acc
                if args.n_gpu == 1:
                    torch.save(model.state_dict(), args.worst_model_path)
                else:
                    torch.save(model.module.state_dict(), args.worst_model_path)
                get_best = True
            else:
                get_best = False
        else:
            if best_worst_acc_psu < worst_acc_psu:
                best_worst_acc_psu = worst_acc_psu
                if args.n_gpu == 1:
                    torch.save(model.state_dict(), args.worst_pseudo_model_path)
                else:
                    torch.save(model.module.state_dict(), args.worst_pseudo_model_path)
            if best_unbiased_pseudo < pseudo_unbiased:
                best_unbiased_pseudo = pseudo_unbiased
                if args.n_gpu == 1:
                    torch.save(model.state_dict(), args.unbiased_pseudo_model_path)
                else:
                    torch.save(
                        model.module.state_dict(), args.unbiased_pseudo_model_path
                    )
                get_best = True
            else:
                get_best = False

            if (
                args.dataset == "waterbirds"
                or args.dataset == "celeba"
                or args.dataset.startswith("spawrious_o2o_")
                or args.dataset.startswith("metashift")
            ):
                if best_worst_acc < worst_acc:
                    best_worst_acc = worst_acc
                    if args.n_gpu == 1:
                        torch.save(model.state_dict(), args.worst_model_path)
                    else:
                        torch.save(model.module.state_dict(), args.worst_model_path)
        train_loss_all = ave_meters["loss_all"].avg
        train_loss_rep = ave_meters["loss_rep"].avg
        train_loss_cls = ave_meters["loss_cls"].avg
        train_acc_rep = ave_meters["acc_rep"].avg
        train_acc_cls = ave_meters["acc_cls"].avg

        writer.add_scalar("Loss/train_all", train_loss_all, epoch)
        writer.add_scalar("Loss/train_rep", train_loss_rep, epoch)
        writer.add_scalar("Loss/train_cls", train_loss_cls, epoch)
        writer.add_scalar("Acc/train_cls", train_acc_cls, epoch)
        writer.add_scalar("Acc/train_rep", train_acc_rep, epoch)

        elapsed_time = timer.t()
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        if args.use_group_label:
            msg = (
                f"[{epoch}] Loss:{train_loss_all:.4f} "
                f"(CLS:{train_loss_cls:.4f}, "
                f"REP:{train_loss_rep:.4f}), "
                f"CLSAcc:{train_acc_cls:.4f}, "
                f"REPAcc:{train_acc_rep:.4f}, "
                f"ValAcc: {avg_acc:.4f}, "
                f"ValWAcc: {worst_acc:.4f}, "
                f"ValUAcc: {unbiased_acc:.4f},"
                f"TestAcc: {test_avg_acc:.4f}, "
                f"TestWAcc: {test_worst_acc:.4f}, "
                f"TestUAcc: {test_unbiased_acc:.4f}"
            )
            writer.add_scalar("Acc/ValAcc", avg_acc, epoch)
            writer.add_scalar("Acc/ValWAcc", worst_acc, epoch)
            writer.add_scalar("Acc/ValUAcc", unbiased_acc, epoch)
            writer.add_scalar("Acc/TestAcc", test_avg_acc, epoch)
            writer.add_scalar("Acc/TestWAcc", test_worst_acc, epoch)
            writer.add_scalar("Acc/TestUAcc", test_unbiased_acc, epoch)
            # if get_best:
            #     msg += "(best)"
        else:
            msg = (
                f"[{epoch}] Loss:{train_loss_all:.4f} "
                f"(CLS:{train_loss_cls:.4f}, "
                f"REP:{train_loss_rep:.4f}), "
                f"CLSAcc:{train_acc_cls:.4f}, "
                f"REPAcc:{train_acc_rep:.4f}, "
                f"ValAcc: {avg_acc:.4f}, "
            )
            if args.dataset != "nico":
                msg += f"ValUAcc: {unbiased_acc:.4f}, "
                msg += f"ValWAcc: {worst_acc:.4f}, "
            msg += (
                f"ValPWAcc: {worst_acc_psu:.4f}, "
                f"ValPUAcc: {pseudo_unbiased:.4f}, "
                f"TestAcc: {test_avg_acc:.4f}, "
                f"TestWAcc: {test_worst_acc:.4f}, "
                f"TestUAcc: {test_unbiased_acc:.4f}"
            )

            writer.add_scalar("Acc/ValAcc", avg_acc, epoch)
            writer.add_scalar("Acc/ValPWAcc", worst_acc_psu, epoch)
            writer.add_scalar("Acc/ValPUAcc", pseudo_unbiased, epoch)
            writer.add_scalar("Acc/TestAcc", test_avg_acc, epoch)
            writer.add_scalar("Acc/TestWAcc", test_worst_acc, epoch)
            writer.add_scalar("Acc/TestUAcc", test_unbiased_acc, epoch)

        msg += f" ({utils.time_str(elapsed_time)}/{utils.time_str(avg_time_per_epoch*args.num_epochs)})"
        logger.debug(msg)
        writer.flush()
        if getattr(args, "last_model_path", ""):
            if args.n_gpu == 1:
                torch.save(model.state_dict(), args.last_model_path)
            else:
                torch.save(model.module.state_dict(), args.last_model_path)
        if getattr(args, "resume_ckpt", ""):
            save_training_state(
                args.resume_ckpt,
                model,
                optimizer,
                lr_scheduler,
                scaler,
                epoch,
                {
                    "best_worst_acc": best_worst_acc,
                    "best_worst_acc_psu": best_worst_acc_psu,
                    "best_avg_acc": best_avg_acc,
                    "best_unbiased_pseudo": best_unbiased_pseudo,
                },
                args,
            )
        if lr_scheduler:
            lr_scheduler.step()
    writer.close()


if __name__ == "__main__":
    args = utils.get_config()
    if args.use_val:  # choose whether to use valiation data for training
        data_tag = "train_and_val"
    else:
        data_tag = "train"
    if args.random_sampler:
        group_str = "random_sampler"
    elif args.use_group_label:
        group_str = "group_labels"
    else:
        group_str = args.vlm
    if args.pretrained:
        model_tag = "imagenet_weights"
    else:
        model_tag = "scratch"

    expr_name = f"{args.dataset}_{args.backbone}_lr_{args.lr:.6f}_{data_tag}_{args.batch_size}B_{args.topk}topK_{args.num_supp}ns_{args.num_query}nq_{args.num_epochs}Epochs_{args.num_episode}Epi_{args.task_num}T_{group_str}_{args.temp:.2f}temp_{args.alpha}alpha_{args.score_func}_{args.scheduler}_scheduler_{model_tag}_{args.tag}"
    expr_folder = os.path.join(args.save_folder, expr_name)
    os.makedirs(expr_folder, exist_ok=True)
    with open(os.path.join(expr_folder, "config.yaml"), "w") as f:
        yaml.dump(args, f)

    args.worst_model_path = os.path.join(expr_folder, "worst_model.pt")
    args.worst_pseudo_model_path = os.path.join(expr_folder, "pseudo_worst_model.pt")
    args.avg_model_path = os.path.join(expr_folder, "avg_model_embed.pt")
    args.unbiased_pseudo_model_path = os.path.join(
        expr_folder, "pseudo_unbiased_model.pt"
    )
    args.last_model_path = os.path.join(expr_folder, "last_model.pt")
    args.tensorboard_path = os.path.join(expr_folder, "tensorboard")
    args.resume_ckpt = os.path.join(expr_folder, "resume_state.pt")
    EXPR_LOG_PATH = os.path.join(expr_folder, "expr_train.log")

    gpu = ",".join([str(i) for i in utils.get_free_gpu()[0 : args.n_gpu]])
    utils.set_gpu(gpu)
    os.makedirs(
        expr_folder,
        exist_ok=True,
    )
    logger = logging.getLogger("expr")
    logger.setLevel(logging.DEBUG)
    fhandler = logging.FileHandler(EXPR_LOG_PATH)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(message)s")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)

    shandler = logging.StreamHandler()
    shandler.setFormatter(formatter)
    logger.addHandler(shandler)
    train_loader, idx_train_loader, val_loader, test_loader = get_loader(args)
    model = REPModel(args.backbone, train_loader.dataset.n_classes, args.pretrained)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    warm_ckpt = getattr(args, "init_ckpt", "")
    if warm_ckpt:
        target_model = model if args.n_gpu == 1 else model.module
        load_backbone_from_checkpoint(target_model, warm_ckpt, logger)

    model.cuda()
    target_model = model if args.n_gpu == 1 else model.module
    target_model.use_checkpoint = bool(getattr(args, "use_checkpoint", False))
    if args.n_gpu == 1:
        init_rep_model(model, idx_train_loader, args, default_use_all=False)
    else:
        init_rep_model(model.module, idx_train_loader, args, default_use_all=False)
    if not args.test:
        meta_train(model, train_loader, idx_train_loader, val_loader, test_loader, args)

    state_dict = torch.load(args.avg_model_path)
    model.load_state_dict(state_dict)
    if args.n_gpu == 1:
        init_rep_model(model, idx_train_loader, args, default_use_all=True)
    else:
        init_rep_model(model.module, idx_train_loader, args, default_use_all=True)
    test_avg_acc, test_worst_acc, test_unbiased_acc = test_model(model, test_loader)

    logger.info(
        f"[AvgModel] Avg acc: {test_avg_acc:.4f}, worst acc: {test_worst_acc:.4f}, unbiased acc:{test_unbiased_acc:.4f}"
    )
    if args.use_group_label or (
        args.dataset == "waterbirds"
        or args.dataset == "celeba"
        or args.dataset.startswith("spawrious_o2o_")
    ):
        state_dict = torch.load(args.worst_model_path)
        model.load_state_dict(state_dict)
        if args.n_gpu == 1:
            init_rep_model(model, idx_train_loader, args, default_use_all=True)
        else:
            init_rep_model(model.module, idx_train_loader, args, default_use_all=True)
        test_avg_acc, test_worst_acc, test_unbiased_acc = test_model(model, test_loader)

        logger.info(
            f"[WorstModel] Avg acc: {test_avg_acc:.4f}, worst acc: {test_worst_acc:.4f}, unbiased acc:{test_unbiased_acc:.4f}"
        )

    if not args.use_group_label:
        state_dict = torch.load(args.worst_pseudo_model_path)
        model.load_state_dict(state_dict)
        if args.n_gpu == 1:
            init_rep_model(model, idx_train_loader, args, default_use_all=True)
        else:
            init_rep_model(model.module, idx_train_loader, args, default_use_all=True)
        test_avg_acc, test_worst_acc, test_unbiased_acc = test_model(model, test_loader)

        logger.info(
            f"[PseudoWorstModel] Avg acc: {test_avg_acc:.4f}, worst acc: {test_worst_acc:.4f}, unbiased acc:{test_unbiased_acc:.4f}"
        )

        state_dict = torch.load(args.unbiased_pseudo_model_path)
        model.load_state_dict(state_dict)
        if args.n_gpu == 1:
            init_rep_model(model, idx_train_loader, args, default_use_all=True)
        else:
            init_rep_model(model.module, idx_train_loader, args, default_use_all=True)
        test_avg_acc, test_worst_acc, test_unbiased_acc = test_model(model, test_loader)

        logger.info(
            f"[PseudoUnbiasedModel] Avg acc: {test_avg_acc:.4f}, worst acc: {test_worst_acc:.4f}, unbiased acc:{test_unbiased_acc:.4f}"
        )
