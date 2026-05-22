import logging
import torch
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn as nn
from models.resnet import resnet18, resnet50
import torchvision
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def get_correlated_features(model, dataloader, score_func="tanh-abs-log", pred_loader=None):
    """Calculate the spuriousness scores of the activated concepts for each class.

    Concept presence is computed from ``dataloader`` (train).  When
    ``pred_loader`` is provided (e.g. validation split), prediction
    accuracies are measured on that loader instead, which can give more
    reliable spuriousness estimates when the training split has highly
    exclusive class–context assignments.

    Args:
        model (torch.nn.Module): a prediction model.
        dataloader (torch.utils.data.DataLoader): loader for concept presence.
        score_func (str, optional): the type of the score function. Defaults to "tanh-abs-log".
        pred_loader (optional): loader for measuring prediction accuracy. Defaults to dataloader.

    Returns:
        dict[int, tuple[np.array, np.array]]: a dictionary of spuriousness scores and the indexes of active features for each class.
    """
    class_wise_data = {}
    ctx_wise_data = {}  # for mi-ratio: track which images go to which context
    model.eval()
    # Use pred_loader for accuracy measurement; dataloader for concept presence
    acc_loader = pred_loader if pred_loader is not None else dataloader
    with torch.no_grad():
        for idx, data, y, g, p, _ in tqdm(acc_loader, leave=False):
            logits = model(data.cuda())
            logits = logits.detach().cpu()
            preds = torch.argmax(logits, dim=1).numpy()
            for i in range(len(y)):
                l = y[i].item()
                if l in class_wise_data:
                    class_wise_data[l].append((idx[i].item(), int(preds[i] == l)))
                else:
                    class_wise_data[l] = [(idx[i].item(), int(preds[i] == l))]
                # Track context for mi-ratio
                ctx_id = int(p[i].item()) if hasattr(p[i], 'item') else int(p[i])
                if l not in ctx_wise_data:
                    ctx_wise_data[l] = []
                ctx_wise_data[l].append(ctx_id)
    embeddings = dataloader.dataset.dataset.embeddings  # always from train
    n_contexts = len(set(v for vals in ctx_wise_data.values() for v in vals))
    class_correlated_feas = {}

    eps = 1e-10
    for c in class_wise_data:
        count_pos = 0
        num_per_class = len(class_wise_data[c])
        counts_pos_w = np.zeros(embeddings.shape[1])
        counts_neg_w = np.zeros(embeddings.shape[1])

        counts_pos_wo = np.zeros(embeddings.shape[1])
        counts_neg_wo = np.zeros(embeddings.shape[1])
        for idx, pred_res in class_wise_data[c]:
            if pred_res == 1:
                counts_pos_w[embeddings[idx] == 1] += 1
                counts_pos_wo[embeddings[idx] != 1] += 1
                count_pos += 1
            else:
                counts_neg_w[embeddings[idx] == 1] += 1
                counts_neg_wo[embeddings[idx] != 1] += 1

        all_indexes = np.arange(embeddings.shape[1])
        active_indexes = all_indexes[(counts_pos_w + counts_neg_w) > 0]

        p_y1_w0 = counts_pos_wo[active_indexes] / (
            counts_pos_wo[active_indexes] + counts_neg_wo[active_indexes] + eps
        )
        p_y1_w1 = counts_pos_w[active_indexes] / (
            counts_pos_w[active_indexes] + counts_neg_w[active_indexes] + eps
        )

        # address the corner cases
        cond = (p_y1_w1 == 0) & (p_y1_w0 == 0)
        p_y1_w1[cond] = 1.0
        p_y1_w0[cond] = 1.0

        if score_func == "tanh-abs-log":
            scores = np.tanh(abs(np.log(p_y1_w1 / (p_y1_w0 + eps) + eps)))
        elif score_func == "tanh-log":
            scores = np.tanh(np.log(p_y1_w1 / (p_y1_w0 + eps) + eps))
        elif score_func == "abs-log":
            scores = abs(np.log(p_y1_w1 / (p_y1_w0 + eps) + eps))
        elif score_func == "log":
            scores = np.log(p_y1_w1 / (p_y1_w0 + eps) + eps)
        elif score_func == "abs-diff":
            scores = abs(p_y1_w1 - p_y1_w0)
        elif score_func == "diff":
            scores = p_y1_w1 - p_y1_w0
        elif score_func == "exp-diff":
            scores = np.exp(p_y1_w1 - p_y1_w0)
        elif score_func == "exp-abs-diff":
            scores = np.exp(abs(p_y1_w1 - p_y1_w0))
        elif score_func == "bayes":
            # Laplace smoothing + sample-size penalty
            alpha = 1.0
            n_w = counts_pos_w[active_indexes] + counts_neg_w[active_indexes]
            n_wo = counts_pos_wo[active_indexes] + counts_neg_wo[active_indexes]
            p_s_w = (counts_pos_w[active_indexes] + alpha) / (counts_pos_w[active_indexes] + counts_neg_w[active_indexes] + 2 * alpha + eps)
            p_s_wo = (counts_pos_wo[active_indexes] + alpha) / (counts_pos_wo[active_indexes] + counts_neg_wo[active_indexes] + 2 * alpha + eps)
            penalty = 1 - np.exp(-0.005 * np.minimum(n_w, n_wo))
            scores = np.abs(p_s_w - p_s_wo) * penalty
        elif score_func == "mi-ratio":
            # Context concentration: concept appearing in fewer unique contexts = spurious
            scores = np.zeros(len(active_indexes))
            for j, feat_idx in enumerate(active_indexes):
                ctx_counts = {}
                n_total = 0
                for ci in range(len(class_wise_data[c])):
                    emb_idx = class_wise_data[c][ci][0]
                    if embeddings[emb_idx, feat_idx] == 1:
                        ctx_id = ctx_wise_data[c][ci]
                        ctx_counts[ctx_id] = ctx_counts.get(ctx_id, 0) + 1
                        n_total += 1
                if n_total < 5:
                    scores[j] = 0.0; continue
                n_unique = len(ctx_counts)
                concentration = max(ctx_counts.values()) / n_total
                scores[j] = concentration / (n_unique + 1)
            scores = np.clip(scores, 0, 10)
        else:
            raise ValueError(f"Unknown score_func: {score_func}")

        class_correlated_feas[c] = (scores, active_indexes)
    model.train()
    return class_correlated_feas


class ERMModel(nn.Module):
    def __init__(self, backbone, num_classes, pretrained):
        """Initliaze an ERM model

        Args:
            backbone (str): specify the backbone of the model.
            num_classes (int): number of classes.
            pretrained (bool): whether to load the pretrained weights.
        """
        super(ERMModel, self).__init__()
        if backbone == "resnet50":
            if pretrained:
                self.backbone = resnet50()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet50_Weights.DEFAULT.get_state_dict(
                        progress=True
                    ),
                    strict=False,
                )
            else:
                self.backbone = resnet50()
        elif backbone == "resnet18":
            if pretrained:
                self.backbone = resnet18()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet18_Weights.DEFAULT.get_state_dict(
                        progress=True
                    ),
                    strict=False,
                )
            else:
                self.backbone = resnet18()
        d = self.backbone.out_dim
        self.num_classes = num_classes
        self.fea_dim = d
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x, get_fea=False):
        """Calculate the logits (and embeddings of x if get_fea=True) of the model

        Args:
            x (torch.tensor): a batch of image tensors.
            get_fea (bool, optional): choose whether to also return feature embeddings. Defaults to False.

        Returns:
            [torch.tensor, Optional[torch.tensor]]: prediction logits and embeddings of x if get_fea=True.
        """
        fea = self.backbone(x)
        logits = self.fc(fea)
        if get_fea:
            return logits, fea
        else:
            return logits


class REPModel(nn.Module):
    def __init__(self, backbone, n_classes, pretrained):
        """Initialize a prediction model that uses a centroid classifier for predictions.
        This model is designed specifically for meta0-learning.

        Args:
            backbone (str): backbone of the model.
            n_classes (int): number of classes.
            pretrained (bool): choose whether to load the pretrained weights.
        """
        super(REPModel, self).__init__()
        if backbone == "resnet50":
            if pretrained:
                self.backbone = resnet50()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet50_Weights.DEFAULT.get_state_dict(
                        progress=True
                    ),
                    strict=False,
                )
            else:
                self.backbone = resnet50()
        elif backbone == "resnet18":
            if pretrained:
                self.backbone = resnet18()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet18_Weights.DEFAULT.get_state_dict(
                        progress=True
                    ),
                    strict=False,
                )
            else:
                self.backbone = resnet18()
        d = self.backbone.out_dim
        self.n_classes = n_classes
        self.fea_dim = d
        self.use_checkpoint = False
        # self.init(idx_dataloader, False)

    def init(self, idx_dataloader, use_all=True, batch_size=64, num_workers=0):
        """Initialize the centroids of the classifier using the dataloader.
        A centroid of a class is the average feature embedding of all the samples in that class.

        Args:
            idx_dataloader (torch.utils.data.DataLoader): a dataloader that also provides the indexes of the images.
            use_all (bool, optional): Choose whether to use all the samples in the dataset. If False, directly use idx_dataloader which can be created with a subset of the original dataset. Defaults to True.
        """
        self.centroids = torch.zeros(self.n_classes, self.fea_dim)
        self.counts = {c: 0 for c in range(self.n_classes)}
        if use_all:
            dataloader = torch.utils.data.DataLoader(
                idx_dataloader.dataset,
                shuffle=False,
                batch_size=batch_size,
                pin_memory=False,
                num_workers=num_workers,
            )
        else:
            dataloader = idx_dataloader
        self.eval()
        with torch.no_grad():
            for _, x, y, _, _, _ in tqdm(
                dataloader, desc="init classifier", leave=False
            ):
                fea = self.backbone(x.cuda()).cpu()
                for i in range(len(y)):
                    self.centroids[y[i].item()] += fea[i]
                    self.counts[y[i].item()] += 1
            for c in self.counts:
                self.centroids[c] /= self.counts[c]
        self.centroids = self.centroids.cuda()

    def forward(self, x, get_fea=False):
        """Predict the logits of x using the centroid classifier.

        Args:
            x (torch.tensor): a batch of image tensors.
            get_fea (bool, optional): choose whether to return the image embeddings. Defaults to False.

        Returns:
            [torch.tensor, Optional[torch.tensor]]: prediction logits and embeddings of x if get_fea=True.
        """
        if self.training and self.use_checkpoint and x.requires_grad:
            fea = checkpoint(self.backbone, x, use_reentrant=False)
        else:
            fea = self.backbone(x)  # B, D
        logits = torch.matmul(
            F.normalize(fea, dim=-1), F.normalize(self.centroids, dim=-1).T
        )
        if get_fea:
            return logits, fea
        else:
            return logits
