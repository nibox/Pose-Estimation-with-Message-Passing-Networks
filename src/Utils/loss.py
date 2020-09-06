# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Nikita Kister
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch import nn as nn
from torch.nn import functional as F

from torch_scatter import scatter_mean
from torch_geometric.utils import subgraph


class HeatmapLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2) * mask[:, None, :, :].expand_as(pred)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        # loss = loss.mean(dim=3).mean(dim=2).sum(dim=1)
        return loss


class MultiLossFactory(nn.Module):

    def __init__(self, config):
        super().__init__()
        # init check

        self.num_joints = config.MODEL.HRNET.NUM_JOINTS
        self.num_stages = config.MODEL.HRNET.LOSS.NUM_STAGES

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in config.MODEL.HRNET.LOSS.WITH_HEATMAPS_LOSS
                ]
            )
        self.heatmaps_loss_factor = config.MODEL.HRNET.LOSS.HEATMAPS_LOSS_FACTOR
        if config.MODEL.LOSS.USE_FOCAL:
            self.classification_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA, logits=True)
        else:
            raise NotImplementedError
        # removed ae loss


    def forward(self, outputs, labels, masks):

        preds_heatmaps, heatmap_labels, heatmap_masks = outputs["heatmap"], labels["heatmap"], masks["heatmap"]
        preds_edges, edge_labels, edge_masks = outputs["edge"], labels["edge"], masks["edge"]

        heatmap_loss = 0.0
        for idx in range(len(preds_heatmaps)):
            if self.heatmaps_loss[idx]:
                heatmaps_pred = preds_heatmaps[idx][:, :self.num_joints]

                heatmaps_loss = self.heatmaps_loss[idx](
                    heatmaps_pred, heatmap_labels[idx], masks[idx]
                )
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmap_loss += heatmaps_loss.mean()  # average over batch

        edge_class_loss = 0.0
        for i in range(len(preds_edges)):
            edge_class_loss += self.classification_loss(preds_edges[i], edge_labels, "mean", edge_masks)

        logging = {"heatmap": heatmap_loss.cpu().item(),
                   "edge": edge_class_loss.cpu().item()}
        return heatmap_loss + edge_class_loss, logging


class ClassMultiLossFactory(nn.Module):

    def __init__(self, config):
        super().__init__()
        # init check

        self.num_joints = config.MODEL.HRNET.NUM_JOINTS
        self.num_stages = config.MODEL.HRNET.LOSS.NUM_STAGES
        self.loss_weights = config.MODEL.LOSS.LOSS_WEIGHTS
        assert len(self.loss_weights) in [2,3]

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in config.MODEL.HRNET.LOSS.WITH_HEATMAPS_LOSS
                ]
            )
        self.heatmaps_loss_factor = config.MODEL.HRNET.LOSS.HEATMAPS_LOSS_FACTOR
        if config.MODEL.LOSS.USE_FOCAL:
            self.edge_loss= FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA, logits=True)
        else:
            raise NotImplementedError
        if config.MODEL.LOSS.NODE_USE_FOCAL:
            self.node_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA,
                                       logits=True)
        else:
            raise NotImplementedError
        self.class_loss = CrossEntropyLossWithLogits()
        # removed ae loss


    def forward(self, outputs, labels, masks):

        preds_heatmaps, heatmap_labels, heatmap_masks = outputs["heatmap"], labels["heatmap"], masks["heatmap"]
        preds_edges, edge_labels, edge_masks = outputs["edge"], labels["edge"], masks["edge"]
        preds_nodes, node_labels, node_masks = outputs["node"], labels["node"], masks["node"]
        preds_classes, class_labels, class_masks = outputs["class"], labels["class"], labels["node"]  # classes use

        heatmap_loss = 0.0
        for idx in range(len(preds_heatmaps)):
            if self.heatmaps_loss[idx]:
                heatmaps_pred = preds_heatmaps[idx][:, :self.num_joints]

                heatmaps_loss = self.heatmaps_loss[idx](
                    heatmaps_pred, heatmap_labels[idx], heatmap_masks[idx]
                )
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmap_loss += heatmaps_loss.mean()  # average over batch

        node_loss = 0.0
        for i in range(len(preds_nodes)):
            node_loss += self.node_loss(preds_nodes[i], node_labels, "mean", node_masks)
        node_loss = node_loss / len(preds_nodes)

        edge_loss = 0.0
        for i in range(len(preds_edges)):
            if preds_edges[i] is None:
                continue
            edge_loss += self.edge_loss(preds_edges[i], edge_labels[i], "mean", edge_masks[i])
        edge_loss = edge_loss / len(preds_edges)
        if torch.isnan(edge_loss):
            edge_loss = 0.0

        class_loss = 0.0
        if preds_classes is not None:
            for i in range(len(preds_classes)):
                class_loss += self.class_loss(preds_classes[i], class_labels, "mean", node_labels)
            class_loss = class_loss / len(preds_classes)

        logging = {"heatmap": heatmap_loss.cpu().item(),
                   "edge": edge_loss.cpu().item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
                   "node": node_loss.cpu().item(),
                   "class_loss": class_loss.cpu().item() if isinstance(class_loss, torch.Tensor) else class_loss}
        if len(self.loss_weights) == 3:
            class_loss *= self.loss_weights[2]

        return self.loss_weights[0] * node_loss + edge_loss * self.loss_weights[1] + heatmap_loss + class_loss, logging


class MPNLossFactory(nn.Module):

    def __init__(self, config):
        super().__init__()

        if config.MODEL.LOSS.USE_FOCAL:
            self.classification_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA,
                                                 logits=True)
        else:
            raise NotImplementedError

    def forward(self, outputs, labels, masks):

        preds_edges = outputs["edge"]
        edge_labels = labels["edge"]
        masks = masks["edge"]

        loss = 0.0
        for i in range(len(preds_edges)):
            loss += self.classification_loss(preds_edges[i], edge_labels, "mean", masks)
        loss = loss / len(preds_edges)

        return loss, {}

class ClassMPNLossFactory(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.loss_weights = config.MODEL.LOSS.LOSS_WEIGHTS
        assert len(self.loss_weights) in [2, 3]

        if config.MODEL.LOSS.USE_FOCAL:
            self.edge_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA,
                                       logits=True)
        else:
            self.edge_loss = BCELossWtihLogits(pos_weight=1.0)

        if config.MODEL.LOSS.NODE_USE_FOCAL:
            self.node_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA,
                                       logits=True)
        else:
            self.node_loss = BCELossWtihLogits(pos_weight=config.MODEL.LOSS.NODE_BCE_POS_WEIGHT)

        self.class_loss = CrossEntropyLossWithLogits()

    def forward(self, outputs, labels, masks):

        preds_edges, edge_labels, edge_masks = outputs["edge"], labels["edge"], masks["edge"]
        print(edge_labels)
        preds_nodes, node_labels, node_masks = outputs["node"], labels["node"], masks["node"]
        preds_classes, class_labels, class_masks = outputs["class"], labels["class"], labels["node"]  # classes use

        node_loss = 0.0
        for i in range(len(preds_nodes)):
            node_loss += self.node_loss(preds_nodes[i], node_labels, "mean", node_masks)
        node_loss = node_loss / len(preds_nodes)

        edge_loss = 0.0
        for i in range(len(preds_edges)):
            if preds_edges[i] is None:
                continue
            print(type(edge_labels[i]))
            edge_loss += self.edge_loss(preds_edges[i], edge_labels[i], "mean", edge_masks[i])
        edge_loss = edge_loss / len(preds_edges)
        if torch.isnan(edge_loss):
            edge_loss = 0.0

        class_loss = 0.0
        if preds_classes is not None:
            for i in range(len(preds_classes)):
                class_loss += self.class_loss(preds_classes[i], class_labels, "mean", node_labels)
            class_loss = class_loss / len(preds_classes)

        logging = {"edge": edge_loss.cpu().item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
                   "node": node_loss.cpu().item(),
                   "class_loss": class_loss.cpu().item() if isinstance(class_loss, torch.Tensor) else class_loss}
        if len(self.loss_weights) == 3:
            class_loss *= self.loss_weights[2]

        return self.loss_weights[0] * node_loss + edge_loss * self.loss_weights[1] + class_loss, logging


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets, reduction, mask=None, batch_index=None):
        assert batch_index is None
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if mask is not None:
            F_loss = F_loss * mask

        if reduction is not None:
            if reduction == "mean" and mask is None:
                return torch.mean(F_loss)
            elif reduction == "mean" and mask is not None:
                return torch.sum(F_loss) / mask.sum()
            elif reduction == "sum":
                return torch.sum(F_loss)
        else:
            return F_loss

class BCELossWtihLogits(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets, reduction, mask=None, batch_index=None):
        bce_loss = self.loss(inputs, targets)
        if mask is not None:
            bce_loss = bce_loss * mask
        if self.pos_weight is not None:
            bce_loss[targets==1.0] *= self.pos_weight
        return bce_loss.mean()

class CrossEntropyLossWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets, reduction, mask=None):
        assert reduction == "mean"
        ce_loss = self.loss(inputs, targets)
        if mask is not None:
            ce_loss = ce_loss * mask
        return ce_loss.mean()


