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
        self._init_check(config)

        self.num_joints = config.MODEL.NUM_JOINTS
        self.num_stages = config.LOSS.NUM_STAGES

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in config.LOSS.WITH_HEATMAPS_LOSS
                ]
            )
        self.heatmaps_loss_factor = config.LOSS.HEATMAPS_LOSS_FACTOR
        if config.MODEL.LOSS.USE_FOCAL:
            self.classification_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA, logits=True)
        else:
            raise NotImplementedError
        # removed ae loss


    def forward(self, outputs_det, outputs_class, heatmaps, edge_labels, masks, label_mask):

        heatmaps_losses = []
        for idx in range(len(outputs_det)):
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs_det[idx][:, :self.num_joints]

                heatmaps_loss = self.heatmaps_loss[idx](
                    heatmaps_pred, heatmaps[idx], masks[idx]
                )
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)

        edge_class_losses = []
        for i in range(len(outputs_class)):
            loss = self.classification_loss(outputs_class[i], edge_labels, "mean", label_mask)
            edge_class_losses.append(loss)

        return heatmaps_losses, edge_class_losses

class MPNLossFactory(nn.Module):

    def __init__(self, config):
        super().__init__()

        if config.MODEL.LOSS.USE_FOCAL:
            self.classification_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA,
                                                 logits=True)
        else:
            raise NotImplementedError

    def forward(self, outputs_class, edge_labels, label_mask):



        loss = 0.0
        for i in range(len(outputs_class)):
            loss += self.classification_loss(outputs_class[i], edge_labels, "mean", label_mask)
        loss = loss / len(outputs_class)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets, reduction, mask=None, batch_index=None):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if mask is not None:
            F_loss = F_loss * mask

        if reduction is not None:
            if batch_index is not None and reduction == "mean":
                F_loss = scatter_mean(F_loss, batch_index)
                assert len(F_loss) == batch_index.max() + 1
                return torch.mean(F_loss)
            elif reduction == "mean" and batch_index is None:
                return torch.mean(F_loss)
            elif reduction == "sum":
                return torch.sum(F_loss)
        else:
            return F_loss