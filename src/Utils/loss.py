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

def _make_input(t, requires_grad=False, need_cuda=True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    if need_cuda:
        inp = inp.cuda()
    return inp

class AELoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """
        associative embedding loss for one image
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return _make_input(torch.zeros(1).float()), \
                   _make_input(torch.zeros(1).float())
        elif num_tags == 1:
            return _make_input(torch.zeros(1).float()), \
                   pull/(num_tags)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unkown ae loss type')

        return push/((num_tags - 1) * num_tags) * 0.5, \
               pull/(num_tags)

    def forward(self, tags, joints):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


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


class BackgroundClassMultiLossFactory(nn.Module):

    def __init__(self, config):
        super().__init__()
        # init check

        self.num_joints = config.MODEL.HRNET.NUM_JOINTS
        self.num_stages = config.MODEL.HRNET.LOSS.NUM_STAGES
        self.loss_weights = config.MODEL.LOSS.LOSS_WEIGHTS
        assert len(self.loss_weights) in [2]

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in config.MODEL.HRNET.LOSS.WITH_HEATMAPS_LOSS
                ]
            )
        self.heatmaps_loss_factor = config.MODEL.HRNET.LOSS.HEATMAPS_LOSS_FACTOR

        self.ae_loss = \
            nn.ModuleList(
                [
                    AELoss(config.MODEL.HRNET.LOSS.AE_LOSS_TYPE) if with_ae_loss else None
                    for with_ae_loss in config.TRAIN.WITH_AE_LOSS
                ]
            )
        self.push_loss_factor = config.MODEL.HRNET.LOSS.PUSH_LOSS_FACTOR
        self.pull_loss_factor = config.MODEL.HRNET.LOSS.PULL_LOSS_FACTOR

        if config.MODEL.LOSS.USE_FOCAL:
            self.edge_loss= FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA, logits=True)
        else:
            raise NotImplementedError
        self.class_loss = CrossEntropyLossWithLogits()
        self.reg_loss = nn.SmoothL1Loss()
        self.score_loss = nn.BCELoss()
        # removed ae loss


    def forward(self, outputs, labels, masks):

        preds_heatmaps, heatmap_labels, heatmap_masks = outputs["heatmap"], labels["heatmap"], masks["heatmap"]
        tag_labels = labels["tag"]
        preds_edges, edge_labels, edge_masks = outputs["edge"], labels["edge"], masks["edge"]
        preds_classes, class_labels, class_masks = outputs["class"], labels["class"], masks["class"]  # classes use

        heatmap_loss = 0.0
        ae_loss = 0.0
        for idx in range(len(preds_heatmaps)):
            if self.heatmaps_loss[idx]:
                heatmaps_pred = preds_heatmaps[idx][:, :self.num_joints]

                heatmaps_loss = self.heatmaps_loss[idx](
                    heatmaps_pred, heatmap_labels[idx], heatmap_masks[idx]
                )
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmap_loss += heatmaps_loss.mean()  # average over batch

            if self.ae_loss[idx]:
                tags_pred = preds_heatmaps[idx][:, 17:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                push_loss, pull_loss = self.ae_loss[idx](
                    tags_pred, tag_labels[idx]
                )
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                ae_loss += push_loss.mean() + pull_loss.mean()

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
                class_loss += self.class_loss(preds_classes[i], class_labels, "mean", class_masks)
            class_loss = class_loss / len(preds_classes)

        logging = {"heatmap": heatmap_loss.cpu().item(),
                   "tag_loss": ae_loss.cpu().item() if isinstance(ae_loss, torch.Tensor) else ae_loss,
                   "edge": edge_loss.cpu().item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
                   "node": None,
                   "class_loss": class_loss.cpu().item() if isinstance(class_loss, torch.Tensor) else class_loss,
                   }

        class_loss *= self.loss_weights[1]
        edge_loss *= self.loss_weights[0]

        loss = edge_loss  + heatmap_loss + ae_loss + class_loss
        logging["loss"] = loss.cpu().item()

        return loss, logging


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

        self.ae_loss = \
            nn.ModuleList(
                [
                    AELoss(config.MODEL.HRNET.LOSS.AE_LOSS_TYPE) if with_ae_loss else None
                    for with_ae_loss in config.TRAIN.WITH_AE_LOSS
                ]
            )
        self.push_loss_factor = config.MODEL.HRNET.LOSS.PUSH_LOSS_FACTOR
        self.pull_loss_factor = config.MODEL.HRNET.LOSS.PULL_LOSS_FACTOR

        if config.MODEL.LOSS.USE_FOCAL:
            self.edge_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA, logits=True)
        else:
            if config.MODEL.LOSS.EDGE_WITH_LOGITS:
                self.edge_loss = BCELossWtihLogits(pos_weight=config.MODEL.LOSS.EDGE_BCE_POS_WEIGHT)
            else:
                self.edge_loss = BCELoss(pos_weight=config.MODEL.LOSS.EDGE_BCE_POS_WEIGHT)
        if config.MODEL.LOSS.NODE_USE_FOCAL:
            self.node_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA,
                                       logits=True)
        else:
            raise NotImplementedError
        self.class_loss = CrossEntropyLossWithLogits()
        self.reg_loss = nn.SmoothL1Loss()
        self.score_loss = nn.BCELoss()
        # removed ae loss


    def forward(self, outputs, labels, masks):

        preds_heatmaps, heatmap_labels, heatmap_masks = outputs["heatmap"], labels["heatmap"], masks["heatmap"]
        tag_labels = labels["tag"]
        preds_edges, edge_labels, edge_masks = outputs["edge"], labels["edge"], masks["edge"]
        preds_nodes, node_labels, node_masks = outputs["node"], labels["node"], masks["node"]
        preds_classes, class_labels, class_masks = outputs["class"], labels["class"], masks["class"]  # classes use

        """
        if "refine" in outputs.keys():
            preds_pos, node_targets = outputs["refine"], labels["refine"]
        else:
            preds_pos = None
        """

        heatmap_loss = 0.0
        ae_loss = 0.0
        for idx in range(len(preds_heatmaps)):
            if self.heatmaps_loss[idx]:
                heatmaps_pred = preds_heatmaps[idx][:, :self.num_joints]

                heatmaps_loss = self.heatmaps_loss[idx](
                    heatmaps_pred, heatmap_labels[idx], heatmap_masks[idx]
                )
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmap_loss += heatmaps_loss.mean()  # average over batch

            if self.ae_loss[idx]:
                tags_pred = preds_heatmaps[idx][:, 17:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                push_loss, pull_loss = self.ae_loss[idx](
                    tags_pred, tag_labels[idx]
                )
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                ae_loss += push_loss.mean() + pull_loss.mean()


        """
        reg_l = 0.0
        if preds_pos is not None:
            node_person = preds_pos[:, 4].long()
            node_types = preds_pos[:, 2].long()
            targets_reg = node_targets[node_person, node_types, :2]
            targets_score = (node_targets[node_person, node_types, 2] != 0).float()
            positions = preds_pos[:, :2].reshape(-1)
            scores = preds_pos[:, 3]

            reg_l += self.reg_loss(positions, targets_reg.view(-1))
            if scores.sum() != 0.0:
                reg_l += self.score_loss(scores, targets_score)

            if torch.isnan(reg_l):
                reg_loss = 0.0
        """

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
                   "tag_loss": ae_loss.cpu().item() if isinstance(ae_loss, torch.Tensor) else ae_loss,
                   "edge": edge_loss.cpu().item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
                   "node": node_loss.cpu().item(),
                   "class_loss": class_loss.cpu().item() if isinstance(class_loss, torch.Tensor) else class_loss,
                   }
        if len(self.loss_weights) == 3:
            class_loss *= self.loss_weights[2]

        loss = self.loss_weights[0] * node_loss + edge_loss * self.loss_weights[1] + heatmap_loss + ae_loss + class_loss
        logging["loss"] = loss.cpu().item()

        return loss, logging

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
            if config.MODEL.LOSS.EDGE_WITH_LOGITS:
                self.edge_loss = BCELossWtihLogits(pos_weight=1.0)
            else:
                self.edge_loss = BCELoss(pos_weight=config.MODEL.LOSS.EDGE_BCE_POS_WEIGHT)

        if config.MODEL.LOSS.NODE_USE_FOCAL:
            self.node_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA,
                                       logits=True)
        else:
            self.node_loss = BCELossWtihLogits(pos_weight=config.MODEL.LOSS.NODE_BCE_POS_WEIGHT)

        self.class_loss = CrossEntropyLossWithLogits()
        self.reg_loss = nn.SmoothL1Loss()

    def forward(self, outputs, labels, masks):

        preds_edges, edge_labels, edge_masks = outputs["edge"], labels["edge"], masks["edge"]
        preds_nodes, node_labels, node_masks = outputs["node"], labels["node"], masks["node"]
        preds_classes, class_labels, class_masks = outputs["class"], labels["class"], masks["class"]  # classes use
        if "refine" in outputs.keys():
            preds_pos, node_targets = outputs["refine"], labels["refine"]
        else:
            preds_pos = None

        reg_loss = 0.0
        if preds_pos is not None:
            node_person = preds_pos[:, 4].long()
            node_types = preds_pos[:, 2].long()
            targets = node_targets[node_person, node_types, :2]
            positions = preds_pos[:, :2].reshape(-1)

            reg_loss += self.reg_loss(positions, targets.view(-1))

            if torch.isnan(reg_loss):
                reg_loss = 0.0

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

        logging = {"edge": edge_loss.cpu().item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
                   "node": node_loss.cpu().item(),
                   "class_loss": class_loss.cpu().item() if isinstance(class_loss, torch.Tensor) else class_loss,
                   "reg": reg_loss.cpu().item() if isinstance(reg_loss, torch.Tensor) else reg_loss}
        if len(self.loss_weights) == 3:
            class_loss *= self.loss_weights[2]
        loss = self.loss_weights[0] * node_loss + edge_loss * self.loss_weights[1] + reg_loss + class_loss
        logging["loss"] = loss.cpu().item()

        return loss, logging


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


