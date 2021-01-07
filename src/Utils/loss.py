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
        tags = []  # contains reference tags
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))  # compute reference tags for the person
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)  # compute squared diff to reference tags

        num_tags = len(tags)
        if num_tags == 0:  # no loss at all
            return _make_input(torch.zeros(1).float()), \
                   _make_input(torch.zeros(1).float())
        elif num_tags == 1:  # no push loss because no other person is there
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


class NodeAELoss(nn.Module):

    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tags, person_label):
        """
        associative embedding loss for one image
        """
        tags = scatter_mean(pred_tags, person_label, dim=0)
        """
        pull_real = 0
        num_persons = person_label.max().item() + 1
        for i in range(num_persons):
            tmp = pred_tags[person_label == i]
            pull_real = pull_real + torch.mean((tmp - tags[i].expand_as(tmp))**2)  # compute squared diff to reference tags
        # alternative
        """
        pull = scatter_mean((pred_tags - tags[person_label])**2, person_label, dim=0).sum()

        num_tags = tags.shape[0]
        if num_tags == 0:  # no loss at all
            return torch.tensor(0.0), torch.tensor(0.0)
        elif num_tags == 1:  # no push loss because no other person is there
            return torch.tensor(0.0, device=pred_tags.device).float(), pull/num_tags

        # tags = torch.stack(tags)

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
               pull/num_tags

    def forward(self, tags, person_label, batch_index):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        batch_size = batch_index.max().cpu().item() + 1
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[batch_index==i], person_label[batch_index==i])
            pushes.append(push.to(tags.device))
            pulls.append(pull.to(tags.device))
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
                    heatmaps_pred, heatmap_labels[idx], heatmap_masks[idx]
                )
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmap_loss += heatmaps_loss.mean()  # average over batch


        edge_loss = 0.0
        for i in range(len(preds_edges)):
            edge_loss += self.classification_loss(preds_edges[i], edge_labels[i], "mean", edge_masks[i])
        edge_loss /= len(preds_edges)
        if torch.isnan(edge_loss):
            edge_loss = 0.0

        logging = {"heatmap": heatmap_loss.cpu().item(),

                   "edge": edge_loss.cpu().item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
                   }
        return heatmap_loss + edge_loss, logging


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

        loss = edge_loss + heatmap_loss + ae_loss + class_loss
        logging["loss"] = loss.cpu().item()

        return loss, logging


class TagMultiLossFactory(nn.Module):

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

        self.tag_loss = NodeAELoss(config.MODEL.HRNET.LOSS.AE_LOSS_TYPE)
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
        preds_tags, edge_labels, edge_masks = outputs["tag"], labels["edge"], masks["edge"]
        batch_index = labels["batch_index"]
        node_person = labels["person"]
        preds_nodes, node_labels, node_masks = outputs["node"], labels["node"], masks["node"]
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


        node_loss = 0.0
        for i in range(len(preds_nodes)):
            node_loss += self.node_loss(preds_nodes[i], node_labels, "mean", node_masks)
        node_loss = node_loss / len(preds_nodes)

        tag_loss = 0
        for i in range(len(preds_tags)):
            if (node_labels == 1.0).sum() > 0.0:
                push, pull = self.tag_loss(preds_tags[i][node_labels == 1.0], node_person[node_labels == 1.0],
                                           batch_index[node_labels == 1.0])
                tag_loss += push.mean() + pull.mean()

        class_loss = 0.0
        if preds_classes is not None:
            for i in range(len(preds_classes)):
                class_loss += self.class_loss(preds_classes[i], class_labels, "mean", node_labels)
            class_loss = class_loss / len(preds_classes)

        logging = {"heatmap": heatmap_loss.cpu().item(),
                   "tag_loss": ae_loss.cpu().item() if isinstance(ae_loss, torch.Tensor) else ae_loss,
                   "tag": tag_loss.cpu().item() if isinstance(tag_loss, torch.Tensor) else tag_loss,
                   "node": node_loss.cpu().item(),
                   "class_loss": class_loss.cpu().item() if isinstance(class_loss, torch.Tensor) else class_loss,
                   }
        if len(self.loss_weights) == 3:
            class_loss *= self.loss_weights[2]

        loss = self.loss_weights[0] * node_loss + tag_loss * self.loss_weights[1] + heatmap_loss + ae_loss + class_loss
        logging["loss"] = loss.cpu().item()

        return loss, logging

class PureTagMultiLossFactory(nn.Module):

    def __init__(self, config):
        super().__init__()
        # init check

        self.num_joints = config.MODEL.HRNET.NUM_JOINTS
        self.num_stages = config.MODEL.HRNET.LOSS.NUM_STAGES
        self.loss_weights = config.MODEL.LOSS.LOSS_WEIGHTS
        self.sync_tags = config.MODEL.LOSS.SYNC_TAGS
        assert len(self.loss_weights) == 1

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

        self.tag_loss = NodeAELoss(config.MODEL.HRNET.LOSS.AE_LOSS_TYPE)


    def forward(self, outputs, labels, masks, graph):

        preds_heatmaps, heatmap_labels, heatmap_masks = outputs["heatmap"], labels["heatmap"], masks["heatmap"]
        tag_labels = labels["tag"]
        preds_tags, edge_labels, edge_masks = outputs["tag"], labels["edge"], masks["edge"]
        batch_index = labels["batch_index"]
        node_person = labels["person"]
        joint_det = graph["nodes"]
        preds_nodes, node_labels, node_masks = outputs["node"], labels["node"], masks["node"]


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
                tags_pred = preds_heatmaps[idx][:, self.num_joints:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                push_loss, pull_loss = self.ae_loss[idx](
                    tags_pred, tag_labels[idx]
                )
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                ae_loss += push_loss.mean() + pull_loss.mean()

        if self.sync_tags:
            assert len(preds_tags) == 1
            heatmap_tags = preds_heatmaps[0][:, self.num_joints:]
            heatmap_tags = torch.nn.functional.interpolate(
                heatmap_tags,
                size=(preds_heatmaps[1].shape[2], preds_heatmaps[1].shape[3]),
                mode='bilinear',
                align_corners=False
            )
            heatmap_tags = heatmap_tags[batch_index, joint_det[:, 2], joint_det[:, 1], joint_det[:, 0]]
            batch_index = torch.stack([batch_index, batch_index])
            node_person = torch.stack([node_person, node_person])
            node_labels = torch.stack([node_labels, node_labels])
            preds_tags[-1] = torch.stack([preds_tags[-1], heatmap_tags])


        tag_loss = 0
        for i in range(len(preds_tags)):
            if (node_labels == 1.0).sum() > 0.0:
                push, pull = self.tag_loss(preds_tags[i][node_labels == 1.0], node_person[node_labels == 1.0],
                                           batch_index[node_labels == 1.0])
                tag_loss += push.mean() + pull.mean()

        logging = {"heatmap": heatmap_loss.cpu().item(),
                   "tag_loss": ae_loss.cpu().item() if isinstance(ae_loss, torch.Tensor) else ae_loss,
                   "tag": tag_loss.cpu().item() if isinstance(tag_loss, torch.Tensor) else tag_loss,
                   }
        tag_loss *= self.loss_weights[0]

        loss = tag_loss * self.loss_weights[0] + heatmap_loss + ae_loss
        logging["loss"] = loss.cpu().item()

        return loss, logging


class ClassMultiLossFactory(nn.Module):

    def __init__(self, config):
        super().__init__()
        # init check
        if isinstance(config.MODEL.LOSS.NAME, str):
            print("the use of names for indicating the used loss is not possible anymore."
                  " Instead, use a list of the used losses.")
            raise NotImplementedError

        self.num_joints = config.MODEL.HRNET.NUM_JOINTS
        self.num_stages = config.MODEL.HRNET.LOSS.NUM_STAGES
        # self.loss_weights = config.MODEL.LOSS.LOSS_WEIGHTS
        self.edge_weight = config.MODEL.LOSS.EDGE_WEIGHT
        self.node_weight = config.MODEL.LOSS.NODE_WEIGHT
        self.class_weight = config.MODEL.LOSS.CLASS_WEIGHT
        self.tag_weight = config.MODEL.LOSS.TAG_WEIGHT

        self.sync_tags = config.MODEL.LOSS.SYNC_TAGS
        self.sync_gt_tags = config.MODEL.LOSS.SYNC_GT_TAGS

        self.heatmaps_loss = None
        self.ae_loss = None
        self.edge_loss = None
        self.node_loss = None
        self.class_loss = None
        self.tag_loss = None

        losses_to_use = config.MODEL.LOSS.NAME
        if "heatmap" in losses_to_use:
            if config.MODEL.KP in ["hrnet", "mmpose_hrnet"]:
                self.heatmaps_loss = \
                    nn.ModuleList(
                        [
                            HeatmapLoss()
                            if with_heatmaps_loss else None
                            for with_heatmaps_loss in config.MODEL.HRNET.LOSS.WITH_HEATMAPS_LOSS
                        ]
                    )
                self.heatmaps_loss_factor = config.MODEL.HRNET.LOSS.HEATMAPS_LOSS_FACTOR

            elif config.MODEL.KP == "hourglass":
                self.heatmaps_loss = \
                    nn.ModuleList(
                        [HeatmapLoss() for _ in range(config.MODEL.HG.NSTACK)]
                    )
                self.ae_loss = [None for _ in range(config.MODEL.HG.NSTACK)]
                self.heatmaps_loss_factor = [1.0, 1.0, 1.0, 1.0]
        if "tagmap" in losses_to_use:
            assert config.MODEL.KP != "hourglass"
            # ensure that ae losses are actually set
            num_ae_loss = 0
            for with_ae_loss in config.TRAIN.WITH_AE_LOSS:
                if with_ae_loss:
                    num_ae_loss += 1
            assert num_ae_loss > 0

            self.ae_loss = \
                nn.ModuleList(
                    [
                        AELoss(config.MODEL.HRNET.LOSS.AE_LOSS_TYPE) if with_ae_loss else None
                        for with_ae_loss in config.TRAIN.WITH_AE_LOSS
                    ]
                )
            self.push_loss_factor = config.MODEL.HRNET.LOSS.PUSH_LOSS_FACTOR
            self.pull_loss_factor = config.MODEL.HRNET.LOSS.PULL_LOSS_FACTOR

        if "edge" in losses_to_use:
            if config.MODEL.LOSS.USE_FOCAL:
                self.edge_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA, logits=True)
            else:
                if config.MODEL.LOSS.EDGE_WITH_LOGITS:
                    self.edge_loss = BCELossWtihLogits(pos_weight=config.MODEL.LOSS.EDGE_BCE_POS_WEIGHT)
                else:
                    self.edge_loss = BCELoss(pos_weight=config.MODEL.LOSS.EDGE_BCE_POS_WEIGHT)
        if "node" in losses_to_use:
            if config.MODEL.LOSS.NODE_USE_FOCAL:
                self.node_loss = FocalLoss(config.MODEL.LOSS.FOCAL_ALPHA, config.MODEL.LOSS.FOCAL_GAMMA,
                                           logits=True)
            else:
                raise NotImplementedError
        if "class" in losses_to_use:
            self.class_loss = CrossEntropyLossWithLogits()
        if "tag_loss" in losses_to_use:
            self.tag_loss = NodeAELoss(config.MODEL.HRNET.LOSS.AE_LOSS_TYPE)

    def forward(self, outputs, labels, masks, graph):

        preds_heatmaps, heatmap_labels, heatmap_masks = outputs["heatmap"], labels["heatmap"], masks["heatmap"]
        tag_labels = labels["tag"]
        preds_edges, edge_labels, edge_masks = outputs["edge"], labels["edge"], masks["edge"]
        preds_nodes, node_labels, node_masks = outputs["node"], labels["node"], masks["node"]
        preds_classes, class_labels, class_masks = outputs["class"], labels["class"], masks["class"]  # classes use

        preds_tags = outputs["tag"]
        batch_index = labels["batch_index"]
        node_person = labels["person"]
        keypoints = labels["keypoints"]
        joint_det = graph["nodes"]

        heatmap_loss = 0.0
        ae_loss = 0.0
        if self.heatmaps_loss is not None:
            for idx in range(len(preds_heatmaps)):
                if self.heatmaps_loss[idx]:
                    heatmaps_pred = preds_heatmaps[idx][:, :self.num_joints]

                    heatmaps_loss = self.heatmaps_loss[idx](
                        heatmaps_pred, heatmap_labels[idx], heatmap_masks[idx]
                    )
                    heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                    heatmap_loss += heatmaps_loss.mean()  # average over batch

        if self.ae_loss is not None:
            for idx in range(len(preds_heatmaps)):
                if self.ae_loss[idx]:
                    tags_pred = preds_heatmaps[idx][:, self.num_joints:]
                    batch_size = tags_pred.size()[0]
                    tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                    push_loss, pull_loss = self.ae_loss[idx](
                        tags_pred, tag_labels[idx]
                    )
                    push_loss = push_loss * self.push_loss_factor[idx]
                    pull_loss = pull_loss * self.pull_loss_factor[idx]

                    ae_loss += push_loss.mean() + pull_loss.mean()


        node_loss = 0.0
        if self.node_loss is not None:
            for i in range(len(preds_nodes)):
                node_loss += self.node_loss(preds_nodes[i], node_labels, "mean", node_masks)
            node_loss = node_loss / len(preds_nodes)
        node_loss *= self.node_weight

        edge_loss = 0.0
        if self.edge_loss is not None:
            for i in range(len(preds_edges)):
                edge_loss += self.edge_loss(preds_edges[i], edge_labels[i], "mean", edge_masks[i])
            edge_loss = edge_loss / len(preds_edges)
            if torch.isnan(edge_loss):
                edge_loss = 0.0
        edge_loss *= self.edge_weight

        class_loss = 0.0
        if self.class_loss is not None:
            for i in range(len(preds_classes)):
                class_loss += self.class_loss(preds_classes[i], class_labels, "mean", node_labels)
            class_loss = class_loss / len(preds_classes)
        class_loss *= self.class_weight

        tag_loss = 0.0
        if self.tag_loss is not None:
            pred_tags = preds_tags[-1]
            heatmap_tags = preds_heatmaps[0][:, self.num_joints:]
            heatmap_tags = torch.nn.functional.interpolate(
                heatmap_tags,
                size=(preds_heatmaps[1].shape[2], preds_heatmaps[1].shape[3]),
                mode='bilinear',
                align_corners=False
            )
            if pred_tags is None:
                pred_tags = heatmap_tags[batch_index, joint_det[:, 2], joint_det[:, 1], joint_det[:, 0]]

            if self.sync_gt_tags:
                print(keypoints.shape)
                batch_size, num_persons, num_joints = keypoints.shape[:3]
                gt_types = torch.arange(0, num_joints, dtype=keypoints.dtype, device=keypoints.device).repeat(num_persons*batch_size).reshape(-1, 1)
                keypoints = keypoints.reshape(-1, 3)
                keypoints = torch.cat([keypoints, gt_types], dim=1).long().clamp(0, max(heatmap_tags.shape[2], heatmap_tags.shape[3])-1)
                valid_keypoints = keypoints[:, 2] > 0.0
                print(f"num_valid_kpt:{valid_keypoints.sum()}")
                if valid_keypoints.sum() > 0.0:
                    keypoints = keypoints[valid_keypoints]
                    gt_node_person = torch.arange(0, num_persons, dtype=torch.long, device=keypoints.device).repeat_interleave(num_joints).repeat(batch_size, 1).reshape(-1)
                    gt_batch_index = torch.arange(0, batch_size, dtype=torch.long, device=keypoints.device).repeat_interleave(num_persons*num_joints)
                    gt_node_labels = torch.ones_like(gt_batch_index, dtype=torch.float32, device=keypoints.device)
                    gt_node_person = gt_node_person[valid_keypoints]
                    gt_batch_index = gt_batch_index[valid_keypoints]
                    gt_node_labels = gt_node_labels[valid_keypoints]
                    batch_index = torch.cat([batch_index, gt_batch_index])
                    node_person = torch.cat([node_person, gt_node_person])
                    node_labels = torch.cat([node_labels, gt_node_labels])

                    print(f"num_valid_kpt:{valid_keypoints.sum()}")
                    print(f"heatmaptags shape:{heatmap_tags.shape}")
                    print(f"max/min :{keypoints[:, 1].max()} {keypoints[:, 1].min()}")
                    print(f"max/min x:{keypoints[:, 0].max()} {keypoints[:, 0].min()}")
                    print(f"max/min t:{keypoints[:, 3].max()} {keypoints[:, 3].min()}")

                    gt_pred_tags = heatmap_tags[gt_batch_index, keypoints[:, 3], keypoints[:, 1], keypoints[:, 0]]
                    pred_tags = torch.cat([pred_tags, gt_pred_tags])

            if self.sync_tags:
                assert len(preds_tags) == 1
                print("wtf")

                heatmap_tags = heatmap_tags[batch_index, joint_det[:, 2], joint_det[:, 1], joint_det[:, 0]]
                batch_index = torch.cat([batch_index, batch_index])
                node_person = torch.cat([node_person, node_person])
                node_labels = torch.cat([node_labels, node_labels])
                pred_tags = torch.cat([pred_tags, heatmap_tags])

            if (node_labels == 1.0).sum() > 0.0:
                push, pull = self.tag_loss(pred_tags[node_labels == 1.0], node_person[node_labels == 1.0],
                                           batch_index[node_labels == 1.0])
                tag_loss += push.mean() + pull.mean()
        tag_loss *= self.tag_weight

        loss = node_loss + edge_loss + class_loss + heatmap_loss + ae_loss + tag_loss

        logging = {"heatmap": heatmap_loss.cpu().item() if isinstance(heatmap_loss, torch.Tensor) else heatmap_loss,
                   "tag_loss": ae_loss.cpu().item() if isinstance(ae_loss, torch.Tensor) else ae_loss,
                   "edge": edge_loss.cpu().item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
                   "node": node_loss.cpu().item() if isinstance(node_loss, torch.Tensor) else node_loss,
                   "class_loss": class_loss.cpu().item() if isinstance(class_loss, torch.Tensor) else class_loss,
                   "loss": loss.cpu().item()}

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
            loss += self.classification_loss(preds_edges[i], edge_labels[i], "mean", masks[i])
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


class BCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        self.loss = nn.BCELoss(reduction="none")

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


