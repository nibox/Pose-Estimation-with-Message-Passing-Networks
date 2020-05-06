import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as gutils
from torch_geometric.data import Data
from Models.MessagePassingNetwork.VanillaMPN2 import VanillaMPN2
from Models.MessagePassingNetwork.VanillaMPN import VanillaMPN
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph
from Utils.dataset_utils import Graph
from Utils.Utils import *
import numpy as np
import matplotlib;

matplotlib.use("Agg")
from Utils.ConstructGraph import NaiveGraphConstructor, graph_cluster_to_persons
import matplotlib.pyplot as plt
import cv2


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def test_construct_labels():

    # test graph
    joint_gt = np.zeros([30, 17, 3])
    person_1 = np.zeros([17, 3])
    person_1[5] = 4, 4, 2
    person_1[6] = 8, 5, 2
    person_1[7] = 7, 6, 2
    person_2 = np.zeros([17, 3])
    person_2[5] = 4, 1, 2
    person_2[7] = 8, 1, 2
    joint_gt[0] = person_1
    joint_gt[1] = person_2
    joint_gt = torch.from_numpy(joint_gt)

    joint_det = torch.tensor([[4, 4, 5], [8, 5, 6], [7, 6, 7],
                              [4, 1, 5], [8, 1, 7],
                              [4, 2, 5], [4, 7, 5]])
    # detected joint is 5 (6ths)
    #
    edge_index = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 0, 1, 2, 3, 4, 5],
                               [1, 2, 3, 4, 5, 2, 3, 4, 5, 0, 3, 4, 5, 1, 0, 4, 5, 1, 2, 0, 5, 1, 2, 3, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6]])
    constr = NaiveGraphConstructor(torch.zeros([1, 17, 128, 128]), torch.zeros(1, 128, 128, 17), torch.zeros(1, 17, 3), device=edge_index.device, use_neighbours = True)
    person_idx_gt, joint_idx_gt = joint_gt[:, :, 2].nonzero(as_tuple=True)
    joints_gt_loc = joint_gt[person_idx_gt, joint_idx_gt, :2].round().long().clamp(0, 127)
    joints_gt_loc = torch.cat([joints_gt_loc, joint_idx_gt.unsqueeze(1)], 1)

    constr.remove_ambigiuous_det(joint_det, joints_gt_loc)
    edge_labels = constr._construct_edge_labels(joint_det, joint_gt, edge_index)
    print(edge_index[0, edge_labels==1])
    print(edge_index[1, edge_labels == 1])


def main():
    test_construct_labels()
    batch_size = 1
    if batch_size == 1:
        imgs, masks, keypoints = torch.load("test_output/imgs.pt"), torch.load("test_output/masks.pt"), torch.load(
            "test_output/keypoints.pt")
        scoremap, features = torch.load("test_output/score.pt"), torch.load("test_output/features.pt")
    else:
        imgs, masks, keypoints = torch.load("test_output/img_batch.pt"), torch.load(
            "test_output/masks_batch.pt"), torch.load("test_output/keypoints_batch.pt")
        scoremap, features = torch.load("test_output/score_map_batch.pt"), torch.load("test_output/features_batch.pt")
        # use half of the batch
        imgs = imgs[:batch_size]
        masks = masks[:batch_size]
        keypoints = keypoints[:batch_size]
        scoremap = scoremap[:batch_size]
        features = features[:batch_size]

    scoremap = scoremap[:, -1, :17]

    feature_gather = nn.AvgPool2d(3, 1, 1)
    features = feature_gather(features)

    constr = NaiveGraphConstructor(scoremap.cuda(), features.cuda(), keypoints.cuda(), masks, use_gt=True,
                                   no_false_positives=False, use_neighbours=True, device=torch.device("cuda"),
                                   edge_label_method=2,
                                   mask_crowds=False)
    x, edge_attr, edge_index, edge_labels, joint_det = constr.construct_graph()
    if batch_size == 1:
        print(f"Num detection: {len(x)}")
        print(f"Num edges : {len(edge_index[0])}")
        print(f"Num active edges: {(edge_labels==1).sum()}")

    config_2 = {"steps": 4,
                "node_input_dim": 256,
                "edge_input_dim": 2 + 17 * 17,
                "node_feature_dim": 128,
                "edge_feature_dim": 128,
                "node_hidden_dim": 128,
                "edge_hidden_dim":128,
                "aggr": "add"}
    config_1 = {"steps": 8,
                "node_input_dim": 256,
                "edge_input_dim": 2 + 17 * 17,
                "node_feature_dim": 32,
                "edge_feature_dim": 32,
                }
    use_focal_loss = True
    focal_loss = FocalLoss(logits=True)
    model = VanillaMPN2(**config_2).cuda()
    print(f"trainable parameters: {count_parameters(model)}")

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=3e-4)
    for i in range(10000):

        optimizer.zero_grad()

        edge = model(x, edge_attr.cuda(), edge_index.cuda())
        if use_focal_loss:
            loss = focal_loss(edge.squeeze(), edge_labels.cuda())
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(edge.squeeze(), edge_labels.cuda(),
                                                                        pos_weight=torch.tensor(78.0).cuda())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        grad_norms = []
        for p in list(filter(lambda p: p.grad is not None, model.parameters())):
            grad_norms.append(p.grad.data.norm(2).item())
        optimizer.step()
        result = edge.cpu().sigmoid().squeeze()
        result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))
        edge_labels[edge_labels<0.9] = 0.0
        accuracy = gutils.accuracy(result, edge_labels.cpu())

        if i % 10 == 0:
            print(f"Iter: {i} loss {loss.item():6f} "
                  f"accuracy: {accuracy},"
                  f"avg_grad_norm: {np.mean(grad_norms)}")

        if accuracy > 0.99999:
            break
    print(f"Iter: {i} loss {loss.item()} "
          f"accuracy: {accuracy}, "
          f"precision: {gutils.precision(result, edge_labels.cpu(), 1)} "
          f"recall: {gutils.recall(result, edge_labels.cpu(), 1)} "
          f"minimal_acc: {gutils.accuracy(torch.zeros_like(edge_labels.cpu()), edge_labels.cpu())}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()
