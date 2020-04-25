import torch
import torch.nn as nn
import torch_geometric.utils as gutils
from torch_geometric.data import Data
from Models.MessagePassingNetwork.VanillaMPN2 import VanillaMPN2
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph
from Utils.dataset_utils import Graph
from Utils.Utils import *
import numpy as np
import matplotlib;matplotlib.use("Agg")
from Utils.ConstructGraph import NaiveGraphConstructor, graph_cluster_to_persons
import matplotlib.pyplot as plt
import cv2


def main():

    # todo use crowd mask
    imgs, masks, keypoints = torch.load("test_output/img_batch.pt"), torch.load("test_output/masks_batch.pt"), torch.load("test_output/keypoints_batch.pt")
    scoremap, features = torch.load("test_output/score_map_batch.pt"), torch.load("test_output/features_batch.pt")
    # use half of the batch
    batch_size = 1
    imgs = imgs[:batch_size]
    keypoints = keypoints[:batch_size]
    scoremap = scoremap[:batch_size]
    features = features[:batch_size]

    scoremap = scoremap[:, -1, :17]

    feature_gather = nn.AvgPool2d(7, 1, 3)
    features = feature_gather(features)

    constr = NaiveGraphConstructor(scoremap.cuda(), features.cuda(), keypoints.cuda(), mode="train")
    x, edge_attr, edge_index, edge_labels, joint_det = constr.construct_graph()

    #################################
    """
    test_graph = Graph(x=x, edge_index=edge_index, edge_attr=edge_labels*1.0, edge_labels=edge_labels)
    sol = cluster_graph(test_graph, "MUT", complete=False)
    sparse_sol, _ = gutils.dense_to_sparse(torch.from_numpy(sol))
    # construct easy solution
    #alternative_sol = edge_index[:, edge_labels == 1]

    persons_det = graph_cluster_to_persons(joint_det, sparse_sol)
    #draw_detection(imgs[0], joint_det, keypoints[0], "img.png")
    #draw_poses(imgs[0], persons_det, "img_poses.png")
    keypoints = keypoints.cpu().numpy()
    # draw_poses(imgs[0], keypoints[0], "img_poses_gt.png")


    #################################
    """
    config = {"steps": 4,
              "node_input_dim": 256,
              "edge_input_dim": 2 + 17*17,
              "node_feature_dim": 128,
              "edge_feature_dim": 128,
              "node_hidden_dim": 128,
              "edge_hidden_dim": 128,
              "aggr": "add"}
    model = VanillaMPN2(**config).cuda()
    print(f"trainable parameters: {count_parameters(model)}")

    #######################

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=3e-4)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR()
    for i in range(10000):

        optimizer.zero_grad()

        edge = model(x, edge_attr.cuda(), edge_index.cuda())
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
