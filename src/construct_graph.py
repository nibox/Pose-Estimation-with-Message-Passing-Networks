import torch
import torch.nn as nn
import torch_geometric.utils as gutils
from torch_geometric.data import Data
from Models.MessagePassingNetwork.VanillaMPN import VanillaMPN
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph
from Utils.dataset_utils import Graph
from Utils.Utils import *
import numpy as np
import matplotlib;matplotlib.use("Agg")
from Utils.ConstructGraph import NaiveGraphConstructor, graph_cluster_to_persons
import matplotlib.pyplot as plt
import cv2


def draw_detection(img: torch.tensor, joint_det: torch.tensor, joint_gt: torch.tensor, fname=None):
    """
    :param img: torcg.tensor. image
    :param joint_det: shape: (num_joints, 2) list of xy positions of detected joints (without classes or clustering)
    :param fname: optional - file name of image to save. If None the image is show with plt.show
    :return:
    """

    fig = plt.figure()
    img = img.detach().cpu().numpy().copy()
    joint_det = joint_det.detach().cpu().numpy()
    joint_gt = joint_gt.detach().cpu().numpy()

    for i in range(len(joint_det)):
        # scale up to 512 x 512
        scale = 512.0 / 128.0
        x, y = int(joint_det[i, 0] * scale), int(joint_det[i, 1] * scale)
        type = joint_det[i, 2]
        if type != -1:
            cv2.circle(img, (x, y), 3, (0., 1., 0.), -1)
    for person in range(len(joint_gt)):
        if np.sum(joint_gt[person]) > 0.0:
            for i in range(len(joint_gt[person])):
                # scale up to 512 x 512
                scale = 512.0 / 128.0
                x, y = int(joint_gt[person, i, 0] * scale), int(joint_gt[person, i, 1] * scale)
                type = i
                if type != -1:
                    cv2.circle(img, (x, y), 3, (0., 0., 1.), -1)

    plt.imshow(img)
    if fname is not None:
        plt.savefig(fig=fig, fname=fname)
    else:
        raise NotImplementedError


def draw_poses(img: [torch.tensor, np.array], persons, fname=None):
    """

    :param img:
    :param persons: (N,17,3) array containing the person in the image img. Detected or ground truth does not matter.
    :param fname: If not none an image will be saved under this name , otherwise the image will be displayed
    :return:
    """
    img = to_numpy(img)
    assert img.shape[0] == 512
    assert img.shape[1] == 512
    assert len(persons.shape) == 3
    pair_ref = [
        [1, 2], [2, 3], [1, 3],
        [6, 8], [8, 10], [12, 14], [14, 16],
        [7, 9], [9, 11], [13, 15], [15, 17],
        [6, 7], [12, 13], [6, 12], [7, 13]
    ]
    bones = np.array(pair_ref) - 1
    colors = np.arange(0, 179, np.ceil(179/len(persons)))
    # image to 8bit hsv (i dont know what hsv values opencv expects in 32bit case=)
    img = img * 255.0
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    assert len(colors) == len(persons)
    for person in range(len(persons)):
        scale = 512.0 / 128.0
        color = (colors[person], 255., 255)
        valid_joints = persons[person, :, 2] > 0
        center_joint = np.mean(persons[person, valid_joints] * scale, axis=0).astype(np.int)
        for i in range(len(bones)):
            # scale up to 512 x 512
            joint_1 = persons[person, bones[i, 0]]
            joint_2 = persons[person, bones[i, 1]]
            joint_1_valid = joint_1[2] > 0
            joint_2_valid = joint_2[2] > 0
            x_1, y_1 = np.multiply(joint_1[:2], scale).astype(np.int)
            x_2, y_2 = np.multiply(joint_2[:2], scale).astype(np.int)
            if joint_1_valid:
                cv2.circle(img, (x_1, y_1), 3, color, -1)
                cv2.line(img, (x_1, y_1), (center_joint[0], center_joint[1]), color)
            if joint_2_valid:
                cv2.circle(img, (x_2, y_2), 3, color, -1)
                cv2.line(img, (x_2, y_2), (center_joint[0], center_joint[1]), color)
            """if joint_1[2] > 0 and joint_2[2] > 0:
                cv2.line(img, (x_1, y_1), (x_2, y_2), (0., 0., 1.)) """

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    fig = plt.figure()
    plt.imshow(img)
    if fname is not None:
        plt.savefig(fig=fig, fname=fname)
    else:
        raise NotImplementedError




def main():

    # todo use crowd mask
    imgs, masks, keypoints = torch.load("test_output/imgs.pt"), torch.load("test_output/masks.pt"), torch.load("test_output/keypoints.pt")
    scoremap, features, early_features = torch.load("test_output/score.pt"), torch.load("test_output/features.pt"), torch.load(
        "test_output/early_features.pt")
    scoremap = scoremap[:, -1, :17]

    # feature_gather = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
    feature_gather = nn.AvgPool2d(7, 1, 3)
    features = feature_gather(features)
    t = features.cpu().numpy()

    constr = NaiveGraphConstructor(scoremap.cuda(), features.cuda(), keypoints.cuda())
    x, edge_attr, edge_index, edge_labels, joint_det = constr.construct_graph()

    #################################
    test_graph = Graph(x=x, edge_index=edge_index, edge_attr=edge_labels*1.0, edge_labels=edge_labels)
    sol = cluster_graph(test_graph, "MUT", complete=False)
    sparse_sol, _ = gutils.dense_to_sparse(torch.from_numpy(sol))
    # construct easy solution
    #alternative_sol = edge_index[:, edge_labels == 1]

    persons_det = graph_cluster_to_persons(joint_det, sparse_sol)
    draw_detection(imgs[0], joint_det, keypoints[0], "img.png")
    draw_poses(imgs[0], persons_det, "img_poses.png")
    keypoints = keypoints.cpu().numpy()
    draw_poses(imgs[0], keypoints[0], "img_poses_gt.png")


    #################################
    # model = MPN(128, 2)
    config = {"steps": 4,
              "node_input_dim": 256,
              "edge_input_dim": 2 + 17*17,
              "node_feature_dim": 128,
              "edge_feature_dim": 128,
              "node_hidden_dim": 256,
              "edge_hidden_dim": 512}
    model = VanillaMPN(**config).cuda()
    print(f"trainable parameters: {count_parameters(model)}")

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels)

    #######################

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=3e-5)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR()
    loss = nn.BCEWithLogitsLoss()
    for i in range(10000):

        optimizer.zero_grad()

        edge = model(data.x.cuda(), data.edge_attr.cuda(), data.edge_index.cuda())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(edge.squeeze(), data.edge_labels.cuda(),
                                                                    pos_weight=torch.tensor(98.0).cuda())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
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
