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
    """"# evil_joint_position = keypoints[0, person_idx_gt, joint_idx_gt, :2][19]
    # np_score = scoremap[joint_idx_gt[19]].detach().numpy()
    # this image has 6 persons, but selecting top 30 makes more sense because then the number of joint per image is same
    batch = 0
    imgs, masks, keypoints = torch.load("imgs.pt"), torch.load("masks.pt"), torch.load("keypoints.pt")
    scoremap, features = torch.load("score.pt"), torch.load("features.pt")
    scoremap = scoremap[0, -1, :17]

    feature_gather = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
    features = feature_gather(features)

    person_idx_gt, joint_idx_gt = keypoints[batch, :, :, 2].nonzero(as_tuple=True)
    num_joints_gt = len(person_idx_gt)

    joint_map = non_maximum_suppression(scoremap, threshold=0.007)
    joint_idx_det, joint_y, joint_x = joint_map.nonzero(as_tuple=True)
    num_joints_det = len(joint_idx_det)
    print(f"gt kp: {num_joints_gt}, d kp: {num_joints_det}")

    # match joint_gt to joint_det
    # calculate distances between joint_det and joint_gt
    # these distances for a cost matrix for the matching between joint_gt and joint_det
    keypoints = keypoints[batch, :, :, :2]
    joint_positions_det = torch.stack([joint_x, joint_y], 1)
    keypoints = keypoints.view(-1, 1, 2)
    distances = torch.norm(joint_positions_det - keypoints, dim=2)
    distances = distances.view(30, 17, num_joints_det)
    # set the distances of joint pairse of different types to high cost s.t. they are not matched
    for jt in range(17):
        distances[:, jt, joint_idx_det != jt] = 1000000.0

    cost_mat = distances[person_idx_gt, joint_idx_gt].detach().numpy()
    sol = linear_sum_assignment(cost_mat)
    cost = np.sum(cost_mat[sol[0], sol[1]])
    print(f"Assignment cost: {cost}")
    # sol maps nodes to gt joints/  gt joints to nodes and connectivity maps between gt joints
    # create mapping joint_det -> joint_gt
    # because not all joint_det have a corresponding partner all other joints_dets are mapped to 0
    # this means the position of joint_gt given joint_idx_det is node_to_gt[joint_idx_det] - 1
    node_to_gt = torch.zeros(num_joints_det, dtype=torch.int) - 1
    node_to_gt[sol[1]] = torch.arange(0, num_joints_gt, dtype=torch.int)
    node_to_gt = node_to_gt.long() + 1


    # joint_locations tuple (joint type, height, width)
    # extract joint features from feature map
    joint_features = features[0, :, joint_y, joint_x]

    # construct node features
    x = joint_features.T
    # construct inital edge_features
    edge_attr_y = joint_y.unsqueeze(1) - joint_y
    edge_attr_x = joint_x.unsqueeze(1) - joint_x
    edge_attr = torch.stack([edge_attr_x, edge_attr_y], 2).view(-1, 2).float()

    # construct joint_det graph (fully connected)
    edge_index, _ = gutils.dense_to_sparse(torch.ones([num_joints_det, num_joints_det], dtype=torch.long))
    edge_index, _ = gutils.remove_self_loops(edge_index)
    edge_index = gutils.to_undirected(edge_index)

    # construct edge labels
    # idea: an edge has label 1 if source and destination node are assigned to joint_gt of the same person
    # this results in an fully connected pose graph per person
    num_edges = num_joints_det * num_joints_det - num_joints_det
    # person_idx_ext_(1/2) map joint_gt_idx to the respective person (node_to_gt maps to joint_gt_idx + 1 and
    # because person_idx_ext_() is shifted one to the right it evens out) node_to_gt maps joint_det without match to 0
    # which gets mapped by person_idx_ext_() to -1/-2 that means that comparing the persons, each joint_det
    # is mapped to, results in no edge for joint_dets without match since -1 != -2 and an edge for joint_dets
    # of same person
    person_idx_ext = torch.zeros(len(person_idx_gt) + 1) - 1
    person_idx_ext[1:] = person_idx_gt
    person_idx_ext_2 = torch.zeros(len(person_idx_gt) + 1) - 2
    person_idx_ext_2[1:] = person_idx_gt
    person_1 = person_idx_ext[node_to_gt[edge_index[0]]]
    person_2 = person_idx_ext_2[node_to_gt[edge_index[1]]]
    edge_label = torch.where(torch.eq(person_1, person_2), torch.ones(num_edges), torch.zeros(num_edges))
    """

    # todo use crowd mask
    imgs, masks, keypoints = torch.load("imgs.pt"), torch.load("masks.pt"), torch.load("keypoints.pt")
    scoremap, features, early_features = torch.load("score.pt"), torch.load("features.pt"), torch.load(
        "early_features.pt")
    scoremap = scoremap[:, -1, :17]

    # feature_gather = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
    feature_gather = nn.AvgPool2d(7, 1, 3)
    features = feature_gather(features)
    t = features.cpu().numpy()

    constr = NaiveGraphConstructor(scoremap, features, keypoints)
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
    model = VanillaMPN(3, 64, 64).cuda()
    print(f"trainable parameters: {count_parameters(model)}")

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels)

    #######################

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=3e-5)
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
        accuracy = gutils.accuracy(result, edge_labels)

        if i % 10 == 0:
            print(f"Iter: {i} loss {loss.item():6f} "
                  f"accuracy: {accuracy},"
                  f"avg_grad_norm: {np.mean(grad_norms)}")

        if accuracy > 0.99999:
            break
    print(f"Iter: {i} loss {loss.item()} "
          f"accuracy: {accuracy}, "
          f"precision: {gutils.precision(result, edge_labels, 1)} "
          f"recall: {gutils.recall(result, edge_labels, 1)} "
          f"minimal_acc: {gutils.accuracy(torch.zeros_like(edge_labels), edge_labels)}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()
