import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter_mean

from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph
from Utils.dataset_utils import Graph


def non_maximum_suppression(scoremap, threshold=0.05):

    pool = nn.MaxPool2d(3, 1, 1)
    pooled = pool(scoremap)
    maxima = torch.eq(pooled, scoremap).float()
    return maxima


def to_numpy(array: [torch.Tensor, np.array]):
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    else:
        return array


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


def draw_detection(img, joint_det, joint_gt, fname=None, output_size=128.0):
    """
    :param img: torcg.tensor. image
    :param joint_det: shape: (num_joints, 2) list of xy positions of detected joints (without classes or clustering)
    :param fname: optional - file name of image to save. If None the image is show with plt.show
    :return:
    """


    img = to_numpy(img)
    if img.dtype != np.uint8:
        img = img * 255.0
        img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    colors_joints = np.arange(0, 179, np.ceil(179 / 17), dtype=np.float)
    colors_joints[1::2] = colors_joints[-2::-2] # swap some colors to have clearer distinction between similar joint types

    for i in range(len(joint_det)):
        # scale up to 512 x 512
        scale = 512.0 / output_size
        x, y = int(joint_det[i, 0] * scale), int(joint_det[i, 1] * scale)
        type = joint_det[i, 2]
        if type != -1:
            color = (colors_joints[type], 255, 255)
            cv2.circle(img, (x, y), 2, color, -1)
    for person in range(len(joint_gt)):
        if np.sum(joint_gt[person]) > 0.0:
            for i in range(len(joint_gt[person])):
                # scale up to 512 x 512
                scale = 512.0 / output_size
                x, y = int(joint_gt[person, i, 0] * scale), int(joint_gt[person, i, 1] * scale)
                type = i
                if type != -1:
                    cv2.circle(img, (x, y), 2, (120, 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    fig = plt.figure()
    plt.imshow(img)
    if fname is not None:
        plt.savefig(fig=fig, fname=fname)
        plt.close(fig)
    else:
        raise NotImplementedError


def draw_poses(img: [torch.tensor, np.array], persons, fname=None, output_size=128.0):
    """

    :param img:
    :param persons: (N,17,3) array containing the person in the image img. Detected or ground truth does not matter.
    :param fname: If not none an image will be saved under this name , otherwise the image will be displayed
    :return:
    """
    img = to_numpy(img)
    assert img.shape[0] == 512 or img.shape[1] == 512

    if len(persons) == 0:
        fig = plt.figure()
        plt.imshow(img)
        plt.savefig(fig=fig, fname=fname)
        plt.close(fig)
        return
    pair_ref = [
        [1, 2], [2, 3], [1, 3],
        [6, 8], [8, 10], [12, 14], [14, 16],
        [7, 9], [9, 11], [13, 15], [15, 17],
        [6, 7], [12, 13], [6, 12], [7, 13]
    ]
    bones = np.array(pair_ref) - 1
    colors = np.arange(0, 179, np.ceil(179 / len(persons)))
    colors_joints = np.arange(0, 179, np.ceil(179 / 17), dtype=np.float)
    colors_joints[1::2] = colors_joints[-2::-2] # swap some colors to have clearer distinction between similar joint types
    # image to 8bit hsv (i dont know what hsv values opencv expects in 32bit case=)
    if img.dtype != np.uint8:
        img = img * 255.0
        img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    assert len(colors) == len(persons)
    for person in range(len(persons)):
        scale = 512.0 / output_size
        color = (colors[person], 255., 255)
        valid_joints = persons[person, :, 2] > 0
        t = persons[person, valid_joints]
        center_joint = np.mean(persons[person, valid_joints] * scale, axis=0).astype(np.int)
        for i in range(len(persons[person])):
            # scale up to 512 x 512
            joint_1 = persons[person, i]
            color_joint = (colors_joints[i], 255., 255.)
            joint_1_valid = joint_1[2] > 0
            x_1, y_1 = np.multiply(joint_1[:2], scale).astype(np.int)
            if joint_1_valid:
                cv2.circle(img, (x_1, y_1), 2, color_joint, -1)
                cv2.line(img, (x_1, y_1), (center_joint[0], center_joint[1]), color)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    fig = plt.figure()
    plt.imshow(img)
    if fname is not None:
        plt.savefig(fig=fig, fname=fname)
        plt.close(fig)
    else:
        raise NotImplementedError


def draw_clusters(img: [torch.tensor, np.array], joints, joint_connections, fname=None, output_size=128.0):
    """

    :param img:
    :param persons: (N,17,3) array containing the person in the image img. Detected or ground truth does not matter.
    :param fname: If not none an image will be saved under this name , otherwise the image will be displayed
    :return:
    """
    img = to_numpy(img)
    assert img.shape[0] == 512 or img.shape[1] == 512

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    joints, joint_connections = to_numpy(joints), to_numpy(joint_connections)
    num_nodes = len(joints)
    adj_matrix = np.zeros([num_nodes, num_nodes])
    adj_matrix[joint_connections[0], joint_connections[1]] = 1
    graph = csr_matrix(adj_matrix)
    n_components, person_labels = connected_components(graph, directed=False, return_labels=True)

    # count number of valid cc
    num_cc = 0
    for i in range(n_components):
        person_joints = joints[person_labels == i]
        if len(person_joints) > 1:
            num_cc += 1

    colors_person = np.arange(0, 179, np.ceil(179 / num_cc))
    colors_joints = np.arange(0, 179, np.ceil(179 / 17), dtype=np.float)
    colors_joints[1::2] = colors_joints[-2::-2] # swap some colors to have clearer distinction between similar joint types

    if img.dtype != np.uint8:
        img = img * 255.0
        img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    color_i = 0
    scale = 512.0 / output_size
    for i in range(n_components):
        person_joints = joints[person_labels == i]
        if len(person_joints) == 1:
            continue

        joint_centers = []
        for joint_type in range(17):  # 17 different joint types
            # take the detected joints of a certain type
            person_joint_for_type = person_joints[person_joints[:, 2] == joint_type]
            color = (colors_joints[joint_type], 255., 255)
            if len(person_joint_for_type) != 0:
                joint_center = np.mean(person_joint_for_type, axis=0)
                joint_centers.append(joint_center)
                for x, y, _ in person_joint_for_type:
                    x, y = int(x * scale), int(y * scale)
                    cv2.circle(img, (x, y), 2, color, -1)
        pose_center = np.array(joint_centers).mean(axis=0)
        pose_center = (pose_center * scale).astype(np.int)
        for x, y, _ in joint_centers:
            x, y = int(x * scale), int(y * scale)
            color = (colors_person[color_i], 255., 255)
            cv2.line(img, (x, y), (pose_center[0], pose_center[1]), color)

        color_i += 1

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    fig = plt.figure()
    plt.imshow(img)
    if fname is not None:
        plt.savefig(fig=fig, fname=fname)
        plt.close(fig)
    else:
        raise NotImplementedError


def pred_to_person(joint_det, edge_index, pred, cc_method):
    test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=pred)
    sol = cluster_graph(test_graph, cc_method, complete=False)
    sparse_sol, _ = dense_to_sparse(torch.from_numpy(sol))
    persons_pred, mutants = graph_cluster_to_persons(joint_det, sparse_sol)  # might crash
    return persons_pred, mutants


def graph_cluster_to_persons(joints, joint_connections):
    """
    :param joints: (N, 2) vector of joints
    :param joint_connections: (2, E) array/tensor that indicates which joint are connected thus belong to the same person
    :return: (N persons, 17, 3) array. 17 joints, 2 positions + visibiilty flag (in case joints are missing)
    """
    joints, joint_connections = to_numpy(joints), to_numpy(joint_connections)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    # construct dense adj matrix
    num_nodes = len(joints)
    adj_matrix = np.zeros([num_nodes, num_nodes])
    adj_matrix[joint_connections[0], joint_connections[1]] = 1
    graph = csr_matrix(adj_matrix)
    n_components, person_labels = connected_components(graph, directed=False, return_labels=True)
    persons = []
    mutant_detected = False
    for i in range(n_components):
        # check if cc has more than one node
        person_joints = joints[person_labels == i]
        if len(person_joints) > 17:
            # print(f"Mutant detected!! It has {len(person_joints)} joints!!")
            # todo change meaning of mutant
            mutant_detected = True

        if len(person_joints) > 1:  # isolated joints also form a cluster -> ignore them
            # rearrange person joints
            keypoints = np.zeros([17, 3])
            for joint_type in range(17):  # 17 different joint types
                # take the detected joints of a certain type
                person_joint_for_type = person_joints[person_joints[:, 2] == joint_type]
                if len(person_joint_for_type) != 0:
                    keypoints[joint_type] = np.mean(person_joint_for_type, axis=0)
            keypoints[np.sum(keypoints, axis=1) != 0, 2] = 1
            keypoints[keypoints[:, 2] == 0, :2] = keypoints[keypoints[:, 2] != 0, :2].mean(axis=0)
            persons.append(keypoints)
    persons = np.array(persons)
    return persons, mutant_detected


def num_non_detected_points(joint_det, keypoints, threshold, use_gt):

    person_idx_gt, joint_idx_gt = keypoints[0, :, :, 2].nonzero(as_tuple=True)
    joints_gt = keypoints[0, person_idx_gt, joint_idx_gt, :2].round().long()
    joints_gt_loc = torch.cat([joints_gt, joint_idx_gt.unsqueeze(1)], 1)
    distance = torch.norm(joints_gt_loc[:, None, :2] - joint_det[:, :2].float(), dim=2)

    different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
    distance[different_type] = 100000.0
    non_valid = distance >= threshold
    distance[non_valid] = 100000.0
    from scipy.optimize import linear_sum_assignment
    distance = distance.cpu().numpy()
    if use_gt:
        distance = distance[:, :-len(person_idx_gt)]
    sol = linear_sum_assignment(distance)
    cost = np.sum(distance[sol[0], sol[1]])
    num_miss_detections = cost // 100000
    return num_miss_detections, len(person_idx_gt)


def adjust(ans, det):
    for people_id, i in enumerate(ans):
        for joint_id, joint in enumerate(i):
            if joint[2] > 0:
                y, x = joint[0], joint[1]# joint[0:2]
                xx, yy = int(x), int(y)
                # print(batch_id, joint_id, det[batch_id].shape)
                tmp = det[joint_id]
                if tmp[xx, min(yy + 1, tmp.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
                    y += 0.25
                else:
                    y -= 0.25

                if tmp[min(xx + 1, tmp.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
                    x += 0.25
                else:
                    x -= 0.25
                ans[people_id, joint_id, 1] = x + 0.5
                ans[people_id, joint_id, 0] = y + 0.5
    return ans


def to_tensor(device, *args):
    out = []
    for a in args:
        out.append(torch.from_numpy(a).to(device).unsqueeze(0))
    return out
