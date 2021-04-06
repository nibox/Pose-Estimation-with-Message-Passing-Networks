import cv2
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import dense_to_sparse, precision, recall, accuracy, f1_score, subgraph
from scipy.optimize import linear_sum_assignment
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph
from Utils.dataset_utils import Graph
from Utils.eval import gen_ann_format, gen_ann_format_mean, gen_ann_format_correct
from Utils.transformations import reverse_affine_map


def non_maximum_suppression(scoremap, threshold=0.05, pool_kernel=None):
    assert pool_kernel % 2 == 1
    pool = nn.MaxPool2d(pool_kernel, 1, pool_kernel//2)
    pooled = pool(scoremap)
    maxima = torch.eq(pooled, scoremap).float()
    return maxima

def topk_accuracy(output, target, topk=1, mask=None):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    if mask is not None:
        output = output[mask == 1.0]
        target = target[mask == 1.0]
    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.float().sum(0).mean()

    return correct.item()

def to_numpy(array: [torch.Tensor, np.array]):
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    else:
        return array


def draw_detection_with_cluster(img, joint_det, person_labels, fname=None, num_joints=17):
    """
    :param img: torcg.tensor. image
    :param joint_det: shape: (num_joints, 2) list of xy positions of detected joints (without classes or clustering)
    :param fname: optional - file name of image to save. If None the image is show with plt.show
    :return:
    """


    if isinstance(img, torch.Tensor):
        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
        if img.dtype != np.uint8:
            img = img * 255.0
            img = img.astype(np.uint8)
    else:
        raise NotImplementedError

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    colors_joints = np.linspace(0, 179, person_labels.max() + 1, dtype=np.float)
    # colors_joints[1::2] = colors_joints[-2::-2] # swap some colors to have clearer distinction between similar joint types

    for i in range(len(joint_det)):
        # scale up to 512 x 512
        x, y = int(joint_det[i, 0]), int(joint_det[i, 1])
        type = joint_det[i, 2]
        cluster = person_labels[i]
        if type != -1:  # not sure why that is here
            color = (colors_joints[cluster], 255, 255)
            cv2.circle(img, (x, y), 2, color, -1)
    """
    for person in range(len(joint_gt)):
        if np.sum(joint_gt[person]) > 0.0:
            for i in range(len(joint_gt[person])):
                # scale up to 512 x 512
                scale = 512.0 / output_size
                x, y = int(joint_gt[person, i, 0] * scale), int(joint_gt[person, i, 1] * scale)
                type = i
                if type != -1:
                    cv2.circle(img, (x, y), 2, (120, 255, 255), -1)
    # """
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imwrite(fname + f"_det_groups.png", img)

def draw_detection_with_conf(img, joint_det, joint_score, joint_gt, fname=None, output_size=128.0):
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
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    colors_joints = [(255, 0, 0), (255, 170, 0), (0, 26, 255), (0, 255, 0)]# np.arange(0, 179, np.ceil(179 / 17), dtype=np.float)
    # colors_joints[1::2] = colors_joints[-2::-2] # swap some colors to have clearer distinction between similar joint types

    for i in range(len(joint_det)):
        # scale up to 512 x 512
        scale = 512.0 / output_size
        x, y = int(joint_det[i, 0] * scale), int(joint_det[i, 1] * scale)
        type = joint_det[i, 2]
        conf = joint_score[i]
        if conf < 0.33:
            conf_class = 0
        elif 0.33 <= conf < 0.5:
            conf_class = 1
        elif 0.5 <= conf <= 0.66:
            conf_class = 2
        elif conf > 0.66:
            conf_class = 3
        else:
            raise NotImplementedError
        if type != -1:  # not sure why that is here
            color = colors_joints[conf_class]
            cv2.circle(img, (x, y), 2, color, -1)
    for person in range(len(joint_gt)):
        if np.sum(joint_gt[person]) > 0.0:
            for i in range(len(joint_gt[person])):
                # scale up to 512 x 512
                scale = 512.0 / output_size
                x, y = int(joint_gt[person, i, 0] * scale), int(joint_gt[person, i, 1] * scale)
                type = i
                if type != -1:
                    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    # img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    fig = plt.figure()
    plt.imshow(img)
    if fname is not None:
        plt.savefig(fig=fig, fname=fname)
        plt.close(fig)
    else:
        raise NotImplementedError

def draw_detection_scoremap(scoremaps, joint_det, joint_gt, inp_type, fname=None, output_size=128.0):
    """
    :param img: torcg.tensor. image
    :param joint_det: shape: (num_joints, 2) list of xy positions of detected joints (without classes or clustering)
    :param fname: optional - file name of image to save. If None the image is show with plt.show
    :return:
    """

    scoremap = scoremaps.cpu().numpy().squeeze()
    if inp_type is not None:
        scoremap = scoremap[None, inp_type].clip(0, 1.0) * 255
    else:
        scoremap = scoremap.max(axis=0)[None].clip(0, 1.0) * 255
    scoremap = np.repeat(scoremap, repeats=3, axis=0).astype(np.uint8).transpose(1, 2, 0)
    scoremap = cv2.resize(scoremap, (512, 512))

    img = to_numpy(scoremap)
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
        j_type = joint_det[i, 2]
        if inp_type is None or j_type == inp_type:
            color = (colors_joints[j_type], 255, 255)
            cv2.circle(img, (x, y), 2, color, -1)
    for person in range(len(joint_gt)):
        if np.sum(joint_gt[person]) > 0.0:
            for i in range(len(joint_gt[person])):
                # scale up to 512 x 512
                scale = 512.0 / output_size
                x, y = int(joint_gt[person, i, 0] * scale), int(joint_gt[person, i, 1] * scale)
                j_type = i
                if inp_type is None or j_type == inp_type:
                    cv2.circle(img, (x, y), 2, (120, 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    fig = plt.figure()
    plt.imshow(img)
    if fname is not None:
        plt.savefig(fig=fig, fname=fname)
        plt.close(fig)
    else:
        raise NotImplementedError


def draw_detection(img, joint_det, joint_gt, fname=None):
    """
    :param img: torcg.tensor. image
    :param joint_det: shape: (num_joints, 2) list of xy positions of detected joints (without classes or clustering)
    :param fname: optional - file name of image to save. If None the image is show with plt.show
    :return:
    """


    if isinstance(img, torch.Tensor):
        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
        if img.dtype != np.uint8:
            img = img * 255.0
            img = img.astype(np.uint8)
    else:
        raise NotImplementedError
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    colors_joints = np.arange(0, 179, np.ceil(179 / 17), dtype=np.float)
    colors_joints[1::2] = colors_joints[-2::-2] # swap some colors to have clearer distinction between similar joint types

    for i in range(len(joint_det)):
        # scale up to 512 x 512
        x, y = int(joint_det[i, 0]), int(joint_det[i, 1])
        type = joint_det[i, 2]
        if type != -1:
            color = (colors_joints[type], 255, 255)
            cv2.circle(img, (x, y), 2, color, -1)
    for person in range(len(joint_gt)):
        if np.sum(joint_gt[person]) > 0.0:
            for i in range(len(joint_gt[person])):
                # scale up to 512 x 512
                x, y = int(joint_gt[person, i, 0]), int(joint_gt[person, i, 1])
                type = i
                if type != -1:
                    cv2.circle(img, (x, y), 2, (120, 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imwrite(fname + f"_det.png", img)

def draw_detection_classification_result(img, joint_det, joint_labels, fname=None):
    """
    :param img: torcg.tensor. image
    :param joint_det: shape: (num_joints, 2) list of xy positions of detected joints (without classes or clustering)
    :param fname: optional - file name of image to save. If None the image is show with plt.show
    :return:
    """


    if isinstance(img, torch.Tensor):
        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
        if img.dtype != np.uint8:
            img = img * 255.0
            img = img.astype(np.uint8)
    else:
        raise NotImplementedError
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    colors_label = [0, 120]

    for i in range(len(joint_det)):
        # scale up to 512 x 512
        x, y = int(joint_det[i, 0]), int(joint_det[i, 1])
        label = joint_labels[i]
        if type != -1:
            color = (colors_label[label], 255, 255)
            cv2.circle(img, (x, y), 2, color, -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imwrite(fname + f"_det_fp.png", img)


def draw_poses(img: [torch.tensor, np.array], persons, fname=None, output_size=128.0):
    """

    :param img:
    :param persons: (N,17,3) array containing the person in the image img. Detected or ground truth does not matter.
    :param fname: If not none an image will be saved under this name , otherwise the image will be displayed
    :return:
    """
    img = to_numpy(img)
    # assert img.shape[0] == 512 or img.shape[1] == 512

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
    colors = np.linspace(0, 179, len(persons))
    colors_joints = np.linspace(0, 179, 17, dtype=np.float)
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



def draw_poses_fp(img: [torch.tensor, np.array], persons, debug_flags, fname=None, output_size=128.0):
    """

    :param img:
    :param persons: (N,17,3) array containing the person in the image img. Detected or ground truth does not matter.
    :param fname: If not none an image will be saved under this name , otherwise the image will be displayed
    :return:
    """
    img = to_numpy(img)
    # assert img.shape[0] == 512 or img.shape[1] == 512

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
    colors = np.linspace(0, 179, len(persons))
    colors_joints = (None, 0, 236/2, 116/2)
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
            joint_1_valid = joint_1[2] > 0
            x_1, y_1 = np.multiply(joint_1[:2], scale).astype(np.int)
            if joint_1_valid:
                color_joint = (colors_joints[int(debug_flags[person, i])], 255., 255.)
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


def draw_clusters(img: [torch.tensor, np.array], joints, person_labels, fname=None, num_joints=17):
    """

    :param joint_classes:
    :param img:
    :param persons: (N,17,3) array containing the person in the image img. Detected or ground truth does not matter.
    :param fname: If not none an image will be saved under this name , otherwise the image will be displayed
    :return:
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
        if img.dtype != np.uint8:
            img = img * 255.0
            img = img.astype(np.uint8)
    else:
        raise NotImplementedError

    # count number of valid cc
    num_cc = 0
    num_components = person_labels.max() + 1
    for i in range(num_components):
        person_joints = joints[person_labels == i]
        if len(person_joints) > 1:
            num_cc += 1

    colors_person = np.arange(0, 179, np.ceil(179 / num_cc))
    colors_joints = np.arange(0, 179, np.ceil(179 / 17), dtype=np.float)
    colors_joints[1::2] = colors_joints[-2::-2] # swap some colors to have clearer distinction between similar joint types

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    color_i = 0
    for i in range(num_components):
        person_joints = joints[person_labels == i]
        if len(person_joints) == 1:
            continue

        joint_centers = []
        for joint_type in range(num_joints):  # 17 different joint types
            # take the detected joints of a certain type
            person_joint_for_type = person_joints[person_joints[:, 2] == joint_type]
            color = (colors_joints[joint_type], 255., 255)
            if len(person_joint_for_type) != 0:
                joint_center = np.mean(person_joint_for_type, axis=0)
                joint_centers.append(joint_center)
                for x, y, _ in person_joint_for_type:
                    x, y = int(x), int(y)
                    cv2.circle(img, (x, y), 2, color, -1)
        pose_center = np.array(joint_centers).mean(axis=0)
        pose_center = (pose_center).astype(np.int)
        for x, y, _ in joint_centers:
            x, y = int(x), int(y)
            color = (colors_person[color_i], 255., 255)
            cv2.line(img, (x, y), (pose_center[0], pose_center[1]), color)

        color_i += 1

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    cv2.imwrite(fname + f"_groups.png", img)


def draw_grouping(img: [torch.tensor, np.array], joints, person_labels, fname=None, num_joints=17):
    """

    :param joint_classes:
    :param img:
    :param persons: (N,17,3) array containing the person in the image img. Detected or ground truth does not matter.
    :param fname: If not none an image will be saved under this name , otherwise the image will be displayed
    :return:
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
        if img.dtype != np.uint8:
            img = img * 255.0
            img = img.astype(np.uint8)
    else:
        raise NotImplementedError

    # count number of valid cc
    num_cc = 0
    num_components = person_labels.max() + 1
    for i in range(num_components):
        person_joints = joints[person_labels == i]
        if len(person_joints) > 1:
            num_cc += 1

    colors_person = np.arange(0, 179, np.ceil(179 / num_cc))
    colors_joints = np.arange(0, 179, np.ceil(179 / 17), dtype=np.float)
    colors_joints[1::2] = colors_joints[-2::-2] # swap some colors to have clearer distinction between similar joint types

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    color_i = 0
    for i in range(num_components):
        person_joints = joints[person_labels == i]
        if len(person_joints) == 1:
            continue

        joint_centers = []
        for joint_type in range(num_joints):  # 17 different joint types
            # take the detected joints of a certain type
            person_joint_for_type = person_joints[person_joints[:, 2] == joint_type]
            color = (colors_joints[joint_type], 255., 255)
            if len(person_joint_for_type) != 0:
                joint_center = np.mean(person_joint_for_type, axis=0)
                joint_centers.append(joint_center)
                for x, y, _ in person_joint_for_type:
                    x, y = int(x), int(y)
                    cv2.circle(img, (x, y), 2, color, -1)
        pose_center = np.array(joint_centers).mean(axis=0)
        pose_center = (pose_center).astype(np.int)
        for x, y, _ in joint_centers:
            x, y = int(x), int(y)
            color = (colors_person[color_i], 255., 255)
            cv2.line(img, (x, y), (pose_center[0], pose_center[1]), color)

        color_i += 1

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    cv2.imwrite(fname + f"_groups.png", img)

def pred_to_person(joint_det, joint_scores, edge_index, pred, class_pred, cc_method, num_joints, score_for_poses=None,
                   allow_single_joint_persons=False):
    test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=pred)
    if cc_method in ["GAEC", "MUT", "KL"]:
        sol = cluster_graph(test_graph, cc_method, complete=False)
        sparse_sol, _ = dense_to_sparse(torch.from_numpy(sol))
    elif cc_method == "greedy":
        persons_pred, person_labels = greedy_person_construction(joint_det, joint_scores, pred, class_pred, edge_index, num_joints)
        return persons_pred, False, person_labels
    elif cc_method =="threshold":
        sparse_sol = edge_index[:, pred > 0.8]
    else:
        raise NotImplementedError
    persons_pred, mutants, person_labels = graph_cluster_to_persons(joint_det, joint_scores, sparse_sol, class_pred,
                                                                    num_joints, score_for_poses, allow_single_joint_persons)  # might crash
    return persons_pred, mutants, person_labels


def greedy_person_construction(joint_det, preds_nodes, preds_edges, preds_classes, edge_index, num_joints):

    # is this just clustering? or is it person construction?
    joint_det = joint_det.cpu().numpy()
    preds_nodes = preds_nodes.cpu().numpy()
    preds_edges = preds_edges.cpu().numpy()
    preds_classes = preds_classes.cpu().numpy() if preds_classes is not None else None
    edge_index = edge_index.cpu().numpy()

    # reclassify predictions
    if preds_classes is not None:
        joint_det[:, 2] = preds_classes.argmax(axis=1)
    adj_mat = np.zeros((len(joint_det), len(joint_det)), dtype=np.float)
    adj_mat[edge_index[0], edge_index[1]] = preds_edges
    adj_mat = (adj_mat.T + adj_mat) / 2.0   # make it symetric by averaging
    adj_mat[np.diag_indices(len(adj_mat))] = 1.0

    # structure similar to ae grouping
    # iterate through types
    # remove assinged nodes?
    # assuming an eye is always detected
    # -- threshold and for each type select maximum edges
    # -- one node two clusters -> both high score edge -> assign to both
    # -- one high one low

    # stores index of core node of cluster
    taken_joints = np.zeros_like(preds_nodes, dtype=np.int32) - 1

    out = []
    for type in range(num_joints):
       # todo rearange types just like ae group
        type_joints = joint_det[:, 2] == type
        for i in range(len(joint_det)):
            # look at joint with type: type and joints that do not belong to a cluster
            if not type_joints[i] or taken_joints[i] != -1:
                continue
            """
            person = np.zeros(num_joints, 3)
            person[type, :2] = joint_det[i]
            person[type, 2] = preds_nodes[i]
            """
            if preds_nodes[i] < 0.5:
                continue
            taken_joints[i] = i

            for j in range(num_joints):
                if j == type:
                    continue
                target_joint_type = joint_det[:, 2] != j
                tmp_mat = adj_mat[i, :].copy()
                tmp_mat[target_joint_type] = 0.0
                target_edge_score = np.max(tmp_mat)
                target_joint_idx = np.argmax(tmp_mat)
                if target_edge_score == 0.0 or target_joint_idx == i:
                    continue
                # if joint is taken compare scores
                if taken_joints[target_joint_idx] != -1:
                    first_score = adj_mat[taken_joints[target_joint_idx], target_joint_idx]
                    current_score = target_edge_score
                    if first_score > current_score:
                        continue
                    else:
                        # if new score is larger than old one -> reassign
                        taken_joints[target_joint_idx] = i
                else:
                    taken_joints[target_joint_idx] = i

    # construct persons based on clusters
    n_cluster = taken_joints.max() + 1
    persons = []
    for i in range(n_cluster):
        # check if cc has more than one node
        select = taken_joints == i
        person_joints = joint_det[select]
        person_scores = preds_nodes[select]

        if len(person_joints) > 1:  # isolated joints also form a cluster -> ignore them
            # rearrange person joints
            keypoints = np.zeros([num_joints, 3])
            for joint_type in range(num_joints):  # 17 different joint types
                # take the detected joints of a certain type
                select = person_joints[:, 2] == joint_type
                person_joint_for_type = person_joints[select]
                person_scores_for_type = person_scores[select]
                if len(person_joint_for_type) != 0:
                    # keypoints[joint_type, 2] = np.max(person_scores_for_type, axis=0)
                    joint_idx = np.argmax(person_scores_for_type, axis=0)
                    keypoints[joint_type] = person_joint_for_type[
                        joint_idx]  # np.mean(person_joint_for_type, axis=0)
                    keypoints[joint_type, 2] = np.max(person_scores_for_type, axis=0)

            # keypoints[np.sum(keypoints, axis=1) != 0, 2] = 1
            if (keypoints[:, 2] > 0).sum() > 0:
                # keypoints[np.sum(keypoints, axis=1) != 0, 2] = 1
                persons.append(keypoints)
        elif len(person_joints) == 1 and False:

            keypoints = np.zeros([num_joints, 3])
            joint_type = person_joints[:, 2]
            person_score = person_scores[0]
            if person_score < 0.1:
                continue

            keypoints[joint_type, 2] = person_score
            keypoints[:, :2] = person_joints[0, :2]

            persons.append(keypoints)

    persons = np.array(persons)
    return persons, taken_joints


def pred_to_person_ae(joint_det, joint_scores, edge_index, pred, class_pred, cc_method, num_joints, score_for_poses=None,
                   allow_single_joint_persons=False):
    if edge_index.shape[1] != 0:
        test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=pred)
        sol = cluster_graph(test_graph, cc_method, complete=False)
        sparse_sol, _ = dense_to_sparse(torch.from_numpy(sol))
    else:
        # each joint is a cluster
        sparse_sol = [[], []]
    persons_pred, mutants, person_labels = graph_cluster_to_persons(joint_det, joint_scores, sparse_sol, class_pred,
                                                                    num_joints, score_for_poses, allow_single_joint_persons)  # might crash
    return persons_pred, mutants, person_labels

def pred_to_person_debug(joint_det, joint_scores, edge_index, edge_pred, class_pred, node_labels, cc_method, num_joints):
    test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=edge_pred)
    if cc_method != "threshold":
        sol = cluster_graph(test_graph, cc_method, complete=False)
        sparse_sol, _ = dense_to_sparse(torch.from_numpy(sol))
    else:
        sparse_sol = edge_index[:, edge_pred > 0.5]
    persons_pred, person_labels, debug_flags = graph_cluster_to_persons_debug(joint_det, joint_scores, sparse_sol,
                                                                    class_pred, node_labels, num_joints)  # might crash
    return persons_pred, person_labels, debug_flags


def pred_to_person_labels(joint_det, edge_index, edge_attr, cc_method="GAEC"):
    test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=edge_attr)

    sol = cluster_graph(test_graph, cc_method, complete=False)
    sparse_sol, _ = dense_to_sparse(torch.from_numpy(sol))

    joint_connections = to_numpy(sparse_sol)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    # construct dense adj matrix
    num_nodes = len(joint_det)
    adj_matrix = np.zeros([num_nodes, num_nodes])
    adj_matrix[joint_connections[0], joint_connections[1]] = 1
    graph = csr_matrix(adj_matrix)
    n_components, person_labels = connected_components(graph, directed=False, return_labels=True)
    return person_labels


def graph_cluster_to_persons(joints, joint_scores, joint_connections, class_pred, num_joints, scores_for_poses=None,
                             allow_single_joint_persons=False):
    """
    :param allow_single_joint_persons:
    :param class_pred:
    :param joints: (N, 2) vector of joints
    :param joint_connections: (2, E) array/tensor that indicates which joint are connected thus belong to the same person
    :return: (N persons, 17, 3) array. 17 joints, 2 positions + visibiilty flag (in case joints are missing)
    """
    joints, joint_connections, joint_scores = to_numpy(joints), to_numpy(joint_connections), to_numpy(joint_scores)
    joint_classes = to_numpy(class_pred) if class_pred is not None else None
    scores_for_poses = to_numpy(scores_for_poses) if scores_for_poses is not None else None
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
        person_scores = joint_scores[person_labels == i]
        person_pose_scores = scores_for_poses[person_labels == i] if scores_for_poses is not None else None
        if joint_classes is not None:
            c = person_joints[:, 2]
            d = np.argmax(joint_classes[person_labels == i], axis=1)
            person_joints[:, 2] = np.argmax(joint_classes[person_labels == i], axis=1)
        if len(person_joints) > num_joints:
            # print(f"Mutant detected!! It has {len(person_joints)} joints!!")
            # todo change meaning of mutant
            mutant_detected = True

        if len(person_joints) > 1:  # isolated joints also form a cluster -> ignore them
            # rearrange person joints
            keypoints = np.zeros([num_joints, 3])
            for joint_type in range(num_joints):  # 17 different joint types
                # take the detected joints of a certain type
                select = person_joints[:, 2] == joint_type
                person_joint_for_type = person_joints[select]
                person_scores_for_type = person_scores[select]
                person_pose_scores_for_type = person_pose_scores[select] if person_pose_scores is not None else None
                if len(person_joint_for_type) != 0:
                    #keypoints[joint_type, 2] = np.max(person_scores_for_type, axis=0)
                    joint_idx = np.argmax(person_scores_for_type, axis=0)
                    keypoints[joint_type] = person_joint_for_type[joint_idx]# np.mean(person_joint_for_type, axis=0)
                    keypoints[joint_type, 2] = np.max(person_scores_for_type, axis=0)
                    if person_pose_scores_for_type is not None:
                        keypoints[joint_type, 2] = person_pose_scores_for_type[joint_idx]

            #keypoints[np.sum(keypoints, axis=1) != 0, 2] = 1
            if (keypoints[:, 2] > 0).sum() > 0:
                # keypoints[np.sum(keypoints, axis=1) != 0, 2] = 1
                persons.append(keypoints)
        elif len(person_joints) == 1 and allow_single_joint_persons:

            keypoints = np.zeros([num_joints, 3])
            joint_type = person_joints[:, 2]
            person_score = person_scores[0]
            if person_score < 0.1:
                 continue

            keypoints[joint_type, 2] = person_score
            keypoints[:, :2] = person_joints[0, :2]

            persons.append(keypoints)

    persons = np.array(persons)
    return persons, mutant_detected, person_labels


def graph_cluster_to_persons_debug(joints, joint_scores, joint_connections, class_pred, node_labels):
    """
    :param class_pred:
    :param joints: (N, 2) vector of joints
    :param joint_connections: (2, E) array/tensor that indicates which joint are connected thus belong to the same person
    :return: (N persons, 17, 3) array. 17 joints, 2 positions + visibiilty flag (in case joints are missing)
    """
    joints, joint_connections, joint_scores = to_numpy(joints), to_numpy(joint_connections), to_numpy(joint_scores)
    joint_classes = to_numpy(class_pred) if class_pred is not None else None
    node_labels = to_numpy(node_labels)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    # construct dense adj matrix
    num_nodes = len(joints)
    adj_matrix = np.zeros([num_nodes, num_nodes])
    adj_matrix[joint_connections[0], joint_connections[1]] = 1
    graph = csr_matrix(adj_matrix)
    n_components, person_labels = connected_components(graph, directed=False, return_labels=True)

    persons = []
    debug = []
    for i in range(n_components):
        # check if cc has more than one node
        person_joints = joints[person_labels == i]
        person_scores = joint_scores[person_labels == i]
        person_node_labels = node_labels[person_labels == i]
        if joint_classes is not None:
            person_joints[:, 2] = np.argmax(joint_classes[person_labels == i], axis=1)

        if len(person_joints) > 1:  # isolated joints also form a cluster -> ignore them
            # rearrange person joints
            # fourth dimension is used for label information to distinguish joint
            keypoints = np.zeros([17, 3])
            person_debug = np.zeros(17)
            for joint_type in range(17):  # 17 different joint types
                # take the detected joints of a certain type
                select = person_joints[:, 2] == joint_type
                person_joint_for_type = person_joints[select]
                person_scores_for_type = person_scores[select]
                person_node_labels_for_type = person_node_labels[select]
                if len(person_joint_for_type) != 0:
                    joint_idx = np.argmax(person_scores_for_type, axis=0)
                    keypoints[joint_type] = person_joint_for_type[joint_idx]
                    keypoints[joint_type, 2] = np.max(person_scores_for_type, axis=0)

                    # coding 0 : nothin
                    #        1 : there is a tp and it is replaced with a fp (swap error)
                    #        2 : there is not tp but there is an fp (fill in)
                    #        3 : there is tp and it is chosen
                    chosen_label = person_node_labels_for_type[joint_idx]
                    available_label = person_node_labels_for_type.max()
                    if chosen_label == 1:  # 3
                        person_debug[joint_type] = 3
                    elif chosen_label == 0 and available_label == 1: # 1
                        person_debug[joint_type] = 1
                    elif chosen_label == 0 and available_label == 0: # 2
                        person_debug[joint_type] = 2
                    else:
                        print(f"chose_label: {chosen_label}, available_label: {available_label}")
                        raise NotImplementedError

            if (keypoints[:, 2] > 0).sum() > 0:
                keypoints[keypoints[:, 2] == 0, :2] = keypoints[keypoints[:, 2] != 0, :2].mean(axis=0)
                persons.append(keypoints)
                assert (person_debug[keypoints[:, 2] > 0] == 0).sum() == 0
                debug.append(person_debug)

    persons = np.array(persons)
    debug = np.array(debug)
    return persons, person_labels, debug


def parse_refinement(joints, joint_scores, person_labels):
    """
    :param class_pred:
    :param joints: (N, 2) vector of joints
    :param joint_connections: (2, E) array/tensor that indicates which joint are connected thus belong to the same person
    :return: (N persons, 17, 3) array. 17 joints, 2 positions + visibiilty flag (in case joints are missing)
    """
    joints, joint_scores = to_numpy(joints), to_numpy(joint_scores)
    person_labels = to_numpy(person_labels.long())

    persons = []
    for i in range(person_labels.max() + 1):
        # check if cc has more than one node
        person_joints = joints[person_labels == i]
        person_scores = joint_scores[person_labels == i]

        if len(person_joints) > 1:  # isolated joints also form a cluster -> ignore them
            # rearrange person joints
            keypoints = np.zeros([17, 3])
            for joint_type in range(17):  # 17 different joint types
                # take the detected joints of a certain type
                person_joint_for_type = person_joints[person_joints[:, 2] == joint_type]
                person_scores_for_type = person_scores[person_joints[:, 2] == joint_type]
                if len(person_joint_for_type) != 0:
                    f = True
                    if f:
                        keypoints[joint_type, 2] = np.max(person_scores_for_type, axis=0)
                        joint_idx = np.argmax(person_scores_for_type, axis=0)
                        keypoints[joint_type] = person_joint_for_type[joint_idx]# np.mean(person_joint_for_type, axis=0)
                        keypoints[joint_type, 2] = np.max(person_scores_for_type, axis=0)
                        # keypoints[joint_type, 2] = np.mean(person_scores_for_type, axis=0)
                    else:
                        keypoints[joint_type] = np.mean(person_joint_for_type, axis=0)

            keypoints[keypoints[:, 2] == 0, :2] = keypoints[keypoints[:, 2] != 0, :2].mean(axis=0)
            persons.append(keypoints)
        elif len(person_joints) == 1 and False:

            keypoints = np.zeros([17, 3])
            joint_type = person_joints[:, 2]
            person_score = person_scores[0]

            keypoints[joint_type, 2] = person_score
            keypoints[:, :2] = person_joints[0, :2]

            persons.append(keypoints)

    persons = np.array(persons)
    return persons


def num_non_detected_points(joint_det, keypoints, factors, min_num_joints=1):
    num_joints_det = len(joint_det)
    person_idx_gt, joint_idx_gt = keypoints[:, :, 2].nonzero(as_tuple=True)
    num_joints_gt = len(person_idx_gt)
    num_persons = len(set(list(person_idx_gt.cpu().numpy())))

    distance = (keypoints[person_idx_gt, joint_idx_gt, :2].unsqueeze(1).round().float() - joint_det[:, :2].float()).pow(2).sum(dim=2)
    factor_per_joint = factors[person_idx_gt, joint_idx_gt]
    similarity = torch.exp(-distance / factor_per_joint[:, None])


    different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
    similarity[different_type] = 0.0
    similarity[similarity < 0.1] = 0.0  # 0.1 worked well for threshold + knn graph

    cost_mat = similarity.cpu().numpy()
    sol = linear_sum_assignment(cost_mat, maximize=True)
    row, col = sol
    # remove mappings with cost 0.0
    valid_match = cost_mat[row, col] != 0.0
    row = row[valid_match]
    num_det_persons = len(set(list(person_idx_gt[row].cpu().numpy())))
    num_missed_detections = (valid_match == False).sum()

    return num_missed_detections, num_joints_gt, num_det_persons, num_persons


def adjust_joints(ans, det):
    assert len(ans.shape) == 2 and ans.shape[1] == 3
    for i, joint in enumerate(ans):
        y, x = joint[0], joint[1]# joint[0:2]
        joint_id = int(joint[2])
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
        ans[i, 1] = x + 0.5
        ans[i, 0] = y + 0.5
    return ans

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


def to_device(device, l):
    out = []
    for a in l:
        out.append(a.to(device))
    return out


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

def set_bn_feeze(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        for param in module.parameters():
            param.requires_grad = False


def calc_metrics(output, targets, mask=None, num_classes=2):
    if output is None:
        # do nothing
        return None

    if mask is not None:
        output = output[mask == 1.0]
        targets = targets[mask == 1.0]
        if len(output) == 0:
            return None

    prec = precision(output, targets, num_classes)[1]
    rec = recall(output, targets, num_classes)[1]
    acc = accuracy(output, targets)
    f1 = f1_score(output, targets, num_classes)[1]

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

def subgraph_mask(subset, edge_index):
    r"""Copy paste of torch_geometric.utils.subgraph but it return the mask instead of the subgraph.
    Subset has to

    Args:
        subset (BoolTensor or [int] ): The nodes to keep.
        edge_index (LongTensor): The edge indices.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    mask = subset[edge_index[0]] & subset[edge_index[1]]

    return mask


def one_hot_encode(tensor, num_classes, dtype=None, device=None):
    assert len(tensor.shape) == 1
    dtype = tensor.dtype if dtype is None else dtype
    device = tensor.device if device is None else device
    one_hot = torch.zeros(len(tensor), num_classes, dtype=dtype, device=device)
    one_hot[list(range(0, len(tensor))), tensor] = 1
    return one_hot


class Logger(object):

    def __init__(self, config):
        self.writer = SummaryWriter(config.LOG_DIR)


    def log_vars(self, name, iter, **kwargs):

        for key in kwargs.keys():
            if isinstance(kwargs[key], list):
                if kwargs[key]:
                    self.writer.add_scalar(f"{name}_{key}", np.mean(kwargs[key]), iter)
                else:
                    continue
            else:
                self.writer.add_scalar(f"{name}_{key}", kwargs[key], iter)

    def log_loss(self, loss, name, iter):
        self.writer.add_scalar(f"{name}", loss, iter)


def refine(scoremaps, tag, keypoints):
    """
    Given initial keypoint predictions, we identify missing joints
    :param det: numpy.ndarray of size (17, 128, 128)
    :param tag: numpy.ndarray of size (17, 128, 128) if not flip
    :param keypoints: numpy.ndarray of size (N, 17, 3) if not flip, last dim is (x, y, det score)
    :return:
    """
    if len(tag.shape) == 3:
        # tag shape: (17, 256, 256, 1)
        tag = tag[:, :, :, None]

    tags = []
    for p in range(keypoints.shape[0]):
        person_tags = []
        for i in range(keypoints.shape[1]):
            if keypoints[p, i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[p, i][:2].astype(np.int32)
                person_tags.append(tag[i, y, x])
        tags.append(np.array(person_tags))

    """
    small experiment whether filtering outlier tags helps -> it does not
    def estimate_mean_tag(data, m=2):
        mean_tag = []
        for d in range(data.shape[1]):
            tmp = data[:, d]
            mean_tag.append(tmp[abs(tmp - np.mean(tmp, axis=0)) < m * np.std(tmp)].mean())

        return np.array(mean_tag)
    """

    # mean tag of current detected people
    for p in range(keypoints.shape[0]):
        prev_tag = np.mean(tags[p], axis=0)
        # prev_tag = estimate_mean_tag(tags[p])  -> the mentioned filtering step -> does not help!!
        ans = []

        for i in range(keypoints.shape[1]):
            # score of joints i at all position
            tmp = scoremaps[i, :, :]
            # distance of all tag values with mean tag of current detected people
            tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
            tmp2 = tmp - np.round(tt)

            # find maximum position
            y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
            xx = x
            yy = y
            # detection score at maximum position
            val = tmp[y, x]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i in range(scoremaps.shape[0]):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[p, i, 2] == 0:
                    # if ans[i, 2] > 0.01 and keypoints[i, 2] == 0:
                    keypoints[p, i, :2] = ans[i, :2]
                    keypoints[p, i, 2] = 0.001 # ans[i, 2]

    return keypoints

def refine_tag(scoremaps, tag, keypoints):
    """
    Given initial keypoint predictions, we identify missing joints
    :param det: numpy.ndarray of size (17, 128, 128)
    :param tag: numpy.ndarray of size (17, 128, 128) if not flip
    :param keypoints: numpy.ndarray of size (N, 17, 3) if not flip, last dim is (x, y, det score)
    :return:
    """
    if len(tag.shape) == 3:
        # tag shape: (17, 256, 256, 1)
        tag = tag[:, :, :, None]

    tags = []
    for p in range(keypoints.shape[0]):
        person_tags = []
        for i in range(keypoints.shape[1]):
            if keypoints[p, i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[p, i][:2].astype(np.int32)
                person_tags.append(tag[i, y, x])
        tags.append(np.array(person_tags))

    # mean tag of current detected people
    for p in range(keypoints.shape[0]):
        prev_tag = np.mean(tags[p], axis=0)
        ans = []
        keypoint_tag_scores = []
        for i in range(keypoints.shape[1]):
            # score of joints i at all position
            tmp = scoremaps[i, :, :]
            # distance of all tag values with mean tag of current detected people
            tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
            tmp2 = tmp - np.round(tt)
            keypoint_tag_scores.append(tmp2[int(keypoints[p, i, 1]), int(keypoints[p, i, 0])])

            # find maximum position
            y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
            val_2 = tmp2[y, x]
            xx = x
            yy = y
            # detection score at maximum position
            val = tmp[y, x]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val, val_2))
        ans = np.array(ans)

        if ans is not None:
            for i in range(scoremaps.shape[0]):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[p, i, 2] == 0:
                    # if ans[i, 2] > 0.01 and keypoints[i, 2] == 0:
                    keypoints[p, i, :2] = ans[i, :2]
                    keypoints[p, i, 2] = ans[i, 2]
                elif ans[i, 2] > 0 and ans[i, 3] > keypoint_tag_scores[i]:
                    keypoints[p, i, :2] = ans[i, :2]
                    keypoints[p, i, 2] = ans[i, 2]

    return keypoints


def add_joints(image, joints, color, dataset):
    crowd_pose_part_labels = [
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        'head', 'neck'
    ]
    crowd_pose_part_idx = {
        b: a for a, b in enumerate(crowd_pose_part_labels)
    }
    crowd_pose_part_orders = [
        ('head', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle')
    ]
    coco_part_labels = [
        'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
        'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
        'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
    ]
    coco_part_idx = {
        b: a for a, b in enumerate(coco_part_labels)
    }
    coco_part_orders = [
        ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
        ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
        ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
        ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
        ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
        ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
        ('kne_r', 'ank_r')
    ]

    VIS_CONFIG = {
        'COCO': {
        'part_labels': coco_part_labels,
        'part_idx': coco_part_idx,
        'part_orders': coco_part_orders
        },
        'CROWDPOSE': {
            'part_labels': crowd_pose_part_labels,
            'part_idx': crowd_pose_part_idx,
            'part_orders': crowd_pose_part_orders
        }
    }
    data = "CROWDPOSE" if dataset == "crowd_pose" else "COCO"
    part_idx = VIS_CONFIG[data]['part_idx']
    part_orders = VIS_CONFIG[data]['part_orders']

    def link(a, b, color):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
            if jointa[2] > 0 and jointb[2] > 0:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    color,
                    2
                )

    # add joints
    for joint in joints:
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 2)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], color)

    return image


def save_valid_image(image, joints, file_name, dataset):
    if isinstance(image, torch.Tensor):
        image = image.cpu().squeeze().permute(1, 2, 0).numpy() * 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for person in joints:
        color = np.random.randint(0, 255, size=3)
        color = [int(i) for i in color]
        add_joints(image, person, color, dataset)

    cv2.imwrite(file_name, image)


def draw_edges_conf(img, joint_det, person_labels, preds_nodes, edge_index, preds_edges, fname=None):
    import cv2
    # filter nodes using the node labels -> subgraph
    # for node in nodes
    # draw node in color of cluster
    # get all connections and draw them in color of confidence
    preds_nodes = preds_nodes
    edge_index = edge_index
    joint_det = joint_det
    person_labels = person_labels
    preds_edges = preds_edges
    num_persons = person_labels.max() + 1
    types_to_draw = [0]

    if isinstance(img, torch.Tensor):
        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
    if img.dtype != np.uint8:
        img = img * 255.0
        img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # colors_joints = np.linspace(0, 179, person_labels.max() + 1, dtype=np.float)
    colors_edges = [0, 20, int(234/2), 50] #np.linspace(0, 179, 3, dtype=np.float)

    for i in range(num_persons):
        person = person_labels == i
        num_nodes_in_cluster = len(joint_det[person])
        if num_nodes_in_cluster < 2:
            # skipp cluster with less than two joints
            continue
    for joint_type in types_to_draw:
        # select the highest scoring joint for this type for this person
        all_img = img.copy()
        for i in range(num_persons):
            person = person_labels == i
            num_nodes_in_cluster = len(joint_det[person])
            if num_nodes_in_cluster < 2:
                # skipp cluster with less than two joints
                continue
            tmp_img = img.copy()
            type_joints =joint_det[:, 2] == joint_type
            tmp_scores = preds_nodes.copy()
            tmp_scores[np.logical_not(type_joints & person)] = -1.0
            if np.max(tmp_scores) == -1.0:
                continue
            joint_idx = np.argmax(tmp_scores)
            focus_point = joint_det[joint_idx]

            joint_edges = edge_index[1, :] == joint_idx
            src_idx = edge_index[0, joint_edges]
            edge_scores = preds_edges[joint_edges]
            for t in range(17):
                if t == joint_type:
                    continue
                tmp_idx = src_idx[joint_det[src_idx, 2] == t]
                tmp_scores = edge_scores[joint_det[src_idx, 2] == t]
                if len(tmp_scores) == 0:
                    continue
                idx_to_viz = np.argmax(tmp_scores)
                score_to_viz = tmp_scores[idx_to_viz]
                point_to_viz = joint_det[tmp_idx[idx_to_viz]]

                # draw
                x, y = int(focus_point[0]), int(focus_point[1])
                x_t, y_t = int(point_to_viz[0]), int(point_to_viz[1])
                if score_to_viz< 0.33:
                    conf_class = 0
                elif 0.33 <= score_to_viz < 0.5:
                    conf_class = 1
                elif 0.5 <= score_to_viz< 0.66:
                    conf_class = 2
                elif score_to_viz> 0.66:
                    conf_class = 3
                else:
                    raise NotImplementedError
                color = (colors_edges[conf_class], 255, 255)
                cv2.line(tmp_img, (x, y), (x_t, y_t), color)
                cv2.line(all_img, (x, y), (x_t, y_t), color)

            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_HSV2BGR)

            cv2.imwrite(fname + f"_{i}_{joint_type}.png", tmp_img)
        all_img= cv2.cvtColor(all_img, cv2.COLOR_HSV2BGR)
        cv2.imwrite(fname + f"_all_{joint_type}.png", all_img)


def draw_inter_person_edge_conf(img, joint_det, person_labels, preds_nodes, edge_index, preds_edges, type_to_draw, fname=None, num_joints=17):
    import cv2
    # filter nodes using the node labels -> subgraph
    # for node in nodes
    # draw node in color of cluster
    # get all connections and draw them in color of confidence
    preds_nodes = preds_nodes
    edge_index = edge_index
    joint_det = joint_det
    person_labels = person_labels
    preds_edges = preds_edges
    num_persons = person_labels.max() + 1

    if isinstance(img, torch.Tensor):
        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
        if img.dtype != np.uint8:
            img = img * 255.0
            img = img.astype(np.uint8)
    else:
        raise NotImplementedError

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # colors_joints = np.linspace(0, 179, person_labels.max() + 1, dtype=np.float)
    colors_edges = [0, 20, int(234/2), 50] #np.linspace(0, 179, 3, dtype=np.float)

    # what does this block do?
    for person in range(num_persons):
        all_img = img.copy()
        # for each person
        person_idx = person_labels == person
        num_nodes_in_cluster = len(joint_det[person_idx])
        if num_nodes_in_cluster < 2:
            # skipp cluster with less than two joints
            continue
        # joint_type == type to draw
        # select the highest scoring joint for this type for this person
        type_joints = joint_det[:, 2] == type_to_draw
        tmp_scores = preds_nodes.copy()
        tmp_scores[np.logical_not(type_joints & person_idx)] = -1.0
        if np.max(tmp_scores) == -1.0:
            continue
        joint_idx = np.argmax(tmp_scores)
        focus_point = joint_det[joint_idx]
        joint_edges = edge_index[1, :] == joint_idx  # edges of focus point
        src_idx = edge_index[0, joint_edges]  # idx of nodes that are connected to focus point
        edge_scores = preds_edges[joint_edges]  # edge scores of edges of focus point
        node_scores = preds_nodes[src_idx].copy()
        for other_person in range(num_persons):
            tmp_img = img.copy()
            src_person = person_labels[src_idx] == other_person
            if src_person.sum() < 2:
                continue
            for other_type in range(num_joints):
                # filter idx for type and for person label
                src_types = joint_det[src_idx, 2] == other_type

                # now select highest scoring node, local index should be same
                tmp_node_scores = node_scores.copy()
                tmp_node_scores[np.logical_not(src_types & src_person)] = -1.0
                if np.max(tmp_node_scores) == -1.0:
                    continue
                node_idx = np.argmax(tmp_node_scores)

                pos = joint_det[src_idx[node_idx]]
                score = edge_scores[node_idx]

                # draw
                x, y = int(focus_point[0]), int(focus_point[1])
                x_t, y_t = int(pos[0]), int(pos[1])
                if score < 0.33:
                    conf_class = 0
                elif 0.33 <= score < 0.5:
                    conf_class = 1
                elif 0.5 <= score < 0.66:
                    conf_class = 2
                elif score > 0.66:
                    conf_class = 3
                else:
                    raise NotImplementedError
                color = (colors_edges[conf_class], 255, 255)
                cv2.line(tmp_img, (x, y), (x_t, y_t), color, thickness=2)
                cv2.line(all_img, (x, y), (x_t, y_t), color, thickness=2)

            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_HSV2BGR)

            cv2.imwrite(fname + f"_{person}_{other_person}_{type_to_draw}.png", tmp_img)
        all_img = cv2.cvtColor(all_img, cv2.COLOR_HSV2BGR)
        cv2.imwrite(fname + f"_{person}_all_{type_to_draw}.png", all_img)


def pred_to_ann(scoremaps, tags, joint_det, joint_scores, edge_index, pred, img_shape, input_size, img_id, cc_method,
                scaling_type, min_scale, adjustment, th, preds_classes, with_refine, score_map_scores, with_filter,
                scoring_method="default", fill_mean=True, num_joints=17):
    if (score_map_scores > 0.1).sum() < 1:
        return None
    true_positive_idx = joint_scores > th
    edge_index, pred = subgraph(true_positive_idx, edge_index, pred)
    if edge_index.shape[1] != 0:
        # pred[joint_det[edge_index[0, :], 2] == joint_det[
        #     edge_index[1, :], 2]] = 0.0  # set edge predictions of same types to zero
        persons_pred, _, _ = pred_to_person(joint_det, joint_scores, edge_index, pred, preds_classes, cc_method, num_joints=num_joints)
    else:
        return None
    # persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))
    if len(persons_pred.shape) == 1:  # this means none persons were detected
        return None
        # persons_pred = np.zeros([1, 17, 3])

    if with_filter:
        max_scores = persons_pred[:, :, 2].max(axis=1)
        keep = max_scores > 0.25
        persons_pred = persons_pred[keep]
        if persons_pred.shape[0] == 0:
            return None
    if fill_mean:
        for i in range(len(persons_pred)):
            persons_pred[i, persons_pred[i, :, 2] == 0, :2] = persons_pred[i, persons_pred[i, :, 2] != 0, :2].mean(axis=0)

    if with_refine and persons_pred[0, :, 2].sum() != 0:
        tags = tags.cpu().numpy()
        scoremaps = scoremaps.cpu().numpy()
        persons_pred = refine(scoremaps, tags, persons_pred)
    if adjustment:
        persons_pred = adjust(persons_pred, scoremaps)
    persons_pred_orig = reverse_affine_map(persons_pred.copy(), img_shape, input_size, scaling_type=scaling_type,
                                           min_scale=min_scale)

    if scoring_method == "default":
        ann = gen_ann_format(persons_pred_orig, img_id)
    elif scoring_method == "mean":
        ann = gen_ann_format_mean(persons_pred_orig, img_id)
    elif scoring_method == "correct":
        ann = gen_ann_format_correct(persons_pred_orig, img_id)
    else:
        raise NotImplementedError
    return ann


def mpn_match_by_tag(joint_det, tag_k, scores, params):
    """

    :param joint_det: (N, 3)
    :param tags: (N, D)
    :param scores: (N)
    :param params:
    :return:
    """
    from munkres import Munkres
    def py_max_match(scores):
        m = Munkres()
        tmp = m.compute(scores)
        tmp = np.array(tmp).astype(np.int32)
        return tmp

    tag_k, loc_k, val_k = tag_k, joint_det[:, :2], scores
    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[1]))

    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]
        select = joint_det[:, 2] == idx

        tags = tag_k[select]
        joints = np.concatenate(
            (loc_k[select], val_k[select, None], tags), 1
        )
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]
        joints = joints[mask]

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            if params.ignore_too_much \
                    and len(grouped_keys) == params.max_num_people:
                continue

            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (
                        diff_normed,
                        np.zeros((num_added, num_added-num_grouped))+1e10
                    ),
                    axis=1
                )

            pairs = py_max_match(diff_normed)
            for row, col in pairs:
                if (
                        row < num_added
                        and col < num_grouped
                        and diff_saved[row][col] < params.tag_threshold
                ):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
    return ans