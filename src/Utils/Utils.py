import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


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


def load_model(path, model_class, config, device, pretrained_path=None):

    assert not (path is not None and pretrained_path is not None)
    def rename_key(key):
        # assume structure is model.module.REAL_NAME
        return ".".join(key.split(".")[2:])

    #model = hourglass.PoseNet(kwargs["nstack"], kwargs["input_dim"], kwargs["output_size"])
    model = model_class(config)
    if path is not None:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict["model_state_dict"])
    elif pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=device)
        state_dict_new = {rename_key(k): v for k, v in state_dict["state_dict"].items()}
        model.backbone.load_state_dict(state_dict_new)

    return model


def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot(np.concatenate((kpt, kpt[:, 0:1] * 0 + 1), axis=1), mat.T).reshape(shape)


def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets, mask=None):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if mask is not None:
            F_loss = F_loss * mask

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def draw_detection(img, joint_det, joint_gt, fname=None):
    """
    :param img: torcg.tensor. image
    :param joint_det: shape: (num_joints, 2) list of xy positions of detected joints (without classes or clustering)
    :param fname: optional - file name of image to save. If None the image is show with plt.show
    :return:
    """

    img = to_numpy(img)
    img = img * 255.0
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    colors_joints = np.arange(0, 179, np.ceil(179 / 17), dtype=np.float)

    for i in range(len(joint_det)):
        # scale up to 512 x 512
        scale = 512.0 / 128.0
        x, y = int(joint_det[i, 0] * scale), int(joint_det[i, 1] * scale)
        type = joint_det[i, 2]
        if type != -1:
            color = (colors_joints[type], 255, 255)
            cv2.circle(img, (x, y), 2, color, -1)
    for person in range(len(joint_gt)):
        if np.sum(joint_gt[person]) > 0.0:
            for i in range(len(joint_gt[person])):
                # scale up to 512 x 512
                scale = 512.0 / 128.0
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
    # image to 8bit hsv (i dont know what hsv values opencv expects in 32bit case=)
    img = img * 255.0
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    assert len(colors) == len(persons)
    for person in range(len(persons)):
        scale = 512.0 / 128.0
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
                cv2.circle(img, (x_1, y_1), 2, color, -1)
                cv2.line(img, (x_1, y_1), (center_joint[0], center_joint[1]), color)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    fig = plt.figure()
    plt.imshow(img)
    if fname is not None:
        plt.savefig(fig=fig, fname=fname)
        plt.close(fig)
    else:
        raise NotImplementedError


def draw_clusters(img: [torch.tensor, np.array], joints, joint_connections, fname=None):
    """

    :param img:
    :param persons: (N,17,3) array containing the person in the image img. Detected or ground truth does not matter.
    :param fname: If not none an image will be saved under this name , otherwise the image will be displayed
    :return:
    """
    img = to_numpy(img)
    assert img.shape[0] == 512
    assert img.shape[1] == 512

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
    img = img * 255.0
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)

    color_i = 0
    scale = 512.0 / 128.0
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
