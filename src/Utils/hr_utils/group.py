# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Some code is from https://github.com/princeton-vl/pose-ae-train/blob/454d4ba113bbb9775d4dc259ef5e6c07c2ceed54/utils/group.py
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from munkres import Munkres
import numpy as np
import torch
from torch_geometric import utils as gutils

from Utils import Graph, adjust, refine
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph


def cat_unique(tensor_1, tensor_2):
    # tensor_1/2 have shape (N, D) D is dim of vector and N is number of vectors
    assert len(tensor_1.shape) == 2 and len(tensor_2.shape) == 2
    vector_dim = tensor_1.shape[1]
    unique_elements = torch.eq(tensor_1.unsqueeze(1), tensor_2)
    unique_elements = unique_elements.int().sum(dim=2)
    unique_elements = unique_elements == vector_dim
    unique_elements = unique_elements.sum(dim=0)
    unique_elements = unique_elements == 0

    return torch.cat([tensor_1, tensor_2[unique_elements]], 0)


def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp


def match_by_tag(inp, params):
    assert isinstance(params, Params), 'params should be class Params()'

    tag_k, loc_k, val_k = inp
    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]))

    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]

        tags = tag_k[idx]
        joints = np.concatenate(
            (loc_k[idx], val_k[idx, :, None], tags), 1
        )
        original = True
        if original:
            mask = joints[:, 2] > params.detection_threshold
        else:
            mask = joints[:, 2] > 0.1
            mask[:5] = True
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


class Params(object):
    def __init__(self, cfg):
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.max_num_people = cfg.DATASET.MAX_NUM_PEOPLE

        self.detection_threshold = 0.1
        self.tag_threshold = 1.0
        self.use_detection_val = True
        self.ignore_too_much = False

        self.joint_order = [
            i-1 for i in [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
        ]


class HeatmapParser(object):
    def __init__(self, cfg):
        self.params = Params(cfg)
        self.tag_per_joint = True
        self.pool = torch.nn.MaxPool2d(
            5, 1, 2
        )

    def nms(self, det):
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def match(self, tag_k, loc_k, val_k):
        match = lambda x: match_by_tag(x, self.params)
        return list(map(match, zip(tag_k, loc_k, val_k)))

    def top_k(self, det, tag):
        # det = torch.Tensor(det, requires_grad=False)
        # tag = torch.Tensor(tag, requires_grad=False)

        det = self.nms(det)
        num_images = det.size(0)
        num_joints = det.size(1)
        h = det.size(2)
        w = det.size(3)
        det = det.view(num_images, num_joints, -1)
        val_k, ind = det.topk(self.params.max_num_people, dim=2)

        scoremap_zero = torch.where(det < 0.1, torch.zeros_like(det), det)
        t = scoremap_zero.nonzero()
        tag = tag.view(tag.size(0), tag.size(1), w*h, -1)
        if not self.tag_per_joint:
            tag = tag.expand(-1, self.params.num_joints, -1, -1)

        tag_k = torch.stack(
            [
                torch.gather(tag[:, :, :, i], 2, ind)
                for i in range(tag.size(3))
            ],
            dim=3
        )

        x = ind % w
        y = (ind // w).long()

        ind_k = torch.stack((x, y), dim=3)

        ans = {
            'tag_k': tag_k.cpu().numpy(),
            'loc_k': ind_k.cpu().numpy(),
            'val_k': val_k.cpu().numpy()
        }

        return ans

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        #print(batch_id, joint_id, det[batch_id].shape)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)] > tmp[xx, max(yy-1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy] > tmp[max(0, xx-1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    def refine(self, det, tag, keypoints):
        """
        Given initial keypoint predictions, we identify missing joints
        :param det: numpy.ndarray of size (17, 128, 128)
        :param tag: numpy.ndarray of size (17, 128, 128) if not flip
        :param keypoints: numpy.ndarray of size (17, 4) if not flip, last dim is (x, y, det score, tag score)
        :return: 
        """
        if len(tag.shape) == 3:
            # tag shape: (17, 128, 128, 1)
            tag = tag[:, :, :, None]

        tags = []
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(np.int32)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        ans = []

        for i in range(keypoints.shape[0]):
            # score of joints i at all position
            tmp = det[i, :, :]
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
            for i in range(det.shape[0]):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                # if ans[i, 2] > 0.01 and keypoints[i, 2] == 0:
                    keypoints[i, :2] = ans[i, :2]
                    keypoints[i, 2] = ans[i, 2]

        return keypoints

    def parse(self, det, tag, adjust=True, refine=True, scoring="default"):
        ans = self.match(**self.top_k(det, tag))

        if adjust:
            ans = self.adjust(ans, det)

        if scoring == "default":
            scores = [i[:, 2].mean() for i in ans[0]]
        elif scoring == "mean":
            scores = [i[i[:, 2] > 0.009, 2].mean() for i in ans[0]]

        if refine:
            ans = ans[0]
            # for every detected person
            for i in range(len(ans)):
                det_numpy = det[0].cpu().numpy()
                tag_numpy = tag[0].cpu().numpy()
                if not self.tag_per_joint:
                    tag_numpy = np.tile(
                        tag_numpy, (self.params.num_joints, 1, 1, 1)
                    )
                ans[i] = self.refine(det_numpy, tag_numpy, ans[i])
            ans = [ans]

        return ans, scores


def cluster_cc(heatmaps, tagmaps, config):
    # heatmap shape: (joints, h, w)
    # tags: (joints, h, w, tag_dim)
    # extract detections
    from Utils import non_maximum_suppression
    heatmaps = non_maximum_suppression(heatmaps, threshold=0.1, pool_kernel=config.TEST.NMS_KERNEL) * heatmaps
    num_joints = heatmaps.shape[0]

    k = 50
    scoremap_shape = heatmaps.shape
    scores, indices = heatmaps.view(num_joints, -1).topk(k=k, dim=1)
    container = torch.zeros_like(heatmaps, device=heatmaps.device, dtype=torch.float).reshape(num_joints, -1)
    container[np.arange(0, num_joints).reshape(num_joints, 1), indices] = scores + 1e-10  #
    container = container.reshape(scoremap_shape)
    joint_idx_det, joint_y, joint_x = container.nonzero(as_tuple=True)

    scores = container[joint_idx_det, joint_y, joint_x]
    det = torch.stack([joint_x, joint_y, joint_idx_det], 1)
    valid = scores > 0.1
    scores = scores[valid]
    det = det[valid]
    tags = tagmaps[det[:, 2], det[:, 1], det[:, 0]]
    # scores (N)
    # det (N, 3)

    # construct graph
    num_joints_det = len(det)
    if num_joints_det == 0:
        return [], []
    elif num_joints_det > 1:
        edge_index, _ = gutils.dense_to_sparse(torch.ones([num_joints_det, num_joints_det], dtype=torch.long))
        edge_index = gutils.to_undirected(edge_index, len(det))
        edge_index, _ = gutils.remove_self_loops(edge_index)

        t_a = 1 # 1.8425  # based on "features for multi-target multi-camera tracking and re-identification"
        distance = (tags[edge_index[1]] - tags[edge_index[0]]).norm(p=None, dim=1,
                                                                                keepdim=True)
        edge_attr = torch.div(t_a - distance, t_a)
        # edge_attr = (edge_attr < 1.0).float()
        edge_attr[det[edge_index[0, :], 2] == det[edge_index[1, :], 2]] = 0.0  # set edge predictions of same types to zero

        # cluster graph
        test_graph = Graph(x=det, edge_index=edge_index, edge_attr=edge_attr)
        sol = cluster_graph(test_graph, "GAEC", complete=False)
    else:
        sol = np.array([[1]])
    #sparse_sol, _ = gutils.dense_to_sparse(torch.from_numpy(sol))
    # construct people
    joints = det.cpu().numpy()
    scores = scores.cpu().numpy()
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    graph = csr_matrix(sol)
    n_components, person_labels = connected_components(graph, directed=False, return_labels=True)
    persons = []
    for i in range(n_components):
        # check if cc has more than one node
        person_joints = joints[person_labels == i]
        person_scores = scores[person_labels == i]

        keypoints = np.zeros([num_joints, 3])
        for joint_type in range(num_joints):  # 17 different joint types
            # take the detected joints of a certain type
            select = person_joints[:, 2] == joint_type
            person_joint_for_type = person_joints[select]
            person_scores_for_type = person_scores[select]
            if len(person_joint_for_type) != 0:
                joint_idx = np.argmax(person_scores_for_type, axis=0)
                keypoints[joint_type, :2] = person_joint_for_type[joint_idx, :2]
                keypoints[joint_type, 2] = np.max(person_scores_for_type, axis=0)

        if (keypoints[:, 2] > 0).sum() > 0:
            # this step is actually really important for performance without refine !!
            keypoints[keypoints[:, 2] == 0, :2] = keypoints[keypoints[:, 2] != 0, :2].mean(axis=0)
            # keypoints[np.sum(keypoints, axis=1) != 0, 2] = 1
            persons.append(keypoints)

    persons = np.array(persons)

    heatmaps = heatmaps.cpu().numpy()
    tagmaps = tagmaps.cpu().numpy()

    persons_scores = [i[:, 2].mean() for i in persons]

    if config.TEST.ADJUST:
        persons = adjust(persons, heatmaps)
    if config.TEST.REFINE:
        persons = refine(heatmaps, tagmaps, persons)
    return persons, persons_scores