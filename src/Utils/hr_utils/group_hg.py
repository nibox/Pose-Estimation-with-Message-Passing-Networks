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
    tmp = m.compute(-scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp

def match_by_tag_2(inp, params, pad=False):
    tag_k, loc_k, val_k = inp
    assert type(params) is Params
    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]))

    dic = {}
    dic2 = {}
    for i in range(params.num_joints):
        ptIdx = params.joint_order[i]

        tags = tag_k[ptIdx]
        joints = np.concatenate((loc_k[ptIdx], val_k[ptIdx, :, None], tags), 1)
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]
        joints = joints[mask]
        if i == 0 or len(dic) == 0:
            for tag, joint in zip(tags, joints):
                dic.setdefault(tag[0], np.copy(default_))[ptIdx] = joint
                dic2[tag[0]] = [tag]
        else:
            actualTags = list(dic.keys())[:params.max_num_people]
            actualTags_key = actualTags
            actualTags = [np.mean(dic2[i], axis=0) for i in actualTags]

            if params.ignore_too_much and len(actualTags) == params.max_num_people:
                continue
            diff = ((joints[:, None, 3:] - np.array(actualTags)[None, :, :]) ** 2).mean(axis=2) ** 0.5
            if diff.shape[0] == 0:
                continue

            diff2 = np.copy(diff)

            if params.use_detection_val:
                diff = np.round(diff) * 100 - joints[:, 2:3]

            if diff.shape[0] > diff.shape[1]:
                diff = np.concatenate((diff, np.zeros((diff.shape[0], diff.shape[0] - diff.shape[1])) + 1e10),
                                      axis=1)

            pairs = py_max_match(-diff)  ##get minimal matching
            for row, col in pairs:
                if row < diff2.shape[0] and col < diff2.shape[1] and diff2[row][col] < params.tag_threshold:
                    dic[actualTags_key[col]][ptIdx] = joints[row]
                    dic2[actualTags_key[col]].append(tags[row])
                else:
                    key = tags[row][0]
                    dic.setdefault(key, np.copy(default_))[ptIdx] = joints[row]
                    dic2[key] = [tags[row]]

    ans = np.array([dic[i] for i in dic])
    if len(ans) == 0:
        print(ans)
    if pad:
        num = len(ans)
        if num < params.max_num_people:
            padding = np.zeros((params.max_num_people - num, params.num_parts, default_.shape[1]))
            if num > 0:
                ans = np.concatenate((ans, padding), axis=0)
            else:
                ans = padding
        return np.array(ans[:params.max_num_people]).astype(np.float32)
    else:
        return np.array(ans).astype(np.float32)


def match_by_tag_1(inp, params):
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


class Params(object):
    def __init__(self):
        self.num_joints = 17
        self.max_num_people = 30

        self.detection_threshold = 0.1
        self.tag_threshold = 1.0
        self.use_detection_val = False
        self.ignore_too_much = False


        self.joint_order = [
            i-1 for i in [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
        ]


class HeatmapParserHG(object):
    def __init__(self, cfg):
        self.params = Params()
        self.tag_per_joint = True
        self.pool = torch.nn.MaxPool2d(
            3, 1, 1
        )

    def nms(self, det):
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def match(self, tag_k, loc_k, val_k):
        match = lambda x: match_by_tag_1(x, self.params)
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

    def parse(self, det, tag, adjust=True, refine=True):
        ans = self.match(**self.top_k(det, tag))

        if adjust:
            ans = self.adjust(ans, det)

        scores = [i[:, 2].mean() for i in ans[0]]

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


def refine(det, tag, keypoints, adjust=True):
    """
    det shape: (joint, h, w)
    tag shape: (joint, h, w, ..)
    keypoints: (joint, x y score tag)
    Given initial keypoint predictions, we identify missing joints
    """

    if len(tag.shape) == 3:
        tag = tag[:,:,:,None]

    tags = []
    for i in range(keypoints.shape[0]):
        if keypoints[i, 2] > 0:
            y, x = keypoints[i][:2].astype(np.int32)
            tags.append(tag[i, x, y])

    prev_tag = np.mean(tags, axis = 0)
    ans = []

    for i in range(keypoints.shape[0]):
        tmp = det[i, :, :]
        tt = (((tag[i, :, :] - prev_tag[None, None, :])**2).sum(axis = 2)**0.5 )
        tmp2 = tmp - np.round(tt)

        x, y = np.unravel_index( np.argmax(tmp2), tmp.shape )
        val = tmp[x, y]
        if adjust:
            xx = x
            yy = y
            x += 0.5
            y += 0.5

            if tmp[xx, min(yy+1, det.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
                y+=0.25
            else:
                y-=0.25

            if tmp[min(xx+1, det.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
                x+=0.25
            else:
                x-=0.25

        x, y = np.array([y,x])
        ans.append((x, y, val))
    ans = np.array(ans)

    if ans is not None:
        for i in range(17):
            if ans[i, 2]>0 and keypoints[i, 2]==0:
                keypoints[i, :2] = ans[i, :2]
                keypoints[i, 2] = 1

    return keypoints

class HeatmapParserHG2():
    def __init__(self, detection_val=0.03, tag_val=1.):
        from torch import nn
        self.pool = nn.MaxPool2d(3, 1, 1)
        param = Params()
        param.detection_threshold = 0.1
        param.tag_threshold = tag_val
        param.ignore_too_much = True
        param.max_num_people = 30
        param.use_detection_val = True
        self.param = param

    def nms(self, det):
        # suppose det is a tensor
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def match(self, tag_k, loc_k, val_k):
        match = lambda x:match_by_tag_2(x, self.param)
        return list(map(match, zip(tag_k, loc_k, val_k)))

    def calc(self, det, tag):
        # det = torch.autograd.Variable(torch.Tensor(det), volatile=True)
        # tag = torch.autograd.Variable(torch.Tensor(tag), volatile=True)

        det = self.nms(det)
        h = det.size()[2]
        w = det.size()[3]
        det = det.view(det.size()[0], det.size()[1], -1)
        tag = tag.view(tag.size()[0], tag.size()[1], det.size()[2], -1)
        # ind (1, 17, 30)
        # val (1, 17, 128*128)
        # tag (1, 17, 128*128, -1)
        val_k, ind = det.topk(self.param.max_num_people, dim=2)
        tag_k = torch.stack([torch.gather(tag[:,:,:,i], 2, ind) for i in range(tag.size()[3])], dim=3)

        x = ind % w
        y = (ind // w).long()
        ind_k = torch.stack((x, y), dim=3)
        ans = {'tag_k': tag_k, 'loc_k': ind_k, 'val_k': val_k}
        return {key:ans[key].cpu().data.numpy() for key in ans}

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2]>0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        #print(batch_id, joint_id, det[batch_id].shape)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
                            y+=0.25
                        else:
                            y-=0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
                            x+=0.25
                        else:
                            x-=0.25
                        ans[batch_id][people_id, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    def parse(self, det, tag, adjust=True):
        ans = self.match(**self.calc(det, tag))
        scores = [i[:, 2].mean() for i in ans[0]]
        if adjust:
            ans = self.adjust(ans, det)
        tag = tag[0].cpu().numpy()
        det = det[0].cpu().numpy()
        for i in range(len(ans[0])):
            ans[0][i] = refine(det, tag, ans[0][i])
        return ans, scores
