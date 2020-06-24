# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Nikita Kister
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from ..transformations import get_affine_transform, get_multi_scale_size, get_transform

FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, joints, factors):
        for t in self.transforms:
            image, mask, joints, factors = t(image, mask, joints, factors)
        return image, mask, joints, factors

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, image, mask, joints, factors):
        return F.to_tensor(image), mask, joints, factors


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, joints, factors):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask, joints, factors


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class RandomHorizontalFlip(object):
    def __init__(self, flip_index, output_size, prob=0.5):
        self.flip_index = flip_index
        self.prob = prob
        self.output_size = output_size if isinstance(output_size, list) \
            else [output_size]

    def __call__(self, image, mask, joints, factors):
        assert isinstance(mask, list)
        assert len(mask) == len(self.output_size)
        assert np.argmax(self.output_size) == len(self.output_size) - 1  # the largest output size should the last item
        # todo adapt to factors

        if random.random() < self.prob:
            image = image[:, ::-1] - np.zeros_like(image)
            for i, _output_size in enumerate(self.output_size):
                mask[i] = mask[i][:, ::-1] - np.zeros_like(mask[i])
                joints[i] = joints[i][:, self.flip_index]
                joints[i][:, :, 0] = _output_size - joints[i][:, :, 0] - 1

        return image, mask, joints, factors


class HRNetEvalTransform(object):
    def __init__(self,
                 input_size,
                 scale_type,
                 ):
        self.input_size = input_size
        self.output_size = [int(input_size / 4), int(input_size / 2)]

        self.scale_type = scale_type
        assert scale_type == "short"

    def _get_multi_scale_size(self, image, input_size, current_scale, min_scale):
        h, w, _ = image.shape
        center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])

        # calculate the size for min_scale
        min_input_size = int((min_scale * input_size + 63) // 64 * 64)
        if w < h:
            w_resized = int(min_input_size * current_scale / min_scale)
            h_resized = int(
                int((min_input_size / w * h + 63) // 64 * 64) * current_scale / min_scale
            )
            scale_w = w / 200.0
            scale_h = h_resized / w_resized * w / 200.0
        else:
            h_resized = int(min_input_size * current_scale / min_scale)
            w_resized = int(
                int((min_input_size / h * w + 63) // 64 * 64) * current_scale / min_scale
            )
            scale_h = h / 200.0
            scale_w = w_resized / h_resized * h / 200.0

        return (w_resized, h_resized), center, np.array([scale_w, scale_h])


    def _get_affine_transform(self, center, scale, output_size):

        def get_3rd_point(a, b):
            direct = a - b
            return b + np.array([-direct[1], direct[0]], dtype=np.float32)

        def get_dir(src_point, rot_rad):
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)

            src_result = [0, 0]
            src_result[0] = src_point[0] * cs - src_point[1] * sn
            src_result[1] = src_point[0] * sn + src_point[1] * cs

            return src_result

        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            print(scale)
            scale = np.array([scale, scale])

        scale_tmp = scale * 200.0
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        src_dir = get_dir([0, src_w * -0.5], 0.0)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def _affine_joints(self, joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    def _affine_factors(self, factors, mat):
        return factors * mat[0, 0] * mat[1, 1]

    def __call__(self, image, mask, joints, factors):
        assert isinstance(mask, list)
        assert len(mask) == len(self.output_size)
        assert len(mask) == len(joints)

        size_resized, center, scale = get_multi_scale_size(image.shape[0], image.shape[1], self.input_size, 1.0, 1.0)

        matrices = []
        factor = 2
        for i, _output_size in reversed(list(enumerate(self.output_size))):
            new_width = int(size_resized[0] / factor)
            new_height = int(size_resized[1] / factor)

            mat_output = self._get_affine_transform(
                center, scale, (new_width, new_height))[:2]
            mask[i] = cv2.warpAffine(
                (mask[i]*255).astype(np.uint8), mat_output,
                (new_width, new_height)
            ) / 255

            mask[i] = (mask[i] > 0.5).astype(np.float32)

            joints[i][:, :, 0:2] = self._affine_joints(
                joints[i][:, :, 0:2], mat_output
            )
            matrices.append(mat_output)
            factor = factor * 2

        factors = self._affine_factors(factors, matrices[0])

        mat_input = get_affine_transform(
            center, scale, size_resized)
        image = cv2.warpAffine(
            image, mat_input, size_resized
        )

        return image, mask, joints, factors


class HRNetMineTransformation(object):
    def __init__(self,
                 input_size,
                 scale_type,
                 ):
        self.input_size = input_size
        self.output_size = [int(input_size / 4), int(input_size / 2)]

        self.scale_type = scale_type
        assert scale_type == "short_mine"

    def _affine_joints(self, joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    def _affine_factors(self, factors, mat):
        return factors * mat[0, 0] * mat[1, 1]

    def __call__(self, image, mask, joints, factors):
        assert isinstance(mask, list)
        assert len(mask) == len(self.output_size)
        assert len(mask) == len(joints)

        size_resized, center, scale = get_multi_scale_size(image.shape[0], image.shape[1], self.input_size, 1.0, 1.0)

        matrices = []
        factor = 2
        for i, _output_size in reversed(list(enumerate(self.output_size))):
            new_width = int(size_resized[0] / (factor))
            new_height = int(size_resized[1] / (factor))

            mat_output = get_transform(
                center, scale, (new_width, new_height))[:2]
            mask[i] = cv2.warpAffine(
                (mask[i]*255).astype(np.uint8), mat_output,
                (new_width, new_height)
            ) / 255

            mask[i] = (mask[i] > 0.5).astype(np.float32)

            joints[i][:, :, 0:2] = self._affine_joints(
                joints[i][:, :, 0:2], mat_output
            )
            matrices.append(mat_output)
            factor = factor * 2

        factors = self._affine_factors(factors, matrices[0])

        mat_input = get_transform(
            center, scale, size_resized)[:2]
        image = cv2.warpAffine(
            image, mat_input, size_resized
        )

        return image, mask, joints, factors

class RandomAffineTransform(object):
    def __init__(self,
                 input_size,
                 output_size,
                 max_rotation,
                 min_scale,
                 max_scale,
                 scale_type,
                 max_translate):
        self.input_size = input_size
        self.output_size = output_size if isinstance(output_size, list) \
            else [output_size]

        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_type = scale_type
        self.max_translate = max_translate

    def _get_affine_matrix(self, center, scale, res, rot=0):
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
            t_mat[0, 2] = -res[1]/2
            t_mat[1, 2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def _affine_joints(self, joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    def _affine_factors(self, factors, mat):
        return factors * mat[0, 0] * mat[1, 1]

    def __call__(self, image, mask, joints, factors):
        assert isinstance(mask, list)
        assert len(mask) == len(self.output_size)
        assert len(mask) == len(joints)

        height, width = image.shape[:2]

        center = np.array((width/2, height/2))
        if self.scale_type == 'long':
            scale = max(height, width)/200
        elif self.scale_type == 'short':
            scale = min(height, width)/200
        else:
            raise ValueError('Unkonw scale type: {}'.format(self.scale_type))
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) \
                    + self.min_scale
        scale *= aug_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

        if self.max_translate > 0:
            dx = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            dy = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            center[0] += dx
            center[1] += dy

        for i, _output_size in enumerate(self.output_size):
            mat_output = self._get_affine_matrix(
                center, scale, (_output_size, _output_size), aug_rot
            )[:2]
            mask[i] = cv2.warpAffine(
                (mask[i]*255).astype(np.uint8), mat_output,
                (_output_size, _output_size)
            ) / 255
            mask[i] = (mask[i] > 0.5).astype(np.float32)

            joints[i][:, :, 0:2] = self._affine_joints(
                joints[i][:, :, 0:2], mat_output
            )
        factors = self._affine_factors(factors, mat_output)

        mat_input = self._get_affine_matrix(
            center, scale, (self.input_size, self.input_size), aug_rot
        )[:2]
        image = cv2.warpAffine(
            image, mat_input, (self.input_size, self.input_size)
        )

        return image, mask, joints, factors

