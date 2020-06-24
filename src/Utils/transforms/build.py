# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import transforms as T
import torchvision


def transforms_hr_train(config):
    assert isinstance(config.DATASET.OUTPUT_SIZE, (list, tuple)), 'DATASET.OUTPUT_SIZE should be list or tuple'

    max_rotation = config.DATASET.MAX_ROTATION
    min_scale = config.DATASET.MIN_SCALE
    max_scale = config.DATASET.MAX_SCALE
    max_translate = config.DATASET.MAX_TRANSLATE
    input_size = config.DATASET.INPUT_SIZE
    output_size = config.DATASET.OUTPUT_SIZE
    flip = config.DATASET.FLIP
    scale_type = config.DATASET.SCALING_TYPE

    coco_flip_index = T.FLIP_CONFIG["COCO"]

    transforms = T.Compose(
        [
            T.RandomAffineTransform(
                input_size,
                output_size,
                max_rotation,
                min_scale,
                max_scale,
                scale_type,
                max_translate
            ),
            T.RandomHorizontalFlip(coco_flip_index, output_size, flip),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    transforms_inv = torchvision.transforms.Compose([T.NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                     torchvision.transforms.ToPILImage()])

    return transforms, transforms_inv


def transforms_hg_eval(config):

    max_rotation = 0
    min_scale = 1
    max_scale = 1
    max_translate = 0
    input_size = 512
    output_size = 128
    scale_type = config.DATASET.SCALING_TYPE
    assert scale_type == "long"

    transforms = T.Compose(
        [
            T.RandomAffineTransform(
                input_size,
                output_size,
                max_rotation,
                min_scale,
                max_scale,
                scale_type,
                max_translate
            ),
            T.ToTensor()
        ]
    )

    return transforms


def transforms_hr_eval(config):

    input_size = 512
    scale_type = config.DATASET.SCALING_TYPE

    affine_transform = T.HRNetMineTransformation if scale_type == "short" else T.HRNetEvalTransform

    transforms = T.Compose(
        [
            affine_transform(
                input_size,
                scale_type,
            ),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    transforms_inv = torchvision.transforms.Compose([T.NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                     torchvision.transforms.ToPILImage()])

    return transforms, transforms_inv
