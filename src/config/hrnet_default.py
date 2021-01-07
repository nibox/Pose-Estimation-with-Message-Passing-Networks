
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.VERBOSE = True
_C.DIST_BACKEND = 'nccl'
_C.MULTIPROCESSING_DISTRIBUTED = True

# FP16 training params
_C.FP16 = CN()
_C.FP16.ENABLED = False
_C.FP16.STATIC_LOSS_SCALE = 1.0
_C.FP16.DYNAMIC_LOSS_SCALE = False

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'pose_multi_resolution_net_v16'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.SYNC_BN = False

_C.LOSS = CN()
_C.LOSS.NUM_STAGES = 1
_C.LOSS.WITH_HEATMAPS_LOSS = (True,)
_C.LOSS.HEATMAPS_LOSS_FACTOR = (1.0,)
_C.LOSS.WITH_AE_LOSS = (True,)
_C.LOSS.AE_LOSS_TYPE = 'max'
_C.LOSS.PUSH_LOSS_FACTOR = (0.001,)
_C.LOSS.PULL_LOSS_FACTOR = (0.001,)

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'coco_kpt'
_C.DATASET.DATASET_TEST = 'coco'
_C.DATASET.NUM_JOINTS = 17
_C.DATASET.MAX_NUM_PEOPLE = 30
_C.DATASET.TRAIN = 'train2017'
_C.DATASET.TEST = 'val2017'
_C.DATASET.DATA_FORMAT = 'jpg'

# training data augmentation
_C.DATASET.MAX_ROTATION = 30
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.25
_C.DATASET.SCALING_TYPE = 'short'
_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.INPUT_SIZE = 512
_C.DATASET.OUTPUT_SIZE = [128, 256, 512]
_C.DATASET.FLIP = 0.5

# heatmap generator (default is OUTPUT_SIZE/64)
_C.DATASET.SIGMA = -1
_C.DATASET.SCALE_AWARE_SIGMA = False
_C.DATASET.BASE_SIZE = 256.0
_C.DATASET.BASE_SIGMA = 2.0
_C.DATASET.INT_SIGMA = False

_C.DATASET.WITH_CENTER = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.IMAGES_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
# _C.TEST.BATCH_SIZE = 32
_C.TEST.IMAGES_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.ADJUST = True
_C.TEST.REFINE = True
_C.TEST.SCALE_FACTOR = [1]
# group
_C.TEST.SPLIT = "coco_17_full"
_C.TEST.DETECTION_THRESHOLD = 0.2
_C.TEST.TAG_THRESHOLD = 1.
_C.TEST.USE_DETECTION_VAL = True
_C.TEST.IGNORE_TOO_MUCH = False
_C.TEST.MODEL_FILE = ''
_C.TEST.IGNORE_CENTER = True
_C.TEST.NMS_KERNEL = 3
_C.TEST.NMS_PADDING = 1
_C.TEST.PROJECT2IMAGE = False

_C.TEST.WITH_HEATMAPS = (True,)
_C.TEST.WITH_AE = (True,)

_C.TEST.LOG_PROGRESS = False

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = True
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = True
_C.DEBUG.SAVE_HEATMAPS_PRED = True
_C.DEBUG.SAVE_TAGMAPS_PRED = True

def get_hrnet_config():
    return _C.clone()

