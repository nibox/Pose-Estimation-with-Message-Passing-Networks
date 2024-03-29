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



_C.MODEL = CN()
_C.MODEL.KP = "hrnet"
_C.MODEL.PRETRAINED = ""  # means the path were the trained model is saved to and where the trained model is loaded from
# _C.MODEL.WITH_FLIP_KERNEL = False   # did not work but i wont remove it in case
_C.MODEL.FEATURE_GATHER_KERNEL = 3
_C.MODEL.FEATURE_GATHER_PADDING = 1
# _C.MODEL.WITH_LOCATION_REFINE = False  # did not work but wont remove it in case
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.NAME = ["edge_loss"]
# _C.MODEL.LOSS.LOSS_WEIGHTS = [1.0]
_C.MODEL.LOSS.NODE_WEIGHT = 1.0
_C.MODEL.LOSS.EDGE_WEIGHT = 1.0
_C.MODEL.LOSS.CLASS_WEIGHT = 1.0
_C.MODEL.LOSS.TAG_WEIGHT = 1.0  # only for tags predicted by the mpn

_C.MODEL.LOSS.SYNC_TAGS = False
_C.MODEL.LOSS.SYNC_GT_TAGS = False
_C.MODEL.LOSS.USE_FOCAL = True
_C.MODEL.LOSS.EDGE_WITH_LOGITS = True
_C.MODEL.LOSS.NODE_USE_FOCAL = True
_C.MODEL.LOSS.FOCAL_ALPHA = 1.0
_C.MODEL.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.LOSS.NODE_BCE_POS_WEIGHT = 1.0
_C.MODEL.LOSS.EDGE_BCE_POS_WEIGHT = 1.0
_C.MODEL.LOSS.INCLUDE_BORDERING_NODES = False  # include connections to adjacent nodes of selected nodes for loss
_C.MODEL.AUX_STEPS = 1
_C.MODEL.KP_OUTPUT_DIM = 32  # 256 for hg, 32 for HR
# common params for Hourglass keypoint detector (for reproducibility of the old experiments)
_C.MODEL.HG = CN()
_C.MODEL.HG.NAME = "hourglass"
_C.MODEL.HG.PRETRAINED = "../PretrainedModels/pretrained/checkpoint.pth.tar"
_C.MODEL.HG.NSTACK = 4
_C.MODEL.HG.INPUT_DIM = 256
_C.MODEL.HG.OUTPUT_DIM = 68

# common params for HRNet keypoint detector
_C.MODEL.HRNET = CN()
_C.MODEL.HRNET.NAME = 'pose_multi_resolution_net_v16'
_C.MODEL.HRNET.PRETRAINED = '../PretrainedModels/pose_higher_hrnet_w32_512.pth'
_C.MODEL.HRNET.NUM_JOINTS = 17
_C.MODEL.HRNET.TAG_PER_JOINT = True
_C.MODEL.HRNET.SYNC_BN = False
_C.MODEL.HRNET.INPUT_SIZE = 512
_C.MODEL.HRNET.OUTPUT_SIZE = [128, 256]
_C.MODEL.HRNET.FEATURE_FUSION = "avg"
_C.MODEL.HRNET.SCOREMAP_MODE = "avg"


# adapted from yaml file
_C.MODEL.HRNET.LOSS = CN()
_C.MODEL.HRNET.LOSS.NUM_STAGES = 2
_C.MODEL.HRNET.LOSS.WITH_HEATMAPS_LOSS = (True, True)
_C.MODEL.HRNET.LOSS.HEATMAPS_LOSS_FACTOR = (1.0, 1.0)
_C.MODEL.HRNET.LOSS.WITH_AE_LOSS = (True, False)
_C.MODEL.HRNET.LOSS.AE_LOSS_TYPE = 'exp'
_C.MODEL.HRNET.LOSS.PUSH_LOSS_FACTOR = (0.001, 0.001)
_C.MODEL.HRNET.LOSS.PULL_LOSS_FACTOR = (0.001, 0.001)

_C.MODEL.HRNET.EXTRA = CN()
_C.MODEL.HRNET.EXTRA.PRETRAINED_LAYERS = ['*']
_C.MODEL.HRNET.EXTRA.STEM_INPLANES = 64
_C.MODEL.HRNET.EXTRA.FINAL_CONV_KERNEL = 1

_C.MODEL.HRNET.EXTRA.STAGE2 = CN()
_C.MODEL.HRNET.EXTRA.STAGE2.NUM_MODULES = 1
_C.MODEL.HRNET.EXTRA.STAGE2.NUM_BRANCHES = 2
_C.MODEL.HRNET.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.HRNET.EXTRA.STAGE2.NUM_CHANNELS = [32, 64]
_C.MODEL.HRNET.EXTRA.STAGE2.BLOCK = 'BASIC'
_C.MODEL.HRNET.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

_C.MODEL.HRNET.EXTRA.STAGE3 = CN()
_C.MODEL.HRNET.EXTRA.STAGE3.NUM_MODULES = 4
_C.MODEL.HRNET.EXTRA.STAGE3.NUM_BRANCHES = 3
_C.MODEL.HRNET.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.HRNET.EXTRA.STAGE3.NUM_CHANNELS = [32, 64, 128]
_C.MODEL.HRNET.EXTRA.STAGE3.BLOCK = 'BASIC'
_C.MODEL.HRNET.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

_C.MODEL.HRNET.EXTRA.STAGE4 = CN()
_C.MODEL.HRNET.EXTRA.STAGE4.NUM_MODULES = 3
_C.MODEL.HRNET.EXTRA.STAGE4.NUM_BRANCHES = 4
_C.MODEL.HRNET.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.HRNET.EXTRA.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
_C.MODEL.HRNET.EXTRA.STAGE4.BLOCK = 'BASIC'
_C.MODEL.HRNET.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

_C.MODEL.HRNET.EXTRA.DECONV = CN()
_C.MODEL.HRNET.EXTRA.DECONV.NUM_DECONVS = 1
_C.MODEL.HRNET.EXTRA.DECONV.NUM_CHANNELS = [32]
_C.MODEL.HRNET.EXTRA.DECONV.NUM_BASIC_BLOCKS = 4
_C.MODEL.HRNET.EXTRA.DECONV.KERNEL_SIZE = [4]
_C.MODEL.HRNET.EXTRA.DECONV.CAT_OUTPUT = [True]

_C.MODEL.MPN = CN(new_allowed=True)
_C.MODEL.MPN.NODE_TYPE_SUMMARY = "not"
_C.MODEL.MPN.NAME = "VanillaMPN"
# _C.MODEL.MPN.PRETRAINED = ""
_C.MODEL.MPN.STEPS = 10
_C.MODEL.MPN.EDGE_MLP = "agnostic"
_C.MODEL.MPN.NODE_INPUT_DIM = 128
_C.MODEL.MPN.AGGR_TYPE = "agnostic"
_C.MODEL.MPN.EDGE_INPUT_DIM = 17 + 2
_C.MODEL.MPN.EDGE_FEATURE_DIM = 64
_C.MODEL.MPN.EDGE_FEATURE_HIDDEN = 64
_C.MODEL.MPN.NODE_FEATURE_DIM = 64
_C.MODEL.MPN.USE_NODE_UPDATE_MLP = False
_C.MODEL.MPN.NODE_EMB = CN(new_allowed=True)
_C.MODEL.MPN.EDGE_EMB = CN(new_allowed=True)
_C.MODEL.MPN.CLASS = CN(new_allowed=True)
_C.MODEL.MPN.BN = True
_C.MODEL.MPN.AGGR = "max"
_C.MODEL.MPN.AGGR_SUB = "None"
_C.MODEL.MPN.UPDATE_TYPE = "mlp"  # only used for type aware mpn
_C.MODEL.MPN.SKIP = False
_C.MODEL.MPN.AUX_LOSS_STEPS = 0  # 0 means only the last prediction is used for loss
_C.MODEL.MPN.DROP_FEATURE = ""
_C.MODEL.MPN.EDGE_STEPS = 0
_C.MODEL.MPN.LATE_FUSION_POS = False
_C.MODEL.MPN.NUM_JOINTS = 17

# _C.MODEL.REFINE = CN(new_allowed=True)


# configuration for the Graph Constructor
_C.MODEL.GC = CN()
_C.MODEL.GC.NAME = "NaiveGraphConstructor"
_C.MODEL.GC.POOL_KERNEL_SIZE = 3
_C.MODEL.GC.CHEAT = False
_C.MODEL.GC.USE_GT = False
_C.MODEL.GC.USE_NEIGHBOURS = False
_C.MODEL.GC.EDGE_LABEL_METHOD = 4
_C.MODEL.GC.MASK_CROWDS = True
_C.MODEL.GC.DETECT_THRESHOLD = 0.005
_C.MODEL.GC.WITH_BACKGROUND = False
_C.MODEL.GC.HYBRID_K = 5
_C.MODEL.GC.MATCHING_RADIUS = 0.1
_C.MODEL.GC.INCLUSION_RADIUS = 0.75
_C.MODEL.GC.GRAPH_TYPE = "knn"
_C.MODEL.GC.CC_METHOD = "GAEC"
_C.MODEL.GC.NORM_NODE_DISTANCE = False
_C.MODEL.GC.IMAGE_CENTRIC_SAMPLING = False
_C.MODEL.GC.NODE_MATCHING_RADIUS = 0.5
_C.MODEL.GC.NODE_INCLUSION_RADIUS = 0.7
_C.MODEL.GC.WEIGHT_CLASS_LOSS = False
_C.MODEL.GC.EDGE_FEATURES_TO_USE = ["position", "connection_type"]
_C.MODEL.GC.NODE_DROPOUT = 0.0

# DATASET and preprocessing related params
_C.DATASET = CN()
_C.DATASET.ROOT = '../../storage/user/kistern/coco'
_C.DATASET.DATASET = 'coco'
_C.DATASET.WITH_CENTER = False
_C.DATASET.MAX_NUM_PEOPLE = 30
_C.DATASET.NUM_JOINTS = 17
_C.DATASET.SCALING_TYPE = "short"
_C.DATASET.SIGMA = 2
_C.DATASET.HEAT_GENERATOR = "default"
# training data augmentation
_C.DATASET.MAX_ROTATION = 30
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.25
_C.DATASET.SCALE_TYPE = 'short'
_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.INPUT_SIZE = 512
_C.DATASET.OUTPUT_SIZE = [128, 256]
_C.DATASET.FLIP = 0.5


# parameters for Upper Bound model (to calculate the upper bound of the constructed labels)
_C.UB = CN()
_C.UB.KP = "hrnet"
_C.UB.GC = "NaiveGraphConstructor"
_C.UB.NUM_EVAL = 500
_C.UB.ADJUST = True
_C.UB.SPLIT = "coco_17_mini"
_C.UB.REFINE = False

_C.TEST = CN()
_C.TEST.SPLIT = "coco_17_mini"
_C.TEST.NUM_EVAL = 500
_C.TEST.ADJUST = True
_C.TEST.WITH_REFINE = False
_C.TEST.REFINE_COMP = False
_C.TEST.FILL_MEAN = True  # in case refine is not active this estimates missing keypoint by computing the average kpt

# TEST stuff for hrnet
_C.TEST.WITH_HEATMAPS = [True, True]
_C.TEST.WITH_AE = [True, False]
_C.TEST.SCALE_FACTOR = [0.5, 1.0, 2.0]
_C.TEST.FLIP_TEST = True
_C.TEST.FLIP_AND_REARANGE = True
_C.TEST.PROJECT2IMAGE = True
_C.TEST.WITH_POSE_FILTER = False
_C.TEST.SCORING = "correct"

# train
_C.TRAIN = CN()

_C.TRAIN.SPLIT = "coco_17_mini"
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [60, 150]
_C.TRAIN.LR = 3e-4
_C.TRAIN.KP_LR = 1e-5
# _C.TRAIN.REG_LR = 3e-4
_C.TRAIN.W_DECAY = 0.0
_C.TRAIN.KP_W_DECAY = 0.0

_C.TRAIN.START_EPOCH = 0
_C.TRAIN.END_EPOCH = 100
_C.TRAIN.CONTINUE = ""

_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.SPLIT_OPTIMIZER = True
_C.TRAIN.END_TO_END = False
_C.TRAIN.FINETUNE = False
_C.TRAIN.LOSS_REDUCTION = "mean"
_C.TRAIN.USE_LABEL_MASK = True
_C.TRAIN.USE_BATCH_INDEX = False
_C.TRAIN.FREEZE_BN = True
_C.TRAIN.KP_FREEZE_MODE = "complete"
_C.TRAIN.WITH_AE_LOSS = [False, False]


def get_config():
    return _C.clone()


def update_config(cfg, config_file):
    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

def update_config_command(cfg, opts):
    cfg.defrost()
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
