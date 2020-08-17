from ..Hourglass import PoseNet, hg_process_output
from ..HigherHRNet import get_pose_net, hr_process_output
from graph_constructor import get_graph_constructor

import torch
import torch.nn as nn

"""
default_config = {"backbone": PoseNet,
                  "backbone_config": hourglass_config,
                  "graph_constructor": NaiveGraphConstructor,
                  "cheat": True,
                  "use_gt": True,
                  "use_focal_loss": False,
                  "use_neighbours": False,
                  "edge_label_method": 1,
                  "mask_crowds": False,
                  "detect_threshold": 0.007,
                  "inclusion_radius": 5.0,
                  "matching_radius": None,
                  "mpn_graph_type": "knn"
                  }
"""


def load_back_bone(config):
    if config.UB.KP == "hourglass":
        hg_config = {"nstack": config.MODEL.HG.NSTACK,
                     "inp_dim": config.MODEL.HG.INPUT_DIM,
                     "oup_dim": config.MODEL.HG.OUTPUT_DIM}
        return PoseNet(**hg_config), hg_process_output
    elif config.UB.KP == "hrnet":
        return get_pose_net(config, False), hr_process_output


def get_upper_bound_model(config, device):

    def rename_key(key):
        # assume structure is model.module.REAL_NAME
        return ".".join(key.split(".")[2:])
    def rename_key_hr(key):
        return key[2:]

    model = UpperBoundModel(config)

    if config.UB.KP == "hrnet":
        state_dict = torch.load(config.MODEL.HRNET.PRETRAINED, map_location=device)
        if config.MODEL.HRNET.PRETRAINED != '../PretrainedModels/pose_higher_hrnet_w32_512.pth':
            state_dict = {rename_key_hr(k): v for k, v in state_dict.items()}

    elif config.UB.KP == "hourglass":
        state_dict = torch.load(config.MODEL.HG.PRETRAINED, map_location=device)
        state_dict = {rename_key(k): v for k, v in state_dict["state_dict"].items()}
    else:
        raise NotImplementedError
    model.backbone.load_state_dict(state_dict)
    model.to(device)

    return model

class UpperBoundModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backbone, self.process_output = load_back_bone(config)
        self.gc_config = config.MODEL.GC
        self.feature_gather = nn.AvgPool2d(3, 1, 1)
        self.scoremap_mode = config.MODEL.HRNET.SCOREMAP_MODE

    def forward(self, imgs: torch.Tensor, keypoints_gt=None, masks=None, factor_list=None) -> torch.tensor:
        if self.gc_config.MASK_CROWDS:
            assert masks is not None

        bb_output = self.backbone(imgs)
        scoremaps, features = self.process_output(bb_output, self.scoremap_mode)
        features = self.feature_gather(features)

        graph_constructor = get_graph_constructor(self.gc_config, scoremaps=scoremaps, features=features,
                                                  joints_gt=keypoints_gt, factor_list=factor_list, masks=masks,
                                                  device=scoremaps.device)


        x, edge_attr, edge_index, edge_labels, node_labels, node_class_labels, joint_det, label_mask, label_mask_node, joint_scores, _ = graph_constructor.construct_graph()

        # prepare class output
        if node_class_labels is not None:
            # one hot encode the class labels
            node_classes = torch.zeros(len(node_class_labels), 17, dtype=torch.float, device=node_class_labels.device)
            node_classes[list(range(0, len(node_class_labels))), node_class_labels] = 1

        else:
            node_classes = None

        return scoremaps, edge_labels, node_labels, node_classes, joint_det, joint_scores, edge_index, edge_labels, node_labels, node_class_labels, label_mask, label_mask_node, bb_output

