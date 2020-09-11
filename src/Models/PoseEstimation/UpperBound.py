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

    def forward(self, imgs: torch.Tensor, keypoints_gt=None, masks=None, factor_list=None, heatmaps=None) -> torch.tensor:
        if self.gc_config.MASK_CROWDS:
            assert masks is not None

        bb_output = self.backbone(imgs)
        scoremaps, features, _ = self.process_output(bb_output, self.scoremap_mode)
        features = self.feature_gather(features)

        graph_constructor = get_graph_constructor(self.gc_config, scoremaps=scoremaps, features=features,
                                                  joints_gt=keypoints_gt, factor_list=factor_list, masks=masks,
                                                  device=scoremaps.device, testing=not self.training, heatmaps=heatmaps)


        x, edge_attr, edge_index, edge_labels, node_labels, class_labels, node_targets, joint_det, label_mask, label_mask_node, class_mask, joint_scores, _, node_persons = graph_constructor.construct_graph()

        # prepare class output
        if class_labels is not None:
            # one hot encode the class labels
            node_classes = torch.zeros(len(class_labels), 17, dtype=torch.float, device=class_labels.device)
            node_classes[list(range(0, len(class_labels))), class_labels] = 1

        else:
            node_classes = None
        """
        if node_targets is not None and False:

            node_mask = node_labels == 1.0  # this might be a function

            joint_det_refine = joint_det[node_mask]
            node_persons_sort = node_persons[node_mask]  # these are the clusters
            node_types = class_labels[node_mask] if class_labels is not None else joint_det_refine[:, 2]
            node_scores = joint_scores[node_mask]
            # sorted
            node_persons_sort, sorted_idx = torch.sort(node_persons_sort)
            joint_det_refine = joint_det_refine[sorted_idx]
            node_types = node_types[sorted_idx]
            node_scores = node_scores[sorted_idx]

            joint_det_refine = torch.zeros(len(joint_det_refine), 3 + 1 + 1, dtype=torch.float,
                                               device=joint_det_refine.device)
            joint_det_refine[:, :2] = node_targets[node_persons_sort, node_types, :2]
            joint_det_refine[:, 2] = node_types
            joint_det_refine[:, 3] = node_scores
            joint_det_refine[:, 4] = node_persons_sort
        else:
            joint_det_refine = None
        """

        output = {}
        output["labels"] = {"edge": edge_labels, "node": node_labels, "class": class_labels, "refine": node_persons}
        output["masks"] = {"edge": label_mask, "node": label_mask_node}
        output["preds"] = {"edge": edge_labels, "node": node_labels, "class": node_classes, "heatmap": bb_output[0]}
        output["graph"] = {"nodes": joint_det, "detector_scores": joint_scores, "edge_index": edge_index}

        return scoremaps, output

