from ..Hourglass.Hourglass import PoseNet
from Utils.ConstructGraph import *

import torch
import torch.nn as nn

default_config = {"backbone": PoseNet,
                  "backbone_config": {"nstack": 4,
                                      "inp_dim": 256,
                                      "oup_dim": 68},
                  "graph_constructor": NaiveGraphConstructor,
                  "cheat": True,
                  "use_gt": True,
                  "use_focal_loss": False,
                  "use_neighbours": False,
                  "edge_label_method": 1,
                  "mask_crowds": False,
                  "detect_threshold": 0.007,
                  "inclusion_radius": 5.0,
                  "mpn_graph_type": "knn"
                  }


class UpperBoundModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backbone = config["backbone"](**config["backbone_config"])
        self.graph_constructor = config["graph_constructor"]
        self.feature_gather = nn.AvgPool2d(3, 1, 1)

        self.config = config
        self.cheat = config["cheat"]
        self.use_gt = config["use_gt"]
        self.use_neighbours = config["use_neighbours"]
        self.edge_label_method = config["edge_label_method"]
        self.mask_crowds = config["mask_crowds"]

    def forward(self, imgs: torch.Tensor, keypoints_gt=None, masks=None) -> torch.tensor:
        if self.mask_crowds:
            assert masks is not None

        scoremap, features, _ = self.backbone(imgs)

        scoremap = scoremap[:, -1, :17]

        features = self.feature_gather(features)

        graph_constructor = self.graph_constructor(scoremap, features, keypoints_gt, masks, use_gt=self.use_gt,
                                                   no_false_positives=self.cheat, use_neighbours=self.use_neighbours,
                                                   device=scoremap.device, edge_label_method=self.edge_label_method,
                                                   detect_threshold=self.config["detect_threshold"],
                                                   mask_crowds=self.mask_crowds,
                                                   inclusion_radius=self.config["inclusion_radius"],
                                                   mpn_graph_type=self.config["mpn_graph_type"])

        x, edge_attr, edge_index, edge_labels, joint_det, label_mask, batch_index = graph_constructor.construct_graph()

        return edge_labels, joint_det, edge_index, edge_labels, label_mask

