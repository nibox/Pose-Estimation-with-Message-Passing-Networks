from ..Hourglass.Hourglass import Hourglass
from ..MessagePassingNetwork.VanillaMPN import VanillaMPN
from Utils.ConstructGraph import *

import torch
import torch.nn as nn
import torch.nn.functional as F

default_config = {"backbone": Hourglass,
                  "backbone_config": {"nstack": 4,
                                      "input_dim": 256,
                                      "output_size": 68},
                  "message_passing": VanillaMPN,
                  "message_passing_config": {

                  },
                  "graph_constructor": NaiveGraphConstructor
                  }


class PoseEstimationBaseline(nn.Module):

    def __init__(self, config, with_logits=False):
        super().__init__()
        self.backbone = config["backbone"](**config["backbone_config"])
        self.mpn = config["message_passing"](**config["message_passing_config"])
        self.graph_constructor = config["graph_constructor"]
        self.feature_gather = nn.Conv2d(256, 128, 3, 1, 1, bias=True)

        self.with_logits = with_logits

        self.pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, imgs: torch.Tensor, keypoints_gt=None) -> torch.tensor:
        score_map, features = self.backbone(imgs)

        features = self.feature_gather(score_map)

        graph_consturctor = self.graph_constructor(score_map, features, keypoints_gt)

        x, edge_attr, edge_index, edge_labels, joint_det = graph_consturctor.construct_graph()

        pred = self.mpn(x, edge_attr, edge_index)

        return pred

    def loss(self, output, targets) -> torch.Tensor:
        if self.with_logits:
            return F.binary_cross_entropy_with_logits(output, targets)
        else:
            return F.binary_cross_entropy(output, targets)
