from ..Hourglass.Hourglass import PoseNet
from ..MessagePassingNetwork.VanillaMPN import VanillaMPN
from Utils.ConstructGraph import *

import torch
import torch.nn as nn
import torch.nn.functional as F

default_config = {"backbone": PoseNet,
                  "backbone_config": {"nstack": 4,
                                      "inp_dim": 256,
                                      "oup_dim": 68},
                  "message_passing": VanillaMPN,
                  "message_passing_config": {"steps": 4,
                                             "node_input_dim": 128,
                                             "edge_input_dim": 2 + 17 * 17,
                                             "node_feature_dim": 128,
                                             "edge_feature_dim": 128,
                                             "node_hidden_dim": 256,
                                             "edge_hidden_dim": 512
                                             },
                  "graph_constructor": NaiveGraphConstructor
                  }


class PoseEstimationBaseline(nn.Module):

    def __init__(self, config, with_logits=True):
        super().__init__()
        self.backbone = config["backbone"](**config["backbone_config"])
        self.mpn = config["message_passing"](**config["message_passing_config"])
        self.graph_constructor = config["graph_constructor"]
        self.feature_gather = nn.Conv2d(256, 128, 3, 1, 1, bias=True)

        self.with_logits = with_logits

        self.pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, imgs: torch.Tensor, keypoints_gt=None) -> torch.tensor:
        scoremap, features, early_features = self.backbone(imgs)
        scoremap = scoremap[:, -1, :17]

        features = self.feature_gather(features)

        graph_constructor = self.graph_constructor(scoremap, features, keypoints_gt)

        x, edge_attr, edge_index, edge_labels, joint_det = graph_constructor.construct_graph()

        pred = self.mpn(x, edge_attr, edge_index).squeeze()
        if not self.with_logits:
            pred = torch.sigmoid(pred)

        return pred, joint_det, edge_index, edge_labels

    def loss(self, output, targets, pos_weight=None) -> torch.Tensor:
        if self.with_logits:
            return F.binary_cross_entropy_with_logits(output, targets, pos_weight=pos_weight)
        else:
            return F.binary_cross_entropy(output, targets)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
