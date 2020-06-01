from ..Hourglass.Hourglass import PoseNet
from ..MessagePassingNetwork.VanillaMPN import VanillaMPN
from Utils.ConstructGraph import *
from Utils.Utils import FocalLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

default_config = {"backbone": PoseNet,
                  "backbone_config": {"nstack": 4,
                                      "inp_dim": 256,
                                      "oup_dim": 68},
                  "message_passing": VanillaMPN,
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


class PoseEstimationBaseline(nn.Module):

    def __init__(self, config, with_logits=True):
        super().__init__()
        self.backbone = config["backbone"](**config["backbone_config"])
        self.mpn = config["message_passing"](**config["message_passing_config"])
        self.graph_constructor = config["graph_constructor"]
        self.feature_gather = nn.Conv2d(256, 128, 3, 1, 1, bias=True)

        self.with_logits = with_logits

        self.config = config
        self.cheat = config["cheat"]
        self.use_gt = config["use_gt"]
        self.use_neighbours = config["use_neighbours"]
        self.edge_label_method = config["edge_label_method"]
        self.mask_crowds = config["mask_crowds"]
        self.focal = None
        if config["use_focal_loss"]:
            self.focal = FocalLoss(logits=True)

        self.pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, imgs: torch.Tensor, keypoints_gt=None, masks=None, factors=None, with_logits=True) -> torch.tensor:
        if self.mask_crowds:
            assert masks is not None
        scoremap, features, early_features = self.backbone(imgs)
        scoremap = scoremap[:, -1, :17]

        features = self.feature_gather(features)

        graph_constructor = self.graph_constructor(scoremap, features, keypoints_gt, factors, masks, use_gt=self.use_gt,
                                                   no_false_positives=self.cheat, use_neighbours=self.use_neighbours,
                                                   device=scoremap.device, edge_label_method=self.edge_label_method,
                                                   detect_threshold=self.config["detect_threshold"],
                                                   mask_crowds=self.mask_crowds,
                                                   inclusion_radius=self.config["inclusion_radius"],
                                                   matching_radius=self.config["matching_radius"],
                                                   mpn_graph_type=self.config["mpn_graph_type"])

        x, edge_attr, edge_index, edge_labels, joint_det, label_mask, batch_index = graph_constructor.construct_graph()

        pred = self.mpn(x, edge_attr, edge_index).squeeze()
        if not with_logits:
            pred = torch.sigmoid(pred)

        return pred, joint_det, edge_index, edge_labels, label_mask, batch_index

    def loss(self, output, targets, with_logits=True, pos_weight=None, mask=None, batch_index=None) -> torch.Tensor:
        if self.focal is not None:
            assert with_logits
            loss = self.focal(output, targets, mask, batch_index)
            return loss

        assert mask is None
        if with_logits:
            return F.binary_cross_entropy_with_logits(output, targets, pos_weight=pos_weight)
        else:
            return F.binary_cross_entropy(output, targets)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False



def load_model(path, model_class, config, device, pretrained_path=None):

    assert not (path is not None and pretrained_path is not None)
    def rename_key(key):
        # assume structure is model.module.REAL_NAME
        return ".".join(key.split(".")[2:])

    #model = hourglass.PoseNet(kwargs["nstack"], kwargs["input_dim"], kwargs["output_size"])
    model = model_class(config)
    if path is not None:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict["model_state_dict"])
    elif pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=device)
        state_dict_new = {rename_key(k): v for k, v in state_dict["state_dict"].items()}
        model.backbone.load_state_dict(state_dict_new)

    return model