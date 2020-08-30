from Utils.Utils import set_bn_eval, set_bn_feeze
from Models.HigherHRNet import get_pose_net, hr_process_output
from Models.Hourglass import PoseNet, hg_process_output
from graph_constructor import get_graph_constructor
from Models.MessagePassingNetwork import get_mpn_model

import torch
import torch.nn as nn

def get_pose_model(config, device):

    def rename_key(key):
        # assume structure is model.module.REAL_NAME
        return ".".join(key.split(".")[2:])
    def rename_key_hr(key):
        return key[2:]

    model = PoseEstimationBaseline(config)

    if config.MODEL.KP == "hrnet":
        state_dict = torch.load(config.MODEL.HRNET.PRETRAINED, map_location=device)
        if config.MODEL.HRNET.PRETRAINED != '../PretrainedModels/pose_higher_hrnet_w32_512.pth':
            state_dict = {rename_key_hr(k): v for k, v in state_dict.items()}
    elif config.MODEL.KP == "hourglass":
        state_dict = torch.load(config.MODEL.HG.PRETRAINED, map_location=device)
        state_dict = {rename_key(k): v for k, v in state_dict["state_dict"].items()}
    else:
        raise NotImplementedError

    model.backbone.load_state_dict(state_dict)

    return model


def load_back_bone(config):
    if config.MODEL.KP == "hourglass":
        hg_config = {"nstack": config.MODEL.HG.NSTACK,
                     "inp_dim": config.MODEL.HG.INPUT_DIM,
                     "oup_dim": config.MODEL.HG.OUTPUT_DIM}
        return PoseNet(**hg_config), hg_process_output
    elif config.MODEL.KP == "hrnet":
        return get_pose_net(config, False), hr_process_output


class PoseEstimationBaseline(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.backbone = config["backbone"](**config["backbone_config"])
        self.backbone, self.process_output = load_back_bone(config)
        self.mpn = get_mpn_model(config.MODEL.MPN)
        self.gc_config = config.MODEL.GC

        input_feature_size = config.MODEL.KP_OUTPUT_DIM if config.MODEL.HRNET.FEATURE_FUSION != "cat_multi" else 352
        self.feature_gather = nn.Conv2d(input_feature_size, config.MODEL.MPN.NODE_INPUT_DIM,
                                        config.MODEL.FEATURE_GATHER_KERNEL, 1, config.MODEL.FEATURE_GATHER_PADDING,
                                        bias=True)
        self.num_aux_steps = config.MODEL.AUX_STEPS
        self.scoremap_mode = config.MODEL.HRNET.SCOREMAP_MODE

    def forward(self, imgs: torch.Tensor, keypoints_gt=None, masks=None, factors=None, heatmaps=None,
                with_logits=True) -> torch.tensor:
        if self.gc_config.MASK_CROWDS:
            assert masks is not None
        bb_output = self.backbone(imgs)
        scoremaps, features = self.process_output(bb_output, self.scoremap_mode)

        features = self.feature_gather(features)
        scoremaps = scoremaps.detach()

        graph_constructor = get_graph_constructor(self.gc_config, scoremaps=scoremaps, features=features,
                                                  joints_gt=keypoints_gt, factor_list=factors, masks=masks,
                                                  device=scoremaps.device, testing=not self.training, heatmaps=heatmaps)

        x, edge_attr, edge_index, edge_labels, node_labels, class_labels, node_targets, joint_det, label_mask, label_mask_node, class_mask, joint_scores, batch_index, node_persons = graph_constructor.construct_graph()

        edge_pred, node_pred, class_pred, _, _ = self.mpn(x, edge_attr, edge_index, node_labels=node_labels,
                                                          edge_labels=edge_labels
                                                          , batch_index=batch_index, node_mask=label_mask_node)

        if not with_logits:
            if edge_pred[-1] is not None:
                edge_pred[-1] = torch.sigmoid(edge_pred[-1])
            if node_pred is not None:
                node_pred[-1] = torch.sigmoid(node_pred[-1])
            if class_pred is not None:
                class_pred[-1] = torch.softmax(class_pred[-1], dim=1)

        output = {}
        output["labels"] = {"edge": edge_labels, "node": node_labels, "class": class_labels}
        output["masks"] = {"edge": label_mask, "node": label_mask_node, "class": class_mask}
        output["preds"] = {"edge": edge_pred, "node": node_pred, "class": class_pred, "heatmap": bb_output[0]}
        output["graph"] = {"nodes": joint_det, "detector_scores": joint_scores, "edge_index": edge_index}

        return scoremaps, output

    def freeze_backbone(self, mode):
        if mode == "complete":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif mode == "stem":
            self.backbone.apply(set_bn_feeze)
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
            for param in self.backbone.conv2.parameters():
                param.requires_grad = False
            for param in self.backbone.bn2.parameters():
                param.requires_grad = False
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False
        elif mode == "nothing":
            self.backbone.apply(set_bn_feeze)
        elif mode == "from_scratch":
            return
        else:
            raise NotImplementedError

    def stop_backbone_bn(self):
        self.backbone.apply(set_bn_eval)

    def test(self, test=True):
        pass


"""
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
"""