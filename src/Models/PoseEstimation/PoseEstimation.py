from Utils.Utils import set_bn_eval, set_bn_feeze
from Models.HigherHRNet import get_pose_net, hr_process_output
from Models.Hourglass import PoseNet, hg_process_output
from graph_constructor import get_graph_constructor
from Models.MessagePassingNetwork import get_mpn_model
from Utils.hr_utils.multi_scales_testing import *

import torch
import torch.nn as nn
import torchvision


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
        scoremaps, features, tags = self.process_output(bb_output, self.scoremap_mode)

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
        output["graph"] = {"nodes": joint_det, "detector_scores": joint_scores, "edge_index": edge_index, "tags": tags}

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

    def multi_scale_inference(self, img, scales, masks, config):
        assert img.shape[0] == 1  # batch size of 1
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        image = img[0].cpu().permute(1, 2, 0).numpy() # prepair for transformation

        base_size, center, scale = get_multi_scale_size(
            image, 512, 1.0, min(scales)
        )
        final_heatmaps = None
        final_features = None
        tags_list = []
        for idx, s in enumerate(sorted(scales, reverse=True)):
            input_size = 512
            image_resized, center, scale = resize_align_multi_scale(
                image, input_size, s, min(scales)
            )
            image_resized = transforms(image_resized)
            image_resized = image_resized[None].cuda()

            outputs, heatmaps, tags, features = _get_multi_stage_outputs(
                config, self.backbone, image_resized, self.feature_gather, with_flip=config.TEST.FLIP_TEST,
                project2image=True, size_projected=base_size
            )

            final_heatmaps, tags_list, final_features = aggregate_results_mpn(
                config, s, final_heatmaps, tags_list, final_features, heatmaps, tags, features
            )
        scoremaps = final_heatmaps / float(len(scales))
        features = final_features / float(len(scales))

        # features = self.feature_gather(features)  # or average after this

        graph_constructor = get_graph_constructor(self.gc_config, scoremaps=scoremaps, features=features,
                                                  joints_gt=None, factor_list=None, masks=masks,
                                                  device=scoremaps.device, testing=not self.training,
                                                  heatmaps=None)

        x, edge_attr, edge_index, edge_labels, node_labels, class_labels, node_targets, joint_det, label_mask, label_mask_node, class_mask, joint_scores, batch_index, node_persons = graph_constructor.construct_graph()

        edge_pred, node_pred, class_pred, _, _ = self.mpn(x, edge_attr, edge_index, node_labels=node_labels,
                                                          edge_labels=edge_labels
                                                          , batch_index=batch_index, node_mask=label_mask_node)

        output = {}
        output["preds"] = {"edge": edge_pred, "node": node_pred, "class": class_pred}
        output["graph"] = {"nodes": joint_det, "detector_scores": joint_scores, "edge_index": edge_index}

        return scoremaps, output

def _get_multi_stage_outputs(
        cfg, model, image, feature_gather, with_flip=False,
        project2image=False, size_projected=None,
):
    # outputs = []
    heatmaps_avg = 0  # this holds the average value of the different heatmaps produced by forward pass
    num_heatmaps = 0
    heatmaps = []  # collects heatmaps for image and flip image (size is always 2)
    tags = []
    features = []

    outputs, feat = model(image)
    feat = feature_gather(feat)
    features.append(feat)
    for i, output in enumerate(outputs):
        # resize all heatmap/ae predictions to the size of the final heatmap prediction
        if len(outputs) > 1 and i != len(outputs) - 1:
            output = torch.nn.functional.interpolate(
                output,
                size=(outputs[-1].size(2), outputs[-1].size(3)),
                mode='bilinear',
                align_corners=False
            )

        offset_feat = cfg.DATASET.NUM_JOINTS \
            if cfg.MODEL.HRNET.LOSS.WITH_HEATMAPS_LOSS[i] else 0
        # sum the heatmaps
        if cfg.MODEL.HRNET.LOSS.WITH_HEATMAPS_LOSS[i] and cfg.TEST.WITH_HEATMAPS[i]:
            heatmaps_avg += output[:, :cfg.DATASET.NUM_JOINTS]
            num_heatmaps += 1
        # append the tag maps for later concatenation
        if cfg.MODEL.HRNET.LOSS.WITH_AE_LOSS[i] and cfg.TEST.WITH_AE[i]:
            tags.append(output[:, offset_feat:])
    # now average heatmaps (or normalization part of average)
    if num_heatmaps > 0:
        heatmaps.append(heatmaps_avg/num_heatmaps)

    if with_flip:
        if 'coco' in cfg.DATASET.DATASET:
            dataset_name = 'COCO'
        elif 'crowd_pose' in cfg.DATASET.DATASET:
            dataset_name = 'CROWDPOSE'
        else:
            raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
        flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER'] \
            if cfg.DATASET.WITH_CENTER else FLIP_CONFIG[dataset_name]
        flip_index = flip_index if cfg.TEST.FLIP_AND_REARANGE else FLIP_CONFIG["COCO_WITHOUT_REARANGING"]

        heatmaps_avg = 0
        num_heatmaps = 0
        outputs_flip, features_flip = model(torch.flip(image, [3]))
        features_flip = feature_gather(features_flip)
        features.append(torch.flip(features_flip, [3]))
        for i in range(len(outputs_flip)):
            output = outputs_flip[i]
            if len(outputs_flip) > 1 and i != len(outputs_flip) - 1:
                output = torch.nn.functional.interpolate(
                    output,
                    size=(outputs_flip[-1].size(2), outputs_flip[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            output = torch.flip(output, [3])
            outputs.append(output)

            offset_feat = cfg.DATASET.NUM_JOINTS \
                if cfg.MODEL.HRNET.LOSS.WITH_HEATMAPS_LOSS[i] else 0

            if cfg.MODEL.HRNET.LOSS.WITH_HEATMAPS_LOSS[i] and cfg.TEST.WITH_HEATMAPS[i]:
                heatmaps_avg += \
                    output[:, :cfg.DATASET.NUM_JOINTS][:, flip_index, :, :]
                num_heatmaps += 1

            if cfg.MODEL.HRNET.LOSS.WITH_AE_LOSS[i] and cfg.TEST.WITH_AE[i]:
                tags.append(output[:, offset_feat:])
                if cfg.MODEL.HRNET.TAG_PER_JOINT:
                    tags[-1] = tags[-1][:, flip_index, :, :]

        heatmaps.append(heatmaps_avg/num_heatmaps)

    assert not cfg.DATASET.WITH_CENTER
    """
    if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
        heatmaps = [hms[:, :-1] for hms in heatmaps]
        tags = [tms[:, :-1] for tms in tags]
    """

    # upscale to input image size
    if project2image and size_projected:
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for tms in tags
        ]
        features = [
            torch.nn.functional.interpolate(
                feat,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for feat in features
        ]

    return outputs, heatmaps, tags, features


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
