from Models.HigherHRNet import get_pose_net, create_process_func_hr, get_mmpose_hrnet
import torch
import torchvision
import torch.nn as nn
from Utils.hr_utils.multi_scales_testing import *


def get_hr_model(config, device):

    def rename_key(key):
        # assume structure is module.REAL_NAME
        return ".".join(key.split(".")[1:])
    model = PoseEstimationAeGroup(config)

    if config.MODEL.KP == "hrnet":

        state_dict = torch.load(config.MODEL.HRNET.PRETRAINED, map_location=device)
        if config.MODEL.HRNET.PRETRAINED == "../PretrainedModels/pose_higher_hrnet_w48_640_crowdpose.pth.tar":
            state_dict = {rename_key(k): v for k, v in state_dict.items()}

    elif config.MODEL.KP =="mmpose_hrnet":
        state_dict = torch.load(config.MODEL.HRNET.PRETRAINED, map_location=device)["state_dict"]

    model.backbone.load_state_dict(state_dict)

    return model


def load_back_bone(config):

    if config.MODEL.KP == "hrnet":
        return get_pose_net(config, False)
    elif config.MODEL.KP == "mmpose_hrnet":
        return get_mmpose_hrnet(config)
    else:
        raise NotImplementedError


class PoseEstimationAeGroup(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backbone = load_back_bone(config)
        self.num_joints = config.DATASET.NUM_JOINTS

    def forward(self, imgs: torch.Tensor) -> torch.tensor:
        outputs, _ = self.backbone(imgs)

        # outputs = []
        heatmaps_avg = 0
        num_heatmaps = 0
        heatmaps = []
        tags = []

        loss_with_heatmaps = (True, True)
        test_with_heatmaps = (True, True)
        los_with_ae = (True, False)
        test_with_ae = (True, False)
        for i, output in enumerate(outputs):
            if len(outputs) > 1 and i != len(outputs) - 1:
                output = torch.nn.functional.interpolate(
                    output,
                    size=(outputs[-1].size(2), outputs[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                )

            offset_feat = self.num_joints \
                if loss_with_heatmaps[i] else 0

            if loss_with_heatmaps[i] and test_with_heatmaps[i]:
                heatmaps_avg += output[:, :self.num_joints]
                num_heatmaps += 1

            if los_with_ae[i] and test_with_ae[i]:
                tags.append(output[:, offset_feat:])

        if num_heatmaps > 0:
            heatmaps.append(heatmaps_avg / num_heatmaps)

        project2image = True
        size_projected = (imgs.shape[3], imgs.shape[2])
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

        return heatmaps, tags


    def multi_scale_inference(self, img, scales, device, config):
        # need: scales
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
        image = img[0].cpu().permute(1, 2, 0).numpy()  # prepair for transformation

        base_size, center, scale = get_multi_scale_size(
            image, config.DATASET.INPUT_SIZE, 1.0, min(scales)
        )
        final_heatmaps = None
        tags_list = []
        for idx, s in enumerate(sorted(scales, reverse=True)):
            input_size = config.DATASET.INPUT_SIZE
            image_resized, center, scale = resize_align_multi_scale(
                image, input_size, s, min(scales)
            )
            image_resized = transforms(image_resized)
            image_resized = image_resized[None].to(device)

            outputs, heatmaps, tags = _get_multi_stage_outputs(
                config, self.backbone, image_resized, with_flip=config.TEST.FLIP_TEST,
                project2image=config.TEST.PROJECT2IMAGE, size_projected=base_size
            )

            final_heatmaps, tags_list = aggregate_results(
                config, s, final_heatmaps, tags_list, heatmaps, tags
            )
        final_heatmaps = final_heatmaps / float(len(scales))
        tags = torch.cat(tags_list, dim=4)

        return final_heatmaps, tags



def _get_multi_stage_outputs(
        cfg, model, image, with_flip=False,
        project2image=False, size_projected=None
):
    # outputs = []
    heatmaps_avg = 0
    num_heatmaps = 0
    heatmaps = []
    tags = []

    outputs, _ = model(image)
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

        heatmaps_avg = 0
        num_heatmaps = 0
        outputs_flip, _ = model(torch.flip(image, [3]))
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

    if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
        heatmaps = [hms[:, :-1] for hms in heatmaps]
        tags = [tms[:, :-1] for tms in tags]

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

    return outputs, heatmaps, tags
