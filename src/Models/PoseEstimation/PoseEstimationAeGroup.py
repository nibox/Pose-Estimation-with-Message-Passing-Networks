from Models.HigherHRNet import get_pose_net, hr_process_output

import torch
import torch.nn as nn


def get_hr_model(config, device):

    model = PoseEstimationAeGroup(config)

    if config.MODEL.KP == "hrnet":
        state_dict = torch.load(config.MODEL.HRNET.PRETRAINED, map_location=device)
    else:
        raise NotImplementedError

    model.backbone.load_state_dict(state_dict)

    return model


def load_back_bone(config):

    if config.MODEL.KP == "hrnet":
        return get_pose_net(config, False)
    else:
        raise NotImplementedError


class PoseEstimationAeGroup(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backbone  = load_back_bone(config)

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

            offset_feat = 17 \
                if loss_with_heatmaps[i] else 0

            if loss_with_heatmaps[i] and test_with_heatmaps[i]:
                heatmaps_avg += output[:, :17]
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

        return outputs, heatmaps, tags

