from ..Hourglass import PoseNet, hg_process_output
import torch, torchvision
import torch.nn as nn
from Utils.hr_utils.multi_scales_testing import *
from Utils.transformations import get_transform


def load_back_bone(config):
    hg_config = {"nstack": config.MODEL.HG.NSTACK,
                 "inp_dim": config.MODEL.HG.INPUT_DIM,
                 "oup_dim": config.MODEL.HG.OUTPUT_DIM}
    return PoseNet(**hg_config), hg_process_output


def get_hg_model(config, device):

    def rename_key(key):
        # assume structure is model.module.REAL_NAME
        return ".".join(key.split(".")[2:])
    def rename_key_hr(key):
        return key[2:]

    model = PoseEstimationHourglass(config)

    state_dict = torch.load(config.MODEL.HG.PRETRAINED, map_location=device)
    state_dict = {rename_key(k): v for k, v in state_dict["state_dict"].items()}

    model.backbone.load_state_dict(state_dict)
    model.to(device)

    return model

class PoseEstimationHourglass(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backbone, self.process_output = load_back_bone(config)

    def forward(self, imgs: torch.Tensor, config) -> torch.tensor:

        def resize(im, res):
            import cv2
            return np.array([cv2.resize(im[i], res) for i in range(im.shape[0])])

        flip_index = FLIP_CONFIG["COCO"]
        """
        output, _ = self.backbone(imgs)
        scoremaps = output[-1][:, :17]
        tags = output[-1][:, 17:34]

        return scoremaps, tags
        """

        scales = [1]

        img = imgs[0].cpu().permute(1, 2, 0).numpy()
        height, width = img.shape[0:2]
        center = (width / 2, height / 2)
        dets, tags = None, []
        for idx, i in enumerate(scales):
            scale = max(height, width) / 200
            input_res = max(height, width)
            inp_res = int((i * 512 + 63) // 64 * 64)
            res = (inp_res, inp_res)

            mat_ = get_transform(center, np.array((scale, scale)), res)[:2]
            inp = cv2.warpAffine(img, mat_, res)# / 255
            inp = torch.from_numpy(inp).permute(2, 0, 1).cuda()

            def array2dict(tmp):
                return {
                    'det': tmp[:, :, :17],
                    'tag': tmp[:, -1, 17:34]
                }

            tmp1, _ = self.backbone(inp[None])
            tmp2, _ = self.backbone(torch.flip(inp[None], [3]))
            tmp1 = torch.stack(tmp1, 1).cpu().numpy()
            tmp2 = torch.stack(tmp2, 1).cpu().numpy()
            tmp1 = array2dict(tmp1)
            tmp2 = array2dict(tmp2)

            tmp = {}
            for ii in tmp1:
                tmp[ii] = np.concatenate((tmp1[ii], tmp2[ii]), axis=0)

            det = tmp['det'][0, -1] + tmp['det'][1, -1, :, :, ::-1][flip_index]
            # det = tmp['det'][0, -1] + tmp['det'][0, -1]
            if det.max() > 10:
                continue
            if dets is None:
                dets = det
                mat = np.linalg.pinv(np.array(mat_).tolist() + [[0, 0, 1]])[:2]
            else:
                dets = dets + resize(det, dets.shape[1:3])

            if abs(i - 1) < 0.5:
                res = dets.shape[1:3]
                tags += [resize(tmp['tag'][0], res), resize(tmp['tag'][1, :, :, ::-1][flip_index], res)]
                # tags += [resize(tmp['tag'][0], res), resize(tmp['tag'][1], res)]

        if dets is None or len(tags) == 0:
            return [], []

        tags = np.concatenate([i[:, :, :, None] for i in tags], axis=3)
        dets = dets / len(scales) / 2  # divide by two because of the additional flipped det

        dets = np.minimum(dets, 1)  # clamp above
        return torch.from_numpy(dets[None]), torch.from_numpy(tags[None])

    def multi_scale_inference(self, img, scales, config):
        assert img.shape[0] == 1  # batch size of 1
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
        image = img[0].cpu().permute(1, 2, 0).numpy() # prepair for transformation

        base_size, center, scale = get_multi_scale_size_hourglass(
            image, 512, 1.0, min(scales)
        )
        final_heatmaps = None
        tags_list = []
        for idx, s in enumerate(sorted(scales, reverse=True)):
            input_size = 512
            image_resized, center, scale = resize_align_multi_scale_hourglass(
                image, input_size, s, min(scales)
            )
            image_resized = transforms(image_resized)
            image_resized = image_resized[None].cuda()

            outputs, heatmaps, tags = _get_multi_stage_outputs_hourglass(
                config, self.backbone, image_resized, with_flip=config.TEST.FLIP_TEST,
                project2image=config.TEST.PROJECT2IMAGE, size_projected=base_size
            )

            final_heatmaps, tags_list = aggregate_results_hourglass(
                config, s, final_heatmaps, tags_list, heatmaps, tags
            )
        final_heatmaps = final_heatmaps / float(len(scales))
        final_heatmaps = final_heatmaps.clamp(0, 1) # clamp above
        tags = torch.cat(tags_list, dim=4)

        return final_heatmaps, tags

def _get_multi_stage_outputs_hourglass(
        cfg, model, image, with_flip=False,
        project2image=False, size_projected=None,
):
    heatmaps = []  # collects heatmaps for image and flip image (size is always 2)
    tags = []
    features = []

    outputs, feat = model(image)
    features.append(feat)
    heatmaps.append(outputs[-1][:, :17])

    tags.append(outputs[-1][:, 17:34])

    if with_flip:
        if 'coco' in cfg.DATASET.DATASET:
            dataset_name = 'COCO'
        else:
            raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
        flip_index = FLIP_CONFIG[dataset_name]

        outputs_flip, features_flip = model(torch.flip(image, [3]))

        output = outputs_flip[-1]
        output = torch.flip(output, [3])

        tags.append(output[:, 17:34])
        tags[-1] = tags[-1][:, flip_index, :, :]

        heatmaps.append(output[:, flip_index])

    assert not cfg.DATASET.WITH_CENTER

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
