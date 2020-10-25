from crowdposetools.coco import COCO
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pickle
import os

from data.utils import _filter_visible, pack_for_batch


class CrowdPoseKeypoints(Dataset):

    def __init__(self, path, mode="train", seed=0, filter_empty=True,
                 transforms=None, heatmap_generator=None,
                 joint_generator=None, mini=False):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.root_path = path
        # todo deal with different setups and with the different splits
        ann_path = f"{self.root_path}/json/crowdpose_{mode}.json"
        self.coco = COCO(ann_path)
        self.num_joints = 14
        self.transforms = transforms
        # assert self.transforms is not None

        assert isinstance(heatmap_generator, (list, tuple)) or heatmap_generator is None
        self.num_scales = len(heatmap_generator) if heatmap_generator is not None else 0

        self.heatmap_generator = heatmap_generator
        self.joint_generator = joint_generator

        self.max_num_people = 30  # from github code
        assert mode in ["train", "val", "test", "trainval"]

        self.img_ids = list(self.coco.imgs.keys())
        assert len(self.img_ids) == len(set(self.img_ids))
        if filter_empty:
            self.img_ids = [id for id in self.img_ids if len(self.coco.getAnnIds(img_ids=id, iscrowd=None)) > 0]
        if mini:
            assert mode in ["test", "val"]
            self.img_ids = np.random.choice(self.img_ids, 500, replace=False)

    def __getitem__(self, idx):
        assert self.transforms is not None
        assert self.heatmap_generator is not None
        img_id = int(self.img_ids[idx])  # img_ids is array of numpy.int32
        ann_ids = self.coco.getAnnIds(imgIds=img_id)  # keypoints, bbox, sem mask?
        img_info = self.coco.loadImgs(img_id)[0]
        ann = self.coco.loadAnns(ids=ann_ids)

        num_people = len(ann)
        img_height = img_info["height"]
        img_width = img_info["width"]

        # load image
        # todo reading the image for each iteration is probably not efficient
        with open(f"{self.root_path}/images/{img_info['file_name']}", "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))

        # load keypoints
        kpt_oks_sigmas = np.array([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79])/10.0
        distance_factor = np.zeros([num_people, self.num_joints])  # this factor combines sigma with scale (taken from OKS computation)
        distance_factor[0:num_people] = (kpt_oks_sigmas * 2) ** 2
        keypoints_list = []
        factor_list = []
        for i in range(num_people):
            if ann[i]["num_keypoints"] > 0:
                keypoints_list.append(np.array(ann[i]["keypoints"]).reshape([-1, 3]))

                area = ann[i]['bbox'][3] * ann[i]['bbox'][2] * 0.53  # not sure what the factor means
                factor_list.append(np.array(kpt_oks_sigmas * 2) ** 2 * (area + np.spacing(1)) * 2.0)

        keypoints = np.array(keypoints_list).astype(np.float)
        factors = np.array(factor_list)
        # in the code keypoints2 is created (seems to include only persons with atleast one visible keypoint)
        # not sure if it is necessary

        # load mask
        mask = np.zeros([img_height, img_width])
        mask = (mask < 0.5).astype(np.float32)

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        keypoint_list = [keypoints.copy() for _ in range(self.num_scales)]
        ae_targets = [keypoints.copy() for _ in range(self.num_scales)]
        heatmaps = []

        if self.transforms is not None:
            img, mask, keypoint_list, factors = self.transforms(img, mask_list, keypoint_list, factors)

        if self.heatmap_generator is not None:
            for scale_idx in range(self.num_scales):
                heatmap = self.heatmap_generator[scale_idx](keypoint_list[scale_idx], None)
                ae_target = self.joint_generator[scale_idx](ae_targets[scale_idx])
                keypoint_list[scale_idx] = _filter_visible(keypoint_list[scale_idx], mask[scale_idx].shape)
                # keypoint_list[scale_idx] = remove_empty_rows(keypoint_list[scale_idx])

                heatmaps.append(heatmap.astype(np.float32))
                ae_targets[scale_idx] = ae_target.astype(np.int32)
                mask_list[scale_idx] = mask_list[scale_idx].astype(np.float32)
                # keypoint_list[scale_idx] = pack_for_batch(keypoint_list[scale_idx].astype(np.float32), 30)

        kpts = keypoint_list[-1]
        if len(kpts) != 0:
            empty_row = kpts[:, :, 2].sum(axis=1) == 0.0
            rows_to_keep = np.logical_not(empty_row)
            keypoint_list[-1] = pack_for_batch(kpts[rows_to_keep].astype(np.float32), 30)
            factors = pack_for_batch(factors[rows_to_keep],
                                     30)  # assuming the visible keypoints are the same for all scales
        else:
            keypoint_list[-1] = pack_for_batch(kpts, 30)
            factors = pack_for_batch(factors, 30)  # assuming the visible keypoints are the same for all scales

        return img, heatmaps, mask, keypoint_list[-1], factors, ae_targets

    def __len__(self):
        return len(self.img_ids)

