from pycocotools.coco import COCO
from pycocotools import mask as maskapi
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import pickle
import os
from .utils import HeatmapGenerator, JointsGenerator, _filter_visible

from Utils.transformations import kpt_affine, factor_affine, get_transform


class CocoKeypoints(Dataset):

    def __init__(self, path, mini=False, mode="train", seed=0, filter_empty=True,
                 img_ids=None, year=14, transforms=None, mask_crowds=True, heatmap_generator=None,
                 joint_generator=None):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.root_path = path
        # todo deal with different setups and with the different splits
        ann_path = f"{self.root_path}/annotations/person_keypoints_{mode}20{year}.json"
        self.coco = COCO(ann_path)
        self.transforms = transforms
        assert self.transforms is not None
        self.heatmap_generator = heatmap_generator
        self.joints_generator = joint_generator
        self.num_scales = len(heatmap_generator)
        # self.input_size = input_size
        # self.output_size = output_size

        self.max_num_people = 30  # from github code
        self.mask_crowds = mask_crowds
        assert mode in ["train", "val"]
        self.data_dir =f"train20{year}" if mode == "train" else f"val20{year}"

        self.cat_ids = self.coco.getCatIds(catNms=["person"])
        self.img_ids = img_ids if img_ids is not None else self.coco.getImgIds(catIds=self.cat_ids)
        assert len(self.img_ids) == len(set(self.img_ids))
        if filter_empty and img_ids is None:
            filtered_ids_fname = f"tmp/usable_ids_{mode}_{year}.p"
            cached = os.path.exists(filtered_ids_fname) and True
            if cached:
                print("loading cached filtered image ids")
                self.img_ids = pickle.load(open(filtered_ids_fname, "rb"))
                assert len(self.img_ids) == len(set(self.img_ids))
            else:
                print("Creating filtered image ids")
                empty_ids = []
                usable_ids = []
                for id in self.img_ids:
                    ann_ids = self.coco.getAnnIds(imgIds=id)  # keypoints, bbox, sem mask?
                    ann = self.coco.loadAnns(ids=ann_ids)
                    not_empty = False
                    for i in range(len(ann)):
                        keypoints = np.array(ann[i]["keypoints"]).reshape([-1, 3])
                        vis_flag_count = len(np.where(keypoints[:, 2] != 0)[0])
                        not_empty = not_empty or vis_flag_count > 1

                    if not_empty:
                        usable_ids.append(id)
                    else:
                        empty_ids.append(id)
                self.img_ids = usable_ids
                pickle.dump(self.img_ids, open(filtered_ids_fname, "wb"))

        if mini and img_ids is None:
            if year == 17 and mode == "val":
                self.img_ids = np.random.choice(self.img_ids, 500, replace=False)
            else:
                self.img_ids = np.random.choice(self.img_ids, 4000, replace=False)
            assert len(self.img_ids) == len(set(self.img_ids))  # assert that the ids are unique

    def __getitem__(self, idx):
        img_id = int(self.img_ids[idx])  # img_ids is array of numpy.int32
        ann_ids = self.coco.getAnnIds(imgIds=img_id)  # keypoints, bbox, sem mask?
        img_info = self.coco.loadImgs(img_id)[0]
        ann = self.coco.loadAnns(ids=ann_ids)

        num_people = len(ann)
        img_height = img_info["height"]
        img_width = img_info["width"]

        # load image
        # todo reading the image for each iteration is probably not efficient
        with open(f"{self.root_path}/{self.data_dir}/{img_info['file_name']}", "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))

        # load keypoints
        kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        distance_factor = np.zeros([num_people, 17])  # this factor combines sigma with scale (taken from OKS computation)
        distance_factor[0:num_people] = (kpt_oks_sigmas * 2) ** 2
        keypoints_list = []
        factor_list = []
        for i in range(num_people):
            if ann[i]["num_keypoints"] > 0:
                keypoints_list.append(np.array(ann[i]["keypoints"]).reshape([-1, 3]))
                factor_list.append(np.array(kpt_oks_sigmas * 2) ** 2 * (ann[i]["area"] + np.spacing(1)) * 2.0)

        keypoints = np.array(keypoints_list).astype(np.float)
        factors = np.array(factor_list)
        # in the code keypoints2 is created (seems to include only persons with atleast one visible keypoint)
        # not sure if it is necessary

        # load mask
        mask = np.zeros([img_height, img_width])
        if self.mask_crowds:
            for j in ann:
                if j["iscrowd"]:
                    rle = maskapi.frPyObjects(
                        j['segmentation'], img_info['height'], img_info['width'])
                    mask += maskapi.decode(rle)
                # testing for the number of keypoints should be sufficient if the assertion above is true
                elif j["num_keypoints"] == 0:
                    rles = maskapi.frPyObjects(
                        j['segmentation'], img_info['height'], img_info['width'])
                    for rle in rles:
                        mask += maskapi.decode(rle)

        mask = (mask < 0.5).astype(np.float32)

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        keypoint_list = [keypoints.copy() for _ in range(self.num_scales)]
        heatmaps = []

        if self.transforms is not None:
            img, mask, keypoint_list, factors = self.transforms(img, mask_list, keypoint_list, factors)

        if self.heatmap_generator is not None:
            for scale_idx in range(self.num_scales):
                heatmap = self.heatmap_generator[scale_idx](keypoint_list[scale_idx], None)
                keypoint_list[scale_idx] = _filter_visible(keypoint_list[scale_idx], mask[scale_idx].shape)
                # keypoint_list[scale_idx] = remove_empty_rows(keypoint_list[scale_idx])

                heatmaps.append(heatmap.astype(np.float32))
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

        return img, heatmaps, mask, keypoint_list[-1], factors, 0

    def __len__(self):
        return len(self.img_ids)

"""
    def get_tensor(self, idx, device) -> torch.tensor:
        img, mask, keypoints, factor_list = self[idx]
        mask = torch.from_numpy(mask).to(device).unsqueeze(0)
        keypoints = torch.from_numpy(keypoints).to(device).unsqueeze(0)
        factor_list = torch.from_numpy(factor_list).to(device).unsqueeze(0)

        return img, mask, keypoints, factor_list
"""


def pack_keypoints_for_batch(keypoints: np.array, max_num_people):
    out = np.zeros([max_num_people, 17, 3])
    out[:len(keypoints)] = keypoints
    return out


def pack_for_batch(array, max_num_people):
    new_shape = list(array.shape)
    new_shape[0] = max_num_people
    out = np.zeros(new_shape)
    out[:len(array)] = array
    return out


if __name__ == "__main__":
    dataset_path = "../../storage/user/kistern/coco"
    dataset = CocoKeypoints(dataset_path, seed=0, mode="train")
