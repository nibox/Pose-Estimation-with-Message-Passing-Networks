from pycocotools.coco import COCO
from pycocotools import mask as maskapi
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import pickle
import os

from Utils.transformations import kpt_affine, factor_affine, get_transform


class CocoKeypoints(Dataset):

    def __init__(self, path, mini=False, input_size=512, output_size=128, mode="train", seed=0, filter_empty=True,
                 img_ids=None, year=14, transforms=None):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.root_path = path
        # todo deal with different setups and with the different splits
        ann_path = f"{self.root_path}/annotations/person_keypoints_{mode}20{year}.json"
        self.coco = COCO(ann_path)
        self.transforms = transforms
        assert self.transforms is not None
        # self.input_size = input_size
        # self.output_size = output_size

        self.max_num_people = 30  # from github code
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
        for j in ann:
            if j["iscrowd"]:
                assert j["num_keypoints"] == 0
            # testing for the number of keypoints should be sufficient if the assertion above is true
            if j["num_keypoints"] == 0:
                encoding = maskapi.frPyObjects(j["segmentation"], img_height, img_width)
                inst_mask = maskapi.decode(encoding)
                if len(inst_mask.shape) == 3:
                    inst_mask = maskapi.decode(encoding).sum(axis=2)
                mask += inst_mask
        mask = (mask < 0.5).astype(np.float32)

        """
        # img processing
        # todo random scaling
        # todo random rotation
        # todo random horizontal flip
        center = np.array([img_width / 2, img_height / 2])  # probably for opencv?
        scale = max(img_height, img_width) / 200
        scale = np.array([scale, scale])

        dx = 0
        dy = 0
        center[0] += dx * center[0]
        center[1] += dy * center[1]

        mat_mask = get_transform(center, scale, (self.output_size, self.output_size), 0)[:2]
        mask = cv2.warpAffine((mask * 255).astype(np.uint8), mat_mask, (self.output_size, self.output_size)) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        mat = get_transform(center, scale, (self.input_size, self.input_size), 0)[:2]
        img = cv2.warpAffine(img, mat, (self.input_size, self.input_size)).astype(np.float32) / 255.0

        keypoints[:, :, :2] = kpt_affine(keypoints[:, :, :2], mat_mask).astype(np.float32)
        keypoints = pack_keypoints_for_batch(keypoints, max_num_people=self.max_num_people)

        factors = factor_affine(factors, mat_mask)
        factors = pack_for_batch(factors, self.max_num_people)
        """
        mask_list = [mask.copy()]
        keypoint_list = [keypoints.copy()]
        heatmaps = []

        if self.transforms is not None:
            img, mask, keypoint_list, factors = self.transforms(img, mask_list, keypoint_list, factors)

        factors = pack_for_batch(factors, 30)
        keypoints = pack_for_batch(keypoint_list[-1], 30)

        return img, heatmaps, mask, keypoints, factors

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
