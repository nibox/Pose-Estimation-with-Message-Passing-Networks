from pycocotools.coco import COCO
from pycocotools import mask as maskapi
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import pickle
import os

from Utils.transformations import kpt_affine, factor_affine, get_transform, get_multi_scale_size


class CocoKeypoints(Dataset):

    def __init__(self, path, mini=False, mode="train", seed=0, filter_empty=True,
                 img_ids=None, year=14, transforms=None, heatmap_generator=None, mask_crowds=True):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.root_path = path
        # todo deal with different setups and with the different splits
        ann_path = f"{self.root_path}/annotations/person_keypoints_{mode}20{year}.json"
        self.coco = COCO(ann_path)
        self.mask_crowds = mask_crowds
        self.transforms = transforms
        assert self.transforms is not None

        assert isinstance(heatmap_generator, (list, tuple)) or heatmap_generator is None
        self.num_scales = len(heatmap_generator) if heatmap_generator is not None else 0

        self.heatmap_generator = heatmap_generator

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

        # load mask ( as in github repo)
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

        """
        center = np.array([img_width / 2, img_height / 2])  # probably for opencv?
        scale = max(img_height, img_width) / 200
        scale = np.array([scale, scale])
        input_size, center, scale = get_multi_scale_size(img.shape[0], img.shape[1], self.input_size, 1.0, 1.0)

        dx = 0
        dy = 0
        center[0] += dx * center[0]
        center[1] += dy * center[1]

        mat_mask = get_transform(center, scale, (int(input_size[0]/2), int(input_size[1]/2)), 0)[:2]
        mask = cv2.warpAffine((mask * 255).astype(np.uint8), mat_mask, (int(input_size[0]/2), int(input_size[1]/2))) / 255.0
        # mat_mask = get_transform(center, scale, (self.output_size, self.output_size), 0)[:2]
        # mask = cv2.warpAffine((mask * 255).astype(np.uint8), mat_mask, (self.output_size, self.output_size)) / 255.0
        mask = (mask > 0.5).astype(np.float32)


        mat = get_transform(center, scale, input_size, 0)[:2]
        img = cv2.warpAffine(img, mat, input_size).astype(np.float32) / 255.0
        # mat = get_transform(center, scale, (self.input_size, self.input_size), 0)[:2]
        # img = cv2.warpAffine(img, mat, (self.input_size, self.input_size)).astype(np.float32) / 255.0

        keypoints[:, :, :2] = kpt_affine(keypoints[:, :, :2], mat_mask).astype(np.float32)
        keypoints = pack_keypoints_for_batch(keypoints, max_num_people=self.max_num_people)

        factors = factor_affine(factors, mat_mask)
        factors = pack_for_batch(factors, self.max_num_people)
        # """

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        keypoint_list = [keypoints.copy() for _ in range(self.num_scales)]
        heatmaps = []

        if self.transforms is not None:
            img, mask, keypoint_list, factors = self.transforms(img, mask_list, keypoint_list, factors)

        if self.heatmap_generator is not None:
            for scale_idx in range(self.num_scales):
                heatmap = self.heatmap_generator[scale_idx](keypoint_list[scale_idx])
                keypoint_list[scale_idx] = _filter_visible(keypoint_list[scale_idx], mask[scale_idx].shape)

                heatmaps.append(heatmap.astype(np.float32))
                mask_list[scale_idx] = mask_list[scale_idx].astype(np.float32)
                keypoint_list[scale_idx] = pack_for_batch(keypoint_list[scale_idx].astype(np.float32), 30)

        factors = pack_for_batch(factors, 30)

        return img, heatmaps, mask, keypoint_list[-1], factors

    def __len__(self):
        return len(self.img_ids)


def pack_keypoints_for_batch(keypoints: np.array, max_num_people):
    out = np.zeros([max_num_people, 17, 3])
    out[:len(keypoints)] = keypoints
    return out

def _filter_visible(keypoints, output_shape):
    out_w = output_shape[1]
    out_h = output_shape[0]
    vis_keypoints = keypoints.copy()
    for i in range(len(keypoints)):
        for j in range(17):
            x, y = keypoints[i, j, :2]
            if x < 0 or x >= out_w or y < 0 or y >= out_h:
                vis_keypoints[i, j] = 0.0
    return vis_keypoints


def pack_for_batch(array, max_num_people):
    new_shape = list(array.shape)
    new_shape[0] = max_num_people
    out = np.zeros(new_shape)
    out[:len(array)] = array
    return out


class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                            x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

if __name__ == "__main__":
    dataset_path = "../../storage/user/kistern/coco"
    dataset = CocoKeypoints(dataset_path, seed=0, mode="train")
