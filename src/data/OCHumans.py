from pycocotools.coco import COCO
from pycocotools import mask as maskapi
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pickle
import os


class OCHumans(Dataset):

    def __init__(self, path, mode="val", seed=0, transforms=None, mask_crowds=False):
        assert mode in ["val", "test"]
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.root_path = path
        # todo deal with different setups and with the different splits
        ann_path = f"{self.root_path}/ochuman_coco_format_{mode}_range_0.00_1.00.json"
        self.coco = COCO(ann_path)
        self.mask_crowds = mask_crowds
        self.transforms = transforms
        # assert self.transforms is not None

        self.max_num_people = 30  # from github code
        self.cat_ids = self.coco.getCatIds(catNms=["person"])
        self.img_ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        assert self.transforms is not None
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
        kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        distance_factor = np.zeros([num_people, 17])  # this factor combines sigma with scale (taken from OKS computation)
        distance_factor[0:num_people] = (kpt_oks_sigmas * 2) ** 2
        keypoints_list = []
        factor_list = []
        scale_list = []
        for i in range(num_people):
            if ann[i]["num_keypoints"] > 0:
                keypoints_list.append(np.array(ann[i]["keypoints"]).reshape([-1, 3]))
                factor_list.append(np.array(kpt_oks_sigmas * 2) ** 2 * (ann[i]["area"] + np.spacing(1)) * 2.0)
                scale_list.append((ann[i]["area"] + np.spacing(1)) * 2.0)

        keypoints = np.array(keypoints_list).astype(np.float)
        factors = np.array(factor_list)
        scales = np.array(scale_list)
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

        if self.transforms is not None:
            img, mask, _, _ = self.transforms(img, mask, None, None)

        return img, mask

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

"""
class OCHumans(Dataset):

    def __init__(self, path, mode="val", seed=0, transforms=None, mask_crowds=True):
        assert mode in ["val", "test"]
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.root_path = path
        # todo deal with different setups and with the different splits
        ann_path = f"{self.root_path}/ochuman.json"

        self.ochuman = OCHuman(AnnoFile=ann_path, Filter="kpt&segm")

        self.mask_crowds = mask_crowds
        self.transforms = transforms
        self.max_num_people = 30
        self.img_ids = self.ochuman.getImgIds()
        if mode=="val":
            self.img_ids = self.img_ids[:2500]
            assert len(self.img_ids) == 2500
        elif mode=="test":
            self.img_ids = self.img_ids[2500:]
            assert len(self.img_ids) == 2231


    def __getitem__(self, idx):
        assert self.transforms is not None
        img_id = int(self.img_ids[idx])  # img_ids is array of numpy.int32
        data = self.ochuman.loadImgs(imgIds=[img_id])  # keypoints, bbox, sem mask?
        img_info = self.coco.loadImgs(img_id)[0]
        ann = data["annotations"]
        num_people = len(ann)
        img_height = data["height"]
        img_width = data["width"]

        # load image
        # todo reading the image for each iteration is probably not efficient
        with open(f"{self.root_path}/images/{data['file_name']}", "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))

        # load mask ( as in github repo)
        mask = np.zeros([img_height, img_width])
        mask = (mask < 0.5).astype(np.float32)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_ids)

"""
