from pycocotools.coco import COCO
from pycocotools import mask as maskapi
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot(np.concatenate((kpt, kpt[:, 0:1] * 0 + 1), axis=1), mat.T).reshape(shape)


class CocoKeypoints(Dataset):

    def __init__(self, path, mini=False, input_size=512, output_size=128, **kwargs):
        np.random.seed(kwargs["seed"])
        torch.manual_seed(kwargs["seed"])

        self.root_path = path
        # todo deal with different setups and with the different splits
        ann_path = f"{self.root_path}/annotations/person_keypoints_{kwargs['mode']}2014.json"
        self.coco = COCO(ann_path)
        self.input_size = input_size
        self.output_size = output_size
        self.max_num_people = 30  # from github code

        self.cat_ids = self.coco.getCatIds(catNms=["person"])
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        if mini:
            self.img_ids = np.random.choice(self.img_ids, 4000)

    def __getitem__(self, idx):
        img_id = int(self.img_ids[idx])  # img_ids is array of numpy.int32
        print(img_id)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)  # keypoints, bbox, sem mask?
        img_info = self.coco.loadImgs(img_id)[0]
        ann = self.coco.loadAnns(ids=ann_ids)

        num_people = len(ann)
        img_height = img_info["height"]
        img_width = img_info["width"]

        # load image
        # todo reading the image for each iteration is probably not efficient
        with open(f"{self.root_path}/train2014/{img_info['file_name']}", "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))

        # load keypoints
        keypoints = np.zeros([num_people, 17, 3])  # 17 joints with xy position and visibility flag
        for i in range(num_people):
            keypoints[i] = np.array(ann[i]["keypoints"]).reshape([-1, 3])
        # in the code keypoints2 is created (seems to include only persons with atleast one visible keypoint)
        # not sure if it is necessary

        # load mask
        mask = np.zeros([img_height, img_width])
        for j in ann:
            if j["iscrowd"]:
                encoding = maskapi.frPyObjects(j["segmentation"], img_height, img_width)
                mask += maskapi.decode(encoding)
        mask = (mask < 0.5).astype(np.float32)

        # img processing
        # todo random scaling
        # todo random rotation
        # todo random horizontal flip
        center = np.array([img_width / 2, img_height / 2])  # probably for opencv?
        scale = max(img_height, img_width) / 200

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
        # todo keypoints ref: constructs list of positions of keypoint in the scoremap. Maybe i dont need id
        # todo random hue saturation brightness and contrast

        return img, mask, keypoints

    def __len__(self):
        return len(self.img_ids)


def ann_to_scoremap(ann, height, width):
    # todo preprocessing before score map creation (
    #
    pass


def pack_keypoints_for_batch(keypoints: np.array, max_num_people):
    out = np.zeros([max_num_people, 17, 3])
    out[:len(keypoints)] = keypoints
    return out



if __name__ == "__main__":
    dataset_path = "../../storage/user/kistern/coco"
    dataset = CocoKeypoints(dataset_path, seed=0, mode="train")

    dataset[0]
