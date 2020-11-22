from pycocotools.coco import COCO
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CocoKeypoints_test(Dataset):

    def __init__(self, path, seed=0, year=14):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.root_path = path
        self.transforms = torchvision.transforms.ToTensor
        # todo deal with different setups and with the different splits
        ann_path = f"{self.root_path}/annotations/image_info_test-dev20{year}.json"
        self.coco = COCO(ann_path)

        self.cat_ids = self.coco.getCatIds(catNms=["person"])
        self.img_ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        img_id = int(self.img_ids[idx])  # img_ids is array of numpy.int32
        img_info = self.coco.loadImgs(img_id)[0]

        # load image
        # todo reading the image for each iteration is probably not efficient
        with open(f"{self.root_path}/test2017/{img_info['file_name']}", "rb") as f:
            img = np.array(Image.open(f).convert("RGB"))

        return self.transforms(img)

    def __len__(self):
        return len(self.img_ids)
