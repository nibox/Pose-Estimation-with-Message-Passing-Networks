import torch
from torch.utils.data import DataLoader
from CocoKeypoints import CocoKeypoints
import numpy as np
import pickle
import os


def create_train_validation_split(data_root, variant, force):
    # todo connect the preprosseing with the model selection (input size etc)
    # todo add validation
    if variant=="mini":
        tv_split_name = "tmp/mini_train_valid_split_4.p"
        if os.path.exists(tv_split_name):
            print("Mini dataset already exists!")
        else:
            print(f"Mini: Creating train validation split {tv_split_name}")
            data_set = CocoKeypoints(data_root, mini=True, seed=0, mode="train")
            train, valid = torch.utils.data.random_split(data_set, [3500, 500])
            assert len(data_set.img_ids) == len(set(data_set.img_ids))
            train_valid_split = [train.dataset.img_ids[train.indices], valid.dataset.img_ids[valid.indices]]
            pickle.dump(train_valid_split, open(tv_split_name, "wb"))
    elif variant=="mini_real":
        tv_split_name = "tmp/mini_real_train_valid_split_1.p"
        if os.path.exists(tv_split_name):
            print("Mini_real dataset already exists!")
        else:
            print(f"Mini_real: Creating train validation split {tv_split_name}")
            # we use the old mini train set so first step is to load it
            print(f"Loading train validation split {tv_split_name}")
            train_ids, _ = pickle.load(open("tmp/mini_train_valid_split_4.p", "rb"))

            # what follows is a bit hacky but whatever
            # hacky: mini=True,
            valid_set = CocoKeypoints(data_root, mini=True, seed=0, mode="val")

            _, valid = torch.utils.data.random_split(valid_set, [3500, 500])
            assert len(valid_set.img_ids) == len(set(valid_set.img_ids))
            train_valid_split = [train_ids, valid.dataset.img_ids[valid.indices]]
            pickle.dump(train_valid_split, open(tv_split_name, "wb"))
    elif variant == "real":
        raise NotImplementedError
    elif variant == "princeton":
        tv_split_name = "tmp/princeton_split.p"
        if os.path.exists(tv_split_name):
            print("Princeton split already exists!")
        else:
            print(f"Princeton: Creating train validation split {tv_split_name}")
            # we use the old mini train set so first step is to load it
            print(f"Loading train validation split {tv_split_name}")
            train_ids, _ = pickle.load(open("tmp/mini_train_valid_split_4.p", "rb"))

            # what follows is a bit hacky but whatever
            # hacky: mini=True,
            valid_ids = np.loadtxt("tmp/valid_id")
            set_intersection = set(list(valid_ids)).intersection(set(list(train_ids)))
            valid_ids = set(list(valid_ids)).difference(set_intersection)
            valid_ids = np.array(list(valid_ids))
            assert len(valid_ids)==500-45
            assert len(set(list(valid_ids)).intersection(set(list(train_ids)))) == 0
            train_valid_split = [train_ids, valid_ids]
            pickle.dump(train_valid_split, open(tv_split_name, "wb"))

def main():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_path = "../../storage/user/kistern/coco"
    split_variant = "princeton"  # mini, mini_real, real, princeton

    create_train_validation_split(dataset_path, variant=split_variant, force=False)

if __name__ == "__main__":
    main()
