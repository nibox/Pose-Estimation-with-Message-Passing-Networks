import torch
from CocoKeypoints import CocoKeypoints
import numpy as np
import pickle
from Models.PoseEstimation.UpperBound import UpperBoundModel, default_config
from Utils.Utils import num_non_detected_points


def load_backbone(config, device, pretrained_path=None):

    def rename_key(key):
        # assume structure is model.module.REAL_NAME
        return ".".join(key.split(".")[2:])

    model = UpperBoundModel(config)

    state_dict = torch.load(pretrained_path, map_location=device)
    state_dict_new = {rename_key(k): v for k, v in state_dict["state_dict"].items()}
    model.backbone.load_state_dict(state_dict_new)
    model.to(device)

    return model


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_path = "../../../storage/user/kistern/coco"
    pretrained_path = "../../PretrainedModels/pretrained/checkpoint.pth.tar"

    ##################################
    config = default_config
    config["cheat"] = False
    config["use_gt"] = True
    config["use_focal_loss"] = True
    config["use_neighbours"] = True
    config["mask_crowds"] = False
    config["detect_threshold"] = 0.007  # default was 0.007
    config["edge_label_method"] = 2
    config["inclusion_radius"] = 5.0

    ##################################

    model = load_backbone(config, device, pretrained_path=pretrained_path)
    model.eval()

    train_ids, valid_ids = pickle.load(open("../tmp/mini_train_valid_split_4.p", "rb"))
    ids = np.concatenate([train_ids, valid_ids])
    img_set = CocoKeypoints(dataset_path, mini=True, seed=0, mode="train", img_ids=ids)

    num_detections = []
    num_det_failures = []
    imbalance = []
    deg = []
    # todo number of not detected keypoints
    for i in range(3500, 4000):  # just test the first 100 images
        # split batch
        imgs, masks, keypoints = img_set[i]
        imgs = torch.from_numpy(imgs).to(device).unsqueeze(0)
        masks = torch.from_numpy(masks).to(device).unsqueeze(0)
        keypoints = torch.from_numpy(keypoints).to(device).unsqueeze(0)

        pred, joint_det, edge_index, edge_labels, _ = model(imgs, keypoints, masks)
        #deg.append(degree(edge_index[1], len(joint_det)).mean())
        imbalance.append(edge_labels.mean().item())
        num_non_detected, num_gt = num_non_detected_points(joint_det, keypoints)
        print(num_non_detected)

        num_detections.append(len(joint_det))
        num_det_failures.append(float(num_non_detected) / num_gt)
    print(f"Average Imbalance: {np.mean(imbalance)}")
    print(f"Average detection failure: {np.mean(num_det_failures)}")
    #print(f"Average node degree: {np.mean(degree)}")


if __name__ == "__main__":
    main()
