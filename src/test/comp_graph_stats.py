import torch
from CocoKeypoints import CocoKeypoints
import numpy as np
import pickle
from Models.PoseEstimation.UpperBound import UpperBoundModel, default_config
from Utils.Utils import num_non_detected_points
from tqdm import tqdm


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
    config["use_gt"] = False
    config["use_focal_loss"] = True
    config["use_neighbours"] = False
    config["mask_crowds"] = True
    config["detect_threshold"] = 0.005 # default was 0.007
    config["edge_label_method"] = 4
    config["inclusion_radius"] = 7.5
    config["matching_radius"] = 0.1
    config["mpn_graph_type"] = "knn"

    ##################################

    model = load_backbone(config, device, pretrained_path=pretrained_path)
    model.eval()

    train_ids, valid_ids = pickle.load(open("../tmp/mini_train_valid_split_4.p", "rb"))
    ids = np.concatenate([train_ids, valid_ids])
    valid_ids = np.random.choice(valid_ids, 100, replace=False)
    img_set = CocoKeypoints(dataset_path, mini=True, seed=0, mode="train", img_ids=valid_ids)

    num_detections = []
    num_edges = []
    num_det_failures = []
    imbalance = []
    deg = []
    # todo number of not detected keypoints
    for i in tqdm(range(100)):  # just test the first 100 images
        # split batch
        imgs, masks, keypoints, factors = img_set.get_tensor(i, device)

        pred, joint_det, edge_index, edge_labels, _ = model(imgs, keypoints, masks, factors)
        #deg.append(degree(edge_index[1], len(joint_det)).mean())
        imbalance.append(edge_labels.mean().item())
        num_non_detected, num_gt = num_non_detected_points(joint_det, keypoints, 6.0,
                                                           config["use_gt"])

        num_detections.append(len(joint_det))
        num_edges.append(len(edge_labels))
        num_det_failures.append(float(num_non_detected) / num_gt)
    print(f"Average number of detections:{np.mean(num_detections)}")
    print(f"Std number of detections:{np.std(num_detections)}")
    print(f"Average number of edges:{np.mean(num_edges)}")
    print(f"Std number of edges:{np.std(num_edges)}")
    print(f"Average Imbalance: {np.mean(imbalance)}")
    print(f"Average detection failure: {np.mean(num_det_failures)}")
    #print(f"Average node degree: {np.mean(degree)}")


if __name__ == "__main__":
    main()
