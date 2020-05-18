import torch
from CocoKeypoints import CocoKeypoints
import numpy as np
import pickle
from torch_geometric.utils import dense_to_sparse
from Models.PoseEstimation.UpperBound import UpperBoundModel, default_config
from Utils.Utils import load_model, draw_detection, draw_poses, draw_clusters, pred_to_person, graph_cluster_to_persons
from Utils.ConstructGraph import NaiveGraphConstructor
from Utils.dataset_utils import Graph
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph
import matplotlib;matplotlib.use("Agg")

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
    config["mpn_graph_type"] = "fully"

    cc_method = "GAEC"
    ##################################

    model = load_backbone(config, device, pretrained_path=pretrained_path)
    model.eval()

    train_ids, _ = pickle.load(open("../tmp/mini_train_valid_split_4.p", "rb"))
    img_set = CocoKeypoints(dataset_path, mini=True, seed=0, mode="train", img_ids=train_ids)

    imgs_without_det = 0
    for i in range(500):  # just test the first 100 images
        # split batch
        print(f"Iter : {i}")
        print(f"img_idx: {img_set.img_ids[i]}")
        imgs, masks, keypoints = img_set.get_tensor(i, device)

        pred, joint_det, edge_index, _, _ = model(imgs, keypoints, masks)

        # construct poses
        persons_pred_cc, _ = pred_to_person(joint_det, edge_index, pred, cc_method)
        # construct solution by using only labeled edges (instead of corr clustering)
        sparse_sol_gt = torch.stack([edge_index[0, pred == 1], edge_index[1, pred == 1]])
        persons_pred_gt, _ = graph_cluster_to_persons(joint_det, sparse_sol_gt)  # might crash
        print(f"Num detection: {len(joint_det)}")
        print(f"Num edges : {len(edge_index[0])}")
        print(f"Num active edges: {(pred==1).sum()}")

        # """
        imgs = imgs.squeeze()
        joint_det = joint_det.cpu().numpy().squeeze()
        keypoints = keypoints.cpu().numpy().squeeze()

        draw_detection(imgs, joint_det, keypoints,
                       fname=f"../tmp/test_construct_graph_img/{img_set.img_ids[i]}_det.png")
        draw_poses(imgs, keypoints, fname=f"../tmp/test_construct_graph_img/{img_set.img_ids[i]}_pose_gt.png")
        if (pred==1).sum() == 0:
            imgs_without_det += 1
            continue
        draw_clusters(imgs, joint_det, sparse_sol_gt, fname=f"../tmp/test_construct_graph_img/{img_set.img_ids[i]}_pose_gt_full.png")
        # draw_poses(imgs, persons_pred_cc, fname=f"../tmp/test_construct_graph_img/{img_set.img_ids[i]}_pose_cc.png")
        draw_poses(imgs, persons_pred_gt, fname=f"../tmp/test_construct_graph_img/{img_set.img_ids[i]}_pose_gt_labels.png")

        # """
    print(f"num images without detection :{imgs_without_det}")


if __name__ == "__main__":
    main()
