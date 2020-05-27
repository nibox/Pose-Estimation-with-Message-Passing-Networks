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
    config["use_gt"] = False
    config["use_focal_loss"] = True
    config["use_neighbours"] = False
    config["mask_crowds"] = True
    config["detect_threshold"] = None  # default was 0.007
    config["edge_label_method"] = 4
    config["inclusion_radius"] = 7.5
    config["matching_radius"] = 0.1
    config["mpn_graph_type"] = "topk"

    cc_method = "GAEC"
    use_subset = True
    id_subset = [471913, 139124, 451781, 437856, 543272, 423250, 304336, 499425, 128905, 389185, 108762, 10925, 464037,
                 121938, 389815, 1145, 1431, 4309, 5028, 5294, 6777, 7650, 10130, 401534]
    ##################################

    model = load_backbone(config, device, pretrained_path=pretrained_path)
    model.eval()

    train_ids, valid_ids = pickle.load(open("../tmp/mini_train_valid_split_4.p", "rb"))
    ids = np.concatenate([train_ids, valid_ids])
    img_set = CocoKeypoints(dataset_path, mini=True, seed=0, mode="train", img_ids=ids)

    imgs_without_det = 0
    for i in range(4000):  # just test the first 100 images
        # split batch
        img_id = img_set.img_ids[i]
        if use_subset:
            if img_id not in id_subset:
                continue
        print(f"Iter : {i}")
        print(f"img_idx: {img_set.img_ids[i]}")
        imgs, masks, keypoints, factor_list = img_set.get_tensor(i, device)

        pred, joint_det, edge_index, _, label_mask = model(imgs, keypoints, masks, factor_list)

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
