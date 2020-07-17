import torch
import torchvision
import numpy as np
import pickle
from config import get_config, update_config
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator
from Utils.transforms import transforms_hr_train, transforms_hr_eval, transforms_hg_eval
from Utils import draw_detection, draw_clusters, draw_poses, pred_to_person, graph_cluster_to_persons, to_device, to_tensor
from Models import get_upper_bound_model
import matplotlib;matplotlib.use("Agg")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_subset = False
    config = get_config()
    config = update_config(config, f"../experiments/upper_bound/hrnet.yaml")

    transforms, transforms_inv = transforms_hr_train(config)


    """
    id_subset = [471913, 139124, 451781, 437856, 543272, 423250, 304336, 499425, 128905, 389185, 108762, 10925, 464037,
                 121938, 389815, 1145, 1431, 4309, 5028, 5294, 6777, 7650, 10130, 401534]
    """
    id_subset = [164013]
    ##################################

    model = get_upper_bound_model(config, device).to(device)
    model.eval()

    train_ids, _ = pickle.load(open("tmp/coco_17_mini_split.p", "rb"))  # mini_train_valid_split_4 old one
    heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
    img_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="train", img_ids=train_ids, year=17,
                               transforms=transforms, heatmap_generator=heatmap_generator)

    imgs_without_det = 0
    with torch.no_grad():
        for i in range(3500):  # just test the first 100 images
            # split batch
            img_id = img_set.img_ids[i]
            if use_subset:
                if img_id not in id_subset:
                    continue
            print(f"Iter : {i}")
            print(f"img_idx: {img_set.img_ids[i]}")
            img, _, masks, keypoints, factor_list = img_set[i]
            mask, keypoints, factor_list = to_tensor(device, masks[-1], keypoints, factor_list)
            img = img.to(device)[None]
            _, pred, preds_nodes, joint_det, joint_scores, edge_index, _, _, label_mask, _ = model(img, keypoints, mask, factor_list)

            # construct poses
            persons_pred_cc, _, _ = pred_to_person(joint_det, joint_scores, edge_index, pred, config.MODEL.GC.CC_METHOD)
            # construct solution by using only labeled edges (instead of corr clustering)
            sparse_sol_gt = torch.stack([edge_index[0, pred == 1], edge_index[1, pred == 1]])
            persons_pred_gt, _, _ = graph_cluster_to_persons(joint_det, joint_scores, sparse_sol_gt)  # might crash
            print(f"Num detection: {len(joint_det)}")
            print(f"Num edges : {len(edge_index[0])}")
            print(f"Num active edges: {(pred==1).sum()}")

            joint_det = joint_det.cpu().numpy().squeeze()
            preds_nodes = preds_nodes.cpu().numpy()
            clean_joint_det = joint_det[preds_nodes == 1]
            keypoints = keypoints.cpu().numpy().squeeze()

            img = np.array(transforms_inv(img.cpu().squeeze()))
            draw_detection(img, joint_det, keypoints,
                           fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_det.png",
                           output_size=256)
            draw_detection(img, clean_joint_det, keypoints,
                           fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_clean.png",
                           output_size=256)
            draw_poses(img, keypoints, fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_pose_gt.png",
                       output_size=256)
            if (pred==1).sum() == 0:
                imgs_without_det += 1
                continue
            draw_clusters(img, joint_det, sparse_sol_gt, fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_pose_gt_full.png",
                          output_size=256)
            # draw_poses(imgs, persons_pred_cc, fname=f"../tmp/test_construct_graph_img/{img_set.img_ids[i]}_pose_cc.png")
            draw_poses(img, persons_pred_gt, fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_pose_gt_labels.png",
                       output_size=256)

            # """
        print(f"num images without detection :{imgs_without_det}")


if __name__ == "__main__":
    main()
