import torch
import torchvision
import numpy as np
import pickle
import cv2
from config import get_config, update_config
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator, ScaleAwareHeatmapGenerator, CrowdPoseKeypoints, JointsGenerator
from Utils.transforms import transforms_hr_train, transforms_hr_eval, transforms_hg_eval
from Utils import (draw_detection, draw_clusters, draw_poses, pred_to_person, parse_refinement,
                   graph_cluster_to_persons, to_device, to_tensor, draw_detection_scoremap)
from Models import get_upper_bound_model
import matplotlib;matplotlib.use("Agg")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and False else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = "coco"
    dir_name = "images_type_agnostic_labels_valid"
    use_subset = False

    if dataset == "crowd_pose":
        config = get_config()
        config = update_config(config, f"../experiments/upper_bound/mmpose_hrnet.yaml")
    elif dataset == "coco":
        config = get_config()
        config = update_config(config, f"../experiments/upper_bound/hrnet.yaml")

    transforms, transforms_inv = transforms_hr_train(config)


    """
    id_subset = [471913, 139124, 451781, 437856, 543272, 423250, 304336, 499425, 128905, 389185, 108762, 10925, 464037,
                 121938, 389815, 1145, 1431, 4309, 5028, 5294, 6777, 7650, 10130, 401534]
    """
    id_subset = [8856]
    ##################################

    model = get_upper_bound_model(config, device).to(device)
    model.eval()


    if dataset == "coco":
        train_ids, valid_ids = pickle.load(open("tmp/coco_17_mini_split.p", "rb"))  # mini_train_valid_split_4 old one
        heatmap_generator = [HeatmapGenerator(128, 17, 2), HeatmapGenerator(256, 17, 2)]
        joint_generator = [JointsGenerator(30, 17, 128, True), JointsGenerator(30, 17, 256, True)]
        img_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids, year=17,
                                   transforms=transforms, heatmap_generator=heatmap_generator, joint_generator=joint_generator)
    elif dataset == "crowd_pose":
        heatmap_generator = [HeatmapGenerator(128, 14, 2), HeatmapGenerator(256, 14, 2)]
        joint_generator = [JointsGenerator(30, 14, 128, True), JointsGenerator(30, 14, 256, True)]
        img_set = CrowdPoseKeypoints(config.DATASET.ROOT, mini=True, seed=0, mode="val", transforms=transforms,
                                   heatmap_generator=heatmap_generator, joint_generator=joint_generator)

    imgs_without_det = 0
    with torch.no_grad():
        for i in range(200):  # just test the first 100 images
            # split batch
            img_id = img_set.img_ids[i]
            if use_subset:
                if img_id not in id_subset:
                    continue
            print(f"Iter : {i}")
            print(f"img_idx: {img_set.img_ids[i]}")
            img, heatmaps, masks, keypoints, factor_list, _ = img_set[i]
            mask, keypoints, factor_list = to_tensor(device, masks[-1], keypoints, factor_list)
            img = img.to(device)[None]
            sm_avg, output = model(img, keypoints, mask, factor_list)

            preds_nodes, preds_edges, preds_classes = output["preds"]["node"], output["preds"]["edge"], output["preds"]["class"]
            node_labels, edge_labels, class_labels = output["labels"]["node"], output["labels"]["edge"], output["labels"]["class"]
            joint_det, edge_index = output["graph"]["nodes"], output["graph"]["edge_index"]
            joint_scores = output["graph"]["detector_scores"]


            persons_pred_cc, _, _ = pred_to_person(joint_det, joint_scores, edge_index, preds_edges, preds_classes,
                                                   config.MODEL.GC.CC_METHOD, config.DATASET.NUM_JOINTS)
            # construct solution by using only labeled edges (instead of corr clustering)
            sparse_sol_gt = torch.stack([edge_index[0, preds_edges == 1], edge_index[1, preds_edges == 1]])
            persons_pred_gt, _, _ = graph_cluster_to_persons(joint_det, joint_scores, sparse_sol_gt, preds_classes,
                                                             config.DATASET.NUM_JOINTS)  # might crash
            print(f"Num detection: {len(joint_det)}")
            print(f"Num edges : {len(edge_index[0])}")
            print(f"Num active edges: {(preds_edges==1).sum()}")

            joint_det = joint_det.cpu().numpy().squeeze()
            joint_classes = preds_classes.cpu().numpy() if preds_classes is not None else None
            preds_nodes = preds_nodes.cpu().numpy()
            clean_joint_det = joint_det[preds_nodes == 1]
            keypoints = keypoints.cpu().numpy().squeeze()

            heatmaps = torch.from_numpy(heatmaps[0])
            img = np.array(transforms_inv(img.cpu().squeeze()))

            draw_detection(img.copy(), joint_det, np.copy(keypoints),
                           fname=f"tmp/test_construct_graph_img_{dataset}/{img_set.img_ids[i]}_det.png",
                           output_size=256)
            # draw scoremaps
            """
            draw_detection_scoremap(heatmaps, joint_det[preds_nodes==1.0], keypoints, 0,
                           fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_nose.png",
                           output_size=256)
            draw_detection_scoremap(sm_avg[0], joint_det[preds_nodes==1.0], keypoints, 0,
                                    fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_nose_avg.png",
                                    output_size=256)
            draw_detection_scoremap(sm_avg[0], joint_det[preds_nodes==1.0], keypoints, 6,
                                    fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_shoulder_avg.png",
                                    output_size=256)
            draw_detection_scoremap(heatmaps, joint_det[preds_nodes==1.0], keypoints, 6,
                                    fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_shoulder.png",
                                    output_size=256)
            draw_detection_scoremap(sm_avg[0], joint_det[preds_nodes==1.0], keypoints, 11,
                                    fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_hip_avg.png",
                                    output_size=256)
            draw_detection_scoremap(heatmaps, joint_det[preds_nodes==1.0], keypoints, 11,
                                    fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_hip.png",
                                    output_size=256)
            draw_detection_scoremap(heatmaps, joint_det[preds_nodes==1.0], keypoints, None,
                                    fname=f"tmp/test_construct_graph_img_crowd_pose/{img_set.img_ids[i]}_nose.png",
                                    output_size=256)
            # """
            # """
            draw_detection(img.copy(), clean_joint_det, np.copy(keypoints),
                           fname=f"tmp/{dir_name}/{img_set.img_ids[i]}_clean.png",
                           output_size=256)
            draw_detection(img.copy(), joint_det, np.copy(keypoints),
                           fname=f"tmp/{dir_name}/{img_set.img_ids[i]}_det.png",
                           output_size=256)
            draw_poses(img.copy(), np.copy(keypoints), fname=f"tmp/{dir_name}/{img_set.img_ids[i]}_pose_gt.png",
                       output_size=256)
            if (preds_edges==1).sum() == 0:
                imgs_without_det += 1
                continue
            # draw_clusters(img.copy(), joint_det, joint_classes, sparse_sol_gt,
            #              fname=f"tmp/{dir_name}/{img_set.img_ids[i]}_pose_gt_full.png", output_size=256)
            # draw_poses(imgs, persons_pred_cc, fname=f"../tmp/test_construct_graph_img/{img_set.img_ids[i]}_pose_cc.png")
            draw_poses(img.copy(), persons_pred_gt, fname=f"tmp/{dir_name}/{img_set.img_ids[i]}_pose_gt_from_labels.png",
                       output_size=256)
            # draw_poses(img.copy(), persons_pred_refine, fname=f"tmp/test_construct_graph_img/{img_set.img_ids[i]}_pose_gt_refined.png",
            #            output_size=256)
            # """

        print(f"num images without detection :{imgs_without_det}")


if __name__ == "__main__":
    main()
