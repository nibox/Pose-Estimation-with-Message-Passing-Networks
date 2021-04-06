import pickle
import os
import torch
import torchvision
import numpy as np
from torch_geometric.utils import precision, recall, subgraph
from tqdm import tqdm

from config import get_config, update_config
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator, JointsGenerator
from Utils import pred_to_person, num_non_detected_points, adjust, to_tensor, calc_metrics, subgraph_mask, one_hot_encode, refine
from Models.PoseEstimation import get_pose_model
from Utils.transformations import reverse_affine_map, reverse_affine_map_points
from Utils.transforms import transforms_to_tensor
from Utils.eval import gen_ann_format, EvalWriter
from Utils.hr_utils.multi_scales_testing import multiscale_keypoints
from Utils import *


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    config_dir = "hybrid_class_agnostic_end2end"
    # config_dir = "train"
    config_name = "model_56_1_0_6_0"
    output_dir = f"tmp/output_{config_name}_multi"
    os.makedirs(output_dir, exist_ok=True)
    config = get_config()
    config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")

    heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
    joint_generator = [JointsGenerator(30, 17, 128, True),
                       JointsGenerator(30, 17, 256, True)]
    transforms, transforms_inv = transforms_to_tensor(config)
    eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=False, seed=0, mode="val", img_ids=None, year=17,
                                transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=False,
                                filter_empty=False, joint_generator=joint_generator)

    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    # baseline : predicting full connections
    # baseline: upper bound

    # eval model
    scaling_type = "short_with_resize" if config.TEST.PROJECT2IMAGE else "short"

    num_iter = 50
    with torch.no_grad():

        for i in tqdm(range(num_iter)):
            img_id = eval_set.img_ids[i]

            img, _, masks, keypoints, factors, _ = eval_set[i]
            img = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)

            if keypoints.sum() == 0.0:
                keypoints = None
                factors = None

            scoremaps, output = model.multi_scale_inference(img, config, keypoints)
            preds_nodes, preds_edges, preds_classes = output["preds"]["node"], output["preds"]["edge"], output["preds"]["class"]
            node_labels, edge_labels, class_labels = output["labels"]["node"], output["labels"]["edge"], output["labels"]["class"]
            joint_det, edge_index = output["graph"]["nodes"], output["graph"]["edge_index"]
            joint_scores = output["graph"]["detector_scores"]

            preds_nodes = preds_nodes[-1].sigmoid()
            preds_edges = preds_edges[-1].sigmoid().squeeze() if preds_edges[-1] is not None else None
            preds_classes = preds_classes[-1].softmax(dim=1) if preds_classes is not None else None
            # preds_baseline_class = joint_det[:, 2]
            # preds_baseline_class_one_hot = one_hot_encode(preds_baseline_class, 17, torch.float)
            # preds_classes_gt = one_hot_encode(class_labels, 17, torch.float) if class_labels is not None else preds_classes

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            persons_pred = compute_poses(edge_index, joint_det, preds_classes, preds_edges, preds_nodes, scoremaps)
            if keypoints is not None:
                # preds_baseline_class = joint_det[:, 2]
                class_labels_one_hot = one_hot_encode(class_labels, 17, torch.float)
                persons_pred_gt = compute_poses(edge_index, joint_det, class_labels_one_hot, edge_labels, node_labels, scoremaps)
            else:
                persons_pred_gt = None

            img = transforms_inv(img.cpu().squeeze())
            img = np.array(img)
            joint_det = joint_det.cpu().numpy()
            persons_pred = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]),
                                                   scaling_type=scaling_type,
                                                   min_scale=min(config.TEST.SCALE_FACTOR))
            joint_det = reverse_affine_map_points(joint_det.copy(), (img_info["width"], img_info["height"]),
                                              scaling_type=scaling_type,
                                              min_scale=min(config.TEST.SCALE_FACTOR))
            draw_poses(img.copy(), persons_pred, f"{output_dir}/{i}_{img_id}_pred.png", output_size=512)
            # draw_poses(img, person_label, f"{output_dir}/{i}_{img_id}_gt_labels.png", output_size=output_size)
            if keypoints is not None:
                keypoints = keypoints[0].cpu().numpy()
                keypoints = keypoints[keypoints[:, :, 2].sum(axis=1) != 0]
                keypoints, factors = multiscale_keypoints(keypoints[None], factors, img, 512, 1.0, min(config.TEST.SCALE_FACTOR),
                                                          config.TEST.PROJECT2IMAGE)
                keypoints = keypoints.astype(np.float32).squeeze()

                keypoints = reverse_affine_map(keypoints.copy(), (img_info["width"], img_info["height"]),
                                               scaling_type=scaling_type,
                                               min_scale=min(config.TEST.SCALE_FACTOR))
                persons_pred_gt = reverse_affine_map(persons_pred_gt.copy(), (img_info["width"], img_info["height"]),
                                                  scaling_type=scaling_type,
                                                  min_scale=min(config.TEST.SCALE_FACTOR))
                draw_poses(img.copy(), persons_pred_gt, f"{output_dir}/{i}_{img_id}_pred_gt.png", output_size=512)

                draw_poses(img.copy(), keypoints, f"{output_dir}/{i}_{img_id}_gt.png", output_size=512)
                draw_detection(img.copy(), joint_det.copy(), keypoints, fname=f"{output_dir}/{i}_{img_id}_det.png", output_size=512)


def compute_poses(edge_index, joint_det, preds_classes, preds_edges, preds_nodes, scoremaps):
    true_positive_idx = preds_nodes > 0.1
    edge_index, pred = subgraph(true_positive_idx, edge_index, preds_edges)
    if edge_index.shape[1] != 0:
        persons_pred, _, _ = pred_to_person(joint_det, preds_nodes, edge_index, pred, preds_classes, "GAEC", 17)
    else:
        persons_pred = np.zeros([1, 17, 3])
    # persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))
    persons_pred = adjust(persons_pred, scoremaps[0])
    return persons_pred


def perd_to_person(scoremaps, joint_det, joint_scores, edge_index, pred, cc_method, th, preds_classes, score_map_scores):
    true_positive_idx = joint_scores > th
    edge_index, pred = subgraph(true_positive_idx, edge_index, pred)
    if edge_index.shape[1] != 0:
        persons_pred, _, _ = pred_to_person(joint_det, joint_scores, edge_index, pred, preds_classes, cc_method, 17)
    else:
        persons_pred = np.zeros([1, 17, 3])
    # persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))
    persons_pred = adjust(persons_pred, scoremaps)

    """
    persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]), scaling_type=scaling_type,
                                           min_scale=min_scale)
    """

    return persons_pred


if __name__ == "__main__":
    main()
