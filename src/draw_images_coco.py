import torch
import numpy as np
import argparse, os
from torch_geometric.utils import subgraph
from tqdm import tqdm

from config import get_config, update_config, update_config_command
from data import CocoKeypoints_hr, HeatmapGenerator, JointsGenerator
from Utils import pred_to_person, save_valid_image, adjust, to_tensor, one_hot_encode, \
    refine, draw_edges_conf
from Models.PoseEstimation import get_pose_model
from Utils.transformations import reverse_affine_map, reverse_affine_map_points
from Utils.transforms import transforms_to_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate poses and draw the results")
    parser.add_argument("--config", help="Config file name for the experiment.", required=True, type=str)
    parser.add_argument("--out_dir", help="Name of the target directory.", required=True, type=str)
    parser.add_argument("options", help="Modifications to config file through the command line. "
                                        "Can be use to specify the evaluation setting (flip test, multi-scale etc.)"
                        , default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and False else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    # config_dir = "hybrid_class_agnostic_end2end"
    # config_name = "model_58_4"
    # config = get_config()
    # config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")

    args = parse_args()
    config = get_config()
    config = update_config(config, f"../experiments/{args.config}")
    config = update_config_command(config, args.options)

    os.makedirs(f"tmp/{args.out_dir}", exist_ok=True)

    heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
    joint_generator = [JointsGenerator(30, 17, 128, True),
                       JointsGenerator(30, 17, 256, True)]
    transforms, _ = transforms_to_tensor(config)
    scaling_type = "short_with_resize" if config.TEST.PROJECT2IMAGE else "short"
    eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=False, seed=0, mode="val", img_ids=None, year=17,
                                transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=False,
                                filter_empty=False, joint_generator=joint_generator)

    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    # baseline : predicting full connections
    # baseline: upper bound

    num_iter = len(eval_set)
    with torch.no_grad():

        for i in tqdm(range(num_iter)):
            img_id = eval_set.img_ids[i]

            img, _, masks, keypoints, factors, _ = eval_set[i]
            img = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)

            if keypoints.sum() == 0.0:
                continue

            scoremaps, output = model.multi_scale_inference(img, device, config, keypoints, factors)
            preds_nodes, preds_edges, preds_classes = output["preds"]["node"], output["preds"]["edge"], output["preds"]["class"]
            node_labels, edge_labels, class_labels = output["labels"]["node"], output["labels"]["edge"], output["labels"]["class"]
            joint_det, edge_index = output["graph"]["nodes"], output["graph"]["edge_index"]
            joint_scores = output["graph"]["detector_scores"]
            tags = output["graph"]["tags"]

            preds_nodes = preds_nodes[-1].sigmoid() if preds_nodes[-1] is not None else joint_scores
            preds_edges = preds_edges[-1].sigmoid().squeeze() if preds_edges[-1] is not None else None
            preds_classes = preds_classes[-1].softmax(dim=1) if preds_classes is not None else None
            preds_baseline_class = joint_det[:, 2]
            preds_baseline_class_one_hot = one_hot_encode(preds_baseline_class, 17, torch.float)
            # preds_classes_gt = one_hot_encode(class_labels, 17, torch.float) if class_labels is not None else preds_classes



            img_shape = (img.shape[3], img.shape[2])

            ann, person_labels = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_shape,
                              config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                              scaling_type, min(config.TEST.SCALE_FACTOR), config.TEST.ADJUST,
                              config.MODEL.MPN.NODE_THRESHOLD, preds_classes, False, joint_scores,
                              False)

            if ann is None:
                continue
            save_valid_image(img, ann, f"tmp/{args.out_dir}/{img_id}_pred.png", "coco")
            save_valid_image(img, keypoints[0], f"tmp/{args.out_dir}/{img_id}_gt.png", "coco")

            joint_det = joint_det.cpu().numpy()
            joint_det = reverse_affine_map_points(joint_det.copy(), img_shape,
                                                   scaling_type=scaling_type,
                                                   min_scale=min(config.TEST.SCALE_FACTOR))
            edge_index = edge_index.cpu().numpy()
            preds_nodes = preds_nodes.cpu().numpy()
            preds_edges = preds_edges.cpu().numpy()
            draw_edges_conf(img, joint_det, person_labels, preds_nodes, edge_index, preds_edges,
                            fname=f"tmp/{args.out_dir}/{img_id}")


def perd_to_ann(scoremaps, tags, joint_det, joint_scores, edge_index, pred, img_shape, input_size, img_id, cc_method,
                scaling_type, min_scale, adjustment, th, preds_classes, with_refine, score_map_scores, with_filter,
                scoring_method=None):
    if (score_map_scores > 0.1).sum() < 1:
        return None, None
    true_positive_idx = joint_scores > th
    edge_index, pred = subgraph(true_positive_idx, edge_index, pred)
    if edge_index.shape[1] != 0:
        pred[joint_det[edge_index[0, :], 2] == joint_det[
            edge_index[1, :], 2]] = 0.0  # set edge predictions of same types to zero
        persons_pred, _, person_labels = pred_to_person(joint_det, joint_scores, edge_index, pred, preds_classes, cc_method, 17)
    else:
        return None, None
    # persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))
    if len(persons_pred.shape) == 1:  # this means none persons were detected
        return None

    if with_refine and persons_pred[0, :, 2].sum() != 0:
        tags = tags.cpu().numpy()
        scoremaps = scoremaps.cpu().numpy()
        persons_pred = refine(scoremaps, tags, persons_pred)
    if adjustment:
        persons_pred = adjust(persons_pred, scoremaps)
    persons_pred_orig = reverse_affine_map(persons_pred.copy(), img_shape, input_size, scaling_type=scaling_type,
                                           min_scale=min_scale)

    return persons_pred_orig, person_labels


if __name__ == "__main__":
    main()
