import pickle
import torch
import torchvision
import sys
import numpy as np
from torch_geometric.utils import precision, recall, subgraph
from tqdm import tqdm

from config import get_config, update_config
from data import OCHumans
from Utils import pred_to_person, num_non_detected_points, adjust, to_tensor, calc_metrics, subgraph_mask, one_hot_encode, refine
from Models.PoseEstimation import get_pose_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_to_tensor
from Utils.eval import gen_ann_format, EvalWriter
from Models.PoseEstimation.PoseEstimation import multi_scale_inference_hourglass


def merge_dicts(dict_1, dict_2):
    for key in dict_1.keys():
        dict_1[key].append(dict_2[key])
    return dict_1


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    config_dir = "hourglass"
    config_name = "model_70"
    file_name = "ochuman_single_scale_flip_confirmation"
    config = get_config()
    config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")
    eval_writer = EvalWriter(config, fname=f"{file_name}.txt")

    transforms, _ = transforms_to_tensor(config)
    eval_set = OCHumans('../../storage/user/kistern/OCHuman', seed=0, mode="val",
                        transforms=transforms, mask_crowds=False)
    # scaling_type = "long_with_multiscale" if config.TEST.PROJECT2IMAGE else "long"
    scaling_type = "long"

    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    # baseline : predicting full connections
    # baseline: upper bound

    # eval model
    anns = []
    anns_back = []
    anns_filter = []
    anns_w_refine = []
    anns_w_adjust = []
    anns_w_adjust_refine = []

    eval_ids = []
    num_iter = len(eval_set)
    with torch.no_grad():

        for i in tqdm(range(num_iter)):
            eval_ids.append(eval_set.img_ids[i])

            img, masks = eval_set[i]
            img = img.to(device)[None]

            scoremaps, output = multi_scale_inference_hourglass(model, img, config.TEST.SCALE_FACTOR, device, config, None, None)
            preds_nodes, preds_edges, preds_classes = output["preds"]["node"], output["preds"]["edge"], output["preds"]["class"]
            node_labels, edge_labels, class_labels = output["labels"]["node"], output["labels"]["edge"], output["labels"]["class"]
            joint_det, edge_index = output["graph"]["nodes"], output["graph"]["edge_index"]
            joint_scores = output["graph"]["detector_scores"]
            tags = output["graph"]["tags"]

            preds_nodes = preds_nodes[-1].sigmoid()
            preds_edges = preds_edges[-1].sigmoid().squeeze() if preds_edges[-1] is not None else None
            preds_classes = preds_classes[-1].softmax(dim=1) if preds_classes is not None else None
            preds_baseline_class = joint_det[:, 2]
            preds_baseline_class_one_hot = one_hot_encode(preds_baseline_class, 17, torch.float)
            preds_classes_gt = one_hot_encode(class_labels, 17, torch.float) if class_labels is not None else preds_classes

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            ann = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                              int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, scaling_type,
                              min(config.TEST.SCALE_FACTOR), False, config.MODEL.MPN.NODE_THRESHOLD,
                              preds_classes, False, joint_scores, False)
            ann_filter = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                              int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, scaling_type,
                              min(config.TEST.SCALE_FACTOR), config.TEST.ADJUST, config.MODEL.MPN.NODE_THRESHOLD,
                              preds_classes, config.TEST.WITH_REFINE, joint_scores, True)
            ann_bone = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                   int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, scaling_type,
                                   min(config.TEST.SCALE_FACTOR), config.TEST.ADJUST, config.MODEL.MPN.NODE_THRESHOLD,
                                   preds_baseline_class_one_hot, config.TEST.WITH_REFINE, joint_scores, False)
            ann_w_refine = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                       int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, scaling_type,
                                       min(config.TEST.SCALE_FACTOR), False, config.MODEL.MPN.NODE_THRESHOLD,
                                       preds_classes, True, joint_scores, False)
            ann_w_adjust = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                       int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, scaling_type,
                                       min(config.TEST.SCALE_FACTOR), True, config.MODEL.MPN.NODE_THRESHOLD,
                                       preds_classes, False, joint_scores, False)
            ann_w_adjust_refine = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges,
                                              img_info, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                              scaling_type, min(config.TEST.SCALE_FACTOR), True,
                                              config.MODEL.MPN.NODE_THRESHOLD, preds_classes, True, joint_scores, False)

            if ann is not None:
                anns.append(ann)
                anns_back.append(ann_bone)
                anns_w_adjust.append(ann_w_adjust)
                anns_w_adjust_refine.append(ann_w_adjust_refine)
                anns_w_refine.append(ann_w_refine)
                if ann_filter is not None:
                    anns_filter.append(ann_filter)


        eval_writer.eval_coco(eval_set.coco, anns, np.array(eval_ids), "General Evaluation", "kpt_det_full_set_multi_scale.json")
        eval_writer.eval_coco(eval_set.coco, anns_filter, np.array(eval_ids), "Using pose proposal filter")
        eval_writer.eval_coco(eval_set.coco, anns_back, np.array(eval_ids), "Using keypoint detector classes")
        eval_writer.eval_coco(eval_set.coco, anns_w_refine, np.array(eval_ids), "With refinment", "full_dt.json")
        eval_writer.eval_coco(eval_set.coco, anns_w_adjust, np.array(eval_ids), "With adjustment", "full_dt.json")
        eval_writer.eval_coco(eval_set.coco, anns_w_adjust_refine, np.array(eval_ids), "Wtih refinement + adjustment", "full_dt.json")

        eval_writer.close()


def perd_to_ann(scoremaps, tags, joint_det, joint_scores, edge_index, pred, img_info, img_id, cc_method, scaling_type,
                min_scale, adjustment, th, preds_classes, with_refine, score_map_scores, with_filter):
    if (score_map_scores > 0.1).sum() < 1:
       return None
    true_positive_idx = joint_scores > th
    edge_index, pred = subgraph(true_positive_idx, edge_index, pred)
    if edge_index.shape[1] != 0:
        pred[joint_det[edge_index[0, :], 2] == joint_det[
            edge_index[1, :], 2]] = 0.0  # set edge predictions of same types to zero
        persons_pred, _, _ = pred_to_person(joint_det, joint_scores, edge_index, pred, preds_classes, cc_method, 17)
    else:
        persons_pred = np.zeros([1, 17, 3])
    # persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))
    if len(persons_pred.shape) == 1:  # this means none persons were detected
        return None
        # persons_pred = np.zeros([1, 17, 3])

    if with_filter:
        max_scores = persons_pred[:, :, 2].max(axis=1)
        keep = max_scores > 0.25
        persons_pred = persons_pred[keep]
        if persons_pred.shape[0] == 0:
            return None

    if adjustment:
        persons_pred = adjust(persons_pred, scoremaps)

    if with_refine and persons_pred[0, :, 2].sum() != 0:
        tags = tags.cpu().numpy()
        scoremaps = scoremaps.cpu().numpy()
        persons_pred = refine(scoremaps, tags, persons_pred)
    persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]), 512, scaling_type=scaling_type,
                                           min_scale=min_scale)

    ann = gen_ann_format(persons_pred_orig, img_id)
    return ann


if __name__ == "__main__":
    main()
