import pickle
import torch
import torchvision
import sys
import numpy as np
from torch_geometric.utils import precision, recall, subgraph
from tqdm import tqdm

from config import get_config, update_config
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator, JointsGenerator
from Utils import pred_to_person, pred_to_person_ae, adjust, to_tensor, calc_metrics, subgraph_mask, one_hot_encode, refine, adjust_joints
from Models.PoseEstimation import get_pose_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_to_tensor
from Utils.eval import gen_ann_format, EvalWriter, gen_ann_format_correct, gen_ann_format_mean


def merge_dicts(dict_1, dict_2):
    for key in dict_1.keys():
        dict_1[key].append(dict_2[key])
    return dict_1


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    config_dir = "hybrid_class_agnostic_end2end"
    # config_name = "model_58_4"
    # file_name = "t"
    config_name = sys.argv[1]
    file_name = sys.argv[2]
    config = get_config()
    config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")
    eval_writer = EvalWriter(config, fname=f"{file_name}.txt")

    heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
    joint_generator = [JointsGenerator(30, 17, 128, True),
                       JointsGenerator(30, 17, 256, True)]
    transforms, _ = transforms_to_tensor(config)
    scaling_type = "short_with_resize" if config.TEST.PROJECT2IMAGE else "short"
    eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=False, seed=0, mode="val", img_ids=None, year=17,
                                transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=False,
                                filter_empty=False, joint_generator=joint_generator)

    model = get_pose_model(config, device)
    if config.MODEL.PRETRAINED != "":
        state_dict = torch.load(config.MODEL.PRETRAINED, map_location=device)
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        assert config_name == "model_63"
    model.to(device)
    model.eval()

    # baseline : predicting full connections
    # baseline: upper bound

    # eval model
    anns = []
    anns_w_correct_scores = []
    anns_w_single = []
    anns_w_mean_scores = []
    anns_full = []
    anns_back = []
    anns_filter = []
    anns_with_people = []
    anns_w_refine = []
    anns_w_adjust = []
    anns_w_correct_adj = []
    anns_w_adjust_refine = []
    imgs_with_people = []

    # classification
    eval_node = {"acc": [], "prec": [], "rec": [], "f1": []}
    eval_edge = {"acc": [], "prec": [], "rec": [], "f1": []}
    eval_edge_masked = {"acc": [], "prec": [], "rec": [], "f1": []}
    eval_roc_auc = {"node": {"pred": [], "label": [], "class": []}, "edge": None}
    eval_classes_baseline_acc = []
    eval_classes_acc = []

    eval_ids = []
    num_iter = len(eval_set)
    with torch.no_grad():

        for i in tqdm(range(num_iter)):
            eval_ids.append(eval_set.img_ids[i])

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
            tags = output["graph"]["tags"]

            preds_nodes = preds_nodes[-1].sigmoid() if preds_nodes[-1] is not None else joint_scores
            preds_edges = preds_edges[-1].sigmoid().squeeze() if preds_edges[-1] is not None else None
            preds_classes = preds_classes[-1].softmax(dim=1) if preds_classes is not None else None
            preds_baseline_class = joint_det[:, 2]
            preds_baseline_class_one_hot = one_hot_encode(preds_baseline_class, 17, torch.float)
            # preds_classes_gt = one_hot_encode(class_labels, 17, torch.float) if class_labels is not None else preds_classes

            # classification metrics
            if node_labels is not None and preds_classes is not None:
                true_positive_idx = preds_nodes > config.MODEL.MPN.NODE_THRESHOLD
                mask = subgraph_mask(true_positive_idx, edge_index)
                result_edges = preds_edges * mask.float()

                result_edges = torch.where(result_edges < 0.5, torch.zeros_like(result_edges), torch.ones_like(result_edges))
                result_nodes = torch.where(preds_nodes < config.MODEL.MPN.NODE_THRESHOLD, torch.zeros_like(preds_nodes), torch.ones_like(preds_nodes))

                eval_roc_auc["node"]["pred"] += list(preds_nodes.cpu().numpy())
                eval_roc_auc["node"]["label"] += list(node_labels.cpu().numpy())
                eval_roc_auc["node"]["class"] += list(preds_classes.argmax(dim=1).cpu().numpy())

                eval_node = merge_dicts(eval_node, calc_metrics(result_nodes, node_labels))
                eval_edge = merge_dicts(eval_edge, calc_metrics(result_edges, edge_labels))
                if preds_classes is not None and node_labels.sum() > 1.0:
                    eval_classes_baseline_acc.append(calc_metrics(preds_baseline_class, class_labels, node_labels, 17)["acc"])
                    eval_classes_acc.append(calc_metrics(preds_classes.argmax(dim=1), class_labels, node_labels, 17)["acc"])

                """
                if node_labels.sum() > 1.0:
                    true_positive_idx = node_labels == 1.0
                    mask = subgraph_mask(true_positive_idx, edge_index)
                    result_edges = torch.where(preds_edges < 0.5, torch.zeros_like(result_edges), torch.ones_like(result_edges))
                    eval_edge_masked = merge_dicts(eval_edge_masked, calc_metrics(result_edges, edge_labels, mask.float()))
                # """


            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            """
            ann = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                              config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                              scaling_type, min(config.TEST.SCALE_FACTOR), config.TEST.ADJUST,
                              config.MODEL.MPN.NODE_THRESHOLD, preds_classes, config.TEST.WITH_REFINE, joint_scores,
                              False)
            ann_w_correct_scores = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges,
                                             img_info, config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]),
                                             config.MODEL.GC.CC_METHOD, scaling_type, min(config.TEST.SCALE_FACTOR),
                                             config.TEST.ADJUST, config.MODEL.MPN.NODE_THRESHOLD, preds_classes,
                                             config.TEST.WITH_REFINE, joint_scores, False, "correct")
            ann_w_mean_scores = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges,
                                          img_info, config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]),
                                          config.MODEL.GC.CC_METHOD, scaling_type, min(config.TEST.SCALE_FACTOR),
                                          config.TEST.ADJUST, config.MODEL.MPN.NODE_THRESHOLD, preds_classes,
                                          config.TEST.WITH_REFINE, joint_scores, False, "mean")
            ann_full = perd_to_ann_2(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info, config.DATASET.INPUT_SIZE,
                              int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, scaling_type,
                              min(config.TEST.SCALE_FACTOR), True, config.MODEL.MPN.NODE_THRESHOLD,
                              preds_classes, True, joint_scores, False)
            ann_filter = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                     config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                     scaling_type, min(config.TEST.SCALE_FACTOR), config.TEST.ADJUST,
                                     config.MODEL.MPN.NODE_THRESHOLD, preds_classes, config.TEST.WITH_REFINE,
                                     joint_scores, True)
            ann_bone = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                   config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                   scaling_type, min(config.TEST.SCALE_FACTOR), config.TEST.ADJUST,
                                   config.MODEL.MPN.NODE_THRESHOLD, preds_baseline_class_one_hot, False, joint_scores,
                                   False)
            ann_w_refine = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                       config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                       scaling_type, min(config.TEST.SCALE_FACTOR), False,
                                       config.MODEL.MPN.NODE_THRESHOLD, preds_classes, True, joint_scores, False)
            ann_w_adjust = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                       config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                       scaling_type, min(config.TEST.SCALE_FACTOR), True,
                                       config.MODEL.MPN.NODE_THRESHOLD, preds_classes, False, joint_scores, False)
            ann_w_correct_adj = perd_to_ann_2(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,config.DATASET.INPUT_SIZE,
                                       int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, scaling_type,
                                       min(config.TEST.SCALE_FACTOR), True, config.MODEL.MPN.NODE_THRESHOLD,
                                       preds_classes, False, joint_scores, False)
            ann_w_adjust_refine = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges,
                                              img_info, config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]),
                                              config.MODEL.GC.CC_METHOD, scaling_type, min(config.TEST.SCALE_FACTOR),
                                              True, config.MODEL.MPN.NODE_THRESHOLD, preds_classes, True, joint_scores,
                                              False)
            # """
            ann = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                              config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                              scaling_type, min(config.TEST.SCALE_FACTOR), config.TEST.ADJUST,
                              config.MODEL.MPN.NODE_THRESHOLD, preds_classes, config.TEST.WITH_REFINE, joint_scores,
                              False, scoring_method="mean")
            ann_w_adjust = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                       config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                       scaling_type, min(config.TEST.SCALE_FACTOR), True,
                                       config.MODEL.MPN.NODE_THRESHOLD, preds_classes, False, joint_scores, False, scoring_method="mean")
            ann_w_correct_scores = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                       config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                       scaling_type, min(config.TEST.SCALE_FACTOR), True,
                                       config.MODEL.MPN.NODE_THRESHOLD, preds_classes, False, joint_scores, False, scoring_method="correct")
            ann_w_single = perd_to_ann_single_joint(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                       config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                       scaling_type, min(config.TEST.SCALE_FACTOR), True,
                                       config.MODEL.MPN.NODE_THRESHOLD, preds_classes, False, joint_scores, False, scoring_method="mean")

            if ann is not None:
                anns.append(ann)
                anns_w_adjust.append(ann_w_adjust)
                anns_w_correct_scores.append(ann_w_correct_scores)
            if ann_w_single is not None:
                anns_w_single.append(ann_w_single)



        eval_writer.eval_coco(eval_set.coco, anns, np.array(eval_ids), "General Evaluation", "kpt_det_full_set_multi_scale.json")
        eval_writer.eval_coco(eval_set.coco, anns_w_adjust, np.array(eval_ids), "With adjustment and mean scores", "full_dt.json")
        eval_writer.eval_coco(eval_set.coco, anns_w_correct_scores, np.array(eval_ids), "With adjustment and correct scores", "full_dt.json")
        eval_writer.eval_coco(eval_set.coco, anns_w_single, np.array(eval_ids), "With single joint persons", "full_dt.json")

        eval_writer.eval_metrics(eval_node, "Node metrics")
        eval_writer.eval_metrics(eval_edge, "Edge metrics")
        eval_writer.eval_metrics(eval_edge_masked, "Edge metrics masked")
        eval_writer.eval_metric(eval_classes_baseline_acc, "Type classification using backbone detections as baseline")
        eval_writer.eval_metric(eval_classes_acc, "Type classification")
        eval_writer.eval_roc_auc(eval_roc_auc, "Roc Auc scores")
        eval_writer.close()


def perd_to_ann(scoremaps, tags, joint_det, joint_scores, edge_index, pred, img_info, input_size, img_id, cc_method,
                scaling_type, min_scale, adjustment, th, preds_classes, with_refine, score_map_scores, with_filter,
                scoring_method=None):
    if (score_map_scores > 0.1).sum() < 1:
        return None
    true_positive_idx = joint_scores > th
    edge_index, pred = subgraph(true_positive_idx, edge_index, pred)
    if edge_index.shape[1] != 0:
        pred[joint_det[edge_index[0, :], 2] == joint_det[
            edge_index[1, :], 2]] = 0.0  # set edge predictions of same types to zero
        persons_pred, _, _ = pred_to_person(joint_det, joint_scores, edge_index, pred, preds_classes, cc_method, 17)
    else:
        return None
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

    if with_refine and persons_pred[0, :, 2].sum() != 0:
        tags = tags.cpu().numpy()
        scoremaps = scoremaps.cpu().numpy()
        persons_pred = refine(scoremaps, tags, persons_pred)
    if adjustment:
        persons_pred = adjust(persons_pred, scoremaps)
    persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]), input_size, scaling_type=scaling_type,
                                           min_scale=min_scale)

    if scoring_method is None:
        ann = gen_ann_format(persons_pred_orig, img_id)
    elif scoring_method == "mean":
        ann = gen_ann_format_mean(persons_pred_orig, img_id)
    elif scoring_method == "correct":
        ann = gen_ann_format_correct(persons_pred_orig, img_id)
    else:
        raise NotImplementedError
    return ann


def perd_to_ann_single_joint(scoremaps, tags, joint_det, joint_scores, edge_index, pred, img_info, input_size, img_id, cc_method,
                scaling_type, min_scale, adjustment, th, preds_classes, with_refine, score_map_scores, with_filter,
                scoring_method=None):
    if (score_map_scores > 0.1).sum() < 1:
        return None
    true_positive_idx = joint_scores > th
    edge_index, pred = subgraph(true_positive_idx, edge_index, pred)

    pred[joint_det[edge_index[0, :], 2] == joint_det[
        edge_index[1, :], 2]] = 0.0  # set edge predictions of same types to zero
    persons_pred, _, _ = pred_to_person_ae(joint_det, joint_scores, edge_index, pred, preds_classes, cc_method, 17,
                                           allow_single_joint_persons=True)

    if len(persons_pred.shape) == 1:  # this means none persons were detected
        return None
        # persons_pred = np.zeros([1, 17, 3])

    if with_refine and persons_pred[0, :, 2].sum() != 0:
        tags = tags.cpu().numpy()
        scoremaps = scoremaps.cpu().numpy()
        persons_pred = refine(scoremaps, tags, persons_pred)
    if adjustment:
        persons_pred = adjust(persons_pred, scoremaps)
    persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]), input_size, scaling_type=scaling_type,
                                           min_scale=min_scale)

    if scoring_method is None:
        ann = gen_ann_format(persons_pred_orig, img_id)
    elif scoring_method == "mean":
        ann = gen_ann_format_mean(persons_pred_orig, img_id)
    elif scoring_method == "correct":
        ann = gen_ann_format_correct(persons_pred_orig, img_id)
    else:
        raise NotImplementedError
    return ann

def perd_to_ann_2(scoremaps, tags, joint_det, joint_scores, edge_index, pred, img_info, input_size, img_id, cc_method, scaling_type,
                min_scale, adjustment, th, preds_classes, with_refine, score_map_scores, with_filter):
    if (score_map_scores > 0.1).sum() < 1:
        return None
    if adjustment:
        joint_det = adjust_joints(joint_det.float(), scoremaps)
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

    if with_refine and persons_pred[0, :, 2].sum() != 0:
        tags = tags.cpu().numpy()
        scoremaps = scoremaps.cpu().numpy()
        persons_pred = refine(scoremaps, tags, persons_pred)
    persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]), input_size, scaling_type=scaling_type,
                                           min_scale=min_scale)

    ann = gen_ann_format(persons_pred_orig, img_id)
    return ann

if __name__ == "__main__":
    main()
