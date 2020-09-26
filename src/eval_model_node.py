import pickle
import torch
import numpy as np
from torch_geometric.utils import subgraph
from tqdm import tqdm

from Utils.eval import EvalWriter, gen_ann_format
from config import get_config, update_config
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator, JointsGenerator
from Utils import parse_refinement, pred_to_person, num_non_detected_points, adjust, to_tensor, calc_metrics, subgraph_mask, one_hot_encode, topk_accuracy
from Models.PoseEstimation import get_pose_model, get_pose_with_ref_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_hg_eval, transforms_hr_eval


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    dataset_path = "../../storage/user/kistern/coco"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    config_dir = "hybrid_class_agnostic_end2end"
    # config_dir = "node_classification_from_scratch"
    # config_dir = "regression_mini"
    # config_dir = "train"
    # config_dir = "node_classification_from_scratch"
    config_name = "model_56_1_0_6_0_12"
    config = get_config()
    config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")
    eval_writer = EvalWriter(config, fname="eval_10_mini.txt")

    if config.TEST.SPLIT == "mini":
        train_ids, valid_ids = pickle.load(open("tmp/mini_train_valid_split_4.p", "rb"))
        assert len(set(train_ids).intersection(set(valid_ids))) == 0

        transforms = transforms_hg_eval(config)
        eval_set = CocoKeypoints_hg(dataset_path, mini=True, seed=0, mode="train",
                                    img_ids=valid_ids, transforms=transforms)
    elif config.TEST.SPLIT == "mini_real":
        train_ids, valid_ids = pickle.load(open("tmp/mini_real_train_valid_split_1.p", "rb"))
        assert len(set(train_ids).intersection(set(valid_ids))) == 0
        eval_set = CocoKeypoints_hg(dataset_path, mini=True, seed=0, mode="val", img_ids=valid_ids)
    elif config.TEST.SPLIT == "princeton":
        train_ids, valid_ids = pickle.load(open("tmp/princeton_split.p", "rb"))
        assert len(set(train_ids).intersection(set(valid_ids))) == 0
        transforms = transforms_hg_eval(config)
        eval_set = CocoKeypoints_hg(dataset_path, mini=True, seed=0, mode="train", img_ids=valid_ids,
                                    transforms=transforms, mask_crowds=False)
    elif config.TEST.SPLIT == "coco_17_mini":
        _, valid_ids = pickle.load(open("tmp/coco_17_mini_split.p", "rb"))  # mini_train_valid_split_4 old one
        heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
        joint_generator = [JointsGenerator(30, 17, 128, True),
                           JointsGenerator(30, 17, 256, True)]
        transforms, _ = transforms_hr_eval(config)
        eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids, year=17,
                                    transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=False,
                                    joint_generator=joint_generator)
    else:
        raise NotImplementedError

    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()
    model.test(True)

    # baseline : predicting full connections
    # baseline: upper bound

    # eval model
    eval_node = {"acc": [], "prec": [], "rec": [], "f1": []}
    eval_edge = {"acc": [], "prec": [], "rec": [], "f1": []}
    eval_edge_masked = {"acc": [], "prec": [], "rec": [], "f1": []}
    eval_node_per_type = {i: {"acc": [], "prec": [], "rec": [], "f1": []} for i in range(17)}
    eval_classes_baseline_acc = []
    eval_classes_acc = []
    eval_classes_top2_acc = []
    def merge_dicts(dict_1, dict_2):
        for key in dict_1.keys():
            dict_1[key].append(dict_2[key])
        return dict_1

    anns = []
    anns_full = []
    anns_perf_node = []  # need second
    anns_perf_edge = []
    anns_class_infl = []
    anns_cc_cluster = []
    anns_refined = []
    anns_with_heatmap_values = []
    eval_ids = []
    imgs_fully_det = []
    with torch.no_grad():
        for i in tqdm(range(config.TEST.NUM_EVAL)):
            eval_ids.append(eval_set.img_ids[i])

            img, _, masks, keypoints, factors, _ = eval_set[i]
            img = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)

            scoremaps, output = model(img, keypoints, masks, factors, with_logits=True)
            preds_nodes, preds_edges, preds_classes = output["preds"]["node"], output["preds"]["edge"], output["preds"]["class"]
            node_labels, edge_labels, class_labels = output["labels"]["node"], output["labels"]["edge"], output["labels"]["class"]
            joint_det, edge_index = output["graph"]["nodes"], output["graph"]["edge_index"]
            tags = output["graph"]["tags"]
            joint_scores = output["graph"]["detector_scores"]

            preds_edges = preds_edges[-1].sigmoid().squeeze() if preds_edges[-1] is not None else None
            preds_classes = preds_classes[-1].softmax(dim=1) if preds_classes is not None else None
            preds_nodes = preds_nodes[-1].sigmoid().squeeze()

            preds_classes_gt = one_hot_encode(class_labels, 17, torch.float) if class_labels is not None else None
            # preds_classes[node_labels == 1.0] = preds_classes_gt[node_labels == 1.0]
            preds_baseline_class = joint_det[:, 2]
            preds_baseline_class_one_hot = one_hot_encode(preds_baseline_class, 17, torch.float)
            # preds_baseline_class_one_hot[node_labels!=1.0] = preds_classes[node_labels!=1.0]
            # preds_classes[node_labels==1.0] = preds_baseline_class_one_hot[node_labels==1.0]

            true_positive_idx = preds_nodes > config.MODEL.MPN.NODE_THRESHOLD
            mask = subgraph_mask(true_positive_idx, edge_index)
            result_edges = preds_edges * mask.float()

            result_edges = torch.where(result_edges < 0.5, torch.zeros_like(result_edges), torch.ones_like(result_edges))
            result_nodes = torch.where(preds_nodes < config.MODEL.MPN.NODE_THRESHOLD, torch.zeros_like(preds_nodes), torch.ones_like(preds_nodes))
            n, _, _, _ = num_non_detected_points(joint_det.cpu(), keypoints[0].cpu(), factors[0].cpu())

            eval_node = merge_dicts(eval_node, calc_metrics(result_nodes, node_labels))
            eval_edge = merge_dicts(eval_edge, calc_metrics(result_edges, edge_labels))
            if preds_classes is not None:
                eval_classes_baseline_acc.append(calc_metrics(preds_baseline_class, class_labels, node_labels, 17)["acc"])
                eval_classes_acc.append(calc_metrics(preds_classes.argmax(dim=1), class_labels, node_labels, 17)["acc"])
                eval_classes_top2_acc.append(topk_accuracy(preds_classes, class_labels, 2, node_labels))

            """
            if node_labels.sum() > 1.0:
                true_positive_idx = node_labels == 1.0
                mask = subgraph_mask(true_positive_idx, edge_index)
                result_edges_2 = torch.where(preds_edges < 0.5, torch.zeros_like(result_edges), torch.ones_like(result_edges))
                eval_edge_masked = merge_dicts(eval_edge_masked, calc_metrics(result_edges_2[mask], edge_labels[mask]))
            """
            mask = subgraph_mask(node_labels > 0.5, edge_index)
            half_perfect_edge_preds = preds_edges.clone()
            half_perfect_edge_preds[mask] = edge_labels[mask]

            for j in range(17):
                m = joint_det[:, 2] == j
                eval_node_per_type[j] = merge_dicts(eval_node_per_type[j], calc_metrics(result_nodes[m], node_labels[m]))


            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            ann = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                              int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, config.DATASET.SCALING_TYPE,
                              config.TEST.ADJUST, config.MODEL.MPN.NODE_THRESHOLD, preds_classes, False)
            ann_heatmap = perd_to_ann(scoremaps[0], tags[0], joint_det, joint_scores, edge_index, preds_edges, img_info,
                              int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, config.DATASET.SCALING_TYPE,
                              config.TEST.ADJUST, 0.1, preds_classes, True)
            ann_cc = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                 int(eval_set.img_ids[i]), "threshold", config.DATASET.SCALING_TYPE, config.TEST.ADJUST,
                                 config.MODEL.MPN.NODE_THRESHOLD, preds_classes, config.TEST.WITH_REFINE)
            ann_refined = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                                      int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, config.DATASET.SCALING_TYPE,
                                      config.TEST.ADJUST, config.MODEL.MPN.NODE_THRESHOLD, preds_classes, True)

            ann_perf_edge = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges,
                                        img_info, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                        config.DATASET.SCALING_TYPE, config.TEST.ADJUST,
                                        config.MODEL.MPN.NODE_THRESHOLD, preds_classes_gt, config.TEST.WITH_REFINE,
                                        )
            ann_perf_node = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes * node_labels, edge_index, preds_edges,
                                        img_info, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                        config.DATASET.SCALING_TYPE, config.TEST.ADJUST,
                                        config.MODEL.MPN.NODE_THRESHOLD, preds_classes, config.TEST.WITH_REFINE)
            ann_class_infl = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes * node_labels, edge_index, edge_labels,
                                         img_info, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                         config.DATASET.SCALING_TYPE, config.TEST.ADJUST,
                                         config.MODEL.MPN.NODE_THRESHOLD, preds_classes, config.TEST.WITH_REFINE)

            anns.append(ann)
            anns_refined.append(ann_refined)
            anns_perf_edge.append(ann_perf_edge)
            anns_perf_node.append(ann_perf_node)
            anns_class_infl.append(ann_class_infl)
            anns_cc_cluster.append(ann_cc)
            anns_with_heatmap_values.append(ann_heatmap)
            if int(n) == 0:
                imgs_fully_det.append(eval_set.img_ids[i])
                anns_full.append(ann)

        print("##################")
        eval_writer.eval_coco(eval_set.coco, anns, np.array(eval_ids), "General Evaluation", "kpt_det.json")
        eval_writer.eval_coco(eval_set.coco, anns_perf_edge, np.array(eval_ids), "Perfect classification")
        eval_writer.eval_coco(eval_set.coco, anns_perf_node, np.array(eval_ids), "Perfect node prediction")
        eval_writer.eval_coco(eval_set.coco, anns_class_infl, np.array(eval_ids), "Perfect node and edge prediction")
        eval_writer.eval_coco(eval_set.coco, anns_cc_cluster, np.array(eval_ids), "Thresholding and connected components")
        eval_writer.eval_coco(eval_set.coco, anns_refined, np.array(eval_ids), "Refined Poses")
        eval_writer.eval_coco(eval_set.coco, anns_with_heatmap_values, np.array(eval_ids), "Using heatmap scores with refinement")
        eval_writer.eval_coco(eval_set.coco, anns_full, imgs_fully_det, f"Evaluation on perfect images {len(anns_full)}")

        eval_writer.eval_metrics(eval_node, "Node metrics")
        eval_writer.eval_metrics(eval_edge, "Edge metrics")
        eval_writer.eval_metrics(eval_edge_masked, "Edge metrics masked")
        eval_writer.eval_metric(eval_classes_baseline_acc, "Type classification using backbone detections as baseline")
        eval_writer.eval_metric(eval_classes_acc, "Type classification")
        eval_writer.eval_metric(eval_classes_top2_acc, "Type classification top 2")
        eval_writer.eval_part_metrics(eval_node_per_type, "Node metrics per type")

        eval_writer.close()


def perd_to_ann(scoremaps, tags, joint_det, joint_scores, edge_index, preds_edges, img_info, img_id, cc_method,
                scaling_type, adjustment, th, preds_classes, with_refine):
    true_positive_idx = joint_scores > th
    edge_index, preds_edges = subgraph(true_positive_idx, edge_index, preds_edges)
    if edge_index.shape[1] != 0:
        persons_pred, _, _ = pred_to_person(joint_det, joint_scores, edge_index, preds_edges, preds_classes, cc_method)
    else:
        persons_pred = np.zeros([1, 17, 3])
    # persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))
    if len(persons_pred.shape) == 1:  # this means none persons were detected
        persons_pred = np.zeros([1, 17, 3])
    if with_refine and persons_pred[0, :, 2].sum() != 0:
        tags = tags.cpu().numpy()
        scoremaps = scoremaps.cpu().numpy()
        persons_pred = refine(scoremaps, tags, persons_pred)
    if adjustment:
        persons_pred = adjust(persons_pred, scoremaps)
    persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]), scaling_type=scaling_type)

    ann = gen_ann_format(persons_pred_orig, img_id)
    return ann


if __name__ == "__main__":
    main()
