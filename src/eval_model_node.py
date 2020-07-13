import pickle
import torch
import torchvision
import numpy as np
from torch_geometric.utils import precision, recall, subgraph
from tqdm import tqdm

from config import get_config, update_config
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator
from Utils import pred_to_person, num_non_detected_points, adjust, to_tensor, calc_metrics, subgraph_mask
from Models.PoseEstimation import get_pose_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_hg_eval, transforms_hr_eval


def specificity(pred, target, num_classes):
    r"""Computes the recall
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}` of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    from torch_geometric.utils import true_negative, false_positive
    tn = true_negative(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tn / (tn + fp)
    out[torch.isnan(out)] = 0

    return out


def sprase_to_dense(edge_index, num_nodes):
    mat = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
    mat[edge_index[0], edge_index[1]] = 1.0
    return mat

"""
def reverse_affine_map(keypoints, img_size_orig):
    gt_width = img_size_orig[0]
    gt_height = img_size_orig[1]
    scale = max(gt_height, gt_width) / 200
    mat = get_transform((gt_width / 2, gt_height / 2), scale, (128, 128))
    inv_mat = np.linalg.inv(mat)[:2]  # might this lead to numerical errors?
    inv_mat = np.zeros([3, 3], dtype=np.float)
    inv_mat[0, 0], inv_mat[1, 1] = 1 / mat[0, 0], 1 / mat[1, 1]
    inv_mat[0, 2], inv_mat[1, 2] = -mat[0, 2] / mat[0, 0], -mat[1, 2] / mat[1, 1]
    inv_mat = inv_mat[:2]
    keypoints[:, :, :2] = kpt_affine(keypoints[:, :, :2], inv_mat)
    return keypoints
"""


def gen_ann_format(pred, image_id=0):
    """
    from https://github.com/princeton-vl/pose-ae-train
    Generate the json-style data for the output
    """
    ans = []
    for i in range(len(pred)):
        person = pred[i]
        # some score is used, not sure how it is used for evaluation.
        # todo what does the score do?
        # how are missing joints handled ?
        tmp = {'image_id': int(image_id), "category_id": 1, "keypoints": [], "score": 1.0}
        score = 0.0
        for j in range(len(person)):
            tmp["keypoints"] += [float(person[j, 0]), float(person[j, 1]), float(person[j, 2])]
            score += float(person[j, 2])
        tmp["score"] = score #/ 17.0
        ans.append(tmp)
    return ans


def eval_single_img(coco, dt, image_id, tmp_dir="tmp"):
    ann = [gen_ann_format(dt, image_id)]
    stats = coco_eval(coco, ann, [image_id], log=False)
    return stats[:2]


def coco_eval(coco, dt, image_ids, tmp_dir="tmp", log=True):
    """
    from https://github.com/princeton-vl/pose-ae-train
    Evaluate the result with COCO API
    """
    from pycocotools.cocoeval import COCOeval

    import json
    with open(tmp_dir + '/dt.json', 'w') as f:
        json.dump(sum(dt, []), f)

    # load coco
    coco_dets = coco.loadRes(tmp_dir + '/dt.json')
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    dataset_path = "../../storage/user/kistern/coco"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    config_name = "model_41"
    config = get_config()
    config = update_config(config, f"../experiments/train/{config_name}.yaml")

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
        transforms, _ = transforms_hr_eval(config)
        eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids, year=17,
                                    transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=False)
    else:
        raise NotImplementedError

    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    # baseline : predicting full connections
    # baseline: upper bound

    # eval model
    eval_node = {"acc": [], "prec": [], "rec": [], "f1": []}
    eval_edge = {"acc": [], "prec": [], "rec": [], "f1": []}
    def merge_dicts(dict_1, dict_2):
        for key in dict_1.keys():
            dict_1[key].append(dict_2[key])
        return dict_1

    anns = []
    anns_full = []
    anns_perf_node = []  # need second
    anns_perf_edge = []
    eval_ids = []
    imgs_fully_det = []
    with torch.no_grad():
        for i in tqdm(range(config.TEST.NUM_EVAL)):
            eval_ids.append(eval_set.img_ids[i])

            img, _, masks, keypoints, factors = eval_set[i]
            img = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)

            scoremaps, pred, preds_nodes, joint_det, joint_scores, edge_index, edge_labels, node_labels,  _, _ = model(img, keypoints, masks, factors, with_logits=True)

            preds_edges = pred[-1].sigmoid().squeeze() if pred[-1] is not None else None
            preds_nodes = preds_nodes[-1].sigmoid().squeeze()

            true_positive_idx = preds_nodes > 0.5
            mask = subgraph_mask(true_positive_idx, edge_index)
            result_edges = torch.zeros(edge_index.shape[1], dtype=torch.float, device=edge_index.device)
            if preds_edges is not None:
                result_edges[mask] = preds_edges
            result_edges = torch.where(result_edges < 0.5, torch.zeros_like(result_edges), torch.ones_like(result_edges))

            result_nodes = torch.where(preds_nodes < 0.5, torch.zeros_like(preds_nodes), torch.ones_like(preds_nodes))
            n, _ = num_non_detected_points(joint_det.cpu(), keypoints.cpu(), 6.0, config.MODEL.GC.USE_GT)

            eval_node = merge_dicts(eval_node, calc_metrics(result_nodes, node_labels))
            eval_edge = merge_dicts(eval_edge, calc_metrics(result_edges, edge_labels))

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            ann = perd_to_ann(scoremaps[0], joint_det, preds_nodes, edge_index, preds_edges, img_info, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD
                              , config.DATASET.SCALING_TYPE, config.TEST.ADJUST)
            perf_edges = edge_labels[mask]
            ann_perf_edge = perd_to_ann(scoremaps[0], joint_det, preds_nodes, edge_index, perf_edges, img_info, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD
                              , config.DATASET.SCALING_TYPE, config.TEST.ADJUST)

            anns.append(ann)
            anns_perf_edge.append(ann_perf_edge)
            if int(n) == 0:
                imgs_fully_det.append(eval_set.img_ids[i])
                anns_full.append(ann)
        model.train()
        model.stop_backbone_bn()
        for i in tqdm(range(config.TEST.NUM_EVAL)):
            eval_ids.append(eval_set.img_ids[i])

            img, _, masks, keypoints, factors = eval_set[i]
            img = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)

            scoremaps, pred, preds_nodes, joint_det, joint_scores, edge_index, edge_labels, node_labels, _, _ = model(
                img, keypoints, masks, factors, with_logits=True)

            preds_edges = pred[-1].sigmoid().squeeze() if pred[-1] is not None else None
            preds_nodes = preds_nodes[-1].sigmoid().squeeze()

            true_positive_idx = preds_nodes > 0.5
            true_positive_idx[node_labels == 1.0] = True
            mask = subgraph_mask(true_positive_idx, edge_index)
            result_edges = torch.zeros(edge_index.shape[1], dtype=torch.float, device=edge_index.device)
            if pred[-1] is not None:
                result_edges[mask] = preds_edges

            _, preds_edges = subgraph(node_labels > 0.5, edge_index, result_edges)

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            ann_perf_node = perd_to_ann(scoremaps[0], joint_det, node_labels, edge_index, preds_edges, img_info,
                              int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD
                              , config.DATASET.SCALING_TYPE, config.TEST.ADJUST)

            anns_perf_node.append(ann_perf_node)

        print("##################")
        print("General Evaluation")
        coco_eval(eval_set.coco, anns, np.array(eval_ids))
        print("Perfect edge prediction")
        coco_eval(eval_set.coco, anns_perf_edge, np.array(eval_ids))
        print("Perfect node prediction")
        coco_eval(eval_set.coco, anns_perf_node, np.array(eval_ids))
        print("##################")
        print("Real Evaluation on perfect images")
        print(f"Number of perfect images: {len(anns_full)}")
        coco_eval(eval_set.coco, anns_full, imgs_fully_det)
        for k in eval_edge.keys():
            eval_edge[k] = np.mean(eval_edge[k])
        for k in eval_node.keys():
            eval_node[k] = np.mean(eval_node[k])
        print("node metrics")
        print(eval_node)
        print("edge metrics")
        print(eval_edge)


def perd_to_ann(scoremaps, joint_det, joint_scores, edge_index, pred, img_info, img_id, cc_method, scaling_type, adjustment):
    true_positive_idx = joint_scores > 0.5
    edge_index, _ = subgraph(true_positive_idx, edge_index)
    if edge_index.shape[1] != 0:
        persons_pred, _, _ = pred_to_person(joint_det, joint_scores, edge_index, pred, cc_method)
    else:
        persons_pred = np.zeros([1, 17, 3])
    # persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))
    if len(persons_pred.shape) == 1:  # this means none persons were detected
        persons_pred = np.zeros([1, 17, 3])
    if adjustment:
        persons_pred = adjust(persons_pred, scoremaps)
    persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]), scaling_type=scaling_type)


    ann = gen_ann_format(persons_pred_orig, img_id)
    return ann


if __name__ == "__main__":
    main()
