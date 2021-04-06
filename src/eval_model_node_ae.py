import pickle, time
import torch
import sys
import numpy as np
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean
from tqdm import tqdm
from Utils.hr_utils import match_by_tag

from Utils.Utils import refine, refine_tag
from Utils.eval import EvalWriter, gen_ann_format
from config import get_config, update_config
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator, JointsGenerator
from Utils import pred_to_person_debug, pred_to_person, num_non_detected_points, adjust, to_tensor, calc_metrics, subgraph_mask, one_hot_encode, topk_accuracy
from Models.PoseEstimation import get_pose_model  #, get_pose_with_ref_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_hg_eval, transforms_hr_eval


def mpn_match_by_tag(joint_det, tag_k, scores, params):
    """

    :param joint_det: (N, 3)
    :param tags: (N, D)
    :param scores: (N)
    :param params:
    :return:
    """
    from munkres import Munkres
    def py_max_match(scores):
        m = Munkres()
        tmp = m.compute(scores)
        tmp = np.array(tmp).astype(np.int32)
        return tmp

    tag_k, loc_k, val_k = tag_k, joint_det[:, :2], scores
    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[1]))

    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]
        select = joint_det[:, 2] == idx

        tags = tag_k[select]
        joints = np.concatenate(
            (loc_k[select], val_k[select, None], tags), 1
        )
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]
        joints = joints[mask]

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            if params.ignore_too_much \
                    and len(grouped_keys) == params.max_num_people:
                continue

            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (
                        diff_normed,
                        np.zeros((num_added, num_added-num_grouped))+1e10
                    ),
                    axis=1
                )

            pairs = py_max_match(diff_normed)
            for row, col in pairs:
                if (
                        row < num_added
                        and col < num_grouped
                        and diff_saved[row][col] < params.tag_threshold
                ):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
    return ans

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
    config_name = "model_56_1_0_6_0_16_1"
    config = get_config()
    config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")
    eval_writer = EvalWriter(config, fname="runtime_test.txt")

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

    """
    if config.MODEL.WITH_LOCATION_REFINE:
        model = get_pose_with_ref_model(config, device)
    else:
        model = get_pose_model(config, device)
    # """
    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()
    model.test(True)
    # model.with_flip_kernel = False

    # baseline : predicting full connections
    # baseline: upper bound

    # eval model
    eval_node = {"acc": [], "prec": [], "rec": [], "f1": []}
    eval_edge = {"acc": [], "prec": [], "rec": [], "f1": []}
    eval_edge_masked = {"acc": [], "prec": [], "rec": [], "f1": []}
    eval_node_per_type = {i: {"acc": [], "prec": [], "rec": [], "f1": []} for i in range(17)}
    eval_roc_auc = {"node": {"pred": [], "label": [], "class": []}, "edge": None}
    eval_classes_baseline_acc = []
    eval_classes_acc = []
    eval_classes_top2_acc = []
    eval_joint_error_types = {"errors": [], "groups": [], "num_free_joints": []}
    def merge_dicts(dict_1, dict_2):
        for key in dict_1.keys():
            dict_1[key].append(dict_2[key])
        return dict_1

    anns = []
    avg_time = []
    eval_ids = []
    imgs_fully_det = []
    with torch.no_grad():
        for i in tqdm(range(len(eval_set))):
            eval_ids.append(eval_set.img_ids[i])

            img, _, masks, keypoints, factors, _ = eval_set[i]
            img = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)

            scoremaps, output = model(img, keypoints, masks, factors, with_logits=True)
            preds_nodes, preds_tags, preds_classes = output["preds"]["node"], output["preds"]["tag"], output["preds"]["class"]
            node_labels, class_labels = output["labels"]["node"],  output["labels"]["class"]
            joint_det = output["graph"]["nodes"]
            tags = output["graph"]["tags"]
            joint_scores = output["graph"]["detector_scores"]

            preds_classes = preds_classes[-1].softmax(dim=1) if preds_classes is not None else None
            preds_nodes = preds_nodes[-1].sigmoid().squeeze()
            preds_tags = preds_tags[-1]

            preds_classes_gt = one_hot_encode(class_labels, 17, torch.float) if class_labels is not None else None
            # preds_classes[node_labels == 1.0] = preds_classes_gt[node_labels == 1.0]
            preds_baseline_class = joint_det[:, 2]
            preds_baseline_class_one_hot = one_hot_encode(preds_baseline_class, 17, torch.float)
            # preds_baseline_class_one_hot[node_labels!=1.0] = preds_classes[node_labels!=1.0]
            # preds_classes[node_labels==1.0] = preds_baseline_class_one_hot[node_labels==1.0]

            result_nodes = torch.where(preds_nodes < config.MODEL.MPN.NODE_THRESHOLD, torch.zeros_like(preds_nodes), torch.ones_like(preds_nodes))
            n, _, _, _ = num_non_detected_points(joint_det.cpu(), keypoints[0].cpu(), factors[0].cpu())

            eval_roc_auc["node"]["pred"] += list(preds_nodes.cpu().numpy())
            eval_roc_auc["node"]["label"] += list(node_labels.cpu().numpy())
            eval_roc_auc["node"]["class"] += list(preds_classes.argmax(dim=1).cpu().numpy())


            eval_node = merge_dicts(eval_node, calc_metrics(result_nodes, node_labels))
            if preds_classes is not None:
                eval_classes_baseline_acc.append(calc_metrics(preds_baseline_class, class_labels, node_labels, 17)["acc"])
                eval_classes_acc.append(calc_metrics(preds_classes.argmax(dim=1), class_labels, node_labels, 17)["acc"])
                eval_classes_top2_acc.append(topk_accuracy(preds_classes, class_labels, 2, node_labels))

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            t1 = time.time()
            ann = perd_to_ann(scoremaps[0], preds_tags, joint_det, joint_scores, img_info, int(eval_set.img_ids[i]), config.DATASET.SCALING_TYPE, preds_classes,
                              tags[0])
            t2 = time.time()
            avg_time.append(t2-t1)


            if ann is not None:
                anns.append(ann)

        print("##################")
        print(f"Average time: {np.mean(avg_time)}")
        eval_writer.eval_coco(eval_set.coco, anns, np.array(eval_ids), "General Evaluation", "kpt_det.json")

        eval_writer.eval_metrics(eval_node, "Node metrics")
        eval_writer.eval_metric(eval_classes_baseline_acc, "Type classification using backbone detections as baseline")
        eval_writer.eval_metric(eval_classes_acc, "Type classification")
        eval_writer.eval_metric(eval_classes_top2_acc, "Type classification top 2")
        eval_writer.eval_roc_auc(eval_roc_auc, "Roc Auc scores")
        eval_writer.close()


def perd_to_ann(scoremaps, tags, joint_det, joint_scores, img_info, img_id,
                scaling_type, preds_classes, tagmap):
    class Params(object):
        def __init__(self):
            self.num_joints = 17
            self.max_num_people = 30

            self.detection_threshold = 0.1
            self.tag_threshold = 1.0
            self.use_detection_val = True
            self.ignore_too_much = False

            self.joint_order = [
                i - 1 for i in [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]

    joint_det = joint_det.cpu().numpy()
    joint_scores = joint_scores.cpu().numpy()
    tags = tags.cpu().numpy()[:, None]
    joint_det[:, 2] = preds_classes.cpu().numpy().argmax(axis=1)
    ans = mpn_match_by_tag(joint_det, tags, joint_scores, Params())

    if True:
        tagmap= tagmap.cpu().numpy()
        scoremaps = scoremaps.cpu().numpy()
        ans = refine(scoremaps, tagmap, ans)
    if True:
        ans = adjust(ans, scoremaps)
    if len(ans) == 0:
        return None
    persons_pred_orig = reverse_affine_map(ans.copy(), (img_info["width"], img_info["height"]), 512, scaling_type=scaling_type,
                                           min_scale=1)

    ann = gen_ann_format(persons_pred_orig, img_id)

    return ann


if __name__ == "__main__":
    main()
