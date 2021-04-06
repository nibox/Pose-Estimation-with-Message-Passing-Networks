import pickle
import torch
import torchvision
import sys
import numpy as np
from torch_geometric.utils import precision, recall, subgraph
from tqdm import tqdm

from config import get_config, update_config
from data import CocoKeypoints_test
from Utils import pred_to_person, num_non_detected_points, adjust, to_tensor, calc_metrics, subgraph_mask, one_hot_encode, refine
from Models.PoseEstimation import get_pose_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_to_tensor
from Utils.eval import gen_ann_format, EvalWriter, create_results_json


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
    # config_dir = "train"
    #config_name = sys.argv[1]
    config_name = "model_56_1_0_6_0_7_6"
    config = get_config()
    config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")

    transforms, _ = transforms_to_tensor(config)
    scaling_type = "short_with_resize" if config.TEST.PROJECT2IMAGE else "short"
    eval_set = CocoKeypoints_test(config.DATASET.ROOT, seed=0, year=17)

    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    # baseline : predicting full connections
    # baseline: upper bound

    # eval model
    anns = []

    eval_ids = []
    num_iter = 3 #len(eval_set)
    with torch.no_grad():

        for i in tqdm(range(num_iter)):
            eval_ids.append(eval_set.img_ids[i])

            img = torch.from_numpy(eval_set[i])
            img = img.to(device)[None]


            scoremaps, output = model.multi_scale_inference(img, config, None)
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



            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            ann = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info, config.DATASET.INPUT_SIZE,
                              int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, scaling_type,
                              min(config.TEST.SCALE_FACTOR), config.TEST.ADJUST, config.MODEL.MPN.NODE_THRESHOLD,
                              preds_classes, config.TEST.WITH_REFINE, joint_scores, False)

            if ann is not None:
                anns.append(ann)


        create_results_json(anns, config.LOG_DIR, "person_keypoints_test-dev2017_mpn_results.json")


def perd_to_ann(scoremaps, tags, joint_det, joint_scores, edge_index, pred, img_info, input_size, img_id, cc_method, scaling_type,
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
        keep = max_scores > 0.30
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

    ann = gen_ann_format(persons_pred_orig, img_id)
    return ann


if __name__ == "__main__":
    main()
