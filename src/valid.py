import pickle
import torch
import torchvision
import numpy as np
from torch_geometric.utils import precision, recall, subgraph
from tqdm import tqdm

from config import get_config, update_config
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator, JointsGenerator
from Utils import *
from Models.PoseEstimation import get_pose_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_hg_eval, transforms_hr_eval
from Utils.eval import gen_ann_format, EvalWriter


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    config_dir = "hybrid_class_agnostic_end2end"
    # config_dir = "train"
    config_name = "model_56_1_0"
    config = get_config()
    config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")
    eval_writer = EvalWriter(config, fname="eval_10_full_with_refine_without_masking.txt")

    heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
    joint_generator = [JointsGenerator(30, 17, 128, True),
                       JointsGenerator(30, 17, 256, True)]
    transforms, _ = transforms_hr_eval(config)
    eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=False, seed=0, mode="val", img_ids=None, year=17,
                                transforms=transforms, heatmap_generator=heatmap_generator, joint_generator=joint_generator,
                                mask_crowds=False,
                                filter_empty=False)

    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    # baseline : predicting full connections
    # baseline: upper bound

    # eval model

    anns = []
    anns_with_people = []
    imgs_with_people = []
    eval_ids = []
    with torch.no_grad():
        for i in tqdm(range(len(eval_set))):
            eval_ids.append(eval_set.img_ids[i])

            img, _, masks, keypoints, factors, _ = eval_set[i]
            img = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)

            if keypoints.sum() == 0.0:
                keypoints = None

            # scoremaps, pred, preds_nodes, preds_classes, joint_det, joint_scores, edge_index, _, node_labels, class_labels, _, _, _ = model(img, None, masks, factors, with_logits=True)

            scoremaps, output = model(img, keypoints, masks, factors, with_logits=True)
            preds_nodes, preds_edges, preds_classes = output["preds"]["node"], output["preds"]["edge"], output["preds"]["class"]
            joint_det, edge_index = output["graph"]["nodes"], output["graph"]["edge_index"]
            det_scores = output["graph"]["detector_scores"]
            tags = output["graph"]["tags"]

            preds_nodes = preds_nodes[-1].sigmoid()
            preds_edges = preds_edges[-1].sigmoid().squeeze() if preds_edges[-1] is not None else None
            preds_classes = preds_classes[-1].softmax(dim=1) if preds_classes is not None else None
            preds_baseline_class = joint_det[:, 2]
            preds_baseline_class_one_hot = one_hot_encode(preds_baseline_class, 17, torch.float)
            # preds_classes_gt = one_hot_encode(class_labels, 17, torch.float) if class_labels is not None else preds_classes

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            ann = perd_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_info,
                              int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD, config.DATASET.SCALING_TYPE,
                              config.TEST.ADJUST, config.MODEL.MPN.NODE_THRESHOLD, preds_classes, det_scores, True)

            if ann is not None:
                anns.append(ann)
            if keypoints is not None and ann is not None:
                imgs_with_people.append(int(eval_set.img_ids[i]))
                anns_with_people.append(ann)


        eval_writer.eval_coco(eval_set.coco, anns, np.array(eval_ids), "General Evaluation", "kpt_det_full_set.json")
        eval_writer.eval_coco(eval_set.coco, anns_with_people, np.array(imgs_with_people), f"General Evaluation on not empty images {len(anns_with_people)}")
        eval_writer.close()


def perd_to_ann(scoremaps, tags, joint_det, joint_scores, edge_index, pred, img_info, img_id, cc_method, scaling_type,
                adjustment, th, preds_classes, det_scores, with_refine):
    # if (det_scores > 0.1).sum() == 0.0:
    #     return None
    true_positive_idx = joint_scores > th
    edge_index, pred = subgraph(true_positive_idx, edge_index, pred)
    if edge_index.shape[1] != 0:
        persons_pred, _, _ = pred_to_person(joint_det, joint_scores, edge_index, pred, preds_classes, cc_method)
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
