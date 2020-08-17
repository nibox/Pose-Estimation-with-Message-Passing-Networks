import os
import pickle
import torch
import numpy as np
import torchvision
from torch_geometric.utils import f1_score, subgraph
from tqdm import tqdm
from config import get_config, update_config
from torch.utils.tensorboard import SummaryWriter

from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator
from Utils.transforms import transforms_hr_eval
from Utils import draw_detection, draw_poses, pred_to_person, to_tensor, draw_detection_with_conf, draw_detection_with_cluster, draw_edges_conf, subgraph_mask
from Models import get_pose_model
import matplotlib
matplotlib.use("Agg")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################
    config_dir = "class_agnostic_end2end"
    # config_dir = "train"
    config_name = "model_57_1_0"
    config = get_config()
    config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")
    # img_ids_to_use = [84015, 84381, 117772, 237976, 281133, 286645, 505421]
    img_ids_to_use = [2299, 68409, 205324, 361586, 171190, 84674, 238410, 523957,
                      49759, 18491, 68409, 280710, 303713, 100723,
                      13774, 496409, 187362, 177861, 258388, 496854, 252716,
                      303713, 49060, 312421, 468332, 559348, 484415, 581206]


    ######################################
    # set is used, "train" means validation set corresponding to the mini train set is used )
    if config.TEST.SPLIT == "mini":
        output_size = 128.0
        train_ids, valid_ids = pickle.load(open("tmp/mini_train_valid_split_4.p", "rb"))
        assert len(set(train_ids).intersection(set(valid_ids))) == 0
        eval_set = CocoKeypoints_hg(config.DATASET.ROOT, mini=True, seed=0, mode="train", img_ids=valid_ids)
    elif config.TEST.SPLIT == "mini_real":
        train_ids, valid_ids = pickle.load(open("tmp/mini_real_train_valid_split_1.p", "rb"))
        assert len(set(train_ids).intersection(set(valid_ids))) == 0
        eval_set = CocoKeypoints_hg(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids)
    elif config.TEST.SPLIT == "princeton":
        _, valid_ids = pickle.load(open("tmp/princeton_split.p", "rb"))
        eval_set = CocoKeypoints_hg(config.DATASET.ROOT, mini=True, seed=0, mode="train", img_ids=valid_ids)
    elif config.TEST.SPLIT == "coco_17_mini":
        _, valid_ids = pickle.load(open("tmp/coco_17_mini_split.p", "rb"))  # mini_train_valid_split_4 old one

        heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
        transforms, transforms_inv = transforms_hr_eval(config)
        output_size = 256.0
        eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids, year=17,
                                    heatmap_generator=heatmap_generator, transforms=transforms)
    elif config.TEST.SPLIT == "coco_17_full":
        _, valid_ids = pickle.load(open("tmp/coco_17_full_split.p", "rb"))  # mini_train_valid_split_4 old one

        heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
        transforms, transforms_inv = transforms_hr_eval(config)
        output_size = 256.0
        eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=False, seed=0, mode="val", img_ids=valid_ids, year=17,
                                    heatmap_generator=heatmap_generator, transforms=transforms, filter_empty=False)
    else:
        raise NotImplementedError


    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    # search for the best/worst samples and draw the prediction/gt
    # low f1 score, missing persons, additional persons, mutants
    class SavingCause:
        def __init__(self, **kwargs):
            self.f1 = kwargs["f1"]
            self.missing_p = kwargs["missing_p"]
            self.additional_p = kwargs["additional_p"]
            self.mutants = kwargs["mutants"]

    image_to_draw = []
    with torch.no_grad():
        for i in tqdm(range(2302)):
            img_id = eval_set.img_ids[i]
            if img_id not in img_ids_to_use:
                continue
            img, _, masks, keypoints, factors = eval_set[i]
            img_transformed = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)
            num_persons_gt = np.count_nonzero(keypoints[0, :, :, 2].sum(axis=1).cpu().numpy())
            _, preds_edges, preds_nodes, preds_classes, joint_det, joint_scores, edge_index, edge_labels, node_labels, _, _, _ , _ = model(img_transformed, keypoints, masks, factors, with_logits=False)

            preds_edges = preds_edges[-1] if preds_edges[-1] is not None else None
            preds_classes = preds_classes[-1] if preds_classes is not None else None
            preds_nodes = preds_nodes[-1]
            result_edges = torch.where(preds_edges < 0.5, torch.zeros_like(preds_edges), torch.ones_like(preds_edges))
            result_nodes = torch.where(preds_nodes < 0.5, torch.zeros_like(preds_nodes), torch.ones_like(preds_nodes))

            f1_s = f1_score(result_nodes, node_labels, 2)[1]
            # draw images that have low f1 score, that could not detect all persons or to many persons, or mutants
            mask = subgraph_mask(preds_nodes > config.MODEL.MPN.NODE_THRESHOLD, edge_index)
            sub_preds_edges = result_edges * mask.float()

            persons_pred, mutants, person_labels = pred_to_person(joint_det, preds_nodes, edge_index, sub_preds_edges,
                                                                  preds_classes, config.MODEL.GC.CC_METHOD)
            # persons_pred_label, _, _ = pred_to_person(joint_det, joint_scores, edge_index, edge_labels, config.MODEL.GC.CC_METHOD)
            num_persons_det = len(persons_pred)


            img = np.array(transforms_inv(img.cpu().squeeze()))
            if (num_persons_gt != num_persons_det) or mutants or f1_s < 0.9:
                keypoints = keypoints[:, :num_persons_gt].squeeze().cpu().numpy()
                joint_det = joint_det.squeeze().cpu().numpy()
                preds_nodes = preds_nodes.squeeze().cpu().numpy()
                edge_index = edge_index.cpu().numpy()
                preds_edges = preds_edges.cpu().numpy()
                node_labels = node_labels.cpu().numpy()
                if len(keypoints.shape) != 3:
                    keypoints = keypoints[np.newaxis]
                saving_cause = SavingCause(f1=f1_s, additional_p=num_persons_det > num_persons_gt,
                                           missing_p=num_persons_det < num_persons_gt, mutants=mutants)
                image_to_draw.append(
                    (eval_set.img_ids[i], img, persons_pred, joint_det, preds_nodes, person_labels, keypoints, saving_cause, edge_index, preds_edges, node_labels ))

        # draw images
        # best image
        output_dir = f"tmp/output_{config_name}"
        os.makedirs(output_dir, exist_ok=True)
        for i, samples in enumerate(image_to_draw):
            img_id, img, persons, joint_det, joint_scores, person_labels, keypoints, saving_cause, edge_index, preds_edges, node_labels = samples
            failures = filter(lambda x: x is not None,
                              [cause if getattr(saving_cause, cause) and cause != "f1" else None for cause in
                               saving_cause.__dict__.keys()])
            failures = "|".join(failures)
            draw_poses(img.copy(), persons, f"{output_dir}/{i}_{img_id}_{int(saving_cause.f1 * 100)}_{failures}.png", output_size=output_size)
            # draw_poses(img, person_label, f"{output_dir}/{i}_{img_id}_gt_labels.png", output_size=output_size)
            draw_poses(img.copy(), keypoints, f"{output_dir}/{i}_{img_id}_gt.png", output_size=output_size)
            draw_detection_with_conf(img.copy(), joint_det, joint_scores, keypoints, fname=f"{output_dir}/{i}_{img_id}_conf_det.png", output_size=output_size)
            draw_detection(img.copy(), joint_det, keypoints, fname=f"{output_dir}/{i}_{img_id}_det.png", output_size=output_size)
            #draw_detection_with_cluster(img, joint_det, person_labels, keypoints, fname=f"{output_dir}/{i}_{img_id}_clust_det.png", output_size=output_size)
            draw_edges_conf(img.copy(), joint_det, person_labels, node_labels, edge_index, preds_edges, fname=f"{output_dir}/{i}_{img_id}_edges.png", output_size=output_size)


if __name__ == "__main__":
    main()
