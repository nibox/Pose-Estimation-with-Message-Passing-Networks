import os
import pickle
import torch
import numpy as np
import torchvision
from torch_geometric.utils import f1_score
from tqdm import tqdm
from config import get_config, update_config

from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator
from Utils.transforms import transforms_hr_eval
from Utils import draw_detection, draw_poses, pred_to_person, to_tensor
from Models import get_pose_model
import matplotlib
matplotlib.use("Agg")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################
    config_name = "model_32"
    config = get_config()
    config = update_config(config, f"../experiments/train/{config_name}.yaml")
    # img_ids_to_use = [84015, 84381, 117772, 237976, 281133, 286645, 505421]

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
        for i in tqdm(range(100)):
            img_id = eval_set.img_ids[i]
            # if img_id not in img_ids_to_use:
            #     continue
            img, _, masks, keypoints, factors = eval_set[i]
            img_transformed = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)
            num_persons_gt = np.count_nonzero(keypoints[0, :, :, 2].sum(axis=1).cpu().numpy())
            _, pred, joint_det, joint_scores, edge_index, edge_labels, _, _ , _, node_features = model(img_transformed, keypoints, masks, factors, with_logits=False)

            result = pred[-1].squeeze()
            result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))
            f1_s = f1_score(result, edge_labels, 2)[1]
            # draw images that have low f1 score, that could not detect all persons or to many persons, or mutants
            persons_pred, mutants, person_labels = pred_to_person(joint_det, joint_scores, edge_index, result, config.MODEL.GC.CC_METHOD)
            persons_pred_label, _, _ = pred_to_person(joint_det, joint_scores, edge_index, edge_labels, config.MODEL.GC.CC_METHOD)
            num_persons_det = len(persons_pred)


            img = np.array(transforms_inv(img.cpu().squeeze()))
            if (num_persons_gt != num_persons_det) or mutants or f1_s < 0.9:
                keypoints = keypoints[:, :num_persons_gt].squeeze().cpu().numpy()
                joint_det = joint_det.squeeze().cpu().numpy()
                if len(keypoints.shape) != 3:
                    keypoints = keypoints[np.newaxis]
                saving_cause = SavingCause(f1=f1_s, additional_p=num_persons_det > num_persons_gt,
                                           missing_p=num_persons_det < num_persons_gt, mutants=mutants)
                image_to_draw.append(
                    (eval_set.img_ids[i], img, persons_pred, persons_pred_label, joint_det, keypoints, saving_cause))

        # draw images
        # best image
        output_dir = f"tmp/output_{config_name}"
        os.makedirs(output_dir, exist_ok=True)
        for i, samples in enumerate(image_to_draw):
            img_id, img, persons, person_label, joint_det, keypoints, saving_cause = samples
            failures = filter(lambda x: x is not None,
                              [cause if getattr(saving_cause, cause) and cause != "f1" else None for cause in
                               saving_cause.__dict__.keys()])
            failures = "|".join(failures)
            draw_poses(img, persons, f"{output_dir}/{img_id}_{int(saving_cause.f1 * 100)}_{failures}.png", output_size=output_size)
            draw_poses(img, person_label, f"{output_dir}/{img_id}_gt_labels.png", output_size=output_size)
            draw_poses(img, keypoints, f"{output_dir}/{img_id}_gt.png", output_size=output_size)
            draw_detection(img, joint_det, keypoints, fname=f"{output_dir}/{img_id}_det.png", output_size=output_size)


if __name__ == "__main__":
    main()
