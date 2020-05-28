import os
import pickle
import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse, f1_score
from tqdm import tqdm

from CocoKeypoints import CocoKeypoints
import Models.PoseEstimation.PoseEstimation as pose
from Models.MessagePassingNetwork.VanillaMPN import VanillaMPN, default_config
from Utils.Utils import load_model, draw_detection, draw_poses, pred_to_person
import matplotlib
matplotlib.use("Agg")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and False else torch.device("cpu")
    dataset_path = "../../storage/user/kistern/coco"
    ######################################
    mini = True
    sample_ids = False

    model_name = "24"
    model_path = f"../log/PoseEstimationBaseline/Real/{model_name}/pose_estimation.pth"
    config = pose.default_config
    config["message_passing"] = VanillaMPN
    config["message_passing_config"] = default_config
    config["message_passing_config"]["aggr"] = "max"
    config["message_passing_config"]["edge_input_dim"] = 2 + 17
    config["message_passing_config"]["edge_feature_dim"] = 64
    config["message_passing_config"]["node_feature_dim"] = 64
    config["message_passing_config"]["steps"] = 10

    config["cheat"] = False
    config["use_gt"] = False
    config["use_focal_loss"] = True
    config["use_neighbours"] = False
    config["mask_crowds"] = True
    config["detect_threshold"] = 0.005  # default was 0.007
    config["edge_label_method"] = 4
    config["inclusion_radius"] = 0.75
    config["matching_radius"] = 0.1
    config["mpn_graph_type"] = "knn"
    img_ids_to_use = [84015, 84381, 117772, 237976, 281133, 286645, 505421]

    ######################################
    # set is used, "train" means validation set corresponding to the mini train set is used )
    modus = "train" if mini else "valid"  # decides which validation set to use. "valid" means the coco2014 validation
    if modus == "train":
        train_ids, valid_ids = pickle.load(open("tmp/mini_train_valid_split_4.p", "rb"))
        assert len(set(train_ids).intersection(set(valid_ids))) == 0
        eval_set = CocoKeypoints(dataset_path, mini=True, seed=0, mode="train", img_ids=valid_ids)
    else:
        raise NotImplementedError
    if sample_ids:
        img_ids_to_use = np.random.choice(eval_set.img_ids, 100, replace=False)

    model = load_model(model_path, pose.PoseEstimationBaseline, pose.default_config, device).to(device)
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
        for i in tqdm(range(len(eval_set))):
            img_id = eval_set.img_ids[i]
            if img_id not in img_ids_to_use:
                continue
            imgs, masks, keypoints, factors = eval_set.get_tensor(i, device)
            num_persons_gt = np.count_nonzero(keypoints[0, :, :, 2].sum(axis=1))
            pred, joint_det, edge_index, edge_labels, _, _ = model(imgs, keypoints, masks, factors, with_logits=False)

            result = pred.squeeze()
            result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))
            f1_s = f1_score(result, edge_labels, 2)[1]
            # draw images that have low f1 score, that could not detect all persons or to many persons, or mutants
            persons_pred, mutants = pred_to_person(joint_det, edge_index, pred, "GAEC")
            num_persons_det = len(persons_pred)

            if (num_persons_gt != num_persons_det) or mutants or f1_s < 0.9:
                keypoints = keypoints[:, :num_persons_gt].squeeze().cpu().numpy()
                joint_det = joint_det.squeeze().cpu().numpy()
                if len(keypoints.shape) != 3:
                    keypoints = keypoints[np.newaxis]
                saving_cause = SavingCause(f1=f1_s, additional_p=num_persons_det > num_persons_gt,
                                           missing_p=num_persons_det < num_persons_gt, mutants=mutants)
                image_to_draw.append(
                    (eval_set.img_ids[i], imgs.squeeze(), persons_pred, joint_det, keypoints, saving_cause))

        # draw images
        # best image
        output_dir = f"tmp/output_{model_name}"
        os.makedirs(output_dir, exist_ok=True)
        for samples in image_to_draw:
            img_id, img, persons, joint_det, keypoints, saving_cause = samples
            failures = filter(lambda x: x is not None,
                              [cause if getattr(saving_cause, cause) and cause != "f1" else None for cause in
                               saving_cause.__dict__.keys()])
            failures = "|".join(failures)
            draw_poses(img, persons, f"{output_dir}/{img_id}_{int(saving_cause.f1 * 100)}_{failures}.png")
            draw_poses(img, keypoints, f"{output_dir}/{img_id}_gt.png")
            draw_detection(img, joint_det, keypoints, fname=f"{output_dir}/{img_id}_det.png")


if __name__ == "__main__":
    main()
