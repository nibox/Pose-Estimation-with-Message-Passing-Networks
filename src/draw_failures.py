import os
import pickle
import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse, f1_score

from CocoKeypoints import CocoKeypoints
from Utils.Utils import load_model, draw_detection, draw_poses
import Models.PoseEstimation.PoseEstimation as pose
from Models.MessagePassingNetwork.VanillaMPN2 import VanillaMPN2, default_config
from Utils.ConstructGraph import graph_cluster_to_persons
from Utils.dataset_utils import Graph
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph
import matplotlib;

matplotlib.use("Agg")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and False else torch.device("cpu")
    ######################################
    mini = True

    dataset_path = "../../storage/user/kistern/coco"
    model_name = "11"
    model_path = f"../log/PoseEstimationBaseline/{model_name}/pose_estimation.pth"
    config = pose.default_config
    config["message_passing"] = VanillaMPN2
    config["message_passing_config"] = default_config
    config["message_passing_config"]["aggr"] = "max"
    config["message_passing_config"]["edge_input_dim"] = 2 + 17
    config["cheat"] = False
    config["use_gt"] = True
    config["use_focal_loss"] = True
    config["use_neighbours"] = True
    config["mask_crowds"] = False
    config["detect_threshold"] = 0.007  # default was 0.007
    config["edge_label_method"] = 1
    img_ids_to_use = [84015, 84381, 117772, 237976, 281133, 286645, 505421]
    # set is used, "train" means validation set corresponding to the mini train set is used )
    ######################################
    modus = "train" if mini else "valid"  # decides which validation set to use. "valid" means the coco2014 validation
    if modus == "train":
        train_ids, valid_ids = pickle.load(open("tmp/mini_train_valid_split_4.p", "rb"))
        assert len(set(train_ids).intersection(set(valid_ids))) == 0
        eval_set = CocoKeypoints(dataset_path, mini=True, seed=0, mode="train", img_ids=valid_ids)
    else:
        raise NotImplementedError

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
        for i in range(250):
            if eval_set.img_ids[i] not in img_ids_to_use:
                continue
            print(f"ITERATION {i}")
            imgs, masks, keypoints = eval_set[i]
            imgs = torch.from_numpy(imgs).to(device).unsqueeze(0)
            masks = torch.from_numpy(masks).to(device).unsqueeze(0)
            num_persons_gt = np.count_nonzero(keypoints[:, :, 2].sum(axis=1))
            # todo use mask to mask of joint predicitions in crowd (is this allowed?)
            keypoints = torch.from_numpy(keypoints).to(device).unsqueeze(0)
            pred, joint_det, edge_index, edge_labels = model(imgs, keypoints, masks, with_logits=False)

            result = pred.squeeze()
            result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))
            f1_s = f1_score(result, edge_labels, 2)[1]
            # draw images that have low f1 score, that could not detect all persons or to many persons, or mutants
            test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=pred)
            sol = cluster_graph(test_graph, "GAEC", complete=False)
            sparse_sol, _ = dense_to_sparse(torch.from_numpy(sol))
            persons_pred, mutants = graph_cluster_to_persons(joint_det, sparse_sol)  # might crash
            num_persons_det = len(persons_pred)

            if (num_persons_gt != num_persons_det) or mutants or f1_s < 0.9:
                keypoints = keypoints[:, :num_persons_gt].squeeze().cpu().numpy()
                joint_det = joint_det.squeeze().cpu().numpy()
                if len(keypoints.shape) != 3:
                    keypoints = keypoints[np.newaxis]
                saving_cause = SavingCause(f1=f1_s, additional_p=num_persons_det>num_persons_gt,
                                           missing_p=num_persons_det<num_persons_gt, mutants=mutants)
                image_to_draw.append((eval_set.img_ids[i], imgs.squeeze(), persons_pred, joint_det, keypoints, saving_cause))

        # draw images
        # best image
        output_dir = f"output_{model_name}"
        os.makedirs(output_dir, exist_ok=True)
        for samples in image_to_draw:
            img_id, img, persons, joint_det, keypoints, saving_cause = samples
            failures = filter(lambda x: x is not None, [cause if getattr(saving_cause, cause) and cause!="f1" else None for cause  in  saving_cause.__dict__.keys()])
            failures = "|".join(failures)
            draw_poses(img, persons, f"{output_dir}/{img_id}_{int(saving_cause.f1 * 100)}_{failures}.png")
            draw_poses(img, keypoints, f"{output_dir}/{img_id}_gt.png")
            draw_detection(img, joint_det, keypoints, fname=f"{output_dir}/{img_id}_det.png")



if __name__ == "__main__":
    main()
