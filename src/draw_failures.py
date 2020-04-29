import os
import cv2
import pickle
import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse, f1_score

from CocoKeypoints import CocoKeypoints
from Utils.Utils import load_model, to_numpy
import Models.PoseEstimation.PoseEstimation as pose
from Models.MessagePassingNetwork.VanillaMPN2 import VanillaMPN2, default_config
from Utils.ConstructGraph import graph_cluster_to_persons
from Utils.dataset_utils import Graph
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph
import matplotlib;

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_detection(img, joint_det, joint_gt, fname=None):
    """
    :param img: torcg.tensor. image
    :param joint_det: shape: (num_joints, 2) list of xy positions of detected joints (without classes or clustering)
    :param fname: optional - file name of image to save. If None the image is show with plt.show
    :return:
    """

    img = to_numpy(img)

    for i in range(len(joint_det)):
        # scale up to 512 x 512
        scale = 512.0 / 128.0
        x, y = int(joint_det[i, 0] * scale), int(joint_det[i, 1] * scale)
        type = joint_det[i, 2]
        if type != -1:
            cv2.circle(img, (x, y), 3, (0., 1., 0.), -1)
    for person in range(len(joint_gt)):
        if np.sum(joint_gt[person]) > 0.0:
            for i in range(len(joint_gt[person])):
                # scale up to 512 x 512
                scale = 512.0 / 128.0
                x, y = int(joint_gt[person, i, 0] * scale), int(joint_gt[person, i, 1] * scale)
                type = i
                if type != -1:
                    cv2.circle(img, (x, y), 3, (0., 0., 1.), -1)

    fig = plt.figure()
    plt.imshow(img)
    if fname is not None:
        plt.savefig(fig=fig, fname=fname)
        plt.close(fig)
    else:
        raise NotImplementedError


def draw_poses(img: [torch.tensor, np.array], persons, fname=None):
    """

    :param img:
    :param persons: (N,17,3) array containing the person in the image img. Detected or ground truth does not matter.
    :param fname: If not none an image will be saved under this name , otherwise the image will be displayed
    :return:
    """
    img = to_numpy(img)
    assert img.shape[0] == 512
    assert img.shape[1] == 512

    if len(persons) == 0:
        fig = plt.figure()
        plt.imshow(img)
        plt.savefig(fig=fig, fname=fname)
        plt.close(fig)
        return
    pair_ref = [
        [1, 2], [2, 3], [1, 3],
        [6, 8], [8, 10], [12, 14], [14, 16],
        [7, 9], [9, 11], [13, 15], [15, 17],
        [6, 7], [12, 13], [6, 12], [7, 13]
    ]
    bones = np.array(pair_ref) - 1
    colors = np.arange(0, 179, np.ceil(179 / len(persons)))
    # image to 8bit hsv (i dont know what hsv values opencv expects in 32bit case=)
    img = img * 255.0
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    assert len(colors) == len(persons)
    for person in range(len(persons)):
        scale = 512.0 / 128.0
        color = (colors[person], 255., 255)
        valid_joints = persons[person, :, 2] > 0
        t = persons[person, valid_joints]
        center_joint = np.mean(persons[person, valid_joints] * scale, axis=0).astype(np.int)
        for i in range(len(persons[person])):
            # scale up to 512 x 512
            joint_1 = persons[person, i]
            joint_1_valid = joint_1[2] > 0
            x_1, y_1 = np.multiply(joint_1[:2], scale).astype(np.int)
            if joint_1_valid:
                cv2.circle(img, (x_1, y_1), 3, color, -1)
                cv2.line(img, (x_1, y_1), (center_joint[0], center_joint[1]), color)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    fig = plt.figure()
    plt.imshow(img)
    if fname is not None:
        plt.savefig(fig=fig, fname=fname)
    else:
        raise NotImplementedError


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    ######################################
    mini = True

    dataset_path = "../../storage/user/kistern/coco"
    model_name = "6"
    model_path = f"../log/PoseEstimationBaseline/{model_name}/pose_estimation_continue.pth"
    config = pose.default_config
    config["message_passing"] = VanillaMPN2
    config["message_passing_config"] = default_config
    config["cheat"] = False
    config["use_gt"] = True
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

            print(f"ITERATION {i}")
            imgs, masks, keypoints = eval_set[i]
            imgs = torch.from_numpy(imgs).to(device).unsqueeze(0)
            num_persons_gt = np.count_nonzero(keypoints[:, :, 2].sum(axis=1))
            # todo use mask to mask of joint predicitions in crowd (is this allowed?)
            keypoints = torch.from_numpy(keypoints).to(device).unsqueeze(0)
            pred, joint_det, edge_index, edge_labels = model(imgs, keypoints, with_logits=False)

            result = pred.squeeze()
            result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))
            f1_s = f1_score(result, edge_labels, 2)[1]
            # draw images that have low f1 score, that could not detect all persons or to many persons, or mutants
            test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=pred)
            sol = cluster_graph(test_graph, "MUT", complete=False)
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
