import os
import cv2
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse

from CocoKeypoints import CocoKeypoints
from Utils.Utils import load_model, get_transform, kpt_affine, to_numpy
import Models.PoseEstimation.PoseEstimation as pose
from Models.MessagePassingNetwork.VanillaMPN2 import VanillaMPN2, default_config
from Utils.ConstructGraph import graph_cluster_to_persons
from Utils.dataset_utils import Graph
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph
import matplotlib;

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reverse_affine_map(keypoints, img_size_orig):
    """
    Reverses the transformation resulting from the input argument (using get_transform). Used to map output keypoints to
    original space in order to evaluate them.
    :param center:
    :param scale:
    :param res:
    :return:
    """
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
        for j in range(len(person)):
            tmp["keypoints"] += [float(person[j, 0]), float(person[j, 1]), int(person[j, 2])]
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
    from pycocotools.coco import COCO
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
    ######################################
    mini = True

    dataset_path = "../../storage/user/kistern/coco"
    model_path = "../log/PoseEstimationBaseline/6/pose_estimation_continue.pth"
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

    # baseline : predicting full connectiosn
    # baseline: upper bound

    anns = []
    for i in range(500):

        imgs, masks, keypoints = eval_set[i]
        num_persons_gt = np.count_nonzero(keypoints[:, :, 2].sum(axis=1))
        persons_pred = keypoints[:num_persons_gt].round()

        img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
        persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))

        ann = gen_ann_format(persons_pred_orig, eval_set.img_ids[i])
        anns.append(ann)
    print("Upper bound")
    coco_eval(eval_set.coco, anns, eval_set.img_ids.astype(np.int))

    print("Upper bound 2")
    anns = []
    for i in range(500):

        imgs, masks, keypoints = eval_set[i]
        imgs = torch.from_numpy(imgs).to(device).unsqueeze(0)
        keypoints = torch.from_numpy(keypoints).to(device).unsqueeze(0)
        _, joint_det, edge_index, edge_labels = model(imgs, keypoints, with_logits=False)

        test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=edge_labels)
        sol = cluster_graph(test_graph, "MUT", complete=False)
        sparse_sol, _ = dense_to_sparse(torch.from_numpy(sol))
        persons_pred, _ = graph_cluster_to_persons(joint_det, sparse_sol)  # might crash

        img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
        if len(persons_pred.shape) == 1:  # this means none persons were detected
            persons_pred = np.zeros([1, 17, 3])
        persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))

        ann = gen_ann_format(persons_pred_orig, eval_set.img_ids[i])
        anns.append(ann)
    coco_eval(eval_set.coco, anns, eval_set.img_ids.astype(np.int))
    # eval model
    anns = []
    with torch.no_grad():
        for i in range(500):

            imgs, masks, keypoints = eval_set[i]
            imgs = torch.from_numpy(imgs).to(device).unsqueeze(0)
            # masks = masks.to(device)
            # todo use mask to mask of joint predicitions in crowd (is this allowed?)
            # todo remove cheating
            # todo lower bound!!
            keypoints = torch.from_numpy(keypoints).to(device).unsqueeze(0)
            pred, joint_det, edge_index, _ = model(imgs, keypoints, with_logits=False)

            # pred = torch.where(pred < 0.5, torch.zeros_like(pred), torch.ones_like(pred))
            test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=pred)
            sol = cluster_graph(test_graph, "MUT", complete=False)
            sparse_sol, _ = dense_to_sparse(torch.from_numpy(sol))
            persons_pred, _ = graph_cluster_to_persons(joint_det, sparse_sol)  # might crash

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
            if len(persons_pred.shape) == 1:  # this means none persons were detected
                persons_pred = np.zeros([1, 17, 3])
            persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))

            ann = gen_ann_format(persons_pred_orig, eval_set.img_ids[i])
            anns.append(ann)
        coco_eval(eval_set.coco, anns, eval_set.img_ids.astype(np.int))


if __name__ == "__main__":
    main()
