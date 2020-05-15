import os
import cv2
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse, precision, recall

from CocoKeypoints import CocoKeypoints
from Utils.Utils import load_model, get_transform, kpt_affine, to_numpy, graph_cluster_to_persons
from Models.PoseEstimation.UpperBound import UpperBoundModel, default_config
from Utils.dataset_utils import Graph
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph
import matplotlib
matplotlib.use("Agg")


def sprase_to_dense(edge_index, num_nodes):
    mat = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
    mat[edge_index[0], edge_index[1]] = 1.0
    return mat

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


def coco_eval(coco, dt, image_ids, tmp_dir="../tmp", log=True):
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

def load_model(config, device, pretrained_path=None):

    def rename_key(key):
        # assume structure is model.module.REAL_NAME
        return ".".join(key.split(".")[2:])

    model = UpperBoundModel(config)

    state_dict = torch.load(pretrained_path, map_location=device)
    state_dict_new = {rename_key(k): v for k, v in state_dict["state_dict"].items()}
    model.backbone.load_state_dict(state_dict_new)
    model.to(device)

    return model


def fill_person_pred(persons):
    """

    :param persons: (Num persons, 17, 3)
    :return: same array but the zero gaps are filled with the average of visible person keypoints
    """
    for i in range(len(persons)):
        vis_joint_idx = persons[i, :, 2] >= 1
        vis_joints = persons[i, vis_joint_idx]
        fill = np.mean(vis_joints[:, :2], axis=0)
        persons[i, persons[i, :, 2] == 0, :2] = fill
    return persons


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and False else torch.device("cpu")
    ######################################
    mini = True
    eval_num = 100  # setting this to 100 results in worse
    cc_method = "GAEC"

    dataset_path = "../../../storage/user/kistern/coco"
    pretrained_path = "../../PretrainedModels/pretrained/checkpoint.pth.tar"

    config = default_config
    config["cheat"] = False
    config["use_gt"] = True
    config["use_focal_loss"] = True
    config["use_neighbours"] = False
    config["mask_crowds"] = False
    config["detect_threshold"] = 0.007  # default was 0.007
    config["edge_label_method"] = 2
    config["inclusion_radius"] = 0.0
    # set is used, "train" means validation set corresponding to the mini train set is used )
    ######################################
    modus = "train" if mini else "valid"  # decides which validation set to use. "valid" means the coco2014 validation
    if modus == "train":
        train_ids, valid_ids = pickle.load(open("../tmp/mini_train_valid_split_4.p", "rb"))
        assert len(set(train_ids).intersection(set(valid_ids))) == 0
        eval_set = CocoKeypoints(dataset_path, mini=True, seed=0, mode="train", img_ids=valid_ids)
    else:
        raise NotImplementedError

    model = load_model(config, device, pretrained_path)
    model.eval()

    anns = []
    for i in range(eval_num):

        imgs, masks, keypoints = eval_set[i]
        num_persons_gt = np.count_nonzero(keypoints[:, :, 2].sum(axis=1))
        persons_pred = keypoints[:num_persons_gt].round()  # rounding lead to ap=0.9

        img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
        persons_pred = fill_person_pred(persons_pred)
        persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))

        ann = gen_ann_format(persons_pred_orig, eval_set.img_ids[i])
        anns.append(ann)
    print("Upper bound")
    coco_eval(eval_set.coco, anns, eval_set.img_ids[:eval_num].astype(np.int))

    print("Upper bound 2")
    anns_cc = []
    anns_gt = []
    for i in range(eval_num):

        imgs, masks, keypoints = eval_set[i]
        imgs = torch.from_numpy(imgs).to(device).unsqueeze(0)
        masks = torch.from_numpy(masks).to(device).unsqueeze(0)
        keypoints = torch.from_numpy(keypoints).to(device).unsqueeze(0)
        _, joint_det, edge_index, edge_labels, _ = model(imgs, keypoints, masks)

        test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=edge_labels)
        sol = cluster_graph(test_graph, cc_method, complete=False)
        sparse_sol_cc, _ = dense_to_sparse(torch.from_numpy(sol))
        sparse_sol_gt = torch.stack([edge_index[0, edge_labels==1], edge_index[1, edge_labels==1]])
        # construct solution by using only labeled edges (instead of corr clustering)
        # sparse_sol = torch.stack([edge_index[0, edge_labels==1], edge_index[1, edge_labels==1]])
        persons_pred_cc, _ = graph_cluster_to_persons(joint_det, sparse_sol_cc)  # might crash
        persons_pred_gt, _ = graph_cluster_to_persons(joint_det, sparse_sol_gt)  # might crash

        if len(persons_pred_gt.shape) == 1:  # this means none persons were detected
            persons_pred_gt = np.zeros([1, 17, 3])
        if len(persons_pred_cc.shape) == 1:  # this means none persons were detected
            persons_pred_cc = np.zeros([1, 17, 3])
        img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
        persons_pred_orig_cc = reverse_affine_map(persons_pred_cc.copy(), (img_info["width"], img_info["height"]))
        persons_pred_orig_gt = reverse_affine_map(persons_pred_gt.copy(), (img_info["width"], img_info["height"]))

        ann_cc = gen_ann_format(persons_pred_orig_cc, eval_set.img_ids[i])
        ann_gt = gen_ann_format(persons_pred_orig_gt, eval_set.img_ids[i])
        anns_cc.append(ann_cc)
        anns_gt.append(ann_gt)
    coco_eval(eval_set.coco, anns_gt, eval_set.img_ids[:eval_num].astype(np.int))
    coco_eval(eval_set.coco, anns_cc, eval_set.img_ids[:eval_num].astype(np.int))


if __name__ == "__main__":
    main()
