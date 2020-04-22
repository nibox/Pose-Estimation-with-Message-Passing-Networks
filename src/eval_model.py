import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse

from CocoKeypoints import CocoKeypoints
from Utils.Utils import load_model, get_transform, kpt_affine
import Models.PoseEstimation.PoseEstimation as pose
from Utils.ConstructGraph import graph_cluster_to_persons
from Utils.dataset_utils import Graph
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph


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


def create_train_validation_split(data_root, batch_size, mini=False):
    # todo connect the preprosseing with the model selection (input size etc)
    # todo add validation
    if mini:
        if not os.path.exists("mini_train_valid_split.p"):
            data_set = CocoKeypoints(data_root, mini=True, seed=0, mode="train")
            train, valid = torch.utils.data.random_split(data_set, [3500, 500])
            train_valid_split = [train.dataset.img_ids[train.indices], valid.dataset.img_ids[valid.indices]]
            pickle.dump(train_valid_split, open("mini_train_valid_split.p", "wb"))
        else:
            train_ids, valid_ids = pickle.load(open("train_valid_split.p", "rb"))
            train = CocoKeypoints(data_root, mini=True, seed=0, mode="train", img_ids=train_ids)
            valid = CocoKeypoints(data_root, mini=True, seed=0, mode="train", img_ids=valid_ids)


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


def coco_eval(coco, dt, image_ids, tmp_dir="tmp"):
    """
    from https://github.com/princeton-vl/pose-ae-train
    Evaluate the result with COCO API
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # why ???
    # for _, i in enumerate(sum(dt, [])):
    #    i['id'] = _+1

    # todo set image id

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
    device = torch.device("cuda") if torch.cuda.is_available() and False else torch.device("cpu")
    mini = True
    modus = "train" if mini else "valid"  # decides which validation set to use. "valid" means the coco2014 validation

    dataset_path = "../../storage/user/kistern/coco"
    model_path = "../log/PoseEstimationBaseline/2/pose_estimation.pth"
    # set is used, "train" means validation set corresponding to the mini train set is used )
    if modus == "train":
        train_ids, valid_ids = pickle.load(open("mini_train_valid_split.p", "rb"))
        eval_set = CocoKeypoints(dataset_path, mini=True, seed=0, mode="train", img_ids=train_ids)
        # valid = CocoKeypoints(data_root, mini=True, seed=0, mode="train", img_ids=valid_ids)
    else:
        raise NotImplementedError

    model = load_model(model_path, pose.PoseEstimationBaseline, pose.default_config, device).to(device)
    model.train()  # todo remove

    anns = []
    with torch.no_grad():
        for i in range(10):
            print(f"ITERATION {i}")
            imgs, masks, keypoints = eval_set[i]
            imgs = torch.from_numpy(imgs).to(device).unsqueeze(0)
            # masks = masks.to(device)
            # todo use mask to mask of joint predicitions in crowd (is this allowed?)
            # todo remove cheating
            keypoints = torch.from_numpy(keypoints).to(device).unsqueeze(0)
            pred, joint_det, edge_index, edge_labels = model(imgs, keypoints, with_logits=False)

            print(f"num_edges: {pred.shape[0]}")
            print(f"{i} num_detect {joint_det.shape[0]}")
            #pred = torch.where(pred < 0.5, torch.zeros_like(pred), torch.ones_like(pred))
            test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=pred)
            sol = cluster_graph(test_graph, "MUT", complete=False)
            sparse_sol, _ = dense_to_sparse(torch.from_numpy(sol))
            persons_pred = graph_cluster_to_persons(joint_det, sparse_sol)  # might crash

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
            gt_height = img_info["height"]
            gt_width = img_info["width"]
            scale = max(gt_height, gt_width) / 200
            mat = get_transform((gt_width / 2, gt_height / 2), scale, (128, 128))
            inv_mat = np.linalg.inv(mat)[:2]  # might this lead to numerical errors?
            tmp = np.ascontiguousarray(np.copy(persons_pred[:, :, :2]))
            persons_pred[:, :, :2] = kpt_affine(tmp, inv_mat)

            ann = gen_ann_format(persons_pred, eval_set.img_ids[i])
            anns.append(ann)
    # for i in range(10):
    #    coco_eval(eval_set.coco, [anns[i]], eval_set.img_ids[i].astype(np.int))
        coco_eval(eval_set.coco, anns, eval_set.img_ids[:10].astype(np.int))


if __name__ == "__main__":
    main()