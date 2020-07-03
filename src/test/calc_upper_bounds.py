import pickle
import torch
import numpy as np
import torchvision
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_hr_eval, transforms_hg_eval
from config import update_config, get_config

from data import CocoKeypoints_hr, CocoKeypoints_hg, HeatmapGenerator
from Utils import graph_cluster_to_persons, adjust, to_tensor
from Utils.dataset_utils import Graph
from Models.PoseEstimation import get_upper_bound_model
from Utils.correlation_clustering.correlation_clustering_utils import cluster_graph
import matplotlib
matplotlib.use("Agg")


def sprase_to_dense(edge_index, num_nodes):
    mat = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
    mat[edge_index[0], edge_index[1]] = 1.0
    return mat


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
        score = 0.0
        for j in range(len(person)):
            tmp["keypoints"] += [float(person[j, 0]), float(person[j, 1]), float(person[j, 2])]
            score += float(person[j, 2])
        tmp["score"] = score
        ans.append(tmp)
    return ans


def coco_eval(coco, dt, image_ids, tmp_dir="tmp", log=True):
    """
    from https://github.com/princeton-vl/pose-ae-train
    Evaluate the result with COCO API
    """
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
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################
    config_name = "hrnet"
    config = get_config()
    config = update_config(config, f"../experiments/upper_bound/{config_name}.yaml")


    # set is used, "train" means validation set corresponding to the mini train set is used )
    ######################################
    if config.UB.SPLIT == "mini":
        train_ids, valid_ids = pickle.load(open("tmp/mini_train_valid_split_4.p", "rb"))
        assert len(set(train_ids).intersection(set(valid_ids))) == 0
        transforms = transforms_hg_eval(config)
        eval_set = CocoKeypoints_hg(config.DATASET.ROOT, mini=True, seed=0, mode="train",
                                    img_ids=valid_ids, transforms=transforms)
    elif config.UB.SPLIT == "mini_real":
        train_ids, valid_ids = pickle.load(open("tmp/mini_real_train_valid_split_1.p", "rb"))
        assert len(set(train_ids).intersection(set(valid_ids))) == 0
        eval_set = CocoKeypoints_hg(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids)
    elif config.UB.SPLIT == "princeton":
        _, valid_ids = pickle.load(open("tmp/princeton_split.p", "rb"))
        transforms = transforms_hg_eval(config)
        eval_set = CocoKeypoints_hg(config.DATASET.ROOT, mini=True, seed=0, mode="train",
                                    img_ids=valid_ids, transforms=transforms, mask_crowds=False)
    elif config.UB.SPLIT == "coco_17_mini":
        _, valid_ids = pickle.load(open("tmp/coco_17_mini_split.p", "rb"))  # mini_train_valid_split_4 old one
        heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
        transforms, _ = transforms_hr_eval(config)
        eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids, year=17,
                                    transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=False)
    else:
        raise NotImplementedError

    model = get_upper_bound_model(config, device)
    model.eval()

    with torch.no_grad():
        anns = []
        for i in tqdm(range(config.UB.NUM_EVAL)):

            imgs, _, masks, keypoints, _ = eval_set[i]
            num_persons_gt = np.count_nonzero(keypoints[:, :, 2].sum(axis=1))
            persons_pred = keypoints[:num_persons_gt].round()  # rounding lead to ap=0.9

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
            persons_pred = fill_person_pred(persons_pred)
            persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]),
                                                   scaling_type=config.DATASET.SCALING_TYPE)

            ann = gen_ann_format(persons_pred_orig, eval_set.img_ids[i])
            anns.append(ann)
        print("Upper bound")
        coco_eval(eval_set.coco, anns, eval_set.img_ids[:config.UB.NUM_EVAL].astype(np.int))

        anns_cc = []
        for i in tqdm(range(config.UB.NUM_EVAL)):

            img, _, masks, keypoints, factors = eval_set[i]
            img = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)
            scoremaps, _, joint_det, joint_scores, edge_index, edge_labels, _ = model(img, keypoints, masks, factors)

            test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=edge_labels)
            sol = cluster_graph(test_graph, str(config.MODEL.GC.CC_METHOD), complete=False)
            sparse_sol_cc, _ = dense_to_sparse(torch.from_numpy(sol))
            persons_pred_cc, _, _ = graph_cluster_to_persons(joint_det, joint_scores, sparse_sol_cc)  # might crash

            if config.UB.ADJUST:
                persons_pred_cc = adjust(persons_pred_cc, scoremaps[0])

            if len(persons_pred_cc.shape) == 1:  # this means none persons were detected
                persons_pred_cc = np.zeros([1, 17, 3])

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
            persons_pred_orig_cc = reverse_affine_map(persons_pred_cc.copy(), (img_info["width"], img_info["height"]),
                                                      scaling_type=config.DATASET.SCALING_TYPE)

            ann_cc = gen_ann_format(persons_pred_orig_cc, eval_set.img_ids[i])
            anns_cc.append(ann_cc)
        print("Upper bound 2")
        coco_eval(eval_set.coco, anns_cc, eval_set.img_ids[:config.UB.NUM_EVAL].astype(np.int))


if __name__ == "__main__":
    main()
