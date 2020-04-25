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
import matplotlib;matplotlib.use("Agg")
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


def create_train_validation_split(data_root, batch_size, mini=False):
    # todo connect the preprosseing with the model selection (input size etc)
    # todo add validation
    if mini:
        tv_split_fname = "tmp/mini_train_valid_split_4.p"
        if not os.path.exists(tv_split_fname):
            data_set = CocoKeypoints(data_root, mini=True, seed=0, mode="train")
            train, valid = torch.utils.data.random_split(data_set, [3500, 500])
            train_valid_split = [train.dataset.img_ids[train.indices], valid.dataset.img_ids[valid.indices]]
            pickle.dump(train_valid_split, open(tv_split_fname, "wb"))
        else:
            train_ids, valid_ids = pickle.load(open(tv_split_fname, "rb"))
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


def draw_detection(img: torch.tensor, joint_det: torch.tensor, joint_gt: torch.tensor, fname=None):
    """
    :param img: torcg.tensor. image
    :param joint_det: shape: (num_joints, 2) list of xy positions of detected joints (without classes or clustering)
    :param fname: optional - file name of image to save. If None the image is show with plt.show
    :return:
    """

    fig = plt.figure()
    img = img.detach().cpu().numpy().copy()
    joint_det = joint_det.detach().cpu().numpy()
    joint_gt = joint_gt.detach().cpu().numpy()

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

    plt.imshow(img)
    if fname is not None:
        plt.savefig(fig=fig, fname=fname)
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
    assert len(persons.shape) == 3
    pair_ref = [
        [1, 2], [2, 3], [1, 3],
        [6, 8], [8, 10], [12, 14], [14, 16],
        [7, 9], [9, 11], [13, 15], [15, 17],
        [6, 7], [12, 13], [6, 12], [7, 13]
    ]
    bones = np.array(pair_ref) - 1
    colors = np.arange(0, 179, np.ceil(179/len(persons)))
    # image to 8bit hsv (i dont know what hsv values opencv expects in 32bit case=)
    img = img * 255.0
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    assert len(colors) == len(persons)
    for person in range(len(persons)):
        scale = 512.0 / 128.0
        color = (colors[person], 255., 255)
        valid_joints = persons[person, :, 2] > 0
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
    model_path = "../log/PoseEstimationBaseline/5_simple/pose_estimation.pth"
    config = pose.default_config
    config["message_passing"] = VanillaMPN2
    config["message_passing_config"] = default_config
    config["cheat"] = True
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
    best_pred = (None, 0, None)
    worst_pred = (None, 2, None)
    # also sample 10/x rnd images and draw them
    chosen_iter = np.random.randint(0, len(eval_set), 10)
    pred_to_draw =[]

    anns = []
    with torch.no_grad():
        for i in range(250):
            print(f"ITERATION {i}")
            imgs, masks, keypoints = eval_set[i]
            imgs = torch.from_numpy(imgs).to(device).unsqueeze(0)
            # masks = masks.to(device)
            # todo use mask to mask of joint predicitions in crowd (is this allowed?)
            # todo remove cheating
            keypoints = torch.from_numpy(keypoints).to(device).unsqueeze(0)
            pred, joint_det, edge_index, _ = model(imgs, keypoints, with_logits=False)

            print(f"num_edges: {pred.shape[0]}")
            print(f"num_detect {joint_det.shape[0]}")
            # pred = torch.where(pred < 0.5, torch.zeros_like(pred), torch.ones_like(pred))
            test_graph = Graph(x=joint_det, edge_index=edge_index, edge_attr=pred)
            sol = cluster_graph(test_graph, "MUT", complete=False)
            sparse_sol, _ = dense_to_sparse(torch.from_numpy(sol))
            persons_pred = graph_cluster_to_persons(joint_det, sparse_sol)  # might crash

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
            if len(persons_pred.shape) == 1:
                persons_pred = np.zeros([1, 17, 3])
            persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))

            ap, ap_50 = eval_single_img(eval_set.coco, persons_pred_orig, eval_set.img_ids[i])
            print(ap)
            if worst_pred[1] > ap:
                worst_pred = (i, ap, persons_pred)
            if best_pred[1] <= ap:
                best_pred = (i, ap, persons_pred)
            if i in chosen_iter:
                pred_to_draw.append((i, ap, persons_pred))

            ann = gen_ann_format(persons_pred_orig, eval_set.img_ids[i])
            anns.append(ann)
    # for i in range(10):
    #    coco_eval(eval_set.coco, [anns[i]], eval_set.img_ids[i].astype(np.int))
        coco_eval(eval_set.coco, anns, eval_set.img_ids[:250].astype(np.int))

        # draw images
        # best image
        (img, _, keypoints), persons, score = eval_set[best_pred[0]], best_pred[2], best_pred[1]
        print(f"Best prediction score: {score}")
        draw_poses(img, persons, f"output_imgs_5_simple/best_{eval_set.img_ids[best_pred[0]]}.png")
        (img, _, keypoints), persons, score = eval_set[worst_pred[0]], worst_pred[2], worst_pred[1]
        print(f"Worst prediction score: {score}")
        draw_poses(img, persons, f"output_imgs_5_simple/worst_{eval_set.img_ids[worst_pred[0]]}.png")



if __name__ == "__main__":
    main()