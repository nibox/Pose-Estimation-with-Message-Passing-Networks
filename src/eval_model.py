import pickle
import torch
import numpy as np
from torch_geometric.utils import precision, recall
from tqdm import tqdm
from CocoKeypoints import CocoKeypoints
from Utils.Utils import load_model, get_transform, kpt_affine, pred_to_person, num_non_detected_points
import Models.PoseEstimation.PoseEstimation as pose
from Models.MessagePassingNetwork.VanillaMPN2 import VanillaMPN2, default_config


def specificity(pred, target, num_classes):
    r"""Computes the recall
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}` of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    from torch_geometric.utils import true_negative, false_positive
    tn = true_negative(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tn / (tn + fp)
    out[torch.isnan(out)] = 0

    return out

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


def eval_single_img(coco, dt, image_id, tmp_dir="tmp"):
    ann = [gen_ann_format(dt, image_id)]
    stats = coco_eval(coco, ann, [image_id], log=False)
    return stats[:2]


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

def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    dataset_path = "../../storage/user/kistern/coco"
    ######################################
    mini = True
    eval_num = 500
    cc_method = "GAEC"

    dataset_path = "../../storage/user/kistern/coco"
    model_path = "../log/PoseEstimationBaseline/9/pose_estimation.pth"
    config = pose.default_config
    config["message_passing"] = VanillaMPN2
    config["message_passing_config"] = default_config
    config["cheat"] = False
    config["use_gt"] = True
    config["use_neighbours"] = True
    config["mask_crowds"] = False
    # set is used, "train" means validation set corresponding to the mini train set is used )
    ######################################
    modus = "train" if mini else "valid"  # decides which validation set to use. "valid" means the coco2014 validation
    if modus == "train":
        train_ids, valid_ids = pickle.load(open("tmp/mini_train_valid_split_4.p", "rb"))
        assert len(set(train_ids).intersection(set(valid_ids))) == 0
        eval_set = CocoKeypoints(dataset_path, mini=True, seed=0, mode="train", img_ids=valid_ids)
    else:
        raise NotImplementedError

    model = load_model(model_path, pose.PoseEstimationBaseline, config, device).to(device)
    model.eval()

    # baseline : predicting full connections
    # baseline: upper bound

    # eval model
    eval_prec_negative = []
    eval_prec_positive = []
    eval_recall_negative = []
    eval_recall_positive = []
    eval_specificity_positive = []

    anns = []
    anns_inter_person = []
    anns_intra_person = []
    imgs_fully_det = []  # save ids with perfect detections so i can eval performance on
    print("Eval on Labeled data")
    with torch.no_grad():
        for i in tqdm(range(eval_num)):

            imgs, masks, keypoints = eval_set[i]
            imgs = torch.from_numpy(imgs).to(device).unsqueeze(0)
            masks = torch.from_numpy(masks).to(device).unsqueeze(0)
            # todo use mask to mask of joint predicitions in crowd (is this allowed?)
            # todo remove cheating
            # todo lower bound!!
            keypoints = torch.from_numpy(keypoints).to(device).unsqueeze(0)
            pred, joint_det, edge_index, edge_labels, _ = model(imgs, keypoints, masks, with_logits=True)

            pred = pred.sigmoid().squeeze()
            result = torch.where(pred < 0.5, torch.zeros_like(pred), torch.ones_like(pred))
            n, _ = num_non_detected_points(joint_det.cpu(), keypoints.cpu(), 6.0, config["use_gt"])

            eval_prec_positive.append(precision(result, edge_labels, 2)[1])
            eval_prec_negative.append(precision(result, edge_labels, 2)[0])
            eval_recall_positive.append(recall(result, edge_labels, 2)[1])
            eval_recall_negative.append(recall(result, edge_labels, 2)[0])
            eval_specificity_positive.append(specificity(result, edge_labels, 2)[1])

            pred_intra_person = pred * edge_labels  # this shows the impact of higher recall
            pred_inter_person = torch.where(edge_labels == 0.0, pred, edge_labels)  # this shows the impact of higher specificity
            # pred = torch.where(pred < 0.5, torch.zeros_like(pred), torch.ones_like(pred))
            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            ann = perd_to_ann(joint_det, edge_index, pred, img_info, int(eval_set.img_ids[i]), cc_method)
            ann_intra = perd_to_ann(joint_det, edge_index, pred_intra_person, img_info, int(eval_set.img_ids[i]), cc_method)
            ann_inter = perd_to_ann(joint_det, edge_index, pred_inter_person, img_info, int(eval_set.img_ids[i]), cc_method)

            anns.append(ann)
            anns_intra_person.append(ann_intra)
            anns_inter_person.append(ann_inter)
            if int(n) == 0:
                imgs_fully_det.append(int(eval_set.img_ids[i]))

        anns_real = []
        anns_full = []
        model.use_gt = False
        print("Eval on Real data")
        with torch.no_grad():
            for i in tqdm(range(eval_num)):
                imgs, masks, keypoints = eval_set[i]
                imgs = torch.from_numpy(imgs).to(device).unsqueeze(0)
                masks = torch.from_numpy(masks).to(device).unsqueeze(0)
                # todo use mask to mask of joint predicitions in crowd (is this allowed?)
                # todo remove cheating
                # todo lower bound!!
                keypoints = torch.from_numpy(keypoints).to(device).unsqueeze(0)
                pred, joint_det, edge_index, _, _ = model(imgs, keypoints, masks, with_logits=True)

                pred = pred.sigmoid().squeeze()

                # pred = torch.where(pred < 0.5, torch.zeros_like(pred), torch.ones_like(pred))
                img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
                ann = perd_to_ann(joint_det, edge_index, pred, img_info, int(eval_set.img_ids[i]), cc_method)
                anns_real.append(ann)
                if int(eval_set.img_ids[i]) in imgs_fully_det:
                    anns_full.append(ann)
        print("##################")
        print("General Evaluation")
        coco_eval(eval_set.coco, anns, eval_set.img_ids[:eval_num].astype(np.int))
        print("##################")
        print("Inter Person Ability")
        coco_eval(eval_set.coco, anns_inter_person, eval_set.img_ids[:eval_num].astype(np.int))
        print("##################")
        print("Intra Person Ability")
        coco_eval(eval_set.coco, anns_intra_person, eval_set.img_ids[:eval_num].astype(np.int))
        print("##################")
        print("Real Evaluation")
        coco_eval(eval_set.coco, anns_real, eval_set.img_ids[:eval_num].astype(np.int))
        print("##################")
        print("Real Evaluation on perfect images")
        coco_eval(eval_set.coco, anns_full, imgs_fully_det)
        print(f"Positive Precision: {np.mean(eval_prec_positive)}")
        print(f"Positive Recall: {np.mean(eval_recall_positive)}")
        print(f"Negative Precision: {np.mean(eval_prec_negative)}")
        print(f"Negative Recall: {np.mean(eval_recall_negative)}")
        print(f"Positive Specificity: {np.mean(eval_specificity_positive)}")


def perd_to_ann(joint_det, edge_index, pred, img_info, img_id, cc_method):
    persons_pred, _ = pred_to_person(joint_det, edge_index, pred, cc_method)

    if len(persons_pred.shape) == 1:  # this means none persons were detected
        persons_pred = np.zeros([1, 17, 3])
    persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))
    ann = gen_ann_format(persons_pred_orig, img_id)
    return ann


if __name__ == "__main__":
    main()
