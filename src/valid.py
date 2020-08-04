import pickle
import torch
import torchvision
import numpy as np
from torch_geometric.utils import precision, recall, subgraph
from tqdm import tqdm

from config import get_config, update_config
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator
from Utils import pred_to_person, num_non_detected_points, adjust, to_tensor, calc_metrics, subgraph_mask
from Models.PoseEstimation import get_pose_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_hg_eval, transforms_hr_eval


class EvalWriter(object):

    def __init__(self, config):
        th = int(config.MODEL.MPN.NODE_THRESHOLD * 100)
        self.f = open(config.LOG_DIR + f"/eval_{th:g}.txt", "w")
    def eval_coco(self, coco, anns, ids, description):
        print(description)
        stats = coco_eval(coco, anns, ids)
        self.f.write(description + "\n")
        self.f.write(f"AP       : {stats[0]: 3f} \n")
        self.f.write(f"AP    0.5: {stats[1]: 3f} \n")
        self.f.write(f"AP   0.75: {stats[2]: 3f} \n")
        self.f.write(f"AP medium: {stats[3]: 3f} \n")
        self.f.write(f"AP  large: {stats[4]: 3f} \n")

    def eval_metrics(self, eval_dict, descirption):
        for k in eval_dict.keys():
            eval_dict[k] = np.mean(eval_dict[k])
        print(descirption)
        print(eval_dict)
        self.f.write(descirption + "\n")
        self.f.write(str(eval_dict) + "\n")
    def eval_part_metrics(self, eval_dict, description):
        part_labels = ['nose','eye_l','eye_r','ear_l','ear_r',
                       'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
                       'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']
        for i in range(17):
            for k in eval_dict[i].keys():
                eval_dict[i][k] = np.mean(eval_dict[i][k])
        print(description)
        self.f.write(description + " \n")
        for i in range(17):
            string = f"{part_labels[i]} acc: {eval_dict[i]['acc']:3f} prec: {eval_dict[i]['prec']:3f} rec: {eval_dict[i]['rec']:3f} f1: {eval_dict[i]['f1']:3f}"
            print(string)
            self.f.write(string + "\n")
    def close(self):
        self.f.close()


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

"""
def reverse_affine_map(keypoints, img_size_orig):
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
"""


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
        max_score = 0.0
        for j in range(len(person)):
            tmp["keypoints"] += [float(person[j, 0]), float(person[j, 1]), float(person[j, 2])]
            score += float(person[j, 2])
            max_score = max(float(person[j, 2]), max_score)
        tmp["score"] = score #/ 17.0
        if max_score > 0.0:
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    config_name = "model_54_0"
    config = get_config()
    config = update_config(config, f"../experiments/train/{config_name}.yaml")
    eval_writer = EvalWriter(config)

    heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
    transforms, _ = transforms_hr_eval(config)
    eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=False, seed=0, mode="val", img_ids=None, year=17,
                                transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=False,
                                filter_empty=False)

    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    # baseline : predicting full connections
    # baseline: upper bound

    # eval model

    anns = []
    anns_with_people = []
    imgs_with_people = []
    eval_ids = []
    with torch.no_grad():
        for i in tqdm(range(len(eval_set))):
            eval_ids.append(eval_set.img_ids[i])

            img, _, masks, keypoints, factors = eval_set[i]
            img = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)

            scoremaps, pred, preds_nodes, joint_det, joint_scores, edge_index, _, node_labels,  _, _, _ = model(img, None, masks, factors, with_logits=True)


            preds_nodes = preds_nodes[-1].sigmoid()
            preds_edges = pred[-1].sigmoid().squeeze() if pred[-1] is not None else None

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            ann = perd_to_ann(scoremaps[0], joint_det, joint_scores, edge_index, preds_edges, img_info, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD
                              , config.DATASET.SCALING_TYPE, config.TEST.ADJUST, config.MODEL.MPN.NODE_THRESHOLD)

            anns.append(ann)
            if keypoints.sum() != 0:
                imgs_with_people.append(int(eval_set.img_ids[i]))
                anns_with_people.append(ann)


        eval_writer.eval_coco(eval_set.coco, anns, np.array(eval_ids), "General Evaluation")
        eval_writer.eval_coco(eval_set.coco, anns_with_people, np.array(imgs_with_people), f"General Evaluation on not empty images {len(anns_with_people)}")
        eval_writer.close()


def perd_to_ann(scoremaps, joint_det, joint_scores, edge_index, pred, img_info, img_id, cc_method, scaling_type, adjustment, th):
    true_positive_idx = joint_scores > th
    edge_index, pred = subgraph(true_positive_idx, edge_index, pred)
    if edge_index.shape[1] != 0:
        persons_pred, _, _ = pred_to_person(joint_det, joint_scores, edge_index, pred, None, cc_method)
    else:
        persons_pred = np.zeros([1, 17, 3])
    # persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]))
    if len(persons_pred.shape) == 1:  # this means none persons were detected
        persons_pred = np.zeros([1, 17, 3])
    if adjustment:
        persons_pred = adjust(persons_pred, scoremaps)
    persons_pred_orig = reverse_affine_map(persons_pred.copy(), (img_info["width"], img_info["height"]), scaling_type=scaling_type)

    ann = gen_ann_format(persons_pred_orig, img_id)
    return ann


if __name__ == "__main__":
    main()
