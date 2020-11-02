import pickle, time
import torch
import sys
import numpy as np
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean
from tqdm import tqdm
from Utils.hr_utils import match_by_tag

from Utils.Utils import refine, refine_tag
from Utils.eval import EvalWriter, gen_ann_format
from config import get_config, update_config
from data import CrowdPoseKeypoints, HeatmapGenerator, JointsGenerator
from Utils import pred_to_person_debug, pred_to_person, num_non_detected_points, adjust, to_tensor, calc_metrics, subgraph_mask, one_hot_encode, topk_accuracy
from Models.PoseEstimation import get_pose_model  #, get_pose_with_ref_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_to_tensor


def mpn_match_by_tag(joint_det, tag_k, scores, params):
    """

    :param joint_det: (N, 3)
    :param tags: (N, D)
    :param scores: (N)
    :param params:
    :return:
    """
    from munkres import Munkres
    def py_max_match(scores):
        m = Munkres()
        tmp = m.compute(scores)
        tmp = np.array(tmp).astype(np.int32)
        return tmp

    tag_k, loc_k, val_k = tag_k, joint_det[:, :2], scores
    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[1]))

    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]
        select = joint_det[:, 2] == idx

        tags = tag_k[select]
        joints = np.concatenate(
            (loc_k[select], val_k[select, None], tags), 1
        )
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]
        joints = joints[mask]

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            if params.ignore_too_much \
                    and len(grouped_keys) == params.max_num_people:
                continue

            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (
                        diff_normed,
                        np.zeros((num_added, num_added-num_grouped))+1e10
                    ),
                    axis=1
                )

            pairs = py_max_match(diff_normed)
            for row, col in pairs:
                if (
                        row < num_added
                        and col < num_grouped
                        and diff_saved[row][col] < params.tag_threshold
                ):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
    return ans

def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    config_dir = "hybrid_class_agnostic_end2end_crowd_pose"
    config_name = "model_86_1"
    config = get_config()
    config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")
    eval_writer = EvalWriter(config, fname="eval_single_scale_flip_wo_refine.txt")


    heatmap_generator = [HeatmapGenerator(128, 14), HeatmapGenerator(256, 14)]
    joint_generator = [JointsGenerator(30, 14, 128, True),
                       JointsGenerator(30, 14, 256, True)]
    transforms, _ = transforms_to_tensor(config)
    scaling_type = "short_with_resize" if config.TEST.PROJECT2IMAGE else "short"
    eval_set = CrowdPoseKeypoints(config.DATASET.ROOT, mini=False, seed=0, mode="test",
                                  transforms=transforms, heatmap_generator=heatmap_generator,
                                  filter_empty=False, joint_generator=joint_generator)

    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()
    model.test(True)
    # model.with_flip_kernel = False

    # baseline : predicting full connections
    # baseline: upper bound

    anns = []
    avg_time = []
    eval_ids = []
    num_iter = len(eval_set)
    with torch.no_grad():
        for i in tqdm(range(num_iter)):
            eval_ids.append(eval_set.img_ids[i])

            img, _, masks, keypoints, factors, _ = eval_set[i]
            img = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)

            if keypoints.sum() == 0.0:
                keypoints = None
                factors = None

            scoremaps, output = model.multi_scale_inference(img, config.TEST.SCALE_FACTOR, config, keypoints, factors)
            preds_nodes, preds_tags, preds_classes = output["preds"]["node"], output["preds"]["tag"], output["preds"]["class"]
            node_labels, class_labels = output["labels"]["node"],  output["labels"]["class"]
            joint_det = output["graph"]["nodes"]
            tags = output["graph"]["tags"]
            joint_scores = output["graph"]["detector_scores"]

            preds_tags = preds_tags[-1]

            t1 = time.time()
            ann = perd_to_ann(scoremaps[0], preds_tags, joint_det, joint_scores, (img.shape[2], img.shape[3]), int(eval_set.img_ids[i]), scaling_type, preds_classes,
                              tags[0])
            t2 = time.time()
            avg_time.append(t2-t1)

            if ann is not None:
                anns.append(ann)

        print("##################")
        print(f"Average time: {np.mean(avg_time)}")
        eval_writer.eval_coco(eval_set.coco, anns, np.array(eval_ids), "General Evaluation", "kpt_det.json")

        eval_writer.close()


def perd_to_ann(scoremaps, tags, joint_det, joint_scores, img_shape, img_id,
                scaling_type, preds_classes, tagmap):
    class Params(object):
        def __init__(self):
            self.num_joints = 14
            self.max_num_people = 30

            self.detection_threshold = 0.1
            self.tag_threshold = 1.0
            self.use_detection_val = True
            self.ignore_too_much = False

            self.joint_order = [
                i - 1 for i in [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]

    joint_det = joint_det.cpu().numpy()
    joint_scores = joint_scores.cpu().numpy()
    tags = tags.cpu().numpy()[:, None]
    ans = mpn_match_by_tag(joint_det, tags, joint_scores, Params())

    if False:
        tagmap= tagmap.cpu().numpy()
        scoremaps = scoremaps.cpu().numpy()
        ans = refine(scoremaps, tagmap, ans)
    if True:
        ans = adjust(ans, scoremaps)
    if len(ans) == 0:
        return None
    persons_pred_orig = reverse_affine_map(ans.copy(), (img_shape[1], img_shape[0]), 512, scaling_type=scaling_type,
                                           min_scale=1)

    ann = gen_ann_format(persons_pred_orig, img_id)

    return ann


if __name__ == "__main__":
    main()
