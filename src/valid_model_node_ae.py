import time
import torch
import numpy as np
from tqdm import tqdm

from Utils.eval import EvalWriter, gen_ann_format
from config import get_config, update_config
from data import CocoKeypoints_hr, HeatmapGenerator, JointsGenerator
from Utils import to_tensor, mpn_match_by_tag
from Models.PoseEstimation import get_pose_model  #, get_pose_with_ref_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_to_tensor


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    config_dir = "hybrid_class_agnostic_end2end"
    config_name = "model_56_1_0_6_0_16_3"
    config = get_config()
    config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")
    eval_writer = EvalWriter(config, fname="eval_single_scale_flip.txt")


    heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
    joint_generator = [JointsGenerator(30, 17, 128, True),
                       JointsGenerator(30, 17, 256, True)]
    transforms, _ = transforms_to_tensor(config)
    scaling_type = "short_with_resize" if config.TEST.PROJECT2IMAGE else "short"
    eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=False, seed=0, mode="val", img_ids=None, year=17,
                                transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=False,
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

            scoremaps, output = model.multi_scale_inference(img, config, keypoints)
            preds_nodes, preds_tags, preds_classes = output["preds"]["node"], output["preds"]["tag"], output["preds"]["class"]
            node_labels, class_labels = output["labels"]["node"],  output["labels"]["class"]
            joint_det = output["graph"]["nodes"]
            tags = output["graph"]["tags"]
            joint_scores = output["graph"]["detector_scores"]

            # preds_classes = preds_classes[-1].softmax(dim=1) if preds_classes is not None else None
            # preds_nodes = preds_nodes[-1].sigmoid().squeeze()
            preds_tags = preds_tags[-1]

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
            t1 = time.time()
            ann = perd_to_ann(scoremaps[0], preds_tags, joint_det, joint_scores, img_info, int(eval_set.img_ids[i]), scaling_type,
                              tags[0])
            t2 = time.time()
            avg_time.append(t2-t1)


            if ann is not None:
                anns.append(ann)

        print("##################")
        print(f"Average time: {np.mean(avg_time)}")
        eval_writer.eval_coco(eval_set.coco, anns, np.array(eval_ids), "General Evaluation", "kpt_det.json")

        eval_writer.close()


def perd_to_ann(scoremaps, tags, joint_det, joint_scores, img_info, img_id,
                scaling_type, tagmap, refine, adjust, input_size):
    class Params(object):
        def __init__(self):
            self.num_joints = 17
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

    if refine:
        tagmap= tagmap.cpu().numpy()
        scoremaps = scoremaps.cpu().numpy()
        ans = refine(scoremaps, tagmap, ans)
    if adjust:
        ans = adjust(ans, scoremaps)
    if len(ans) == 0:
        return None
    persons_pred_orig = reverse_affine_map(ans.copy(), (img_info["width"], img_info["height"]), input_size, scaling_type=scaling_type,
                                           min_scale=1)

    ann = gen_ann_format(persons_pred_orig, img_id)

    return ann


if __name__ == "__main__":
    main()
