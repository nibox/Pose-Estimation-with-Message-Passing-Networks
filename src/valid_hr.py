import torch
import numpy as np
from tqdm import tqdm
import argparse

from Utils.hr_utils.group import cluster_cc
from config import update_config, get_hrnet_config, update_config_command
from data import CocoKeypoints_hr, HeatmapGenerator, JointsGenerator, CrowdPoseKeypoints, OCHumans
from Models.PoseEstimation import get_hr_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_to_tensor
from Utils.hr_utils import HeatmapParser


class EvalWriter(object):

    def __init__(self, config, fname=None):
        if fname is None:
            raise NotImplementedError
        else:
            self.f = open(config.LOG_DIR + "/" + fname, "w")
    def eval_coco(self, coco, anns, ids, description):
        print(description)
        stats = coco_eval(coco, anns, ids)
        self.f.write(description + "\n")
        self.f.write(f"AP       : {stats[0]: 3f} \n")
        self.f.write(f"AP    0.5: {stats[1]: 3f} \n")
        self.f.write(f"AP   0.75: {stats[2]: 3f} \n")
        self.f.write(f"AP medium: {stats[3]: 3f} \n")
        self.f.write(f"AP  large: {stats[4]: 3f} \n")

    def close(self):
        self.f.close()
        pass

def gen_ann_format(pred, scores, image_id=0):
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
            tmp["keypoints"] += [float(person[j, 0]), float(person[j, 1]), float(person[j, 2])]
        tmp["score"] = float(scores[i])
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
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HigherHRNet.")
    parser.add_argument("--config", help="Config file name for the experiment.", required=True, type=str)
    parser.add_argument("--out_file", help="Name of the output log file containing the results.", required=True, type=str)
    parser.add_argument("options", help="Modifications to config file through the command line. "
                                        "Can be use to specify the evaluation setting (flip test, multi-scale etc.)"
                        , default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    # config = get_hrnet_config()
    # config = update_config(config, f"../experiments/hrnet/w32_512_adam_lr1e-3.yaml")
    # eval_writer = EvalWriter(config, fname="w32_512_adam_lr1e-3.yaml")
    args = parse_args()
    config = get_hrnet_config()
    config = update_config(config, f"../experiments/{args.config}")
    config = update_config_command(config, args.options)
    eval_writer = EvalWriter(config, fname=args.out_file)

    scaling_type = "short_with_resize" if config.TEST.PROJECT2IMAGE else "short"
    output_sizes = config.DATASET.OUTPUT_SIZE
    max_num_people = config.DATASET.MAX_NUM_PEOPLE
    num_joints = config.DATASET.NUM_JOINTS
    transforms, _ = transforms_to_tensor(config)

    parser = HeatmapParser(config)
    heatmap_generator = [HeatmapGenerator(output_sizes[0], num_joints),
                         HeatmapGenerator(output_sizes[1], num_joints)]
    joint_generator = [JointsGenerator(max_num_people, num_joints, output_sizes[0], True),
                       JointsGenerator(max_num_people, num_joints, output_sizes[1], True)]
    if config.TEST.SPLIT == "coco_17_full":
        assert config.DATASET.NUM_JOINTS == 17
        eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=False, seed=0, mode="val", img_ids=None, year=17,
                                    transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=False,
                                    filter_empty=False, joint_generator=joint_generator)
    elif config.TEST.SPLIT == "test-dev2017":
        raise NotImplementedError
    elif config.TEST.SPLIT == "crowd_pose_test":
        assert config.DATASET.NUM_JOINTS == 14
        eval_set = CrowdPoseKeypoints(config.DATASET.ROOT, mini=False, seed=0, mode="test",
                                      transforms=transforms, heatmap_generator=heatmap_generator,
                                      filter_empty=False, joint_generator=joint_generator)
    elif config.TEST.SPLIT == "ochuman_valid":
        assert config.DATASET.NUM_JOINTS == 17
        eval_set = OCHumans('../../storage/user/kistern/OCHuman', seed=0, mode="val",
                            transforms=transforms, mask_crowds=False)
    elif config.TEST.SPLIT == "ochuman_test":
        assert config.DATASET.NUM_JOINTS == 17
        eval_set = OCHumans('../../storage/user/kistern/OCHuman', seed=0, mode="test",
                            transforms=transforms, mask_crowds=False)
    else:
        raise NotImplementedError

    model = get_hr_model(config, device)
    model.to(device)
    model.eval()

    anns_ae = []
    anns_cc = []

    eval_ids = []
    num_iter = len(eval_set)
    with torch.no_grad():
        for i in tqdm(range(num_iter)):
            eval_ids.append(eval_set.img_ids[i])

            img = eval_set[i][0]
            img = img.to(device)[None]

            heatmaps, tags = model.multi_scale_inference(img, device, config)

            grouped_heu, scores_heu = parser.parse(heatmaps, tags, adjust=config.TEST.ADJUST, refine=config.TEST.REFINE)
            grouped_cc, scores_cc = cluster_cc(heatmaps[0], tags[0], config)


            img_shape = (img.shape[3], img.shape[2])
            if len(grouped_heu[0]) != 0:
                ann = perd_to_ann(grouped_heu[0], scores_heu, img_shape, int(eval_set.img_ids[i]), config.DATASET.INPUT_SIZE, scaling_type,
                                  min(config.TEST.SCALE_FACTOR))
                anns_ae.append(ann)
            if len(grouped_cc) != 0:
                ann = perd_to_ann(grouped_cc, scores_cc, img_shape, int(eval_set.img_ids[i]), config.DATASET.INPUT_SIZE, scaling_type,
                                  min(config.TEST.SCALE_FACTOR))
                anns_cc.append(ann)


        eval_writer.eval_coco(eval_set.coco, anns_ae, np.array(eval_ids), "General Evaluation with heuristic grouping")
        eval_writer.eval_coco(eval_set.coco, anns_cc, np.array(eval_ids), "General Evaluation with correlation clustering")
        eval_writer.close()


def perd_to_ann(grouped, scores, img_shape, img_id, input_size, scaling_type, min_scale):
    persons_pred_orig = reverse_affine_map(grouped.copy(), img_shape, input_size, scaling_type, min_scale)

    ann = gen_ann_format(persons_pred_orig, scores, img_id)
    return ann


if __name__ == "__main__":
    main()
