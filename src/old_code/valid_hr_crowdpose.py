import torch
import numpy as np
from tqdm import tqdm

from config import update_config, get_hrnet_config
from data import CrowdPoseKeypoints, HeatmapGenerator, JointsGenerator
from Utils import to_tensor
from Models.PoseEstimation import get_hr_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_minimal, transforms_to_tensor
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
        self.f.write(f"AP   easy: {stats[8]: 3f} \n")
        self.f.write(f"AP medium: {stats[9]: 3f} \n")
        self.f.write(f"AP   hard: {stats[10]: 3f} \n")

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
    from crowdposetools.cocoeval import COCOeval

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

    config = get_hrnet_config()
    config = update_config(config, f"../experiments/hrnet/w32_512_adam_lr1e-3_crowdpose.yaml")
    print(f"test: {config.LOG_DIR}")
    eval_writer = EvalWriter(config, fname="crowd_pose_w32_multi_scale_diff_factors.txt")

    parser = HeatmapParser(config)
    heatmap_generator = [HeatmapGenerator(160, 14), HeatmapGenerator(320, 14)]
    joint_generator = [JointsGenerator(30, 14, 160, True),
                       JointsGenerator(30, 14, 320, True)]
    transforms, _ = transforms_to_tensor(config)
    eval_set = CrowdPoseKeypoints(config.DATASET.ROOT, mini=False, seed=0, mode="test", transforms=transforms,
                                  heatmap_generator=heatmap_generator, filter_empty=False,
                                  joint_generator=joint_generator)

    model = get_hr_model(config, device)
    model.to(device)
    model.eval()

    anns = []

    eval_ids = []
    num_iter = len(eval_set)
    with torch.no_grad():
        for i in tqdm(range(num_iter)):
            eval_ids.append(eval_set.img_ids[i])

            img, _, masks, keypoints, factors, _ = eval_set[i]
            img = img.to(device)[None]

            heatmaps, tags = model.multi_scale_inference(img, device, config)

            grouped, scores = parser.parse(heatmaps, tags, adjust=True, refine=True)

            if len(grouped[0]) != 0:
                ann, debug_tmp = perd_to_ann(grouped[0], scores, (img.shape[2], img.shape[3]), int(eval_set.img_ids[i]), config.DATASET.INPUT_SIZE, "short_with_resize",
                                  min(config.TEST.SCALE_FACTOR))
                anns.append(ann)

        eval_writer.eval_coco(eval_set.coco, anns, np.array(eval_ids), "General Evaluation")
        eval_writer.close()


def perd_to_ann(grouped, scores, img_shape, img_id, input_size, scaling_type, min_scale):
    persons_pred_orig = reverse_affine_map(grouped.copy(), (img_shape[1], img_shape[0]), input_size, scaling_type=scaling_type,
                                           min_scale=min_scale)

    ann = gen_ann_format(persons_pred_orig, scores, img_id)
    return ann, persons_pred_orig


if __name__ == "__main__":
    main()
