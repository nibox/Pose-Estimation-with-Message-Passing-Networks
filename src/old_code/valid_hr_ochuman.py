import torch
import numpy as np
from tqdm import tqdm

from config import update_config, get_hrnet_config
from data import OCHumans
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

def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    config = get_hrnet_config()
    config = update_config(config, f"../experiments/hrnet/w32_512_adam_lr1e-3.yaml")
    eval_writer = EvalWriter(config, fname="hourglass_multiscale_flip_ochuman.txt")

    parser = HeatmapParser(config)
    transforms, _ = transforms_to_tensor(config)
    eval_set = OCHumans('../../storage/user/kistern/OCHuman', seed=0, mode="val",
                                transforms=transforms, mask_crowds=False)

    model = get_hr_model(config, device)
    model.to(device)
    model.eval()

    # baseline : predicting full connections
    # baseline: upper bound

    # eval model

    anns = []
    eval_ids = []
    num_iter = len(eval_set)
    with torch.no_grad():

        for i in tqdm(range(num_iter)):
            eval_ids.append(eval_set.img_ids[i])

            img,masks = eval_set[i]
            img = img.to(device)[None]

            heatmaps, tags = model.multi_scale_inference(img, config, )

            grouped, scores = parser.parse(heatmaps, tags, adjust=True, refine=True)

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]

            if len(grouped[0]) != 0:
                ann = perd_to_ann(grouped[0], scores, img_info, int(eval_set.img_ids[i]), "short_with_resize",
                                  min(config.TEST.SCALE_FACTOR))
                anns.append(ann)


        eval_writer.eval_coco(eval_set.coco, anns, np.array(eval_ids), "General Evaluation")
        eval_writer.close()


def perd_to_ann(grouped, scores, img_info, img_id, scaling_type, min_scale):
    persons_pred_orig = reverse_affine_map(grouped.copy(), (img_info["width"], img_info["height"]), scaling_type=scaling_type,
                                           min_scale=min_scale)

    ann = gen_ann_format(persons_pred_orig, scores, img_id)
    return ann


if __name__ == "__main__":
    main()
