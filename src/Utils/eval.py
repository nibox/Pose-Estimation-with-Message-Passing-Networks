import numpy as np


class EvalWriter(object):

    def __init__(self, config, fname=None):
        th = int(config.MODEL.MPN.NODE_THRESHOLD * 100)
        self.dir = config.LOG_DIR
        if fname is None:
            self.f = open(config.LOG_DIR + f"/eval_{th:g}.txt", "w")
        else:
            self.f = open(config.LOG_DIR + "/" + f"{fname}", "w")
    def eval_coco(self, coco, anns, ids, description, dt_file_name="dt.json"):
        print(description)
        stats = coco_eval(coco, anns, ids, tmp_dir=self.dir, dt_file_name=dt_file_name)
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

    def eval_metric(self, eval_list, description):
        value = np.mean(eval_list)
        print(description)
        print(value)
        self.f.write(description + "\n")
        self.f.write(str(value) + "\n")

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


def coco_eval(coco, dt, image_ids, tmp_dir="tmp", dt_file_name="dt.json", log=True):
    """
    from https://github.com/princeton-vl/pose-ae-train
    Evaluate the result with COCO API
    """
    from pycocotools.cocoeval import COCOeval

    import json
    with open(tmp_dir + '/' + dt_file_name, 'w') as f:
        json.dump(sum(dt, []), f)

    # load coco
    coco_dets = coco.loadRes(tmp_dir + '/' + dt_file_name)
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


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
        tmp["score"] = score #/ 17.0
        ans.append(tmp)
    return ans