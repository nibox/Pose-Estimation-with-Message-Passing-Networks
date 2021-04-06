import numpy as np
import torch
from torch_scatter import scatter_mean
from sklearn.metrics import roc_auc_score


class EvalWriter(object):

    def __init__(self, config, fname=None):
        th = int(config.MODEL.MPN.NODE_THRESHOLD * 100)
        self.dir = config.LOG_DIR
        self.dataset = config.DATASET.DATASET
        self.num_joints = config.DATASET.NUM_JOINTS
        assert self.dataset in ["coco", "crowd_pose"]
        if fname is None:
            self.f = open(config.LOG_DIR + f"/eval_{th:g}.txt", "w")
        else:
            self.f = open(config.LOG_DIR + "/" + f"{fname}", "w")
    def eval_coco(self, coco, anns, ids, description, dt_file_name="dt.json"):
        print(description)
        if self.dataset == "coco":
            stats = coco_eval(coco, anns, ids, tmp_dir=self.dir, dt_file_name=dt_file_name)
            self.f.write(description + "\n")
            self.f.write(f"AP       : {stats[0]: 3f} \n")
            self.f.write(f"AP    0.5: {stats[1]: 3f} \n")
            self.f.write(f"AP   0.75: {stats[2]: 3f} \n")
            self.f.write(f"AP medium: {stats[3]: 3f} \n")
            self.f.write(f"AP  large: {stats[4]: 3f} \n")
        else:
            stats = crowd_pose_eval(coco, anns, ids, tmp_dir=self.dir, dt_file_name=dt_file_name)
            self.f.write(description + "\n")
            self.f.write(f"AP         : {stats[0]: 3f} \n")
            self.f.write(f"AP      0.5: {stats[1]: 3f} \n")
            self.f.write(f"AP     0.75: {stats[2]: 3f} \n")
            self.f.write(f"AR         : {stats[5]: 3f} \n")
            self.f.write(f"AR      0.5: {stats[6]: 3f} \n")
            self.f.write(f"AR     0.75: {stats[7]: 3f} \n")
            self.f.write(f"AP     easy: {stats[8]: 3f} \n")
            self.f.write(f"AR   medium: {stats[9]: 3f} \n")
            self.f.write(f"AR     hard: {stats[10]: 3f} \n")


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

    def eval_speed(self, *args):
        # 0, 2, 4, 6 ... are names
        # 1, 3, 5, 7 ... are values
        assert len(args) % 2 == 0
        print(f"Runtime measurement")
        self.f.write("Runtime measurement")
        for i in range(0, len(args), 2):
            print(f"{args[i]}: {np.mean(args[i+1])}")
            self.f.write(f"{args[i]}: {np.mean(args[i+1])} \n")



    def eval_part_metrics(self, eval_dict, description):
        if self.dataset == "coco":
            part_labels = ['nose','eye_l','eye_r','ear_l','ear_r',
                           'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
                           'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']
        else:
            part_labels = [
                           'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
                           'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r', "head_1", "head_2"]
        for i in range(self.num_joints):
            for k in eval_dict[i].keys():
                eval_dict[i][k] = np.mean(eval_dict[i][k])
        print(description)
        self.f.write(description + " \n")
        for i in range(self.num_joints):
            string = f"{part_labels[i]} acc: {eval_dict[i]['acc']:3f} prec: {eval_dict[i]['prec']:3f} rec: {eval_dict[i]['rec']:3f} f1: {eval_dict[i]['f1']:3f}"
            print(string)
            self.f.write(string + "\n")

    def eval_joint_error_types(self, eval_dict, description):
        assert "groups" in eval_dict.keys() and "errors" in eval_dict.keys()

        print(description)
        self.f.write(description + " \n")
        errors = torch.from_numpy(np.array(eval_dict["errors"]))
        groups = torch.from_numpy(np.array(eval_dict["groups"]))
        errors = scatter_mean(errors, groups).cpu().numpy()
        num_free_joints = np.mean(eval_dict["num_free_joints"])
        string = f" <= 5: {errors[0]} \n <= 10: {errors[1]} \n <= 15: {errors[2]} \n > 15: {errors[3]} \n num_free_joints: {num_free_joints}"
        print(string)
        self.f.write(string + "\n")




        print(f" <= 5: {errors[0]}")
        print(f" <= 10: {errors[1]}")
        print(f" <= 15: {errors[2]}")
        print(f" < 15: {errors[3]}")

    def eval_roc_auc(self, eval_dict, description):
        print(description)
        self.f.write(description + " \n")

        part_labels = ['nose','eye_l','eye_r','ear_l','ear_r',
                       'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
                       'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']
        if eval_dict["node"] is not None or len(eval_dict["node"]) != 0:
            pred = np.array(eval_dict["node"]["pred"])
            label = np.array(eval_dict["node"]["label"]).astype(np.int)
            score = roc_auc_score(label, pred)
            string = f"node roc_auc: {score}"
            print(string)
            self.f.write(string + "\n")

            classes = np.array(eval_dict["node"]["class"])
            for i in range(self.num_joints):
                mask = classes == i
                class_score = roc_auc_score(label[mask], pred[mask])
                string = f"{part_labels[i]}  roc_auc: {class_score}"
                print(string)
                self.f.write(string + "\n")


        if eval_dict["edge"] is not None:
            raise NotImplementedError


    def close(self):
        self.f.close()


def coco_eval(coco, dt, image_ids, tmp_dir="tmp", dt_file_name="dt.json"):
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

def create_results_json(dt, dir, name):
    import json
    with open(dir + '/' + name, 'w') as f:
        json.dump(sum(dt, []), f)

def crowd_pose_eval(coco, dt, image_ids, tmp_dir="tmp", dt_file_name="dt.json"):
    """
    from https://github.com/princeton-vl/pose-ae-train
    Evaluate the result with COCO API
    """
    from crowdposetools.cocoeval import COCOeval

    import json
    with open(tmp_dir + f'/{dt_file_name}', 'w') as f:
        json.dump(sum(dt, []), f)

    # load coco
    coco_dets = coco.loadRes(tmp_dir + f'/{dt_file_name}')
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
        # todo what does the score do?
        # how are missing joints handled ?
        tmp = {'image_id': int(image_id), "category_id": 1, "keypoints": [], "score": 1.0}
        score = 0.0
        if (person[:, 2] > 0.09).sum() > 0:
            score = float(person[person[:, 2] > 0.09, 2].mean())
        else:
            score = 0.0

        for j in range(len(person)):
            tmp["keypoints"] += [float(person[j, 0]), float(person[j, 1]), float(person[j, 2])]
            score += float(person[j, 2])
        tmp["score"] = score #/ 17.0
        ans.append(tmp)
    return ans

def gen_ann_format_correct(pred, image_id=0):
    """
    from https://github.com/princeton-vl/pose-ae-train
    Generate the json-style data for the output
    """
    ans = []
    for i in range(len(pred)):
        person = pred[i]
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

def gen_ann_format_mean(pred, image_id=0):
    """
    from https://github.com/princeton-vl/pose-ae-train
    Generate the json-style data for the output
    """
    ans = []
    for i in range(len(pred)):
        person = pred[i]
        # todo what does the score do?
        # how are missing joints handled ?
        tmp = {'image_id': int(image_id), "category_id": 1, "keypoints": [], "score": 1.0}
        if (person[:, 2] > 0.09).sum() > 0:
            score = float(person[person[:, 2] > 0.09, 2].mean())
        else:
            score = 0.0

        for j in range(len(person)):
            tmp["keypoints"] += [float(person[j, 0]), float(person[j, 1]), float(person[j, 2])]
        tmp["score"] = score #/ 17.0
        ans.append(tmp)
    return ans
