import argparse
import time
from tqdm import tqdm

from config import get_config, update_config, update_config_command
from data import CocoKeypoints_hr, HeatmapGenerator, JointsGenerator, CrowdPoseKeypoints, CocoKeypoints_test, OCHumans
from Utils import *
from Models.PoseEstimation import get_pose_model
from Utils.transforms import transforms_to_tensor
from Utils.eval import EvalWriter, create_results_json




def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate message passing network for keypoint detection")
    parser.add_argument("--config", help="Config file name for the experiment.", required=True, type=str)
    parser.add_argument("--out_file", help="Name of the output log file containing the results.", required=True, type=str)
    parser.add_argument("options", help="Modifications to config file through the command line. "
                                        "Can be use to specify the evaluation setting (flip test, multi-scale etc.)"
                        , default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def get_scaling_type(config):
    if config.DATASET.SCALING_TYPE == "short":
        if len(config.TEST.SCALE_FACTOR) > 1:
            assert config.TEST.PROJECT2IMAGE

        return "short_with_resize" if config.TEST.PROJECT2IMAGE else "short"
    elif config.DATASET.SCALING_TYPE == "long":
        assert not config.TEST.PROJECT2IMAGE
        return "long_with_multiscale" if len(config.TEST.SCALE_FACTOR) > 1 else "long"

def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    args = parse_args()
    config = get_config()
    config = update_config(config, f"../experiments/{args.config}")
    config = update_config_command(config, args.options)
    eval_writer = EvalWriter(config, fname=args.out_file)

    # load datasets
    output_sizes = config.DATASET.OUTPUT_SIZE
    max_num_people = config.DATASET.MAX_NUM_PEOPLE
    num_joints = config.DATASET.NUM_JOINTS
    transforms, _ = transforms_to_tensor(config)
    scaling_type = get_scaling_type(config)

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
        assert config.DATASET.NUM_JOINTS == 17
        eval_set = CocoKeypoints_test(config.DATASET.ROOT, seed=0, year=17)
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

    model = get_pose_model(config, device)
    state_dict = torch.load(config.MODEL.PRETRAINED, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    anns = []
    duration_pose_constr = []
    duration_kpt = []
    duration_mpn = []
    eval_ids = []
    num_iter = len(eval_set)
    with torch.no_grad():
        for i in tqdm(range(num_iter)):
            eval_ids.append(eval_set.img_ids[i])

            img = eval_set[i][0]
            img = img.to(device)[None]

            scoremaps, output = model.multi_scale_inference(img, device, config)
            preds_nodes, preds_edges, preds_classes = output["preds"]["node"], output["preds"]["edge"], output["preds"]["class"]
            joint_det, edge_index = output["graph"]["nodes"], output["graph"]["edge_index"]
            joint_scores = output["graph"]["detector_scores"]
            tags = output["graph"]["tags"]
            # mpn + tag stuff
            preds_tags = output["preds"]["tag"][-1]

            preds_nodes = preds_nodes[-1].sigmoid() if preds_nodes[-1] is not None else joint_scores
            preds_edges = preds_edges[-1].sigmoid().squeeze() if preds_edges[-1] is not None else None
            preds_classes = preds_classes[-1].softmax(dim=1) if preds_classes is not None else None


            img_shape = (img.shape[3], img.shape[2])
            if preds_tags is None:
                start_constr = time.clock()
                ann = pred_to_ann(scoremaps[0], tags[0], joint_det, preds_nodes, edge_index, preds_edges, img_shape,
                                  config.DATASET.INPUT_SIZE, int(eval_set.img_ids[i]), config.MODEL.GC.CC_METHOD,
                                  scaling_type, min(config.TEST.SCALE_FACTOR), config.TEST.ADJUST,
                                  config.MODEL.MPN.NODE_THRESHOLD, preds_classes, config.TEST.WITH_REFINE, joint_scores,
                                  False, scoring_method=config.TEST.SCORING, fill_mean=config.TEST.FILL_MEAN,
                                  num_joints=num_joints)
                end_constr = time.clock()
            else:
                end_constr, start_constr = 0, 0
                ann = perd_to_ann_ae(scoremaps[0], preds_tags, joint_det, joint_scores, img_shape, int(eval_set.img_ids[i]),
                                  scaling_type,
                                  tags[0], config.TEST.WITH_REFINE, config.TEST.ADJUST, config.DATASET.INPUT_SIZE,
                                     min(config.TEST.SCALE_FACTOR))

            if ann is not None:
                anns.append(ann)
            duration_kpt.append(output["debug"]["kpt"])
            duration_mpn.append(output["debug"]["mpn"])
            duration_pose_constr.append(end_constr - start_constr)


        if config.TEST.SPLIT == "test-dev2017":
            create_results_json(anns, config.LOG_DIR, "person_keypoints_test-dev2017_mpn_results.json")
        else:
            eval_writer.eval_coco(eval_set.coco, anns, np.array(eval_ids), "General Evaluation", f"person_keypoints_{config.TEST.SPLIT}_mpn_results.json")
            eval_writer.eval_speed("kpt_detector", duration_kpt, "mpn", duration_mpn, "pose_constr", duration_pose_constr)
            eval_writer.close()


def perd_to_ann_ae(scoremaps, tags, joint_det, joint_scores, img_shape, img_id,
                scaling_type, tagmap, with_refine, with_adjust, input_size, min_scale):
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

    if with_refine:
        tagmap= tagmap.cpu().numpy()
        scoremaps = scoremaps.cpu().numpy()
        ans = refine(scoremaps, tagmap, ans)
    if with_adjust:
        ans = adjust(ans, scoremaps)
    if len(ans) == 0:
        return None
    persons_pred_orig = reverse_affine_map(ans.copy(), img_shape, input_size, scaling_type=scaling_type,
                                           min_scale=min_scale)

    ann = gen_ann_format(persons_pred_orig, img_id)

    return ann
if __name__ == "__main__":
    main()
