import torch
import os
import argparse
import numpy as np
from tqdm import tqdm

from config import update_config, get_hrnet_config, update_config_command
from data import CocoKeypoints_hr, HeatmapGenerator, JointsGenerator, CrowdPoseKeypoints, OCHumans
from Utils import to_tensor
from Models.PoseEstimation import get_hr_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_minimal, transforms_to_tensor
from Utils.hr_utils import HeatmapParser, cluster_cc
from Utils import draw_detection, draw_poses, pred_to_person, save_valid_image, draw_detection_with_conf, draw_detection_with_cluster, draw_edges_conf, subgraph_mask, one_hot_encode


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate poses and draw the results")
    parser.add_argument("--config", help="Config file name for the experiment.", required=True, type=str)
    parser.add_argument("--out_dir", help="Name of the target directory.", required=True, type=str)
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

    """
    config = get_hrnet_config()
    config = update_config(config, f"../experiments/hrnet/w32_512_adam_lr1e-3.yaml")
    output_dir = f"tmp/output_hr_multi"
    os.makedirs(output_dir, exist_ok=True)
    """

    args = parse_args()
    config = get_hrnet_config()
    config = update_config(config, f"../experiments/{args.config}")
    config = update_config_command(config, args.options)

    os.makedirs(f"tmp/{args.out_dir}", exist_ok=True)

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
    body_type = None
    if config.TEST.SPLIT == "coco_17_full":
        assert config.DATASET.NUM_JOINTS == 17
        body_type = "coco"
        eval_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=False, seed=0, mode="val", img_ids=None, year=17,
                                    transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=False,
                                    filter_empty=False, joint_generator=joint_generator)
    elif config.TEST.SPLIT == "test-dev2017":
        raise NotImplementedError
    elif config.TEST.SPLIT == "crowd_pose_test":
        assert config.DATASET.NUM_JOINTS == 14
        body_type = "crowd_pose"
        eval_set = CrowdPoseKeypoints(config.DATASET.ROOT, mini=False, seed=0, mode="test",
                                      transforms=transforms, heatmap_generator=heatmap_generator,
                                      filter_empty=False, joint_generator=joint_generator)
    elif config.TEST.SPLIT == "ochuman_valid":
        assert config.DATASET.NUM_JOINTS == 17
        body_type = "coco"
        eval_set = OCHumans('../../storage/user/kistern/OCHuman', seed=0, mode="val",
                            transforms=transforms, mask_crowds=False)
    elif config.TEST.SPLIT == "ochuman_test":
        assert config.DATASET.NUM_JOINTS == 17
        body_type = "coco"
        eval_set = OCHumans('../../storage/user/kistern/OCHuman', seed=0, mode="test",
                            transforms=transforms, mask_crowds=False)
    else:
        raise NotImplementedError

    model = get_hr_model(config, device)
    model.to(device)
    model.eval()


    num_iter = 50
    with torch.no_grad():
        for i in tqdm(range(num_iter)):

            img_id = eval_set.img_ids[i]
            img, _, masks, keypoints, factors, _ = eval_set[i]
            img = img.to(device)[None]

            heatmaps, tags = model.multi_scale_inference(img, device, config)

            grouped_heu, scores_heu = parser.parse(heatmaps, tags, adjust=config.TEST.ADJUST, refine=config.TEST.REFINE)
            grouped_cc, scores_cc = cluster_cc(heatmaps[0], tags[0], config)

            img_shape = (img.shape[3], img.shape[2])

            # gt keypoints
            # keypoints = keypoints.cpu().numpy()
            if keypoints.sum() != 0.0:
                save_valid_image(img, keypoints, f"tmp/{args.out_dir}/{i}_{img_id}_gt.png", body_type)
            if len(grouped_heu[0]) == 0:
                # dfasdf
                pass
            else:

                persons = reverse_affine_map(grouped_heu[0].copy(), img_shape,
                                                       config.DATASET.INPUT_SIZE,
                                                       scaling_type=scaling_type,
                                                       min_scale=min(config.TEST.SCALE_FACTOR))
                save_valid_image(img, persons, f"tmp/{args.out_dir}/{i}_{img_id}_heu.png", body_type)

            if len(grouped_cc) == 0:
                pass
            else:
                persons = reverse_affine_map(grouped_cc.copy(), img_shape,
                                             config.DATASET.INPUT_SIZE,
                                             scaling_type=scaling_type,
                                             min_scale=min(config.TEST.SCALE_FACTOR))
                save_valid_image(img, persons, f"tmp/{args.out_dir}/{i}_{img_id}_cc.png", body_type)


if __name__ == "__main__":
    main()
