import torch
import os
import numpy as np
from tqdm import tqdm

from config import update_config, get_hrnet_config
from data import CrowdPoseKeypoints, HeatmapGenerator, JointsGenerator
from Utils import to_tensor
from Models.PoseEstimation import get_hr_model
from Utils.transformations import reverse_affine_map
from Utils.transforms import transforms_minimal, transforms_to_tensor
from Utils.hr_utils import HeatmapParser
from Utils import draw_detection, draw_poses, pred_to_person, to_tensor, draw_detection_with_conf, draw_detection_with_cluster, draw_edges_conf, subgraph_mask, one_hot_encode


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and False else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ######################################

    config = get_hrnet_config()
    config = update_config(config, f"../experiments/hrnet/w32_512_adam_lr1e-3_crowdpose.yaml")
    output_dir = f"tmp/output_hr_crowdpose"
    os.makedirs(output_dir, exist_ok=True)

    parser = HeatmapParser(config)
    heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
    joint_generator = [JointsGenerator(30, 17, 128, True),
                       JointsGenerator(30, 17, 256, True)]
    transforms, transforms_inv = transforms_to_tensor(config)
    eval_set = CrowdPoseKeypoints(config.DATASET.ROOT, mini=True, seed=0, mode="val",
                                transforms=transforms, heatmap_generator=heatmap_generator,
                                 joint_generator=joint_generator)

    model = get_hr_model(config, device)
    model.to(device)
    model.eval()

    anns = []
    anns_with_people = []
    imgs_with_people = []

    eval_ids = []
    num_iter = 10
    with torch.no_grad():
        for i in tqdm(range(num_iter)):
            img_id = eval_set.img_ids[i]

            img, _, masks, keypoints, factors, _ = eval_set[i]
            img = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)

            heatmaps, tags = model.multi_scale_inference(img, device, config)

            grouped, scores = parser.parse(heatmaps, tags, adjust=True, refine=True)

            img_info = eval_set.coco.loadImgs(int(eval_set.img_ids[i]))[0]
            img = transforms_inv(img.cpu().squeeze())
            img = np.array(img)


            if len(grouped[0]) != 0:
                persons_pred_orig = reverse_affine_map(grouped[0].copy(), (img_info["width"], img_info["height"]),
                                                       config.DATASET.INPUT_SIZE,
                                                       scaling_type="short_with_resize",
                                                       min_scale=1.0)
                draw_poses(img.copy(), persons_pred_orig, f"{output_dir}/{i}_{img_id}_pred.png", output_size=512)


if __name__ == "__main__":
    main()
