import torch
import torchvision
from data import CocoKeypoints_hr, HeatmapGenerator, JointsGenerator
import numpy as np
import pickle
from config import get_config, update_config
from Models import get_upper_bound_model
from Utils import num_non_detected_points, to_tensor
from tqdm import tqdm
from Utils.transforms import transforms_hr_eval
from matplotlib import pyplot as plt
import pylab as pl


def score_dist(keypoints: torch.Tensor, scoremaps: torch.Tensor):

    pool = torch.nn.MaxPool2d(5, 1, 2)
    scoremaps = pool(scoremaps)
    keypoints = keypoints.squeeze()
    person_idx, joint_type_idx = keypoints[:, :, 2].nonzero(as_tuple=True)

    points = keypoints[person_idx, joint_type_idx].round().long()
    x, y, flag = points[:, 0], points[:, 1], points[:, 2]
    scores = scoremaps[0, joint_type_idx, y, x]  # todo max from neighourhood
    scores = scores.cpu().numpy()
    flag = flag.cpu().numpy()

    return scores, scores[flag == 2], scores[flag==1]

def plot_hist(log_space, bins=None, **kwargs):
    for key in kwargs.keys():
        ## pylab
        fname = f"tmp/{key}_dist.png"
        if log_space:
            pl.hist(np.array(kwargs[key]), bins=np.logspace(np.log10(0.0001), np.log(1.0), 50))
            pl.gca().set_xscale("log")
        else:
            pl.hist(np.array(kwargs[key]), bins=np.linspace(0, 30, 15))
        pl.savefig(fname)
        pl.close()
        """
        fig = plt.figure()
        plt.hist(np.array(kwargs[key]), bins=100)

        fname = f"tmp/{key}_dist.png"
        plt.savefig(fig=fig, fname=fname)
        plt.close(fig)
        """

def regression_target_distribution(joint_det, node_labels, node_persons, node_target, node_classes):

    node_mask = node_labels == 1.0  # this might be a function

    joint_det_refine = joint_det[node_mask]
    node_persons = node_persons[node_mask]  # these are the clusters
    node_types = node_classes[node_mask] if node_classes is not None else joint_det_refine[:, 2]
    # sorted
    node_persons, sorted_idx = torch.sort(node_persons)
    joint_det_refine = joint_det_refine[sorted_idx]
    node_types = node_types[sorted_idx]

    distances = torch.norm(joint_det_refine[:, :2] - node_target[:, :2], dim=1)

    return distances.cpu().numpy()


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = get_config()
    config = update_config(config, "../experiments/upper_bound/hrnet.yaml")

    model = get_upper_bound_model(config, device=device)
    model.eval()

    _, valid_ids = pickle.load(open("tmp/coco_17_mini_split.p", "rb"))


    heatmap_generator = [HeatmapGenerator(128, 17, 2), HeatmapGenerator(256, 17, 2)]
    joint_generator = [JointsGenerator(30, 17, 128, True), JointsGenerator(30, 17, 256, True)]
    transforms, _ = transforms_hr_eval(config)
    img_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids, year=17,
                                transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=False,
                               joint_generator=joint_generator)

    num_detections = []
    num_edges = []
    num_det_failures = []
    total_missed_persons = 0
    total_persons = 0
    num_persons = []
    imbalance_edge = []
    imbalance_node = []
    scores_all = []
    scores_all_visible = []
    scores_all_occluded = []
    scores_by_kpt_num = []
    scores_by_size = []
    reg_distances = []
    deg = []
    # todo number of not detected keypoints
    with torch.no_grad():
        for i in tqdm(range(500)):  # just test the first 100 images
            # split batch
            img, _, masks, keypoints, factors, _ = img_set[i]
            img_transformed = img.to(device)[None]
            masks, keypoints, factors = to_tensor(device, masks[-1], keypoints, factors)
            # scoremaps, pred_edge, pred_node, pred_class, joint_det, joint_scores, edge_index, edge_labels, node_labels, class_labels, _, _, _ = model(img_transformed, keypoints, masks, factors)
            scoremaps, output = model(img_transformed, keypoints, masks, factors)

            imbalance_edge.append(output["labels"]["edge"].mean().item())
            imbalance_node.append(output["labels"]["node"].mean().item())
            num_non_detected, num_gt, num_missed_persons, num_persons = num_non_detected_points(output["graph"]["nodes"], keypoints[0], factors[0])
            total_missed_persons += num_missed_persons
            total_persons += num_persons
            num_detections.append(len(output["graph"]["nodes"]))
            num_edges.append(len(output["labels"]["edge"]))
            num_det_failures.append(float(num_non_detected) / num_gt)
            scores, scores_visible, scores_occluded = score_dist(keypoints, scoremaps)
            scores_all += list(scores)
            scores_all_visible += list(scores_visible)
            scores_all_occluded += list(scores_occluded)
            # joint_refined = output["preds"]["refine"]
            # reg_distances += list(regression_target_distribution(output["graph"]["nodes"], output["labels"]["node"], output["labels"]["refine"],
            #                                                 joint_refined[:, :2], output["preds"]["class"]))


        print(f"Average number of detections:{np.mean(num_detections)}")
        print(f"Std number of detections:{np.std(num_detections)}")
        print(f"Average number of edges:{np.mean(num_edges)}")
        print(f"Std number of edges:{np.std(num_edges)}")
        print(f"Average Edge Imbalance: {np.mean(imbalance_edge)}")
        print(f"Average Node Imbalance: {np.mean(imbalance_node)}")
        print(f"Average detection failure: {np.mean(num_det_failures)}")
        print(f"Detected persons: {float(total_missed_persons) / total_persons}")
        plot_hist(True, scores_all=scores_all,
                  scores_visible=scores_all_visible,
                  scores_occluded=scores_all_occluded,
                  )
        plot_hist(False, bins=50, location_distances=reg_distances)



if __name__ == "__main__":
    main()
