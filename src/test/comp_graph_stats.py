import torch
import torchvision
from data import CocoKeypoints_hr
import numpy as np
import pickle
from config import get_config, update_config
from Models import get_upper_bound_model
from Utils import num_non_detected_points, to_tensor
from tqdm import tqdm



def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = get_config()
    config = update_config(config, "../experiments/upper_bound/hrnet.yaml")

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    model = get_upper_bound_model(config, device=device)
    model.eval()

    train_ids, valid_ids = pickle.load(open("tmp/coco_17_mini_split.p", "rb"))
    ids = np.concatenate([train_ids, valid_ids])
    valid_ids = np.random.choice(valid_ids, 100, replace=False)
    img_set = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids, year=17,
                                output_size=256)

    num_detections = []
    num_edges = []
    num_det_failures = []
    imbalance = []
    deg = []
    # todo number of not detected keypoints
    for i in tqdm(range(100)):  # just test the first 100 images
        # split batch
        img, masks, keypoints, factors = img_set[i]
        img_transformed = transforms(img).to(device)[None]
        masks, keypoints, factors = to_tensor(device, masks, keypoints, factors)
        _, pred, joint_det, edge_index, edge_labels, _ = model(img_transformed, keypoints, masks, factors)
        #deg.append(degree(edge_index[1], len(joint_det)).mean())
        imbalance.append(edge_labels.mean().item())
        num_non_detected, num_gt = num_non_detected_points(joint_det, keypoints, 6.0,
                                                           config.MODEL.GC.USE_GT)

        num_detections.append(len(joint_det))
        num_edges.append(len(edge_labels))
        num_det_failures.append(float(num_non_detected) / num_gt)
    print(f"Average number of detections:{np.mean(num_detections)}")
    print(f"Std number of detections:{np.std(num_detections)}")
    print(f"Average number of edges:{np.mean(num_edges)}")
    print(f"Std number of edges:{np.std(num_edges)}")
    print(f"Average Imbalance: {np.mean(imbalance)}")
    print(f"Average detection failure: {np.mean(num_det_failures)}")
    #print(f"Average node degree: {np.mean(degree)}")


if __name__ == "__main__":
    main()
