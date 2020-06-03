import torch
from torch.utils.data import DataLoader
from CocoKeypoints import CocoKeypoints
import numpy as np
import pickle
import Models.PoseEstimation.PoseEstimation as pose
from Models.MessagePassingNetwork.VanillaMPN import default_config, VanillaMPN
from torch_geometric.utils import recall, accuracy, precision, f1_score
from torch.utils.tensorboard import SummaryWriter
import os


def load_data(data_root, batch_size, device):
    tv_split_name = "../tmp/mini_train_valid_split_4.p"
    train_ids, _ = pickle.load(open(tv_split_name, "rb"))
    train_ids = np.random.choice(train_ids, batch_size, replace=False)
    train = CocoKeypoints(data_root, mini=True, seed=0, mode="train", img_ids=train_ids)

    imgs = []
    masks = []
    keypoints = []
    factors = []
    for i in range(len(train)):
        sample = train.get_tensor(i, device)
        imgs.append(sample[0])
        masks.append(sample[1])
        keypoints.append(sample[2])
        factors.append(sample[3])
    batch = (torch.cat(imgs), torch.cat(masks), torch.cat(keypoints), torch.cat(factors))

    return batch

def make_train_func(model, optimizer, **kwargs):
    def func(batch):
        if kwargs["end_to_end"]:
            optimizer.zero_grad()
            # split batch
            imgs, masks, keypoints, factors = batch
            imgs = imgs.to(kwargs["device"])
            masks = masks.to(kwargs["device"])
            keypoints = keypoints.to(kwargs["device"])
            factors = factors.to(kwargs["device"])

            loss = 0.0
            preds, labels = [], []
            batch_size = imgs.shape[0]

            num_edges_in_batch = 0
            for i in range(batch_size):

                scoremaps, pred, joint_det, edge_index, edge_labels, label_mask, batch_index = model(imgs[None, i],
                                                                                          keypoints[None, i],
                                                                                          masks[None, i],
                                                                                          factors[None, i])

                label_mask = label_mask if kwargs["use_label_mask"] else None
                batch_index = batch_index if kwargs["use_batch_index"] else None

                local_loss = model.loss(pred, edge_labels, reduction=kwargs["loss_reduction"],
                                        mask=label_mask, batch_index=batch_index)
                num_edges_in_batch += len(edge_labels)
                # in case of loss_reduction="sum" the scaling is done at the gradient directly
                local_loss.backward()
                loss += local_loss.item()

                preds.append(pred.detach())
                labels.append(edge_labels.detach())

            if kwargs["loss_reduction"] == "sum":
                norm_const = num_edges_in_batch
            elif kwargs["loss_reduction"] == "mean":
                norm_const = batch_size
            else:
                raise NotImplementedError
            loss /= norm_const  # this is for logging
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data /= norm_const
            optimizer.step()

            preds = torch.cat(preds, 1)
            edge_labels = torch.cat(labels)
        else:
            optimizer.zero_grad()
            # split batch
            imgs, masks, keypoints, factors = batch
            imgs = imgs.to(kwargs["device"])
            masks = masks.to(kwargs["device"])
            keypoints = keypoints.to(kwargs["device"])
            factors = factors.to(kwargs["device"])

            scoremaps, preds, joint_det, edge_index, edge_labels, label_mask, batch_index = model(imgs, keypoints, masks, factors)

            label_mask = label_mask if kwargs["use_label_mask"] else None
            batch_index = batch_index if kwargs["use_batch_index"] else None

            loss = model.loss(preds, edge_labels, reduction=kwargs["loss_reduction"], mask=label_mask, batch_index=batch_index)
            loss.backward()
            optimizer.step()
            loss = loss.item()

        return preds[-1], edge_labels, loss

    return func


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    ##########################################################
    dataset_path = "../../../storage/user/kistern/coco"
    pretrained_path = "../../PretrainedModels/pretrained/checkpoint.pth.tar"

    learn_rate = 3e-4
    hr_learn_rate = 3e-8
    max_num_iter = 100
    batch_size = 8  # 16 is pretty much largest possible batch size
    config = pose.default_config
    config["message_passing"] = VanillaMPN
    config["message_passing_config"] = default_config
    config["message_passing_config"]["aggr"] = "max"
    config["message_passing_config"]["edge_input_dim"] = 2 + 17
    config["message_passing_config"]["edge_feature_dim"] = 64
    config["message_passing_config"]["node_feature_dim"] = 64
    config["message_passing_config"]["steps"] = 10

    config["num_aux_steps"] = 6
    config["cheat"] = False
    config["use_gt"] = False
    config["use_focal_loss"] = True
    config["use_neighbours"] = True
    config["mask_crowds"] = True
    config["detect_threshold"] = 0.005  # default was 0.007
    config["mpn_graph_type"] = "knn"
    config["edge_label_method"] = 4  # this only applies if use_gt==True
    config["matching_radius"] = 0.1
    config["inclusion_radius"] = 0.75

    end_to_end = True  # this also enables batching simulation where the gradient is accumulated for multiple batches
    loss_reduction = "sum" if end_to_end else "mean"
    loss_reduction = "mean"

    use_label_mask = True
    use_batch_index = False

    ##########################################################
    if config["use_gt"]:
        assert use_label_mask  # this ensures that images with no "persons"/clusters do not contribute to the loss
    print("Load model")
    model = pose.load_model(None, pose.PoseEstimationBaseline, config, device,
                            pretrained_path=pretrained_path)
    model.to(device)

    if end_to_end:
        model.freeze_backbone(partial=True)
        optimizer = torch.optim.Adam([{"params": model.mpn.parameters(), "lr": learn_rate},
                                      {"params": model.backbone.parameters(), "lr": hr_learn_rate}])
    else:
        model.freeze_backbone(partial=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    print("Load dataset")

    update_model = make_train_func(model, optimizer, use_batch_index=use_batch_index, use_label_mask=use_label_mask,
                                   device=device, end_to_end=end_to_end, batch_size=batch_size,
                                   loss_reduction=loss_reduction)
    print("#####Begin Training#####")
    batch = load_data(dataset_path, batch_size, device)
    model.train()
    for iter in range(max_num_iter):

        pred, edge_labels, loss = update_model(batch)

        result = pred.sigmoid().squeeze()
        result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))

        prec = precision(result, edge_labels, 2)[1]
        rec = recall(result, edge_labels, 2)[1]
        acc = accuracy(result, edge_labels)
        f1 = f1_score(result, edge_labels, 2)[1]
        print(f"Iter: {iter}, loss:{loss:6f}, "
              f"Precision : {prec:5f} "
              f"Recall: {rec:5f} "
              f"Accuracy: {acc:5f} "
              f"F1 score: {f1:5f}")

if __name__ == "__main__":
    main()
