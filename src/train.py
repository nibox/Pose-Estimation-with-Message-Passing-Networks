import torch
from torch.utils.data import DataLoader
from CocoKeypoints import CocoKeypoints
import numpy as np
import pickle
import Models.PoseEstimation.PoseEstimation as pose
from Models.MessagePassingNetwork.VanillaMPN2 import default_config, VanillaMPN2
from torch_geometric.utils import recall, accuracy, precision, f1_score
from torch.utils.tensorboard import SummaryWriter
import os
from Utils.Utils import load_model


def create_train_validation_split(data_root, batch_size, mini=False):
    # todo connect the preprosseing with the model selection (input size etc)
    # todo add validation
    if mini:
        print("Using mini dataset")
        tv_split_name = "tmp/mini_train_valid_split_4.p"
        if not os.path.exists(tv_split_name):
            print(f"Creating train validation split {tv_split_name}")
            data_set = CocoKeypoints(data_root, mini=True, seed=0, mode="train")
            train, valid = torch.utils.data.random_split(data_set, [3500, 500])
            assert len(data_set.img_ids) == len(set(data_set.img_ids))
            train_valid_split = [train.dataset.img_ids[train.indices], valid.dataset.img_ids[valid.indices]]
            pickle.dump(train_valid_split, open(tv_split_name, "wb"))
        else:
            print(f"Loading train validation split {tv_split_name}")
            train_ids, valid_ids = pickle.load(open(tv_split_name, "rb"))
            print(set(train_ids).intersection(set(valid_ids)))
            assert len(set(train_ids).intersection(set(valid_ids))) == 0
            train = CocoKeypoints(data_root, mini=True, seed=0, mode="train", img_ids=train_ids)
            valid = CocoKeypoints(data_root, mini=True, seed=0, mode="train", img_ids=valid_ids)

        return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8), \
               DataLoader(valid, batch_size=batch_size, num_workers=8)
    else:
        raise NotImplementedError


def load_checkpoint(path, model_class, model_config, device):
    model = load_model(path, model_class, model_config, device)
    model.to(device)
    model.freeze_backbone()

    state_dict = torch.load(path)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    return model, optimizer, state_dict["epoch"]


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_path = "../../storage/user/kistern/coco"
    pretrained_path = "../PretrainedModels/pretrained/checkpoint.pth.tar"
    model_path =   "../log/PoseEstimationBaseline/9/pose_estimation.pth"

    log_dir = "../log/PoseEstimationBaseline/9"
    model_save_path = f"{log_dir}/pose_estimation_continue.pth"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # hyperparameters and other stuff
    learn_rate = 3e-5
    num_epochs = 100
    batch_size = 16  # pretty much largest possible batch size
    config = pose.default_config
    config["message_passing"] = VanillaMPN2
    config["message_passing_config"] = default_config
    config["cheat"] = False
    config["use_gt"] = True
    config["use_focal_loss"] = True
    config["use_neighbours"] = True

    ##########################################################
    print("Load model")
    if model_path is not None:
        model, optimizer, start_epoch = load_checkpoint(model_path,
                                                        pose.PoseEstimationBaseline, pose.default_config, device)
        start_epoch += 1
    else:
        model = load_model(model_path, pose.PoseEstimationBaseline, pose.default_config, device,
                           pretrained_path=pretrained_path)
        model.to(device)
        model.freeze_backbone()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        start_epoch = 0
    print("Load dataset")
    train_loader, valid_loader = create_train_validation_split(dataset_path, batch_size=batch_size, mini=True)
    print("#####Begin Training#####")
    epoch_len = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            iter = i + (epoch_len * (start_epoch + epoch))
            optimizer.zero_grad()
            # split batch
            imgs, masks, keypoints = batch
            imgs = imgs.to(device)
            #masks = masks.to(device)
            keypoints = keypoints.to(device)
            pred, joint_det, edge_index, edge_labels = model(imgs, keypoints)

            if len(edge_labels[edge_labels == 1]) != 0 and len(edge_labels[edge_labels == 0]) != 0:
                pos_weight = torch.tensor(len(edge_labels[edge_labels == 0]) / len(edge_labels[edge_labels == 1]))
            else:
                pos_weight = torch.tensor(1.0)
            loss = model.loss(pred, edge_labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            result = pred.sigmoid().squeeze()
            result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))

            # metrics
            # edge_labels[edge_labels<0.99] = 0.0  # some edge labels might be
            prec = precision(result, edge_labels, 2)[1]
            rec = recall(result, edge_labels, 2)[1]
            acc = accuracy(result, edge_labels)
            f1 = f1_score(result, edge_labels, 2)[1]
            print(f"Iter: {iter}, loss:{loss.item():6f}, "
                  f"Precision : {prec:5f} "
                  f"Recall: {rec:5f} "
                  f"Accuracy: {acc:5f} "
                  f"F1 score: {f1:5f}")

            writer.add_scalar("Loss/train", loss.item(), iter)
            writer.add_scalar("Metric/train_f1:", f1, iter)
            writer.add_scalar("Metric/train_prec:", prec, iter)
            writer.add_scalar("Metric/train_rec:", rec, iter)
        model.eval()

        print("#### BEGIN VALIDATION ####")
        valid_loss = []
        valid_acc = []
        valid_f1 = []
        valid_prec = []
        valid_recall = []
        with torch.no_grad():
            for batch in valid_loader:
                # split batch
                imgs, masks, keypoints = batch
                imgs = imgs.to(device)
                # masks = masks.to(device)
                keypoints = keypoints.to(device)
                pred, joint_det, edge_index, edge_labels = model(imgs, keypoints)

                if len(edge_labels[edge_labels == 1]) != 0:
                    pos_weight = torch.tensor(len(edge_labels[edge_labels == 0]) / len(edge_labels[edge_labels == 1]))
                else:
                    pos_weight = torch.tensor(1.0)
                loss = model.loss(pred, edge_labels, pos_weight=pos_weight)
                result = pred.sigmoid().squeeze()
                result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))

                valid_loss.append(loss.item())
                valid_acc.append(accuracy(result, edge_labels))
                valid_f1.append(f1_score(result, edge_labels, 2)[1])
                valid_prec.append(precision(result, edge_labels, 2)[1])
                valid_recall.append(recall(result, edge_labels, 2)[1])
        print(f"Epoch: {epoch + start_epoch}, loss:{np.mean(valid_loss):6f}, "
              f"Accuracy: {np.mean(valid_acc)}")

        writer.add_scalar("Loss/valid", np.mean(valid_loss), epoch + start_epoch)
        writer.add_scalar("Metric/valid_acc", np.mean(valid_acc), epoch + start_epoch)
        writer.add_scalar("Metric/valid_f1:", np.mean(valid_f1), epoch + start_epoch)
        writer.add_scalar("Metric/valid_prec:", np.mean(valid_prec), epoch + start_epoch)
        writer.add_scalar("Metric/valid_recall:", np.mean(valid_recall), epoch + start_epoch)
        torch.save({"epoch": epoch + start_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    }, model_save_path)
    writer.close()


if __name__ == "__main__":
    main()
