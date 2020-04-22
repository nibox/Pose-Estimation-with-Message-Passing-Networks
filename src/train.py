import torch
from torch.utils.data import DataLoader
from CocoKeypoints import CocoKeypoints
import numpy as np
import pickle
import Models.PoseEstimation.PoseEstimation as pose
from torch_geometric.utils import recall, accuracy, precision, f1_score
from torch.utils.tensorboard import SummaryWriter
import os
from Utils.Utils import load_model


def create_train_validation_split(data_root, batch_size, mini=False):
    # todo connect the preprosseing with the model selection (input size etc)
    # todo add validation
    if mini:
        if not os.path.exists("mini_train_valid_split.p"):
            data_set = CocoKeypoints(data_root, mini=True, seed=0, mode="train")
            train, valid = torch.utils.data.random_split(data_set, [3500, 500])
            train_valid_split = [train.dataset.img_ids[train.indices], valid.dataset.img_ids[valid.indices]]
            pickle.dump(train_valid_split, open("mini_train_valid_split.p", "wb"))
        else:
            train_ids, valid_ids = pickle.load(open("mini_train_valid_split.p", "rb"))
            train = CocoKeypoints(data_root, mini=True, seed=0, mode="train", img_ids=train_ids)
            valid = CocoKeypoints(data_root, mini=True, seed=0, mode="train", img_ids=valid_ids)

        return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(valid, batch_size=batch_size)
    else:
        raise NotImplementedError




def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_path = "../../storage/user/kistern/coco"
    pretrained_path = "../PretrainedModels/pretrained/checkpoint.pth.tar"
    model_path = None

    log_dir = "../log/PoseEstimationBaseline/2"
    model_save_path = f"{log_dir}/pose_estimation.pth"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)


    # hyperparameters
    learn_rate = 3e-5
    num_epochs = 30
    batch_size = 16  # pretty much largest possible batch size

    print("Load model")
    model = load_model(model_path, pose.PoseEstimationBaseline, pose.default_config, device, pretrained_path=pretrained_path)
    model.to(device)
    model.freeze_backbone()
    print("Load dataset")
    train_loader, valid_loader = create_train_validation_split(dataset_path, batch_size=batch_size, mini=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    print(len(train_loader))
    print("#####Begin Training#####")
    for epoch in range(num_epochs):
        model.train()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # split batch
            imgs, masks, keypoints = batch
            imgs = imgs.to(device)
            masks = masks.to(device)
            keypoints = keypoints.to(device)
            pred, joint_det, edge_index, edge_labels = model(imgs, keypoints)

            if len(edge_labels[edge_labels==1]) != 0:
                pos_weight = torch.tensor(len(edge_labels[edge_labels == 0]) / len(edge_labels[edge_labels == 1]))
            else:
                pos_weight = torch.tensor(1.0)
            loss = model.loss(pred, edge_labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            result = pred.sigmoid().squeeze()
            result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))

            print(f"Iter: {iter}, loss:{loss.item():6f}, "
                  f"Precision : {precision(result, edge_labels, 2)[1]:5f} "
                  f"Recall: {recall(result, edge_labels, 2)[1]:5f} "
                  f"Accuracy: {accuracy(result, edge_labels):5f} "
                  f"F1 score: {f1_score(result, edge_labels, 2)[1]:5f}")

            writer.add_scalar("Loss/train", loss.item(), iter + epoch * len(train_loader))
            writer.add_scalar("Metric/train_f1:", f1_score(result, edge_labels, 2)[1], iter + epoch * len(train_loader))
        model.eval()

        print("#### BEGIN VALIDATION ####")
        valid_loss = []
        valid_acc = []
        valid_f1 = []
        with torch.no_grad():
            for iter, batch in enumerate(valid_loader):
                # split batch
                imgs, masks, keypoints = batch
                imgs = imgs.to(device)
                masks = masks.to(device)
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
        print(f"Epoch: {epoch}, loss:{np.mean(valid_loss):6f}, "
              f"Accuracy: {np.mean(valid_acc)}")

        writer.add_scalar("Loss/valid", np.mean(valid_loss), epoch)
        writer.add_scalar("Metric/valid_acc", np.mean(valid_acc), epoch)
        writer.add_scalar("Metric/valid_f1:", np.mean(valid_f1), epoch)
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    }, model_save_path)
    writer.close()


if __name__ == "__main__":
    main()
