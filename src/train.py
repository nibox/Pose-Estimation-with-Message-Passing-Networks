import torch
from torch.utils.data import DataLoader
from CocoKeypoints import CocoKeypoints
import numpy as np
import Models.PoseEstimation.PoseEstimation as pose
from torch_geometric.utils import recall, accuracy, precision
from torch.utils.tensorboard import SummaryWriter
import os


def create_train_validation_split(data_root, batch_size, mini=False):
    # todo connect the preprosseing with the model selection (input size etc)
    # todo add validation
    if mini:
        data_set = CocoKeypoints(data_root, mini=True, seed=0, mode="train")
        train, valid = torch.utils.data.random_split(data_set, [3500, 500])
        return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(valid, batch_size=batch_size)
    else:
        raise NotImplementedError


def load_model(path, device, pretrained_path=None):

    assert not (path is not None and pretrained_path is not None)
    def rename_key(key):
        # assume structure is model.module.REAL_NAME
        return ".".join(key.split(".")[2:])

    #model = hourglass.PoseNet(kwargs["nstack"], kwargs["input_dim"], kwargs["output_size"])
    model = pose.PoseEstimationBaseline(pose.default_config)
    if path is not None:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    elif pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=device)
        state_dict_new = {rename_key(k): v for k, v in state_dict["state_dict"].items()}
        model.backbone.load_state_dict(state_dict_new)

    return model


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_path = "../../storage/user/kistern/coco"
    pretrained_path = "../PretrainedModels/pretrained/checkpoint.pth.tar"
    model_path = None

    log_dir = "../log/PoseEstimationBaseline/1"
    model_save_path = f"{log_dir}/pose_estimation.pth"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)


    # hyperparameters
    learn_rate = 3e-5
    num_epochs = 10
    batch_size = 16

    print("Load model")
    model = load_model(model_path, device, pretrained_path=pretrained_path)
    model.to(device)
    model.freeze_backbone()
    print("Load dataset")
    train_loader, valid_loader = create_train_validation_split(dataset_path, batch_size=batch_size, mini=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

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
                  f"Precision : {precision(result, edge_labels, 2)} "
                  f"Recall: {recall(result, edge_labels, 2)} "
                  f"Accuracy: {accuracy(result, edge_labels)}")

            writer.add_scalar("Loss/train", loss.item(), iter)
        model.eval()

        print("#### BEGIN VALIDATION ####")
        valid_loss = []
        valid_acc = []
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
        print(f"Epoch: {epoch}, loss:{np.mean(valid_loss):6f}, "
              f"Accuracy: {np.mean(valid_acc)}")

        writer.add_scalar("Loss/valid", np.mean(valid_loss), epoch)
        writer.add_scalar("Metric/valid_acc", np.mean(valid_acc), epoch)
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    }, model_save_path)
    writer.close()


if __name__ == "__main__":
    main()
