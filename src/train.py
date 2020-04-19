import torch
from torch.utils.data import DataLoader
import Models.Hourglass.Hourglass as hourglass
from CocoKeypoints import CocoKeypoints
import numpy as np
import Models.PoseEstimation.PoseEstimation as pose


def create_train_validation_split(data_root, batch_size):
    # todo connect the preprosseing with the model selection (input size etc)
    # todo add validation
    train_set = CocoKeypoints(data_root, mini=True, seed=0, mode="train")
    return DataLoader(train_set, batch_size=batch_size, shuffle=True), None


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


    # hyperparameters
    learn_rate = 3e-5
    num_epochs = 10
    batch_size = 1

    print("Load model")
    model = load_model(model_path, device, pretrained_path=pretrained_path)
    model.to(device)
    model.freeze_backbone()
    print("Load dataset")
    train_loader, valid_loader = create_train_validation_split(dataset_path, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    print("#####Begin Training#####")
    for epoch in range(num_epochs):
        for iter, batch in enumerate(train_loader):

            # split batch
            imgs, masks, keypoints = batch
            imgs = imgs.to(device)
            masks = masks.to(device)
            keypoints = keypoints.to(device)
            pred, joint_det, edge_index, edge_labels = model(imgs, keypoints)

            loss = model.loss(pred, edge_labels)
            loss.backward()
            optimizer.step()


            print(f"Iter: {iter}, loss:{loss.item()}")
            # scoremap, features, early_features = model(imgs.to(device))
            # torch.save(early_features.detach().cpu(), "early_features.pt")
            """torch.save(imgs.detach().cpu(), "imgs.pt")
            torch.save(masks.detach().cpu(), "masks.pt")
            torch.save(keypoints.detach().cpu(), "keypoints.pt")
            torch.save(scoremap.detach().cpu(), "score.pt")
            torch.save(features.detach().cpu(), "features.pt")"""

            # forward pass

            # loss calculation


            # backwards

            # metrics
        break
        # same for validation with more metrics?

if __name__ == "__main__":
    main()
