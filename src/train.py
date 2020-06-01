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


def create_train_validation_split(data_root, batch_size, mini=False):
    # todo connect the preprosseing with the model selection (input size etc)
    # todo add validation
    if mini:
        print("Using mini dataset")
        tv_split_name = "tmp/mini_train_valid_split_4.p"
        if not os.path.exists(tv_split_name):
            print(f"Split cannot be created {tv_split_name} is missing.")
            raise FileNotFoundError
        else:
            print(f"Loading train validation split {tv_split_name}")
            train_ids, valid_ids = pickle.load(open(tv_split_name, "rb"))
            print(set(train_ids).intersection(set(valid_ids)))
            assert len(set(train_ids).intersection(set(valid_ids))) == 0
            train = CocoKeypoints(data_root, mini=True, seed=0, mode="train", img_ids=train_ids)
            valid = CocoKeypoints(data_root, mini=True, seed=0, mode="train", img_ids=valid_ids)

            return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8), \
                   DataLoader(valid, batch_size=1, num_workers=8)
    else:
        raise NotImplementedError


def load_checkpoint(path, model_class, model_config, device):
    model = pose.load_model(path, model_class, model_config, device)
    model.to(device)
    model.freeze_backbone()

    state_dict = torch.load(path)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    #"""
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 60)
    if "lr_scheduler_state_dict" in state_dict:
        scheduler.load_state_dict(state_dict["lr_scheduler_state_dict"])
    # """

    return model, optimizer, state_dict["epoch"], scheduler


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

                pred, joint_det, edge_index, edge_labels, label_mask, batch_index = model(imgs[None, i],
                                                                                          keypoints[None, i],
                                                                                          masks[None, i],
                                                                                          factors[None, i])

                label_mask = label_mask if kwargs["use_label_mask"] else None
                batch_index = batch_index if kwargs["use_batch_index"] else None

                local_loss = model.loss(pred, edge_labels, reduction=kwargs["loss_reduction"],
                                        pos_weight=None, mask=label_mask, batch_index=batch_index)
                if kwargs["loss_reduction"] == "mean":
                    local_loss /= batch_size
                num_edges_in_batch += len(edge_labels)
                # in case of loss_reduction="sum" the scaling is done at the gradient directly
                local_loss.backward()
                loss += local_loss.item()
                preds.append(pred.detach())
                labels.append(edge_labels.detach())
            pred = torch.cat(preds)
            edge_labels = torch.cat(labels)
            if kwargs["loss_reduction"] == "sum":
                loss /= num_edges_in_batch
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data /= num_edges_in_batch
            optimizer.step()
        else:
            optimizer.zero_grad()
            # split batch
            imgs, masks, keypoints, factors = batch
            imgs = imgs.to(kwargs["device"])
            masks = masks.to(kwargs["device"])
            keypoints = keypoints.to(kwargs["device"])
            factors = factors.to(kwargs["device"])
            pred, joint_det, edge_index, edge_labels, label_mask, batch_index = model(imgs, keypoints, masks, factors)

            label_mask = label_mask if kwargs["use_label_mask"] else None
            batch_index = batch_index if kwargs["use_batch_index"] else None

            loss = model.loss(pred, edge_labels, pos_weight=None, mask=label_mask, batch_index=batch_index)
            loss.backward()
            optimizer.step()
            loss = loss.item()

        return pred, edge_labels, loss

    return func


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    ##########################################################
    dataset_path = "../../storage/user/kistern/coco"
    pretrained_path = "../PretrainedModels/pretrained/checkpoint.pth.tar"
    model_path = None  # "../log/PoseEstimationBaseline/13/pose_estimation.pth"

    log_dir = "../log/PoseEstimationBaseline/Real/24"
    model_save_path = f"{log_dir}/pose_estimation.pth"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    learn_rate = 3e-4
    num_epochs = 100
    batch_size = 8  # 16 is pretty much largest possible batch size
    config = pose.default_config
    config["message_passing"] = VanillaMPN
    config["message_passing_config"] = default_config
    config["message_passing_config"]["aggr"] = "max"
    config["message_passing_config"]["edge_input_dim"] = 2 + 17
    config["message_passing_config"]["edge_feature_dim"]= 64
    config["message_passing_config"]["node_feature_dim"]= 64
    config["message_passing_config"]["steps"] = 10

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
    loss_reduction = "sum"  # default is "mean"

    use_label_mask = True
    use_batch_index = False

    ##########################################################
    if config["use_gt"]:
        assert use_label_mask  # this ensures that images with no "persons"/clusters do not contribute to the loss
    print("Load model")
    if model_path is not None:
        model, optimizer, start_epoch, scheduler = load_checkpoint(model_path,
                                                        pose.PoseEstimationBaseline, config, device)
        start_epoch += 1
    else:
        model = pose.load_model(model_path, pose.PoseEstimationBaseline, config, device,
                           pretrained_path=pretrained_path)
        model.to(device)

        if end_to_end:
            model.freeze_backbone(partial=True)
            optimizer = torch.optim.Adam([{"params": model.mpn.parameters(), "lr": learn_rate},
                                          {"params": model.backbone.parameters(), "lr": hr_learn_rate}])
        else:
            model.freeze_backbone(partial=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 150], 0.1)
        start_epoch = 0
    print("Load dataset")
    train_loader, valid_loader = create_train_validation_split(dataset_path, batch_size=batch_size, mini=True)

    update_model = make_train_func(model, optimizer, use_batch_index=use_batch_index, use_label_mask=use_label_mask,
                                   device=device, end_to_end=end_to_end, batch_size=batch_size,
                                   loss_reduction=loss_reduction)

    print("#####Begin Training#####")
    epoch_len = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            iter = i + (epoch_len * (start_epoch + epoch))

            pred, edge_labels, loss = update_model(batch)

            result = pred.sigmoid().squeeze()
            result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))

            # metrics
            # edge_labels[edge_labels<0.99] = 0.0  # some edge labels might be
            prec = precision(result, edge_labels, 2)[1]
            rec = recall(result, edge_labels, 2)[1]
            acc = accuracy(result, edge_labels)
            f1 = f1_score(result, edge_labels, 2)[1]
            print(f"Iter: {iter}, loss:{loss:6f}, "
                  f"Precision : {prec:5f} "
                  f"Recall: {rec:5f} "
                  f"Accuracy: {acc:5f} "
                  f"F1 score: {f1:5f}")

            writer.add_scalar("Loss/train", loss, iter)
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
                imgs, masks, keypoints, factors = batch
                imgs = imgs.to(device)
                masks = masks.to(device)
                keypoints = keypoints.to(device)
                factors = factors.to(device)
                pred, joint_det, edge_index, edge_labels, label_mask, batch_index = model(imgs, keypoints, masks, factors)

                if len(edge_labels[edge_labels == 1]) != 0:
                    pos_weight = torch.tensor(len(edge_labels[edge_labels == 0]) / len(edge_labels[edge_labels == 1]))
                else:
                    pos_weight = torch.tensor(1.0)
                label_mask = label_mask if use_label_mask else None
                loss = model.loss(pred, edge_labels, reduction="mean", pos_weight=pos_weight, mask=label_mask)
                result = pred.sigmoid().squeeze()
                result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))

                valid_loss.append(loss.item())
                valid_acc.append(accuracy(result, edge_labels))
                valid_f1.append(f1_score(result, edge_labels, 2)[1])
                valid_prec.append(precision(result, edge_labels, 2)[1])
                valid_recall.append(recall(result, edge_labels, 2)[1])
        print(f"Epoch: {epoch + start_epoch}, loss:{np.mean(valid_loss):6f}, "
              f"Accuracy: {np.mean(valid_acc)}, "
              f"Precision: {np.mean(valid_prec)}")
        scheduler.step()

        writer.add_scalar("Loss/valid", np.mean(valid_loss), epoch + start_epoch)
        writer.add_scalar("Metric/valid_acc", np.mean(valid_acc), epoch + start_epoch)
        writer.add_scalar("Metric/valid_f1:", np.mean(valid_f1), epoch + start_epoch)
        writer.add_scalar("Metric/valid_prec:", np.mean(valid_prec), epoch + start_epoch)
        writer.add_scalar("Metric/valid_recall:", np.mean(valid_recall), epoch + start_epoch)
        torch.save({"epoch": epoch + start_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": scheduler.state_dict()
                    }, model_save_path)
    writer.close()


if __name__ == "__main__":
    main()
