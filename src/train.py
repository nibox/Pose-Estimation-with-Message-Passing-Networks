import torch
from torch.utils.data import DataLoader
from data import CocoKeypoints_hg
from config import get_config, update_config
import numpy as np
import pickle
from Models import get_pose_model
from Models.MessagePassingNetwork.VanillaMPNNew import default_config, VanillaMPN
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
            train = CocoKeypoints_hg(data_root, mini=True, seed=0, mode="train", img_ids=train_ids)
            valid = CocoKeypoints_hg(data_root, mini=True, seed=0, mode="train", img_ids=valid_ids)

            return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8), \
                   DataLoader(valid, batch_size=1, num_workers=8)
    else:
        raise NotImplementedError

"""
def load_checkpoint(path, model_class, model_config, device):
    model = pose.load_model(path, model_class, model_config, device)
    model.to(device)
    model.freeze_backbone()

    state_dict = torch.load(path)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
   
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 60)
    if "lr_scheduler_state_dict" in state_dict:
        scheduler.load_state_dict(state_dict["lr_scheduler_state_dict"])
    

    return model, optimizer, state_dict["epoch"], scheduler
"""


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

                _, pred, joint_det, edge_index, edge_labels, label_mask, batch_index = model(imgs[None, i],
                                                                                          keypoints[None, i],
                                                                                          masks[None, i],
                                                                                          factors[None, i])

                label_mask = label_mask if kwargs["use_label_mask"] else None
                batch_index = batch_index if kwargs["use_batch_index"] else None

                local_loss = model.mpn_loss(pred, edge_labels, reduction=kwargs["loss_reduction"],
                                            pos_weight=None, mask=label_mask, batch_index=batch_index)

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

            _, preds, joint_det, edge_index, edge_labels, label_mask, batch_index = model(imgs, keypoints, masks, factors)

            label_mask = label_mask if kwargs["use_label_mask"] else None
            batch_index = batch_index if kwargs["use_batch_index"] else None

            loss = model.mpn_loss(preds, edge_labels, reduction="mean", mask=label_mask, batch_index=batch_index)
            loss.backward()
            optimizer.step()
            loss = loss.item()

        return preds, edge_labels, loss

    return func


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    ##########################################################
    config_name = "model_29"
    config = get_config()
    config = update_config(config, f"../experiments/train/{config_name}.yaml")

    """
    dataset_path = "../../storage/user/kistern/coco"
    pretrained_path = "../PretrainedModels/pretrained/checkpoint.pth.tar"
    model_path = None  # "../log/PoseEstimationBaseline/13/pose_estimation.pth"
    """

    # log_dir = "../log/PoseEstimationBaseline/Real/24_5"
    # model_save_path = f"{log_dir}/pose_estimation.pth"
    os.makedirs(config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(config.LOG_DIR)

    # learn_rate = 3e-4
    # hr_learn_rate = 3e-8
    # num_epochs = 100
    # batch_size = 8  # 16 is pretty much largest possible batch size
    """
    config = pose.default_config
    config["message_passing"] = VanillaMPN
    config["message_passing_config"] = default_config
    config["message_passing_config"]["aggr"] = "max"
    config["message_passing_config"]["edge_input_dim"] = 2 + 17
    config["message_passing_config"]["edge_feature_dim"] = 64
    config["message_passing_config"]["node_feature_dim"] = 64
    config["message_passing_config"]["steps"] = 10
    config["message_passing_config"]["skip"] = True
    config["message_passing_config"]["edge_from_node"] = True
    #config["message_passing_config"]["bn"] = True

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
    config["num_aux_steps"] = 1  # default is 1: only the last
    """

    # end_to_end = False  # this also enables batching simulation where the gradient is accumulated for multiple batches
    # loss_reduction = "mean"  # default is "mean"

    # use_label_mask = True
    # use_batch_index = False

    ##########################################################
    if not config.MODEL.GC.USE_GT:
        assert config.TRAIN.USE_LABEL_MASK  # this ensures that images with no "persons"/clusters do not contribute to the loss
    print("Load model")
    """
    if model_path is not None:
        raise 
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
    """
    model = get_pose_model(config, device)
    if config.TRAIN.END_TO_END:
        model.freeze_backbone(partial=True)
        optimizer = torch.optim.Adam([{"params": model.mpn.parameters(), "lr": config.TRAIN.LR},
                                      {"params": model.backbone.parameters(), "lr": config.TRAIN.KP_LR}])
    else:
        model.freeze_backbone(partial=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)
    model.to(device)

    if config.TRAIN.CONTINUE != "":
        state_dict = torch.load(config.TRAIN.CONTINUE)
        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict["lr_scheduler_state_dict"])

    print("Load dataset")
    train_loader, valid_loader = create_train_validation_split(config.DATASET.ROOT, batch_size=config.TRAIN.BATCH_SIZE,
                                                               mini=True)

    update_model = make_train_func(model, optimizer, use_batch_index=config.TRAIN.USE_BATCH_INDEX,
                                   use_label_mask=config.TRAIN.USE_LABEL_MASK, device=device,
                                   end_to_end=config.TRAIN.END_TO_END, batch_size=config.TRAIN.BATCH_SIZE,
                                   loss_reduction=config.TRAIN.LOSS_REDUCTION)

    print("#####Begin Training#####")
    epoch_len = len(train_loader)
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.END_EPOCH):
        model.train()
        for i, batch in enumerate(train_loader):
            iter = i + (epoch_len * epoch)

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
                _, preds, joint_det, edge_index, edge_labels, label_mask, batch_index = model(imgs, keypoints, masks, factors)

                label_mask = label_mask if config.TRAIN.USE_LABEL_MASK else None
                loss = model.mpn_loss(preds, edge_labels, reduction="mean", mask=label_mask)
                result = preds.sigmoid().squeeze()
                result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))

                valid_loss.append(loss.item())
                valid_acc.append(accuracy(result, edge_labels))
                valid_f1.append(f1_score(result, edge_labels, 2)[1])
                valid_prec.append(precision(result, edge_labels, 2)[1])
                valid_recall.append(recall(result, edge_labels, 2)[1])
        print(f"Epoch: {epoch}, loss:{np.mean(valid_loss):6f}, "
              f"Accuracy: {np.mean(valid_acc)}, "
              f"Precision: {np.mean(valid_prec)}")
        scheduler.step()

        writer.add_scalar("Loss/valid", np.mean(valid_loss), epoch)
        writer.add_scalar("Metric/valid_acc", np.mean(valid_acc), epoch)
        writer.add_scalar("Metric/valid_f1:", np.mean(valid_f1), epoch)
        writer.add_scalar("Metric/valid_prec:", np.mean(valid_prec), epoch)
        writer.add_scalar("Metric/valid_recall:", np.mean(valid_recall), epoch)
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": scheduler.state_dict()
                    }, config.MODEL.PRETRAINED)
    writer.close()


if __name__ == "__main__":
    main()
