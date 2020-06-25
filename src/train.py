import torch
from torch.utils.data import DataLoader
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator
from Utils.transforms import transforms_hr_eval, transforms_hr_train, transforms_hg_eval
from Utils.loss import MPNLossFactory
from config import get_config, update_config
import numpy as np
import pickle
from Models import get_pose_model
from torch_geometric.utils import recall, accuracy, precision, f1_score
from torch.utils.tensorboard import SummaryWriter
import os


def create_train_validation_split(config):
    batch_size = config.TRAIN.BATCH_SIZE
    # todo connect the preprosseing with the model selection (input size etc)
    # todo add validation
    if config.TRAIN.SPLIT == "mini":
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
            transforms = transforms_hg_eval(config)
            train = CocoKeypoints_hg(config.DATASET.ROOT, mini=True, seed=0, mode="train", img_ids=train_ids, transforms=transforms)
            valid = CocoKeypoints_hg(config.DATASET.ROOT, mini=True, seed=0, mode="train", img_ids=valid_ids, transforms=transforms)

            return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8), \
                   DataLoader(valid, batch_size=1, num_workers=8)
    elif config.TRAIN.SPLIT == "coco_17_mini":
        train_ids, valid_ids = pickle.load(open("tmp/coco_17_mini_split.p", "rb"))  # mini_train_valid_split_4 old one
        heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
        transforms, _ = transforms_hr_train(config)
        train = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="train", img_ids=train_ids, year=17,
                                    transforms=transforms, heatmap_generator=heatmap_generator)
        valid = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids, year=17,
                                 transforms=transforms, heatmap_generator=heatmap_generator)
        return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8), \
               DataLoader(valid, batch_size=1, num_workers=8)

    else:
        raise NotImplementedError

def make_train_func(model, optimizer, loss_func, **kwargs):
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
            imgs, _, masks, keypoints, factors = batch
            imgs = imgs.to(kwargs["device"])
            masks = masks[-1].to(kwargs["device"])
            keypoints = keypoints.to(kwargs["device"])
            factors = factors.to(kwargs["device"])

            _, preds, joint_det, edge_index, edge_labels, label_mask, batch_index = model(imgs, keypoints, masks, factors)

            label_mask = label_mask if kwargs["use_label_mask"] else None
            batch_index = batch_index if kwargs["use_batch_index"] else None

            loss = loss_func(preds, edge_labels, label_mask=label_mask)
            loss.backward()
            optimizer.step()
            loss = loss.item()

        return preds, edge_labels, loss, label_mask

    return func


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    ##########################################################
    config_name = "model_24_7"
    config = get_config()
    config = update_config(config, f"../experiments/train/{config_name}.yaml")

    os.makedirs(config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(config.LOG_DIR)

    ##########################################################
    if not config.MODEL.GC.USE_GT:
        assert config.TRAIN.USE_LABEL_MASK  # this ensures that images with no "persons"/clusters do not contribute to the loss
    print("Load model")
    model = get_pose_model(config, device)
    if config.TRAIN.END_TO_END:
        model.freeze_backbone(partial=True)
        optimizer = torch.optim.Adam([{"params": model.mpn.parameters(), "lr": config.TRAIN.LR},
                                      {"params": model.backbone.parameters(), "lr": config.TRAIN.KP_LR}])
    else:
        model.freeze_backbone(partial=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)
        loss_func = MPNLossFactory(config)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)
    model.to(device)

    if config.TRAIN.CONTINUE != "":
        state_dict = torch.load(config.TRAIN.CONTINUE)
        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict["lr_scheduler_state_dict"])

    print("Load dataset")
    train_loader, valid_loader = create_train_validation_split(config)

    update_model = make_train_func(model, optimizer, loss_func, use_batch_index=config.TRAIN.USE_BATCH_INDEX,
                                   use_label_mask=config.TRAIN.USE_LABEL_MASK, device=device,
                                   end_to_end=config.TRAIN.END_TO_END, batch_size=config.TRAIN.BATCH_SIZE,
                                   loss_reduction=config.TRAIN.LOSS_REDUCTION)

    print("#####Begin Training#####")
    epoch_len = len(train_loader)
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.END_EPOCH):
        model.train()
        if config.TRAIN.FREEZE_BN:
            model.stop_backbone_bn()
        for i, batch in enumerate(train_loader):
            iter = i + (epoch_len * epoch)

            pred, edge_labels, loss, label_mask = update_model(batch)

            result = pred.sigmoid().squeeze()
            result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))

            result = result[label_mask==1.0]
            edge_labels = edge_labels[label_mask==1.0]

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
                imgs, _, masks, keypoints, factors = batch
                imgs = imgs.to(device)
                masks = masks[-1].to(device)
                keypoints = keypoints.to(device)
                factors = factors.to(device)
                _, preds, joint_det, edge_index, edge_labels, label_mask, batch_index = model(imgs, keypoints, masks, factors)

                label_mask = label_mask if config.TRAIN.USE_LABEL_MASK else None
                loss = loss_func(preds, edge_labels, label_mask)
                result = preds.sigmoid().squeeze()
                result = torch.where(result < 0.5, torch.zeros_like(result), torch.ones_like(result))

                # remove masked connections from score calculation
                result = result[label_mask == 1.0]
                edge_labels = edge_labels[label_mask == 1.0]
                if len(result) == 0:
                    continue

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
