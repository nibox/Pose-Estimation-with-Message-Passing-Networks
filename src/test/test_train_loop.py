import torch
import numpy as np
import pickle
from config import get_config, update_config
from torch_geometric.utils import recall, accuracy, precision, f1_score
from Utils.transforms import transforms_hr_train
from Utils.loss import MPNLossFactory
from Models import get_cached_model
from data import CocoKeypoints_hr, HeatmapGenerator


def load_data(config, batch_size, device):
    tv_split_name = "tmp/coco_17_mini_split.p"
    train_ids, _ = pickle.load(open(tv_split_name, "rb"))
    train_ids = np.random.choice(train_ids, batch_size, replace=False)
    transforms, _ = transforms_hr_train(config)
    heatmap_generator = [HeatmapGenerator(128, 17), HeatmapGenerator(256, 17)]
    train = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="train", img_ids=train_ids, year=17,
                             transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=True)

    imgs = []
    masks = []
    keypoints = []
    factors = []
    target_scoremaps = []
    for i in range(len(train)):
        img, _, mask, keypoint, factor = train[i]
        img = img.to(device)
        mask = torch.from_numpy(mask[-1]).to(device)
        keypoint = torch.from_numpy(keypoint).to(device)
        factor = torch.from_numpy(factor).to(device)

        imgs.append(img[None])
        masks.append(mask[None])
        keypoints.append(keypoint[None])
        factors.append(factor[None])
    return torch.cat(imgs), torch.cat(masks), torch.cat(keypoints), torch.cat(factors)

def make_train_func(model, optimizer, loss_func, **kwargs):
    def func(batch):
        if kwargs["end_to_end"]:
            optimizer.zero_grad()
            # split batch
            imgs, masks, keypoints, factors, target_scoremaps = batch
            imgs = imgs.to(kwargs["device"])
            masks = masks.to(kwargs["device"])
            keypoints = keypoints.to(kwargs["device"])
            factors = factors.to(kwargs["device"])
            target_scoremaps = target_scoremaps.to(kwargs["device"])

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

                local_loss = model.mpn_loss(pred, edge_labels, reduction=kwargs["loss_reduction"],
                                            mask=label_mask, batch_index=batch_index)
                # scoremap_loss = model.scoremap_loss(scoremaps, target_scoremaps[i], masks[None, i])
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

            scoremaps, preds, joint_det, _, edge_index, edge_labels, label_mask, batch_index = model(imgs, keypoints, masks, factors)

            label_mask = label_mask if kwargs["use_label_mask"] else None
            batch_index = batch_index if kwargs["use_batch_index"] else None

            loss = loss_func(preds, edge_labels, label_mask=label_mask)
            loss.backward()
            optimizer.step()
            loss = loss.item()

        return preds[label_mask==1], edge_labels[label_mask==1], loss

    return func


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    ##########################################################
    config_name = "model_35"
    config = get_config()
    config = update_config(config, f"../experiments/train/{config_name}.yaml")


    ##########################################################
    if not config.MODEL.GC.USE_GT:
        assert config.TRAIN.USE_LABEL_MASK  # this ensures that images with no "persons"/clusters do not contribute to the loss
    print("Load model")
    model = get_cached_model(config, device)
    if config.TRAIN.END_TO_END:
        model.freeze_backbone(partial=True)
        optimizer = torch.optim.Adam([{"params": model.mpn.parameters(), "lr": config.TRAIN.LR},
                                      {"params": model.backbone.parameters(), "lr": config.TRAIN.KP_LR}])
    else:
        model.freeze_backbone()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)
        loss_func = MPNLossFactory(config)
    model.to(device)

    update_model = make_train_func(model, optimizer, loss_func, use_batch_index=config.TRAIN.USE_BATCH_INDEX,
                                   use_label_mask=config.TRAIN.USE_LABEL_MASK, device=device,
                                   end_to_end=config.TRAIN.END_TO_END, batch_size=config.TRAIN.BATCH_SIZE,
                                   loss_reduction=config.TRAIN.LOSS_REDUCTION)
    print("#####Begin Training#####")
    batch = load_data(config, config.TRAIN.BATCH_SIZE, device)

    model.train()
    if config.TRAIN.FREEZE_BN:
            model.stop_backbone_bn()
    for iter in range(10000):

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
