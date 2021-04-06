import torch
import numpy as np
import pickle
from config import get_config, update_config
from torch_geometric.utils import recall, accuracy, precision, f1_score, subgraph
from Utils.transforms import transforms_hr_train
from Utils.loss import *
from Utils.Utils import Logger, subgraph_mask, calc_metrics
from Models import get_cached_model
from data import CocoKeypoints_hr, HeatmapGenerator, JointsGenerator


def load_data(config, batch_size, device):
    tv_split_name = "tmp/coco_17_mini_split.p"
    train_ids, _ = pickle.load(open(tv_split_name, "rb"))
    train_ids = np.random.choice(train_ids, batch_size, replace=False)
    transforms, _ = transforms_hr_train(config)
    heatmap_generator = [HeatmapGenerator(160, 17), HeatmapGenerator(320, 17)]
    joints_generator = [JointsGenerator(30, 17, 320, True),
                        JointsGenerator(30, 17, 320, True)
                        ]
    train = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="train", img_ids=train_ids, year=17,
                             transforms=transforms, heatmap_generator=heatmap_generator, mask_crowds=True,
                             joint_generator=joints_generator)

    imgs = []
    masks = []
    keypoints = []
    factors = []
    for i in range(len(train)):
        img, target_scoremap, mask, keypoint, factor, _ = train[i]
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
        optimizer.zero_grad()
        # split batch
        imgs, masks, keypoints, factors = batch
        imgs = imgs.to(kwargs["device"])
        masks = masks.to(kwargs["device"])
        keypoints = keypoints.to(kwargs["device"])
        factors = factors.to(kwargs["device"])

        _, output = model(imgs, keypoints, masks, factors)

        output["masks"]["heatmap"] = masks
        output["labels"]["heatmap"] = []
        output["labels"]["tag"] = []
        output["labels"]["keypoints"] = []
        output["preds"]["heatmap"] = None

        if isinstance(loss_func, (MultiLossFactory, MPNLossFactory, TagMultiLossFactory)):
            loss, _ = loss_func(output["preds"], output["labels"], output["masks"])
        elif isinstance(loss_func, ClassMultiLossFactory):
            # adapt labels and mask to reduced graph
            loss_masks = []
            loss_edge_labels = []
            for i in range(len(output["preds"]["node"])):
                true_positive_idx = output["preds"]["node"][i].sigmoid() > 1.0
                true_positive_idx[output["labels"]["node"]== 1.0] = True
                mask = subgraph_mask(true_positive_idx, output["graph"]["edge_index"])
                loss_edge_labels.append(output["labels"]["edge"])
                loss_masks.append(output["masks"]["edge"] * mask.float())

            output["labels"]["edge"] = loss_edge_labels
            output["masks"]["edge"] = loss_masks
            loss, logging = loss_func(output["preds"], output["labels"], output["masks"], output["graph"])

            #
            default_pred = torch.zeros(output["graph"]["edge_index"].shape[1],
                                       dtype=torch.float, device=output["graph"]["edge_index"].device) - 1.0
            default_pred[mask] = output["preds"]["edge"][-1][mask].detach()
        elif isinstance(loss_func, BackgroundClassMultiLossFactory):
            # adapt labels and mask to reduced graph
            loss_masks = []
            loss_edge_labels = []
            for i in range(len(output["preds"]["node"])):
                true_positive_idx = output["labels"]["node"] == 1.0
                mask = subgraph_mask(true_positive_idx, output["graph"]["edge_index"])
                loss_edge_labels.append(output["labels"]["edge"])
                loss_masks.append(output["masks"]["edge"] * mask.float())

            output["labels"]["edge"] = loss_edge_labels
            output["masks"]["edge"] = loss_masks
            loss, logging = loss_func(output["preds"], output["labels"], output["masks"])

            #
            default_pred = torch.zeros(output["graph"]["edge_index"].shape[1],
                                       dtype=torch.float, device=output["graph"]["edge_index"].device) - 1.0
            default_pred[mask] = output["preds"]["edge"][-1][mask].detach()

        else:
            raise NotImplementedError

        loss.backward()
        optimizer.step()

        if isinstance(loss_func, ClassMPNLossFactory):
            output["preds"]["edge"] = [default_pred]

        loss = loss.item()
        preds = output["preds"]
        labels = output["labels"]
        masks = output["masks"]

        return loss, preds, labels, masks, logging

    return func


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ##########################################################
    config_dir = "hybrid_class_agnostic_end2end"
    config_name = "model_58_4_4"
    # config_dir = "feature_importance"
    # config_name = "model_nothing"
    config = get_config()
    config = update_config(config, f"../experiments/{config_dir}/{config_name}.yaml")


    ##########################################################
    if not config.MODEL.GC.USE_GT:
        assert config.TRAIN.USE_LABEL_MASK  # this ensures that images with no "persons"/clusters do not contribute to the loss
    print("Load model")
    model = get_cached_model(config, device)

    model.freeze_backbone(mode="complete")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)
    loss_func = ClassMultiLossFactory(config)
    """
    if config.MODEL.LOSS.NAME == "edge_loss":
        loss_func = MPNLossFactory(config)
    elif config.MODEL.LOSS.NAME == "node_edge_loss":
        loss_func = ClassMPNLossFactory(config)
    elif config.MODEL.LOSS.NAME == "node_with_background_edge_loss":
        loss_func = BackgroundClassMultiLossFactory(config)
    elif config.MODEL.LOSS.NAME == "tag_loss":
        loss_func = TagMultiLossFactory(config)
    else:
        raise NotImplementedError
    # """
    model.to(device)

    update_model = make_train_func(model, optimizer, loss_func, use_batch_index=config.TRAIN.USE_BATCH_INDEX,
                                   use_label_mask=config.TRAIN.USE_LABEL_MASK, device=device,
                                   end_to_end=config.TRAIN.END_TO_END, batch_size=config.TRAIN.BATCH_SIZE,
                                   loss_reduction=config.TRAIN.LOSS_REDUCTION, config=config)
    print("#####Begin Training#####")
    batch = load_data(config, config.TRAIN.BATCH_SIZE, device)

    model.train()
    if config.TRAIN.FREEZE_BN:
            model.stop_backbone_bn()
    for iter in range(10000):

        loss, preds, labels, masks, logging = update_model(batch)

        preds_nodes, preds_edges, preds_classes = preds["node"][-1], preds["edge"][-1], preds["class"]
        node_labels, edge_labels, class_labels = labels["node"], labels["edge"][-1], labels["class"]
        node_mask, edge_mask, class_mask = masks["node"], masks["edge"][-1], masks["class"]

        if preds_nodes is not None:
            result_nodes = preds_nodes.sigmoid().squeeze()
            result_nodes = torch.where(result_nodes < 0.5, torch.zeros_like(result_nodes),
                                       torch.ones_like(result_nodes))
        else:
            if preds_classes is not None:
                result_nodes = preds_classes[-1].argmax(dim=1).squeeze() != 18
            else:
                result_nodes = None
        result_classes = preds_classes[-1].argmax(dim=1).squeeze() if preds_classes is not None else None

        if preds_edges is not None:
            result_edges = preds_edges.sigmoid().squeeze()
            result_edges = torch.where(result_edges < 0.5, torch.zeros_like(result_edges), torch.ones_like(result_edges))
        else:
                result_edges = None

        node_metrics = calc_metrics(result_nodes, node_labels, node_mask)
        class_metrics = calc_metrics(result_classes, class_labels, class_mask, 17)
        edge_metrics = calc_metrics(result_edges, edge_labels, edge_mask)

        s = f"Iter: {iter}, loss:{loss:6f}, "
        if edge_metrics is not None:
            s += f"Edge_Prec : {edge_metrics['prec']:5f} " \
                 f"Edge_Rec: {edge_metrics['rec']:5f} " \
                 f"Edge_Acc: {edge_metrics['acc']:5f} "
        if node_metrics is not None:
                s += f"Node_Prec: {node_metrics['prec']:5f} " \
                     f"Node_Rec: {node_metrics['rec']:5f} "
        if preds_classes is not None:
            s += f"Class Acc: {class_metrics['acc']:5f} "
        if "reg" in logging.keys():
            s += f"Reg: {logging['reg']}"
        print(s)


if __name__ == "__main__":
    main()
