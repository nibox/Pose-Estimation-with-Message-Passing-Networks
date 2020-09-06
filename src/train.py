import torch
from torch.utils.data import DataLoader
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator, ScaleAwareHeatmapGenerator
from Utils.transforms import transforms_hr_train, transforms_hg_eval
from Utils.loss import MPNLossFactory, MultiLossFactory, ClassMPNLossFactory, ClassMultiLossFactory
from Utils.Utils import to_device, calc_metrics, subgraph_mask, Logger
from config import get_config, update_config
import numpy as np
import pickle
from Models import get_pose_model
import os
import sys


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
        heatmap_generator = [HeatmapGenerator(128, 17, config.DATASET.SIGMA), HeatmapGenerator(256, 17, config.DATASET.SIGMA)]
        transforms, _ = transforms_hr_train(config)
        train = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="train", img_ids=train_ids, year=17,
                                    transforms=transforms, heatmap_generator=heatmap_generator)
        valid = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids, year=17,
                                 transforms=transforms, heatmap_generator=heatmap_generator)
        return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8), \
               DataLoader(valid, batch_size=1, num_workers=8)
    elif config.TRAIN.SPLIT == "coco_17_full":
        train_ids, _ = pickle.load(open("tmp/coco_17_full_split.p", "rb"))  # mini_train_valid_split_4 old one
        _, valid_ids = pickle.load(open("tmp/coco_17_mini_split.p", "rb"))  # mini_train_valid_split_4 old one
        if config.DATASET.HEAT_GENERATOR == "default":
            heatmap_generator = [HeatmapGenerator(128, 17, config.DATASET.SIGMA),
                                 HeatmapGenerator(256, 17, config.DATASET.SIGMA)]
        elif config.DATASET.HEAT_GENERATOR == "scale_aware":
            heatmap_generator = [ScaleAwareHeatmapGenerator(128, 17, config.DATASET.SIGMA),
                                 ScaleAwareHeatmapGenerator(256, 17, config.DATASET.SIGMA)]
        transforms, _ = transforms_hr_train(config)
        train = CocoKeypoints_hr(config.DATASET.ROOT, mini=False, seed=0, mode="train", img_ids=train_ids, year=17,
                                 transforms=transforms, heatmap_generator=heatmap_generator)
        valid = CocoKeypoints_hr(config.DATASET.ROOT, mini=True, seed=0, mode="val", img_ids=valid_ids, year=17,
                                 transforms=transforms, heatmap_generator=heatmap_generator)
        return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8), \
               DataLoader(valid, batch_size=1, num_workers=8)

    else:
        raise NotImplementedError

def mask_node_connections(preds_nodes, edge_index, th, node_labels=None):
    true_positive_idx = preds_nodes > th
    if node_labels is not None:
        true_positive_idx[node_labels == 1.0] = True
    mask = subgraph_mask(true_positive_idx, edge_index)
    return mask


def make_train_func(model, optimizer, loss_func, **kwargs):
    def func(batch):
        optimizer.zero_grad()
        # split batch
        imgs, heatmaps, masks, keypoints, factors = batch
        imgs = imgs.to(kwargs["device"])
        masks = to_device(kwargs["device"], masks)
        keypoints = keypoints.to(kwargs["device"])
        factors = factors.to(kwargs["device"])
        heatmaps = to_device(kwargs["device"], heatmaps)

        # _, preds, preds_nodes, preds_classes, joint_det, _, edge_index, edge_labels, node_labels, class_labels, label_mask, label_mask_node, bb_output = model(imgs, keypoints, masks[-1], factors)
        _, output = model(imgs, keypoints, masks[-1], factors, heatmaps)
        output["masks"]["heatmap"] = masks
        output["labels"]["heatmap"] = heatmaps

        if isinstance(loss_func, (MultiLossFactory, MPNLossFactory)):
            loss, _ = loss_func(output["preds"], output["labels"], output["masks"])
        elif isinstance(loss_func, (ClassMultiLossFactory, ClassMPNLossFactory)):
            edge_masks = []
            edge_labels = []
            # first the graph reduction
            for i in range(len(output["preds"]["node"])):
                mask = mask_node_connections(output["preds"]["node"][i].sigmoid().detach(), output["graph"]["edge_index"],
                                             kwargs["config"].MODEL.MPN.NODE_THRESHOLD, output["labels"]["node"])
                edge_labels.append(output["labels"]["edge"])
                edge_masks.append(output["masks"]["edge"] * mask.float())
            output["labels"]["edge"] = edge_labels
            output["masks"]["edge"] = edge_masks

            loss, logging = loss_func(output["preds"], output["labels"], output["masks"])
        else:
            raise NotImplementedError

        loss.backward()
        optimizer.step()

        loss = loss.item()
        preds = output["preds"]
        labels = output["labels"]
        masks = output["masks"]

        return loss, preds, labels, masks, logging

    return func


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    ##########################################################
    config_name = sys.argv[1]
    config = get_config()
    config = update_config(config, f"../experiments/{config_name}.yaml")

    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger = Logger(config)

    ##########################################################
    if not config.MODEL.GC.USE_GT:
        assert config.TRAIN.USE_LABEL_MASK  # this ensures that images with no "persons"/clusters do not contribute to the loss
    print("Load model")
    if config.MODEL.WITH_LOCATION_REFINE:
        model = get_pose_with_ref_model(config, device)
    else:
        model = get_pose_model(config, device)
    if config.TRAIN.END_TO_END:
        assert config.TRAIN.KP_FREEZE_MODE != "complete"
        model.freeze_backbone(mode=config.TRAIN.KP_FREEZE_MODE)
        model_params = list(model.mpn.parameters()) + list(model.feature_gather.parameters())
        if config.TRAIN.SPLIT_OPTIMIZER and not config.MODEL.WITH_LOCATION_REFINE:
            optimizer = torch.optim.Adam([{"params": model_params, "lr": config.TRAIN.LR, "weight_decay": config.TRAIN.W_DECAY},
                                          {"params": model.backbone.parameters(), "lr": config.TRAIN.KP_LR, "weight_decay": config.TRAIN.W_DECAY}])
        elif config.TRAIN.SPLIT_OPTIMIZER and config.MODEL.WITH_LOCATION_REFINE:
            reg_param = model.refinement_network.parameters()
            optimizer = torch.optim.Adam(
                [{"params": model_params, "lr": config.TRAIN.LR, "weight_decay": config.TRAIN.W_DECAY},
                 {"params": model.backbone.parameters(), "lr": config.TRAIN.KP_LR,
                  "weight_decay": config.TRAIN.W_DECAY},
                 {"params": reg_param, "lr": config.TRAIN.REG_LR, "weight_decay": config.TRAIN.W_DECAY}])
        else:
            raise NotImplementedError

        if config.MODEL.LOSS.NAME == "edge_loss":
            loss_func = MultiLossFactory(config)
        elif config.MODEL.LOSS.NAME == "node_edge_loss":
            loss_func = ClassMultiLossFactory(config)
        else:
            raise NotImplementedError
    else:
        model.freeze_backbone(mode="complete")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.W_DECAY)
        if config.MODEL.LOSS.NAME == "edge_loss":
            loss_func = MPNLossFactory(config)
        elif config.MODEL.LOSS.NAME == "node_edge_loss":
            loss_func = ClassMPNLossFactory(config)
        else:
            raise NotImplementedError
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)
    model.to(device)

    if config.TRAIN.CONTINUE != "" and not config.TRAIN.FINETUNE:
        state_dict = torch.load(config.TRAIN.CONTINUE)
        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict["lr_scheduler_state_dict"])
    elif config.TRAIN.CONTINUE != "" and config.TRAIN.FINETUNE:
        state_dict = torch.load(config.TRAIN.CONTINUE)
        model.load_state_dict(state_dict["model_state_dict"])

    print("Load dataset")
    train_loader, valid_loader = create_train_validation_split(config)

    update_model = make_train_func(model, optimizer, loss_func, use_batch_index=config.TRAIN.USE_BATCH_INDEX,
                                   use_label_mask=config.TRAIN.USE_LABEL_MASK, device=device,
                                   end_to_end=config.TRAIN.END_TO_END, batch_size=config.TRAIN.BATCH_SIZE,
                                   loss_reduction=config.TRAIN.LOSS_REDUCTION, config=config)

    print("#####Begin Training#####")
    epoch_len = len(train_loader)
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.END_EPOCH):
        model.train()
        if config.TRAIN.FREEZE_BN:
            model.stop_backbone_bn()
        for i, batch in enumerate(train_loader):
            iter = i + (epoch_len * epoch)

            loss, preds, labels, masks, logging = update_model(batch)
            preds_nodes, preds_edges, preds_classes = preds["node"][-1], preds["edge"][-1], preds["class"]
            node_labels, edge_labels, class_labels = labels["node"], labels["edge"][-1], labels["class"]
            node_mask, edge_mask, class_mask = masks["node"], masks["edge"][-1], masks["class"]

            if preds_nodes is not None:
                result_nodes = preds_nodes.sigmoid().squeeze()
                result_nodes = torch.where(result_nodes < 0.5, torch.zeros_like(result_nodes), torch.ones_like(result_nodes))
            else:
                result_nodes = None
            result_classes = preds_classes[-1].argmax(dim=1).squeeze() if preds_classes is not None else None
            result_edges = preds_edges.sigmoid().squeeze()
            result_edges = torch.where(result_edges < 0.5, torch.zeros_like(result_edges), torch.ones_like(result_edges))

            node_metrics = calc_metrics(result_nodes, node_labels, node_mask)
            edge_metrics = calc_metrics(result_edges, edge_labels, edge_mask)
            class_metrics = calc_metrics(result_classes, class_labels, node_labels, 17)

            logger.log_vars("Metric/train", iter, **(edge_metrics or {}))
            logger.log_vars("Metric/Node/train", iter, **(node_metrics or {}))

            if class_metrics is not None:
                logger.log_vars("Metric/Node/train_class", iter, acc=class_metrics["acc"])
            if "reg" in logging.keys():
                logger.log_loss(logging["reg"], "Loss/train_reg", iter)
            logger.log_loss(loss, "Loss/train", iter)


            s = f"Iter: {iter}, loss:{loss:6f}, "
            if edge_metrics is not None:
                s += f"Edge_Prec : {edge_metrics['prec']:5f} " \
                     f"Edge_Rec: {edge_metrics['rec']:5f} " \
                     f"Edge_Acc: {edge_metrics['acc']:5f} "
            if node_metrics is not None:
                s += f"Node_Prec: {node_metrics['prec']:5f} " \
                     f"Node_Rec: {node_metrics['rec']:5f} "
            if class_metrics is not None:
                s += f"Class Acc: {class_metrics['acc']:5f} "
            if "reg" in logging.keys():
                s += f"Reg: {logging['reg']}"
            print(s)

        model.eval()

        print("#### BEGIN VALIDATION ####")

        valid_dict = {"node": {"acc": [], "prec": [], "rec": [], "f1": []},
                      "edge": {"acc": [], "prec": [], "rec": [], "f1": []},
                      "edge_masked": {"prec": [], "rec": []}}
        valid_loss = []
        valid_reg = []
        class_valid_acc = []
        valid_heatmap = []
        with torch.no_grad():
            for batch in valid_loader:
                # split batch
                imgs, heatmaps, masks, keypoints, factors = batch
                if keypoints.sum() == 0.0:
                    continue
                imgs = imgs.to(device)
                masks = to_device(device, masks)
                keypoints = keypoints.to(device)
                factors = factors.to(device)
                heatmaps = to_device(device, heatmaps)

                _, output = model(imgs, keypoints, masks[-1], factors, heatmaps)
                output["masks"]["heatmap"] = masks
                output["labels"]["heatmap"] = heatmaps

                preds, labels, masks = output["preds"], output["labels"], output["masks"]
                if isinstance(loss_func, (MultiLossFactory, MPNLossFactory)):
                    loss, logging = loss_func(preds, labels, masks)
                    loss_mask = masks["edge"]
                elif isinstance(loss_func, (ClassMultiLossFactory, ClassMPNLossFactory)):
                    loss_mask = masks["edge"].detach()
                    edge_masks = []
                    edge_labels = []
                    # first the graph reduction
                    for i in range(len(preds["node"])):
                        mask = mask_node_connections(preds["node"][i].sigmoid(), output["graph"]["edge_index"], 0.5,
                                                     None)
                        edge_labels.append(labels["edge"])
                        edge_masks.append(masks["edge"] * mask.float())
                    labels["edge"] = edge_labels
                    masks["edge"] = edge_masks

                    loss, logging = loss_func(preds, labels, masks)
                    #
                    default_pred = torch.zeros(output["graph"]["edge_index"].shape[1], dtype=torch.float,
                                               device=output["graph"]["edge_index"].device) - 1.0
                    if preds["edge"][-1] is not None:
                        default_pred[mask] = preds["edge"][-1][mask].detach()
                        preds["edge"][-1] = default_pred
                    else:
                        preds["edge"] = [default_pred]
                preds_nodes, preds_edges, preds_classes = preds["node"], preds["edge"], preds["class"]
                node_labels, edge_labels, class_labels = labels["node"], labels["edge"][-1], labels["class"]
                node_mask, edge_mask, class_mask = masks["node"], masks["edge"][-1], masks["class"]

                result_edges = preds_edges[-1].sigmoid().squeeze()
                result_edges = torch.where(result_edges < 0.5, torch.zeros_like(result_edges), torch.ones_like(result_edges))

                if preds_nodes is not None:
                    result_nodes = preds_nodes[-1].sigmoid().squeeze()
                    result_nodes = torch.where(result_nodes < 0.5, torch.zeros_like(result_nodes), torch.ones_like(result_nodes))
                else:
                    result_nodes = None
                    node_labels = None
                result_classes = preds_classes[-1].argmax(dim=1).squeeze() if preds_classes is not None else None

                node_metrics = calc_metrics(result_nodes, node_labels, node_mask)
                class_metrics = calc_metrics(result_classes, class_labels, node_labels, 17)
                edge_metrics_complete = calc_metrics(result_edges, edge_labels, loss_mask)
                edge_metrics_masked = calc_metrics(result_edges, edge_labels, edge_mask)

                valid_loss.append(loss.item())
                if "heatmap" in logging:
                    valid_heatmap.append(logging["heatmap"])
                if "reg" in logging:
                    valid_reg.append(logging["reg"])
                if edge_metrics_complete is not None:
                    for key in edge_metrics_complete.keys():
                        valid_dict["edge"][key].append(edge_metrics_complete[key])
                if edge_metrics_masked is not None:
                    for key in ["prec", "rec"]:
                        valid_dict["edge_masked"][key].append(edge_metrics_masked[key])
                if class_metrics is not None:
                    class_valid_acc.append(class_metrics["acc"])

                if node_metrics is not None:
                    for key in node_metrics.keys():
                        valid_dict["node"][key].append(node_metrics[key])

        print(f"Epoch: {epoch}, loss:{np.mean(valid_loss):6f}, "
              f"Accuracy: {np.mean(valid_dict['edge']['acc']):5f}, "
              f"Precision: {np.mean(valid_dict['edge']['prec']):5f}, "
              f"Recall: {np.mean(valid_dict['edge']['rec']):5f}")
        scheduler.step()

        logger.log_loss(np.mean(valid_loss), "Loss/valid", epoch)
        logger.log_loss(np.mean(valid_heatmap), "Loss/valid_heat", epoch)
        logger.log_loss(np.mean(valid_reg), "Loss/valid_reg", epoch)
        logger.log_vars("Metric/valid", epoch, **valid_dict['edge'])
        logger.log_vars("Metric/valid_masked", epoch, **valid_dict['edge_masked'])
        logger.log_vars("Metric/Node/valid", epoch, **valid_dict['node'])
        if len(class_valid_acc) != 0:
            logger.log_vars("Metric/Node/valid_class", epoch, acc=class_valid_acc)

        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": scheduler.state_dict()
                    }, config.MODEL.PRETRAINED)
        if epoch + 1 in config.TRAIN.LR_STEP:
            save_name = config.LOG_DIR + f"/pose_estimation_epoch_{epoch}.pth"
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": scheduler.state_dict()
                        }, save_name)


if __name__ == "__main__":
    main()
