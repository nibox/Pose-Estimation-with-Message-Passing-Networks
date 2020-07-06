import torch
from torch.utils.data import DataLoader
from data import CocoKeypoints_hg, CocoKeypoints_hr, HeatmapGenerator
from Utils.transforms import transforms_hr_eval, transforms_hr_train, transforms_hg_eval
from Utils.loss import MPNLossFactory, MultiLossFactory, ClassMPNLossFactory
from Utils.Utils import to_device, calc_metrics, subgraph_mask
from torch_geometric.utils import subgraph
from config import get_config, update_config
import numpy as np
import pickle
from Models import get_pose_model
from torch.utils.tensorboard import SummaryWriter
import os


class Logger(object):

    def __init__(self, config):
        self.writer = SummaryWriter(config.LOG_DIR)


    def log_vars(self, name, iter, **kwargs):

        for key in kwargs.keys():
            if isinstance(kwargs[key], list):
                if kwargs[key]:
                    self.writer.add_scalar(f"{name}_{key}", np.mean(kwargs[key]), iter)
                else:
                    continue
            else:
                self.writer.add_scalar(f"{name}_{key}", kwargs[key], iter)

    def log_loss(self, loss, name, iter):
        self.writer.add_scalar(f"{name}", loss, iter)


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
            imgs, heatmaps, masks, keypoints, factors = batch
            imgs = imgs.to(kwargs["device"])
            masks = to_device(kwargs["device"], masks)
            keypoints = keypoints.to(kwargs["device"])
            factors = factors.to(kwargs["device"])
            heatmaps = to_device(kwargs["device"], heatmaps)

            _, preds, preds_nodes, joint_det, _, edge_index, edge_labels, node_labels, label_mask, bb_output = model(imgs, keypoints, masks[-1], factors)

            label_mask = label_mask if kwargs["use_label_mask"] else None

            loss = loss_func(bb_output, preds, heatmaps, edge_labels, masks, label_mask)
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            # split batch
            imgs, _, masks, keypoints, factors = batch
            imgs = imgs.to(kwargs["device"])
            masks = masks[-1].to(kwargs["device"])
            keypoints = keypoints.to(kwargs["device"])
            factors = factors.to(kwargs["device"])
            _, preds, preds_nodes, joint_det, _, edge_index, edge_labels, node_labels, label_mask, bb_output = model(imgs, keypoints, masks, factors)

            label_mask = label_mask if kwargs["use_label_mask"] else None

            if isinstance(loss_func, MPNLossFactory):
                loss = loss_func(preds, edge_labels, label_mask=label_mask)
            elif isinstance(loss_func, ClassMPNLossFactory):
                # adapt labels and mask to reduced graph
                """
                true_positive_idx = preds_nodes[-1] > 0.0
                true_positive_idx[node_labels == 1.0] = True
                _, attr = subgraph(true_positive_idx, edge_index, torch.stack([edge_labels, label_mask], dim=1))
                edge_labels = attr[:, 0]
                label_mask = attr[:, 1]
                loss = loss_func(preds, preds_nodes, edge_labels, node_labels, label_mask)
                # """
                # """
                true_positive_idx = preds_nodes[-1].detach() > 0.0
                true_positive_idx[node_labels == 1.0] = True
                mask = subgraph_mask(true_positive_idx, edge_index)
                label_mask[mask == 0] = 0.0
                loss = loss_func(preds, preds_nodes, edge_labels, node_labels, label_mask)
                label_mask[mask == 0] = 1.0
                # """
            else:
                raise NotImplementedError
            loss.backward()
            optimizer.step()
        loss = loss.item()
        preds_edges = preds[-1].detach()
        preds_nodes = None if preds_nodes is None else preds_nodes[-1].detach()
        edge_labels = edge_labels.detach()
        node_labels = None if preds_nodes is None else node_labels.detach()
        edge_label_mask = label_mask.detach()

        return loss, preds_nodes, preds_edges, node_labels, edge_labels, edge_label_mask

    return func


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    ##########################################################
    config_name = "model_35_2"
    config = get_config()
    config = update_config(config, f"../experiments/train/{config_name}.yaml")

    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger = Logger(config)

    ##########################################################
    if not config.MODEL.GC.USE_GT:
        assert config.TRAIN.USE_LABEL_MASK  # this ensures that images with no "persons"/clusters do not contribute to the loss
    print("Load model")
    model = get_pose_model(config, device)
    if config.TRAIN.END_TO_END:
        model.freeze_backbone(partial=True)
        optimizer = torch.optim.Adam([{"params": model.mpn.parameters(), "lr": config.TRAIN.LR},
                                      {"params": model.backbone.parameters(), "lr": config.TRAIN.KP_LR}])
        loss_func = MultiLossFactory(config)
    else:
        model.freeze_backbone(partial=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)
        if config.MODEL.LOSS.NAME == "edge_loss":
            loss_func = MPNLossFactory(config)
        elif config.MODEL.LOSS.NAME == "node_edge_loss":
            loss_func = ClassMPNLossFactory(config)
        else:
            raise NotImplementedError
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

            loss, preds_nodes, preds_edges, node_labels, edge_labels, edge_label_mask = update_model(batch)

            if preds_nodes is not None:
                result_nodes = preds_nodes.sigmoid().squeeze()
                result_nodes = torch.where(result_nodes < 0.5, torch.zeros_like(result_nodes), torch.ones_like(result_nodes))
            else:
                result_nodes = None
            result_edges = preds_edges.sigmoid().squeeze()
            result_edges = torch.where(result_edges < 0.5, torch.zeros_like(result_edges), torch.ones_like(result_edges))

            node_metrics = calc_metrics(result_nodes, node_labels)
            edge_metrics = calc_metrics(result_edges, edge_labels, edge_label_mask)

            logger.log_vars("Metric/train", iter, **edge_metrics)
            logger.log_vars("Metric/Node/train", iter, **(node_metrics or {}))
            logger.log_loss(loss, "Loss/train", iter)

            if preds_nodes is not None:
                print(f"Iter: {iter}, loss:{loss:6f}, "
                      f"Edge_Prec : {edge_metrics['prec']:5f} "
                      f"Edge_Rec: {edge_metrics['rec']:5f} "
                      f"Edge_Acc: {edge_metrics['acc']:5f} "
                      f"Node_Prec: {node_metrics['prec']:5f} "
                      f"Node_Rec: {node_metrics['rec']:5f} "
                      )
            else:
                print(f"Iter: {iter}, loss:{loss:6f}, "
                      f"Edge_Prec : {edge_metrics['prec']:5f} "
                      f"Edge_Rec: {edge_metrics['rec']:5f} "
                      f"Edge_Acc: {edge_metrics['acc']:5f} "
                      )
            """
            result = result[label_mask==1.0]
            edge_labels = edge_labels[label_mask==1.0]


            # metrics
            # edge_labels[edge_labels<0.99] = 0.0  # some edge labels might be
            prec = precision(result, edge_labels, 2)[1]
            rec = recall(result, edge_labels, 2)[1]
            acc = accuracy(result, edge_labels)
            f1 = f1_score(result, edge_labels, 2)[1]

            node_rec = recall(result_nodes, node_labels, 2)[1]
            node_prec = precision(result_nodes, node_labels, 2)[1]
            node_acc = accuracy(result_nodes, node_labels)
            print(f"Iter: {iter}, loss:{loss:6f}, "
                  f"Precision : {prec:5f} "
                  f"Recall: {rec:5f} "
                  f"Accuracy: {acc:5f} "
                  f"F1 score: {f1:5f}")

            writer.add_scalar("Loss/train", loss, iter)
            writer.add_scalar("Metric/train_f1:", f1, iter)
            writer.add_scalar("Metric/train_prec:", prec, iter)
            writer.add_scalar("Metric/train_rec:", rec, iter)
            writer.add_scalar("Metrics/Node/train_prec", node_prec, iter)
            writer.add_scalar("Metrics/Node/train_rec", node_rec, iter)
            writer.add_scalar("Metrics/Node/train_acc", node_acc, iter)
            """
        model.eval()

        print("#### BEGIN VALIDATION ####")

        valid_dict = {"node": {"acc": [], "prec": [], "rec": [], "f1": []},
                      "edge": {"acc": [], "prec": [], "rec": [], "f1": []}}
        valid_loss = []
        with torch.no_grad():
            for batch in valid_loader:
                # split batch
                imgs, heatmaps, masks, keypoints, factors = batch
                imgs = imgs.to(device)
                masks = to_device(device, masks)
                keypoints = keypoints.to(device)
                factors = factors.to(device)
                heatmaps = to_device(device, heatmaps)

                _, preds, preds_nodes, joint_det, _, edge_index, edge_labels, node_labels, label_mask, bb_output = model(imgs, keypoints, masks[-1], factors)

                label_mask = label_mask if config.TRAIN.USE_LABEL_MASK else None
                if config.TRAIN.END_TO_END:
                    loss = loss_func(bb_output, preds, heatmaps, edge_labels, masks, label_mask)
                else:
                    if isinstance(loss_func, MPNLossFactory):
                        loss = loss_func(preds, edge_labels, label_mask=label_mask)
                    elif isinstance(loss_func, ClassMPNLossFactory):
                        # adapt labels and mask to reduced graph
                        true_positive_idx = preds_nodes[-1] > 0.0
                        true_positive_idx[node_labels == 1.0] = True
                        _, attr = subgraph(true_positive_idx, edge_index, torch.stack([edge_labels, label_mask], dim=1))
                        edge_labels = attr[:, 0]
                        label_mask = attr[:, 1]
                        loss = loss_func(preds, preds_nodes, edge_labels, node_labels, label_mask)
                    else:
                        raise NotImplementedError

                result_edges = preds[-1].sigmoid().squeeze()
                result_edges = torch.where(result_edges < 0.5, torch.zeros_like(result_edges), torch.ones_like(result_edges))

                if preds_nodes is not None:
                    result_nodes = preds_nodes[-1].sigmoid().squeeze()
                    result_nodes = torch.where(result_nodes < 0.5, torch.zeros_like(result_nodes), torch.ones_like(result_nodes))
                else:
                    result_nodes = None
                    node_labels = None

                # remove masked connections from score calculation
                """
                if len(result) == 0:
                    continue
                """
                print(result_nodes)
                node_metrics = calc_metrics(result_nodes, node_labels)
                edge_metrics = calc_metrics(result_edges, edge_labels, label_mask)

                valid_loss.append(loss.item())
                if edge_metrics is not None:
                    for key in edge_metrics.keys():
                        valid_dict["edge"][key].append((edge_metrics[key]))

                if node_metrics is not None:
                    for key in node_metrics.keys():
                        valid_dict["edge"][key].append((node_metrics[key]))

        print(f"Epoch: {epoch}, loss:{np.mean(valid_loss):6f}, "
              f"Accuracy: {np.mean(valid_dict['edge']['acc']):5f}, "
              f"Precision: {np.mean(valid_dict['edge']['prec']):5f}, "
              f"Recall: {np.mean(valid_dict['edge']['rec']):5f}")
        scheduler.step()

        logger.log_loss(np.mean(valid_loss), "Loss/valid", epoch)
        logger.log_vars("Metrics/valid", epoch, **valid_dict['edge'])
        logger.log_vars("Metrics/Node/valid", epoch, **valid_dict['node'])

        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": scheduler.state_dict()
                    }, config.MODEL.PRETRAINED)


if __name__ == "__main__":
    main()
