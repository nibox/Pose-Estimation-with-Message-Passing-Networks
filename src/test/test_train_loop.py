import torch
import numpy as np
import pickle
from config import get_config, update_config
from torch_geometric.utils import recall, accuracy, precision, f1_score, subgraph
from Utils.transforms import transforms_hr_train
from Utils.loss import MPNLossFactory, MultiLossFactory, ClassMPNLossFactory
from Utils.Utils import Logger, subgraph_mask, calc_metrics
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
    for i in range(len(train)):
        img, target_scoremap, mask, keypoint, factor = train[i]
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

        scoremaps, preds_edges, preds_nodes, preds_classes, joint_det, _, edge_index, edge_labels, node_labels, class_labels, label_mask, label_mask_node = model(imgs, keypoints, masks, factors)

        label_mask = label_mask if kwargs["use_label_mask"] else None

        if isinstance(loss_func, MultiLossFactory):
            loss, _ = loss_func(preds_edges, edge_labels, label_mask=label_mask)
        elif isinstance(loss_func, ClassMPNLossFactory):
            # adapt labels and mask to reduced graph
            loss_masks = []
            loss_edge_labels = []
            for i in range(len(preds_nodes)):
                true_positive_idx = preds_nodes[i].sigmoid() > kwargs["config"].MODEL.MPN.NODE_THRESHOLD
                true_positive_idx[node_labels == 1.0] = True
                mask = subgraph_mask(true_positive_idx, edge_index)
                loss_edge_labels.append(edge_labels)
                loss_masks.append(label_mask * mask.float())

            loss, _ = loss_func(preds_edges, preds_nodes, preds_classes, loss_edge_labels, node_labels, class_labels, loss_masks, label_mask_node)

            #
            default_pred = torch.zeros(edge_index.shape[1], dtype=torch.float, device=edge_index.device) - 1.0
            default_pred[mask] = preds_edges[-1][mask].detach()

        else:
            raise NotImplementedError

        loss.backward()
        optimizer.step()

        loss = loss.item()
        if isinstance(loss_func, ClassMPNLossFactory):
            preds_edges = default_pred
        else:
            preds_edges = preds_edges[-1].detach()
        preds_nodes = None if preds_nodes is None else preds_nodes[-1].detach()
        preds_classes = None if preds_classes is None else preds_classes[-1].detach()
        edge_labels = edge_labels.detach()
        node_labels = None if preds_nodes is None else node_labels.detach()
        class_labels = None if preds_classes is None else class_labels.detach()
        edge_label_mask = label_mask.detach()
        node_label_mask = label_mask_node.detach()

        return loss, preds_nodes, preds_edges, preds_classes, node_labels, edge_labels, class_labels, edge_label_mask, node_label_mask

    return func


def main():
    device = torch.device("cuda") if torch.cuda.is_available() and True else torch.device("cpu")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ##########################################################
    config_name = "model_56"
    config = get_config()
    config = update_config(config, f"../experiments/train/{config_name}.yaml")


    ##########################################################
    if not config.MODEL.GC.USE_GT:
        assert config.TRAIN.USE_LABEL_MASK  # this ensures that images with no "persons"/clusters do not contribute to the loss
    print("Load model")
    model = get_cached_model(config, device)
    if config.TRAIN.END_TO_END:
        raise NotImplementedError
    else:
        model.freeze_backbone(mode=config.TRAIN.KP_FREEZE_MODE)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)
        if config.MODEL.LOSS.NAME == "edge_loss":
            loss_func = MPNLossFactory(config)
        elif config.MODEL.LOSS.NAME == "node_edge_loss":
            loss_func = ClassMPNLossFactory(config)
        else:
            raise NotImplementedError
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

        loss, preds_nodes, preds_edges, preds_classes, node_labels, edge_labels, class_labels, edge_label_mask, node_label_mask = update_model(batch)

        if preds_nodes is not None:
            result_nodes = preds_nodes.sigmoid().squeeze()
            result_nodes = torch.where(result_nodes < 0.5, torch.zeros_like(result_nodes),
                                       torch.ones_like(result_nodes))
        else:
            result_nodes = None
        result_classes = preds_classes.argmax(dim=1).squeeze() if preds_classes is not None else None

        result_edges = preds_edges.sigmoid().squeeze()
        result_edges = torch.where(result_edges < 0.5, torch.zeros_like(result_edges), torch.ones_like(result_edges))

        node_metrics = calc_metrics(result_nodes, node_labels, node_label_mask)
        class_metrics = calc_metrics(result_classes, class_labels, node_labels, 17)
        edge_metrics = calc_metrics(result_edges, edge_labels, edge_label_mask)


        if preds_nodes is not None:

            s = f"Iter: {iter}, loss:{loss:6f}, " \
                  f"Edge_Prec : {edge_metrics['prec']:5f} " \
                  f"Edge_Rec: {edge_metrics['rec']:5f} " \
                  f"Edge_Acc: {edge_metrics['acc']:5f} " \
                  f"Node_Prec: {node_metrics['prec']:5f} " \
                  f"Node_Rec: {node_metrics['rec']:5f} "
            if preds_classes is not None:
                s += f"Class Acc: {class_metrics['acc']:5f}"
            print(s)
        else:
            print(f"Iter: {iter}, loss:{loss:6f}, "
                  f"Edge_Prec : {edge_metrics['prec']:5f} "
                  f"Edge_Rec: {edge_metrics['rec']:5f} "
                  f"Edge_Acc: {edge_metrics['acc']:5f} "
                  )

if __name__ == "__main__":
    main()
