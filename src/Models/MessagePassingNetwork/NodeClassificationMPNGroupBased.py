import torch
import torch.nn as nn
from .layers import _make_mlp, MPLayer, TypeAwareMPNLayer
from .utils import sum_node_types


class LateFusionEdgeMLP(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        single_mlps = [size // 2 for size in config.EDGE_EMB.OUTPUT_SIZES[:-1]]
        self.pos_mlp = _make_mlp(2, single_mlps, bn=config.EDGE_EMB.BN,
                                 end_with_relu=config.EDGE_EMB.END_WITH_RELU)
        self.edge_mlp = _make_mlp(17, single_mlps, bn=config.EDGE_EMB.BN,
                                  end_with_relu=config.EDGE_EMB.END_WITH_RELU)
        self.out = nn.Linear(single_mlps[-1]*2, config.EDGE_EMB.OUTPUT_SIZES[-1])

    def forward(self, edge_attr):
        pos = edge_attr[:, :2]
        edge = edge_attr[:, 2:]
        return self.out(nn.functional.relu(torch.cat([self.pos_mlp(pos), self.edge_mlp(edge)], dim=1), inplace=True))

class NodeClassificationMPNGroupBased(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_skip_connections = config.SKIP
        self.node_summary = config.NODE_TYPE_SUMMARY
        if config.AGGR_TYPE == "agnostic":
            self.mpn_node_cls = MPLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                      aggr=config.AGGR, skip=config.SKIP, use_node_update_mlp=config.USE_NODE_UPDATE_MLP,
                                        edge_mlp=config.EDGE_MLP)
        elif config.AGGR_TYPE == "per_type":
            num_types = None
            if self.node_summary == "per_body_part":
                num_types = 6
            elif self.node_summary == "not":
                num_types = 17
            elif self.node_summary == "left_right":
                num_types = 9
            self.mpn_node_cls = TypeAwareMPNLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                        aggr=config.AGGR, skip=config.SKIP,
                                        edge_mlp=config.EDGE_MLP, num_types=num_types)

        if config.LATE_FUSION_POS:
            self.edge_embedding = LateFusionEdgeMLP(config)
        else:
            self.edge_embedding = _make_mlp(config.EDGE_INPUT_DIM, config.EDGE_EMB.OUTPUT_SIZES, bn=config.EDGE_EMB.BN,
                                            end_with_relu=config.EDGE_EMB.END_WITH_RELU)
        self.node_embedding = _make_mlp(config.NODE_INPUT_DIM, config.NODE_EMB.OUTPUT_SIZES, bn=config.NODE_EMB.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)

        self.edge_classification = _make_mlp(config.EDGE_FEATURE_DIM, config.EDGE_CLASS.OUTPUT_SIZES, bn=config.BN)
        self.node_classification = _make_mlp(config.NODE_FEATURE_DIM, config.NODE_CLASS.OUTPUT_SIZES, bn=config.BN)
        self.classification = _make_mlp(config.NODE_FEATURE_DIM, config.CLASS.OUTPUT_SIZES, bn=config.BN)

        self.edge_steps = config.STEPS
        self.node_steps = config.NODE_STEPS
        self.aux_loss_steps = config.AUX_LOSS_STEPS
        self.node_summary = config.NODE_TYPE_SUMMARY

    def forward(self, x, edge_attr, edge_index, **kwargs):

        node_types = sum_node_types(self.node_summary, kwargs["node_types"])
        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)

        node_features_initial = node_features
        edge_features_initial = edge_features

        preds_edge = []
        preds_node = []
        preds_class = []
        mask, mask_2 = get_sub_graphs(node_types, edge_index)
        for i in range(self.edge_steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)

            out_edge_features = torch.zeros_like(edge_features_initial)
            edge_index_1 = edge_index[:, mask]
            node_features, out_edge_features[mask] = self.mpn_node_cls(node_features, edge_features[mask], edge_index_1,
                                                             node_types=node_types)
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
            edge_index_2 = edge_index[:, mask_2]
            node_features, out_edge_features[mask_2] = self.mpn_node_cls(node_features, edge_features[mask_2], edge_index_2,
                                                                   node_types=node_types)
            edge_features = out_edge_features

        preds_edge.append(self.edge_classification(edge_features).squeeze())

        for i in range(self.node_steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)
            node_features, edge_features = self.mpn_node_cls(node_features, edge_features, edge_index)

        preds_node.append(self.node_classification(node_features).squeeze())
        preds_class.append(self.classification(node_features))


        return preds_edge, preds_node, preds_class


def get_sub_graphs(node_cat, edge_index):
    # 'nose','eye_l','eye_r','ear_l','ear_r', 'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
    # 'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r'
    mask = None
    for i in range(6):
        cat = node_cat == i
        if mask is None:
            mask = cat[edge_index[0]] & cat[edge_index[1]]
        else:
            mask = mask | (cat[edge_index[0]] & cat[edge_index[1]])
    return mask, torch.logical_not(mask)


