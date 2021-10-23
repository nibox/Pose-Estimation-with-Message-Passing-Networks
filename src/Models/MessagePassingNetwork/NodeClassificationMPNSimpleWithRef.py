import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from Utils.Utils import subgraph_mask
from .layers import _make_mlp, MPLayer

def _make_classification_mlp(input_dim, hidden_dims, bn):

    layers = []
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    if len(hidden_dims) != 1:
        layers.append(nn.ReLU(inplace=True))
    if bn and len(hidden_dims) != 1:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
    for i in range(1, len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        if i != len(hidden_dims) - 1:
            layers.append(nn.ReLU(inplace=True))
            if bn:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))

    classification_layer = nn.Linear(hidden_dims[-2], hidden_dims[-1], bias=True)

    return nn.Sequential(*layers), classification_layer

class NodeClassificationMPNSimpleWithRef(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_skip_connections = config.SKIP
        self.mpn_edge_cls = MPLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                    aggr=config.AGGR, skip=config.SKIP, use_node_update_mlp=config.USE_NODE_UPDATE_MLP)
        self.mpn_node_cls = MPLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM + 32, config.EDGE_FEATURE_HIDDEN,
                                  aggr=config.AGGR, skip=config.SKIP, use_node_update_mlp=config.USE_NODE_UPDATE_MLP)

        self.edge_embedding = _make_mlp(config.EDGE_INPUT_DIM, config.EDGE_EMB.OUTPUT_SIZES, bn=config.EDGE_EMB.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)
        self.node_embedding = _make_mlp(config.NODE_INPUT_DIM, config.NODE_EMB.OUTPUT_SIZES, bn=config.NODE_EMB.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)

        self.edge_out, self.edge_classification = _make_classification_mlp(config.EDGE_FEATURE_DIM, config.EDGE_CLASS.OUTPUT_SIZES, bn=config.BN)
        self.node_classification = _make_mlp(config.NODE_FEATURE_DIM, config.NODE_CLASS.OUTPUT_SIZES, bn=config.BN)
        self.classification = _make_mlp(config.NODE_FEATURE_DIM, config.CLASS.OUTPUT_SIZES, bn=config.BN)

        self.edge_steps = config.STEPS
        self.node_steps = config.NODE_STEPS
        self.aux_loss_steps = config.AUX_LOSS_STEPS

    def forward(self, x, edge_attr, edge_index, **kwargs):

        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)

        node_features_initial = node_features
        edge_features_initial = edge_features

        preds_edge = []
        preds_node = []
        preds_class = []
        for i in range(self.edge_steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)
            node_features, edge_features = self.mpn_edge_cls(node_features, edge_features, edge_index)

        edge_features_out = self.edge_out(edge_features)
        preds_edge.append(self.edge_classification(edge_features_out).squeeze())

        node_features_initial = node_features
        edge_features = torch.cat([edge_features_out, edge_features], dim=1)
        edge_features_initial = edge_features

        for i in range(self.node_steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)
            node_features, edge_features = self.mpn_node_cls(node_features, edge_features, edge_index)

        preds_node.append(self.node_classification(node_features).squeeze())
        preds_class.append(self.classification(node_features))


        return preds_edge, preds_node, preds_class
