import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from .layers import _make_mlp


class PerInvMLP(nn.Module):

    def __init__(self, node_feature_dim, edge_feature_dim):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.node_mlp = nn.Sequential(nn.Linear(node_feature_dim, edge_feature_dim),
                                      nn.ReLU(),
                                      nn.Linear(edge_feature_dim, edge_feature_dim))
        self.edge_mlp = nn.Sequential(nn.Linear(edge_feature_dim, edge_feature_dim),
                                      nn.ReLU(),
                                      nn.Linear(edge_feature_dim, edge_feature_dim))
        self.fusion = nn.Sequential(nn.Linear(2 * edge_feature_dim, edge_feature_dim),
                                    nn.ReLU(),
                                    nn.Linear(edge_feature_dim, edge_feature_dim))

    def forward(self, x):
        node_1 = self.node_mlp(x[:, :self.node_feature_dim])
        node_2 = self.node_mlp(x[:, self.node_feature_dim: 2 * self.node_feature_dim])
        edge = self.edge_mlp(x[:, self.node_feature_dim * 2:])
        edge_attr = self.fusion(
            torch.cat([node_1 + node_2, edge], dim=1))  # todo ask if edge features are permutation inv
        return edge_attr


class VanillaMPLayer(MessagePassing):

    # todo with or without inital feature skip connection
    def __init__(self, node_feature_dim, edge_feature_dim, edge_feature_hidden, aggr, use_node_update_mlp, skip=False):
        super(VanillaMPLayer, self).__init__(aggr=aggr)
        # todo better architecture

        node_factor = 2 if skip else 1
        edge_factor = 2 if skip else 1

        self.mlp_edge = nn.Sequential(nn.Linear(node_feature_dim * 2 * node_factor + edge_feature_dim * edge_factor, edge_feature_hidden),
                                      nn.ReLU(),
                                      nn.Linear(edge_feature_hidden, edge_feature_dim),
                                      nn.ReLU(),
                                      )

        # self.mlp_edge = PerInvMLP(node_feature_dim, edge_feature_dim)
        self.mlp_node = nn.Sequential(nn.Linear(node_feature_dim * node_factor + edge_feature_dim, node_feature_dim),
                                      nn.ReLU(),
                                      )
        self.update_mlp = nn.Sequential(nn.Linear(node_feature_dim, node_feature_dim), nn.ReLU()) if use_node_update_mlp else None


    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)

        j, i = edge_index  # message is from j to i
        x_i, x_j = x[i], x[j]
        e_ij = edge_attr
        edge_attr = self.mlp_edge(torch.cat([x_i, x_j, e_ij], dim=1))  # todo ask if edge features are permutation inv

        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x, edge_attr=edge_attr), edge_attr

    def message(self, x_i, x_j, edge_attr):
        # edge_attr = self.mlp_edge(torch.cat([x_i, x_j, edge_attr], dim=1))
        out = self.mlp_node(torch.cat([x_i, edge_attr], dim=1))
        return out

    def update(self, aggr_out):
        if self.update_mlp is not None:
            aggr_out = self.update_mlp(aggr_out)
        return aggr_out


class VanillaMPN(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_skip_connections = config.SKIP
        self.mpn = VanillaMPLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                  aggr=config.AGGR, skip=config.SKIP, use_node_update_mlp=config.USE_NODE_UPDATE_MLP)

        self.edge_embedding = _make_mlp(config.EDGE_INPUT_DIM, config.EDGE_EMB.OUTPUT_SIZES, bn=config.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)
        self.node_embedding = _make_mlp(config.NODE_INPUT_DIM, config.NODE_EMB.OUTPUT_SIZES, bn=config.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)
        self.classification = _make_mlp(config.EDGE_FEATURE_DIM, config.CLASS.OUTPUT_SIZES, bn=config.BN)

        self.aux_loss_steps = config.AUX_LOSS_STEPS

        self.steps = config.STEPS

    def forward(self, x, edge_attr, edge_index):
        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)

        node_features_initial = node_features
        edge_features_initial = edge_features

        preds_edge = []
        preds_node = []
        for i in range(self.steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)

            node_features, edge_features = self.mpn(node_features, edge_features, edge_index)
            if i >= self.steps - self.aux_loss_steps - 1:
                preds_edge.append(self.classification(edge_features).squeeze())
                preds_node.append(node_features)

        return preds_edge, preds_node
