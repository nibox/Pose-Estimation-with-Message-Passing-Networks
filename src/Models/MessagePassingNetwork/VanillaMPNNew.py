import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from .layers import _make_mlp

class VanillaMPLayer(MessagePassing):

    # todo with or without inital feature skip connection
    def __init__(self, node_feature_dim, edge_feature_dim, aggr, skip=False):
        super(VanillaMPLayer, self).__init__(aggr=aggr)
        # todo better architecture

        non_lin = nn.ReLU()
        node_factor = 2 if skip else 1
        edge_factor = 2 if skip else 1

        self.mlp_edge = nn.Sequential(nn.Linear(node_feature_dim * 2 * node_factor + edge_feature_dim * edge_factor, 64),
                                      non_lin,
                                      nn.Linear(64, edge_feature_dim),
                                      non_lin,
                                      )

        # self.mlp_edge = PerInvMLP(node_feature_dim, edge_feature_dim)
        self.mlp_node = nn.Sequential(nn.Linear(node_feature_dim * node_factor + edge_feature_dim, node_feature_dim),
                                      non_lin,
                                      )

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
        return aggr_out


class VanillaMPN(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.mpn = nn.ModuleList([VanillaMPLayer(node_feature_dim, edge_feature_dim) for i in range(steps)])
        self.use_skip_connections = config.SKIP
        self.mpn = VanillaMPLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, aggr=config.AGGR, skip=config.SKIP)

        non_linearity = nn.ReLU()
        self.edge_embedding = _make_mlp(config.EDGE_INPUT_DIM, config.EDGE_EMB.OUTPUT_SIZES, False)
        self.node_embedding = _make_mlp(config.NODE_INPUT_DIM, config.NODE_EMB.OUTPUT_SIZES, False)
        self.classification = _make_mlp(config.EDGE_FEATURE_DIM, config.CLASS.OUTPUT_SIZES, False)
        """
        self.edge_embedding = nn.Sequential(nn.Linear(config.EDGE_INPUT_DIM, 32),  # 2 + 17*17,
                                            non_linearity,
                                            nn.Linear(32, 64),
                                            non_linearity,
                                            nn.Linear(64, 64),
                                            non_linearity,
                                            nn.Linear(64, config.EDGE_FEATURE_DIM))
        self.node_embedding = nn.Sequential(nn.Linear(config.NODE_INPUT_DIM, 128),
                                            non_linearity,
                                            nn.Linear(128, 64),
                                            non_linearity,
                                            nn.Linear(64, config.NODE_FEATURE_DIM))

        self.classification = nn.Sequential(nn.Linear(config.EDGE_FEATURE_DIM, 64),
                                            non_linearity,
                                            nn.Linear(64, 32),
                                            non_linearity,
                                            nn.Linear(32, 1))
        """

        self.steps = config.STEPS

    def forward(self, x, edge_attr, edge_index, **kwargs):
        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)

        node_features_initial = node_features
        edge_features_initial = edge_features

        for i in range(self.steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)

            node_features, edge_features = self.mpn(node_features, edge_features, edge_index)

        return self.classification(edge_features)
