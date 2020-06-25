import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from .layers import _make_mlp

default_config = {"steps": 4,
                  "node_input_dim": 128,
                  "edge_input_dim": 2 + 17 * 17,
                  "node_feature_dim": 128,
                  "edge_feature_dim": 128,
                  "node_hidden_dim": 128,
                  "edge_hidden_dim": 128,
                  "aggr": "add"}


class VanillaMPLayer2(MessagePassing):

    # todo with or without inital feature skip connection
    def __init__(self, n_in, n_out, e_in, e_out, aggr="add"):
        super(VanillaMPLayer2, self).__init__(aggr=aggr)
        # todo better architecture

        non_lin = nn.ReLU()
        self.mlp_edge = nn.Sequential(nn.Linear(n_in * 2 + e_in, e_out),
                                      non_lin,
                                      nn.BatchNorm1d(e_out),
                                      )

        self.mlp_node = nn.Sequential(nn.Linear(n_in + e_out, n_out),
                                      non_lin,
                                      nn.BatchNorm1d(n_out)
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


class VanillaMPN2(torch.nn.Module):

    def __init__(self, config, node_hidden_dim=128, edge_hidden_dim=128):
        super().__init__()
        self.mpn = nn.ModuleList([VanillaMPLayer2(config.NODE_FEATURE_DIM, config.NODE_FEATURE_DIM,
                                                  config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_DIM, aggr=config.AGGR)
                                  for _ in range(config.STEPS)])
        self.edge_embedding = _make_mlp(config.EDGE_INPUT_DIM, config.EDGE_EMB.OUTPUT_SIZES, bn=config.BN)
        self.node_embedding = _make_mlp(config.NODE_INPUT_DIM, config.NODE_EMB.OUTPUT_SIZES, bn=config.BN)
        self.classification = _make_mlp(config.EDGE_FEATURE_DIM, config.CLASS.OUTPUT_SIZES, bn=config.BN, init_trick=True)

        """
        self.mpn = nn.ModuleList([VanillaMPLayer2(config.NODE_FEATURE_DIM, config.NODE_FEATURE_DIM,
                                                  config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_DIM, aggr=config.AGGR),
                                  VanillaMPLayer2(config.NODE_FEATURE_DIM, config.NODE_FEATURE_DIM,
                                                  config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_DIM, aggr=config.AGGR),
                                  VanillaMPLayer2(config.NODE_FEATURE_DIM, config.NODE_FEATURE_DIM,
                                                  config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_DIM, aggr=config.AGGR),
                                  VanillaMPLayer2(config.NODE_FEATURE_DIM, config.NODE_FEATURE_DIM,
                                                  config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_DIM, aggr=config.AGGR)])


        non_linearity = nn.ReLU()
        self.edge_embedding = nn.Sequential(nn.Linear(config.EDGE_INPUT_DIM, config.EDGE_INPUT_DIM),  # 2 + 17*17,
                                            non_linearity,
                                            nn.BatchNorm1d(config.EDGE_INPUT_DIM),
                                            nn.Linear(config.EDGE_INPUT_DIM, config.EDGE_INPUT_DIM),
                                            non_linearity,
                                            nn.BatchNorm1d(config.EDGE_INPUT_DIM),
                                            nn.Linear(config.EDGE_INPUT_DIM, edge_hidden_dim),
                                            non_linearity,
                                            nn.BatchNorm1d(edge_hidden_dim),
                                            nn.Linear(edge_hidden_dim, config.EDGE_FEATURE_DIM))
        self.node_embedding = nn.Sequential(nn.Linear(config.NODE_INPUT_DIM, config.NODE_INPUT_DIM),
                                            non_linearity,
                                            nn.BatchNorm1d(config.NODE_INPUT_DIM),
                                            nn.Linear(config.NODE_INPUT_DIM, config.NODE_INPUT_DIM),
                                            non_linearity,
                                            nn.BatchNorm1d(config.NODE_INPUT_DIM),
                                            nn.Linear(config.NODE_INPUT_DIM, node_hidden_dim),
                                            non_linearity,
                                            nn.BatchNorm1d(node_hidden_dim),
                                            nn.Linear(node_hidden_dim, config.NODE_FEATURE_DIM))

        self.classification = nn.Sequential(nn.Linear(config.EDGE_FEATURE_DIM, 1))
        # """

        self.steps = config.STEPS

    def forward(self, x, edge_attr, edge_index):
        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)

        for mpn in self.mpn:
            node_features, edge_features = mpn(node_features, edge_features, edge_index)
        return self.classification(edge_features)

