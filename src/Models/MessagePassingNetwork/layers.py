import torch
import torch.nn as nn
from torch import nn as nn
from torch_geometric.nn import MessagePassing


def _make_mlp(input_dim, hidden_dims, bn=False, init_trick=False, end_with_relu=False):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    if len(hidden_dims) != 1:
        layers.append(nn.ReLU(inplace=True))
    if bn and len(hidden_dims) != 1:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
    for i in range(1, len(hidden_dims)):
        layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        if i == len(hidden_dims) - 1 and init_trick:   # last layer
            nn.init.constant_(layers[-1].bias, -2.0)

        if i != len(hidden_dims) - 1:
            layers.append(nn.ReLU(inplace=True))
            if bn:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
    if end_with_relu:
        layers.append(nn.ReLU(inplace=True))
        if bn:
            layers.append(nn.BatchNorm1d(hidden_dims[i]))

    return nn.Sequential(*layers)


class MPLayer(MessagePassing):

    # todo with or without inital feature skip connection
    def __init__(self, node_feature_dim, edge_feature_dim, edge_feature_hidden, aggr, use_node_update_mlp, skip=False):
        super(MPLayer, self).__init__(aggr=aggr)
        # todo better architecture

        node_factor = 2 if skip else 1
        edge_factor = 2 if skip else 1

        self.mlp_edge = nn.Sequential(nn.Linear(node_feature_dim * 2 * node_factor + edge_feature_dim * edge_factor, edge_feature_hidden),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(edge_feature_hidden, edge_feature_dim),
                                      nn.ReLU(inplace=True),
                                      )

        # self.mlp_edge = PerInvMLP(node_feature_dim, edge_feature_dim)
        self.mlp_node = nn.Sequential(nn.Linear(node_feature_dim * node_factor + edge_feature_dim, node_feature_dim),
                                      nn.ReLU(inplace=True),
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