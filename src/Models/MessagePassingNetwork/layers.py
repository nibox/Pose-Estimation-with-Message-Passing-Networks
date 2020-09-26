import torch
import torch.nn as nn
from torch import nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, scatter_max


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
    def __init__(self, node_feature_dim, edge_feature_dim, edge_feature_hidden, aggr, use_node_update_mlp, skip=False,
                 edge_mlp="agnostic"):
        super(MPLayer, self).__init__(aggr=aggr)
        # todo better architecture

        node_factor = 2 if skip else 1
        edge_factor = 2 if skip else 1

        self.edge_mlp = edge_mlp
        if edge_mlp == "agnostic":
            self.mlp_edge = nn.Sequential(nn.Linear(node_feature_dim * 2 * node_factor + edge_feature_dim * edge_factor, edge_feature_hidden),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(edge_feature_hidden, edge_feature_dim),
                                          nn.ReLU(inplace=True),
                                          )

        elif edge_mlp == "per_type":
            self.mlp_edge = TypeAwareEdgeUpdate(node_feature_dim*node_factor, edge_feature_dim*edge_factor,
                                                edge_feature_hidden)


        # self.mlp_edge = PerInvMLP(node_feature_dim, edge_feature_dim)
        self.mlp_node = nn.Sequential(nn.Linear(node_feature_dim * node_factor + edge_feature_dim, node_feature_dim),
                                      nn.ReLU(inplace=True),
                                      )
        self.update_mlp = nn.Sequential(nn.Linear(node_feature_dim, node_feature_dim), nn.ReLU()) if use_node_update_mlp else None


    def forward(self, x, edge_attr, edge_index, **kwargs):
        num_nodes = x.size(0)

        j, i = edge_index  # message is from j to i
        x_i, x_j = x[i], x[j]
        e_ij = edge_attr
        if self.edge_mlp == "agnostic":
            edge_attr = self.mlp_edge(torch.cat([x_i, x_j, e_ij], dim=1))  # todo ask if edge features are permutation inv
        elif self.edge_mlp == "per_type":
            node_types_1 = kwargs["node_types"][i]
            node_types_2 = kwargs["node_types"][j]
            edge_attr = self.mlp_edge(x_i, x_j, e_ij, node_types_1, node_types_2)  # todo ask if edge features are permutation inv

        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x, edge_attr=edge_attr), edge_attr

    def message(self, x_i, x_j, edge_attr):
        # edge_attr = self.mlp_edge(torch.cat([x_i, x_j, edge_attr], dim=1))
        out = self.mlp_node(torch.cat([x_i, edge_attr], dim=1))
        return out

    def update(self, aggr_out):
        if self.update_mlp is not None:
            aggr_out = self.update_mlp(aggr_out)
        return aggr_out


class TypeAwareMPNLayer(MessagePassing):

    # todo with or without inital feature skip connection
    def __init__(self, node_feature_dim, edge_feature_dim, edge_feature_hidden, aggr, skip=False,
                 edge_mlp="agnostic"):
        super().__init__(aggr=aggr)
        # todo better architecture

        node_factor = 2 if skip else 1
        edge_factor = 2 if skip else 1

        self.edge_mlp = edge_mlp
        if edge_mlp == "agnostic":
            self.mlp_edge = nn.Sequential(nn.Linear(node_feature_dim * 2 * node_factor + edge_feature_dim * edge_factor, edge_feature_hidden),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(edge_feature_hidden, edge_feature_dim),
                                          nn.ReLU(inplace=True),
                                          )

        elif edge_mlp == "per_type":
            self.mlp_edge = TypeAwareEdgeUpdate(node_feature_dim * node_factor, edge_feature_dim * node_factor, edge_feature_hidden)


        # self.mlp_edge = PerInvMLP(node_feature_dim, edge_feature_dim)

        self.mlp_node = TypeAwareNodeUpdate(node_feature_dim * node_factor + edge_feature_dim, node_feature_dim)
        self.update_mlp = nn.Sequential(nn.Linear(node_feature_dim * 17, node_feature_dim), nn.ReLU(inplace=True))


    def forward(self, x, edge_attr, edge_index, **kwargs):
        num_nodes = x.size(0)

        j, i = edge_index  # message is from j to i
        x_i, x_j = x[i], x[j]
        e_ij = edge_attr
        if self.edge_mlp == "agnostic":
            edge_attr = self.mlp_edge(torch.cat([x_i, x_j, e_ij], dim=1))  # todo ask if edge features are permutation inv
        elif self.edge_mlp == "per_type":
            node_types_1 = kwargs["node_types"][i]
            node_types_2 = kwargs["node_types"][j]
            edge_attr = self.mlp_edge(x_i, x_j, e_ij, node_types_1, node_types_2)  # todo ask if edge features are permutation inv

        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x, edge_attr=edge_attr, source_type=kwargs["node_types"][j], num_nodes=num_nodes, target_nodes=i), edge_attr

    def message(self, x_i, x_j, edge_attr, source_type):
        out = self.mlp_node(x_i, edge_attr, source_type)
        return out

    def aggregate(self, inputs, index, source_type, num_nodes, target_nodes):

        feature_dim = inputs.shape[1]
        updates = torch.zeros(num_nodes, 17, feature_dim, dtype=torch.float32, device=inputs.device)
        for i in range(17):
            types = source_type==i
            updates[:, i] = scatter(inputs[types], index[types], dim=0, reduce=self.aggr, dim_size=num_nodes)
        return updates.reshape(num_nodes, -1)


    def update(self, aggr_out):
        aggr_out = self.update_mlp(aggr_out)
        return aggr_out

class TypeAwareNodeUpdate(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, output_dim),
                      nn.ReLU(inplace=True),
                      ) for i in range(17)])
        self.output_dim = output_dim
    def forward(self, x, edge_attr, node_types):

        out = torch.zeros(x.shape[0], self.output_dim, dtype=torch.float32, device=x.device)
        for i in range(17):
            types = node_types == i
            out[types] = self.mlp[i](torch.cat([x[types], edge_attr[types]], dim=1))
        return out

class TypeAwareEdgeUpdate(nn.Module):

    def __init__(self, node_feature_dim, edge_feature_dim, output_dim):
        super().__init__()
        self.layer_1 = nn.ModuleList([nn.Linear(node_feature_dim, output_dim) for i in range(17)])
        self.layer_2 = nn.ModuleList([nn.Linear(node_feature_dim, output_dim) for i in range(17)])
        self.edge_layer = nn.Linear(edge_feature_dim, output_dim)
        self.out = nn.Sequential(nn.ReLU(inplace=True),
                                 nn.Linear(3 * output_dim, output_dim),
                                 nn.ReLU(inplace=True))
        self.output_dim = output_dim

    def forward(self, nodes_1, nodes_2, edges, node_types_1, node_types_2):
        num_nodes = nodes_1.shape[0]
        device = nodes_1.device

        tmp_1 = torch.zeros(num_nodes, self.output_dim, dtype=torch.float32, device=device)
        tmp_2 = torch.zeros(num_nodes, self.output_dim, dtype=torch.float32, device=device)
        for i in range(17):
            types_1 = node_types_1 == i
            tmp_1[types_1] = self.layer_1[i](nodes_1[types_1])
        for i in range(17):
            types_2 = node_types_2 == i
            tmp_2[types_2] = self.layer_2[i](nodes_2[types_2])
        edges = self.edge_layer(edges)
        cat_inp = torch.cat([tmp_1, tmp_2, edges], dim=1)
        return self.out(cat_inp)


class MPLayer2(MessagePassing):

    # todo with or without inital feature skip connection
    def __init__(self, node_feature_dim, edge_feature_dim, edge_feature_hidden, aggr, use_node_update_mlp, skip=False,
                 aggregation_method="agnostic", edge_mlp="agnostic"):
        super(MPLayer2, self).__init__(aggr=aggr)
        # todo better architecture

        node_factor = 2 if skip else 1
        edge_factor = 2 if skip else 1

        if edge_mlp == "agnostic":
            self.mlp_edge = nn.Sequential(nn.Linear(node_feature_dim * 2 * node_factor + edge_feature_dim * edge_factor, edge_feature_hidden),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(edge_feature_hidden, edge_feature_dim),
                                          nn.ReLU(inplace=True),
                                          )

        elif edge_mlp == "per_type":
            self.mlp_edge = TypeAwareEdgeUpdate(node_feature_dim*node_factor, edge_feature_dim*edge_factor,
                                                edge_feature_hidden)

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

"""
class NaiveNodeMPN(MessagePassing):

    def __init__(self, node_feature_dim, edge_feature_dim, edge_feature_hidden, aggr, use_node_update_mlp, skip=False):
        super(NaiveNodeMPN, self).__init__(aggr=aggr)
        # todo better architecture

        node_factor = 2 if skip else 1
        edge_factor = 2 if skip else 1

        self.mlp_edge = nn.Sequential(nn.Linear(node_feature_dim * 2 * node_factor, edge_feature_hidden),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(edge_feature_hidden, edge_feature_dim),
                                      nn.ReLU(inplace=True),
                                      )

        # self.mlp_edge = PerInvMLP(node_feature_dim, edge_feature_dim)
        self.mlp_node = nn.Sequential(nn.Linear(node_feature_dim * node_factor + edge_feature_dim, node_feature_dim),
                                      nn.ReLU(inplace=True),
                                      )
        self.update_mlp = nn.Sequential(nn.Linear(node_feature_dim, node_feature_dim), nn.ReLU()) if use_node_update_mlp else None


    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        j, i = edge_index  # message is from j to i
        x_i, x_j = x[i], x[j]
        edge_attr = self.mlp_edge(torch.cat([x_i, x_j], dim=1))

        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # edge_attr = self.mlp_edge(torch.cat([x_i, x_j, edge_attr], dim=1))
        out = self.mlp_node(torch.cat([x_i, edge_attr], dim=1))
        return out

    def update(self, aggr_out):
        if self.update_mlp is not None:
            aggr_out = self.update_mlp(aggr_out)
        return aggr_out
"""