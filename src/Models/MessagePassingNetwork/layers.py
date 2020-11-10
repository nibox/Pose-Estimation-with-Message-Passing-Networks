import torch
import torch.nn as nn
from torch import nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, scatter_max, scatter_softmax


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


class HierarchUpdateMlp(torch.nn.Module):

    def __init__(self, node_dim, num_joints):
        super().__init__()

        self.node_dim = node_dim
        self.num_joints = num_joints
        assert self.num_joints in [17, 14]
        if self.num_joints == 17:
            self.first_layer = nn.ModuleList([nn.Linear(node_dim * 5, node_dim//2)] +
                                             [nn.Linear(node_dim * 2, node_dim//2) for _ in range(6)])
        else:
            self.first_layer = nn.ModuleList([nn.Linear(node_dim * 2, node_dim//2)] +
                                             [nn.Linear(node_dim * 2, node_dim//2) for _ in range(6)])

        self.second_layer = nn.ModuleList([nn.Linear(2 * node_dim//2, node_dim//2) for _ in range(6)])
        self.final = nn.Linear(6 * node_dim//2, node_dim)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, update):
        # update.shape: N x F x D
        # 'nose','eye_l','eye_r','ear_l','ear_r', 'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
        # 'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r'
        num_nodes = update.shape[0]
        if self.num_joints == 17:
            order_1 = [(0, 1, 2, 3, 4), (5, 6), (7, 9), (8, 10), (11, 12), (13, 15), (14, 16)]
            order_2 = [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6)]
        else:
            order_1 = [(0, 1), (2, 3), (4, 6), (5, 7), (8, 9), (10, 12), (11, 13)]
            order_2 = [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6)]

        out_1 = torch.zeros(num_nodes, 7, self.node_dim//2, device=update.device, dtype=torch.float32)
        out_2 = torch.zeros(num_nodes, 6, self.node_dim//2, device=update.device, dtype=torch.float32)
        for i, types in enumerate(order_1):
            out_1[:, i] = self.relu(self.first_layer[i](update[:, types].view(num_nodes, -1)))
        for i, types in enumerate(order_2):
            out_2[:, i] = self.relu(self.second_layer[i](out_1[:, types].view(num_nodes, -1)))

        return self.relu(self.final(out_2.reshape(num_nodes, -1)))


class HierarchUpdateCnn(torch.nn.Module):

    def __init__(self, node_dim):
        super().__init__()

        self.head_layer = nn.Linear(node_dim  * 4, node_dim//2)
        self.conv_1 = nn.Conv1d(node_dim, node_dim // 2, 2, 2)
        self.conv_2 = nn.Conv1d(node_dim // 2, node_dim // 2, 2, 2)
        self.final = nn.Linear(5 * node_dim // 2, node_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, update):
        # update.shape: N x F x D
        # 'nose','eye_l','eye_r','ear_l','ear_r', 'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
        # 'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r'
        num_nodes = update.shape[0]
        update = update.permute(0, 2, 1)
        order_1 = [5, 6, 7, 9, 8, 10, 11, 12, 13, 15, 14, 16]
        order_2 = [0, 1, 0, 2, 0, 3, 3, 4, 3, 5]
        out_1 = self.relu(self.conv_1(update[:, :, order_1]))
        head = self.relu(self.head_layer(update[:, :, :4].reshape(num_nodes, -1)))
        update = torch.cat([head[:, :, None], out_1], dim=2)
        update = self.relu(self.conv_2(update[:, :,order_2]))
        return self.relu(self.final(update.reshape(num_nodes, -1)))


class TypeAwareMPNLayer(MessagePassing):

    # todo with or without inital feature skip connection
    def __init__(self, node_feature_dim, edge_feature_dim, edge_feature_hidden, aggr, skip=False,
                 edge_mlp="agnostic", num_types=17, aggr_sub=None, update_type="mlp"):
        super().__init__(aggr=aggr)
        # todo better architecture

        node_factor = 2 if skip else 1
        edge_factor = 2 if skip else 1
        self.num_types = num_types

        self.edge_mlp = edge_mlp
        if edge_mlp == "agnostic":
            self.mlp_edge = nn.Sequential(nn.Linear(node_feature_dim * 2 * node_factor + edge_feature_dim * edge_factor, edge_feature_hidden),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(edge_feature_hidden, edge_feature_dim),
                                          nn.ReLU(inplace=True),
                                          )

        elif edge_mlp == "per_type":
            self.mlp_edge = TypeAwareEdgeUpdate(node_feature_dim * node_factor, edge_feature_dim * node_factor, edge_feature_hidden,
                                                self.num_types)

        self.mlp_node = TypeAwareNodeUpdate(node_feature_dim * node_factor + edge_feature_dim, node_feature_dim)

        self.update_type = update_type
        if update_type == "mlp":
            self.update_mlp = nn.Sequential(nn.Linear(node_feature_dim * num_types, node_feature_dim), nn.ReLU(inplace=True))
        elif update_type == "hierarch_mlp":
            self.update_mlp = HierarchUpdateMlp(node_feature_dim, num_types)
        elif update_type == "hierarch_cnn":
            self.update_mlp = HierarchUpdateCnn(node_feature_dim)
        else:
            raise NotImplementedError

        self.aggr_sub = aggr_sub
        if self.aggr_sub == "node_edge_attn":
            # self.attn_net = nn.Sequential(nn.Linear(edge_feature_dim, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.attn_net = nn.Sequential(nn.Linear(edge_feature_dim, 1))
        else:
            self.attn_net = None


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

    def aggregate(self, inputs, index, source_type, num_nodes, target_nodes, edge_attr):

        if self.aggr in ["add", "max", "mean"] and self.aggr_sub == "None":
            return self.aggr_vanilla(inputs, index, source_type, num_nodes, target_nodes, edge_attr)
        elif self.aggr_sub in ["node_edge_attn", "node_edge_attn_per_type"]:
            return self.aggr_node_attn(inputs, index, source_type, num_nodes, target_nodes, edge_attr)


    def aggr_vanilla(self, inputs, index, source_type, num_nodes, target_nodes, edge_attr):
        feature_dim = inputs.shape[1]
        updates = torch.zeros(num_nodes, self.num_types, feature_dim, dtype=torch.float32, device=inputs.device)
        for i in range(self.num_types):
            types = source_type==i
            updates[:, i] = scatter(inputs[types], index[types], dim=0, reduce=self.aggr, dim_size=num_nodes)
        return updates

    def aggr_node_attn(self, inputs, index, source_type, num_nodes, target_nodes, edge_attr):
        feature_dim = inputs.shape[1]
        updates = torch.zeros(num_nodes, self.num_types, feature_dim, dtype=torch.float32, device=inputs.device)
        attn_scores = self.attn_net(edge_attr)
        for i in range(self.num_types):
            attn_index = 0 if self.aggr_sub == "node_edge_attn" else i
            types = source_type==i
            attn = scatter_softmax(attn_scores[types, attn_index], index[types], dim=0)
            updates[:, i] = scatter(inputs[types] * attn[:, None], index[types], dim=0, reduce="add", dim_size=num_nodes)
        return updates

    def update(self, aggr_out):
        num_nodes = aggr_out.shape[0]
        if self.update_type  == "mlp":
            aggr_out = aggr_out.reshape(num_nodes, -1)
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

    def __init__(self, node_feature_dim, edge_feature_dim, output_dim, num_joints):
        super().__init__()
        self.layer_1 = nn.ModuleList([nn.Linear(node_feature_dim, output_dim) for i in range(num_joints)])
        self.layer_2 = nn.ModuleList([nn.Linear(node_feature_dim, output_dim) for i in range(num_joints)])
        self.edge_layer = nn.Linear(edge_feature_dim, output_dim)
        self.out = nn.Sequential(nn.ReLU(inplace=True),
                                 nn.Linear(3 * output_dim, output_dim),
                                 nn.ReLU(inplace=True))
        self.output_dim = output_dim
        self.num_joints = num_joints

    def forward(self, nodes_1, nodes_2, edges, node_types_1, node_types_2):
        num_nodes = nodes_1.shape[0]
        device = nodes_1.device

        tmp_1 = torch.zeros(num_nodes, self.output_dim, dtype=torch.float32, device=device)
        tmp_2 = torch.zeros(num_nodes, self.output_dim, dtype=torch.float32, device=device)
        for i in range(self.num_joints):
            types_1 = node_types_1 == i
            tmp_1[types_1] = self.layer_1[i](nodes_1[types_1])
        for i in range(self.num_joints):
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