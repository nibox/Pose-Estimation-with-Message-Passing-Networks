import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from .layers import _make_mlp, TypeAwareMPNLayer
from .utils import sum_node_types


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
            self.mlp_edge = nn.Sequential(nn.Linear(node_feature_dim * 2 * node_factor + edge_feature_dim * edge_factor + 32, edge_feature_hidden),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(edge_feature_hidden, edge_feature_dim),
                                          nn.ReLU(inplace=True),
                                          )

        elif edge_mlp == "per_type":
            raise NotImplementedError


        # self.mlp_edge = PerInvMLP(node_feature_dim, edge_feature_dim)
        self.mlp_node = nn.Sequential(nn.Linear(node_feature_dim * node_factor + edge_feature_dim + 16, node_feature_dim),
                                      nn.ReLU(inplace=True),
                                      )
        self.update_mlp = None


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
            raise NotImplementedError
        return aggr_out


class NodeClassificationMPNSelfAttention(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_skip_connections = config.SKIP
        if config.AGGR_TYPE == "agnostic":
            self.mpn_node_cls = MPLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                      aggr=config.AGGR, skip=config.SKIP, use_node_update_mlp=config.USE_NODE_UPDATE_MLP,
                                        edge_mlp=config.EDGE_MLP)

        if config.LATE_FUSION_POS:
            raise NotImplementedError
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

        # attention stuff
        self.key_transform = nn.Linear(config.NODE_FEATURE_DIM, 16)
        self.query_transform = nn.Conv2d(32, 16, 1)
        self.value_transform = nn.Conv2d(32, 16, 1)


    def forward(self, x, edge_attr, edge_index, **kwargs):

        feature_maps = kwargs["feature_maps"]
        values = self.value_transform(feature_maps)
        values = values.view(values.shape[0], values.shape[1], -1)
        queries = self.query_transform(feature_maps)
        queries = queries.view(queries.shape[0], queries.shape[1], -1)

        node_types = sum_node_types(self.node_summary, kwargs["node_types"])
        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)

        node_features_initial = node_features
        edge_features_initial = edge_features
        """
        if i >= self.steps - self.aux_loss_steps - 1:
            preds_node.append(self.node_classification(node_features).squeeze())
            preds_class.append(self.classification(node_features))
        """

        preds_edge = []
        preds_node = []
        preds_class = []
        for i in range(self.edge_steps):
            # attention stuff
            key = self.key_transform(node_features)[None]  # key: 1, N_2, C
            attn = torch.matmul(key, queries)  # queries: B, C, N_1 key: 1, N_2, C, sim: B, N_2, N_1
            attn = attn.softmax(dim=2)
            result = torch.bmm(values, attn.transpose(1, 2))  # result B, C, N_2
            result = result.transpose(0, 2)
            result = result[torch.arange(result.shape[0]), :, kwargs["batch_index"]]
            node_features = torch.cat([node_features, result], dim=1)
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)
            node_features, edge_features = self.mpn_node_cls(node_features, edge_features, edge_index,
                                                             node_types=node_types)

        preds_edge.append(self.edge_classification(edge_features).squeeze())
        preds_node.append(self.node_classification(node_features).squeeze())
        preds_class.append(self.classification(node_features))


        return preds_edge, preds_node, preds_class, node_features, edge_features
