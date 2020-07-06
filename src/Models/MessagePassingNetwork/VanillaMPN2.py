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
    def __init__(self, n_in, n_out, e_in, e_out, aggr="add", use_node_update_mlp=False):
        super(VanillaMPLayer2, self).__init__(aggr=aggr)
        # todo better architecture

        self.mlp_edge = nn.Sequential(nn.Linear(n_in * 2 + e_in, e_out),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(e_out),
                                      )

        self.mlp_node = nn.Sequential(nn.Linear(n_in + e_out, n_out),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(n_out)
                                      )

        self.node_update_mlp = nn.Sequential(nn.Linear(n_out, n_out), nn.ReLU(), nn.BatchNorm1d(n_out)) if use_node_update_mlp else None

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
        if self.node_update_mlp is not None:
            aggr_out = self.node_update_mlp(aggr_out)
        return aggr_out


class VanillaMPN2(torch.nn.Module):

    def __init__(self, config, node_hidden_dim=128, edge_hidden_dim=128):
        super().__init__()
        self.mpn = nn.ModuleList([VanillaMPLayer2(config.NODE_FEATURE_DIM, config.NODE_FEATURE_DIM,
                                                  config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_DIM, aggr=config.AGGR,
                                                  use_node_update_mlp=config.USE_NODE_UPDATE_MLP)
                                  for _ in range(config.STEPS)])
        self.edge_embedding = _make_mlp(config.EDGE_INPUT_DIM, config.EDGE_EMB.OUTPUT_SIZES, bn=config.BN,
                                        end_with_relu=config.EDGE_EMB.END_WITH_RELU)
        self.node_embedding = _make_mlp(config.NODE_INPUT_DIM, config.NODE_EMB.OUTPUT_SIZES, bn=config.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)
        self.classification = _make_mlp(config.EDGE_FEATURE_DIM, config.CLASS.OUTPUT_SIZES, bn=config.BN, init_trick=True,
                                        end_with_relu=False)

        self.steps = config.STEPS
        self.aux_loss_steps = config.AUX_LOSS_STEPS

    def forward(self, x, edge_attr, edge_index, **kwargs):
        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)

        # node_features_initial = node_features
        # edge_features_initial = edge_features

        preds_edge = []
        for i, mpn in enumerate(self.mpn):
            # if self.use_skip_connections:
            #     node_features = torch.cat([node_features_initial, node_features], dim=1)
            #     edge_features = torch.cat([edge_features_initial, edge_features], dim=1)
            node_features, edge_features = mpn(node_features, edge_features, edge_index)

            if i >= self.steps - self.aux_loss_steps - 1:
                preds_edge.append(self.classification(edge_features).squeeze())
        return preds_edge, None

