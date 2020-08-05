import torch
import torch.nn as nn
from Utils.Utils import subgraph_mask
from .layers import _make_mlp, MPLayer


class JointTypeClassification(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_skip_connections = config.SKIP
        self.mpn_node_cls = MPLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                  aggr=config.AGGR, skip=config.SKIP, use_node_update_mlp=config.USE_NODE_UPDATE_MLP)

        self.edge_embedding = _make_mlp(config.EDGE_INPUT_DIM, config.EDGE_EMB.OUTPUT_SIZES, bn=config.EDGE_EMB.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)
        self.node_embedding = _make_mlp(config.NODE_INPUT_DIM, config.NODE_EMB.OUTPUT_SIZES, bn=config.NODE_EMB.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)

        # self.edge_classification = _make_mlp(config.EDGE_FEATURE_DIM, config.EDGE_CLASS.OUTPUT_SIZES, bn=config.BN)
        # self.node_classification = _make_mlp(config.NODE_FEATURE_DIM, config.NODE_CLASS.OUTPUT_SIZES, bn=config.BN)
        self.classification = _make_mlp(config.NODE_FEATURE_DIM, config.CLASS.OUTPUT_SIZES, bn=config.BN)

        self.node_steps = config.STEPS

        # mask edge features containing the edge type, otherwise it would be cheating
        self.mask_features = torch.ones(config.EDGE_INPUT_DIM, dtype=torch.float, requires_grad=False)

    def forward(self, x, edge_attr, edge_index, **kwargs):

        edge_attr = edge_attr * self.mask_features.to(edge_attr.device)

        edge_index = edge_index[:, kwargs["edge_labels"] == 1.0]

        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)

        node_features_initial = node_features
        edge_features_initial = edge_features[kwargs["edge_labels"]==1.0]
        edge_features = edge_features[kwargs["edge_labels"]==1.0]


        preds_edge = []
        preds_node = []
        preds_class = []
        for i in range(self.node_steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)

            node_features, edge_features = self.mpn_node_cls(node_features, edge_features, edge_index)

        preds_node.append(torch.zeros_like(kwargs["node_labels"], dtype=torch.float32, device=node_features.device))
        preds_class.append(self.classification(node_features))
        edge_pred = torch.zeros_like(kwargs["edge_labels"], dtype=torch.float32, device=edge_features.device)
        preds_edge.append(edge_pred)

        return preds_edge, preds_node, preds_class
