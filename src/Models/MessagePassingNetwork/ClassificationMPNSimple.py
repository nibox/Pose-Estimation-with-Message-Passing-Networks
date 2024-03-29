import torch
from .layers import _make_mlp, MPLayer, TypeAwareMPNLayer
from .utils import sum_node_types


class ClassificationMPNSimple(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_skip_connections = config.SKIP
        self.node_summary = config.NODE_TYPE_SUMMARY
        if config.AGGR_TYPE == "agnostic":
            self.mpn_node_cls = MPLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                        aggr=config.AGGR, skip=config.SKIP, use_node_update_mlp=config.USE_NODE_UPDATE_MLP,
                                        edge_mlp=config.EDGE_MLP)
        elif config.AGGR_TYPE == "per_type":
            num_types = None
            if self.node_summary == "per_body_part":
                num_types = 6
            elif self.node_summary == "not":
                num_types = config.NUM_JOINTS
            elif self.node_summary == "left_right":
                num_types = 9
            self.mpn_node_cls = TypeAwareMPNLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                                  aggr=config.AGGR, skip=config.SKIP,
                                                  edge_mlp=config.EDGE_MLP, num_types=num_types, aggr_sub=config.AGGR_SUB,
                                                  update_type=config.UPDATE_TYPE)

        self.edge_embedding = _make_mlp(config.EDGE_INPUT_DIM, config.EDGE_EMB.OUTPUT_SIZES, bn=config.EDGE_EMB.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)
        self.node_embedding = _make_mlp(config.NODE_INPUT_DIM, config.NODE_EMB.OUTPUT_SIZES, bn=config.NODE_EMB.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)

        self.edge_classification = _make_mlp(config.EDGE_FEATURE_DIM, config.EDGE_CLASS.OUTPUT_SIZES, bn=config.BN)
        self.node_classification = _make_mlp(config.NODE_FEATURE_DIM, config.NODE_CLASS.OUTPUT_SIZES, bn=config.BN)

        self.node_steps = config.STEPS
        self.edge_steps = config.EDGE_STEPS

    def forward(self, x, edge_attr, edge_index, **kwargs):

        node_types = sum_node_types("not", kwargs["node_types"])
        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)

        node_features_initial = node_features
        edge_features_initial = edge_features

        preds_edge = []
        preds_node = []
        for i in range(self.node_steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)
            node_features, edge_features = self.mpn_node_cls(node_features, edge_features, edge_index, node_types=node_types)
        preds_node.append(self.node_classification(node_features).squeeze())

        for i in range(self.edge_steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)
            node_features, edge_features = self.mpn_node_cls(node_features, edge_features, edge_index, node_types=node_types)

        preds_edge.append(self.edge_classification(edge_features).squeeze())

        return preds_edge, preds_node, None, [None]
