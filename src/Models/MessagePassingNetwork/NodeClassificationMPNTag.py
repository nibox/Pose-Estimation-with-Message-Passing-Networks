import torch
import torch.nn as nn
from .layers import _make_mlp, MPLayer, TypeAwareMPNLayer
from .utils import sum_node_types


class NodeClassificationMPNTag(torch.nn.Module):

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
                num_types = 17
            elif self.node_summary == "left_right":
                num_types = 9
            self.mpn_node_cls = TypeAwareMPNLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                        aggr=config.AGGR, skip=config.SKIP,
                                        edge_mlp=config.EDGE_MLP, num_types=num_types)

        self.edge_embedding = _make_mlp(config.EDGE_INPUT_DIM, config.EDGE_EMB.OUTPUT_SIZES, bn=config.EDGE_EMB.BN,
                                        end_with_relu=config.EDGE_EMB.END_WITH_RELU)
        self.node_embedding = _make_mlp(config.NODE_INPUT_DIM, config.NODE_EMB.OUTPUT_SIZES, bn=config.NODE_EMB.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)

        self.node_classification = _make_mlp(config.NODE_FEATURE_DIM, config.NODE_CLASS.OUTPUT_SIZES, bn=config.BN)
        self.tag_pred = _make_mlp(config.NODE_FEATURE_DIM, config.NODE_TAG.OUTPUT_SIZES, bn=config.BN)
        self.classification = _make_mlp(config.NODE_FEATURE_DIM, config.CLASS.OUTPUT_SIZES, bn=config.BN)

        self.edge_steps = config.STEPS
        self.node_steps = config.NODE_STEPS
        self.aux_loss_steps = config.AUX_LOSS_STEPS
        self.skip = config.TAG_SKIP

    def forward(self, x, edge_attr, edge_index, **kwargs):

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

        preds_node = []
        preds_class = []
        preds_tags = []
        for i in range(self.edge_steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)
            node_features, edge_features = self.mpn_node_cls(node_features, edge_features, edge_index,
                                                             node_types=node_types)

        if self.skip:
            preds_tags.append(self.tag_pred(node_features).squeeze() + kwargs["joint_tags"])
        else:
            preds_tags.append(self.tag_pred(node_features).squeeze())

        for i in range(self.node_steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)
            node_features, edge_features = self.mpn_node_cls(node_features, edge_features, edge_index)

        preds_node.append(self.node_classification(node_features).squeeze())
        preds_class.append(self.classification(node_features))

        return [None], preds_node, preds_class, preds_tags
