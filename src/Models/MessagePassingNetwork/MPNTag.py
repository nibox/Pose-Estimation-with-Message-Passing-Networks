import torch
import torch.nn as nn
from .layers import _make_mlp, MPLayer, TypeAwareMPNLayer
from .utils import sum_node_types


class MPNTag(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_skip_connections = config.SKIP
        if config.AGGR_TYPE == "agnostic":
            self.mpn_node_cls = MPLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                      aggr=config.AGGR, skip=config.SKIP, use_node_update_mlp=config.USE_NODE_UPDATE_MLP,
                                        edge_mlp=config.EDGE_MLP)
        else:
            raise NotImplementedError

        self.edge_embedding = _make_mlp(config.EDGE_INPUT_DIM, config.EDGE_EMB.OUTPUT_SIZES, bn=config.EDGE_EMB.BN,
                                        end_with_relu=config.EDGE_EMB.END_WITH_RELU)
        self.node_embedding = _make_mlp(config.NODE_INPUT_DIM, config.NODE_EMB.OUTPUT_SIZES, bn=config.NODE_EMB.BN,
                                        end_with_relu=config.NODE_EMB.END_WITH_RELU)

        self.tag_pred = _make_mlp(config.NODE_FEATURE_DIM, config.NODE_TAG.OUTPUT_SIZES, bn=config.BN)

        self.edge_steps = config.STEPS
        self.aux_loss_steps = config.AUX_LOSS_STEPS
        self.skip = config.TAG_SKIP

    def forward(self, x, edge_attr, edge_index, **kwargs):

        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)

        node_features_initial = node_features
        edge_features_initial = edge_features

        preds_tags = []
        for i in range(self.edge_steps):
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)
            node_features, edge_features = self.mpn_node_cls(node_features, edge_features, edge_index)

        preds_tags.append(self.tag_pred(node_features).squeeze())


        return [None], [None], None, preds_tags
