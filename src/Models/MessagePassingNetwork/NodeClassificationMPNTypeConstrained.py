import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from Utils.Utils import subgraph_mask
from torch_scatter.composite import scatter_softmax
from .layers import _make_mlp, MPLayer, TypeAwareMPNLayer


class NodeClassificationMPNTypeConstrained(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_skip_connections = config.SKIP
        if config.AGGR_TYPE == "agnostic":
            self.mpn_node_cls = MPLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                      aggr=config.AGGR, skip=config.SKIP, use_node_update_mlp=config.USE_NODE_UPDATE_MLP,
                                        edge_mlp=config.EDGE_MLP)
        elif config.AGGR_TYPE == "per_type":
            self.mpn_node_cls = TypeAwareMPNLayer(config.NODE_FEATURE_DIM, config.EDGE_FEATURE_DIM, config.EDGE_FEATURE_HIDDEN,
                                        aggr=config.AGGR, skip=config.SKIP,
                                        edge_mlp=config.EDGE_MLP)


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
        self.edge_const_emb = nn.Linear(config.NODE_FEATURE_DIM, config.NODE_FEATURE_DIM)

    def forward(self, x, edge_attr, edge_index, **kwargs):

        node_types = self.sum_node_types(kwargs["node_types"])
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
            if self.use_skip_connections:
                node_features = torch.cat([node_features_initial, node_features], dim=1)
                edge_features = torch.cat([edge_features_initial, edge_features], dim=1)
            node_features, edge_features = self.mpn_node_cls(node_features, edge_features, edge_index,
                                                             node_types=node_types)

        preds_node.append(self.node_classification(node_features).squeeze())
        preds_class.append(self.classification(node_features))

        edge_pred = self.edge_classification(edge_features).squeeze()
        source_idx = edge_index[0]
        target_idx = edge_index[1]
        source_types = preds_class[-1].argmax(dim=1).detach()[source_idx] # node_types[source_idx]
        node_emb = self.edge_const_emb(node_features).squeeze()
        edge_out = torch.zeros_like(edge_pred, dtype=torch.float32, device=x.device)
        edge_scores = (node_emb[source_idx] * node_emb[target_idx]).sum(dim=1)
        for i in range(17):
            types = source_types == i
            edge_out[types] = scatter_softmax(edge_scores[types], target_idx[types], dim=0)

        preds_edge.append(edge_out * edge_pred.sigmoid())


        return preds_edge, preds_node, preds_class, node_features, edge_features

    def sum_node_types(self, node_types):
        # 'nose','eye_l','eye_r','ear_l','ear_r', 'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
        # 'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r'
        if self.node_summary == "not":
            return node_types
        elif self.node_summary == "left_right":
            raise NotImplementedError
        elif self.node_summary == "per_body_part":
            # head, shoulder, arm left, arm right, left leg, right ,leg
            mapping = torch.from_numpy(np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5])).to(node_types.device)
            node_types = mapping[node_types]
            return node_types
