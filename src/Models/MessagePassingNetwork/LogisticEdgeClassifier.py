import torch
from .layers import _make_mlp


class LogisticEdgeClassifier(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.edge_classifier = _make_mlp(config.EDGE_FEATURE_DIM, config.EDGE_CLASS.OUTPUT_SIZES, bn=config.BN)

    def forward(self, x, edge_attr, edge_index, **kwargs):

        preds_edges = []
        preds_edges.append(self.edge_classifier(edge_attr.detach()).squeeze())


        return preds_edges, [None], None, [None]
