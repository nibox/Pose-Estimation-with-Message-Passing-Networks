import torch
from .layers import _make_mlp


class TagThreshold(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

    def forward(self, x, edge_attr, edge_index, **kwargs):

        preds_edges = []
        preds_edges.append((edge_attr < 1.0).float())


        return preds_edges, [None], None, [None]
