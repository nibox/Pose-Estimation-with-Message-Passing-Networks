import torch
import torch.nn as nn
import numpy as np

def non_maximum_suppression(scoremap, threshold=0.05):

    pool = nn.MaxPool2d(3, 1, 1)
    pooled = pool(scoremap)
    scoremap = torch.where(scoremap < threshold, torch.zeros_like(scoremap), scoremap)
    maxima = torch.eq(pooled, scoremap).float()
    return maxima


def to_numpy(array: [torch.Tensor, np.array]):
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    else:
        return array
