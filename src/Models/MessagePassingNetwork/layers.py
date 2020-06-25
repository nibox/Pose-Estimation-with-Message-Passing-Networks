import torch.nn as nn

def _make_mlp(input_dim, hidden_dims, bn=False, init_trick=False):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    if len(hidden_dims) != 1:
        layers.append(nn.ReLU())
    if bn and len(hidden_dims) != 1:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
    for i in range(1, len(hidden_dims)):
        layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        if i == len(hidden_dims) - 1 and init_trick:   # last layer
            nn.init.constant_(layers[-1].bias, -2.0)

        if i != len(hidden_dims) - 1:
            layers.append(nn.ReLU())
            if bn:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))

    return nn.Sequential(*layers)
