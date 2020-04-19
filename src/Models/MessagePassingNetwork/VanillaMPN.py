import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class MPN(MessagePassing):
    def __init__(self, node_feature_dim, edge_feature_dim):
        super(MPN, self).__init__(aggr="add")
        self.mlp_edge = torch.nn.Linear(node_feature_dim * 2 + edge_feature_dim, edge_feature_dim)
        self.mlp_node = torch.nn.Linear(node_feature_dim + edge_feature_dim, node_feature_dim)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        i, j = edge_index[0], edge_index[1]

        x_i, x_j = x[i], x[j]
        e_ij = edge_attr
        edge_attr = self.mlp_edge(torch.cat([x_j, x_i, e_ij], dim=1))  # todo ask if edge features are permutation inv

        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr), edge_attr

    def message(self, x_i, edge_attr):
        out = self.mlp_node(torch.cat([x_i, edge_attr], dim=1))
        return out

    def update(self, aggr_out):
        return aggr_out


class PerInvMLP(nn.Module):

    def __init__(self, node_feature_dim, edge_feature_dim):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.node_mlp = nn.Sequential(nn.Linear(node_feature_dim, edge_feature_dim),
                                      nn.ReLU(),
                                      nn.Linear(edge_feature_dim, edge_feature_dim))
        self.edge_mlp = nn.Sequential(nn.Linear(edge_feature_dim, edge_feature_dim),
                                      nn.ReLU(),
                                      nn.Linear(edge_feature_dim, edge_feature_dim))
        self.fusion = nn.Sequential(nn.Linear(2 * edge_feature_dim, edge_feature_dim),
                                    nn.ReLU(),
                                    nn.Linear(edge_feature_dim, edge_feature_dim))

    def forward(self, x):
        node_1 = self.node_mlp(x[:, :self.node_feature_dim])
        node_2 = self.node_mlp(x[:, self.node_feature_dim: 2 * self.node_feature_dim])
        edge = self.edge_mlp(x[:, self.node_feature_dim * 2:])
        edge_attr = self.fusion(
            torch.cat([node_1 + node_2, edge], dim=1))  # todo ask if edge features are permutation inv
        return edge_attr


class VanillaMPLayer(MessagePassing):

    # todo with or without inital feature skip connection
    def __init__(self, node_feature_dim, edge_feature_dim):
        super(VanillaMPLayer, self).__init__(aggr="add")
        # todo better architecture

        non_lin = nn.ReLU()
        self.mlp_edge = nn.Sequential(nn.Linear(node_feature_dim * 2 + edge_feature_dim, edge_feature_dim),
                                      non_lin,
                                      nn.Linear(edge_feature_dim, edge_feature_dim),
                                      non_lin,
                                      nn.Linear(edge_feature_dim, edge_feature_dim),
                                      non_lin,
                                      nn.Linear(edge_feature_dim, edge_feature_dim),
                                      )


        # self.mlp_edge = PerInvMLP(node_feature_dim, edge_feature_dim)
        self.mlp_node = nn.Sequential(nn.Linear(node_feature_dim + edge_feature_dim, node_feature_dim),
                                      non_lin,
                                      nn.Linear(node_feature_dim, node_feature_dim),
                                      non_lin,
                                      nn.Linear(node_feature_dim, node_feature_dim),
                                      )

    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)

        j, i = edge_index # message is from j to i
        x_i, x_j = x[i], x[j]
        e_ij = edge_attr
        edge_attr = self.mlp_edge(torch.cat([x_i, x_j, e_ij], dim=1))  # todo ask if edge features are permutation inv

        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x, edge_attr=edge_attr), edge_attr

    def message(self, x_i, x_j, edge_attr):
        # edge_attr = self.mlp_edge(torch.cat([x_i, x_j, edge_attr], dim=1))
        out = self.mlp_node(torch.cat([x_i, edge_attr], dim=1))
        return out

    def update(self, aggr_out):
        return aggr_out


class VanillaMPN(torch.nn.Module):

    def __init__(self, steps, node_feature_dim, edge_feature_dim):
        super().__init__()
        self.mpn = VanillaMPLayer(node_feature_dim, edge_feature_dim)

        non_linearity = nn.ReLU()
        self.edge_embedding = nn.Sequential(nn.Linear(2 + 17*17, 512),
                                            non_linearity,
                                            nn.Linear(512, edge_feature_dim))
        self.node_embedding = nn.Sequential(nn.Linear(256, 128),
                                            non_linearity,
                                            nn.Linear(128, node_feature_dim))

        """
        self.classification = nn.Sequential(nn.Linear(edge_feature_dim, edge_feature_dim),
                                            non_linearity,
                                            nn.Linear(edge_feature_dim, edge_feature_dim),
                                            non_linearity,
                                            nn.Linear(edge_feature_dim, 1))
        """
        self.classification = nn.Sequential(nn.Linear(edge_feature_dim, 1))

        self.steps = steps

    def forward(self, x, edge_attr, edge_index):
        node_features = self.node_embedding(x)
        edge_features = self.edge_embedding(edge_attr)

        for i in range(self.steps):
            node_features, edge_features = self.mpn(node_features, edge_features, edge_index)

        return self.classification(edge_features)


def main():
    model = MPN(1, 1)

    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)

    edge_attr = torch.tensor([[1], [1], [12], [12]], dtype=torch.float32)
    x = torch.tensor([[0], [1], [2]], dtype=torch.float32)
    y = torch.tensor([[1], [1], [0], [0]], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2, eps=1e-10)
    for i in range(100):
        optimizer.zero_grad()

        node, edge = model(data)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(edge, data.y)
        loss.backward()
        optimizer.step()
        print(f"{i} {loss.item()}")


if __name__ == "__main__":
    main()
