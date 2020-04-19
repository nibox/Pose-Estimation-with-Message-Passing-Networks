import torch
from torch_geometric.data import Data
import numpy as np


########################################################################################################################
### The classes below are used to create graphs from detections and ground truth files and create datasets with them.
########################################################################################################################

def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    
    elif isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()


class Graph(Data):
    """
    This is the class we use to instantiate our graph objects. We inherit from torch_geometric's Data class and add a
    few convenient methods to it, mostly related to changing data types in a single call.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def change_attrs_types(self, attr_change_fun):
        """
        Base method for all methods related to changing attribute types. Iterates over the attributes names in
        self.data_attr_names, and changes its type via attr_change_fun

        Args:
            attr_change_fun: callable function to change a variable's type
        """
        # These are our standard 'data-related' attribute names.
        data_attr_names = ['x', # Node feature vecs
                           'edge_attr', # Edge Feature vecs
                           'edge_index', # Sparse Adjacency
                           'node_names', # Node names (integer values)
                           'node_labels', # Node binary values
                           'edge_labels'] # Edge binary values
        for attr_name in data_attr_names:
            if hasattr(self, attr_name):
                old_attr_val = getattr(self, attr_name)
                setattr(self, attr_name, attr_change_fun(old_attr_val))

    def tensor(self):
        self.change_attrs_types(attr_change_fun = torch.tensor)
        return self

    def float(self):
        self.change_attrs_types(attr_change_fun = lambda x: x.float())
        return self

    def numpy(self):
        self.change_attrs_types(attr_change_fun = to_numpy)
        return self

    def cpu(self):
        self.change_attrs_types(attr_change_fun = lambda x: x.cpu())
        return self

    def cuda(self):
        self.change_attrs_types(attr_change_fun=lambda x: x.cuda())
        return self
