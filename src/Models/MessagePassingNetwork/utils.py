import numpy as np
import torch
from Cython.Includes import numpy


def sum_node_types(node_summary, node_types):
    # 'nose','eye_l','eye_r','ear_l','ear_r', 'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
    # 'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r'
    if node_summary == "not":
        return node_types
    elif node_summary == "left_right":
        mapping = torch.from_numpy(np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8])).to(node_types.device)
        node_types = mapping[node_types]
        return node_types
    elif node_summary == "per_body_part":
        # head, shoulder, arm left, arm right, left leg, right ,leg
        mapping = torch.from_numpy(np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5])).to(node_types.device)
        node_types = mapping[node_types]
        return node_types