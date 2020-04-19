import torch
import torch_geometric
import torch_geometric.utils as gutils
from scipy.optimize import linear_sum_assignment
import numpy as np
from Utils.Utils import *


class NaiveGraphConstructor:

    def __init__(self, scoremaps, features, joints_gt):
        self.scoremaps = scoremaps
        self.features = features
        self.joints_gt = joints_gt
        self.batch_size = scoremaps.shape[0]

    def _construct_mpn_graph(self, joint_det, features):
        # joint_locations tuple (joint type, height, width)
        # extract joint features from feature map
        # joint_idx_det, joint_y, joint_x = joint_map.nonzero(as_tuple=True)

        num_joints_det = len(joint_det)

        joint_x = joint_det[:, 0]
        joint_y = joint_det[:, 1]
        joint_features = features[:, joint_y, joint_x]

        # construct node features
        x = joint_features.T

        # construct joint_det graph (fully connected)
        # edge_index, _ = gutils.dense_to_sparse(torch.ones([num_joints_det, num_joints_det], dtype=torch.long))
        # todo using k_nn and setting distances between joints of certain type on can create the different graphs
        temp = joint_det[:, :2].float()
        edge_index = torch_geometric.nn.knn_graph(temp, 50)
        edge_index = gutils.to_undirected(edge_index)
        edge_index, _ = gutils.remove_self_loops(edge_index)

        # construct inital edge_features
        edge_attr_y = joint_y[edge_index[1]] - joint_y[edge_index[0]]
        edge_attr_x = joint_x[edge_index[1]] - joint_x[edge_index[0]]
        edge_attr = torch.stack([edge_attr_x, edge_attr_y], 1).view(-1, 2).float()
        return x, edge_attr, edge_index

    def construct_graph(self):
        for batch in range(self.batch_size):
            joint_det = joint_det_from_scoremap(self.scoremaps[batch], threshold=0.007)

            # joint_map = non_maximum_suppression(self.scoremaps[batch], threshold=0.007)
            # print(f"gt kp: {num_joints_gt}, d kp: {num_joints_det}")

            ###############cheating#################
            # extend joint_det with joints_gt in order to have a perfect matching at train time
            # !!! be careufull to use it at test time!!!
            # todo move in function
            person_idx_gt, joint_idx_gt = self.joints_gt[batch, :, :, 2].nonzero(as_tuple=True)
            tmp = self.joints_gt[batch, person_idx_gt, joint_idx_gt, :2].long()
            joints_gt_position = torch.cat([tmp, joint_idx_gt.unsqueeze(1)], 1)
            unique_elements = torch.eq(joints_gt_position[:, :2].unsqueeze(1), joint_det[:, :2])
            unique_elements = unique_elements[:, :, 0] & unique_elements[:, :, 1]
            unique_elements = unique_elements.sum(dim=0)
            joint_det = torch.cat([joint_det[unique_elements == 0], joints_gt_position], 0)

            x, edge_attr, edge_index = self._construct_mpn_graph(joint_det, self.features[batch])

            # sol maps nodes to gt joints/  gt joints to nodes and connectivity maps between gt joints
            edge_labels = None
            if self.joints_gt is not None:
                edge_labels = NaiveGraphConstructor._construct_edge_labels(joint_det, self.joints_gt[batch], edge_index)

            return x, edge_attr, edge_index, edge_labels, joint_det

    @staticmethod
    def _construct_edge_labels(joint_det, joints_gt, edge_index):
        # joint_idx_det, joint_y, joint_x = joint_map.nonzero(as_tuple=True)
        # joint_positions_det = torch.stack([joint_x, joint_y], 1)

        num_joints_det = len(joint_det)

        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)

        joints_position_gt = joints_gt[:, :, :2]
        joints_position_gt = joints_position_gt.view(-1, 1, 2).long().float()  # !!! cast to long !!! and then to float
        distances = torch.norm(joint_det[:, :2] - joints_position_gt, dim=2)
        distances = distances.view(30, 17, num_joints_det)  # todo include ref to max number of people
        # set the distances of joint pairse of different types to high cost s.t. they are not matched
        for jt in range(17):
            distances[:, jt, joint_det[:, 2] != jt] = 1000000.0
        cost_mat = distances[person_idx_gt, joint_idx_gt].detach().cpu().numpy()
        sol = linear_sum_assignment(cost_mat)
        cost = np.sum(cost_mat[sol[0], sol[1]])
        print(f"Assignment cost: {cost}")

        # construct edge labels
        # idea: an edge has label 1 if source and destination node are assigned to joint_gt of the same person
        # this results in an fully connected pose graph per person
        num_edges = edge_index.shape[1]  # num_joints_det * num_joints_det - num_joints_det
        # create mapping joint_det -> joint_gt
        # because not all joint_det have a corresponding partner all other joints_dets are mapped to 0
        # this means the position of joint_gt given joint_idx_det is node_to_gt[joint_idx_det] - 1
        node_to_gt = torch.zeros(num_joints_det, dtype=torch.int) - 1
        node_to_gt[sol[1]] = torch.arange(0, num_joints_gt, dtype=torch.int)
        node_to_gt = node_to_gt.long() + 1

        # person_idx_ext_(1/2) map joint_gt_idx to the respective person (node_to_gt maps to joint_gt_idx + 1 and
        # because person_idx_ext_() is shifted one to the right it evens out) node_to_gt maps joint_det without match to 0
        # which gets mapped by person_idx_ext_() to -1/-2 that means that comparing the persons, each joint_det
        # is mapped to, results in no edge for joint_dets without match since -1 != -2 and an edge for joint_dets
        # of same person
        person_idx_ext = torch.zeros(len(person_idx_gt) + 1) - 1
        person_idx_ext[1:] = person_idx_gt
        person_idx_ext_2 = torch.zeros(len(person_idx_gt) + 1) - 2
        person_idx_ext_2[1:] = person_idx_gt
        person_1 = person_idx_ext[node_to_gt[edge_index[0]]]
        person_2 = person_idx_ext_2[node_to_gt[edge_index[1]]]
        edge_labels = torch.where(torch.eq(person_1, person_2), torch.ones(num_edges), torch.zeros(num_edges))

        return edge_labels.to(joint_det.device)


def joint_det_from_scoremap(scoremap, threshold=0.007):
    joint_map = non_maximum_suppression(scoremap, threshold=threshold)
    joint_idx_det, joint_y, joint_x = joint_map.nonzero(as_tuple=True)
    joint_positions_det = torch.stack([joint_x, joint_y, joint_idx_det], 1)
    return joint_positions_det


def graph_cluster_to_persons(joints, joint_connections):
    """
    :param joints: (N, 2) vector of joints
    :param joint_connections: (2, E) array/tensor that indicates which joint are connected thus belong to the same person
    :return: (N persons, 17, 3) array. 17 joints, 2 positions + visibiilty flag (in case joints are missing)
    """
    joints, joint_connections = to_numpy(joints), to_numpy(joint_connections)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    # consturct dense adj matrix
    num_nodes = len(joints)
    adj_matrix = np.zeros([num_nodes, num_nodes])
    adj_matrix[joint_connections[0], joint_connections[1]] = 1
    graph = csr_matrix(adj_matrix)
    n_components, person_labels = connected_components(graph, directed=False, return_labels=True)
    persons = []
    for i in range(n_components):
        # check if cc has more than one node
        person_joints = joints[person_labels == i]
        person_joint_types = person_joints[:, 2]
        if len(person_joints) > 17:
            print("Mutant detected!!")

        if len(person_joints) > 1:  # isolated joints also form a cluster -> ignore them
            # rearrange person joints
            keypoints = np.zeros([17, 3])
            keypoints[person_joint_types, :2] = person_joints[:, :2]
            keypoints[np.sum(keypoints, axis=1) != 0, 2] = 1
            persons.append(keypoints)
    persons = np.array(persons)
    return persons
