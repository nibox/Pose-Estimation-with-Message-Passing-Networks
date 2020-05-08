import torch
import torch_geometric
import torch_geometric.utils as gutils
from scipy.optimize import linear_sum_assignment
import numpy as np
from Utils.Utils import *


class NaiveGraphConstructor:

    def __init__(self, scoremaps, features, joints_gt, masks, device, use_gt=True, no_false_positives=False,
                 use_neighbours=False, edge_label_method=1, mask_crowds=False, detect_threshold=0.007):
        self.scoremaps = scoremaps.to(device)
        self.features = features.to(device)
        self.joints_gt = joints_gt.to(device)
        self.masks = masks.to(device)
        self.batch_size = scoremaps.shape[0]
        self.device = device
        self.use_gt = use_gt
        self.no_false_positives = no_false_positives
        self.include_neighbouring_keypoints = use_neighbours
        self.edge_label_method = edge_label_method
        self.mask_crowds = mask_crowds
        self.detect_threshold = detect_threshold

    def _construct_mpn_graph(self, joint_det, features):
        """

        :param joint_det: Shape: (Num_dets, 3) x, y, type,
        :param features: Shape: (C, H, W)
        :return: x (Num_dets, feature_dim), edge_attr (Num_edges, feature_dim), edge_index (2, Num_dets)
        """

        joint_x = joint_det[:, 0]
        joint_y = joint_det[:, 1]
        joint_type = joint_det[:, 2]
        joint_features = features[:, joint_y, joint_x]
        num_joints_det = len(joint_x)

        # construct node features
        x = joint_features.T

        # construct joint_det graph (fully connected)
        # edge_index, _ = gutils.dense_to_sparse(torch.ones([num_joints_det, num_joints_det], dtype=torch.long))
        # todo using k_nn and setting distances between joints of certain type on can create the different graphs
        # todo remove connections between same type
        temp = joint_det[:, :2].float()  # nn.knn_graph (cuda) can't handle int64 tensors
        edge_index = torch_geometric.nn.knn_graph(temp, 50)
        edge_index = gutils.to_undirected(edge_index)
        edge_index, _ = gutils.remove_self_loops(edge_index)

        # construct edge labels (each connection type is one label
        labels = torch.arange(0, 17 * 17, dtype=torch.long, device=self.device).view(17, 17)
        connection_type = labels[joint_type[edge_index[0]], joint_type[edge_index[1]]]
        # remove connections between same type
        same_type_connection_types = torch.arange(0, 17 * 17, 18, dtype=torch.long, device=self.device)
        assert same_type_connection_types.shape[0] == 17
        same_type_connections = torch.eq(connection_type.unsqueeze(1), same_type_connection_types).sum(dim=1)
        edge_index = edge_index.T
        edge_index = edge_index[same_type_connections == 0].T
        # create edge features
        """
        connection_type = connection_type[same_type_connections == 0]
        connection_label = torch.nn.functional.one_hot(connection_type, num_classes=17 * 17)
        """
        # connection label 2
        num_edges = edge_index.shape[1]
        connection_label_2 = torch.zeros(num_edges, 17, dtype=torch.long, device=self.device)
        connection_label_2[list(range(0, num_edges)), joint_type[edge_index[0]]] = 1
        connection_label_2[list(range(0, num_edges)), joint_type[edge_index[1]]] = 1

        edge_attr_y = joint_y[edge_index[1]] - joint_y[edge_index[0]]
        edge_attr_x = joint_x[edge_index[1]] - joint_x[edge_index[0]]

        edge_attr = torch.cat([edge_attr_x.unsqueeze(1), edge_attr_y.unsqueeze(1), connection_label_2], dim=1).float()

        return x, edge_attr, edge_index

    def construct_graph(self):
        x_list, edge_attr_list, edge_index_list, edge_labels_list, joint_det_list = [], [], [], [], []
        num_node_list = [0]
        for batch in range(self.batch_size):
            if self.mask_crowds:
                joint_det = joint_det_from_scoremap(self.scoremaps[batch], threshold=self.detect_threshold, mask=self.masks[batch])
            else:
                joint_det = joint_det_from_scoremap(self.scoremaps[batch], threshold=self.detect_threshold)


            # joint_map = non_maximum_suppression(self.scoremaps[batch], threshold=0.007)
            # print(f"gt kp: {num_joints_gt}, d kp: {num_joints_det}")

            # ##############cheating#################
            # extend joint_det with joints_gt in order to have a perfect matching at train time
            # !!! be careufull to use it at test time!!!
            # todo move in function

            # remove detected joints that are close to multiple gt joints
            person_idx_gt, joint_idx_gt = self.joints_gt[batch, :, :, 2].nonzero(as_tuple=True)
            joints_gt_loc = self.joints_gt[batch, person_idx_gt, joint_idx_gt, :2].round().long().clamp(0, 127)
            joints_gt_loc = torch.cat([joints_gt_loc, joint_idx_gt.unsqueeze(1)], 1)

            joint_det = self.remove_ambigiuous_det(joint_det, joints_gt_loc)

            if self.use_gt:
                person_idx_gt, joint_idx_gt = self.joints_gt[batch, :, :, 2].nonzero(as_tuple=True)
                tmp = self.joints_gt[batch, person_idx_gt, joint_idx_gt, :2].round().long().clamp(0, 127)
                joints_gt_position = torch.cat([tmp, joint_idx_gt.unsqueeze(1)], 1)
                unique_elements = torch.eq(joints_gt_position[:, :2].unsqueeze(1), joint_det[:, :2])
                unique_elements = unique_elements[:, :, 0] & unique_elements[:, :, 1]
                unique_elements = unique_elements.sum(dim=0)
                joint_det = torch.cat([joint_det[unique_elements == 0], joints_gt_position], 0)
                if self.no_false_positives:
                    joint_det = joints_gt_position

            x, edge_attr, edge_index = self._construct_mpn_graph(joint_det, self.features[batch])

            # sol maps nodes to gt joints/  gt joints to nodes and connectivity maps between gt joints
            edge_labels = None
            if self.joints_gt is not None:
                if self.edge_label_method == 1:
                    edge_labels = self._construct_edge_labels_1(joint_det, self.joints_gt[batch], edge_index)
                elif self.edge_label_method == 2:
                    edge_labels = self._construct_edge_labels_2(joint_det, self.joints_gt[batch], edge_index)
            x_list.append(x)
            num_node_list.append(x.shape[0] + num_node_list[-1])
            edge_attr_list.append(edge_attr)
            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)
            joint_det_list.append(joint_det)
        # update edge_indices for batching
        for i in range(1, len(x_list)):
            edge_index_list[i] += num_node_list[i]

        x_list = torch.cat(x_list, 0)
        edge_attr_list = torch.cat(edge_attr_list, 0)
        edge_index_list = torch.cat(edge_index_list, 1)
        edge_labels_list = torch.cat(edge_labels_list, 0)
        joint_det_list = torch.cat(joint_det_list, 0)

        return x_list, edge_attr_list, edge_index_list, edge_labels_list, joint_det_list

    def remove_ambigiuous_det(self, joint_det, joints_gt_loc):
        distances = torch.norm(joint_det[:, :2].unsqueeze(1).float() - joints_gt_loc[:, :2].float(), dim=2)
        # set distances of joint pairs of different type to some high value
        type_det = torch.logical_not(torch.eq(joint_det[:, 2].unsqueeze(1), joints_gt_loc[:, 2]))
        distances[type_det] = 1000.0  # different type joint are exempt
        distances[distances >= 5] = 0.0  # todo set radius and maybe norm
        distances = distances != 0
        joints_to_remove = distances.sum(dim=1) > 1
        joint_det = joint_det[torch.logical_not(joints_to_remove)]
        return joint_det

    def _construct_edge_labels_1(self, joint_det, joints_gt, edge_index):
        # joint_idx_det, joint_y, joint_x = joint_map.nonzero(as_tuple=True)
        # joint_positions_det = torch.stack([joint_x, joint_y], 1)

        num_joints_det = len(joint_det)

        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)
        num_edges = edge_index.shape[1]  # num_joints_det * num_joints_det - num_joints_det

        joints_position_gt = joints_gt[:, :, :2]
        joints_position_gt = joints_position_gt.view(-1, 1, 2).round().float()  # !!! cast to long !!! and then to float
        distances = torch.norm(joint_det[:, :2] - joints_position_gt, dim=2)
        distances = distances.view(30, 17, num_joints_det)  # todo include ref to max number of people
        # set the distances of joint pairse of different types to high cost s.t. they are not matched
        for jt in range(17):
            distances[:, jt, joint_det[:, 2] != jt] = 1000000.0
        cost_mat = distances[person_idx_gt, joint_idx_gt].detach().cpu().numpy()
        sol = linear_sum_assignment(cost_mat)
        cost = np.sum(cost_mat[sol[0], sol[1]])

        sol_torch = torch.stack(list(map(torch.from_numpy, sol)), dim=1).to(joint_det.device).T
        edge_labels = NaiveGraphConstructor.match_cc(person_idx_gt, joint_det, edge_index, sol_torch)

        if self.include_neighbouring_keypoints:
            source_nodes_part_of_body = edge_index[0, edge_labels == 1]
            target_nodes_part_of_body = edge_index[1, edge_labels == 1]

            distances = torch.norm(joint_det[:, :2].unsqueeze(1).float() - joint_det[:, :2].float(), dim=2)
            # set distances of joint pairs of different type to some high value
            type_det = torch.logical_not(torch.eq(joint_det[:, 2].unsqueeze(1), joint_det[:, 2]))
            distances[type_det] = 1000.0
            distances[:, source_nodes_part_of_body] = 1000.0
            distances[distances >= 5] = 0.0
            distances = distances != 0
            # remove ambiguous cases here
            distances_tmp = distances[source_nodes_part_of_body.unique(sorted=True)]
            mult_assignments = distances_tmp.sum(dim=0)
            distances = distances[source_nodes_part_of_body]
            distances[:, mult_assignments > 1] = False

            target_idxs, source_idxs = distances.long().nonzero(as_tuple=True)
            new_target_nodes = target_nodes_part_of_body[target_idxs]
            # at this point i have new edges from new canditates to old keypoints and label constructing
            # but i would have to remvoe duplicates edges from edge_index
            # so goal is to identifiy the indices of these duplicates
            edge_to_label_1 = torch.stack([source_idxs, new_target_nodes])
            edge_to_label_2 = torch.stack([new_target_nodes, source_idxs])
            edge_to_label = torch.cat([edge_to_label_1, edge_to_label_2], 1)
            edge_comparision = torch.eq(edge_index.T.unsqueeze(1), edge_to_label.T)
            edge_comparision = edge_comparision[:, :, 0] & edge_comparision[:, :, 1]
            duplicate_edges = edge_comparision.sum(dim=1)
            existing_edges = edge_comparision.sum(dim=0)
            edge_labels_2 = torch.zeros(num_edges, device=joint_det.device)
            edge_labels_2[duplicate_edges == 1] = 1.0

            edge_labels += edge_labels_2.to(joint_det.device)
        return edge_labels

    def _construct_edge_labels_2(self, joint_det, joints_gt, edge_index):
        assert self.use_gt
        num_joints_det = len(joint_det)

        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)

        distance = torch.norm(joints_gt[person_idx_gt, joint_idx_gt, :2].unsqueeze(1).round().float().clamp(0, 127) - joint_det[:, :2].float(), dim=2)
        # set distance between joints of different type to some high value
        different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
        distance[different_type] = 1000.0
        if self.include_neighbouring_keypoints:
            assignment = distance < -1  # fast way to generate array initialized with false
            assignment[:, :-num_joints_gt] = distance[:, :-num_joints_gt] < 5.0
            assignment[np.arange(0, num_joints_gt), np.arange(num_joints_det-num_joints_gt, num_joints_det)] = True
        else:
            assignment = distance < -1  # fast way to generate array initialized with false
            assignment[np.arange(0, num_joints_gt), np.arange(num_joints_det-num_joints_gt, num_joints_det)] = True
            # set diagonal to true because it should be true
        assert assignment.sum(dim=0)[:-num_joints_gt].max() < 2
        # this means that one source node is assigned two target nodes
        # this should not happen for detected keypoints as these cases are removed
        # this happens for gt keypoints in case where the original positions are close and even closer after resizing
        # and rounding
        assert assignment.sum(dim=1).min() >= 1
        # this means that each target node is assigned at least one source node (which is the corresponding gt node)
        target_joint_gt, source_joint_det = assignment.long().nonzero(as_tuple=True)

        sol = target_joint_gt, source_joint_det

        edge_labels = NaiveGraphConstructor.match_cc(person_idx_gt, joint_det, edge_index, sol)
        return edge_labels

    @staticmethod
    def match_cc(a_to_clusters, nodes_b, edges_b, edges_a_to_b):
        """
        Given two graph A and B, with A consisting of k connected components. Given a biparite matching between all
        nodes in A and a subset B' of nodes in B, chose edges E' in B using the given mapping between nodes of A and B'
         s.t.the connected components of A are reconstructed in B. I.e nodes in B that are connected by edges from E'
         should have corresponding nodes in A that are part of the same connected component.
        :param a_to_clusters: Mapping of nodes from A to the corresponding connected component (cluster)
        :param nodes_b: List of nodes of B
        :param edges_b: List of all edges of B
        :param edges_a_to_b: matching between A and B'
        :return: list of labels indicating wether edge in edges_b belongs to E'
        """
        device = a_to_clusters.device
        num_nodes_b = len(nodes_b)
        num_edges_b = edges_b.shape[1]  # num_joints_det * num_joints_det - num_joints_det
        # construct edge labels
        # idea: an edge has label 1 if source and destination node are assigned to joint_gt of the same person
        # this results in an fully connected pose graph per person
        # create mapping nodes_b -> nodes_a (for EACH node in nodes_b)
        # because not all joint_det have a corresponding partner all other joints_dets are mapped to 0
        # this means the position of joint_gt given joint_idx_det is node_to_gt[joint_idx_det] - 1
        edges_b_to_a = torch.zeros(num_nodes_b, dtype=torch.long, device=device) - 1
        edges_b_to_a[edges_a_to_b[1]] = edges_a_to_b[0]
        edges_b_to_a = edges_b_to_a + 1

        person_idx_ext = torch.zeros(len(person_idx_gt) + 1, device=self.device) - 1
        person_idx_ext[1:] = person_idx_gt
        person_idx_ext_2 = torch.zeros(len(person_idx_gt) + 1, device=self.device) - 2
        person_idx_ext_2[1:] = person_idx_gt
        person_1 = person_idx_ext[node_to_gt[edge_index[0]]]
        person_2 = person_idx_ext_2[node_to_gt[edge_index[1]]]
        edge_labels = torch.where(torch.eq(person_1, person_2), torch.ones(num_edges, device=self.device),
                                  torch.zeros(num_edges, device=self.device))
        return edge_labels


def joint_det_from_scoremap(scoremap, threshold=0.007, mask=None):
    joint_map = non_maximum_suppression(scoremap, threshold=threshold)
    if mask is not None:
        joint_map = joint_map * mask.unsqueeze(0)
    scoremap = scoremap * joint_map
    if threshold is not None:
        scoremap = torch.where(scoremap < threshold, torch.zeros_like(scoremap), scoremap)
        joint_idx_det, joint_y, joint_x = scoremap.nonzero(as_tuple=True)
    else:
        scoremap_shape = scoremap.shape
        _, indices = scoremap.view(17, -1).topk(k=30, dim=1)
        container = torch.zeros_like(scoremap, device=scoremap.device, dtype=torch.int).reshape(17, -1)
        container[:, indices] = 1
        container = container.reshape(scoremap_shape)
        joint_idx_det, joint_y, joint_x = container.nonzero(as_tuple=True)

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
    # construct dense adj matrix
    num_nodes = len(joints)
    adj_matrix = np.zeros([num_nodes, num_nodes])
    adj_matrix[joint_connections[0], joint_connections[1]] = 1
    graph = csr_matrix(adj_matrix)
    n_components, person_labels = connected_components(graph, directed=False, return_labels=True)
    persons = []
    mutant_detected = False
    for i in range(n_components):
        # check if cc has more than one node
        person_joints = joints[person_labels == i]
        if len(person_joints) > 17:
            # print(f"Mutant detected!! It has {len(person_joints)} joints!!")
            # todo change meaning of mutant
            mutant_detected = True

        if len(person_joints) > 1:  # isolated joints also form a cluster -> ignore them
            # rearrange person joints
            """
            keypoints = np.zeros([17, 3])
            keypoints[person_joint_types, :2] = person_joints[:, :2]
            keypoints[np.sum(keypoints, axis=1) != 0, 2] = 1
            keypoints[:, 2] = 0
            """

            keypoints = np.zeros([17, 3])
            for joint_type in range(17):  # 17 different joint types
                # take the detected joints of a certain type
                person_joint_for_type = person_joints[person_joints[:, 2] == joint_type]
                if len(person_joint_for_type) != 0:
                    keypoints[joint_type] = np.mean(person_joint_for_type, axis=0)
            keypoints[np.sum(keypoints, axis=1) != 0, 2] = 1
            keypoints[keypoints[:, 2] == 0, :2] = keypoints[keypoints[:, 2] != 0, :2].mean(axis=0)
            persons.append(keypoints)
            # print(test)
    persons = np.array(persons)
    return persons, mutant_detected
