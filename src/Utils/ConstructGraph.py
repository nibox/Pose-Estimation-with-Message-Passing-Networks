import torch
import torch_geometric
import torch_geometric.utils as gutils
from scipy.optimize import linear_sum_assignment
import numpy as np
from Utils.Utils import non_maximum_suppression


class NaiveGraphConstructor:

    def __init__(self, scoremaps, features, joints_gt, factor_list, masks, device, use_gt, no_false_positives,
                 use_neighbours, edge_label_method, mask_crowds, detect_threshold, matching_radius,
                 inclusion_radius, mpn_graph_type):
        self.scoremaps = scoremaps.to(device)
        self.features = features.to(device)
        self.joints_gt = joints_gt.to(device)
        self.factor_list = factor_list
        assert factor_list.shape[0] == self.joints_gt.shape[0] and factor_list.shape[1] == self.joints_gt.shape[1]
        self.masks = masks.to(device)
        self.batch_size = scoremaps.shape[0]
        self.device = device
        self.use_gt = use_gt
        self.no_false_positives = no_false_positives
        self.include_neighbouring_keypoints = use_neighbours
        self.edge_label_method = edge_label_method
        self.mask_crowds = mask_crowds
        self.detect_threshold = detect_threshold
        self.matching_radius = matching_radius  # this is for the initial matching
        self.inclusion_radius = inclusion_radius  # this is for the neighbouring matching
        self.mpn_graph_type = mpn_graph_type

    def construct_graph(self):
        x_list, edge_attr_list, edge_index_list, edge_labels_list, joint_det_list = [], [], [], [], []
        label_mask_list = []
        batch_index = []
        num_node_list = [0]
        for batch in range(self.batch_size):
            if self.mask_crowds:
                joint_det = joint_det_from_scoremap(self.scoremaps[batch], threshold=self.detect_threshold,
                                                    mask=self.masks[batch])
            else:
                joint_det = joint_det_from_scoremap(self.scoremaps[batch], threshold=self.detect_threshold)

            # ##############cheating#################
            # extend joint_det with joints_gt in order to have a perfect matching at train time
            # !!! be careufull to use it at test time!!!
            # todo move in function

            # remove detected joints that are close to multiple gt joints
            if self.use_gt:
                person_idx_gt, joint_idx_gt = self.joints_gt[batch, :, :, 2].nonzero(as_tuple=True)
                joints_gt_loc = self.joints_gt[batch, person_idx_gt, joint_idx_gt, :2].round().long().clamp(0, 127)
                joints_gt_loc = torch.cat([joints_gt_loc, joint_idx_gt.unsqueeze(1)], 1)

                joint_det = NaiveGraphConstructor.remove_ambigiuous_det(joint_det, joints_gt_loc, self.inclusion_radius)

                person_idx_gt, joint_idx_gt = self.joints_gt[batch, :, :, 2].nonzero(as_tuple=True)
                tmp = self.joints_gt[batch, person_idx_gt, joint_idx_gt, :2].round().long().clamp(0, 127)
                joints_gt_position = torch.cat([tmp, joint_idx_gt.unsqueeze(1)], 1)
                unique_elements = torch.eq(joints_gt_position[:, :2].unsqueeze(1), joint_det[:, :2])
                unique_elements = unique_elements[:, :, 0] & unique_elements[:, :, 1]
                unique_elements = unique_elements.sum(dim=0)
                joint_det = torch.cat([joint_det[unique_elements == 0], joints_gt_position], 0)
                if self.no_false_positives:
                    joint_det = joints_gt_position

            x, edge_attr, edge_index = self._construct_mpn_graph(joint_det, self.features[batch], self.mpn_graph_type)


            # sol maps nodes to gt joints/  gt joints to nodes and connectivity maps between gt joints
            edge_labels = None
            if self.use_gt:
                if self.edge_label_method == 1:
                    edge_labels = self._construct_edge_labels_1(joint_det, self.joints_gt[batch], edge_index)
                elif self.edge_label_method == 2:
                    edge_labels = self._construct_edge_labels_2(joint_det, self.joints_gt[batch], edge_index)
            elif self.joints_gt is not None:
                assert self.edge_label_method in [3, 4]
                if self.edge_label_method == 3:
                    edge_labels = self._construct_edge_labels_3(joint_det, self.joints_gt[batch], edge_index)
                elif self.edge_label_method == 4:
                    edge_labels = self._construct_edge_labels_4(joint_det, self.joints_gt[batch], self.factor_list[batch], edge_index)

                label_mask = torch.ones_like(edge_labels)
                if edge_labels.max() == 0:
                    label_mask = label_mask - 1
                label_mask_list.append(label_mask)
            x_list.append(x)
            batch_index.append(torch.ones_like(edge_labels, dtype=torch.long) * batch)
            num_node_list.append(x.shape[0] + num_node_list[-1])
            edge_attr_list.append(edge_attr)
            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)
            joint_det_list.append(joint_det)
        # update edge_indices for batching
        for i in range(1, len(x_list)):
            edge_index_list[i] += num_node_list[i]

        x_list = torch.cat(x_list, 0)
        batch_index = torch.cat(batch_index, 0)
        edge_attr_list = torch.cat(edge_attr_list, 0)
        edge_index_list = torch.cat(edge_index_list, 1)
        joint_det_list = torch.cat(joint_det_list, 0)
        if self.joints_gt is not None:
            assert edge_labels_list[0] is not None
            edge_labels_list = torch.cat(edge_labels_list, 0)
        if len(label_mask_list) == 0:
            label_mask_list = None  # preperation for label mask
        else:
            label_mask_list = torch.cat(label_mask_list, 0)

        return x_list, edge_attr_list, edge_index_list, edge_labels_list, joint_det_list, label_mask_list, batch_index

    def _construct_mpn_graph(self, joint_det, features, graph_type):
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

        if graph_type == "fully":
            edge_index = self.fully_connected_mpn_graph(joint_det)
        elif graph_type == "knn":
            edge_index = self.knn_mpn_graph(joint_det)
        elif graph_type == "topk":
            assert self.detect_threshold is None  # for now only this works
            edge_index = self.top_k_mpn_graph(joint_det, k=20)
        else:
            raise NotImplementedError

        # construct edge labels (each connection type is one label
        labels = torch.arange(0, 17 * 17, dtype=torch.long, device=self.device).view(17, 17)
        connection_type = labels[joint_type[edge_index[0]], joint_type[edge_index[1]]]
        """  use this line to remove connections between same type
        same_type_connection_types = torch.arange(0, 17 * 17, 18, dtype=torch.long, device=self.device)
        assert same_type_connection_types.shape[0] == 17
        same_type_connections = torch.eq(connection_type.unsqueeze(1), same_type_connection_types).sum(dim=1)
        edge_index = edge_index.T
        edge_index = edge_index[same_type_connections == 0].T
        # """
        # create edge features
        """  use this line to remove old edge attr
        connection_type = connection_type[same_type_connections == 0]
        connection_label = torch.nn.functional.one_hot(connection_type, num_classes=17 * 17)
        # """
        # connection label 2
        # """
        num_edges = edge_index.shape[1]
        connection_label_2 = torch.zeros(num_edges, 17, dtype=torch.long, device=self.device)
        connection_label_2[list(range(0, num_edges)), joint_type[edge_index[0]]] = 1
        connection_label_2[list(range(0, num_edges)), joint_type[edge_index[1]]] = 1
        # """

        edge_attr_y = joint_y[edge_index[1]] - joint_y[edge_index[0]]
        edge_attr_x = joint_x[edge_index[1]] - joint_x[edge_index[0]]

        edge_attr = torch.cat([edge_attr_x.unsqueeze(1), edge_attr_y.unsqueeze(1), connection_label_2], dim=1).float()

        return x, edge_attr, edge_index

    def knn_mpn_graph(self, joint_det):
        temp = joint_det[:, :2].float()  # nn.knn_graph (cuda) can't handle int64 tensors
        edge_index = torch_geometric.nn.knn_graph(temp, k=50)
        edge_index = gutils.to_undirected(edge_index, len(joint_det))
        edge_index, _ = gutils.remove_self_loops(edge_index)
        return edge_index

    def fully_connected_mpn_graph(self, joint_det):
        num_joints_det = len(joint_det)
        edge_index, _ = gutils.dense_to_sparse(torch.ones([num_joints_det, num_joints_det], dtype=torch.long))
        edge_index = gutils.to_undirected(edge_index, len(joint_det))
        edge_index, _ = gutils.remove_self_loops(edge_index)
        return edge_index.to(self.device)

    def top_k_mpn_graph(self, joint_det, k):
        edge_index = torch.zeros(2, len(joint_det), 17*k, dtype=torch.long, device=self.device)
        edge_index[0] = edge_index[0] + torch.arange(0, len(joint_det), dtype=torch.long, device=self.device)[:, None]
        edge_index = edge_index.reshape(2, -1)
        joint_det = joint_det.float()
        if self.detect_threshold is not None:
            # code assumes that each joint type has same number of detections
            raise NotImplementedError
        _, indices = joint_det[:, 2].sort()
        distance = torch.norm(joint_det[:, None, :2] - joint_det[indices][:, :2], dim=2).view(-1, 40)
        # distance shape is (num_det * 17, num_det_per_type)
        _, top_k_idx = distance.topk(k=k, dim=1, largest=False)
        # top_k_idx shape (num_det * 17, k)
        top_k_idx = top_k_idx.view(len(joint_det), 17, k) + \
                    torch.arange(0, 17, dtype=torch.long, device=self.device)[:, None] * 40
        top_k_idx = top_k_idx.view(-1)
        edge_index[1] = indices[top_k_idx]

        edge_index = gutils.to_undirected(edge_index, len(joint_det))
        edge_index, _ = gutils.remove_self_loops(edge_index)
        return edge_index

    @staticmethod
    def remove_ambigiuous_det(joint_det, joints_gt_loc, radius):
        distances = torch.norm(joint_det[:, :2].unsqueeze(1).float() - joints_gt_loc[:, :2].float(), dim=2)
        # set distances of joint pairs of different type to some high value
        type_det = torch.logical_not(torch.eq(joint_det[:, 2].unsqueeze(1), joints_gt_loc[:, 2]))
        distances[type_det] = 1000.0  # different type joint are exempt
        distances[distances >= radius] = 0.0  # todo set radius and maybe norm
        distances = distances != 0
        joints_to_remove = distances.sum(dim=1) > 1
        joint_det_smaller = joint_det[torch.logical_not(joints_to_remove)]
        return joint_det_smaller

    @staticmethod
    def soft_remove_neighbours(joint_det, joints_gt_loc, radius):
        distances = torch.norm(joint_det[:, :2].unsqueeze(1).float() - joints_gt_loc[:, :2].float(), dim=2)
        # set distances of joint pairs of different type to some high value
        type_det = torch.logical_not(torch.eq(joint_det[:, 2].unsqueeze(1), joints_gt_loc[:, 2]))
        distances[type_det] = 1000.0  # different type joint are exempt
        distances[distances >= radius] = 0.0  # todo set radius and maybe norm
        distances = distances != 0
        joints_to_remove = distances.sum(dim=1) >= 1
        removed_idx = joints_to_remove.int().nonzero(as_tuple=True)[0]
        return removed_idx

    def _construct_edge_labels_1(self, joint_det, joints_gt, edge_index):

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
            distances[distances >= self.inclusion_radius] = 0.0
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
            assignment[:, :-num_joints_gt] = distance[:, :-num_joints_gt] < self.inclusion_radius
            assignment[np.arange(0, num_joints_gt), np.arange(num_joints_det-num_joints_gt, num_joints_det)] = True
        else:
            assignment = distance < -1  # fast way to generate array initialized with false
            assignment[np.arange(0, num_joints_gt), np.arange(num_joints_det-num_joints_gt, num_joints_det)] = True
            assert assignment.sum(dim=1).min() == 1
            # set diagonal to true because it should be true
        if num_joints_det != num_joints_gt and self.use_gt:
            assert assignment.sum(dim=0)[:-num_joints_gt].max() < 2
        # this means that one source node is assigned two target nodes
        # this should not happen for detected keypoints as these cases are removed
        # this happens for gt keypoints in case where the original positions are close and even closer after resizing
        # and rounding
        assert assignment.sum(dim=1).min() >= 1
        # print(f"max num neighbours: {assignment.sum(dim=1).max()}")
        # this means that each target node is assigned at least one source node (which is the corresponding gt node)
        target_joint_gt, source_joint_det = assignment.long().nonzero(as_tuple=True)

        sol = target_joint_gt, source_joint_det

        edge_labels = NaiveGraphConstructor.match_cc(person_idx_gt, joint_det, edge_index, sol)
        return edge_labels

    def _construct_edge_labels_3(self, joint_det, joints_gt, edge_index):
        assert not self.use_gt
        num_joints_det = len(joint_det)

        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)

        distance = torch.norm(joints_gt[person_idx_gt, joint_idx_gt, :2].unsqueeze(1).round().float().clamp(0, 127) - joint_det[:, :2].float(), dim=2)
        # set distance between joints of different type to some high value
        different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
        distance[different_type] = 10000.0
        distance[distance >= self.matching_radius] = 10000.0
        if self.include_neighbouring_keypoints:
            raise NotImplementedError
        else:
            cost_mat = distance.cpu().numpy()
            sol = linear_sum_assignment(cost_mat)
            row, col = sol
            # remove mappings with cost 10000.0
            valid_match = cost_mat[row, col] != 10000.0
            row = row[valid_match]
            col = col[valid_match]
            person_idx_gt = person_idx_gt[row]
            row = np.arange(0, len(row), dtype=np.int64)
            row, col = torch.from_numpy(row).to(self.device), torch.from_numpy(col).to(self.device)
            sol = row, col

        edge_labels = NaiveGraphConstructor.match_cc(person_idx_gt, joint_det, edge_index, sol)
        return edge_labels

    def _construct_edge_labels_4(self, joint_det, joints_gt, factors, edge_index):
        assert not self.use_gt
        num_joints_det = len(joint_det)
        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)

        distance = (joints_gt[person_idx_gt, joint_idx_gt, :2].unsqueeze(1).round().float().clamp(0, 127) - joint_det[:, :2].float()).pow(2).sum(dim=2)
        factor_per_joint = factors[person_idx_gt, joint_idx_gt]
        similarity = torch.exp(-distance / factor_per_joint[:, None])

        different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
        similarity[different_type] = 0.0
        similarity[similarity < self.matching_radius] = 0.0  # 0.1 worked well for threshold + knn graph

        cost_mat = similarity.cpu().numpy()
        sol = linear_sum_assignment(cost_mat, maximize=True)
        row, col = sol
        # remove mappings with cost 0.0
        valid_match = cost_mat[row, col] != 0.0
        row = row[valid_match]
        col = col[valid_match]
        person_idx_gt = person_idx_gt[row]

        row_1 = np.arange(0, len(row), dtype=np.int64)
        row_1, col_1 = torch.from_numpy(row_1).to(self.device), torch.from_numpy(col).to(self.device)
        if self.include_neighbouring_keypoints:
            # use inclusion radius to filter more agressivly
            cost_mat[cost_mat < self.inclusion_radius] = 0.0
            # remove already chosen keypoints from next selection
            cost_mat[np.arange(0, num_joints_gt).reshape(-1, 1), col] = 0.0
            # "remove" ambiguous cases
            # identify them
            ambiguous_dets = (cost_mat != 0.0).sum(axis=0) > 1.0
            # remove them
            cost_mat[np.arange(0, num_joints_gt).reshape(-1, 1), ambiguous_dets] = 0.0
            row_2, col_2 = np.nonzero(cost_mat)
            assert (set(list(row_2)).issubset(set(list(row))))

            # some gt joints have no match and have to be removed for the next step
            # create translation table for new indices in order to translate row_2
            mapping = np.zeros(num_joints_gt, dtype=np.int64) - 1
            mapping[row] = np.arange(0, len(row), dtype=np.int64)
            row_2 = mapping[row_2]
            assert (row_2 == -1).sum() == 0
            row_2, col_2 = torch.from_numpy(row_2).to(self.device), torch.from_numpy(col_2).to(self.device)

            row_1 = torch.cat([row_1, row_2], dim=0)
            col_1 = torch.cat([col_1, col_2], dim=0)

        sol = row_1, col_1

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

        # person_idx_ext_(1/2) map joint_gt_idx to the respective person (node_to_gt maps to joint_gt_idx + 1 and
        # because person_idx_ext_() is shifted one to the right it evens out) node_to_gt maps joint_det without match
        # to 0 which gets mapped by person_idx_ext_() to -1/-2 that means that comparing the persons, each joint_det
        # is mapped to, results in no edge for joint_dets without match since -1 != -2 and an edge for joint_dets of
        # same person
        cluster_idx_ext_1 = torch.zeros(len(a_to_clusters) + 1, device=device) - 1
        cluster_idx_ext_2 = torch.zeros(len(a_to_clusters) + 1, device=device) - 2
        cluster_idx_ext_1[1:] = a_to_clusters
        cluster_idx_ext_2[1:] = a_to_clusters
        b_to_clusters_1= cluster_idx_ext_1[edges_b_to_a[edges_b[0]]]
        b_to_clusters_2 = cluster_idx_ext_2[edges_b_to_a[edges_b[1]]]
        edge_labels = torch.where(torch.eq(b_to_clusters_1, b_to_clusters_2), torch.ones(num_edges_b, device=device),
                                  torch.zeros(num_edges_b, device=device))
        return edge_labels

    @staticmethod
    def create_loss_mask(joints, edge_index):
        """
        Given a list os joint idx whose connections should not contribute to the loss and the edges, create the
        loss mask
        :param joints: tensor (Num joints to consider) contains joint idxs
        :param edge_index: edges of graph
        :return: loss mask (same shape as edge labels)
        """
        loss_mask = torch.ones(edge_index.shape[1], dtype=torch.float, device=edge_index.device)
        # mask all edges starting at points from joints
        source_edges = torch.eq(edge_index[0].unsqueeze(1), joints).sum(dim=1)
        assert source_edges.max() <= 1.0
        source_edges = source_edges == 1.0

        target_edges = torch.eq(edge_index[1].unsqueeze(1), joints).sum(dim=1)
        assert target_edges.max() <= 1.0
        target_edges = target_edges == 1.0
        loss_mask[source_edges | target_edges] = 0.0
        return loss_mask


def joint_det_from_scoremap(scoremap, threshold=0.007, mask=None):
    joint_map = non_maximum_suppression(scoremap, threshold=threshold)
    if mask is not None:
        joint_map = joint_map * mask.unsqueeze(0)
    scoremap = scoremap * joint_map
    if threshold is not None:
        scoremap = torch.where(scoremap < threshold, torch.zeros_like(scoremap), scoremap)
        joint_idx_det, joint_y, joint_x = scoremap.nonzero(as_tuple=True)
    else:
        k = 40
        scoremap_shape = scoremap.shape
        _, indices = scoremap.view(17, -1).topk(k=k, dim=1)
        container = torch.zeros_like(scoremap, device=scoremap.device, dtype=torch.int).reshape(17, -1)
        container[np.arange(0, 17).reshape(17, 1), indices] = 1
        container = container.reshape(scoremap_shape)
        joint_idx_det, joint_y, joint_x = container.nonzero(as_tuple=True)
        assert len(joint_idx_det) == k * 17

    joint_positions_det = torch.stack([joint_x, joint_y, joint_idx_det], 1)
    return joint_positions_det
