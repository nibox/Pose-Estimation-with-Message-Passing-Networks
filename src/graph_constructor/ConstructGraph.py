import torch
import torch_geometric
import torch_geometric.utils as gutils
from scipy.optimize import linear_sum_assignment
import numpy as np
from Utils.Utils import non_maximum_suppression, subgraph_mask


class NaiveGraphConstructor:

    def __init__(self, scoremaps, tagmaps, features, joints_gt, factor_list, masks, device, config, testing, heatmaps, num_joints):
        self.scoremaps = scoremaps.to(device)
        self.tagmaps = tagmaps.to(device)
        self.heatmaps = heatmaps[1].to(device).float() if heatmaps is not None else None
        self.features = features.to(device)
        self.joints_gt = joints_gt.to(device) if joints_gt is not None else None
        self.factor_list = factor_list
        self.masks = masks.to(device) if masks is not None else None
        self.batch_size = scoremaps.shape[0]
        self.device = device
        self.num_joints = num_joints

        self.use_gt = config.USE_GT
        self.no_false_positives = config.CHEAT
        self.edge_label_method = config.EDGE_LABEL_METHOD
        self.include_neighbouring_keypoints = config.USE_NEIGHBOURS
        self.mask_crowds = config.MASK_CROWDS
        self.detect_threshold = config.DETECT_THRESHOLD if config.DETECT_THRESHOLD <= 1.5 else None
        self.hybrid_k = config.HYBRID_K
        self.matching_radius = config.MATCHING_RADIUS  # this is for the initial matching
        self.inclusion_radius = config.INCLUSION_RADIUS  # this is for the neighbouring matching
        self.mpn_graph_type = config.GRAPH_TYPE
        self.normalize_node_distance = config.NORM_NODE_DISTANCE
        self.image_centric_sampling = config.IMAGE_CENTRIC_SAMPLING
        self.edge_features_to_use = config.EDGE_FEATURES_TO_USE
        self.pool_kernel_size = config.POOL_KERNEL_SIZE

        self.node_dropout = config.NODE_DROPOUT if config.NODE_DROPOUT != 0.0 else None
        self.use_weighted_class_loss = config.WEIGHT_CLASS_LOSS
        self.testing = testing

        self.node_matching_radius = config.NODE_MATCHING_RADIUS
        self.node_inclusion_radius = config.NODE_INCLUSION_RADIUS
        self.with_background_class = config.WITH_BACKGROUND

    def construct_graph(self):
        x_list, edge_attr_list, edge_index_list, edge_labels_list, joint_det_list = [], [], [], [], []
        joint_score_list = []
        node_label_list = []
        node_class_list = []
        node_persons_list = []
        label_mask_list = []
        label_mask_node_list = []
        class_mask_list = []
        joint_tag_list = []
        batch_index = []
        num_node_list = [0]
        for batch in range(self.batch_size):
            if self.mask_crowds:
                joint_det, joint_scores = joint_det_from_scoremap(self.scoremaps[batch], self.num_joints,
                                                                  threshold=self.detect_threshold,
                                                                  pool_kernel=self.pool_kernel_size,
                                                                  mask=self.masks[batch], hybrid_k=self.hybrid_k)
            else:
                joint_det, joint_scores = joint_det_from_scoremap(self.scoremaps[batch], self.num_joints,
                                                                  threshold=self.detect_threshold,
                                                                  pool_kernel=self.pool_kernel_size,
                                                                  hybrid_k=self.hybrid_k)

            # ##############cheating#################
            # extend joint_det with joints_gt in order to have a perfect matching at train time
            # !!! be careufull to use it at test time!!!
            # todo move in function

            # remove detected joints that are close to multiple gt joints
            if self.use_gt:
                person_idx_gt, joint_idx_gt = self.joints_gt[batch, :, :, 2].nonzero(as_tuple=True)
                clamp_max = max(self.scoremaps.shape[2], self.scoremaps.shape[3]) - 1
                joints_gt_loc = self.joints_gt[batch, person_idx_gt, joint_idx_gt, :2].round().long().clamp(0,
                                                                                                            clamp_max)
                joints_gt_position = torch.cat([joints_gt_loc, joint_idx_gt.unsqueeze(1)], 1)
                if len(joints_gt_loc) >= 2:
                    # idea is that for images without present gt joint (happens due to crop and masking), i use
                    # the detected joints as dummy input but because there are not gt joint the labels are all
                    # inactive and the loss of the image should be masked out. It is a bit hacky but it works
                    joint_det = joints_gt_position
                    joint_scores = torch.ones(len(joint_det), dtype=torch.float, device=joint_det.device)
            elif self.edge_label_method == 7 and not self.testing:
                person_idx_gt, joint_idx_gt = self.joints_gt[batch, :, :, 2].nonzero(as_tuple=True)
                clamp_max = max(self.scoremaps.shape[2], self.scoremaps.shape[3]) - 1
                tmp = self.joints_gt[batch, person_idx_gt, joint_idx_gt, :2].round().long()
                tmp += torch.randint(-2, 3, (tmp.shape[0], 2), device=self.device)
                tmp = tmp.clamp(0, clamp_max)
                joints_gt_position = torch.cat([tmp, joint_idx_gt.unsqueeze(1)], 1)

                joint_det = torch.cat([joint_det, joints_gt_position], 0)
                joint_scores = self.scoremaps[batch][joint_det[:, 2], joint_det[:, 1], joint_det[:, 0]]
                # joint_scores[-len(joints_gt_position):] = 1.0

            x, edge_attr, edge_index = self._construct_mpn_graph(joint_det, self.tagmaps[batch].squeeze(), self.features[batch],
                                                                 self.mpn_graph_type, joint_scores,
                                                                 self.edge_features_to_use)
            joint_tags = self.tagmaps[batch, joint_det[:, 2], joint_det[:, 1], joint_det[:, 0]]

            # sol maps nodes to gt joints/  gt joints to nodes and connectivity maps between gt joints
            edge_labels = None
            node_labels = None
            node_classes = None
            node_persons = None
            # label_mask_node = torch.ones(joint_det.shape[0], dtype=torch.float, device=self.device)
            label_mask_node = None
            label_mask = None
            class_mask = None
            if self.use_gt:
                if self.edge_label_method == 1:
                    edge_labels, label_mask = self._construct_edge_labels_1(joint_det, self.joints_gt[batch],
                                                                self.factor_list[batch], edge_index)
                elif self.edge_label_method == 2:
                    edge_labels = self._construct_edge_labels_2(joint_det, self.joints_gt[batch], edge_index)
                label_mask = torch.ones_like(edge_labels, device=self.device, dtype=torch.float)
                if edge_labels.max() == 0:
                    label_mask = torch.zeros_like(label_mask, device=self.device, dtype=torch.float)
            elif self.joints_gt is not None:
                assert self.edge_label_method in [3, 4, 5, 6, 7]
                label_mask_node = torch.ones(joint_det.shape[0], dtype=torch.float, device=self.device)

                if self.edge_label_method == 3:
                    edge_labels, node_persons, label_mask = self._construct_edge_labels_3(joint_det, self.joints_gt[batch],
                                                                                          self.factor_list[batch], edge_index)
                elif self.edge_label_method == 4:
                    edge_labels, node_labels, node_persons, label_mask = self._construct_edge_labels_4(joint_det.detach(),
                                                                                         self.joints_gt[batch],
                                                                                         self.factor_list[batch],
                                                                                         edge_index.detach())
                elif self.edge_label_method == 5:
                    edge_labels, node_labels, label_mask, label_mask_node = self._construct_edge_labels_5(
                        joint_det.detach(), self.joints_gt[batch], self.factor_list[batch], edge_index.detach())
                elif self.edge_label_method == 6:
                    edge_labels, node_labels, node_classes, node_persons, label_mask, label_mask_node, class_mask= self._construct_edge_labels_6(
                        joint_det.detach(), self.joints_gt[batch], self.factor_list[batch], edge_index.detach())
                elif self.edge_label_method == 7:
                    edge_labels, node_labels, node_classes, node_persons, label_mask = self._construct_edge_labels_7(
                        joint_det.detach(), self.joints_gt[batch], self.factor_list[batch], edge_index.detach())

                if edge_labels.max() == 0:
                    label_mask = torch.zeros_like(label_mask, device=self.device, dtype=torch.float)

                if self.edge_label_method != 5 and not (self.edge_label_method == 6 and self.include_neighbouring_keypoints):
                    # masks should only be used for edge_label_method == 6
                    assert label_mask_node.sum() == label_mask_node.shape[0]
                # node dropout
                if self.node_dropout is not None and not self.testing:
                    node_mask = torch.ones_like(node_labels, dtype=torch.float, device=self.device) * self.node_dropout
                    rnd = torch.bernoulli(node_mask)
                    node_mask = (rnd * node_labels) == 0.0  # if zero then keep
                    mask = subgraph_mask(node_mask, edge_index)
                    edge_index, _ = gutils.subgraph(node_mask, edge_index, relabel_nodes=True)
                    joint_det = joint_det[node_mask]
                    x = x[node_mask]
                    node_labels = node_labels[node_mask] if node_labels is not None else None
                    label_mask_node = label_mask_node[node_mask]
                    node_classes = node_classes[node_mask] if node_classes is not None else None
                    node_persons = node_persons[node_mask] if node_persons is not None else None
                    joint_scores = joint_scores[node_mask]

                    edge_attr = edge_attr[mask]
                    edge_labels = edge_labels[mask]
                    label_mask = label_mask[mask]

                # create class labels
                if self.use_weighted_class_loss and class_mask is not None:
                    weights = self.heatmaps[batch, node_classes, joint_det[:, 1], joint_det[:, 2]]

                    weights = torch.where(weights < 0.1, torch.ones_like(weights, device=weights.device) * 0.1,
                                          weights)
                    class_mask = weights * class_mask

            # old label code
            # node_labels_2 = torch.zeros(joint_det.shape[0], device=joint_det.device, dtype=torch.float32)
            # node_labels_2[edge_index[0][edge_labels == 1]] = 1.0

            if self.image_centric_sampling:
                num_pos = int(node_labels.sum().item())
                idx_to_take = node_labels == 1.0

                num_pos = 20 if num_pos == 0 else num_pos
                idxs = torch.arange(0, node_labels.size(0))[idx_to_take == False]
                idxs = idxs[torch.randperm(node_labels.shape[0] - num_pos)[:num_pos * 3]]
                idx_to_take[idxs] = True
                edge_index, edge_attr_2 = gutils.subgraph(idx_to_take, edge_index, torch.cat(
                    [edge_attr, edge_labels[:, None], label_mask[:, None]], dim=1), relabel_nodes=True)
                x = x[idx_to_take]
                edge_attr, edge_labels, label_mask = edge_attr_2[:, :19], edge_attr_2[:, 19], edge_attr_2[:, 20]
                node_labels = node_labels[idx_to_take]
                joint_det = joint_det[idx_to_take]
                joint_scores = joint_scores[idx_to_take]
                knn = True
                if knn:
                    x, edge_attr, edge_index = self._construct_mpn_graph(joint_det, None, self.features[batch], "knn",
                                                                         joint_scores, )
                    edge_labels, node_labels, label_mask = self._construct_edge_labels_4(joint_det.detach(),
                                                                                         self.joints_gt[batch],
                                                                                         self.factor_list[batch],
                                                                                         edge_index.detach())

            x_list.append(x)
            batch_index.append(torch.ones(joint_det.shape[0], dtype=torch.long, device=self.device) * batch)
            num_node_list.append(x.shape[0] + num_node_list[-1])
            edge_attr_list.append(edge_attr)
            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)
            joint_det_list.append(joint_det)
            joint_score_list.append(joint_scores)
            node_label_list.append(node_labels)
            node_class_list.append(node_classes)
            node_persons_list.append(node_persons)
            label_mask_list.append(label_mask)
            label_mask_node_list.append(label_mask_node)
            class_mask_list.append(class_mask)
            joint_tag_list.append(joint_tags)
        # update edge_indices for batching
        for i in range(1, len(x_list)):
            edge_index_list[i] += num_node_list[i]

        x_list = torch.cat(x_list, 0)
        batch_index = torch.cat(batch_index, 0)
        edge_attr_list = torch.cat(edge_attr_list, 0)
        edge_index_list = torch.cat(edge_index_list, 1)
        joint_det_list = torch.cat(joint_det_list, 0)
        joint_score_list = torch.cat(joint_score_list, 0)
        joint_tag_list = torch.cat(joint_tag_list, 0)
        if self.joints_gt is not None:
            assert edge_labels_list[0] is not None
            edge_labels_list = torch.cat(edge_labels_list, 0)
            node_label_list = torch.cat(node_label_list, 0) if self.edge_label_method in [4, 6, 7] else None
            label_mask_list = torch.cat(label_mask_list, 0)
            label_mask_node_list = torch.cat(label_mask_node_list, 0) if self.edge_label_method in [4, 6, 7] else None
            class_mask_list = torch.cat(class_mask_list, 0) if self.edge_label_method in [6, 7] else None
            node_class_list = torch.cat(node_class_list, 0) if self.edge_label_method in [6, 7] else None
            keypoint_positions = None
            node_persons_list = torch.cat(node_persons_list, 0) if self.edge_label_method in [4, 6, 7] else None

        else:
            edge_labels_list, node_label_list, label_mask_list, label_mask_node_list, class_mask_list = None, None, None, None, None
            node_class_list = None
            keypoint_positions = None

        return x_list, edge_attr_list, edge_index_list, edge_labels_list, node_label_list, node_class_list, keypoint_positions, \
               joint_det_list, label_mask_list, label_mask_node_list, class_mask_list, joint_score_list, batch_index, node_persons_list, joint_tag_list

    def _construct_mpn_graph(self, joint_det, tag_maps, features, graph_type, joint_scores, edge_features_to_use):
        """

        :param tag_maps:
        :param joint_scores:
        :param joint_det: Shape: (Num_dets, 3) x, y, type,
        :param features: Shape: (C, H, W)
        :return: x (Num_dets, feature_dim), edge_attr (Num_edges, feature_dim), edge_index (2, Num_dets)
        """
        assert len(edge_features_to_use) >= 1

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
            edge_index = self.top_k_mpn_graph(joint_det, k=10)
        elif graph_type == "feature_knn":
            edge_index = self.feature_knn_mpn_graph(joint_det, x)
        elif graph_type == "score_based":
            k = 75
            edge_index = self.score_based_graph(joint_det, joint_scores, k)
        elif graph_type == "score_based_per_type":
            edge_index = self.score_based_k_per_type(joint_det, joint_scores)
        else:
            raise NotImplementedError

        # construct edge labels (each connection type is one label
        labels = torch.arange(0, self.num_joints * self.num_joints, dtype=torch.long, device=self.device).view(self.num_joints, self.num_joints)
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
        connection_label_2 = torch.zeros(num_edges, self.num_joints, dtype=torch.long, device=self.device)
        connection_label_2[list(range(0, num_edges)), joint_type[edge_index[0]]] = 1
        connection_label_2[list(range(0, num_edges)), joint_type[edge_index[1]]] = 1
        # """

        if self.normalize_node_distance:
            norm_factor = max(self.scoremaps.shape[3], self.scoremaps.shape[2])
        else:
            norm_factor = 1

        edge_attr_y = (joint_y[edge_index[1]] - joint_y[edge_index[0]]).float() / norm_factor
        edge_attr_x = (joint_x[edge_index[1]] - joint_x[edge_index[0]]).float() / norm_factor

        a_x, a_y = (joint_x[edge_index[0]] - joint_x[edge_index[1]]).float(), (joint_y[edge_index[0]] - joint_y[edge_index[1]]).float()
        edge_attr_theta = torch.abs(torch.acos(a_x * torch.rsqrt(a_x**2 + a_y**2)))
        edge_attr_theta[torch.isnan(edge_attr_theta)] = 0.0

        if {"position", "connection_type"} == set(edge_features_to_use):
            edge_attr = torch.cat([edge_attr_x.unsqueeze(1), edge_attr_y.unsqueeze(1), connection_label_2.float()],
                                  dim=1)
        elif {"connection_type"} == set(edge_features_to_use):
                edge_attr = torch.cat([connection_label_2.float()],
                                      dim=1)
        elif {"nothing"} == set(edge_features_to_use):
            edge_attr = torch.zeros_like(edge_attr_x).to(self.device).float().unsqueeze(1)
        elif {"position"} == set(edge_features_to_use):
            edge_attr = torch.cat([edge_attr_x.unsqueeze(1), edge_attr_y.unsqueeze(1)], dim=1)
        elif {"position", "angle", "connection_type"} == set(edge_features_to_use):
            edge_attr = torch.cat([edge_attr_x[:, None], edge_attr_y[:, None], edge_attr_theta[:, None], connection_label_2.float()],
                                  dim=1)
        elif {"ae"} == set(edge_features_to_use):
            joint_tags = tag_maps[joint_type, joint_y, joint_x]
            edge_attr = torch.unsqueeze(joint_tags[edge_index[1]] - joint_tags[edge_index[0]], 1).norm(p=None, dim=1, keepdim=True)
        elif {"ae_normed"} == set(edge_features_to_use):
            joint_tags = tag_maps[joint_type, joint_y, joint_x]
            edge_attr = (joint_tags[edge_index[1]] - joint_tags[edge_index[0]]).norm(p=None, dim=1, keepdim=True).round() * 100\
                        - joint_scores[edge_index[0], None]
        elif {"ae_tracking_1"} == set(edge_features_to_use):
            t_a = 1.8425  # based on "features for multi-target multi-camera tracking and re-identification"
            joint_tags = tag_maps[joint_type, joint_y, joint_x]
            if len(joint_tags.shape) == 1:
                joint_tags = joint_tags[:, None]
            distance = (joint_tags[edge_index[1]] - joint_tags[edge_index[0]]).norm(p=None, dim=1,
                                                                                     keepdim=True)
            edge_attr = torch.div(t_a - distance, t_a)
        elif {"position", "connection_type", "ae_normed"} == set(edge_features_to_use):
            joint_tags = tag_maps[joint_type, joint_y, joint_x]
            if len(joint_tags.shape) == 1:
                joint_tags = joint_tags[:, None]
            distance = (joint_tags[edge_index[1]] - joint_tags[edge_index[0]]).norm(p=None, dim=1, keepdim=True)
            edge_attr = torch.cat([edge_attr_x.unsqueeze(1), edge_attr_y.unsqueeze(1), connection_label_2.float(), distance],
                                  dim=1)
        else:
            raise NotImplementedError

        return x, edge_attr, edge_index

    def knn_mpn_graph(self, joint_det):
        temp = joint_det[:, :2].float()  # nn.knn_graph (cuda) can't handle int64 tensors
        edge_index = torch_geometric.nn.knn_graph(temp, k=50)
        edge_index = gutils.to_undirected(edge_index, len(joint_det))
        edge_index, _ = gutils.remove_self_loops(edge_index)
        return edge_index

    def feature_knn_mpn_graph(self, joint_det, features):
        edge_index = torch_geometric.nn.knn_graph(features, k=50)
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
        edge_index = torch.zeros(2, len(joint_det), self.num_joints * k, dtype=torch.long, device=self.device)
        edge_index[0] = edge_index[0] + torch.arange(0, len(joint_det), dtype=torch.long, device=self.device)[:, None]
        edge_index = edge_index.reshape(2, -1)
        joint_det = joint_det.float()
        if self.detect_threshold is not None:
            # code assumes that each joint type has same number of detections
            raise NotImplementedError
        _, indices = joint_det[:, 2].sort()
        distance = torch.norm(joint_det[:, None, :2] - joint_det[indices][:, :2], dim=2).view(-1, 30)
        # distance shape is (num_det * 17, num_det_per_type)
        _, top_k_idx = distance.topk(k=k, dim=1, largest=False)
        # top_k_idx shape (num_det * 17, k)
        top_k_idx = top_k_idx.view(len(joint_det), self.num_joints, k) + \
                    torch.arange(0, self.num_joints, dtype=torch.long, device=self.device)[:, None] * 30
        top_k_idx = top_k_idx.view(-1)
        edge_index[1] = indices[top_k_idx]

        edge_index = gutils.to_undirected(edge_index, len(joint_det))
        edge_index, _ = gutils.remove_self_loops(edge_index)
        return edge_index

    def score_based_graph(self, joint_det, joint_scores, k):
        """
        Idea: create a fully connected graph using high scoring joints called root joints
        (assumption is that for each person at least one joint has a high score)
        the rest of the joint are connected to all of these root joints
        :param joint_det:
        :param joint_scores:
        :return:
        """
        num_joints = len(joint_det)
        _, root_joint_idx = joint_scores.topk(k=k)
        adj_mat = torch.zeros([num_joints, num_joints], dtype=torch.long)
        adj_mat[root_joint_idx] = 1
        edge_index, _ = gutils.dense_to_sparse(adj_mat)
        edge_index = gutils.to_undirected(edge_index, len(joint_det))
        edge_index, _ = gutils.remove_self_loops(edge_index)

        return edge_index.to(self.device)

    def score_based_k_per_type(self, joint_det, joint_scores):
        """
        Idea: create a fully connected graph using high scoring joints called root joints
        (assumption is that for each person at least one joint has a high score)
        the rest of the joint are connected to all of these root joints
        :param joint_det:
        :param joint_scores:
        :return:
        """
        num_joints = len(joint_det)
        joint_det = joint_det.view(self.num_joints, 30, 3)
        joint_scores = joint_scores.view(self.num_joints, 30)
        _, indices = joint_scores.topk(k=2, dim=1)

        type_idx = joint_det[np.arange(0, self.num_joints).reshape(self.num_joints, 1), indices, 2]
        indices = indices.reshape(-1)
        root_joint_idx = indices + type_idx.view(-1) * self.num_joints

        adj_mat = torch.zeros([num_joints, num_joints], dtype=torch.long)
        adj_mat[root_joint_idx] = 1
        adj_mat[joint_scores.view(-1) > 0.1] = 1
        edge_index, _ = gutils.dense_to_sparse(adj_mat)
        edge_index = gutils.to_undirected(edge_index, num_joints)
        edge_index, _ = gutils.remove_self_loops(edge_index)

        return edge_index.to(self.device)

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

    def _construct_edge_labels_1(self, joint_det, joints_gt, factors, edge_index):
        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)

        clamp_max = max(self.scoremaps.shape[2], self.scoremaps.shape[3]) - 1
        distance = (joints_gt[person_idx_gt, joint_idx_gt, :2].unsqueeze(1).round().float().clamp(0,
                                                                                                  clamp_max) - joint_det[
                                                                                                               :,
                                                                                                               :2].float()).pow(
            2).sum(dim=2)
        factor_per_joint = factors[person_idx_gt, joint_idx_gt]
        similarity = torch.exp(-distance / factor_per_joint[:, None])

        different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
        similarity[different_type] = 0.0
        similarity[similarity < self.node_matching_radius] = 0.0  # 0.1 worked well for threshold + knn graph

        cost_mat = similarity.cpu().numpy()
        sol = linear_sum_assignment(cost_mat, maximize=True)
        row, col = sol
        # remove mappings with cost 0.0
        valid_match = cost_mat[row, col] != 0.0
        if num_joints_gt >= 2:  # in this case there are no detected joints and we should have a perfect matching
            assert valid_match.sum() == len(valid_match)
        row = row[valid_match]
        col = col[valid_match]
        person_idx_gt_1 = person_idx_gt[row]

        row_1 = np.arange(0, len(row), dtype=np.int64)
        row_1, col_1 = torch.from_numpy(row_1).to(self.device), torch.from_numpy(col).to(self.device)

        sol = row_1, col_1

        edge_labels = NaiveGraphConstructor.match_cc(person_idx_gt_1, joint_det, edge_index, sol)
        if num_joints_gt >= 2:
            label_mask_edge = torch.ones_like(edge_labels, dtype=torch.float32, device=self.device)
        else:
            label_mask_edge = torch.zeros_like(edge_labels, dtype=torch.float32, device=self.device)

        return edge_labels, label_mask_edge

    def _construct_edge_labels_2(self, joint_det, joints_gt, edge_index):
        assert self.use_gt
        num_joints_det = len(joint_det)

        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)

        clamp_max = max(self.scoremaps.shape[2], self.scoremaps.shape[3]) - 1  # -1 to get the same value as was added
        distance = torch.norm(
            joints_gt[person_idx_gt, joint_idx_gt, :2].unsqueeze(1).round().float().clamp(0, clamp_max) - joint_det[:,
                                                                                                          :2].float(),
            dim=2)
        # set distance between joints of different type to some high value
        different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
        distance[different_type] = 1000.0
        if self.include_neighbouring_keypoints:
            assignment = distance < -1  # fast way to generate array initialized with false
            assignment[:, :-num_joints_gt] = distance[:, :-num_joints_gt] < self.inclusion_radius
            assignment[np.arange(0, num_joints_gt), np.arange(num_joints_det - num_joints_gt, num_joints_det)] = True
        else:
            assignment = distance < -1  # fast way to generate array initialized with false
            assignment[np.arange(0, num_joints_gt), np.arange(num_joints_det - num_joints_gt, num_joints_det)] = True
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

    def _construct_edge_labels_3(self, joint_det, joints_gt, factors, edge_index):
        assert not self.use_gt
        num_joints_det = len(joint_det)
        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)

        clamp_max = max(self.scoremaps.shape[2], self.scoremaps.shape[3])
        distance = (joints_gt[person_idx_gt, joint_idx_gt, :2].unsqueeze(1).round().float().clamp(0,
                                                                                                  clamp_max) - joint_det[
                                                                                                               :,
                                                                                                               :2].float()).pow(
            2).sum(dim=2)
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
        person_idx_gt_1 = person_idx_gt[row]

        row_1 = np.arange(0, len(row), dtype=np.int64)
        row_1, col_1 = torch.from_numpy(row_1).to(self.device), torch.from_numpy(col).to(self.device)

        ambiguous_dets = []
        if self.include_neighbouring_keypoints:
            # use inclusion radius to filter more aggressively
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
        node_labels = torch.zeros(joint_det.shape[0], dtype=torch.float32, device=self.device)
        node_labels[col_1] = 1.0

        edge_labels = NaiveGraphConstructor.match_cc(person_idx_gt_1, joint_det, edge_index, sol)
        joint_det_idx = torch.arange(0, len(joint_det), dtype=torch.int64, device=self.device)
        label_mask = NaiveGraphConstructor.create_loss_mask(joint_det_idx[ambiguous_dets], edge_index)
        label_mask *= subgraph_mask(node_labels == 1.0, edge_index)  # apply loss only to gt nodes


        # node clusters
        node_persons = torch.zeros(num_joints_det, dtype=torch.long, device=self.device) - 1
        node_persons[col_1] = person_idx_gt_1[row_1].long()
        return edge_labels, node_persons, label_mask

    def _construct_edge_labels_4(self, joint_det, joints_gt, factors, edge_index):
        assert not self.use_gt
        num_joints_det = len(joint_det)
        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)

        clamp_max = max(self.scoremaps.shape[2], self.scoremaps.shape[3])
        distance = (joints_gt[person_idx_gt, joint_idx_gt, :2].unsqueeze(1).round().float().clamp(0,
                                                                                                  clamp_max) - joint_det[
                                                                                                               :,
                                                                                                               :2].float()).pow(
            2).sum(dim=2)
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
        person_idx_gt_1 = person_idx_gt[row]

        row_1 = np.arange(0, len(row), dtype=np.int64)
        row_1, col_1 = torch.from_numpy(row_1).to(self.device), torch.from_numpy(col).to(self.device)

        ambiguous_dets = []
        if self.include_neighbouring_keypoints:
            # use inclusion radius to filter more aggressively
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
        node_labels = torch.zeros(joint_det.shape[0], dtype=torch.float32, device=self.device)
        node_labels[col_1] = 1.0

        edge_labels = NaiveGraphConstructor.match_cc(person_idx_gt_1, joint_det, edge_index, sol)
        joint_det_idx = torch.arange(0, len(joint_det), dtype=torch.int64, device=self.device)
        label_mask = NaiveGraphConstructor.create_loss_mask(joint_det_idx[ambiguous_dets], edge_index)

        # node clusters
        node_persons = torch.zeros(num_joints_det, dtype=torch.long, device=self.device) - 1
        node_persons[col_1] = person_idx_gt_1[row_1].long()
        return edge_labels, node_labels, node_persons, label_mask

    def _construct_edge_labels_5(self, joint_det, joints_gt, factors, edge_index):
        assert not self.use_gt
        assert self.include_neighbouring_keypoints
        num_joints_det = len(joint_det)
        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)

        clamp_max = max(self.scoremaps.shape[2], self.scoremaps.shape[3])
        distance = (joints_gt[person_idx_gt, joint_idx_gt, :2].unsqueeze(1).round().float().clamp(0,
                                                                                                  clamp_max) - joint_det[
                                                                                                               :,
                                                                                                               :2].float()).pow(
            2).sum(dim=2)
        factor_per_joint = factors[person_idx_gt, joint_idx_gt]
        similarity = torch.exp(-distance / factor_per_joint[:, None])

        different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
        similarity[different_type] = 0.0
        similarity_orig = similarity.clone()
        similarity[similarity < self.node_matching_radius] = 0.0  # 0.1 worked well for threshold + knn graph

        cost_mat = similarity.cpu().numpy()
        sol = linear_sum_assignment(cost_mat, maximize=True)
        row, col = sol
        # remove mappings with cost 0.0
        valid_match = cost_mat[row, col] != 0.0
        row = row[valid_match]
        col = col[valid_match]
        person_idx_gt_1 = person_idx_gt[row]

        row_1 = np.arange(0, len(row), dtype=np.int64)
        row_1, col_1 = torch.from_numpy(row_1).to(self.device), torch.from_numpy(col).to(self.device)

        ambiguous_dets = []
        if self.include_neighbouring_keypoints:
            # use inclusion radius to filter more aggressively
            cost_mat[cost_mat < self.node_inclusion_radius] = 0.0
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
        node_labels = torch.zeros(joint_det.shape[0], dtype=torch.float32, device=self.device)
        node_labels[col_1] = 1.0

        edge_labels = NaiveGraphConstructor.match_cc(person_idx_gt_1, joint_det, edge_index, sol)
        joint_det_idx = torch.arange(0, len(joint_det), dtype=torch.int64, device=self.device)
        label_mask_edge = NaiveGraphConstructor.create_loss_mask(joint_det_idx[ambiguous_dets], edge_index)
        label_mask_node = torch.ones_like(node_labels, dtype=torch.float32, device=self.device)
        if num_joints_gt != 0:
            # similarity_orig[:, col_1] = 1.0
            similarity_orig, _ = similarity_orig.max(dim=0)
            label_mask_node[(similarity_orig >= 0.1) & (similarity_orig <= 0.8)] = 0.0

        return edge_labels, node_labels, label_mask_edge, label_mask_node

    def _construct_edge_labels_6(self, joint_det, joints_gt, factors, edge_index):
        assert not self.use_gt
        num_joints_det = len(joint_det)
        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)

        clamp_max = max(self.scoremaps.shape[2], self.scoremaps.shape[3])
        distance = (joints_gt[person_idx_gt, joint_idx_gt, :2].unsqueeze(1).round().float().clamp(0,
                                                                                                  clamp_max) - joint_det[
                                                                                                               :,
                                                                                                               :2].float()).pow(
            2).sum(dim=2)
        factor_per_joint = factors[person_idx_gt, joint_idx_gt]
        similarity = torch.exp(-distance / factor_per_joint[:, None])

        different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
        method = 2
        # 2 semi agnostic
        # 3 pure agnostic
        # 1 do not use it
        # """
        if method == 1:
            # different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
            similarity_constrained = similarity.clone()
            similarity_constrained[
                similarity_constrained < self.matching_radius] = 0.0  # 0.1 worked well for threshold + knn graph
            similarity_constrained[similarity_constrained == 0.0] -= 1.0  # 0.1 worked well for threshold + knn graph
            similarity_constrained[different_type] -= 1.0

            cost_mat = similarity_constrained.cpu().numpy()
            sol = linear_sum_assignment(cost_mat, maximize=True)
            row_con, col_con = sol
            # remove mappings with cost 0.0
            valid_match = cost_mat[row_con, col_con] != -1.0
            row_con = row_con[valid_match]
            col_con = col_con[valid_match]
            person_idx_gt_1 = person_idx_gt[row_con]
            joint_idx_gt_1 = joint_idx_gt[row_con]
        elif method == 2:
            # different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
            similarity_same = similarity.clone()
            similarity_diff = similarity.clone()
            similarity_same[different_type] = 0.0
            similarity_same[similarity_same < self.matching_radius] = 0.0  # 0.1 worked well for threshold + knn graph
            similarity_diff[torch.logical_not(different_type)] = 0.0
            similarity_diff[similarity_diff < self.matching_radius] = 0.0

            cost_mat_same = similarity_same.cpu().numpy()
            cost_mat_diff = similarity_diff.cpu().numpy()
            sol_same = linear_sum_assignment(cost_mat_same, maximize=True)
            sol_diff = linear_sum_assignment(cost_mat_diff, maximize=True)
            row_con, col_con = sol_same
            # remove mappings with cost 0.0
            valid_match = cost_mat_same[row_con, col_con] != 0.0
            fill_in = np.logical_not(valid_match)
            col_con[fill_in] = sol_diff[1][fill_in]
            valid_match = cost_mat_diff[sol_diff] + cost_mat_same[sol_same] != 0.0
            row_con = row_con[valid_match]
            col_con = col_con[valid_match]
            person_idx_gt_1 = person_idx_gt[row_con]
            joint_idx_gt_1 = joint_idx_gt[row_con]
        elif method == 3:
            similarity_constrained = similarity.clone()
            similarity_constrained[
                similarity_constrained < self.matching_radius] = 0.0  # 0.1 worked well for threshold + knn graph

            cost_mat = similarity_constrained.cpu().numpy()
            sol = linear_sum_assignment(cost_mat, maximize=True)
            row_con, col_con = sol
            # remove mappings with cost 0.0
            valid_match = cost_mat[row_con, col_con] != 0.0
            row_con = row_con[valid_match]
            col_con = col_con[valid_match]
            person_idx_gt_1 = person_idx_gt[row_con]
            joint_idx_gt_1 = joint_idx_gt[row_con]
        else:
            raise NotImplementedError

        row_1 = np.arange(0, len(row_con), dtype=np.int64)
        row_1, col_1 = torch.from_numpy(row_1).to(self.device), torch.from_numpy(col_con).to(self.device)

        ambiguous_dets = []
        if self.include_neighbouring_keypoints:
            if method == 10:  # previously 2
                different_type = different_type.cpu().numpy()  # todo make the move earlier
                cost_mat = similarity.cpu().numpy()
                cost_mat[np.arange(0, num_joints_gt).reshape(-1, 1), col_con] = 0.0

                cost_first = cost_mat.copy()
                cost_first[different_type] = 0.0
                cost_first[cost_first < self.inclusion_radius] = 0.0
                ambiguous_dets = (cost_first != 0.0).sum(axis=0) > 1.0
                cost_first[np.arange(0, num_joints_gt).reshape(-1, 1), ambiguous_dets] = 0.0
                row_2_1, col_2_1 = np.nonzero(cost_first)
                cost_second = cost_mat.copy()
                cost_second[np.logical_not(different_type)] = 0.0
                cost_second[cost_second < 0.75] = 0.0
                ambiguous_dets = (cost_second != 0.0).sum(axis=0) > 1.0
                cost_second[np.arange(0, num_joints_gt).reshape(-1, 1), ambiguous_dets] = 0.0
                row_2_2, col_2_2 = np.nonzero(cost_second)
                # it might happen that there are more gt joints than detected joints and if matching does not work out
                bad_rows_1 = np.array(list(set((set(list(row_2_1)).difference(set(list(row_con)))))))
                bad_rows_2 = np.array(list(set((set(list(row_2_2)).difference(set(list(row_con)))))))
                for r in bad_rows_1:
                    cost_first[r] = 0.0
                for r in bad_rows_2:
                    cost_second[r] = 0.0
                row_2_1, col_2_1 = np.nonzero(cost_first)
                row_2_2, col_2_2 = np.nonzero(cost_second)
                row_2, col_2 = np.concatenate([row_2_1, row_2_2], axis=0), np.concatenate([col_2_1, col_2_2], axis=0)
                if not (set(list(row_2)).issubset(set(list(row_con)))):
                    print("sdfad")
                assert (set(list(row_2)).issubset(set(list(row_con))))
            else:
                cost_mat = similarity.cpu().numpy()
                # use inclusion radius to filter more aggressively
                cost_mat[cost_mat < self.inclusion_radius] = 0.0
                # remove already chosen keypoints from next selection
                cost_mat[np.arange(0, num_joints_gt).reshape(-1, 1), col_con] = 0.0
                # "remove" ambiguous cases
                # identify them
                ambiguous_dets = (cost_mat != 0.0).sum(axis=0) > 1.0
                # remove them
                cost_mat[np.arange(0, num_joints_gt).reshape(-1, 1), ambiguous_dets] = 0.0
                row_2, col_2 = np.nonzero(cost_mat)
                # it might happen that there are more gt joints than detected joints and if matching does not work out
                bad_rows = np.array(list(set((set(list(row_2)).difference(set(list(row_con)))))))
                for r in bad_rows:
                    cost_mat[r] = 0.0
                row_2, col_2 = np.nonzero(cost_mat)
                if not (set(list(row_2)).issubset(set(list(row_con)))):
                    print("sdfad")
                assert (set(list(row_2)).issubset(set(list(row_con))))

            # some gt joints have no match and have to be removed for the next step
            # create translation table for new indices in order to translate row_2
            mapping = np.zeros(num_joints_gt, dtype=np.int64) - 1
            mapping[row_con] = np.arange(0, len(row_con), dtype=np.int64)
            row_2 = mapping[row_2]
            assert (row_2 == -1).sum() == 0
            row_2, col_2 = torch.from_numpy(row_2).to(self.device), torch.from_numpy(col_2).to(self.device)

            row_1 = torch.cat([row_1, row_2], dim=0)
            col_1 = torch.cat([col_1, col_2], dim=0)

        sol = row_1, col_1

        node_labels, node_mask = None, None
        # edges
        edge_labels = NaiveGraphConstructor.match_cc(person_idx_gt_1, joint_det, edge_index, sol)
        joint_det_idx = torch.arange(0, len(joint_det), dtype=torch.int64, device=self.device)
        label_mask = NaiveGraphConstructor.create_loss_mask(joint_det_idx[ambiguous_dets], edge_index)

        # nodes
        node_labels = torch.zeros(joint_det.shape[0], dtype=torch.float32, device=self.device)
        node_labels[col_1] = 1.0

        node_mask = torch.ones_like(node_labels, dtype=torch.float32, device=self.device)
        if self.include_neighbouring_keypoints:
            node_mask[ambiguous_dets] = 0

        # classes
        node_classes = torch.zeros(num_joints_det, dtype=torch.long, device=self.device)
        node_classes[col_1] = joint_idx_gt_1[row_1]
        class_mask = node_labels.clone() * node_mask
        if self.with_background_class:
            node_classes[node_labels != 1.0] = self.num_joints  # background class
            class_mask[:] = 1.0

        # node clusters
        node_persons = torch.zeros(num_joints_det, dtype=torch.long, device=self.device) - 1
        node_persons[col_1] = person_idx_gt_1[row_1].long()

        return edge_labels, node_labels, node_classes, node_persons, label_mask, node_mask, class_mask

    def _construct_edge_labels_7(self, joint_det, joints_gt, factors, edge_index):
        assert not self.use_gt
        person_idx_gt, joint_idx_gt = joints_gt[:, :, 2].nonzero(as_tuple=True)
        num_joints_gt = len(person_idx_gt)
        if self.testing:
            nodes_det = joint_det
        else:
            nodes_det = joint_det[:-num_joints_gt]
        num_joints_det = len(nodes_det)

        clamp_max = max(self.scoremaps.shape[2], self.scoremaps.shape[3])
        distance = (joints_gt[person_idx_gt, joint_idx_gt, :2].unsqueeze(1).round().float().clamp(0,
                                                                                                  clamp_max) - nodes_det[
                                                                                                               :,
                                                                                                               :2].float()).pow(
            2).sum(dim=2)
        factor_per_joint = factors[person_idx_gt, joint_idx_gt]
        similarity = torch.exp(-distance / factor_per_joint[:, None])

        different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), nodes_det[:, 2]))
        method = 3
        # """
        if method == 2:
            # different_type = torch.logical_not(torch.eq(joint_idx_gt.unsqueeze(1), joint_det[:, 2]))
            similarity_same = similarity.clone()
            similarity_diff = similarity.clone()
            similarity_same[different_type] = 0.0
            similarity_same[similarity_same < self.matching_radius] = 0.0  # 0.1 worked well for threshold + knn graph
            similarity_diff[torch.logical_not(different_type)] = 0.0
            similarity_diff[similarity_diff < self.matching_radius] = 0.0

            cost_mat_same = similarity_same.cpu().numpy()
            cost_mat_diff = similarity_diff.cpu().numpy()
            sol_same = linear_sum_assignment(cost_mat_same, maximize=True)
            sol_diff = linear_sum_assignment(cost_mat_diff, maximize=True)
            row_con, col_con = sol_same
            # remove mappings with cost 0.0
            valid_match = cost_mat_same[row_con, col_con] != 0.0
            fill_in = np.logical_not(valid_match)
            col_con[fill_in] = sol_diff[1][fill_in]
            valid_match = cost_mat_diff[sol_diff] + cost_mat_same[sol_same] != 0.0
            row_con = row_con[valid_match]
            col_con = col_con[valid_match]
        elif method == 3:
            similarity_constrained = similarity.clone()
            similarity_constrained[
                similarity_constrained < self.matching_radius] = 0.0  # 0.1 worked well for threshold + knn graph

            cost_mat = similarity_constrained.cpu().numpy()
            sol = linear_sum_assignment(cost_mat, maximize=True)
            row_con, col_con = sol
            # remove mappings with cost 0.0
            valid_match = cost_mat[row_con, col_con] != 0.0
            row_con = row_con[valid_match]
            col_con = col_con[valid_match]
        else:
            raise NotImplementedError
        if self.testing:
            person_idx_gt_1 = person_idx_gt[row_con]
            joint_idx_gt_1 = joint_idx_gt[row_con]
            row_1 = np.arange(0, len(row_con), dtype=np.int64)
        else:
            person_idx_gt_1 = person_idx_gt
            joint_idx_gt_1 = joint_idx_gt
            row_1 = row_con

        row_1, col_1 = torch.from_numpy(row_1).to(self.device), torch.from_numpy(col_con).to(self.device)

        ambiguous_dets = []
        if self.include_neighbouring_keypoints:
            if method == 2:
                different_type = different_type.cpu().numpy()  # todo make the move earlier
                cost_mat = similarity.cpu().numpy()
                cost_mat[np.arange(0, num_joints_gt).reshape(-1, 1), col_con] = 0.0

                cost_first = cost_mat.copy()
                cost_first[different_type] = 0.0
                cost_first[cost_first < self.inclusion_radius] = 0.0
                ambiguous_dets = (cost_first != 0.0).sum(axis=0) > 1.0
                cost_first[np.arange(0, num_joints_gt).reshape(-1, 1), ambiguous_dets] = 0.0
                row_2_1, col_2_1 = np.nonzero(cost_first)
                cost_second = cost_mat.copy()
                cost_second[np.logical_not(different_type)] = 0.0
                cost_second[cost_second < 0.75] = 0.0
                ambiguous_dets = (cost_second != 0.0).sum(axis=0) > 1.0
                cost_second[np.arange(0, num_joints_gt).reshape(-1, 1), ambiguous_dets] = 0.0
                row_2_2, col_2_2 = np.nonzero(cost_second)
                # it might happen that there are more gt joints than detected joints and if matching does not work out
                bad_rows_1 = np.array(list(set((set(list(row_2_1)).difference(set(list(row_con)))))))
                bad_rows_2 = np.array(list(set((set(list(row_2_2)).difference(set(list(row_con)))))))
                for r in bad_rows_1:
                    cost_first[r] = 0.0
                for r in bad_rows_2:
                    cost_second[r] = 0.0
                row_2_1, col_2_1 = np.nonzero(cost_first)
                row_2_2, col_2_2 = np.nonzero(cost_second)
                row_2, col_2 = np.concatenate([row_2_1, row_2_2], axis=0), np.concatenate([col_2_1, col_2_2], axis=0)
                if not (set(list(row_2)).issubset(set(list(row_con)))):
                    print("sdfad")
                assert (set(list(row_2)).issubset(set(list(row_con))))
            else:
                cost_mat = similarity.cpu().numpy()
                # use inclusion radius to filter more aggressively
                cost_mat[cost_mat < self.inclusion_radius] = 0.0
                # remove already chosen keypoints from next selection
                cost_mat[np.arange(0, num_joints_gt).reshape(-1, 1), col_con] = 0.0
                # "remove" ambiguous cases
                # identify them
                ambiguous_dets = (cost_mat != 0.0).sum(axis=0) > 1.0
                # remove them
                cost_mat[np.arange(0, num_joints_gt).reshape(-1, 1), ambiguous_dets] = 0.0
                row_2, col_2 = np.nonzero(cost_mat)
                # it might happen that there are more gt joints than detected joints and if matching does not work out
                bad_rows = np.array(list(set((set(list(row_2)).difference(set(list(row_con)))))))
                for r in bad_rows:
                    cost_mat[r] = 0.0
                row_2, col_2 = np.nonzero(cost_mat)
                if not (set(list(row_2)).issubset(set(list(row_con)))):
                    print("sdfad")
                assert (set(list(row_2)).issubset(set(list(row_con))))

            # some gt joints have no match and have to be removed for the next step
            # create translation table for new indices in order to translate row_2
            mapping = np.zeros(num_joints_gt, dtype=np.int64) - 1
            mapping[row_con] = np.arange(0, len(row_con), dtype=np.int64)
            row_2 = mapping[row_2]
            assert (row_2 == -1).sum() == 0
            row_2, col_2 = torch.from_numpy(row_2).to(self.device), torch.from_numpy(col_2).to(self.device)

            row_1 = torch.cat([row_1, row_2], dim=0)
            col_1 = torch.cat([col_1, col_2], dim=0)

        if not self.testing:
            row_1 = torch.cat([row_1, torch.arange(0, num_joints_gt, dtype=torch.long, device=self.device)])
            col_1 = torch.cat([col_1, torch.arange(num_joints_det, num_joints_det + num_joints_gt, dtype=torch.long,
                                                   device=self.device)])
        sol = row_1, col_1
        node_labels = torch.zeros(joint_det.shape[0], dtype=torch.float32, device=self.device)
        node_labels[col_1] = 1.0

        edge_labels = NaiveGraphConstructor.match_cc(person_idx_gt_1, joint_det, edge_index, sol)
        joint_det_idx = torch.arange(0, len(nodes_det), dtype=torch.int64, device=self.device)
        label_mask = NaiveGraphConstructor.create_loss_mask(joint_det_idx[ambiguous_dets], edge_index)

        node_classes = torch.zeros(joint_det.shape[0], dtype=torch.long, device=self.device)
        node_classes[col_1] = joint_idx_gt_1[row_1]

        node_persons = torch.zeros(joint_det.shape[0], dtype=torch.long, device=self.device) - 1
        node_persons[col_1] = person_idx_gt_1[row_1].long()
        return edge_labels, node_labels, node_classes, node_persons, label_mask

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
        b_to_clusters_1 = cluster_idx_ext_1[edges_b_to_a[edges_b[0]]]
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
        # todo next line consumes to much memory for topk selection
        source_edges = torch.eq(edge_index[0].unsqueeze(1), joints).sum(dim=1)
        assert source_edges.max() <= 1.0
        source_edges = source_edges == 1.0

        target_edges = torch.eq(edge_index[1].unsqueeze(1), joints).sum(dim=1)
        assert target_edges.max() <= 1.0
        target_edges = target_edges == 1.0
        loss_mask[source_edges | target_edges] = 0.0
        if len(joints) == 0:
            assert loss_mask.min() == 1.0
        return loss_mask


def joint_det_from_scoremap(scoremap, num_joints, threshold=0.007, pool_kernel=None, mask=None, hybrid_k=5):
    joint_map = non_maximum_suppression(scoremap, threshold=threshold, pool_kernel=pool_kernel)
    if mask is not None:
        joint_map = joint_map * mask.unsqueeze(0)
    scoremap = scoremap * joint_map
    if threshold is not None:
        # use atleast the top 5 keypoints per type
        k = hybrid_k
        scoremap_shape = scoremap.shape
        scores, indices = scoremap.view(num_joints, -1).topk(k=k, dim=1)
        container = torch.zeros_like(scoremap, device=scoremap.device, dtype=torch.float).reshape(num_joints, -1)
        container[np.arange(0, num_joints).reshape(num_joints, 1), indices] = scores
        container = container.reshape(scoremap_shape)
        top_joint_idx_det, top_joint_y, top_joint_x = container.nonzero(as_tuple=True)
        # top_joint_scores = container[top_joint_idx_det, top_joint_y, top_joint_x]

        scoremap_zero = torch.where(scoremap < threshold, torch.zeros_like(scoremap), scoremap)
        joint_idx_det, joint_y, joint_x = scoremap_zero.nonzero(as_tuple=True)

        top_joint_pos = torch.stack([top_joint_x, top_joint_y, top_joint_idx_det], 1)
        thresh_joint_pos = torch.stack([joint_x, joint_y, joint_idx_det], 1)
        joint_positions_det = cat_unique(top_joint_pos, thresh_joint_pos)
        joint_scores = scoremap[joint_positions_det[:, 2], joint_positions_det[:, 1], joint_positions_det[:, 0]]
    else:
        k = 20
        scoremap_shape = scoremap.shape
        scores, indices = scoremap.view(num_joints, -1).topk(k=k, dim=1)
        container = torch.zeros_like(scoremap, device=scoremap.device, dtype=torch.float).reshape(num_joints, -1)
        container[np.arange(0, num_joints).reshape(num_joints, 1), indices] = scores + 1e-10  #
        container = container.reshape(scoremap_shape)
        joint_idx_det, joint_y, joint_x = container.nonzero(as_tuple=True)
        joint_scores = container[joint_idx_det, joint_y, joint_x]
        assert len(joint_idx_det) == k * num_joints

        joint_positions_det = torch.stack([joint_x, joint_y, joint_idx_det], 1)
    return joint_positions_det, joint_scores


def cat_unique(tensor_1, tensor_2):
    # tensor_1/2 have shape (N, D) D is dim of vector and N is number of vectors
    assert len(tensor_1.shape) == 2 and len(tensor_2.shape) == 2
    vector_dim = tensor_1.shape[1]
    unique_elements = torch.eq(tensor_1.unsqueeze(1), tensor_2)
    unique_elements = unique_elements.int().sum(dim=2)
    unique_elements = unique_elements == vector_dim
    unique_elements = unique_elements.sum(dim=0)
    unique_elements = unique_elements == 0

    return torch.cat([tensor_1, tensor_2[unique_elements]], 0)
