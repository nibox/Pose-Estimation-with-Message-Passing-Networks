"""
This file contains all the functions used to solve the correlation clustering problem. It is recommended to use these
functions compared to using the functions from andres_graph_wrapper directly unless you are really sure about the
required form of the different parameters.
"""

import numpy as np
import torch

from scipy.sparse import csr_matrix
from torch_geometric.utils import to_dense_adj
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import accuracy_score, precision_score, recall_score

from Utils.correlation_clustering.andres_graph import andres_graph_wrapper

########################################################################################################################
# Main clustering function
########################################################################################################################

def cluster_graph(graph, method, complete=True):
    """
    General function that can call all the other clustering functions

    Args:
        graph:          PyTorch geometric graph object for which the correlation clustering will be done. This function
                        also overwrites the edges in this graph, so a copy of the graph must be created first if this is
                        not desired.
        method:         Clustering method to use
        complete:       Whether to use complete graphs to calculate the correlation clustering or not
                        If complete=True, all edges that are not contained in the graph object are considered as
                        disconnected but can change when solving the correlation clustering problem. If
                        complete=False, only the edges that are currently in the graph are considered when solving
                        the correlation clustering problem. This seems to lead to worse results for synthetic data.
                        After clustering, only the edges that are originally in the graph are changed.
                        TODO: Check if a different behavior might be better here (e.g. changing all edges)

    Returns:            Matrix with binary dissimilarities for each edge (1 is connected, 0 is cut). This result
                        might not be needed when you only want to change the values in the graph object.

    """

    graph = graph.cpu()
    # Updating the graph is only necessary for incomplete graphs --> Averages edge values
    input_edge_matrix = extract_edge_matrix(graph, update=not complete)

    if method is 'RD':
        if not complete:
            print(f'The clustering method \'{method}\' only works for complete graphs. Therefore, the method for complete graphs is used.')

        # Note: One could make a function that rounds an incomplete graph, but it wouldn't really make sense if it
        #       is an undirected graph because the same edge might get a different value depending on the direction.
        output_edge_matrix = cluster_round_complete(input_edge_matrix)
    elif method is 'KL':
        output_edge_matrix = cluster_andres_graph(graph, andres_graph_wrapper.cluster_KL, complete)
    elif method is 'GAEC':
        output_edge_matrix = cluster_andres_graph(graph, andres_graph_wrapper.cluster_GAEC, complete)
    elif method is 'MUT':
        output_edge_matrix = cluster_andres_graph(graph, andres_graph_wrapper.cluster_MUT, complete)

    # Overwrite existing edges in graph with the values in output_edge_matrix
    update_graph_with_edge_matrix(graph, output_edge_matrix)

    return output_edge_matrix

########################################################################################################################
# General helper functions for clustering
########################################################################################################################

def calculate_metrics(gt_matrix, input_edge_matrix, output_edge_matrix):
    '''
    Function that calculates all the metrics for two matrices with binary dissimilarities for each edge (1 is connected, 0 is cut)
    Args:
        gt_matrix:          Ground truth edge matrix with binary with binary dissimilarities for each edge (1 is connected, 0 is cut)
        input_edge_matrix:  Input edge matrix (e.g. from neural network) with binary dissimilarities for each edge (1 is connected, 0 is cut)
        output_edge_matrix: Output edge matrix (from clustering algorithm) with binary dissimilarities for each edge (1 is connected, 0 is cut)

    Returns:
        accuracy:           Accuracy ((TP + TN)/(TP + FP + FN + TN)) of output_edge_matrix compared to gt_matrix
        precision:          Precision (TP / (TP + FP)) of output_edge_matrix compared to gt_matrix
        recall:             Recall (TP / (TP + FN)) of output_edge_matrix compared to gt_matrix
        L2:                 L2 loss between output_edge_matrix and input_edge_matrix (how much the values had to be changed in order to satisfy all triangle inequalities)
        clusters:           The number of clusters resulting from the output_edge_matrix

    '''

    # Note: Could probably make this more efficient (instead of calling each method separately calculate TP, TN, FP, FN)

    gt = gt_matrix.reshape(-1).astype(np.bool)
    output = output_edge_matrix.reshape(-1).astype(np.bool)
    accuracy = accuracy_score(gt, output)
    precision = precision_score(gt, output)
    recall = recall_score(gt, output)
    L2 = np.sum((output_edge_matrix - input_edge_matrix)**2)
    clusters, _ = connected_components(csgraph=csr_matrix(output_edge_matrix), directed=False, return_labels=True)

    return accuracy, precision, recall, L2, clusters

def extract_edge_matrix(graph, update=False):
    '''
    Helper function that extracts the edge matrix from a PyTorch geometric graph object. Specifically, the values in the
    edge_attr attribute  of the graph object are extracted.

    The definition of the edges is as follows: 1 = connected, 0 otherwise

    Args:
        graph:  PyTorch geometric graph object
        update: If update=True, the attributes in the graph object will be overwritten. Makes sense if you want to
                average the values in a bidirectional graph.

    Returns:
                Matrix giving the edge connections

    '''

    # Fill matrix with values saved in graph
    edge_matrix = to_dense_adj(graph.edge_index, edge_attr=graph.edge_attr).numpy().squeeze()

    if np.tril(edge_matrix).sum() == 0:
        # Directed graph with edges of the form (i, j) with i < j
        # Obtain bidirectional graph by filling lower triangle (except diagonal) due to symmetry
        edge_matrix += edge_matrix.transpose()
    else:
        # Bidirectional graph
        # Note: This assumes that both directions are in the graph
        # Average values of the same edge in opposite directions
        edge_matrix = (edge_matrix + edge_matrix.transpose()) / 2

    # Add self-connections by filling diagonal with 1's (self-connections)
    np.fill_diagonal(edge_matrix, 1)

    # Update graph attributes
    if update:
        update_graph_with_edge_matrix(graph, edge_matrix)

    return edge_matrix

def update_graph_with_edge_matrix(graph, edge_matrix):
    '''
    Helper function that updates all of the edges in the graph with the values in the edge matrix

    Args:
        graph:          PyTorch geometric graph object
        edge_matrix:    Matrix with binary dissimilarities for each edge (1 is connected, 0 is cut)

    Returns:
                        Nothing because the values in graph are overwritten directly

    '''
    graph.edge_attr = torch.tensor(edge_matrix[graph.edge_index[0, :], graph.edge_index[1, :]])

def make_symmetric_by_adding_transpose(matrix):
    """
    Helper function that adds the transpose of a matrix to itself to create a symmetric function

    Args:
        matrix:     Input numpy matrix

    Returns:        Symmetric output numpy matrix

    """

    # Subtracting diagonal in case the original diagonal of the matrix contains nonzero entries
    return matrix + np.transpose(matrix) - np.diag(np.diag(matrix))

########################################################################################################################
# Function for RD clustering
########################################################################################################################

def cluster_round_complete(input_edge_matrix):
    '''
    Rounds all edges of an input edge matrix (simple thresholding with threshold at 0.5)

    Args:
        input_edge_matrix:  Input edge matrix with binary dissimilarities for each edge (1 is connected, 0 is cut)

    Returns:
        Output edge matrix with binary dissimilarities for each edge (1 is connected, 0 is cut)

    '''
    return np.round(input_edge_matrix)

########################################################################################################################
# Function for Andres clustering (KL, GAEC, MUT)
########################################################################################################################

def cluster_andres_graph(graph, clustering_function, complete):
    '''
    Calculates the clustering with functions from the Andres graph C++ framework

    Args:
        graph:                  PyTorch geomtric graph object
        clustering_function:    Clustering method to use. Possible values:
                                    'andres_graph_wrapper.cluster_KL',
                                    'andres_graph_wrapper.cluster_GAEC',
                                    'andres_graph_wrapper.cluster_MUT'
        complete:               Whether to use a complete graph in correlation clustering or not

    Returns:
                                Matrix with binary dissimilarities for each edge (1 is connected, 0 is cut)

    '''

    N = graph.num_nodes

    # Set up Andres graph object
    if complete:

        input_edge_matrix = extract_edge_matrix(graph)

        # Extract all values from the upper triangle (except diagonal) and put it in weights 1-d array
        weight_indices = np.triu_indices(N, 1)
        # Note: there must be negative weights for these function to work, so range is moved from [0, 1] to [-0.5, 0.5]
        weights = input_edge_matrix[weight_indices] - 0.5

        # Create Andres graph object
        g = andres_graph_wrapper.CompleteGraph(weights)

    else:

        edges = graph.edge_index.numpy()
        # Note: there must be negative weights for these functions to work, so range is moved from [0, 1] to [-0.5, 0.5]
        weights = graph.edge_attr.numpy() - 0.5

        # Only take edges and corresponding weights where (from node) is smaller than (to node) (upper triangle)
        weights = np.ascontiguousarray(weights[edges[0] < edges[1]])
        edges = np.ascontiguousarray(edges[:, edges[0] < edges[1]])

        # Create Andres graph object
        g = andres_graph_wrapper.Graph(edges, weights, N)

    # Call clustering wrapper function
    output_edge_attr = clustering_function(g)

    # Create output edge matrix
    if output_edge_attr is not None:

        # Create empty output edge matrix
        output_edge_matrix = np.zeros((N, N), dtype=np.int)

        # Fill upper triangle + redefine so that 1 is connected, 0 not
        if complete:
            output_edge_matrix[weight_indices] = 1 - output_edge_attr
        else:
            output_edge_matrix[edges[0], edges[1]] = 1 - output_edge_attr

        # Fill lower triangle
        output_edge_matrix += output_edge_matrix.transpose()
        # Fill diagonal (self-connections)
        np.fill_diagonal(output_edge_matrix, 1)

        return output_edge_matrix

    else:
        print('Some error occurred while executing Andres graph functions --> Simple thresholding was used to round graph outputs')
        return cluster_round_complete(extract_edge_matrix(graph))
