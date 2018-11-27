"""
Markov Clustering Algorithm Implementation
"""

import glob
import math
import os
import os.path
import random

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

OUTPUT_FOLER = "../output/videos"


def main():
    labels = list(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"])
    matrix = [
        [0, 20, 20, 20, 40, 60, 60, 60, 100, 120, 120, 120],
        [20, 0, 20, 20, 60, 80, 80, 80, 120, 140, 140, 140],
        [20, 20, 0, 20, 60, 80, 80, 80, 120, 140, 140, 140],
        [20, 20, 20, 0, 60, 80, 80, 80, 120, 140, 140, 140],
        [40, 60, 60, 60, 0, 20, 20, 20, 60, 80, 80, 80],
        [60, 80, 80, 80, 20, 0, 20, 20, 40, 60, 60, 60],
        [60, 80, 80, 80, 20, 20, 0, 20, 60, 80, 80, 80],
        [60, 80, 80, 80, 20, 20, 20, 0, 60, 80, 80, 80],
        [100, 120, 120, 120, 60, 40, 60, 60, 0, 20, 20, 20],
        [120, 140, 140, 140, 80, 60, 80, 80, 20, 0, 20, 20],
        [120, 140, 140, 140, 80, 60, 80, 80, 20, 20, 0, 20],
        [120, 140, 140, 140, 80, 60, 80, 80, 20, 20, 20, 0]
    ]

    matrix = np.asarray(matrix)

    # Normalize the data
    xmax, xmin = matrix.max(), matrix.min()
    matrix = (matrix - xmin) / (xmax - xmin)

    # Turn into a similarity matrix
    matrix = 1 - matrix

    results = mcl_clusterinig(matrix, labels, inflation_power=2, expansion_power=2, plot=True)

    print(results)

    plt.show()


def normalize(matrix):
    """
    Columns normalization
    :param matrix:
    :return:
    """
    return matrix / np.sum(matrix, axis=0)


def expand(matrix, power):
    """
    Expansion of the matrix for the MCL
    :param matrix:
    :param power:
    :return:
    """
    return np.linalg.matrix_power(matrix, power)


def inflate(matrix, power):
    """
    Inflation of the matrix for the MCL
    :param matrix:
    :param power:
    :return:
    """
    for entry in np.nditer(matrix, op_flags=['readwrite']):
        entry[...] = math.pow(entry, power)
    return matrix


def compute_sparsity(matrix):
    """
    Compute the compute_sparsity of the matrix
    :param matrix:
    :return:
    """
    return 1 - np.count_nonzero(matrix) / (len(matrix) * len(matrix))


def translate_clustering(matrix, labels):
    """
    Translate the results of the MCL clustering
    :param matrix:
    :param labels:
    :return: The clusters, one with the id and the other one with labels
    """

    # clusters = np.column_stack(np.where(matrix > 0))

    clusters = matrix.argmax(0)
    cluster_ids = list(np.unique(clusters))

    clusters_dict = {}
    clusters_graph = {}

    for label_id in range(len(matrix)):
        label = labels[label_id]
        cluster_id = cluster_ids.index(clusters[label_id])  # to have it sorted like 0, 1, 2, ...

        if cluster_id in clusters_dict:
            clusters_dict[cluster_id].append(label_id)
            clusters_graph[cluster_id].append(label)
        else:
            clusters_dict[cluster_id] = [label_id]
            clusters_graph[cluster_id] = [label]

    return clusters_dict, clusters_graph


def plot_graph(matrix, labels, plot=False):
    rows, cols = np.where(matrix > 0)

    gr = nx.Graph()

    for label in labels:
        gr.add_node(label)

    for i in range(len(rows)):
        row = rows[i]
        col = cols[i]

        gr.add_edge(labels[row], labels[col], weight=matrix[row][col])

    # gr.add_edges_from(edges)
    # nx.draw(gr, node_size=500, labels=labels, with_labels=True)

    if plot:
        nx.draw(gr, node_size=500, with_labels=True)
        plt.show()

    return gr


def mcl_clusterinig(matrix, labels, expansion_power=4, inflation_power=2, nb_iterations_max=300, edges_treshold=0.7,
                    plot=True, gif=False):
    """
    Run the MCL clustering algorithm
    :param matrix: A similarity matrix
    :param expansion_power: How far you'd like your random-walkers to go (bigger number -> more walking)
    :param inflation_power: How tightly clustered you'd like your final picture to be (bigger number -> more clusters)
    :param nb_iterations_max: The number max of iterations
    :param edges_treshold: Threshold to link similar nodes
    :return: the clusters
    """

    # TODO: Set a better convergence metric (inter-cluster and intra-cluster distance)

    # Cut weak edges
    threshold_indices = edges_treshold > matrix
    matrix[threshold_indices] = 0

    # Create the initial graph
    graph = plot_graph(matrix, labels)

    ## CLUSTERING
    np.fill_diagonal(matrix, 1)
    matrix = normalize(matrix)

    for _ in range(nb_iterations_max):
        matrix = normalize(inflate(expand(matrix, expansion_power), inflation_power))

        num_ones = np.count_nonzero(matrix == 1)
        num_zeros = (matrix == 0).sum()

        if num_ones == len(matrix) and num_ones + num_zeros == len(matrix) * len(matrix):
            # we converged
            break

    ## Transform the MCL result matrix to clusters
    clusters, labels_clusters = translate_clustering(matrix, labels)

    clusters_color = {}

    # Colors the nodes
    color_dict = {}

    colors = generate_random_color(len(clusters))

    for cluster_id, cluster_label in labels_clusters.items():

        color = colors[cluster_id]

        # Add color to the cluster
        clusters_color[cluster_id] = color
        print(color)
        for label in cluster_label:
            color_dict[label] = color

    if plot:
        color_map = []

        for node in graph:
            color_map.append(color_dict[node])

        # Draw the graph
        plt.figure()
        pos = nx.drawing.spring_layout(graph)
        nx.draw(graph, node_size=300, pos=pos, node_color=color_map, with_labels=True)

        if gif:
            empty_folder(OUTPUT_FOLER)

            fake_graph = graph.copy()

            while len(labels) > 0:
                label = labels.pop()
                fake_graph.remove_node(label)

                fig = plt.figure()
                nx.draw(fake_graph, pos=pos, node_size=500, node_color=color_map, with_labels=True)
                # plt.show()
                plt.savefig(OUTPUT_FOLER + '/{}.png'.format(len(labels)))
                plt.close(fig)

            list_files = glob.glob(OUTPUT_FOLER + '/*.png')
            list_files.sort(key=os.path.getmtime, reverse=True)
            images = []

            for filename in list_files:
                images.append(imageio.imread(filename))
            imageio.mimsave(OUTPUT_FOLER + '/video.gif', images, duration=0.1)

    return clusters, clusters_color


def generate_random_color(n):
    """
    Generate n random colors
    :return:
    """
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += 3 * step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r, g, b))

    colors = ['#%02x%02x%02x' % (c[0], c[1], c[2]) for c in ret]
    return colors


def empty_folder(path):
    """
    Delete all the files in the folder
    :param path:
    :return:
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))


if __name__ == '__main__':
    main()
