"""
Markov Clustering Algorithm Implementation
"""
import glob
import os.path

# import imageio
import markov_clustering as mc
import math

from Utils import *

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

    # Turn into a ks_similarity matrix
    matrix = 1 - matrix

    results = mcl_clusterinig(matrix, labels, plot=True)

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





def mcl_clusterinig(matrix, labels, threshold_filter=0.8, inflation_power=1.5, plot=True, gif=False):
    """
    Run the MCL clustering algorithm
    :param matrix: A similarity matrix
    :param labels : labels of the nodes
    :param treshold_filter: Threshold to filter weak edges
    :param inflation_power: How tightly clustered you'd like your final picture to be (bigger number -> more clusters)

    :return: the clusters
    """



    # Cut weak edges
    if inflation_power is None:
        inflations = [i / 10 for i in range(14, 17)]
    else:
        inflations = [inflation_power]

    if threshold_filter is None:
        thresholds = [i / 100 for i in range(80, 100, 5)]
    else:
        thresholds = [threshold_filter]

    Qs_matrix = np.zeros((len(thresholds), len(inflations)))

    for threshold in thresholds:
        threshold_indices = threshold > matrix

        weak_matrix = matrix.copy()
        weak_matrix[threshold_indices] = 0

        for inflation in inflations:
            result = mc.run_mcl(weak_matrix, inflation=inflation)
            clusters = mc.get_clusters(result)
            Q = mc.modularity(matrix=result, clusters=clusters)
            # print("Threshold:", threshold, "inflation:", inflation, "modularity:", Q)
            Qs_matrix[thresholds.index(threshold)][inflations.index(inflation)] = Q

    threshold_index, inflation_index = np.unravel_index(Qs_matrix.argmax(), Qs_matrix.shape)

    threshold_filter = thresholds[threshold_index]
    inflation_power = inflations[inflation_index]

    print('Threshold : {}'.format(threshold_filter))
    print('Inflation Power : {}'.format(inflation_power))

    threshold_indices = threshold_filter > matrix

    weak_matrix = matrix.copy()
    weak_matrix[threshold_indices] = 0

    # Create the initial graph
    graph = plot_graph(weak_matrix, labels)

    ## CLUSTERING
    np.fill_diagonal(weak_matrix, 1)
    weak_matrix = normalize(weak_matrix)

    for _ in range(200):
        weak_matrix = normalize(inflate(expand(weak_matrix, 4), inflation_power))

        num_ones = np.count_nonzero(weak_matrix == 1)
        num_zeros = (weak_matrix == 0).sum()

        if num_ones == len(weak_matrix) and num_ones + num_zeros == len(weak_matrix) * len(weak_matrix):
            # we converged
            break

    ## Transform the MCL result weak_matrix to clusters
    clusters, labels_clusters = translate_clustering(weak_matrix, labels)

    clusters_color = {}

    # Colors the nodes
    color_dict = {}

    colors = generate_random_color(len(clusters))

    for cluster_id, cluster_label in labels_clusters.items():

        color = colors[cluster_id]

        # Add color to the cluster
        clusters_color[cluster_id] = color
        for label in cluster_label:
            color_dict[label] = color

    if plot:
        color_map = []

        for node in graph:
            color_map.append(color_dict[node])

        # Draw the graph
        plt.figure()
        pos = nx.drawing.spring_layout(graph)
        nx.draw(graph, node_size=150, font_size=8, pos=pos, node_color=color_map, with_labels=True)

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


if __name__ == '__main__':
    main()
