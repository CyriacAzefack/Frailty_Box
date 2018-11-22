import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Data_Drift.MCL import mcl_clusterinig
from Utils import *

sns.set_style('darkgrid')


def main():
    data_name = 'hh101'

    data = pick_dataset(data_name)
    similarity_metric = 'relax'
    window_size = dt.timedelta(days=5)

    drift_detector(data, window_size, similarity_metric, plot=True)

    plt.title(similarity_metric + " Analysis")
    plt.show()


def drift_detector(data, window_size=dt.timedelta(days=5), similarity_metric="sleeping", plot=True):
    """
    Detect the drift point in the data, only for activities occurrence time
    :param data:
    :param window_size: [Timedelta Object] Duration of a time window
    :return:
    """

    search_end_time = data.date.max().to_pydatetime() - window_size  # End time of the dataset

    window_start_time = data.date.min().to_pydatetime()

    windows_occ_times = {}

    labels = data.label.unique()

    # Compute the timestamp of all the events
    # Timestamp : Nb of seconds since the beginning of the day
    data['day_date'] = data['date'].dt.date.apply(lambda x: dt.datetime.combine(x, dt.datetime.min.time()))
    data['timestamp'] = (data['date'] - data['day_date']).apply(lambda x: x.total_seconds())

    window_id = 0
    while window_start_time <= search_end_time:

        window_end_time = window_start_time + window_size

        window_data = data[(data.date >= window_start_time) & (data.date < window_end_time)].copy()

        labels_dict = {}

        for label in labels:
            label_occ_time = window_data[window_data.label == label].timestamp.values
            labels_dict[label] = label_occ_time

        window_start_time += dt.timedelta(days=1)

        windows_occ_times[window_id] = labels_dict

        window_id += 1

    similarity_matrix = np.empty((window_id, window_id))

    for idA, windA in windows_occ_times.items():
        for idB, windB in windows_occ_times.items():
            if idA == idB:
                similarity_matrix[idA][idB] = 1
            else:
                similarity_matrix[idA][idB] = similarity(windA, windB, label=similarity_metric)

    graph_labels = ['W_{}'.format(i) for i in windows_occ_times.keys()]

    clusters, clusters_color = mcl_clusterinig(similarity_matrix, graph_labels, inflation_power=3, plot=plot, gif=False)

    ##################################
    ### VALIDATION OF THE CLUSTERS ###
    ##################################

    cluster_validation_matrix = np.empty((len(clusters), len(clusters)))

    ## Intra-Cluster similarity.

    for cluster_id, window_ids in clusters.items():
        window_ids = np.asarray(window_ids)

        # Extract the cluster from the similarity_matrix

        sub_matrix = similarity_matrix[window_ids[:, None], window_ids]

        # The similarity minimum between 2 object of the same cluster
        cluster_validation_matrix[cluster_id][cluster_id] = sub_matrix.mean()

    ## Inter-Cluster similarity.
    for cluster_id_1, window_ids_1 in clusters.items():
        for cluster_id_2, window_ids_2 in clusters.items():
            if cluster_id_1 != cluster_id_2:
                window_ids_1 = np.asarray(window_ids_1)
                window_ids_2 = np.asarray(window_ids_2)

                # Extract the cluster from the similarity_matrix
                sub_matrix = similarity_matrix[window_ids_1[:, None], window_ids_2]

                # The similarity mean between
                cluster_validation_matrix[cluster_id_1][cluster_id_2] = sub_matrix.mean()

    print(cluster_validation_matrix)

    # sns.heatmap(original_similarity_matrix, vmin=0, vmax=1)
    # plt.title("Similarity Matrix between Time Windows")

    plt.figure()
    sns.heatmap(cluster_validation_matrix, vmin=0, vmax=1, annot=True, fmt=".2f")
    plt.title("Cluster Distances")

    # Cluster Histograms
    plt.figure()
    bins = []
    for cluster_id, window_ids in clusters.items():
        occ_times = []

        if len(window_ids) < 2:
            continue

        for window_id in window_ids:
            occ_times += list(windows_occ_times[window_id][similarity_metric])

        occ_times = [x / 3600 for x in occ_times]
        occ_times = np.asarray(occ_times)
        # if len(bins) == 0:
        #     _, bins, _ = plt.hist(occ_times, bins=100, alpha=0.5, label='Cluster {}'.format(cluster_id))
        # else :
        #     plt.hist(occ_times, bins=bins,  alpha=0.5, label='Cluster {}'.format(cluster_id))

        sns.kdeplot(occ_times, label='Cluster {}'.format(cluster_id), shade=True, color=clusters_color[cluster_id])

    plt.title("Cluster Occurrence Time distributions")
    plt.legend(loc='upper right')
    plt.xlim(0, 24)

    ## Visualization on the whole data
    fig = plt.figure()
    data2 = data[data.label == similarity_metric].reset_index().copy()
    data2['cluster'] = -1
    data_start_time = data.date.min().to_pydatetime()
    for cluster_id, window_ids in clusters.items():
        color = clusters_color[cluster_id]

        result = window_separation(window_ids)

        for interval in result:
            win_start_time = data_start_time + interval[0] * window_size
            win_end_time = data_start_time + interval[1] * window_size
            window_data = data2[(data2.label == similarity_metric) & (data2.date >= win_start_time)
                                & (data2.date < win_end_time)]

            plt.plot(window_data.date, window_data.timestamp / 3600, 'bo', color=color)

        # for window_id in window_ids:
        #     win_start_time = data_start_time + window_id * window_size
        #     win_end_time = win_start_time + window_size
        #     window_data = data2[(data2.label == similarity_metric) & (data2.date >= win_start_time)
        #                        & (data2.date < win_end_time)]
        #
        #     data2.loc[(data2.label == similarity_metric) & (data2.date >= win_start_time)
        #          & (data2.date < win_end_time), 'cluster'] = cluster_id

    plt.title(similarity_metric)

    return None


def similarity(windowA, windowB, label='sleeping'):
    """
    Compute the similarity between 2 time windows
    :param windowA:
    :param windowB:
    :return:
    """

    arrayA = np.asarray(windowA[label])
    arrayB = np.asarray(windowB[label])

    if (len(arrayA) == 0) and (len(arrayB) == 0):
        return 1
    elif (len(arrayA) == 0) or (len(arrayB) == 0):
        return 0

    _, p_val = stats.ttest_ind(arrayA, arrayB, equal_var=True)

    if np.isnan(p_val):
        return 0

    return p_val


def plot_cluster_heatmap(matrix, labels):
    """
    Plot the cluster heatmap
    :param matrix:
    :param labels:
    :return:
    """
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, round(matrix[i, j], 2),
                           ha="center", va="center")

    ax.set_title("Clusters Distance")
    fig.tight_layout()


def window_separation(array):
    result = []
    c = None
    interval = []
    for x in array:
        if c == None:
            interval.append(x)
            c = x
            continue

        if x - c > 1:
            if c == interval[0]:
                c += 1
            interval.append(c)
            result.append(interval)
            interval = []
            interval.append(x)

        c = x

    interval.append(c)
    result.append(interval)

    return result


if __name__ == '__main__':
    main()
