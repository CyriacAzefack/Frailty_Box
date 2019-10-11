from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from Data_Drift.Features_Extraction import *
from Data_Drift.MCL import mcl_clusterinig


def main():
    data_name = 'KA'

    data = pick_dataset(data_name)

    window_size = dt.timedelta(days=14)

    label = "leave_home"

    # windows_dataset = activity_drift_detector(data, window_size, label, plot=True, gif=False)

    # windows_dataset.to_csv("{}_windows_data.csv".format(data_name), period_ts_index=False)

    plt.show()


def activity_drift_detector(data, time_window_duration, label, plot=True, gif=False):
    """
    Detect the drift point in the data, focusing on one activity
    :param data:
    :param window_size: [Timedelta Object] Duration of a time window
    :return:
    """

    data = data[data.label == label].copy()

    ##########################
    ## Pre-Treatment of data #
    ##########################

    data['day_date'] = data['date'].dt.date.apply(lambda x: dt.datetime.combine(x, dt.datetime.min.time()))
    data['timestamp'] = (data['date'] - data['day_date']).apply(lambda x: x.total_seconds())

    data['duration'] = (data['end_date'] - data['date']).apply(lambda x: x.total_seconds() / 3600)

    #############################
    ## Creation of time windows #
    #############################

    time_windows_data = create_time_windows(data, time_window_duration)

    #########################
    ## Clustering Algorithm #
    #########################

    # clusters, clusters_color = features_clustering(time_windows_data, activity_labels = [label])

    clusters, clusters_color = similarity_clustering(time_windows_data, label=label, plot=True, gif=False)

    ##################################
    ### VALIDATION OF THE CLUSTERS ###
    ##################################

    # Check if there is a real difference between the clusters

    cluster_val_occ_matrix = np.empty((len(clusters), len(clusters)))
    cluster_val_dur_matrix = np.empty((len(clusters), len(clusters)))

    clusters_occ_times = []
    clusters_duration_times = []
    for cluster_id, window_ids in clusters.items():
        occ_times = []
        durations = []

        for window_id in window_ids:
            occ_times += list(time_windows_data[window_id].timestamp.values)
            durations += list(time_windows_data[window_id].duration.values)

        occ_times = np.asarray(occ_times)
        clusters_occ_times.append(occ_times)

        durations = np.asarray(durations)
        clusters_duration_times.append(durations)

    for i in range(len(clusters)):
        for j in range(len(clusters)):
            cluster_val_occ_matrix[i][j] = ks_similarity(clusters_occ_times[i], clusters_occ_times[j])
            cluster_val_dur_matrix[i][j] = ks_similarity(clusters_duration_times[i], clusters_duration_times[j])

    valid_clusters_occ = len(cluster_val_occ_matrix < 0.05) / 2

    print("{} meaningful clustering for activity occurrence time".format(valid_clusters_occ))

    valid_clusters_dur = len(cluster_val_dur_matrix < 0.05) / 2

    print("{} meaningful clustering for activity duration time".format(valid_clusters_dur))

    plt.figure()
    sns.heatmap(cluster_val_occ_matrix, vmin=0, vmax=1, annot=True, fmt=".2f")
    plt.title("Validation of clusters Activity Occurrence Time")

    plt.figure()
    sns.heatmap(cluster_val_dur_matrix, vmin=0, vmax=1, annot=True, fmt=".2f")
    plt.title("Validation of Clusters Activity Duration Time")

    # ## Inter-Cluster ks_similarity.
    # for cluster_id_1, window_ids_1 in clusters.items():
    #     for cluster_id_2, window_ids_2 in clusters.items():
    #         if cluster_id_1 != cluster_id_2:
    #             window_ids_1 = np.asarray(window_ids_1)
    #             window_ids_2 = np.asarray(window_ids_2)
    #
    #             # Extract the cluster from the similarity_matrix
    #             sub_matrix = similarity_matrix[window_ids_1[:, None], window_ids_2]
    #
    #             # The ks_similarity mean between
    #             cluster_validation_matrix[cluster_id_1][cluster_id_2] = sub_matrix.mean()
    #
    #
    # plt.figure()
    # sns.heatmap(cluster_validation_matrix, vmin=0, vmax=1, annot=True, fmt=".2f")
    # plt.title("Cluster Distances")

    ############################"
    ## CLUSTER INTERPRETATION ##
    ############################

    # Cluster Occurrence Time
    fig, (ax1, ax2) = plt.subplots(2)
    fig2, (ax11, ax22) = plt.subplots(2)

    for cluster_id, window_ids in clusters.items():
        occ_times = []
        durations = []

        if len(window_ids) < 4:
            continue

        for window_id in window_ids:
            occ_times += list(time_windows_data[window_id].timestamp.values)
            durations += list(time_windows_data[window_id].duration.values)

        occ_times = [x / 3600 for x in occ_times]
        occ_times = np.asarray(occ_times)

        durations = np.asarray(durations)

        ax1.hist(occ_times, bins=100, alpha=0.3, label='Cluster {}'.format(cluster_id),
                 color=clusters_color[cluster_id])

        ax11.hist(durations, bins=100, alpha=0.3, label='Cluster {}'.format(cluster_id),
                  color=clusters_color[cluster_id])


        sns.kdeplot(occ_times, label='Cluster {}'.format(cluster_id), shade_lowest=False, shade=True,
                    color=clusters_color[cluster_id], ax=ax2)

        sns.kdeplot(durations, label='Cluster {}'.format(cluster_id), shade_lowest=False, shade=True,
                    color=clusters_color[cluster_id], ax=ax22)

    ax1.set_title("{}\nCluster : Occurrence Time distribution".format(label))
    ax1.set_xlabel('Hour of the day')
    ax1.set_ylabel('Number of occurrences')
    ax1.set_xlim(0, 24)

    ax2.set_title("Density Distribution")
    ax2.set_xlabel('Hour of the day')
    ax2.set_ylabel('Density')
    ax2.set_xlim(0, 24)

    ax11.set_title("{}\nCluster : Activity Duration Distribution".format(label))
    ax11.set_xlabel('Duration (hours)')
    ax11.set_ylabel('Number of occurrences')
    xmin, xmax = ax11.get_xlim()

    ax22.set_title("Density Distribution")
    ax22.set_xlabel('Duration (hours)')
    ax22.set_ylabel('Density')
    ax22.set_xlim(xmin, xmax)


    plt.legend(loc='upper right')




    return None


def features_clustering(time_windows_data, activity_labels, plot=True):
    """
    Clustering of the time windows with features
    :param time_windows_data:
    :param plot:
    :return: clusters (dict with cluster_id as key and corresponding time_windows id list as item),
    clusters_color (list of clusters colors)
    """
    nb_windows = len(time_windows_data)

    time_windows_labels = ['W_' + str(i) for i in range(len(time_windows_data))]

    data_features = pd.DataFrame()  # Dataset for clustering
    for window_id in range(nb_windows):
        tw_data = time_windows_data[window_id]
        tw_features = activities_features(data=tw_data, activity_labels=activity_labels)

        tw_df = pd.DataFrame.from_dict(tw_features, orient='columns')
        if len(data_features) == 0:
            data_features = tw_df
        else:
            data_features = data_features.append(tw_df, ignore_index=True)

    data_features.fillna(0, inplace=True)
    norm_data = StandardScaler().fit_transform(data_features)

    # Clustering
    linked = linkage(norm_data, method='ward')

    plt.figure(figsize=(10, 7))

    dendrogram(
        linked,
        orientation='top',
        labels=time_windows_labels,
        distance_sort='descending',
        show_leaf_counts=True)

    plt.title("Dendogram")
    max_d = 7
    plt.axhline(y=max_d, c='k')

    clusters = fcluster(linked, max_d, criterion='distance')
    nb_clusters = len(set(list(clusters)))

    print('{} clusters detected using dendograms :)'.format(nb_clusters))

    colors = generate_random_color(nb_clusters)

    # Compute TSNE
    vizu_model = TSNE(learning_rate=100)
    # Fitting Model
    transformed = vizu_model.fit_transform(norm_data)

    plt.figure()
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]

    clusters_dict = {}

    # Associate to each cluster his time_windows
    for cluster_id in set(clusters):
        ids = [i for i, x in enumerate(clusters) if x == cluster_id]
        cluster_id -= 1  # cause the clustering algo cluster starts at 1
        clusters_dict[cluster_id] = ids

        plt.scatter(x_axis[ids], y_axis[ids], c=colors[cluster_id], label='Cluster ' + str(cluster_id))

    # plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
    #             marker="x", color='k', s=150, linewidths=5, zorder=10)

    # plt.scatter(x_axis, y_axis)
    plt.title('2D-Projection')
    plt.legend()

    return clusters_dict, colors


def similarity_clustering(time_windows_data, label, plot=True, gif=False):
    """
    Clustering of the time windows using a similarity metric
    :param time_windows_data:
    :return:
    """
    nb_windows = len(time_windows_data)

    time_windows_data = [data[data.label == label].copy() for data in time_windows_data]

    ######################
    ## Similarity Matrix #
    ######################

    print('Starting building similarity matrix...')

    similarity_matrix = np.zeros((nb_windows, nb_windows))

    for i in range(nb_windows):
        tw_data_A = time_windows_data[i]
        for j in range(i, nb_windows):
            tw_data_B = time_windows_data[j]

            # TODO : Add weights for the different type of similarity

            # 1- Occurrence time similarity
            arrayA = tw_data_A.timestamp.values
            arrayB = tw_data_B.timestamp.values

            occ_time_similarity = ks_similarity(arrayA, arrayB)

            # arrayA = tw_data_A.duration.values
            # arrayB = tw_data_B.duration.values
            #
            # duration_similarity = hi(arrayA, arrayB)
            # similarity = (duration_similarity + occ_time_similarity) / 2

            similarity = occ_time_similarity

            similarity_matrix[i][j] = similarity

    # Little trick for speed purposes ;)
    missing_part = np.transpose(similarity_matrix.copy())
    np.fill_diagonal(missing_part, 0)
    similarity_matrix = similarity_matrix + missing_part

    print('Finish building similarity matrix...')

    # Plotting similarity matrix
    # plt.figure()
    # sns.heatmap(similarity_matrix, vmin=0, vmax=1)
    # plt.title("Similarity Matrix between Time Windows")
    # plt.show()

    ###################
    ## MCL Clustering #
    ###################

    graph_labels = ['W_{}'.format(i) for i in range(nb_windows)]

    clusters, clusters_color = mcl_clusterinig(matrix=similarity_matrix, labels=graph_labels, inflation_power=2,
                                               plot=plot, gif=gif)

    return clusters, clusters_color




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


def create_time_windows(data, time_window_duration):
    """
    Slide a time window among the data and create the different time_windows
    :return: the list of time windows data
    """
    data_start_time = data.date.min().to_pydatetime()  # Starting time of the dataset

    # Starting point of the last time window
    data_end_time = data.date.max().to_pydatetime() - time_window_duration

    window_start_time = data_start_time

    time_windows_data = []

    while window_start_time <= data_end_time:
        window_end_time = window_start_time + time_window_duration

        window_data = data[
            (data.date >= window_start_time) & (data.date < window_end_time)].copy()

        time_windows_data.append(window_data)

        window_start_time += dt.timedelta(days=1)  # We slide the time window by 1 day

    return time_windows_data



if __name__ == '__main__':
    main()
