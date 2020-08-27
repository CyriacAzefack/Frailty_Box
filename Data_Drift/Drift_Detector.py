import math

import matplotlib.dates as dat
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcl
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from Data_Drift.Features_Extraction import *

sns.set_style('darkgrid')
sns.set(font_scale=1.4)


def main():
    data_name = 'hh113'

    # data = pick_dataset(data_name, nb_days=-1)

    path = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/input/Drift_Toy/3_drift_toy_data_21.csv"
    data = pick_custom_dataset(path)
    window_size = dt.timedelta(days=7)

    label = "eating"
    profile = "time"

    clusters, silhouette = activity_drift_detector(data, window_size, label, validation=True, profile=profile,
                                                   display=True)
    plt.show()

    start_date = data.date.min().to_pydatetime()

    cluster_colors = generate_random_color(len(clusters))
    display_behavior_evolution(clusters=clusters, colors=cluster_colors, start_date=start_date,
                               time_window_duration=window_size)

    # windows_dataset.to_csv("{}_windows_data.csv".format(data_name), period_ts_index=False)


def clustering_algorithm(distance_matrix, nb_clusters):
    model = AgglomerativeClustering(n_clusters=nb_clusters, affinity='precomputed', linkage='average')
    labels = model.fit_predict(distance_matrix)

    return labels


def find_optimal_nb_clusters(distance_matrix, display=False):
    """
    Plot the differents silhoutte values
    :return:
    """

    range_n_clusters = list(range(2, 11))

    silhouettes = []

    # filename = './output/encoded_data.csv'
    # outfile = open(filename, 'wb')
    # pickle.dump(data, outfile)
    # outfile.close()

    optimal_n_clusters = 1
    avg_silhouette = 0

    for n_clusters in range_n_clusters:

        cluster_labels = clustering_algorithm(distance_matrix, nb_clusters=n_clusters)

        silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        silhouettes.append(silhouette_avg)
        # sample_silhouette_values = silhouette_samples(distance_matrix, cluster_labels, metric='precomputed')
        if display:
            print(f"For n_clusters ={n_clusters} \tSilhouette = {silhouette_avg}")
        # print("\tThe average silhouette_score is :", silhouette_avg)
        # print("\tThe average silhouette_score is :", silhouette_avg)
        if silhouette_avg > avg_silhouette:
            avg_silhouette = silhouette_avg
            optimal_n_clusters = n_clusters

    print(f"Chosen Number of Clusters : {optimal_n_clusters}")

    if display:
        linked = hcl.linkage(squareform(distance_matrix), method="average")
        plt.figure(figsize=(10, 7))
        hcl.dendrogram(
            linked,
            orientation='top',
            # labels=labels,
            distance_sort='ascending',
            show_leaf_counts=True)

        plt.title("Dendogram")
        plt.ylabel('height')
        plt.xlabel('Time Windows ID')

        plt.figure()
        plt.plot([1] + range_n_clusters, [0] + silhouettes, marker='o')
        plt.axvline(x=optimal_n_clusters, ymin=0, ymax=optimal_n_clusters + 0.1, linestyle='--', color='r')
        plt.ylabel("Silhouette")
        plt.xlabel("Nombre de clusters")
        plt.xticks(range_n_clusters)
        plt.show()

    return optimal_n_clusters


def activity_drift_detector(data, time_window_duration, label, profile='time', validation=False, display=False):
    """
    Detect the drift point in the data, focusing on one activity
    :param display:
    :param profile:
    :param label:
    :param time_window_duration:
    :param validation:
    :param data:
    :return:
    """

    data = data[data.label == label].copy()

    if len(data) == 0:
        raise ValueError(f"Zero occurrences of '{label}' in the dataset")

    ##########################
    ## Pre-Treatment of data #
    ##########################

    data['day_date'] = data['date'].dt.date.apply(lambda x: dt.datetime.combine(x, dt.datetime.min.time()))
    data['timestamp'] = (data['date'] - data['day_date']).apply(lambda x: x.total_seconds())

    data['duration'] = (data['end_date'] - data['date']).apply(lambda x: x.total_seconds() / 60)  # In minutes

    max_duration = data['duration'].max()
    #############################
    ## Creation of time windows #
    #############################

    time_windows_data = create_time_windows(data, time_window_duration)

    #########################
    ## Clustering Algorithm #
    #########################

    # clusters, clusters_color = features_clustering(time_windows_data, activity_labels=[label])

    nb_clusters = None
    clusters, clusters_color, silhouette = similarity_clustering(time_windows_data, nb_clusters=nb_clusters,
                                                                 profile=profile,
                                                                 display=display)

    ##################################
    ### VALIDATION OF THE CLUSTERS ###
    ##################################

    # Check if there is a real difference between the clusters
    if validation:

        ############################"
        ## CLUSTER INTERPRETATION ##
        ############################

        # Cluster Occurrence Time
        # fig, (ax1, ax2) = plt.subplots(2)
        # fig2, (ax11, ax22) = plt.subplots(2)

        durations_list = []
        for cluster_id, window_ids in clusters.items():
            occ_times = []
            durations = []

            if len(window_ids) < 4:
                continue

            for window_id in window_ids:
                occ_times += list(time_windows_data[window_id].timestamp.values)
                durations += list(time_windows_data[window_id].duration.values)

            if profile == 'time':
                occ_times = [x / 3600 for x in occ_times]
                occ_times = np.asarray(occ_times)

                plot_time_circle(data_points=occ_times, title=f'Cluster {1 + cluster_id}', plot=True)
                # ax1.hist(occ_times, range=(0, 24), bins=48, alpha=0.3, label='Cluster {}'.format(cluster_id),
                #          color=clusters_color[cluster_id])
                # sns.kdeplot(occ_times, label='Cluster {}'.format(cluster_id), shade_lowest=False, shade=True, bw=.5,
                #             color=clusters_color[cluster_id], ax=ax2)
                # ax2.xaxis.set_ticks(np.arange(0, 24, 1))

            elif profile == 'duration':
                durations = np.asarray(durations) / 60

                durations_list.append(durations)

        if profile == 'time':
            plt.show()

        elif profile == 'duration':
            fig, ax = plt.subplots()

            # build a box plot
            ax.boxplot(durations_list, vert=False)

            # title and axis labels
            ax.set_xlabel('Durée (heures)')
            yticklabels = [f'Cluster {i + 1}' for i in range(len(durations_list))]
            ax.set_yticklabels(yticklabels)

            # add horizontal grid lines
            ax.yaxis.grid(True)

            # show the plot
            plt.show()

    return clusters, silhouette


def features_clustering(time_windows_data, activity_labels, plot=True):
    """
    Clustering of the time windows with features
    :param activity_labels:
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
    linked = hcl.linkage(norm_data, method='ward')

    plt.figure(figsize=(10, 7))

    hcl.dendrogram(
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


def similarity_clustering(time_windows_data, nb_clusters=None, profile='time', display=True):
    """
    Clustering of the time windows using a similarity metric
    :param profile:
    :param nb_clusters:
    :param display:
    :param time_windows_data:
    :return:
    """
    nb_windows = len(time_windows_data)

    ######################
    ## Similarity Matrix #
    ######################

    print('Starting building similarity distance_matrix...')

    similarity_matrix = np.zeros((nb_windows, nb_windows))

    # print('Daily Profile : ', profile)
    for i in trange(nb_windows, desc='Similarity Matrix Construction'):
        tw_data_A = time_windows_data[i]
        for j in range(i + 1, nb_windows):
            tw_data_B = time_windows_data[j]

            # TODO : Add weights for the different type of similarity

            # 1- Occurrence time similarity
            timesA = tw_data_A.timestamp.values
            durationsA = tw_data_A.duration.values
            timesB = tw_data_B.timestamp.values
            durationsB = tw_data_B.duration.values

            # A_hist_2D, _, _ = np.histogram2d(timesA, durationsA, bins=24, range=[[0, 24], [0, max_duration]],
            #                                  density=True)
            # B_hist_2D, _, _ = np.histogram2d(timesB, durationsB, bins=24, range=[[0, 24], [0, max_duration]],
            #                                  density=True)
            #
            # plt.hist2d(timesA/3600, durationsA, bins=30, cmap=plt.cm.jet)
            # plt.show()
            #
            #
            # sns.jointplot("timestamp", "duration", data=tw_data_A,
            #               kind="kde", space=0, color="g", xlim=(0,24), ylim=(0, max_duration))
            #
            #
            # sns.jointplot("timestamp", "duration", data=tw_data_B,
            #               kind="kde", space=0, color="r", xlim=(0,24), ylim=(0, max_duration))
            #
            # plt.show()
            #
            # plt.hexbin(timesA,durationsA)
            # plt.show()

            # arrayA = tw_data_A[['timestamp', 'duration']].values
            # arrayB = tw_data_B[['timestamp', 'duration']].values
            # similarity = hotelling_test(arrayA, arrayB)

            if profile == 'time':
                similarity = ks_similarity(timesA, timesB)
            elif profile == 'duration':
                similarity = ks_similarity(durationsA, durationsB)
            else:
                raise ValueError(f'"{profile}" is not a valid profile')
            # similarity = occ_time_similarity

            similarity_matrix[i][j] = similarity

    # Little trick for speed purposes ;)
    missing_part = np.transpose(similarity_matrix.copy())
    similarity_matrix = similarity_matrix + missing_part
    np.fill_diagonal(similarity_matrix, 1)
    similarity_matrix[similarity_matrix > 1] = 1

    print('Finish building similarity distance_matrix...')

    distance_matrix = 1 - np.asarray(similarity_matrix)

    if display:
        # Plotting similarity distance_matrix
        plt.figure()
        sns.heatmap(distance_matrix, vmin=0, vmax=1, center=0.5)
        plt.title("Distance Matrix between Time Windows")
        plt.xlabel('Time Window ID')
        plt.ylabel('Time Window ID')
        # plt.show()

        plt.figure()
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=6)
        results = mds.fit(distance_matrix)

        coords = results.embedding_

        plt.scatter(coords[:, 0], coords[:, 1], marker='o')
        plt.title('MDS Distance matrix')
        plt.show()

    # labels = [f'tw_{i}' for i in range(nb_windows)]
    # mcl_clusterinig(similarity_matrix, labels, plot=True)

    if not nb_clusters:
        nb_clusters = find_optimal_nb_clusters(distance_matrix=distance_matrix, display=display)

    cluster_labels = clustering_algorithm(distance_matrix=distance_matrix, nb_clusters=nb_clusters)

    silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')

    clusters_dict = {}
    # Associate to each cluster his time_windows
    for cluster_id in range(nb_clusters):
        ids = [i for i, x in enumerate(cluster_labels) if x == cluster_id]
        # cause the clustering algo cluster starts at 1
        clusters_dict[cluster_id] = ids

    colors = generate_random_color(nb_clusters)



    ###################
    ## MCL Clustering #
    ###################
    #
    # graph_labels = ['W_{}'.format(i) for i in range(nb_windows)]
    #
    # clusters, clusters_color = mcl_clusterinig(matrix=similarity_matrix, labels=graph_labels, inflation_power=2,
    #                                            plot=display, gif=False)

    return clusters_dict, colors, silhouette_avg


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
        if c is None:
            interval.append(x)
            c = x
            continue

        if x - c > 1:
            if c == interval[0]:
                c += 1
            interval.append(c)
            result.append(interval)
            interval = [x]
        c = x

    interval.append(c)
    result.append(interval)

    return result


def create_time_windows(data, time_window_duration):
    """
    Slide a time window among the data and create the different time_windows
    :return: the list of time windows data
    """
    data_start_time = data.date.min().to_pydatetime()  # Starting time of the log_dataset

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


def time_periods_from_windows(window_ids):
    """
        Compute the time period where a cluster is valid
        :param window_ids: list of time_window id
        :return:
        """

    time_periods = []
    current_time_period = (window_ids[0],)

    # Create time period interval
    for i in range(len(window_ids) - 1):
        if window_ids[i + 1] != window_ids[i] + 1:
            current_time_period += (window_ids[i],)
            time_periods.append(current_time_period)
            current_time_period = (window_ids[i + 1],)
    current_time_period += (window_ids[-1],)
    time_periods.append(current_time_period)

    return time_periods


def display_behavior_evolution(clusters, colors, start_date, time_window_duration):
    """
    Plot the evolution of the different behavior throughout the log_dataset
    :param clusters:
    :param colors:
    :return:
    """
    fig, ax = plt.subplots()
    # xfmt = dat.DateFormatter('%d-%m-%y')
    # months = dat.MonthLocator()  # every month
    # monthsFmt = dat.DateFormatter('%b %Y')  # Eg. Jan 2012

    months = dat.AutoDateLocator()
    monthsFmt = dat.AutoDateFormatter(locator=months)

    # format the ticks
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.xaxis.set_minor_locator(months)

    for cluster_id, window_ids in clusters.items():
        lvl = cluster_id * 2

        time_periods = time_periods_from_windows(window_ids)

        print("Cluster {} :".format(cluster_id))
        for period in time_periods:
            start_date_period = start_date + period[0] * dt.timedelta(days=1)
            end_date_period = start_date + (1 + period[1]) * dt.timedelta(days=1)

            print("\t{} - {}".format(start_date_period, end_date_period))

            if time_periods.index(period) == 0:
                plt.text(dat.date2num(start_date_period), lvl, 'Cluster {}'.format(1 + cluster_id), fontsize=16)
            ax.hlines(lvl, dat.date2num(start_date_period), dat.date2num(end_date_period),
                      label='Cluster {}'.format(1 + cluster_id),
                      linewidth=75, color=colors[cluster_id])

    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.autofmt_xdate()
    # plt.title("Activity : '{}'".format(self.label))
    plt.xlabel('Date')
    plt.yticks([])
    plt.show()


def plot_time_circle(data_points, title, plot=False):
    """
    plot time data points into a circle
    :param data_points:
    :return:
    """

    degrees = []
    x = []
    y = []

    for point in data_points:
        alpha = 2 * np.pi * point / 24
        x.append(math.sin(alpha))
        y.append(math.cos(alpha))

        degrees.append(360 * point / 24)

    if plot:
        radians = np.deg2rad(degrees)

        bin_size = 4
        a, b = np.histogram(degrees, bins=np.arange(0, 360 + bin_size, bin_size))
        centers = np.deg2rad(np.ediff1d(b) // 2 + b[:-1])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ax.set_title(title)

        # hours = np.arange(24)
        #
        # hours = [f"{i:02}h" for i in hours]

        ax.set_xticklabels(['00h', '3h', '6h', '9h', '12h', '15h', '18h', '21h'])
        # ax.set_xticklabels(hours)
        ax.tick_params(direction='out', length=6, width=6, colors='r', grid_alpha=1, labelsize=20)

    return x, y


if __name__ == '__main__':
    main()
