from __future__ import absolute_import

import json
import time as t
from optparse import OptionParser

import math
import matplotlib.dates as dat
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from Data_Drift.Features_Extraction import *
from Data_Drift.MCL import mcl_clusterinig
from Utils import *

sns.set_style('darkgrid')
sns.set(font_scale=2)


def main():
    # Default values
    """
    Extraction of parameters
    :return:
    """

    parser = OptionParser(usage='Usage: %prog <options>')
    parser.add_option('-n', '--dataset_name', help='Name of the Input event log', dest='dataset_name', action='store',
                      type='string')
    parser.add_option('-w', '--window_size', help='Size of the time windows in days', dest='window_size',
                      action='store', type='int')
    parser.add_option('-p', '--plot', help='Display of the behavior changes', dest='plot', action='store_true',
                      default=False)
    parser.add_option('-d', '--debug', help='Display of the Drift methods used for behavior changes detection',
                      dest='debug', action='store_true', default=False)
    parser.add_option('-b', '--behavior', help='Behavior to focus on', dest='behavior_type', action='store',
                      type='choice', choices=['duration', 'start_time'], default='start_time')
    parser.add_option('-m', '--drift_method', help='Drift method used', dest='drift_method', action='store',
                      type='choice', choices=['features', 'stat_test', 'histogram_intersect', 'density_intersect',
                                              'all_methods'], default='stat_test')

    (options, args) = parser.parse_args()

    # Mandatory Options
    if options.dataset_name is None:
        print("The name of the Input event log is missing\n")
        parser.print_help()
        exit(-1)

    if options.window_size is None:
        print("The size of the time windows is missing\n")
        parser.print_help()
        exit(-1)

    dataset_name = options.dataset_name
    window_size = options.window_size

    if options.behavior_type == 'start_time':
        behavior_type = Behavior.OCC_TIME
    elif options.behavior_type == 'duration':
        behavior_type = Behavior.DURATION

    drift_method = options.drift_method
    plot = options.plot
    debug = options.debug

    labels = None
    labels = ['eating']

    df = data_drift(dataset_name, window_size, behavior_type, drift_method, plot, debug, labels)


def data_drift(dataset_name, window_size, behavior_type, drift_method, plot, debug, labels):
    OUTDIR = '../output/Data_Drift'

    print("Dataset Name : {}".format(dataset_name.upper()))
    print("Windows size : {}".format(window_size))
    print("Behavior Type : {}".format(behavior_type))
    print("Drift Method : {}".format(drift_method))
    print("Display : {}".format(plot))
    print("Mode debug : {}".format(debug))

    data = pick_dataset(dataset_name)

    time_window_size = dt.timedelta(days=window_size)

    inhabitant_behavior = Behavior(dataset=data, time_window_duration=time_window_size)
    inhabitant_behavior.create_activities_behavior()

    if drift_method != "All Methods":

        changes = inhabitant_behavior.drift_detector(behavior_type=behavior_type, method=drift_method, plot=plot,
                                                     labels=labels, debug=debug)
        changes_str = stringify_keys(changes)
        changes_str.pop('clusters', None)
        # print(json.dumps(changes_str, indent=4))

        df = pd.DataFrame(
            columns=['label', 'nb_behaviors', 'behavior_duration', 'nb_drift_points', 'interpretation', 'silhouette'])
        for label, change in changes_str.items():
            df.loc[len(df)] = [label, change['nb_clusters'], change['duration'], change['nb_drift_points'],
                               change['interpretation'], '{:.2f}'.format(change['silhouette'])]

        # if not os.path.exists(outdir):
        #     os.mkdir(outdir)

        fullname = os.path.join(OUTDIR, '{}_{}_{}_W{}.csv'.format(dataset_name.upper(), behavior_type.replace(' ', '-'),
                                                                  drift_method, window_size))

        df.to_csv(fullname, index=False, sep=';')

        return df


    else:
        methods = ['features', 'stat_test', 'histogram_intersect', 'density_intersect']

        for drift_method in methods:
            print('### {} ###'.format(drift_method))
            changes = inhabitant_behavior.drift_detector(behavior_type=behavior_type, method=drift_method, plot=plot,
                                                         labels=labels, debug=debug)

            changes_str = stringify_keys(changes)
            changes_str.pop('clusters', None)
            print(json.dumps(changes_str, indent=4))


class Behavior:
    OCC_TIME = 'Activity Starting Time'
    DURATION = 'Duration'
    NB_OCC = 'Number of Occurrences'

    def __init__(self, dataset, time_window_duration, instant_events=False):
        """
        Create a Behavior corresponding to a house data
        :param dataset: event sequence
        :param time_window_duration: duration of the time_window for the drift detection
        :param instant_events : True if events in the dataset have no durations
        """

        self.dataset = dataset
        self.instant_events = instant_events
        self.data_preprocessing()
        self.time_window_duration = time_window_duration
        self.labels = dataset.label.unique()
        self.activities_behavior = {}  # {'activity_label' : ActivityBehavior}
        self.time_windows_data = self.create_time_windows()
        self.begin_date = self.dataset.date.min().to_pydatetime()

    def create_activities_behavior(self):
        """
        Create all the activities behavior
        """
        for label in self.labels:
            self.activities_behavior[label] = ActivityBehavior(dataset=self.dataset,
                                                               time_window_duration=self.time_window_duration,
                                                               label=label, instant_events=self.instant_events)

    def data_preprocessing(self):
        """
        Pre-processing of the data
        :return:
        """
        self.dataset['day_date'] = self.dataset['date'].dt.date.apply(
            lambda x: dt.datetime.combine(x, dt.datetime.min.time()))
        self.dataset['timestamp'] = (self.dataset['date'] - self.dataset['day_date']).apply(
            lambda x: x.total_seconds())  # In seconds

        if not self.instant_events:
            self.dataset['duration'] = (self.dataset['end_date'] - self.dataset['date']).apply(
                lambda x: x.total_seconds())  # Duration in seconds

    def drift_detector(self, behavior_type, method, labels=None, plot=False, debug=False):
        """
        Drift detection of a specific behavior
        :param behavior_type: the behavior type for the drift detection. availables ones : Behavior.OCC_TIME,
        Behavior.DURATION
        :param method: The method use for clustering ('similarity', 'features', 'hist_intersect')
        :param plot: if True, plot the changes detected
        :return: changes
        """
        changes = {}  # {('label', (clusters, colors)) : *changes*}


        if labels is None:
            labels = self.labels

        for label in labels:
            activity_behavior = self.activities_behavior[label]
            print("### Behavior Drift on '{}' ###".format(label))
            clusters, colors = activity_behavior.drift_detector(behavior_type=behavior_type, method=method, plot=debug,
                                                                debug=debug)
            if len(clusters) > 1:
                ################################
                # EVALUATION OF THE CLUSTERING #
                ################################

                print('CLUSTERING EVALUATION'.center(30, '*'))

                db_index, dunn_index, silhouette = activity_behavior.clustering_quality(clusters,
                                                                                        behavior_type=behavior_type)
                nb_drifting_points = -1
                clusters_durations = {}
                for cluster_id, window_ids in clusters.items():
                    nb_drifting_points += len(self.time_periods_from_windows(window_ids))
                    time_periods = self.time_periods_from_windows(window_ids)
                    nb_days = 0
                    for time_period in time_periods:
                        start_date = self.begin_date + dt.timedelta(days=time_period[0])
                        end_date = self.begin_date + dt.timedelta(days=time_period[1] + 1)
                        nb_days += (end_date - start_date).days
                    clusters_durations[cluster_id] = nb_days

                ##################################
                # INTERPRETATION OF THE CLUSTERS #
                ##################################

                print('CLUSTERING INTERPRETATION'.center(30, '*'))

                clusters_interpreations = activity_behavior.clustering_interpretation(clusters,
                                                                                      behavior_type=behavior_type)

                changes[label] = {
                    'clusters': clusters,
                    'colors': colors,
                    'nb_clusters': len(clusters),
                    'nb_drift_points': nb_drifting_points,
                    'interpretation': clusters_interpreations,
                    'duration': clusters_durations,
                    'db_index': db_index,
                    'dunn_index': dunn_index,
                    'silhouette': silhouette
                }

        changes = dict(sorted(changes.items(), key=lambda kv: kv[1]['silhouette'], reverse=True))

        print(' +' * 20)
        print(' +' + 'RESULTS'.center(30, ' ') + ' +')
        print(' +' * 20)

        for label, change in changes.items():

            print('#' * 30)
            print('#' + '{}'.format(label.upper()).center(28, ' ') + '#')
            print('#' * 30)

            print('Number of behaviors detected : {}'.format(change['nb_clusters']))
            print('Silhouette : {:.2f}'.format(change['silhouette']))
            print('Number of drift points : {}'.format(change['nb_drift_points']))

            for cluster_id in range(change['nb_clusters']):
                print('Cluster {}'.format(cluster_id).center(20, '*'))
                print('Interpretation : '.ljust(20) + str(change['interpretation'][cluster_id]))
                print('Duration : '.ljust(20) + str(change['duration'][cluster_id]) + ' days')

            if plot:
                # sns.set(font_scale=5)
                # plt.rcParams['figure.figsize'] = [20, 20]
                activity_behavior = self.activities_behavior[label]
                activity_behavior.display_behavior_evolution(change['clusters'], change['colors'])
                activity_behavior.display_drift(change['clusters'], change['colors'], behavior_type)

                plt.show()

        changes.pop('clusters', None)  # Remove the detail of the clusters
        return changes

    def create_time_windows(self):
        """
        Slide a time window among the data and create the different time_windows
        :return: the list of time windows data
        """
        data_start_time = self.dataset.date.min().to_pydatetime()  # Starting time of the dataset

        # Starting point of the last time window
        data_end_time = self.dataset.date.max().to_pydatetime() - self.time_window_duration

        window_start_time = data_start_time

        time_windows_data = []

        while window_start_time <= data_end_time:
            window_end_time = window_start_time + self.time_window_duration

            window_data = self.dataset[
                (self.dataset.date >= window_start_time) & (self.dataset.date < window_end_time)].copy()

            time_windows_data.append(window_data)

            window_start_time += dt.timedelta(days=1)  # We slide the time window by 1 day

        return time_windows_data

    def time_periods_from_windows(self, window_ids):
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

    def sort_clusters(self, clusters):
        """
        Sort the behaviors by the time they first occur
        :param clusters:
        :return:
        """

        clusters_begin_tw = {}  # The first time window of the cluster
        for cluster_id, window_ids in clusters.items():
            start_tw_id = self.time_periods_from_windows(window_ids)[0][0]
            clusters_begin_tw[cluster_id] = start_tw_id

        sorted_clusters_id = [k for k in sorted(clusters_begin_tw, key=clusters_begin_tw.get, reverse=False)]

        sorted_clusters = {}
        for i in range(len(sorted_clusters_id)):
            sorted_clusters[i] = clusters[sorted_clusters_id[i]]

        return sorted_clusters


class ActivityBehavior(Behavior):

    def __init__(self, dataset, time_window_duration, label, instant_events=False):
        """
        Create an Activity Behavior
        :param dataset: event sequence on one activity
        :param time_window_duration:
        :param label:
        """

        Behavior.__init__(self, dataset, time_window_duration, instant_events)
        self.dataset = dataset[dataset.label == label]
        self.time_windows_data = [data[data.label == label].copy() for data in self.time_windows_data]
        self.label = label

    def drift_detector(self, behavior_type=Behavior.OCC_TIME, method='similarity', plot=True, debug=False):
        """
        Drift detection of this activity
        :param behavior_type: the behavior type for the drift detection
        :param
        :return: clusters, clusters_color, changes (dict like {(clusterA, clusterB) : density intersection area}
        """

        time_start = t.process_time()

        if method == 'features':
            clusters, clusters_color = self.features_clustering(behavior_type=behavior_type, plot=debug)
        else:
            clusters, clusters_color = self.similarity_clustering(behavior_type=behavior_type, method=method,
                                                                  plot=debug)

        clusters = self.sort_clusters(clusters)
        elapsed_time = dt.timedelta(seconds=round(t.process_time() - time_start, 1))

        print("\t#### Time elapsed for the clustering  : \t{} ####".format(elapsed_time))


        if plot:
            self.display_behavior_evolution(clusters, clusters_color)
            self.display_drift(clusters, clusters_color, behavior_type=behavior_type)

            plt.show()

        return clusters, clusters_color

    def features_clustering(self, behavior_type, plot=False):
        """
        Clustering of the time windows with features
        :param time_windows_data:
        :param plot: Plot all the graphs if True
        :return: clusters (dict with cluster_id as key and corresponding time_windows id list as item),
        clusters_color (list of clusters colors)
        """
        nb_windows = len(self.time_windows_data)

        time_windows_labels = ['W_' + str(i) for i in range(nb_windows)]

        data_features = pd.DataFrame()  # Dataset for clustering

        # Build the features dataset for all time windows
        for window_id in range(nb_windows):
            tw_data = self.time_windows_data[window_id]


            tw_features = activities_features(window_data=tw_data, activity_labels=[self.label])

            tw_df = pd.DataFrame.from_dict(tw_features, orient='columns')
            if len(data_features) == 0:
                data_features = tw_df
            else:
                data_features = data_features.append(tw_df, ignore_index=True)

        # Preprocessing
        data_features.fillna(0, inplace=True)

        # Normalization
        norm_data = StandardScaler().fit_transform(data_features)

        ## Find optimal number of clusters
        range_n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        silhouettes = []

        # Compute TSNE
        vizu_model = TSNE(learning_rate=1000)
        # Fitting Model
        transformed = vizu_model.fit_transform(norm_data)

        for n_clusters in range_n_clusters:
            if n_clusters == 1:
                silhouette_avg = 0
            else:
                clus = KMeans(n_clusters=n_clusters)
                cluster_labels = clus.fit_predict(norm_data)
                silhouette_avg = silhouette_score(norm_data, cluster_labels)
            silhouettes.append(silhouette_avg)
            # print("For n_clusters =", n_clusters,
            #       "The average silhouette_score is :", silhouette_avg)

        k = range_n_clusters[np.argmax(silhouettes)]

        print('Best number of clusters :', k)

        # Clustering
        linked = linkage(norm_data, method='ward', metric='euclidean')

        if plot:
            plt.figure(figsize=(10, 7))

            dendrogram(
                linked,
                orientation='right',
                labels=time_windows_labels,
                distance_sort='descending',
                show_leaf_counts=True)

            plt.title("{}\n{} - Dendogram\nBest Number of Clusters {}".format(self.label, behavior_type, k))

            plt.ylabel('Time Windows')

        clusters = fcluster(linked, k, criterion='maxclust')
        nb_clusters = len(set(list(clusters)))

        if plot:
            print('{} clusters detected using dendograms :)'.format(nb_clusters))

        colors = generate_random_color(nb_clusters)

        if plot:
            # Compute TSNE
            vizu_model = TSNE(learning_rate=100)
            # Fitting Model
            transformed = vizu_model.fit_transform(norm_data)

            plt.figure()
            x_axis = transformed[:, 0]
            y_axis = transformed[:, 1]

            for cluster_id in set(clusters):
                ids = [i for i, x in enumerate(clusters) if x == cluster_id]
                plt.scatter(x_axis[ids], y_axis[ids], c=colors[cluster_id - 1], label='Cluster ' + str(cluster_id))

            plt.title('{}\n{} - 2D-Projection'.format(self.label, behavior_type))
            plt.legend()

        clusters_dict = {}

        # Associate to each cluster his time_windows
        for cluster_id in set(clusters):
            ids = [i for i, x in enumerate(clusters) if x == cluster_id]
            cluster_id -= 1  # cause the clustering algo cluster starts at 1
            clusters_dict[cluster_id] = ids

        return clusters_dict, colors

    def build_similarity_matrix(self, behavior_type, method):
        """
        Clustering of the time windows using histogram intersection surface as similarity metric
        :param behavior_type:
        :param plot:
        :return:
        """

        nb_windows = len(self.time_windows_data)

        ######################
        ## Similarity Matrix #
        ######################

        similarity_matrix = np.zeros((nb_windows, nb_windows))
        for i in range(nb_windows):
            tw_data_A = self.time_windows_data[i]
            for j in range(i, nb_windows):
                tw_data_B = self.time_windows_data[j]

                if behavior_type == Behavior.OCC_TIME:
                    arrayA = tw_data_A.timestamp.values
                    arrayB = tw_data_B.timestamp.values

                    # In case of histogram_intersection
                    hist_max_bin = 24 * 3600  # 24 hours

                elif behavior_type == Behavior.DURATION:
                    arrayA = tw_data_A.duration.values
                    arrayB = tw_data_B.duration.values

                    # In case of histogram_intersection
                    hist_max_bin = 8 * 3600  # 8 hours

                else:
                    raise ValueError("Illegal value of behavior_type")

                if method == 'stat_test':
                    similarity = ks_similarity(arrayA, arrayB)
                elif method == 'density_intersect':
                    similarity = density_intersection_area(arrayA, arrayB)
                elif method == 'histogram_intersect':
                    similarity = histogram_intersection(arrayA, arrayB, max_bin=hist_max_bin)
                else:
                    raise ValueError("Illegal value of Similarity Method")

                # print('[{}, {}] : {}'.format(i, j, similarity))
                if np.isnan(similarity):
                    similarity = 0
                similarity_matrix[i][j] = similarity

        missing_part = np.transpose(similarity_matrix.copy())
        np.fill_diagonal(missing_part, 0)
        similarity_matrix = similarity_matrix + missing_part


        return similarity_matrix

    def similarity_clustering(self, behavior_type, method, plot=False, debug=False):
        """
        Clustering of the time windows into behavior clusters
        :param behavior_type:
        :param method:
        :param plot:
        :param debug:
        :return:
        """

        nb_windows = len(self.time_windows_data)

        # Build the similarity matrix
        similarity_matrix = self.build_similarity_matrix(behavior_type=behavior_type, method=method)

        print("Similarity Matrix Built")

        # Clustering of the time windows
        graph_labels = ['{}'.format(i) for i in range(nb_windows)]

        inflation_power = None  # Automatic search
        threshold_edges = None  # Automatic search
        if method == 'stat_test':
            threshold_edges = max(0.85, np.median(similarity_matrix.flatten()))
        elif method == 'histogram_intersect':
            threshold_edges = max(0.5, np.median(similarity_matrix.flatten()))
            inflation_power = 1.8
        elif method == 'density_intersect':
            # threshold_edges = max(0.5, np.median(similarity_matrix.flatten()))
            inflation_power = 1.4

        if threshold_edges is not None and debug:
            threshold_indices = threshold_edges > similarity_matrix

            weak_matrix = similarity_matrix.copy()
            weak_matrix[threshold_indices] = 0

            # Similarity Matrix Heatmap
            sns.heatmap(weak_matrix, vmin=0, vmax=1)
            plt.title('Time Windows Similarity Matrix')
            plt.xlabel('Time Windows ID')
            plt.ylabel('Time Windows ID')

            plt.show()


        clusters, clusters_color = mcl_clusterinig(matrix=similarity_matrix, labels=graph_labels,
                                                   threshold_filter=threshold_edges, inflation_power=inflation_power,
                                                   plot=plot, gif=debug)
        return clusters, clusters_color

    def display_drift(self, clusters, colors, behavior_type):
        """
        Display the Distribution drifts discovered on the Activity
        :param clusters: dict like : {'cluster_id': [window_id list]}
        :param colors: a list of the clusters color
        :return:
        """

        fig, (ax1, ax2) = plt.subplots(2)
        # f, ax = plt.subplots(figsize=(8, 8))

        for cluster_id, window_ids in clusters.items():
            durations = []
            occ_times = []

            for window_id in window_ids:
                occ_times += list(self.time_windows_data[window_id].timestamp.values)
                durations += list(self.time_windows_data[window_id].duration.values)

            durations = np.asarray(list(set(durations))) / 60  # Display in minutes # 'set' to remove duplicates
            occ_times = np.asarray(list(set(occ_times))) / 3600  # Display in hours # 'set' to remove duplicates

            # # Describe the time period of the cluster
            # time_periods = self.time_periods_from_windows(window_ids)
            #
            # msg = ''
            # nb_days = 0
            #
            # for time_period in time_periods:
            #     start_date = self.begin_date + dt.timedelta(days=time_period[0])
            #     end_date = self.begin_date + dt.timedelta(days=time_period[1] + 1)
            #     msg += "[{} - {}]\t".format(start_date.date(), end_date.date())
            #     nb_days += (end_date - start_date).days
            #
            # nb_occ_per_days = len(durations) / nb_days
            #
            # print("Cluster {} : {} days ({:.2f} occ/day) ** {}".format(cluster_id, nb_days, nb_occ_per_days, msg))

            array = []
            range = None
            if behavior_type == Behavior.OCC_TIME:
                array = occ_times
                range = (0, 24)
            elif behavior_type == Behavior.DURATION:
                array = durations

            ax1.hist(array, bins=100, range=range, alpha=0.3,
                     label='Behavior {}'.format(cluster_id),
                     color=colors[cluster_id])

            sns.kdeplot(array, label='Behavior {}'.format(cluster_id), gridsize=50,
                        shade_lowest=False, shade=True, color=colors[cluster_id], ax=ax2)

            # Draw the two density plots
            # ax = sns.kdeplot(occ_times, durations, color=colors[cluster_id],
            #                  label='Cluster {} : {:.2f}/day'.format(cluster_id, nb_occ_per_wind), shade=True,
            #                  shade_lowest=False)



        if behavior_type == Behavior.OCC_TIME:
            ax1.set_title("{}\nCluster : Occurrence Time distribution".format(self.label))
            ax1.set_xlabel('Hour of the day')
            ax1.set_xlim(0, 24)

            ax2.set_xlabel('Hour of the day')
            ax2.set_xlim(0, 24)

            plt.xticks(np.arange(0, 24, 2))

        elif behavior_type == Behavior.DURATION:
            ax1.set_title("{}\nCluster : Activity Duration distribution".format(self.label))
            ax1.set_xlabel('Duration (Minutes)')
            ax1.set_xlim(left=0)

            ax2.set_xlabel('Duration (Minutes)')
            ax2.set_xlim(left=0)


        ax1.set_ylabel('Number of occurrences')
        ax2.set_title("Density Distribution")
        ax2.set_ylabel('Density')


        plt.legend(loc='upper right')



    def display_behavior_evolution(self, clusters, colors):
        """
        Plot the evolution of the different behavior throughout the dataset
        :param clusters:
        :param colors:
        :return:
        """
        fig, ax = plt.subplots()
        # xfmt = dat.DateFormatter('%d-%m-%y')
        months = dat.MonthLocator()  # every month
        monthsFmt = dat.DateFormatter('%b %Y')  # Eg. Jan 2012

        # format the ticks
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        ax.xaxis.set_minor_locator(months)



        for cluster_id, window_ids in clusters.items():
            lvl = cluster_id * 2

            time_periods = self.time_periods_from_windows(window_ids)

            print("Cluster {} :".format(cluster_id))
            for period in time_periods:
                start_date = self.begin_date + dt.timedelta(days=period[0])
                end_date = self.begin_date + dt.timedelta(days=period[1] + 1)

                print("\t{} - {}".format(start_date, end_date))


                if time_periods.index(period) == 0:
                    plt.text(dat.date2num(start_date), lvl, 'Behavior {}'.format(cluster_id), fontsize=16)
                ax.hlines(lvl, dat.date2num(start_date), dat.date2num(end_date), label='Behavior {}'.format(cluster_id),
                                linewidth=75, color=colors[cluster_id])

        ax.tick_params(axis='both', which='major', labelsize=12)
        fig.autofmt_xdate()
        # plt.title("Activity : '{}'".format(self.label))
        plt.xlabel('Time')
        plt.ylabel('Behaviors')


    def clustering_quality(self, clusters, behavior_type, method='stat_test'):
        """
        Compute the quality metrics of the clustering
        :param clusters:
        :param behavior_type:
        :return: Davies-Bouldin Index, Dunn Index, Silhouette score
        """

        # Time Windows Similarity Matrix
        tw_similarity_matrix = 1 - self.build_similarity_matrix(behavior_type=behavior_type,
                                                                method=method)  # Turn similarity into distance

        nb_clusters = len(clusters)

        # Inter-Cluster distance matrix
        inter_cluster_distance = np.zeros((nb_clusters, nb_clusters))
        clusters_data = {}

        hist_max_bin = 24 * 3600  # For Occurrence time
        for cluster_id, window_ids in clusters.items():
            array = []

            if behavior_type == Behavior.OCC_TIME:
                for window_id in window_ids:
                    array += list(self.time_windows_data[window_id].timestamp.values)
            elif behavior_type == Behavior.DURATION:
                hist_max_bin = 8 * 3600  # For Durations
                for window_id in window_ids:
                    array += list(self.time_windows_data[window_id].duration.values)

            clusters_data[cluster_id] = np.asarray(list(set(array)))

        for cluster_i in range(nb_clusters):
            array_i = clusters_data[cluster_i]
            for cluster_j in range(cluster_i, nb_clusters):
                array_j = clusters_data[cluster_j]

                if method == 'stat_test':
                    similarity = ks_similarity(array_i, array_j)
                elif method == 'density_intersect':
                    similarity = density_intersection_area(array_i, array_j)
                elif method == 'histogram_intersect':
                    similarity = histogram_intersection(array_i, array_j, max_bin=hist_max_bin)

                inter_cluster_distance[cluster_i][cluster_j] = 1 - similarity


        # Little trick for speed purposes ;)
        # Cause the similarity matrix is triangular

        missing_part = np.transpose(inter_cluster_distance.copy())
        np.fill_diagonal(missing_part, 0)
        inter_cluster_distance = inter_cluster_distance + missing_part

        # Intra-Cluster distance

        intra_cluster_distance = np.zeros((nb_clusters))

        for cluster_id, window_ids in clusters.items():

            distance_to_centroid = []

            cluster_array = clusters_data[cluster_id]
            for window_id in window_ids:
                array = []
                if behavior_type == Behavior.OCC_TIME:
                    array = self.time_windows_data[window_id].timestamp.values
                elif behavior_type == Behavior.DURATION:
                    array = self.time_windows_data[window_id].duration.values

                if method == 'stat_test':
                    similarity = ks_similarity(array, cluster_array)
                elif method == 'density_intersect':
                    similarity = density_intersection_area(array, cluster_array)
                elif method == 'histogram_intersect':
                    similarity = histogram_intersection(array, cluster_array, max_bin=hist_max_bin)

                distance_to_centroid.append(1 - similarity)

            intra_cluster_distance[cluster_id] = np.mean(distance_to_centroid)

        ########################
        # Davies?Bouldin index #
        ########################

        # Based on https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index

        # Mi,j the separation between the ith and the jth cluster
        # Si, the within cluster scatter for cluster i
        # Ri,j = (Si + Sj)/Mi,j
        # Di = max Ri,j (i =/ j)
        # db_index = mean(all Di)

        R_ij = np.zeros((nb_clusters, nb_clusters))

        for cluster_i in range(nb_clusters):
            for cluster_j in range(nb_clusters):

                if cluster_i == cluster_j:
                    R_ij[cluster_i][cluster_j] = 0
                    continue

                Si = intra_cluster_distance[cluster_i]
                Sj = intra_cluster_distance[cluster_j]
                Mij = inter_cluster_distance[cluster_i][cluster_j]

                if Mij == 0:
                    R_ij[cluster_i][cluster_j] = 0
                    continue
                R_ij[cluster_i][cluster_j] = (Si + Sj) / Mij

        db_index = np.mean(R_ij.max(axis=1))

        ##############
        # Dunn index #
        ##############

        # Based on https://en.wikipedia.org/wiki/Dunn_index

        fake_inter_cluster_distance = inter_cluster_distance.copy()
        np.fill_diagonal(fake_inter_cluster_distance, 1)
        dunn_index = fake_inter_cluster_distance.min() / intra_cluster_distance.max()

        ##########################
        # Silhouette Score index #
        ##########################

        # Based on https://en.wikipedia.org/wiki/Silhouette_(clustering)

        # a(i) : average distance between i and all other points in its cluster
        # b(i) : the smallest average distance of i to all points in any other cluster, of which i is not a member

        nb_windows = len(self.time_windows_data)
        tw_silhouettes = np.zeros((nb_windows))

        for cluster_id, window_ids in clusters.items():

            for i in window_ids:
                same_cluster_ids = window_ids.copy()
                same_cluster_ids.remove(i)

                if len(same_cluster_ids) == 0:
                    a_i = 0
                else:
                    a_i = tw_similarity_matrix[
                        np.ix_(same_cluster_ids, same_cluster_ids)].mean()  # Extract submatrix and average

                b_i = np.inf
                for other_cluster_id, other_window_ids in clusters.items():
                    if other_cluster_id == cluster_id:
                        continue
                    current_b_i = tw_similarity_matrix[
                        np.ix_([i], other_window_ids)].mean()  # Extract submatrix and average
                    if current_b_i < b_i:
                        b_i = current_b_i

                tw_silhouettes[i] = (b_i - a_i) / max([a_i, b_i])

        silhouette_score = np.mean(tw_silhouettes)

        if np.isnan(silhouette_score):
            silhouette_score = 0
        return db_index, dunn_index, silhouette_score

    def clustering_interpretation(self, clusters, behavior_type):
        """
        Use DBSCAN to intepret the cluster
        :param clusters:
        :param behavior_type:
        :return:
        """

        clusters_interpretations = {}
        for cluster_id, window_ids in clusters.items():
            durations = []
            occ_times = []

            for window_id in window_ids:
                occ_times += list(self.time_windows_data[window_id].timestamp.values)
                durations += list(self.time_windows_data[window_id].duration.values)

            durations = np.asarray(list(set(durations))).reshape(-1, 1)
            occ_times = np.asarray(list(set(occ_times))).reshape(-1, 1)

            if behavior_type == Behavior.DURATION:
                data = durations
                interpretation = {}
                mu = np.mean(data)
                sigma = np.std(data)
                interpretation[str(dt.timedelta(seconds=mu))] = str(dt.timedelta(seconds=sigma))
                clusters_interpretations[cluster_id] = interpretation
                continue

            else:
                data = occ_times

            if len(data) <= 2:
                clusters_interpretations[cluster_id] = {}  # No interpretation
                continue


            data_clusters = univariate_clustering(data)

            interpretation = {}  # mean_time (in seconds) as key and std_duration (in seconds) as value

            for i, array in data_clusters.items():
                mu = math.ceil(np.mean(array))
                sigma = math.ceil(np.std(array))

                interpretation[str(dt.timedelta(seconds=mu))] = str(dt.timedelta(seconds=sigma))

            clusters_interpretations[cluster_id] = interpretation

        return clusters_interpretations

if __name__ == '__main__':
    main()
