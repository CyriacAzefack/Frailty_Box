import getopt
import json
import sys

import matplotlib.dates as dat
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from Data_Drift.Features_Extraction import *
from Data_Drift.MCL import mcl_clusterinig
from Utils import *


def main(argv):
    # Default values
    dataset_name = 'toy_no_changes'
    window_size = 30
    behavior_type = Behavior.OCC_TIME
    drift_method = 'student_test'
    labels = None
    plot = False
    debug = False

    try:
        opts, args = getopt.getopt(argv, "hn:w:pdATDsimf",
                                   ["help", "dataset_name=", "window_size=", "plot", "debug", "all_methods"])
    except getopt.GetoptError:
        print('Command Error :')
        print('Behavior.py -n <dataset_name> -w <window_size> [-p --plot] [-d --debug]')
        print('-T\t For Behavior drift on activities occurrence time (default one used)')
        print('-D\t For Behavior drift on activities durations')
        print('-s\t Use the student-test as similarity metric for clustering')
        print('-i\t Use the density intersection as similarity metric for clustering')
        print('-m\t Use the Mean Squared Error of normed histograms as similarity metric for clustering ')
        print('-f\t Use features and a classic clustering algorithm ')

        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('How to use the command :')
            print('Behavior.py -n <dataset_name> -w <window_size> [-p --plot] [-d --debug]')
            print('-T\t For Behavior drift on activities occurrence time (default one used)')
            print('-D\t For Behavior drift on activities durations')
            print('-s\t Use the student-test as similarity metric for clustering')
            print('-i\t Use the density intersection as similarity metric for clustering')
            print('-m\t Use the Mean Squared Error of normed histograms as similarity metric for clustering ')
            print('-f\t Use features and a classic clustering algorithm ')
            print('-A\t Use all drift methods and compare them')

            sys.exit()
        elif opt in ("-n", "--dataset_name"):
            dataset_name = arg
        elif opt in ("-w", "--window_size"):
            window_size = int(arg)
        elif opt in ("-p", "--plot"):
            plot = True
        elif opt in ("-d", "--debug"):
            debug = True
        elif opt in ("-T"):
            behavior_type = Behavior.OCC_TIME
        elif opt in ("-D"):
            behavior_type = Behavior.DURATION
        elif opt in ("-s"):
            drift_method = 'student_test'
        elif opt in ("-i"):
            drift_method = 'density_intersect'
        elif opt in ("-m"):
            drift_method = 'histogram_mse'
        elif opt in ("-f"):
            drift_method = 'features'
        elif opt in ("-A", "--all_methods"):
            drift_method = "All Methods"

    print("Dataset Name : {}".format(dataset_name.upper()))
    print("Windows size : {}".format(window_size))
    print("Behavior Type : {}".format(behavior_type))
    print("Drift Method : {}".format(drift_method))
    print("Display : {}".format(plot))
    print("Mode debug : {}".format(debug))

    data = pick_dataset(dataset_name)

    time_window_size = dt.timedelta(days=window_size)
    # labels = ['get drink']
    labels = None

    inhabitant_behavior = Behavior(dataset=data, time_window_duration=time_window_size)
    inhabitant_behavior.create_activities_behavior()
    if drift_method != "All Methods":

        changes = inhabitant_behavior.drift_detector(behavior_type=behavior_type, method=drift_method, plot=plot,
                                                     labels=labels, debug=debug)
        changes_str = stringify_keys(changes)
        changes_str.pop('clusters', None)
        print(json.dumps(changes_str, indent=4))

    else:
        methods = ['features', 'student_test', 'histogram_mse', 'density_intersect']

        for drift_method in methods:
            print('### {} ###'.format(drift_method))
            changes = inhabitant_behavior.drift_detector(behavior_type=behavior_type, method=drift_method, plot=plot,
                                                         labels=labels, debug=debug)

            changes_str = stringify_keys(changes)
            changes_str.pop('clusters', None)
            print(json.dumps(changes_str, indent=4))


class Behavior:
    OCC_TIME = 'Activity Occurrence Time'
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
        self.dataset['timestamp'] = (self.dataset['date'] - self.dataset['day_date']).apply(lambda x: x.total_seconds())

        if not self.instant_events:
            self.dataset['duration'] = (self.dataset['end_date'] - self.dataset['date']).apply(
                lambda x: x.total_seconds() / 60)  # Duration in minutes

    def drift_detector(self, behavior_type, method, plot=False, labels=None, debug=False):
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
            clusters, colors = activity_behavior.drift_detector(behavior_type=behavior_type, method=method, plot=plot,
                                                                debug=debug)
            score = activity_behavior.clustering_quality(clusters, behavior_type=behavior_type)

            changes[label] = {
                # 'clusters': clusters,
                'colors': colors,
                'nb_clusters': len(clusters),
                'score': score
            }


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

        if method == 'features':
            clusters, clusters_color = self.features_clustering(behavior_type=behavior_type, plot=debug)
        else:
            clusters, clusters_color = self.distribution_similarity(behavior_type=behavior_type, method=method,
                                                                    plot=debug)



        ##################################################
        # Density Intersection Area for all the clusters #
        ##################################################

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

            # TODO : select the features according to the behavior_type observed
            tw_features = activities_features(window_data=tw_data, activity_labels=[self.label])

            tw_df = pd.DataFrame.from_dict(tw_features, orient='columns')
            if len(data_features) == 0:
                data_features = tw_df
            else:
                data_features = data_features.append(tw_df, ignore_index=True)

        data_features.fillna(0, inplace=True)
        norm_data = StandardScaler().fit_transform(data_features)

        # Clustering
        linked = linkage(norm_data, method='ward')

        # TODO : automatic threshold decision

        last = linked[-10:, 2]
        last_rev = last[::-1]
        idxs = np.arange(1, len(last) + 1)
        #

        acceleration = np.diff(last, 2)  # 2nd derivative of the distances
        acceleration_rev = acceleration[::-1]
        #
        # x
        k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters

        if plot:
            plt.plot(idxs, last_rev)
            plt.plot(idxs[:-2] + 1, acceleration_rev)
            plt.title("Elbow Method\nBest Number of clusters : {}".format(k))

        print('Best number of clusters :', k)


        if plot:
            plt.figure(figsize=(10, 7))

            dendrogram(
                linked,
                orientation='top',
                labels=time_windows_labels,
                distance_sort='descending',
                show_leaf_counts=True)

            plt.title("{}\n{} - Dendogram\nBest Number of Clusters {}".format(self.label, behavior_type, k))

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

    def distribution_similarity(self, behavior_type, method, plot, gif_debug=False):
        """
        Clustering of the time windows using histogram intersection surface as similarity metric
        :param behavior_type:
        :param plot:
        :return:
        """

        nb_windows = len(self.time_windows_data)

        edges_treshold = 0.8

        ######################
        ## Similarity Matrix #
        ######################

        # print('Building intersection area matrix...')

        similarity_matrix = np.zeros((nb_windows, nb_windows))
        for i in range(nb_windows):
            tw_data_A = self.time_windows_data[i]
            for j in range(i, nb_windows):
                tw_data_B = self.time_windows_data[j]

                if behavior_type == Behavior.OCC_TIME:
                    arrayA = tw_data_A.timestamp.values
                    arrayB = tw_data_B.timestamp.values
                elif behavior_type == Behavior.DURATION:
                    arrayA = tw_data_A.duration.values
                    arrayB = tw_data_B.duration.values
                else:
                    raise ValueError("Illegal value of behavior_type")

                if method == 'student_test':
                    similarity = array_similarity(arrayA, arrayB)

                elif method == 'density_intersect':
                    similarity = density_intersection_area(arrayA, arrayB)
                elif method == 'histogram_mse':
                    similarity = mse(arrayA, arrayB)
                else:
                    raise ValueError("Illegal value of Similarity Method")

                # print('[{}, {}] : {}'.format(i, j, similarity))
                similarity_matrix[i][j] = similarity

        missing_part = np.transpose(similarity_matrix.copy())
        np.fill_diagonal(missing_part, 0)
        similarity_matrix = similarity_matrix + missing_part

        if method == 'histogram_mse':
            similarity_matrix = 1 - (similarity_matrix - similarity_matrix.min()) / (
                    similarity_matrix.max() - similarity_matrix.min())

        # plt.figure()
        # sns.heatmap(mse_matrix, vmin=mse_matrix.min(), vmax=mse_matrix.max(), annot=False, fmt=".2f")
        # plt.title("Validation of Clusters Activity Duration Time")
        # plt.show()

        # print('Markov Clustering...')

        graph_labels = ['W_{}'.format(i) for i in range(nb_windows)]

        clusters, clusters_color = mcl_clusterinig(matrix=similarity_matrix, labels=graph_labels, inflation_power=2,
                                                   plot=plot, gif=gif_debug, edges_treshold=edges_treshold)

        return clusters, clusters_color

    def display_drift(self, clusters, colors, behavior_type):
        """
        Display the drift discovered for the duration of the activity
        :param clusters: dict like : {'cluster_id': [window_id list]}
        :param colors: a list of the clusters color
        :return:
        """

        fig, (ax1, ax2) = plt.subplots(2)

        for cluster_id, window_ids in clusters.items():
            durations = []
            occ_times = []

            for window_id in window_ids:
                occ_times += list(self.time_windows_data[window_id].timestamp.values)
                durations += list(self.time_windows_data[window_id].duration.values)

            durations = np.asarray(durations) / 3600  # Display in hours
            occ_times = np.asarray(occ_times) / 3600  # Display in hours

            if len(window_ids) == 0:
                nb_occ_per_wind = 0
            else:
                nb_occ_per_wind = len(durations) / (len(window_ids) * self.time_window_duration.days)

            # Describe the time period of the cluster
            time_periods = self.time_periods_from_windows(window_ids)

            msg = ''
            nb_days = 0
            for time_period in time_periods:
                start_date = self.begin_date + dt.timedelta(days=time_period[0])
                end_date = self.begin_date + dt.timedelta(days=time_period[1] + 1)
                msg += "[{} - {}]\t".format(start_date.date(), end_date.date())

                nb_days += (end_date - start_date).days
            print("Cluster {} : {} days  ** {}".format(cluster_id, nb_days, msg))

            array = []
            if behavior_type == Behavior.OCC_TIME:
                array = occ_times
            elif behavior_type == Behavior.DURATION:
                array = durations

            ax1.hist(array, bins=100, alpha=0.3,
                     label='Cluster {} : {:.2f}/day'.format(cluster_id, nb_occ_per_wind),
                     color=colors[cluster_id])

            sns.kdeplot(array, label='Cluster {} : {:.2f}/day'.format(cluster_id, nb_occ_per_wind),
                        shade_lowest=False, shade=True, color=colors[cluster_id], ax=ax2)

        if behavior_type == Behavior.OCC_TIME:
            ax1.set_title("{}\nCluster : Occurrence Time distribution".format(self.label))
            ax1.set_xlabel('Hour of the day')
            ax1.set_xlim(0, 24)

            ax2.set_xlabel('Hour of the day')
            ax2.set_xlim(0, 24)

        elif behavior_type == Behavior.DURATION:
            ax1.set_title("{}\nCluster : Activity Duration distribution".format(self.label))
            ax1.set_xlabel('Duration (hour)')

            ax2.set_xlabel('Duration (hour)')


        ax1.set_ylabel('Number of occurrences')
        ax2.set_title("Density Distribution")
        ax2.set_ylabel('Density')


        plt.legend(loc='upper right')

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

    def display_behavior_evolution(self, clusters, colors):
        """
        Plot the evolution of the different behavior throughout the dataset
        :param clusters:
        :param colors:
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xfmt = dat.DateFormatter('%d-%m-%y %H:%M')
        ax.xaxis.set_major_formatter(xfmt)
        ax = ax.xaxis_date()
        for cluster_id, window_ids in clusters.items():
            lvl = cluster_id * 5

            time_periods = self.time_periods_from_windows(window_ids)

            for period in time_periods:
                start_date = self.begin_date + dt.timedelta(days=period[0])
                end_date = self.begin_date + dt.timedelta(days=period[1] + 1)

                plt.text(dat.date2num(start_date), lvl, 'cluster {}'.format(cluster_id), fontsize=14)
                ax = plt.hlines(lvl, dat.date2num(start_date), dat.date2num(end_date),
                                label='cluster {}'.format(cluster_id),
                                linewidth=75, color=colors[cluster_id])

        plt.title("{}\nBehavior evolution".format(self.label))

    def clustering_quality(self, clusters, behavior_type):
        """
        Compute the mean of inter-clusters density intersection area
        :param clusters:
        :param behavior_type:
        :return: clustering score
        """

        nb_clusters = len(clusters)

        clustering_quality_matrix = np.zeros((nb_clusters, nb_clusters))
        clusters_data = {}
        for cluster_id, window_ids in clusters.items():

            array = []

            if behavior_type == Behavior.OCC_TIME:
                for window_id in window_ids:
                    array += list(self.time_windows_data[window_id].timestamp.values)
            elif behavior_type == Behavior.DURATION:
                for window_id in window_ids:
                    array += list(self.time_windows_data[window_id].duration.values)

            clusters_data[cluster_id] = array

        for cluster_i in range(nb_clusters):
            array_i = clusters_data[cluster_i]
            for cluster_j in range(cluster_i, nb_clusters):
                array_j = clusters_data[cluster_j]

                clustering_quality = 1 - density_intersection_area(array_i, array_j)

                clustering_quality_matrix[cluster_i][cluster_j] = clustering_quality

        # Little trick for speed purposes ;)
        # Cause the similarity matrix is triangular

        missing_part = np.transpose(clustering_quality_matrix.copy())
        np.fill_diagonal(missing_part, 0)
        clustering_quality_matrix = clustering_quality_matrix + missing_part

        overall_clustering_quality = clustering_quality_matrix.mean()

        return overall_clustering_quality


if __name__ == '__main__':
    main(sys.argv[1:])
