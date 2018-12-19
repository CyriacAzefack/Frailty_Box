import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from Data_Drift.Features_Extraction import *
from Data_Drift.MCL import mcl_clusterinig
from Utils import *


def main():
    data_name = 'KA'

    data = pick_dataset(data_name)

    window_size = dt.timedelta(days=5)
    # labels = ['get drink']
    labels = None

    inhabitant_behavior = Behavior(dataset=data, time_window_duration=window_size)
    inhabitant_behavior.create_activities_behavior()
    changes = inhabitant_behavior.drift_detector(behavior_type=Behavior.OCC_TIME, method='hist_intersect', plot=False,
                                                 labels=labels)
    print(changes)


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

    def drift_detector(self, behavior_type, method, plot=False, labels=None):
        """
        Drift detection of a specific behavior
        :param behavior_type: the behavior type for the drift detection. availables ones : Behavior.OCC_TIME,
        Behavior.DURATION
        :param method: The method use for clustering ('similarity', 'features', 'hist_intersect')
        :param plot: if True, plot the changes detected
        :return: changes
        """
        changes = {}  # {('label', (clusterA, clusterB)) : *changes*}
        label_clusters = {}

        if labels is None:
            labels = self.labels

        for label in labels:
            activity_behavior = self.activities_behavior[label]
            clusters, colors, label_changes = activity_behavior.drift_detector(behavior_type=behavior_type,
                                                                               method=method, plot=plot)

            for cluster_id_tuple, area in label_changes.items():
                changes[(label, cluster_id_tuple)] = area

            label_clusters[label] = clusters

            print("{} clusters found for the Activity '{}'".format(len(clusters), label))

            if plot:
                plt.show()

        sorted_changes = [(k, changes[k]) for k in sorted(changes, key=changes.get, reverse=False)]

        print(sorted_changes)

        for change in sorted_changes:
            label = change[0][0]
            idA = change[0][1][0]
            idB = change[0][1][1]

            activity_behavior = self.activities_behavior[label]
            clusters = {}
            clusters[0] = label_clusters[label][idA]
            clusters[1] = label_clusters[label][idB]

            colors = generate_random_color(2)
            activity_behavior.display_occurrence_time_drift(clusters, colors)
            plt.title('Important changes detected')
            plt.show()

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

    def drift_detector(self, behavior_type=Behavior.OCC_TIME, method='similarity', plot=True):
        """
        Drift detection of this activity
        :param behavior_type: the behavior type for the drift detection
        :param
        :return: clusters, clusters_color, changes (dict like {(clusterA, clusterB) : density intersection area}
        """
        changes = {}

        if method == 'similarity':
            clusters, clusters_color = self.similarity_clustering(behavior_type=behavior_type, plot=plot)
        elif method == 'features':
            clusters, clusters_color = self.features_clustering(behavior_type=behavior_type, plot=plot)
        elif method == 'hist_intersect':
            clusters, clusters_color = self.histogram_intersection_clustering(behavior_type=behavior_type, plot=plot)

        # Check the difference between cluster to rank changes

        # changes_matrix = np.empty((len(clusters), len(clusters)))

        clusters_data_points = []
        for i in range(len(clusters)):
            data_points = []
            for window_id in clusters[i]:

                if behavior_type == Behavior.OCC_TIME:
                    data_points += list(self.time_windows_data[window_id].timestamp.values)

                elif behavior_type == Behavior.DURATION:
                    data_points += list(self.time_windows_data[window_id].duration.values)

                else:
                    raise ValueError("Illegal value of behavior_type")
            clusters_data_points.append(data_points)

        for i in range(len(clusters)):
            for j in range(i, len(clusters)):

                if (len(clusters_data_points[i]) <= 4) or (len(clusters_data_points[j]) <= 4):
                    area = 1
                else:
                    area = density_intersection_area(clusters_data_points[i], clusters_data_points[j])
                # changes_matrix[i][j] = area
                changes[(i, j)] = area

        # sort by intersection area
        # sorted_changes = [(k, changes[k]) for k in sorted(changes, key=changes.get, reverse=False)]

        # plt.figure()
        # sns.heatmap(changes_matrix, vmin=0, vmax=1, annot=True, fmt=".2f")
        # plt.title("Validation of clusters")

        return clusters, clusters_color, changes

    def features_clustering(self, behavior_type, plot=True):
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
        max_d = 7
        if plot:
            plt.figure(figsize=(10, 7))

            dendrogram(
                linked,
                orientation='top',
                labels=time_windows_labels,
                distance_sort='descending',
                show_leaf_counts=True)
            plt.axhline(y=max_d, c='k')
            plt.title("{}\n{} - Dendogram".format(self.label, behavior_type))

        clusters = fcluster(linked, max_d, criterion='distance')
        nb_clusters = len(set(list(clusters)))

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

    def similarity_clustering(self, behavior_type, plot=True, gif=False):
        """
        Clustering of the time windows using a student-test as similarity metric
        :param time_windows_data:
        :return:
        """
        nb_windows = len(self.time_windows_data)

        ######################
        ## Similarity Matrix #
        ######################

        print('Starting building similarity matrix...')

        similarity_matrix = np.zeros((nb_windows, nb_windows))

        for i in range(nb_windows):
            tw_data_A = self.time_windows_data[i]
            for j in range(i, nb_windows):
                tw_data_B = self.time_windows_data[j]

                # 1- Occurrence time similarity
                if behavior_type == Behavior.OCC_TIME:
                    arrayA = tw_data_A.timestamp.values
                    arrayB = tw_data_B.timestamp.values
                elif behavior_type == Behavior.DURATION:
                    arrayA = tw_data_A.duration.values
                    arrayB = tw_data_B.duration.values
                else:
                    raise ValueError("Illegal value of behavior type")

                similarity = array_similarity(arrayA, arrayB)

                similarity_matrix[i][j] = similarity

        # Little trick for speed purposes ;)
        # Cause the similarity matrix is triangular
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

    def histogram_intersection_clustering(self, behavior_type, plot=True):
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

        # print('Building intersection area matrix...')

        mse_matrix = np.zeros((nb_windows, nb_windows))
        for i in range(nb_windows):
            tw_data_A = self.time_windows_data[i]
            for j in range(i, nb_windows):
                tw_data_B = self.time_windows_data[j]

                # TODO : Add weights for the different type of similarity

                # 1- Occurrence time similarity
                if behavior_type == Behavior.OCC_TIME:
                    arrayA = tw_data_A.timestamp.values
                    arrayB = tw_data_B.timestamp.values
                elif behavior_type == Behavior.DURATION:
                    arrayA = tw_data_A.duration.values
                    arrayB = tw_data_B.duration.values
                else:
                    raise ValueError("Illegal value of behavior_type")

                similarity = mse(arrayA, arrayB)

                # print('[{}, {}] : {}'.format(i, j, similarity))
                mse_matrix[i][j] = similarity

        # Little trick for speed purposes ;)
        # Cause the similarity matrix is triangular

        mse_matrix = (mse_matrix - mse_matrix.min()) / (mse_matrix.max() - mse_matrix.min())
        missing_part = np.transpose(mse_matrix.copy())
        np.fill_diagonal(missing_part, 0)
        mse_matrix = mse_matrix + missing_part

        mse_matrix = 1 - (mse_matrix - mse_matrix.min()) / (mse_matrix.max() - mse_matrix.min())

        # plt.figure()
        # sns.heatmap(mse_matrix, vmin=mse_matrix.min(), vmax=mse_matrix.max(), annot=False, fmt=".2f")
        # plt.title("Validation of Clusters Activity Duration Time")
        # plt.show()

        # print('Markov Clustering...')

        graph_labels = ['W_{}'.format(i) for i in range(nb_windows)]

        clusters, clusters_color = mcl_clusterinig(matrix=mse_matrix, labels=graph_labels, inflation_power=2,
                                                   plot=plot, gif=True, edges_treshold=0.85)

        return clusters, clusters_color

    def display_occurrence_time_drift(self, clusters, colors):
        """
        Display the drift discovered for the occurrence time of the activity
        :param clusters: dict like : {'cluster_id': [window_id list]}
        :param colors: a list of the clusters color
        :return:
        """
        nb_clusters = len(clusters)

        # Cluster Occurrence Time
        fig, (ax1, ax2) = plt.subplots(2)

        for cluster_id, window_ids in clusters.items():
            occ_times = []

            # if len(window_ids) < 4:
            #     continue

            for window_id in window_ids:
                occ_times += list(self.time_windows_data[window_id].timestamp.values)

            occ_times = [x / 3600 for x in occ_times]
            occ_times = np.asarray(occ_times)

            if len(window_ids) == 0:
                nb_occ_per_wind = 0
            else:
                nb_occ_per_wind = len(occ_times) / (len(window_ids) * self.time_window_duration.days)

            ax1.hist(occ_times, bins=100, alpha=0.3, label='Cluster {} : {.2f}/day'.format(cluster_id, nb_occ_per_wind),
                     color=colors[cluster_id])

            sns.kdeplot(occ_times, label='Cluster {} : {.2f}/day'.format(cluster_id, nb_occ_per_wind),
                        shade_lowest=False, shade=True,
                        color=colors[cluster_id], ax=ax2)

        ax1.set_title("{}\nCluster : Occurrence Time distribution".format(self.label))
        ax1.set_xlabel('Hour of the day')
        ax1.set_ylabel('Number of occurrences')
        ax1.set_xlim(0, 24)

        ax2.set_title("Density Distribution")
        ax2.set_xlabel('Hour of the day')
        ax2.set_ylabel('Density')
        ax2.set_xlim(0, 24)

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


if __name__ == '__main__':
    main()
