from __future__ import absolute_import

import errno
import pickle
import random
from optparse import OptionParser

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.dates as dat
import seaborn as sns
import tensorflow as tf
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import trange

from AutoEncoder.AutoEncoder import AutoEncoderModel
from Utils import *

font = {'family': 'normal',
        'weight': 'bold',
        'size': 14}

matplotlib.rc('font', **font)


# sns.set_style('darkgrid')


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
    parser.add_option('-m', '--drift_method', help='Drift method used', dest='drift_method', action='store',
                      type='choice', choices=['features', 'stat_test', 'histogram_intersect', 'density_intersect',
                                              'all_methods'], default='stat_test')

    (options, args) = parser.parse_args()

    # Mandatory Options
    # if options.dataset_name is None:
    #     print("The name of the Input event log is missing\n")
    #     parser.print_help()
    #     exit(-1)
    #
    # if options.window_size is None:
    #     print("The size of the time windows is missing\n")
    #     parser.print_help()
    #     exit(-1)

    # dataset_name = options.dataset_name
    # window_size = options.window_size
    #
    # drift_method = options.drift_method
    # plot = options.plot
    # debug = options.debug

    dataset_name = "aruba"
    window_size = 7
    time_step = dt.timedelta(minutes=5)
    window_step = dt.timedelta(days=1)
    latent_dim = 3
    plot = True
    debug = False

    drift(dataset_name=dataset_name, window_size=window_size, window_step=window_step, latent_dim=latent_dim,
          time_step=time_step, plot=plot, debug=debug)


def drift(dataset_name, window_size, window_step, time_step, latent_dim, plot, debug):
    print("Dataset Name : {}".format(dataset_name.upper()))
    print("Windows size : {}".format(window_size))
    print("Time step : {}".format(time_step))
    print("Display : {}".format(plot))
    print("Mode debug : {}".format(debug))

    data = pick_dataset(dataset_name)

    time_window_size = dt.timedelta(days=window_size)

    behavior = ImageBehaviorClustering(name=dataset_name, dataset=data, time_window_step=window_step,
                                       time_window_duration=time_window_size, time_step=time_step)

    behavior.extract_features(store=True, display=debug)

    n_clusters = 2
    clusters_indices = behavior.time_windows_clustering(display=plot, debug=debug, latent_dim=latent_dim,
                                                        n_clusters=n_clusters)

    behavior.cluster_interpretability(clusters_indices=clusters_indices)

    for cluster_id, indices in clusters_indices.items():
        behavior.plot_day_bars(sublogs_indices=indices, title=f"Days for Cluster {cluster_id}")

        print(f"Cluster {cluster_id} Days Plot finished!")

    cluster_colors = generate_random_color(len(clusters_indices))
    behavior.display_behavior_evolution(clusters=clusters_indices, colors=cluster_colors)
    plt.show()


def clustering_algorithm(data, n_clusters):
    """
    Cluster the data
    :param data:
    :param n_clusters:
    :return:
    """

    # transformed_data = PCA(n_components=2, random_state=0).fit_transform(data)

    labels = KMeans(n_clusters=n_clusters, n_init=100, random_state=0).fit_predict(data)
    # labels = clustering.predict(transformed_data)

    return labels


def silhouette_plots(data, display=True):
    """
    Plot the differents silhoutte values
    :return:
    """

    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

    filename = './output/encoded_data.csv'
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()

    tsne_data = PCA(n_components=2).fit_transform(data)

    optimal_n_clusters = 1
    avg_silhouette = 0

    for n_clusters in range_n_clusters:

        cluster_labels = clustering_algorithm(data, n_clusters=n_clusters)

        silhouette_avg = silhouette_score(data, cluster_labels)
        sample_silhouette_values = silhouette_samples(data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        if silhouette_avg > avg_silhouette:
            avg_silhouette = silhouette_avg
            optimal_n_clusters = n_clusters

        if display:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            # fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters

            # Compute the silhouette scores for each sample

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(tsne_data[:, 0], tsne_data[:, 1], marker='.', s=100, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # # Labeling the clusters
            # centers = clusterer.cluster_centers_
            #
            # # Draw white circles at cluster centers
            # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
            #             c="white", alpha=1, s=200, edgecolor='k')
            #
            # for i, c in enumerate(centers):
            #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
            #                 s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=8, fontweight='bold')
    if display:
        print()
        print(f"Choose Number of Clusters : {optimal_n_clusters}")
        plt.show()

    # print(f"Choosen Number of PCA Components : {min_n_components}")

    return optimal_n_clusters


class BehaviorClustering:

    def __init__(self, name, dataset, time_window_duration, time_window_step):
        """
        Clustering of the behavior
        :param name: Name of the dataset
        :param dataset: event log
        :param time_window_duration: duration of a time window
        :param time_window_step: Duration of the sliding step
        """
        self.name = name
        self.log_dataset = dataset
        self.time_window_duration = time_window_duration
        self.time_window_step = time_window_step

        self.data_preprocessing()

        self.start_date = self.log_dataset.day_date.min().to_pydatetime()
        self.end_date = self.log_dataset.day_date.max().to_pydatetime() + dt.timedelta(days=1)

        # Rank the label by decreasing order of durations
        self.labels = self.log_dataset.groupby(['label'])['duration'].sum().sort_values().index

        # We take the 10 most present activities
        # self.labels = self.labels[-10:]

        self.label_color = {}
        colors = generate_random_color(len(self.labels))
        for i in range(len(self.labels)):
            self.label_color[self.labels[i]] = colors[i]

        self.time_windows_logs = self.create_time_windows()
        print("Time Windows Logs Extracted !!")

        self.output_folder = f'./output/{name}/tw_images/'
        print(os.path.dirname(self.output_folder))

        if not os.path.exists(os.path.dirname(self.output_folder)):
            try:
                os.makedirs(os.path.dirname(self.output_folder))

            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def data_preprocessing(self):
        """
        Pre-processing of the data
        Create the columns : day_date, timestamp(number of seconds since the start of the day), duration(in seconds)
        :return:
        """

        indexes_to_drop = []

        def nightly(row):
            next_day_date = row.day_date + dt.timedelta(days=1)

            # self.log_dataset.drop([row.name], inplace=True) # Remove old activity
            indexes_to_drop.append(row.name)

            # Add the first part
            self.log_dataset = self.log_dataset.append(
                {
                    'date': row.date,
                    'end_date': next_day_date,
                    'label': row.label,
                    'day_date': row.day_date,
                    'start_ts': row.start_ts,
                    'end_ts': dt.timedelta(days=1).total_seconds()
                }, ignore_index=True)

            # Add the second part
            self.log_dataset = self.log_dataset.append(
                {
                    'date': next_day_date,
                    'end_date': row.end_date,
                    'label': row.label,
                    'day_date': next_day_date,
                    'start_ts': 0,
                    'end_ts': (row.end_date - next_day_date).total_seconds()
                }, ignore_index=True)

        self.log_dataset['day_date'] = self.log_dataset['date'].dt.date.apply(
            lambda x: dt.datetime.combine(x, dt.datetime.min.time()))
        self.log_dataset['start_ts'] = (self.log_dataset['date'] - self.log_dataset['day_date']).apply(
            lambda x: x.total_seconds())  # In seconds
        self.log_dataset['end_ts'] = (self.log_dataset['end_date'] - self.log_dataset['day_date']).apply(
            lambda x: x.total_seconds())  # In seconds

        nightly_log = self.log_dataset[self.log_dataset.end_ts > dt.timedelta(hours=24).total_seconds()].copy(
            deep=False)

        nightly_log.apply(nightly, axis=1)

        self.log_dataset.drop(indexes_to_drop, inplace=True)

        self.log_dataset['duration'] = (self.log_dataset['end_date'] - self.log_dataset['date']).apply(
            lambda x: x.total_seconds())  # Duration in seconds

        self.log_dataset.sort_values(['date'], ascending=True, inplace=True)
        self.log_dataset.reset_index(inplace=True, drop=True)

    def time_windows_clustering(self, n_clusters):
        """
        Clustering of the Resident Behavior
        :param n_clusters:
        :return:
        """
        raise Exception("Not Implemented")

    def create_time_windows(self):
        """
        Slide a time window among the data and create the different time windows logs
        :return: the list of time windows data
        """

        # Starting point of the last time window

        window_start_time = self.start_date

        time_windows_logs = []

        while window_start_time <= self.end_date - self.time_window_duration:
            window_end_time = window_start_time + self.time_window_duration

            window_data = self.log_dataset[
                (self.log_dataset.date >= window_start_time) & (self.log_dataset.date < window_end_time)].copy()

            time_windows_logs.append(window_data)

            window_start_time += self.time_window_step  # We slide the time window by the time window step

        return time_windows_logs

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
        for i in range(len(clusters)):
            sorted_clusters[i] = clusters[sorted_clusters_id[i]]

        return sorted_clusters, sorted_clusters_id

    def plot_day_bars(self, sublogs_indices, title="Days"):
        """
        Plot the days bars
        :return:
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        def timeTicks(x, pos):
            d = dt.timedelta(seconds=x)
            return int(d.seconds / 3600)

        formatter = matplotlib.ticker.FuncFormatter(timeTicks)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(plt.MultipleLocator(3600))  # Ticks every hour
        ax.set_xlim(0, 24 * 3600)

        yticks = []
        yticks_labels = []

        for label in self.labels:

            label_segments = []

            dates_list = []

            kwargs = {'color': self.label_color[label], 'linewidth': 300 / len(sublogs_indices)}

            for day_id in sublogs_indices:
                plot_index = sublogs_indices.index(day_id)

                day_start_date = self.start_date + day_id * self.time_window_step
                day_end_date = day_start_date + dt.timedelta(days=1)
                day_dataset = self.log_dataset[(self.log_dataset.date >= day_start_date)
                                               & (self.log_dataset.date < day_end_date)
                                               & (self.log_dataset.label == label)]

                dates_list.append(day_start_date)

                segments = list(day_dataset[['start_ts', 'end_ts']].values)

                if len(segments) > 0:
                    for x in segments:
                        label_segments.append([x[0], plot_index, x[1], plot_index])

            label_segments = np.asarray(label_segments)
            if len(label_segments) > 0:
                xs = label_segments[:, ::2]
                ys = label_segments[:, 1::2]
                lines = LineCollection([list(zip(x, y)) for x, y in zip(xs, ys)], label=label, **kwargs)
                ax.add_collection(lines)

        for i in range(len(sublogs_indices)):
            yticks.append(i)
            yticks_labels.append(sublogs_indices[i])

        ax.legend()
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks_labels)
        plt.xlabel("Hour of the day")
        plt.title(title)
        plt.legend(loc='upper left', fancybox=True, shadow=True, ncol=1, bbox_to_anchor=(1, 0.5))


class ImageBehaviorClustering(BehaviorClustering):
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 1280

    def __init__(self, name, dataset, time_window_duration, time_window_step, time_step):
        super().__init__(name, dataset, time_window_duration, time_window_step)
        self.time_step = time_step
        self.nb_daily_steps = int(dt.timedelta(hours=24) / time_step)

        self.time_labels = [str(i * self.time_step)[:-3] for i in range(self.nb_daily_steps)]

    def extract_features(self, store=False, display=False):
        """
        Build the Heatmap for all the data points (time windows logs)
        :return: list of the heatmap matrix
        """

        tw_heatmaps = []

        nb_days_per_time_window = int(self.time_window_duration / dt.timedelta(days=1))

        for tw_id in trange(len(self.time_windows_logs), desc='Extract features from Time Windows'):
            tw_log = self.time_windows_logs[tw_id]
            # for tw_log in self.time_windows_logs:
            #     tw_id += 1
            heatmap = self.build_heatmap(log=tw_log, nb_days=nb_days_per_time_window, display=display)
            if store:
                self.save_heatmap(heatmap, tw_id + 1)

            # self.plot_day_bars(days_range)
            # plt.show()

            tw_heatmaps.append(heatmap.values)
        #
        #     sys.stdout.write(f"\r{tw_id}/{len(self.time_windows_logs)} Time Windows Heatmap Created")
        #     sys.stdout.flush()
        # sys.stdout.write("\n")

        self.sublogs_heatmaps = np.asarray(tw_heatmaps)

    def build_heatmap(self, log, nb_days, display=False):
        """
        Build a daily heatmap from an event log
        :param log:
        :param display:
        :return:
        """
        log['start_step'] = log.start_ts.apply(lambda x: int(x / self.time_step.total_seconds()))
        log['end_step'] = log.end_ts.apply(lambda x: int(x / self.time_step.total_seconds()))

        heatmap = {}
        for label in self.labels:
            label_log = log[log.label == label]
            actives_time_steps = []
            for _, row in label_log.iterrows():
                actives_time_steps += list(range(row.start_step, row.end_step + 1))

            steps_activity_ratio = []
            for step in range(self.nb_daily_steps):
                ratio = min(actives_time_steps.count(step) / nb_days, 1)
                steps_activity_ratio.append(ratio)

            heatmap[label] = steps_activity_ratio

        heatmap = pd.DataFrame.from_dict(heatmap, orient='index')

        if display:
            heatmap.columns = self.time_labels
            sns.heatmap(heatmap, vmin=0, vmax=1)
            plt.tight_layout()
            plt.xticks(rotation=30)
            plt.show()

        return heatmap

    def save_heatmap(self, heatmap, id):
        """
        Build and save heatmap
        :param heatmap:
        :param id:
        :return:
        """

        image_matrix = cv2.resize(heatmap.values * 255,
                                  (ImageBehaviorClustering.DISPLAY_HEIGHT, ImageBehaviorClustering.DISPLAY_WIDTH),
                                  interpolation=cv2.INTER_AREA)

        img_path = self.output_folder + f'{self.name}_tw_{id}.png'

        if not cv2.imwrite(img_path, image_matrix):
            raise

    def time_windows_clustering(self, latent_dim, n_clusters=None, display=True, debug=False, ):
        """
        Clustering of the time windows
        :return: dict-like object with cluster id as key and tw_ids list as value
        """

        # Build the AutoEncoder Model
        model = self.build_AE_model(latent_dim=latent_dim, display=display)

        tensor_dataset = tf.data.Dataset.from_tensor_slices(self.sublogs_heatmaps).batch(len(self.sublogs_heatmaps))
        encoded_points = []
        for d in tensor_dataset:
            z = model.encode(d)
            encoded_points += [list(x) for x in z.numpy()]

        encoded_points = np.asarray(encoded_points)

        if not n_clusters:
            n_clusters = silhouette_plots(encoded_points, display=True)

        # # if not n_clusters:
        #
        # norm_data = StandardScaler().fit_transform(encoded_points)
        # linked = linkage(norm_data, method='ward', metric='euclidean')
        # clusters = fcluster(linked, n_clusters, criterion='maxclust')  # first cluster #id is 1
        # clusters = [i - 1 for i in clusters]
        #
        # if display:
        #     dendrogram(
        #         linked,
        #         orientation='left',
        #         # labels=time_windows_labels,
        #         distance_sort='descending',
        #         show_leaf_counts=True)
        #
        #     plt.title("Dendogram ")
        #
        #     plt.ylabel('Time Windows')
        #
        #     plt.show()

        clusters = clustering_algorithm(encoded_points, n_clusters=n_clusters)
        # clusters = clustering_algorithm(encoded_points, n_clusters)

        clusters_indices = {}
        for n in range(n_clusters):
            indices = [i for i, e in enumerate(clusters) if e == n]
            clusters_indices[n] = indices

        clusters_indices, sorted_clusters_id = self.sort_clusters(clusters_indices)

        silhouette_avg = silhouette_score(encoded_points, clusters)

        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        return clusters_indices

    def cluster_interpretability(self, clusters_indices, display=True):
        """
        Details the differences between the clustes.
        Hightlight the changes in the resident behavior
        :param clusters_indices:
        :return:
        """

        n_clusters = len(clusters_indices)
        clusters_centers = self.compute_clusters_centers(clusters_indices)

        fig, ax = plt.subplots(n_clusters, n_clusters, sharex=False, sharey=False)
        fig.suptitle("Clusters Differences", fontsize=14)

        for cluster_i in range(n_clusters):
            cluster_i_center = clusters_centers[cluster_i]
            for cluster_j in range(n_clusters):
                cluster_j_center = clusters_centers[cluster_j]

                change_img = cluster_j_center - cluster_i_center

                df_change_img = pd.DataFrame(change_img, columns=self.time_labels, index=self.labels)

                sns.heatmap(df_change_img, center=0, cmap='RdYlGn', vmin=-1, vmax=1, cbar=True,
                            ax=ax[cluster_i][cluster_j])

                ax[cluster_i][cluster_j].set_title(f'Cluster {cluster_i} --> Cluster {cluster_j}')
                ax[cluster_i][cluster_j].set_yticklabels(self.labels)
                ax[cluster_i][cluster_j].set_yticks(np.arange(len(self.labels)))

        plt.yticks(rotation=45)
        plt.xticks(rotation=30)

        if display:
            fig, ax = plt.subplots(n_clusters, 1)
            fig.suptitle("Clusters Centers", fontsize=14)

            img_centers = []
            for i in range(n_clusters):
                img = clusters_centers[i]

                # img_centers.append(img)
                img_centers.append(cv2.resize(img, (1280, 1280), interpolation=cv2.INTER_AREA))

            img_centers = np.asarray(img_centers)
            yticks = []
            yticks_labels = []
            for i in range(len(self.labels)):
                s = i * img_centers.shape[1] / len(self.labels)
                e = (i + 1) * img_centers.shape[1] / len(self.labels)
                yticks.append((s + e) / 2)
                yticks_labels.append(self.labels[i])

            i = 0
            for axi, img in zip(ax.flat, img_centers):
                axi.set(yticks=yticks)
                axi.set_yticklabels(yticks_labels)
                axi.imshow(img, vmin=0, vmax=1)
                axi.set_title(f'Cluster {i}')
                i += 1

    def build_AE_model(self, train_ratio=0.8, latent_dim=10, display=False):
        """
        Build the AutoEncoderModel model
        :param train_ratio:
        :param latent_dim:
        :return:
        """

        height = self.sublogs_heatmaps.shape[1]
        width = self.sublogs_heatmaps.shape[2]

        TRAIN_BUF = int(self.sublogs_heatmaps.shape[0] * train_ratio)
        data_train = self.sublogs_heatmaps[:TRAIN_BUF]
        data_test = self.sublogs_heatmaps[TRAIN_BUF:]

        epochs = 1000
        batch_size = 10

        model = AutoEncoderModel(input_width=width, input_height=height, latent_dim=latent_dim)

        # Model parameters
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # loss = tf.keras.losses.MeanSquaredError()
        loss = tf.keras.losses.MeanSquaredError()
        # loss = tf.keras.losses.SquaredHinge()
        # metric = tf.keras.metrics.BinaryAccuracy()
        metric = tf.keras.metrics.MeanSquaredError()
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

        checkpoint_path = f"./output/{self.name}/AutoEncoder_logs/checkpoint.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                 save_weights_only=True,
                                                                 verbose=1)

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        model.compile(optimizer, loss=loss)

        if latest_checkpoint:
            model.load_weights(latest_checkpoint)

        model.fit(data_train, data_train, epochs=epochs, batch_size=batch_size, validation_data=(data_test, data_test),
                  shuffle=True, callbacks=[es_callback, save_model_callback], verbose=display)

        if display:
            model.plot_history()

            # TEST THE ENCODE-DECODE OPERATION
            nb_test = 5

            x = np.asarray(random.choices(data_train, k=nb_test)).reshape((nb_test, height, width))

            z = model.predict(x).reshape((nb_test, height, width))
            print(z.shape)

            fig, ax = plt.subplots(nb_test, 2)

            ximg = []
            zimg = []
            for i in range(nb_test):
                img = z[i]
                # img[img < 0.5] = 0
                # img[img > 0.5] = 1
                ximg.append(cv2.resize(x[i], (1280, 1280), interpolation=cv2.INTER_AREA))
                zimg.append(cv2.resize(img, (1280, 1280), interpolation=cv2.INTER_AREA))

            # images = images.reshape((2, 1280, 1280))
            for axi, xi, zi in zip(ax, ximg, zimg):
                axi[0].set(xticks=[], yticks=[])
                axi[1].set(xticks=[], yticks=[])
                axi[0].imshow(xi, vmin=0, vmax=1)
                axi[1].imshow(zi, vmin=0, vmax=1)

            plt.show()

        # accuracy = []
        #
        # for j in range(10):
        #     prediction = model.predict(data_train)
        #
        #     flat_test = []
        #     flat_prediction = []
        #
        #     for i in range(len(data_train)):
        #         flat_test += list(data_train[i].flatten())
        #         p = prediction[i].flatten()
        #         p[p>0.1*j] = 1
        #         p[p<0.1*j] = 0
        #         flat_prediction += list(p)
        #
        #     accuracy.append(accuracy_score(flat_test, flat_prediction))
        #
        # plt.plot(accuracy)
        # plt.title("Accuracy evolution with threshold")
        # plt.show()
        #
        # best_threshold = np.argmax(accuracy)
        # print(f"Reconstruction Binary Accuracy = {acc}")

        return model

    def display_behavior_evolution(self, clusters, colors):
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

            time_periods = self.time_periods_from_windows(window_ids)

            print("Cluster {} :".format(cluster_id))
            for period in time_periods:
                start_date = self.start_date + period[0] * self.time_window_step
                end_date = self.start_date + period[1] * self.time_window_step

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

    def compute_clusters_centers(self, clusters_indices):
        """
        Compute the center of each cluster
        :param clusters_indices:
        :return:
        """

        clusters_centers = []

        for cluster_id, indices in clusters_indices.items():

            cluster_heatmaps = []
            for day_id in indices:
                start_date = self.start_date + day_id * self.time_window_step
                end_date = start_date + dt.timedelta(days=1)
                day_dataset = self.log_dataset[
                    (self.log_dataset.date >= start_date) & (self.log_dataset.date < end_date)].copy()

                heatmap = self.build_heatmap(day_dataset, nb_days=1)
                cluster_heatmaps.append(heatmap.values)

            cluster_center = np.mean(np.asarray(cluster_heatmaps), axis=0)
            clusters_centers.append(cluster_center)

        clusters_centers = np.asarray(clusters_centers)

        return clusters_centers


if __name__ == '__main__':
    main()
