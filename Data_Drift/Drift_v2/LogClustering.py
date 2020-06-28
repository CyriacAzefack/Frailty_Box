import datetime as dt
import errno
import os
import sys

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

import Utils


def main():
    # for i in range(101, 121):
    name = "aruba"
    frequency = dt.timedelta(minutes=10)
    log_dataset = Utils.pick_dataset(name)

    model = LogImageClustering(name, log_dataset, frequency=frequency)
    model.extract_features()
    print(model.features_data.values.shape)
    n_clusters, n_pca_components = model.silhouette_plots(data=model.features_data.values, display=True)

    clustering_indices = model.clustering(n_clusters=n_clusters, pca_n_components=n_pca_components)

    model.logHeatmap()


class LogClustering:
    """
    Log Image Clustering
    """

    def __init__(self, name, log_dataset, nb_days=-1):
        """
        Initialize the encoding of the dataset
        :param name: Name of the studied dataset
        :param log_dataset:
        :param nb_days:
        """
        self.name = name

        # self.output_path = f'../../output/{name}/Daily_Images/'
        self.output_path = f'../../output/{name}/Daily_Images/'
        # Create the folder if it does not exist yet
        if not os.path.exists(os.path.dirname(self.output_path)):
            try:
                os.makedirs(os.path.dirname(self.output_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        start_date = log_dataset.date.min().to_pydatetime()
        # Get the date of the beginning of the day
        start_date = start_date.date()
        start_date = dt.datetime.combine(start_date, dt.datetime.min.time())

        end_date = start_date + dt.timedelta(days=nb_days)
        if nb_days <= 0:
            end_date = log_dataset.end_date.max().to_pydatetime()
            # Get the date of the beginning of the day
            end_date = end_date.date()
            end_date = dt.datetime.combine(end_date, dt.datetime.min.time())
            end_date += dt.timedelta(days=1)

        self.start_date = start_date
        self.end_date = end_date
        self.nb_days = int((end_date - start_date) / dt.timedelta(days=1))

        # Extract the part of the log we are interested in
        self.log_dataset = log_dataset[(log_dataset.date >= start_date) & (log_dataset.date < end_date)].copy()
        self.labels = log_dataset.label.unique()
        self.labels.sort()

        colors = Utils.generate_random_color(len(self.labels))
        self.label_color = {}
        for i in range(len(self.labels)):
            self.label_color[self.labels[i]] = colors[i]

        self.time_windows_logs = self.extract_time_windows_logs()

    def extract_time_windows_logs(self):
        """
        Extract each days from the log_dataset,
        :return:
        """

        # We do a local copy for different operations
        log_dataset = self.log_dataset.copy()

        def nightly(row):
            # Add the first part to the sublog_data
            sublog_data.loc[len(sublog_data)] = [row.date, day_end_date, row.label]

            # Add the second part to the rest of the log_dataset
            log_dataset.loc[len(log_dataset)] = [day_end_date, row.end_date, row.label]

        daily_sublogs = []
        for day_index in range(self.nb_days):
            day_start_date = self.start_date + dt.timedelta(days=day_index)
            day_end_date = day_start_date + dt.timedelta(days=1)
            sublog_data = log_dataset[(log_dataset.date >= day_start_date)
                                      & (log_dataset.date <= day_end_date)].copy()

            # Retrieve nightly events and split them in two
            nightly_events = sublog_data[sublog_data.end_date > day_end_date].copy()

            nightly_events.apply(nightly, axis=1)

            daily_sublogs.append(sublog_data)

        return daily_sublogs

    def clustering(self, n_clusters=2):
        raise Exception("Not Implemented")

    def plot_day_bars(self, daily_sublogs):
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

        day_id = 0
        for day_dataset in daily_sublogs:

            # Retrieve the date of the day
            day_start_date = day_dataset.date.min().to_pydatetime()
            day_start_date = day_start_date.date()

            if day_id % 5 == 0:
                yticks.append(day_id)
                yticks_labels.append(day_start_date)

            day_start_date = dt.datetime.combine(day_start_date, dt.datetime.min.time())

            day_dataset['start_second'] = day_dataset.date.apply(lambda x: (x - day_start_date).total_seconds())
            day_dataset['end_second'] = day_dataset.end_date.apply(lambda x: (x - day_start_date).total_seconds())

            for label in self.labels:
                segments = list(day_dataset.loc[day_dataset.label == label, ['start_second', 'end_second']].values)
                segments = [tuple(x) for x in segments]

                label_dataset = day_dataset[day_dataset.label == label]
                plt.hlines(day_id, label_dataset.start_second, label_dataset.end_second, linewidth=300 / self.nb_days,
                           color=self.label_color[label])
                # for seg in segments:
                #     plt.plot(seg, [day_id, day_id], color=self.label_color[label])

            day_id += 1

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks_labels)
        plt.xlabel("Hour of the day")
        plt.title("Days from Cluster")

    def silhouette_plots(self, data, variance_treshold=0.25, display=True):
        """
        Plot the differents silhoutte values
        :return:
        """

        range_n_clusters = [2, 3, 4, 5, 6]

        n_components = 10
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data)

        min_n_components = list(np.cumsum(pca.explained_variance_ratio_) > variance_treshold).index(True)

        if display:
            plt.plot(range(n_components), pca.explained_variance_ratio_)
            plt.axhline(y=variance_treshold, color="red", linestyle="--")
            plt.axvline(x=min_n_components - 1, color='blue', linestyle="--")
            plt.plot(range(n_components), np.cumsum(pca.explained_variance_ratio_))
            plt.title("Component-wise and Cumulative Explained Variance")
            plt.show()

        X = PCA(n_components=min_n_components).fit_transform(data)

        optimal_n_clusters = 1
        avg_silhouette = 0

        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

            silhouette_avg = silhouette_score(X, cluster_labels)
            sample_silhouette_values = silhouette_samples(X, cluster_labels)
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
                ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

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
                ax2.scatter(X[:, 0], X[:, 1], marker='.', s=100, lw=0, alpha=0.7,
                            c=colors, edgecolor='k')

                # Labeling the clusters
                centers = clusterer.cluster_centers_
                # Draw white circles at cluster centers
                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')

                for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                s=50, edgecolor='k')

                ax2.set_title("The visualization of the clustered data.")
                ax2.set_xlabel("Feature space for the 1st feature")
                ax2.set_ylabel("Feature space for the 2nd feature")

                plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                              "with n_clusters = %d" % n_clusters),
                             fontsize=14, fontweight='bold')
        if display:
            plt.show()

        print()
        print(f"Choose Number of Clusters : {optimal_n_clusters}")
        print(f"Choosen Number of PCA Components : {min_n_components}")

        return optimal_n_clusters, min_n_components


class LogImageClustering(LogClustering):
    HEIGHT = 28
    WIDTH = 28

    def __init__(self, name, log_dataset, nb_days=-1, frequency=dt.timedelta(seconds=5)):
        super(LogImageClustering, self).__init__(name, log_dataset, nb_days)
        self.frequency = frequency
        self.nb_tstep = int(dt.timedelta(days=1) / frequency)

    def extract_features(self):
        """
        Save images for each sublog extracted
        :return:
        """
        day_id = 0
        binary_matrixes = []
        for daily_log in self.time_windows_logs:
            day_id += 1
            binary_matrix = self.extract_log_features(daily_log)
            binary_matrixes.append(binary_matrix)

            sys.stdout.write(f"\r{day_id}/{self.nb_days} Days Image Created")
            sys.stdout.flush()
        sys.stdout.write("\n")

        self.binary_matrixes = binary_matrixes

        flattened_data = []
        for matrix in binary_matrixes:
            flattened_data.append(matrix.flatten())

        self.features_data = pd.DataFrame(flattened_data)

        # self.features_data = self.features_data.loc[:, (self.features_data != 0).any(axis=0)]

    def binaryMatrix2Image(self, binary_matrix):
        """
        Convert the binary distance_matrix to an image distance_matrix
        :param binary_matrix:
        :return:
        """
        image_matrix = cv2.resize(binary_matrix, (LogImageClustering.WIDTH, LogImageClustering.HEIGHT))
        # image_matrix[(image_matrix == 1)] = 255

        return image_matrix

    def extract_log_features(self, log):
        """
        Turn a Log
        :param sublog_data:
        :return:
        """

        day_start_date = log.date.min().to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)

        log['start_minutes'] = log.date.apply(lambda x: int((x - day_start_date) / self.frequency))
        log['end_minutes'] = log.end_date.apply(lambda x: int((x - day_start_date) / self.frequency))

        day_period = dt.timedelta(hours=24)

        nb_steps = int(day_period / self.frequency)

        image_width = nb_steps

        binary_matrix = []

        for label_id in range(len(self.labels)):
            label = self.labels[label_id]
            # Build the Binary Vector
            label_log = log[log.label == label]

            binary_vector = np.full(image_width, False)

            for _, row in label_log.iterrows():
                binary_vector[row.start_minutes:row.end_minutes + 1] = True

            binary_matrix.append(binary_vector)

        binary_matrix = np.asarray(binary_matrix, dtype=np.uint8)

        return binary_matrix

    def clustering(self, n_clusters=2, pca_n_components=2):
        """
        Do the clustering
        :param n_clusters:
        :return:
        """
        pca = PCA(n_components=pca_n_components)
        pca_result = pca.fit_transform(self.features_data.values)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(pca_result)
        silhouette_avg = silhouette_score(pca_result, clusters)

        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        clusters_indices = []
        for n in range(n_clusters):
            indices = [i for i, e in enumerate(clusters) if e == n]
            clusters_indices.append(indices)

        centers = pca.inverse_transform(kmeans.cluster_centers_)

        centers = np.asarray(centers)

        center_images = []

        for c in centers:
            c[c >= 0.5] = 255
            c[c < 0.5] = 0
            image = self.binaryMatrix2Image(c.reshape(self.nb_tstep, len(self.labels)))
            center_images.append(image)

        fig, ax = plt.subplots(1, n_clusters)

        center_images = np.asarray(center_images)

        for axi, img in zip(ax.flat, center_images):
            axi.set(xticks=[], yticks=[])
            axi.imshow(img, cmap='gray')

        plt.show()

        #
        # fig, ax1 = plt.subplots()
        # color = 'tab:blue'
        # list_centers = list(centers.flatten())
        # list_centers.sort()
        #
        # ax1.set_ylabel('Original curve', color=color)
        # ax1.tick_params(axis='y', labelcolor=color)
        # ax1.plot(list_centers, color=color)
        #
        # color = 'tab:red'
        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Order 1 Diff', color=color)
        # ax2.tick_params(axis='y', labelcolor=color)
        # ax2.plot(np.diff(list_centers, 1), color=color)
        # # plt.legend()
        # plt.title('Courbe bizarre')
        # plt.show()
        #
        #
        # couples = []
        # for i in range(n_clusters):
        #     [couples.append((i, j)) for j in self.features_data.columns]
        #
        # list_centers = list(centers.flatten())
        #
        # df = pd.DataFrame()
        # df['value'] = list_centers
        # df['couple'] = couples
        # df['feature'] = df.couple.apply(lambda x: x[1])
        #
        # df.sort_values(['value'], ascending=False, inplace=True)
        #
        # features_order = df.feature.unique()
        #
        # centers_df = pd.DataFrame(centers, columns=self.features_data.columns)
        # centers_df = centers_df[features_order]
        #
        # # self.features_data = self.features_data[features_order]
        #
        # sns.heatmap(centers_df, vmin=0, vmax=1, xticklabels=centers_df.columns)
        # plt.title('Cluster Centers')
        # plt.tight_layout()
        # plt.show()

        self.features_data['class'] = clusters
        self.features_data.sort_values(['class'], axis=0, ascending=True, inplace=True, kind='quicksort',
                                       na_position='last')
        return clusters_indices

    def logHeatmap(self):
        """
        Plot an image of the all dataset
        :return:
        """

        sns.heatmap(self.features_data, xticklabels=self.features_data.columns)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
