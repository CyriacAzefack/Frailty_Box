import datetime as dt
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

import Utils
from Data_Drift.Drift_v2.LogClustering import LogClustering

sns.set_style('darkgrid')


def main():
    # for i in range(101, 121):
    name = "hh101"

    log_dataset = Utils.pick_dataset(name)

    model = LogFeatureClustering(name, log_dataset)
    model.extract_features()
    model.logHeatmap()

    n_clusters, n_pca_components = model.silhouette_plots(data=model.features_data.values, display=True)

    model.clustering(n_clusters=n_clusters, pca_n_components=n_pca_components)

    # model.logHeatmap()

    print(f'DATASET {name} DONE!!')


class LogFeatureClustering(LogClustering):

    def __init__(self, name, log_dataset, nb_days=-1):
        super(LogFeatureClustering, self).__init__(name, log_dataset, nb_days)

    def extract_log_features(self, log):
        """
        Extract a set of features from a log
        :param log:
        :return:
        """

        log['day_date'] = log['date'].dt.date.apply(
            lambda x: dt.datetime.combine(x, dt.datetime.min.time()))
        log['timestamp'] = (log['date'] - log['day_date']).apply(lambda x: x.total_seconds())  # In seconds

        features = {}

        # nb_days = np.ceil((data.end_date.max() - data.date.min()) / dt.timedelta(days=1))

        for label in self.labels:
            label_data = log[log.label == label]
            ## 1 - Features on activities occurrence time
            label_occ_time = label_data.timestamp.values
            features[label + "_mean_occ_time"] = np.mean(label_occ_time)
            features[label + "_std_occ_time"] = np.std(label_occ_time)

            ## 2 - Feature on the duration of activities
            label_durations = (label_data.end_date - label_data.date).apply(lambda x: x.total_seconds() / 60)
            features[label + "_mean_duration"] = np.mean(label_durations)
            features[label + "_std_duration"] = np.std(label_durations)

            ## 3 - Feature on the number of occurrences
            features[label + '_nb_occ'] = len(label_data)

        return features

    def extract_features(self):
        """
        Build the dataset for clustering
        :return:
        """

        dataset = []
        day_id = 0
        for daily_log in self.time_windows_logs:
            day_id += 1
            features = self.extract_log_features(daily_log)
            dataset.append(features)

            sys.stdout.write(f"\r{day_id}/{self.nb_days} Days Treated")
            sys.stdout.flush()
        sys.stdout.write("\n")

        self.features_data = pd.DataFrame(dataset)
        self.features_data.fillna(0, inplace=True)

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

        # for i in range(n_clusters):
        #     print(f'Center {i}\t{centers[i]}')

        scaler = MinMaxScaler()

        scaler.fit(self.features_data)

        scaled_centers = scaler.transform(centers)

        couples = []
        for i in range(n_clusters):
            [couples.append((i, j)) for j in self.features_data.columns]

        list_scaled_centers = list(scaled_centers.flatten())

        df = pd.DataFrame()
        df['value'] = list_scaled_centers
        df['couple'] = couples
        df['feature'] = df.couple.apply(lambda x: x[1])

        df.sort_values(['value'], ascending=False, inplace=True)

        features_order = df.feature.unique()

        centers_df = pd.DataFrame(scaled_centers, columns=self.features_data.columns)
        centers_df = centers_df[features_order]

        # self.features_data = self.features_data[features_order]

        sns.heatmap(centers_df, vmin=0, vmax=1, xticklabels=centers_df.columns)
        plt.title('Cluster Centers')
        plt.tight_layout()
        plt.show()

        self.features_data['class'] = clusters
        self.features_data.sort_values(['class'], axis=0, ascending=True, inplace=True, kind='quicksort',
                                       na_position='last')
        return clusters_indices

    def logHeatmap(self):
        """
        Plot an image of the all dataset
        :return:
        """

        scaler = MinMaxScaler()

        scaled_values = scaler.fit_transform(self.features_data)

        sns.heatmap(scaled_values, xticklabels=self.features_data.columns)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
