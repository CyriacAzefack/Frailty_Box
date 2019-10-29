from __future__ import absolute_import

import errno
import glob
import sys
from optparse import OptionParser

import cv2
import imageio
import seaborn as sns

from Utils import *

sns.set_style('darkgrid')


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
    window_size = 30
    time_step = dt.timedelta(minutes=30)
    plot = False
    debug = False

    drift(dataset_name=dataset_name, window_size=window_size, window_step=dt.timedelta(days=1),
          time_step=time_step, plot=plot, debug=debug)


def drift(dataset_name, window_size, window_step, time_step, plot, debug):
    print("Dataset Name : {}".format(dataset_name.upper()))
    print("Windows size : {}".format(window_size))
    print("Window step : {}".format(window_step))
    print("Display : {}".format(plot))
    print("Mode debug : {}".format(debug))

    data = pick_dataset(dataset_name)

    time_window_size = dt.timedelta(days=window_size)

    behavior = ImageBehaviorClustering(name=dataset_name, dataset=data, time_window_step=window_step,
                                       time_window_duration=time_window_size, time_step=time_step)

    heatmaps = behavior.extract_features(store=True)

    # if store:  # Create the GIF
    list_files = glob.glob(behavior.output_folder + '*.png')
    list_files.sort(key=os.path.getmtime, reverse=True)
    images = []

    for filename in list_files:
        images.append(imageio.imread(filename))
    imageio.mimsave(behavior.output_folder + '___all.gif', images, duration=0.1)
    print('GIF Created')

    # changes = behavior_clustering.drift_detector(method=drift_method, plot=plot, debug=debug)
    # changes_str = stringify_keys(changes)
    # changes_str.pop('clusters', None)
    # print(json.dumps(changes_str, indent=4))

    # df = pd.DataFrame(
    #     columns=['label', 'nb_behaviors', 'behavior_duration', 'nb_drift_points', 'interpretation', 'silhouette'])
    # for label, change in changes_str.items():
    #     df.loc[len(df)] = [label, change['nb_clusters'], change['duration'], change['nb_drift_points'],
    #                        change['interpretation'], '{:.2f}'.format(change['silhouette'])]
    #
    # # if not os.path.exists(outdir):
    # #     os.mkdir(outdir)
    #
    # fullname = os.path.join(OUTDIR, '{}_{}_{}_W{}.csv'.format(dataset_name.upper(), behavior_type.replace(' ', '-'),
    #                                                           drift_method, window_size))
    #
    # df.to_csv(fullname, index=False, sep=';')


class BehaviorClustering:

    def __init__(self, name, dataset, time_window_duration, time_window_step):
        """
        Create a Behavior corresponding to a house data
        :param dataset: event sequence
        :param time_window_duration: duration of the time_window for the drift detection
        :param instant_events : True if events in the log_dataset have no durations
        """
        self.name = name
        self.dataset = dataset
        self.time_window_duration = time_window_duration
        self.time_window_step = time_window_step

        self.start_date = self.dataset.date.min().to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        self.end_date = (self.dataset.date.max().to_pydatetime() + dt.timedelta(days=1)).replace(hour=0, minute=0,
                                                                                                 second=0,
                                                                                                 microsecond=0)

        self.data_preprocessing()

        # Rank the label by decreasing order of durations
        self.labels = dataset.groupby(['label'])['duration'].sum().sort_values().index

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

        self.dataset = None


    def data_preprocessing(self):
        """
        Pre-processing of the data
        Create the columns : day_date, timestamp(number of seconds since the start of the day), duration(in seconds)
        :return:
        """
        self.dataset['day_date'] = self.dataset['date'].dt.date.apply(
            lambda x: dt.datetime.combine(x, dt.datetime.min.time()))
        self.dataset['start_ts'] = (self.dataset['date'] - self.dataset['day_date']).apply(
            lambda x: x.total_seconds())  # In seconds
        self.dataset['end_ts'] = (self.dataset['end_date'] - self.dataset['day_date']).apply(
            lambda x: x.total_seconds())  # In seconds

        self.dataset['duration'] = (self.dataset['end_date'] - self.dataset['date']).apply(
            lambda x: x.total_seconds())  # Duration in seconds

    def time_windows_clustering(self, n_clusters=2):
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

            window_data = self.dataset[
                (self.dataset.date >= window_start_time) & (self.dataset.date < window_end_time)].copy()

            time_windows_logs.append(window_data)

            window_start_time += self.time_window_step  # We slide the time window by 1 day

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
        for i in range(len(sorted_clusters_id)):
            sorted_clusters[i] = clusters[sorted_clusters_id[i]]

        return sorted_clusters


class ImageBehaviorClustering(BehaviorClustering):

    def __init__(self, name, dataset, time_window_duration, time_window_step, time_step):
        super().__init__(name, dataset, time_window_duration, time_window_step)
        self.time_step = time_step
        self.nb_daily_steps = int(dt.timedelta(days=1) / time_step)

    def extract_features(self, store=False, display=False):
        """
        Build the Heatmap for all the data points (time windows logs)
        :return: list of the heatmap matrix
        """

        tw_id = 0
        tw_heatmaps = []

        for tw_log in self.time_windows_logs:
            tw_id += 1
            heatmap = self.build_heatmap(tw_log, display=display)
            if store:
                self.build_and_save_heatmap(heatmap, tw_id)

            tw_heatmaps.append(heatmap)

            sys.stdout.write(f"\r{tw_id}/{len(self.time_windows_logs)} Time Windows Heatmap Created")
            sys.stdout.flush()
        sys.stdout.write("\n")

        self.dataset = np.asarray(tw_heatmaps)

    def build_heatmap(self, log, display=False):
        """
        Build a daily heatmap from an event log
        :param log:
        :param display:
        :return:
        """
        log['start_step'] = log.start_ts.apply(lambda x: int(x / self.time_step.total_seconds()))
        log['end_step'] = log.end_ts.apply(lambda x: int(x / self.time_step.total_seconds()))

        nb_days_per_time_window = int(self.time_window_duration / dt.timedelta(days=1))

        heatmap = {}
        for label in self.labels:
            label_log = log[log.label == label]
            actives_ts = []
            for _, row in label_log.iterrows():
                actives_ts += list(range(row.start_step, row.end_step))

            steps_activity_ratio = []
            for step in range(self.nb_daily_steps):
                ratio = actives_ts.count(step) / nb_days_per_time_window
                steps_activity_ratio.append(ratio)

            heatmap[label] = steps_activity_ratio

        heatmap = pd.DataFrame.from_dict(heatmap, orient='index')

        if display:
            sns.heatmap(heatmap, vmin=0, vmax=1)
            plt.tight_layout()
            plt.show()

        return heatmap

    def build_and_save_heatmap(self, heatmap, id):
        """
        Build and save heatmap
        :param heatmap:
        :return:
        """

        image_matrix = heatmap.values * 255
        img_path = self.output_folder + f'{self.name}_tw_{id}.png'

        if not cv2.imwrite(img_path, image_matrix):
            raise

    def time_windows_clustering(self):
        """
        Clustering of the time windows
        :return: dict-like object with cluster id as key and tw_ids list as value
        """

if __name__ == '__main__':
    main()
