import imageio
import matplotlib
import matplotlib.dates as dat
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy import random

import Utils
# from Data_Drift import Drift_Detector
from Pattern_Mining.Pattern_Discovery import *

sns.set_style('darkgrid')


# plt.xkcd()



def main():
    dataset_name = 'aruba'

    dataset = pick_dataset(dataset_name, nb_days=-1)

    output_directory = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/{}/Simulation/DYNAMIC_step_15mn/dataset_simulation_rep_8.csv".format(
        dataset_name)
    simu_dataset = pick_custom_dataset(path=output_directory)

    start_date = simu_dataset.date.min().to_pydatetime()

    # visualize(dataset)

    # simu_graph = ActivityOccurrencesGraph(dataset_name, simu_dataset, nb_days=-1)

    real_graph = ActivityOccurrencesGraph(dataset_name, simu_dataset, nb_days=-1)


def visualize(data, start_date=None, end_date=None):
    '''
    Visualize the log log_dataset
    :param data:
    :param start_date:
    :param end_date:
    :return:
    '''

    if start_date is None:
        start_date = data.date.min().to_pydatetime()

    if end_date is None:
        end_date = data.date.max().to_pydatetime()

    data = data[(data.date >= start_date) & (data.date <= end_date)].copy()
    # Turn the log_dataset into an activity log_dataset

    data['duration'] = data['end_date'] - data['date']
    data['duration'] = data['duration'].apply(lambda x: x.total_seconds() / 60)

    # print(data.describe())
    # sns.distplot(data.duration)
    # plt.show()

    activities = list(data.groupby(['label'], as_index=False).agg({'duration': 'sum'}).sort_values("duration",
                                                                                                   ascending=False).label.values)

    # activities = list(data.label.unique())

    df_data = pd.DataFrame(columns=['activity', 'start', 'end', 'level'])

    fig = plt.figure()
    # fig.set_size_inches(1800 / 1200, 1, forward=False)
    ax = fig.add_subplot(111)
    xfmt = dat.DateFormatter('%d-%m-%y %H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    ax = ax.xaxis_date()

    for activity in activities:
        lvl = activities.index(activity) * 5
        data_activity = data[data.label == activity].copy()
        data_activity['level'] = lvl

        date = data_activity.date.min().to_pydatetime()

        data_activity.date = data_activity.date.dt.to_pydatetime()
        data_activity.end_date = data_activity.end_date.dt.to_pydatetime()

        color = random.rand(3, )
        plt.text(dat.date2num(date), lvl, activity, fontsize=14)
        ax = plt.hlines(data_activity.level, dat.date2num(data_activity.date), dat.date2num(data_activity.end_date),
                        label=activity,
                        linewidth=75, color=color)
        # df_data = pd.concat([df_data, result], axis=0)
    # plt.legend()

    plt.savefig('out.png', transparent=True)
    plt.show()

    print("done")


def plot_activity_occurrence_time(data, label, start_date=None, end_date=None, duration=True):
    data = data[data.label == label].copy()

    if not start_date:
        start_date = data.date.min().to_pydatetime()

    if not end_date:
        end_date = data.date.max().to_pydatetime()

    data = data[(data.date >= start_date) & (data.date < end_date)].copy()

    # Compute the timestamp of all the events
    # Timestamp : Nb of seconds since the beginning of the day
    data['day_date'] = data['date'].dt.date.apply(lambda x: dt.datetime.combine(x, dt.datetime.min.time()))
    data['timestamp'] = (data['date'] - data['day_date']).apply(lambda x: x.total_seconds()) / 3600  # Hour in the day

    # Duration in minutes
    data['duration'] = (data.end_date - data.date).apply(lambda x: x.total_seconds() / 60)

    fig = plt.figure()
    plt.plot(data.date, data.timestamp, 'bo')
    # plt.legend()
    plt.title("Occurrences of '{}'".format(label))
    plt.xlabel("Date")
    plt.ylabel("Hour of the day")
    plt.xticks(rotation=45)

    if duration:
        fig = plt.figure()
        plt.plot(data.timestamp, data.duration, 'bo')
        plt.title("Occurrences and Duration of '{}'".format(label))
        plt.ylabel("Duration (hour)")
        plt.xlabel("Hour of the day")

    plt.show()

    # plt.savefig('output/videos/foo.png')
    # plt.close(fig)


def plot_activiy_duration(data, label, start_date=None, end_date=None):
    data = data[data.label == label].copy()

    if not start_date:
        start_date = data.date.min().to_pydatetime()

    if not end_date:
        end_date = data.date.max().to_pydatetime()

    data = data[(data.date >= start_date) & (data.date < end_date)].copy()

    # Duration in minutes
    data['duration'] = (data.end_date - data.date).apply(lambda x: x.total_seconds() / 60)

    plt.title('Duration of the activity {}'.format(label))
    plt.plot(data.date, data.duration, 'bo')
    plt.legend()
    plt.show()


def distribution_evolution(data, time_window_duration, label, output_folder="./output/videos"):
    """
    Plot the evolution of the occurrence time and duration distribution through the log_dataset
    :param data:
    :param time_window_duration:
    :param label:
    :return:
    """
    data['day_date'] = data['date'].dt.date.apply(lambda x: dt.datetime.combine(x, dt.datetime.min.time()))
    data['timestamp'] = (data['date'] - data['day_date']).apply(lambda x: x.total_seconds())

    data['duration'] = (data['end_date'] - data['date']).apply(lambda x: x.total_seconds() / 3600)

    time_windows_data = Drift_Detector.create_time_windows(data, time_window_duration)

    ## Visualisation of the evolution of some distribution

    empty_folder(output_folder)
    images_occ_times = []
    images_durations = []

    duration_max = data[data.label == label].duration.max()

    for tw_index in range(len(time_windows_data)):
        tw_data = time_windows_data[tw_index]
        tw_data = tw_data[tw_data.label == label]
        occ_times = tw_data.timestamp.values / 3600

        fig = plt.figure()
        canvas = FigureCanvas(fig)
        sns.kdeplot(occ_times, shade_lowest=False, shade=True, color='green')
        plt.title('"{}" Distribution\nWindow {}'.format(label, tw_index))
        plt.xlim(0, 24)
        plt.ylim(0, 1)
        plt.xlabel('Hour of the day')

        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image_occ_times = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        images_occ_times.append(image_occ_times)

        durations = tw_data.duration.values
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        sns.kdeplot(durations, shade_lowest=False, shade=True, color='green')
        plt.title('"{}" Distribution\nWindow {}'.format(label, tw_index))

        plt.xlim(0, duration_max)
        # plt.ylim(0, 1/duration_max)
        plt.xlabel('Duration (hours)')

        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image_durations = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        images_durations.append(image_durations)

    imageio.mimsave(output_folder + '/dist_occ_times.gif', images_occ_times, duration=0.2)
    imageio.mimsave(output_folder + '/dist_durations.gif', images_durations, duration=0.2)


"""
Plot a grapj
"""


class ActivityOccurrencesGraph:

    def __init__(self, plot_label, dataset, start_date=None, nb_days=10):
        if start_date is None:
            start_date = dataset.date.min().to_pydatetime()

            # Get the date of the beginning of the day
            start_date = start_date.date()
            start_date = dt.datetime.combine(start_date, dt.datetime.min.time())

        end_date = start_date + dt.timedelta(days=nb_days)
        if nb_days <= 0:
            end_date = dataset.end_date.max().to_pydatetime()
            # Get the date of the beginning of the day
            end_date = end_date.date()
            end_date = dt.datetime.combine(end_date, dt.datetime.min.time())
            end_date += dt.timedelta(days=1)

        self.start_date = start_date
        self.nb_days = int((end_date - start_date) / dt.timedelta(days=1))
        self.dataset = dataset[(dataset.date >= start_date) & (dataset.date < end_date)].copy()
        self.plot_label = plot_label

        # Choose a color for each activity
        self.labels = dataset.label.unique()
        self.labels.sort()
        colors = Utils.generate_random_color(len(self.labels))
        self.label_color = {}
        for i in range(len(self.labels)):
            self.label_color[self.labels[i]] = colors[i]

        self.days_dataset = self.extract_days()
        print('#######################')
        print('# Days Extracted ...  #')
        print('#######################')

        self.plot_day_bars()
        self.duration_pie_chart()

        plt.show()

    def extract_days(self):
        """
        Extract each days from the log_dataset,
        :return:
        """

        def nightly(row):
            self.dataset.loc[len(self.dataset)] = [end_date, row.end_date, row.label]
            day_dataset.loc[len(day_dataset)] = [row.date, end_date, row.label]

        days_data = []
        # Split nightly activities in 2
        for day_index in range(self.nb_days):
            start_date = self.start_date + dt.timedelta(days=day_index)
            end_date = start_date + dt.timedelta(days=1)
            day_dataset = self.dataset[(self.dataset.date >= start_date) & (self.dataset.date <= end_date)].copy()

            # Split nightly events in 2
            dropping_indexes = []
            nightly_dataset = day_dataset[day_dataset.end_date > end_date].copy()

            nightly_dataset.apply(nightly, axis=1)
            # day_dataset.drop(dropping_indexes, axis=0, inplace=True)

            days_data.append(day_dataset)

        return days_data

    def plot_day_bars(self):
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
        for day_dataset in self.days_dataset:

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

    def duration_pie_chart(self):

        #####################################
        #   Activities Frequency Pie Chart  #
        #####################################
        # Duration Validation
        # confidence_error = 0.9
        labels_count_df = pd.DataFrame(columns=['duration'])

        colors = []
        for label in self.labels:
            labels_count_df.loc[label] = len(self.dataset[self.dataset.label == label])
            colors.append(self.label_color[label])

        # labels_count_df.sort_values(by=['duration'], ascending=False, inplace=True)

        # explode = (0, 0, 0, 0, 0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

        # cmap = plt.get_cmap('tab20c')
        # colors = cmap(np.linspace(0., 1., len(labels_count_df)))
        labels_count_df.duration.plot(kind='pie', fontsize=18, colors=colors, startangle=90)

        # wedges, texts = ax.pie(labels_count_df.duration, wedgeprops=dict(input_width=0.5), startangle=0, colors=colors)
        # bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        # kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
        #           bbox=bbox_props, zorder=0, va="center")
        # for i, p in enumerate(wedges):
        #     ang = (p.theta2 - p.theta1) / 2. + p.theta1
        #     y = np.sin(np.deg2rad(ang))
        #     x = np.cos(np.deg2rad(ang))
        #     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        #     connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        #     kw["arrowprops"].update({"connectionstyle": connectionstyle})
        #     ax.annotate(labels_count_df.period_ts_index[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
        #                 horizontalalignment=horizontalalignment, **kw)

        plt.legend(labels=labels_count_df.index, loc="best")
        plt.axis('equal')
        plt.tight_layout()
        plt.ylabel('')



if __name__ == '__main__':
    main()
