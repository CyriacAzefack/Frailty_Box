from datetime import datetime

import imageio
import matplotlib.dates as dat
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy import random

# from Data_Drift import Drift_Detector
from Pattern_Mining.Pattern_Discovery import *

sns.set_style('darkgrid')



def main():
    dataset_name = 'aruba'
    replication_id = 1

    dataset = pick_dataset(dataset_name)
    #
    # activities_generation_method = 'Macro'
    # duration_generation_method = 'Normal'
    # time_step_min = 5
    #
    # path = "./output/{}/{} Activities Model - {} - Time Step {}mn//dataset_simulation_rep_{}.csv".format(dataset_name,
    #                                                                                                      activities_generation_method + "_recent_patterns",
    #                                                                                                      duration_generation_method,
    #                                                                                                      time_step_min,
    #                                                                                                      replication_id)
    # path = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/{}/Simulation/Simulation_X1_Pattern_ID_0/dataset_simulation_rep_{}.csv".format(
    #     dataset_name, replication_id)
    # dataset = pick_custom_dataset(path)
    # dataset = pick_dataset(dataset_name)

    start_date = dataset.date.min().to_pydatetime()
    end_date = start_date + dt.timedelta(days=20)

    visualize(dataset, start_date=start_date, end_date=end_date)
    # visualize(dataset, start_date=start_date, end_date=end_date, start_suffix='_begin', end_suffix='_end')


def visualize(data, start_date, end_date):
    '''
    Visualize the log dataset
    :param data:
    :param start_date:
    :param end_date:
    :return:
    '''

    data = data[(data.date >= start_date) & (data.date <= end_date)].copy()
    # Turn the dataset into an activity dataset

    data['duration'] = data['end_date'] - data['date']
    data['duration'] = data['duration'].apply(lambda x: x.total_seconds() / 60)

    print(data.describe())
    sns.distplot(data.duration)
    plt.show()

    activities = list(data.groupby(['label'], as_index=False).agg({'duration': 'sum'}).sort_values("duration",
                                                                                                   ascending=False).label.values)

    # activities = list(data.label.unique())

    df_data = pd.DataFrame(columns=['activity', 'start', 'end', 'level'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xfmt = dat.DateFormatter('%d-%m-%y %H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    ax = ax.xaxis_date()

    for activity in activities:
        lvl = activities.index(activity) * 5
        data_activity = data[data.label == activity].copy()
        data_activity['level'] = lvl

        date = data_activity.date.min().to_pydatetime()

        data_activity.date = pd.to_datetime(data_activity.date).astype(datetime)
        data_activity.end_date = pd.to_datetime(data_activity.end_date).astype(datetime)

        color = random.rand(3, )
        plt.text(dat.date2num(date), lvl, activity, fontsize=14)
        ax = plt.hlines(data_activity.level, dat.date2num(data_activity.date), dat.date2num(data_activity.end_date),
                        label=activity,
                        linewidth=75, color=color)
        # df_data = pd.concat([df_data, result], axis=0)
    # plt.legend()

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
    Plot the evolution of the occurrence time and duration distribution through the dataset
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

if __name__ == '__main__':
    main()
