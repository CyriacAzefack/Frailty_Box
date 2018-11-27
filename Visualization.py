from datetime import datetime

import matplotlib.dates as dat
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random

from xED.Pattern_Discovery import *

sns.set_style('darkgrid')



def main():
    dataset_name = 'hh101'

    dataset = pick_dataset(dataset_name)

    label = "cook_dinner"

    plot_activity_occurrence_time(dataset, label=label)
    # plot_activiy_duration(dataset, label=label)

    # sim_id = 0
    # replication_id = 1

    # activities_generation_method = 'Macro'
    # duration_generation_method = 'Normal'
    # time_step_min = 5
    #
    # path = "./output/{}/{} Activities Model - {} - Time Step {}mn//dataset_simulation_rep_{}.csv".format(dataset_name,
    #                                                                                                      activities_generation_method + "_recent_patterns",
    #                                                                                                      duration_generation_method,
    #                                                                                                      time_step_min,
    #                                                                                                      replication_id)
    # # path = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/{}/Simple Model & TS Model Simulation results 5mn/dataset_simulation_rep_1.csv".format(
    # #     dataset_name)
    # dataset = pick_custom_dataset(path)
    #
    # start_date = dataset.date.min().to_pydatetime()
    # end_date = start_date + dt.timedelta(days=20)
    #
    # visualize(dataset, start_date=start_date, end_date=end_date)
    # # visualize(dataset, start_date=start_date, end_date=end_date, start_suffix='_begin', end_suffix='_end')


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
    data['duration'] = data['duration'].apply(lambda x: x.total_seconds())

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


def plot_activity_occurrence_time(data, label, start_date=None, end_date=None):
    data = data[data.label == label].copy()

    if not start_date:
        start_date = data.date.min().to_pydatetime()

    if not end_date:
        end_date = data.date.max().to_pydatetime()

    data = data[(data.date >= start_date) & (data.date < end_date)].copy()

    # Compute the timestamp of all the events
    # Timestamp : Nb of seconds since the beginning of the day
    data['day_date'] = data['date'].dt.date.apply(lambda x: dt.datetime.combine(x, dt.datetime.min.time()))
    data['timestamp'] = (data['date'] - data['day_date']).apply(lambda x: x.total_seconds()) / 3600

    # fig = plt.figure()
    plt.plot(data.date, data.timestamp, 'bo', label=label)
    # plt.legend()
    plt.title("Occurrences of '{}'".format(label))
    plt.xlabel("Date")
    plt.ylabel("Hour of the day")
    plt.xticks(rotation=45)
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

    plt.plot(data.date, data.duration, 'bo', label=label)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
