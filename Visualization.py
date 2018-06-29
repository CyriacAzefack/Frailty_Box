from datetime import datetime

import matplotlib.dates as dat
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random

from xED.Pattern_Discovery import *

sns.set_style('darkgrid')

# Map value to color
color_mapper = np.vectorize(lambda x: {10: 'red', 40: 'blue'}.get(x))


def main():
    dataset_name = 'aruba'

    dataset = pick_dataset(dataset_name)
    sim_id = 0
    replication_id = 1
    path = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/{}/Macro Activities Simulation results 15mn/dataset_simulation_rep_1.csv".format(
        dataset_name)
    dataset = pick_custom_dataset(path)

    start_date = dataset.date.min().to_pydatetime()
    end_date = start_date + dt.timedelta(days=5)

    visualize(dataset, start_date=start_date, end_date=end_date)
    # visualize(dataset, start_date=start_date, end_date=end_date, start_suffix='_begin', end_suffix='_end')


def visualize(data, start_date, end_date):
    """
    Visualize the log dataset
    :param data:
    :param start_date:
    :param end_date:
    :param start_suffix:
    :param end_suffix:
    :return:
    """

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


if __name__ == '__main__':
    main()
