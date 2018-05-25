from datetime import datetime

import matplotlib.dates as dat
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random

from Pattern_Discovery.Pattern_Discovery import *

sns.set_style('darkgrid')

# Map value to color
color_mapper = np.vectorize(lambda x: {10: 'red', 40: 'blue'}.get(x))


def main():
    dataset_name = 'aruba'

    dataset = pick_dataset(dataset_name, nb_days=40)
    # sim_id = 0
    # replication_id = 6
    # path = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/{}/ID_{}/Simulation Replications/dataset_simulation_{}_{}.csv".format(
    #     dataset_name, sim_id, sim_id + 1, replication_id)
    # dataset = pd.read_csv(path, delimiter=';')
    # date_format = '%Y-%m-%d %H:%M:%S.%f'
    # dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)

    start_date = dataset.date.min().to_pydatetime()
    end_date = start_date + dt.timedelta(days=10)

    visualize(dataset, start_date=start_date, end_date=end_date)
    # visualize(dataset, start_date=start_date, end_date=end_date, start_suffix='_begin', end_suffix='_end')


def visualize(data, start_date, end_date, start_suffix=' START', end_suffix=' END'):
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

    # 1. List the set of activities
    data['activity'] = ''
    data.loc[data.label.str.endswith(start_suffix), "activity"] = data.loc[
        data.label.str.endswith(start_suffix), "label"].apply(lambda x: x[0: x.rindex(start_suffix)])
    data.loc[data.label.str.endswith(end_suffix), "activity"] = data.loc[
        data.label.str.endswith(end_suffix), "label"].apply(lambda x: x[0: x.rindex(end_suffix)])

    activities = list(data.activity.unique())

    df_data = pd.DataFrame(columns=['activity', 'start', 'end', 'level'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xfmt = dat.DateFormatter('%d-%m-%y %H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    ax = ax.xaxis_date()

    for activity in activities:
        start_label = activity + start_suffix
        end_label = activity + end_suffix

        lvl = activities.index(activity)
        data_activity = data[data.label.isin([start_label, end_label])].copy()
        result = pd.DataFrame(columns=['activity', 'start', 'end', 'level'])

        last_label = None
        last_date = None
        for index, row in data_activity.iterrows():
            label = row['label']
            date = row['date'].to_pydatetime()
            if not last_label:
                last_label = label
                last_date = date
                continue

            if last_label == label:
                last_date = date
            else:
                if label == end_label:
                    # mean_ts = (date.timestamp() + last_date.timestamp()) / 2
                    # mean_date = dt.datetime.fromtimestamp(mean_ts).date()
                    # mean_date = dt.datetime.combine(mean_date, dt.datetime.min.time())
                    result.loc[len(result)] = [activity, last_date, date, lvl]

            last_label = label
            last_date = date
        if result.empty:
            print("Activity '{}' is not complete in the dataset".format(activity))
            continue
        result.start = pd.to_datetime(result.start).astype(datetime)
        result.end = pd.to_datetime(result.end).astype(datetime)

        color = random.rand(3, 1)
        plt.text(dat.date2num(date), lvl, activity, fontsize=14)
        ax = plt.hlines(result.level, dat.date2num(result.start), dat.date2num(result.end), label=activity,
                        linewidth=75, color=color)
        df_data = pd.concat([df_data, result], axis=0)
    # plt.legend()

    plt.show()

    print("done")


if __name__ == '__main__':
    main()
