# -*- coding: utf-8 -*-
import datetime as dt
import glob
import math

# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

from xED_Algorithm.xED_Algorithm import pick_dataset


# sns.set_style("whitegrid")


def main():
    print("\n")
    print("###############################")
    print("EVALUATION OF THE MODEL on the KA HOUSE Dataset")
    print("##############################")
    print("\n")

    dirname = "./output/KA/Simulation Replications/*.csv"

    list_files = glob.glob(dirname)

    nb_replications = len(list_files)

    activity = ["leave house START", "leave house END"]

    # Original data
    original_dataset = pick_dataset('KA')

    # Plot a simple histogram with binsize determined automatically
    f, ax = plt.subplots(1, 1)
    # plot

    evaluation_sim_results = {}
    for filename in list_files:
        dataset = pd.read_csv(filename, delimiter=";")
        date_format = '%Y-%m-%d %H:%M:%S'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)

        evaluation_result = compute_activity_time(data=dataset, start_label=activity[0], end_label=activity[1],
                                                  time_step_in_days=1)
        if not evaluation_result.empty:
            plt.plot_date(evaluation_result.index, evaluation_result.duration.cumsum(),
                          label="Simulation N°{}/{}".format(list_files.index(filename) + 1, nb_replications),
                          linestyle="-")

        evaluation_sim_results[list_files.index(filename)] = evaluation_result

    base_result = compute_activity_time(data=original_dataset, start_label=activity[0], end_label=activity[1],
                                        start_date=evaluation_result.index[0])

    plt.plot_date(base_result.index, base_result.duration.cumsum(), label="Original Data", linestyle=":")

    plt.title("Duration of the Activity [{} -- {}]".format(activity[0], activity[1]))
    ax.legend()
    plt.gcf().autofmt_xdate()
    plt.show()

    # Build a large Dataframe with a date range index
    start_date = base_result.index[0]
    end_date = base_result.index[-1]
    big_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))

    for i in range(len(list_files)):
        evaluation_result = evaluation_sim_results[i]
        evaluation_result.columns = ["simulation_{}".format(i)]

        big_df = pd.concat([big_df, evaluation_result[["simulation_{}".format(i)]]], axis=1)

    big_df = big_df.loc[(big_df.index >= start_date) & (big_df.index <= end_date)].copy()

    big_df['stoc_error'] = big_df.apply(compute_stochastic_error, axis=1)

    print("")


def compute_stochastic_error(row):
    '''
    Compute the stochastic error given by the array
    :param array: an array of numbers
    :return: mean and the error
    '''

    array = row.values

    # We remove all the 'NaN' values
    array = array[~np.isnan(array)]

    mean = np.mean(array)
    std = np.std(array)

    n = len(array)
    # We find the t-distribution value of the student law
    t_sdt = stats.t.ppf(q=0.9, df=n - 1)
    error = t_sdt * (std / math.sqrt(n))

    return mean, error


def compute_activity_time(data, start_label, end_label, start_date=None, end_date=None, time_step_in_days=1):
    '''
    Compute the Average time between start_label and end_label (in seconds) time into the data
    :param data:
    :param start_date:
    :param end_date:
    :param time_step_in_days: Number of days
    :return: the average sleeping time
    '''

    if not start_date:
        start_date = data.date.min().to_pydatetime()

    if not end_date:
        end_date = data.date.max().to_pydatetime()

    labels = [start_label, end_label]

    # Filter the interesting data
    data = data[(data.date >= start_date) & (data.date <= end_date) & (data.label.isin(labels))].copy()
    data.reset_index(inplace=True, drop=True)
    # Sort data by date
    data.sort_values(by=['date'], ascending=True, inplace=True)

    result = pd.DataFrame(columns=['date', 'duration'])

    last_label = None
    last_date = None
    for index, row in data.iterrows():
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
                mean_ts = (date.timestamp() + last_date.timestamp()) / 2
                mean_date = dt.datetime.fromtimestamp(mean_ts).date()
                mean_date = dt.datetime.combine(mean_date, dt.datetime.min.time())
                result.loc[len(result)] = [mean_date, (date - last_date).total_seconds()]

        last_label = label
        last_date = date

    if result.empty:
        return result

    result.sort_values(by=['date'], ascending=True, inplace=True)
    result.set_index('date', inplace=True)
    result['date'] = result.index

    try:
        result = result.groupby(pd.Grouper(key='date', freq='{}D'.format(time_step_in_days))).sum()
    except:
        print("Error happenned!!")

    return result


if __name__ == "__main__":
    main()
