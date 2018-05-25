# -*- coding: utf-8 -*-
import datetime as dt
import glob
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from Pattern_Discovery.Pattern_Discovery import pick_dataset

sns.set_style("darkgrid")
# plt.xkcd()

def main():
    dataset_name = 'KA'
    print("\n")
    print("###############################")
    print("EVALUATION OF THE MODEL on the {} HOUSE Dataset".format(dataset_name))
    print("##############################")
    print("\n")

    dirname = "./output/{}/Simulation Replications/*.csv".format(dataset_name)

    list_files = glob.glob(dirname)

    confidence_error = 0.9

    activity = ["prepare Breakfast START", "prepare Breakfast END"]

    # Original data
    original_dataset = pick_dataset(dataset_name)

    original_data_evaluation = compute_activity_time(data=original_dataset, start_label=activity[0],
                                                     end_label=activity[1])
    original_data_evaluation.fillna(0, inplace=True)
    plt.plot_date(original_data_evaluation.index, original_data_evaluation.duration / 3600, label="Original Data",
                  linestyle="-")
    plt.show()


    if len(list_files) == 0:
        raise FileNotFoundError("'{}' does not contains *.csv files".format(dirname))


    evaluation_sim_results = {}
    for filename in list_files:
        dataset = pd.read_csv(filename, delimiter=";")
        date_format = '%Y-%m-%d %H:%M:%S'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)

        evaluation_result = compute_activity_time(data=dataset, start_label=activity[0], end_label=activity[1],
                                                  time_step_in_days=1)
        if not evaluation_result.empty:
            evaluation_sim_results[filename] = evaluation_result
            end_drawing_date = evaluation_result.index[-1]

    original_data_evaluation = compute_activity_time(data=original_dataset, start_label=activity[0],
                                                     end_label=activity[1], end_date=end_drawing_date)

    # Build a large Dataframe with a date range index
    start_date = original_data_evaluation.index[0]
    end_date = original_data_evaluation.index[-1]
    big_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))

    for filename, evaluation_result in evaluation_sim_results.items():
        i = list_files.index(filename)
        evaluation_result.columns = ["simulation_{}".format(i)]

        big_df = pd.concat([big_df, evaluation_result[["simulation_{}".format(i)]]], axis=1)

    # Compute the Stochastic Mean & Error

    big_df['stoc_results'] = big_df.apply(compute_stochastic_error, args=(confidence_error,), axis=1)
    big_df['stoc_mean'] = big_df.stoc_results.apply(lambda x: x[0])
    big_df['stoc_lower'] = big_df.stoc_results.apply(lambda x: x[0] - (x[1] if not math.isnan(x[1]) else 0))
    big_df['stoc_upper'] = big_df.stoc_results.apply(lambda x: x[0] + (x[1] if not math.isnan(x[1]) else 0))

    big_df.drop(['stoc_results'], axis=1, inplace=True)

    # big_df[big_df.apply(lambda row: row.fillna(row.mean()), axis=1)

    big_df = pd.concat([big_df, original_data_evaluation['duration']], axis=1)
    big_df = big_df.loc[(big_df.index >= start_date) & (big_df.index <= end_date)].copy()
    big_df.fillna(0, inplace=True)

    # Turn the seconds into hours
    big_df = big_df / 3600

    ####################
    # DISPLAY RESULTS  #
    ####################
    fig, (ax1, ax2) = plt.subplots(2)

    # TIME STEP PLOT
    ax1.plot_date(big_df.index, big_df.duration, label="Original Data", linestyle="-")

    ax1.plot(big_df.index, big_df.stoc_mean, label="MEAN simulation", linestyle="-")

    ax1.fill_between(big_df.index, big_df.stoc_lower, big_df.stoc_upper,
                     label='{0:.0f}% Confidence Error'.format(confidence_error * 100), color='y', alpha=.3)

    # CUMSUM PLOT
    ax2.plot_date(big_df.index, big_df.duration.cumsum(), label="Original Data", linestyle="-")

    ax2.plot(big_df.index, big_df.stoc_mean.cumsum(), label="MEAN simulation", linestyle="--")

    ax2.fill_between(big_df.index, big_df.stoc_lower.cumsum(), big_df.stoc_upper.cumsum(),
                     label='{0:.0f}% Confidence Error'.format(confidence_error * 100), color='k', alpha=.3)

    ax1.title.set_text("{} House Dataset\nActivity [{} -- {}]".format(dataset_name, activity[0], activity[1]))
    ax1.set_ylabel('Duration (hours)')
    ax2.set_ylabel('Duration (hours)')
    ax2.set_xlabel('Date')
    ax2.set_title('Cumulative Evolution')
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    plt.gcf().autofmt_xdate()
    plt.show()


def compute_stochastic_error(row, error_confidence=0.9):
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
    t_sdt = stats.t.ppf(q=error_confidence, df=n - 1)
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

    return result[['duration']]


if __name__ == "__main__":
    main()