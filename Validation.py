# -*- coding: utf-8 -*-
import datetime as dt
import glob
import math
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import mean_squared_error

from xED.Candidate_Study import modulo_datetime
from xED.Pattern_Discovery import pick_dataset, pick_custom_dataset

sns.set_style("darkgrid")
# plt.xkcd()

def main():

    period = dt.timedelta(days=1)
    time_step = dt.timedelta(minutes=1)

    #####################
    #  COMPARE MODELS   #
    ####################

    models = []
    models.append({
        'name': 'hh101',
        'label': 'Tstep=5mn',
        'sim_id': 1,
        'pattern_id': 0
    })

    models.append({
        'name': 'hh101',
        'label': 'Tstep=15mn',
        'sim_id': 2,
        'pattern_id': 0
    })

    models.append({
        'name': 'hh101',
        'label': 'Tstep=30mn',
        'sim_id': 3,
        'pattern_id': 0
    })

    compare_models(models, period=period, time_step=time_step)

    # dataset_name = 'hh101'
    # # Original data
    # original_dataset = pick_dataset(dataset_name)
    #
    # print("\n")
    # print("###############################")
    # print("EVALUATION OF THE MODEL on the {} HOUSE Dataset".format(dataset_name))
    # print("##############################")
    #
    # activity = "sleeping"
    #
    # simulation_id = 1
    # pattern_folder_id = 0
    #
    # dirname = "./output/{}/Simulation/Simulation_X{}_Pattern_ID_{}/".format(dataset_name, simulation_id,
    #                                                                         pattern_folder_id)
    #
    # # dirname = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/aruba/Macro Activities Model - Normal - Simulation results 15mn/"
    #
    # # Occurrence time validation
    # #r = validation_periodic_time_distribution(activity, original_dataset, dirname, period, time_step, display=True)
    #
    # all_activities_validation(original_dataset, dirname, period, time_step, display=True)


    # Duration Validation
    confidence_error = 0.9
    # activity_duration_validation(activity, original_dataset, dirname, confidence_error)

    plt.show()


def all_activities_validation(original_dataset, dirname, period, time_step, display=True):
    index = np.arange(int(period.total_seconds() / time_step.total_seconds()))
    validation_df = pd.DataFrame(index=index)

    all_validation_df = pd.DataFrame(columns=['rmse', 'cum_in'])
    labels_rmse = []


    labels = original_dataset.label.unique()

    labels.sort()
    for label in labels:
        label_validation_df, label_rmse = validation_periodic_time_distribution(label, original_dataset, dirname,
                                                                                period, time_step,
                                                                                display=False)
        daily_in, cum_in = activity_duration_validation(label, original_dataset, dirname, display=False)

        errors = label_validation_df[['prob_error']]
        errors.columns = [label]
        validation_df = pd.concat([validation_df, errors], axis=1)
        all_validation_df.loc[label] = [label_rmse, cum_in]

        labels_rmse.append(label_rmse)

    labels_rmse = np.asarray(labels_rmse)


    if display:
        sns.distplot(labels_rmse, norm_hist=1, kde=1)
        plt.title("Activities RMSE distribution")

        sns.heatmap(validation_df.T, annot=False, fmt="d", cmap="Blues", linewidths=0.3, vmax=1)
        plt.title('Activities beginning time probability errors')

        sns.jointplot(x='rmse', y='cum_in', kind="kde", data=all_validation_df, ylim=(0, 1))
        plt.title('Labels representation')
        plt.show()

    return validation_df, labels_rmse, all_validation_df




def compute_activity_time(data, label, start_date=None, end_date=None, time_step_in_days=1):
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

    # Filter the interesting data
    result = data[(data.date >= start_date) & (data.date <= end_date) & (data.label == label)].copy()

    result['duration'] = result.end_date - result.date
    result['duration'] = result['duration'].apply(
        lambda x: x.total_seconds())
    result.reset_index(inplace=True, drop=True)
    # Sort data by date
    result.sort_values(by=['date'], ascending=True, inplace=True)

    result.set_index('date', inplace=True)
    result['date'] = result.index

    try:
        result = result.groupby(pd.Grouper(key='date', freq='{}D'.format(time_step_in_days))).sum()
    except:
        print("Error happenned!!")

    return result[['duration']]


def periodic_time_distribution(data, label, period, time_step, display=False):
    """
    Return the probability of the label occurring for each time step id
    :param data:
    :param label:
    :param period:
    :param time_step:
    :param display:
    :return:
    """

    occurrences = data[data.label == label].copy()
    index = np.arange(int(period.total_seconds() / time_step.total_seconds()))
    index_df = pd.DataFrame(index, columns=['time_step_id'])

    if occurrences.empty:
        print('The label "{}" does not exist in the dataset'.format(label))
        return None
    occurrences['relative_date'] = occurrences.date.apply(lambda x: modulo_datetime(x.to_pydatetime(), period))
    occurrences['time_step_id'] = occurrences['relative_date'] / time_step.total_seconds()
    occurrences['time_step_id'] = occurrences['time_step_id'].apply(math.floor)

    time_dist = occurrences.groupby(['time_step_id']).size().reset_index(name='prob')
    time_dist['prob'] = time_dist['prob'] / len(occurrences)

    time_dist.sort_values(['time_step_id'], ascending=True, inplace=True)

    # time_dist = pd.concat([time_dist, index_df], axis=1)
    time_dist = index_df.join(time_dist.set_index('time_step_id'))

    time_dist.fillna(0, inplace=True)

    if display:
        now = dt.date.today()
        begin_day = dt.datetime.fromordinal(now.toordinal())
        time_dist['date'] = time_dist['time_step_id'].apply(lambda x: begin_day + x * time_step)

        fig, ax = plt.subplots()
        sns.lineplot(x='date', y='prob', data=time_dist)
        plt.title('Occurrence probability of  : \'{}\''.format(label), fontsize=14)
        plt.xlabel('Hour of the day')
        plt.ylabel('Probability')
        ax.set_xlim(time_dist.date.min(), time_dist.date.max())
        # plt.gcf().autofmt_xdate()
        myFmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(myFmt)
        plt.show()
    return time_dist[['time_step_id', 'prob']]


def validation_periodic_time_distribution(label, original_dataset, replications_directory, period, time_step,
                                          display=True):
    """
    Validation of the occurrence time of a specific label
    :param label:
    :param original_dataset:
    :param replications_directory:
    :param period:
    :param time_step:
    :param confidence:
    :param display:
    :return:
    """

    # Original Dataset
    original_time_dist = periodic_time_distribution(original_dataset, label, period, time_step, display=False)

    # Simulation replications
    list_files = glob.glob(replications_directory + '*.csv')
    if len(list_files) == 0:
        raise FileNotFoundError("'{}' does not contains csv files".format(replications_directory))

    replications_time_dist = {}

    for filename in list_files:
        dataset = pick_custom_dataset(filename)
        repl_time_dist = periodic_time_distribution(data=dataset, label=label, period=period, time_step=time_step,
                                                    display=False)
        if repl_time_dist is None:
            continue
        replications_time_dist[filename] = repl_time_dist
        duo_df = original_time_dist.join(repl_time_dist.set_index('time_step_id'), on='time_step_id', rsuffix='_simul')
        duo_df.fillna(0, inplace=True)


    big_df = pd.DataFrame()
    for filename, repl_time_dist in replications_time_dist.items():
        i = list_files.index(filename)
        repl_time_dist.columns = ["time_step_id", "simulation_{}".format(i)]
        if len(big_df) == 0:
            big_df = pd.concat([big_df, repl_time_dist], axis=1)
        else:
            big_df = big_df.join(repl_time_dist.set_index('time_step_id'), on='time_step_id')

    big_df.fillna(0, inplace=True)
    big_df.set_index(['time_step_id'], inplace=True)
    # big_df.drop(['time_step_id'], axis=1, inplace=True)

    # Compute the Stochastic Mean & Error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        big_df['prob_repl_mean'] = big_df.apply(np.mean, axis=1)


    big_df = big_df.join(original_time_dist.set_index('time_step_id'), on='time_step_id')
    big_df.fillna(0, inplace=True)

    big_df['prob_error'] = abs(big_df['prob'] - big_df['prob_repl_mean'])

    rmse = mean_squared_error(big_df.prob, big_df.prob_repl_mean)

    if display:
        now = dt.date.today()
        begin_day = dt.datetime.fromordinal(now.toordinal())
        big_df['date'] = pd.Series({x: begin_day + x * time_step for x in big_df.index})

        big_df.set_index(['date'], inplace=True)

        fig, ax = plt.subplots()

        # TIME STEP PLOT
        ax.plot(big_df.index, big_df.prob, label="Real Data", linestyle="-", color='blue')

        ax.plot(big_df.index, big_df.prob_repl_mean, label="Simulated Data", linestyle="--", color='red')

        # ax.fill_between(big_df.index, big_df.prob_lower, big_df.prob_upper,
        #                 label='{0:.0f}% Confidence Error'.format(confidence * 100), color='k', alpha=.25)

        ax.set_ylabel('Probability')
        ax.set_xlabel('Hour of the day')
        ax.legend()
        plt.title("Daily time distribution of the label '{}'".format(label))
        myFmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(myFmt)
        #ax.set_xlim(big_df.index.min(), right=big_df.index.max())
        plt.show()

    return big_df[['prob', 'prob_repl_mean', 'prob_error']], rmse


def activity_duration_validation(label, original_dataset, replications_directory, confidence=0.9,
                                 display=True):
    """
    Validation of the simulation replications using the Duration of the activity throughout the data
    :param label:
    :param original_dataset:
    :param replications_directory:
    :param dataset_name:
    :param confidence:
    :param display:
    :return:
    """

    list_files = glob.glob(replications_directory + '*.csv')

    if len(list_files) == 0:
        raise FileNotFoundError("'{}' does not contains csv files".format(replications_directory))

    evaluation_sim_results = {}
    for filename in list_files:
        dataset = pick_custom_dataset(filename)
        evaluation_result = compute_activity_time(data=dataset, label=label, time_step_in_days=1)
        if not evaluation_result.empty:
            evaluation_sim_results[filename] = evaluation_result
            end_drawing_date = evaluation_result.index[-1]

    original_data_evaluation = compute_activity_time(data=original_dataset, label=label, end_date=end_drawing_date)

    # Build a large Dataframe with a date range index
    start_date = original_data_evaluation.index[0]
    end_date = original_data_evaluation.index[-1]
    repl_durations = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))

    for filename, evaluation_result in evaluation_sim_results.items():
        i = list_files.index(filename)
        evaluation_result.columns = ["simulation_{}".format(i)]
        repl_durations = pd.concat([repl_durations, evaluation_result[["simulation_{}".format(i)]]], axis=1)

    # Compute the Stochastic Mean & Error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        repl_durations['stoc_results'] = repl_durations.apply(compute_stochastic, args=(confidence,), axis=1)
        repl_durations['stoc_mean'] = repl_durations.stoc_results.apply(lambda x: x[0])
        repl_durations['stoc_lower'] = repl_durations.stoc_results.apply(
            lambda x: x[0] - (x[1] if not math.isnan(x[1]) else 0))
        repl_durations['stoc_upper'] = repl_durations.stoc_results.apply(
            lambda x: x[0] + (x[1] if not math.isnan(x[1]) else 0))

    repl_durations.drop(['stoc_results'], axis=1, inplace=True)

    # repl_durations[repl_durations.apply(lambda row: row.fillna(row.mean()), axis=1)

    repl_durations = pd.concat([repl_durations, original_data_evaluation['duration']], axis=1)
    repl_durations = repl_durations.loc[
        (repl_durations.index >= start_date) & (repl_durations.index <= end_date)].copy()
    repl_durations.fillna(0, inplace=True)

    repl_durations['in_error_daily'] = (repl_durations['stoc_lower'] <= repl_durations['duration']) & (
            repl_durations['stoc_upper'] >= repl_durations['duration'])

    repl_durations['in_error_cum'] = (repl_durations['stoc_lower'].cumsum() <= repl_durations['duration'].cumsum()) & (
            repl_durations['stoc_upper'].cumsum() >= repl_durations['duration'].cumsum())

    daily_in = repl_durations['in_error_daily'].sum() / len(repl_durations)
    cum_in = repl_durations['in_error_cum'].sum() / len(repl_durations)


    ####################
    # DISPLAY RESULTS  #
    ####################

    if display:
        # Turn the seconds into hours
        repl_durations = repl_durations / 3600
        fig, (ax1, ax2) = plt.subplots(2)

        # TIME STEP PLOT
        ax1.plot_date(repl_durations.index, repl_durations.duration, label="Original Data", linestyle="-")

        ax1.plot(repl_durations.index, repl_durations.stoc_mean, label="MEAN simulation", linestyle="--")

        ax1.fill_between(repl_durations.index, repl_durations.stoc_lower, repl_durations.stoc_upper,
                         label='{0:.0f}% Confidence Error'.format(confidence * 100), color='k', alpha=.25)

        # CUMSUM PLOT
        ax2.plot_date(repl_durations.index, repl_durations.duration.cumsum(), label="Original Data", linestyle="-")

        ax2.plot(repl_durations.index, repl_durations.stoc_mean.cumsum(), label="MEAN simulation", linestyle="--")

        ax2.fill_between(repl_durations.index, repl_durations.stoc_lower.cumsum(), repl_durations.stoc_upper.cumsum(),
                         label='{0:.0f}% Confidence Error'.format(confidence * 100), color='k', alpha=.25)

        ax1.title.set_text("Event duration for activity '{}'".format(label))
        ax1.set_ylabel('Duration (hours)')
        ax2.set_ylabel('Duration (hours)')
        ax2.set_xlabel('Date')
        ax2.set_title('Cumulative Evolution')
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper left")
        plt.gcf().autofmt_xdate()
        plt.show()

    return daily_in, cum_in


def compare_models(models, period=dt.timedelta(days=1),
                   time_step=dt.timedelta(minutes=10)):
    """
    Compare 2 simulation models
    :param original_dataset:
    :param model_A_name:
    :param model_B_name:
    :param model_A_dir:
    :param model_B_dir:
    :return:
    """

    list_validation_df = []

    all_df = pd.DataFrame(columns=['rmse', 'cum_in', 'model_name'])

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for model in models:
        dataset = pick_dataset(model['name'])

        dirname = "./output/{}/Simulation/Simulation_X{}_Pattern_ID_{}/".format(model['name'],
                                                                                model['sim_id'],
                                                                                model['pattern_id'])

        validation_df, rmse, all_validation_df = all_activities_validation(dataset, dirname, period, time_step,
                                                                           display=False)

        rmse_df = pd.DataFrame(rmse * 1e5, columns=[model['name']])

        label = "{}_Sim{}_Pattern{} : Mean_RMSE={:.2e}".format(model['name'], model['sim_id'], model['pattern_id'],
                                                               np.mean(rmse))

        # label = "{}\tMean RMSE={:.2e}".format(model['name'], np.mean(rmse))
        label = model['label']

        validation_df['model_name'] = label
        all_validation_df['model_name'] = label

        all_df = all_df.append(all_validation_df)

        list_validation_df.append(validation_df)

        sns.kdeplot(rmse_df[model['name']], label=label, shade=True, shade_lowest=False, ax=ax1)
        sns.kdeplot(all_validation_df['cum_in'], label=label, shade=True, shade_lowest=False, ax=ax2)

    df = pd.concat(list_validation_df, sort=True)
    df.to_csv('validation_results.csv', index=False)

    ax1.set_xlabel('Labels RMSE (x 1e-05)')
    ax1.set_ylabel('Density')
    ax2.set_xlabel('Ratio in the confidence interval')
    ax2.set_ylabel('Density')

    ax1.set_title('RMSE distribution')
    ax2.set_title('Confidence interval ratio distribution')
    ax1.set_xlim([0, 35])
    ax2.set_xlim([0, 1])
    plt.legend()
    plt.show()

    sns.boxplot(x='model_name', y='cum_in', data=all_df)
    plt.xlabel('Dataset')
    plt.ylabel('Ratio in the confidence interval')
    # plt.title('Labels cumulative event duration ratio in the confidence interval')
    plt.show()


def compute_stochastic(row, error_confidence=0.9):
    array = row.values

    # We remove all the 'NaN' values
    array = array[~np.isnan(array)]

    return compute_stochastic_error(array=array, confidence=0.9)


def compute_stochastic_error(array, confidence=0.9):
    '''
    Compute the stochastic error given by the array
    :param array: an array of numbers
    :return: mean and the error
    '''

    mean = np.mean(array)
    std = np.std(array)

    n = len(array)
    # We find the t-distribution value of the student law
    t_sdt = stats.t.ppf(q=confidence, df=n - 1)
    error = t_sdt * (std / math.sqrt(n))

    return mean, error

if __name__ == "__main__":
    main()