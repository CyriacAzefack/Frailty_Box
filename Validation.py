# -*- coding: utf-8 -*-
import datetime as dt
import glob
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from scipy import stats

from xED.Candidate_Study import modulo_datetime
from xED.Pattern_Discovery import pick_dataset, pick_custom_dataset

sns.set_style("darkgrid")
# plt.xkcd()

def main():
    dataset_name = 'hh102'
    # Original data
    original_dataset = pick_dataset(dataset_name)

    print("\n")
    print("###############################")
    print("EVALUATION OF THE MODEL on the {} HOUSE Dataset".format(dataset_name))
    print("##############################")
    print("\n")

    period = dt.timedelta(days=1)
    time_step = dt.timedelta(minutes=5)

    #####################
    #  COMPARE MODELS   #
    ####################
    # model_A_activities_generation_method = 'Simple'
    # model_A_duration_generation_method = 'Normal'
    # model_A_time_step_min = 5
    # model_A_name = "{} Activities Model - {} - Simulation results {}mn".format(model_A_activities_generation_method,
    #                                                                                             model_A_duration_generation_method,
    #                                                                                             model_A_time_step_min)
    # model_A_dirname = "./output/{}/{}/".format(dataset_name, model_A_name)
    #
    # model_B_activities_generation_method = 'Macro'
    # model_B_duration_generation_method = 'Normal'
    # model_B_time_step_min = 15
    # model_B_name = "{} Activities Model - {} - Simulation results {}mn".format(model_B_activities_generation_method,
    #                                                                            model_B_duration_generation_method,
    #                                                                            model_B_time_step_min)
    # model_B_dirname = "./output/{}/{}/".format(dataset_name, model_B_name)
    #
    # compare_models(original_dataset, model_A_name=model_A_name, model_A_dir=model_A_dirname, model_B_name=model_B_name,
    #                model_B_dir=model_B_dirname, period=period, time_step=time_step)

    activity = "work"

    activities_generation_method = 'Macro'
    duration_generation_method = 'Normal'
    time_step_min = 5

    dirname = "./output/{}/{} Activities Model - {} - Time Step {}mn/".format(dataset_name,
                                                                                       activities_generation_method,
                                                                                       duration_generation_method,
                                                                                       time_step_min)


    # Occurrence time validation
    # r = validation_periodic_time_distribution(activity, original_dataset, dirname, period, time_step, display=True)

    all_activities_validation(original_dataset, dirname, period, time_step, display=False)


    # Duration Validation
    confidence_error = 0.9
    #activity_duration_validation(activity, original_dataset, dirname, dataset_name, confidence_error)

    plt.show()


def all_activities_validation(original_dataset, dirname, period, time_step, display=True):
    index = np.arange(int(period.total_seconds() / time_step.total_seconds()) + 1)
    validation_df = pd.DataFrame(index=index)

    labels = original_dataset.label.unique()

    labels.sort()
    for label in labels:
        label_validation_df = validation_periodic_time_distribution(label, original_dataset, dirname, period, time_step,
                                                                    confidence=0.9, display=False)
        errors = label_validation_df[['prob_error']]
        errors.columns = [label]
        validation_df = pd.concat([validation_df, errors], axis=1)

    validation_df = validation_df.T
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(validation_df, annot=False, fmt="d", ax=ax, cmap="Blues", linewidths=0.3, vmax=1)
    plt.title('Activities beginning time probability errors')




    if display:
        # AUC plot
        y = list(validation_df.mae_auc_percentage.values)
        x = list(validation_df.original_auc.values)
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='r')

        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))

        plt.xlabel('Area Under the Histogram')
        plt.ylabel('Mean Absolute Error (%)')
        plt.title('Area Under Histogram')

        # SSE plot
        y = list(validation_df.mae_sse_percentage.values)
        x = list(validation_df.original_sse.values)
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='r')

        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))

        plt.xlabel('SSE on the fitted distribution')
        plt.ylabel('Mean Absolute Error (%)')
        plt.title('SSE on the fitted distribution')

    return validation_df


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
    index = np.arange(int(period.total_seconds() / time_step.total_seconds()) + 1)
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
        plt.title('Probability of occurrence of the label : \'{}\''.format(label))
        plt.xlabel('Day hour')
        plt.ylabel('Probability')
        plt.gcf().autofmt_xdate()
        plt.show()
    return time_dist[['time_step_id', 'prob']]


def validation_periodic_time_distribution(label, original_dataset, replications_directory, period, time_step,
                                          confidence=0.9, display=True):
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
    replications_errors = []
    for filename in list_files:
        dataset = pick_custom_dataset(filename)
        repl_time_dist = periodic_time_distribution(data=dataset, label=label, period=period, time_step=time_step,
                                                    display=False)
        if repl_time_dist is None:
            continue
        replications_time_dist[filename] = repl_time_dist
        duo_df = original_time_dist.join(repl_time_dist.set_index('time_step_id'), on='time_step_id', rsuffix='_simul')
        duo_df.fillna(0, inplace=True)
        duo_df['error'] = (duo_df['prob_simul'] - duo_df['prob']) / duo_df['prob']
        error = np.sqrt(np.sum(np.power(duo_df.error.values, 2)))  # Quadratic error

        # error = duo_df.error.corr(duo_df.prob_simul)
        replications_errors.append(error)

    replications_errors = np.asarray(replications_errors)


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
        big_df['prob_results'] = big_df.apply(compute_stochastic, args=(confidence,), axis=1)
        big_df['prob_mean'] = big_df.prob_results.apply(lambda x: x[0])
        big_df['prob_lower'] = big_df.prob_results.apply(lambda x: x[0] - (x[1] if not math.isnan(x[1]) else 0))
        big_df['prob_upper'] = big_df.prob_results.apply(lambda x: x[0] + (x[1] if not math.isnan(x[1]) else 0))

    big_df.drop(['prob_results'], axis=1, inplace=True)
    big_df = big_df.join(original_time_dist.set_index('time_step_id'), on='time_step_id')
    big_df.fillna(0, inplace=True)

    big_df['prob_error'] = abs(big_df['prob'] - big_df['prob_mean'])

    if display:
        now = dt.date.today()
        begin_day = dt.datetime.fromordinal(now.toordinal())
        big_df['date'] = pd.Series({x: begin_day + x * time_step for x in big_df.index})

        big_df.set_index(['date'], inplace=True)

        fig, ax = plt.subplots()

        # TIME STEP PLOT
        ax.plot(big_df.index, big_df.prob, label="Original Data", linestyle="-")

        ax.plot(big_df.index, big_df.prob_mean, label="MEAN simulation", linestyle="-")

        # ax.fill_between(big_df.index, big_df.prob_lower, big_df.prob_upper,
        #                 label='{0:.0f}% Confidence Error'.format(confidence * 100), color='k', alpha=.25)

        ax.set_ylabel('Probability')
        ax.set_xlabel('Day hour')
        ax.legend()
        plt.title("Daily time distribution of the label '{}'".format(label))
        plt.gcf().autofmt_xdate()
        plt.show()

    return big_df[['prob', 'prob_mean', 'prob_error']]




def activity_duration_validation(label, original_dataset, replications_directory, dataset_name, confidence=0.9,
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
    big_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))

    for filename, evaluation_result in evaluation_sim_results.items():
        i = list_files.index(filename)
        evaluation_result.columns = ["simulation_{}".format(i)]
        big_df = pd.concat([big_df, evaluation_result[["simulation_{}".format(i)]]], axis=1)

    # Compute the Stochastic Mean & Error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        big_df['stoc_results'] = big_df.apply(compute_stochastic, args=(confidence,), axis=1)
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

    if display:
        fig, (ax1, ax2) = plt.subplots(2)

        # TIME STEP PLOT
        ax1.plot_date(big_df.index, big_df.duration, label="Original Data", linestyle="-")

        ax1.plot(big_df.index, big_df.stoc_mean, label="MEAN simulation", linestyle="--")

        ax1.fill_between(big_df.index, big_df.stoc_lower, big_df.stoc_upper,
                         label='{0:.0f}% Confidence Error'.format(confidence * 100), color='k', alpha=.25)

        # CUMSUM PLOT
        ax2.plot_date(big_df.index, big_df.duration.cumsum(), label="Original Data", linestyle="-")

        ax2.plot(big_df.index, big_df.stoc_mean.cumsum(), label="MEAN simulation", linestyle="--")

        ax2.fill_between(big_df.index, big_df.stoc_lower.cumsum(), big_df.stoc_upper.cumsum(),
                         label='{0:.0f}% Confidence Error'.format(confidence * 100), color='k', alpha=.25)

        ax1.title.set_text("{} House Dataset\nActivity [{}]".format(dataset_name, label))
        ax1.set_ylabel('Duration (hours)')
        ax2.set_ylabel('Duration (hours)')
        ax2.set_xlabel('Date')
        ax2.set_title('Cumulative Evolution')
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper left")
        plt.gcf().autofmt_xdate()


def compare_models(original_dataset, model_A_name, model_B_name, model_A_dir, model_B_dir, period=dt.timedelta(days=1),
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
    model_A_validation_df = all_activities_validation(original_dataset, model_A_dir, period, time_step, display=False)
    model_B_validation_df = all_activities_validation(original_dataset, model_B_dir, period, time_step, display=False)

    model_A_validation_df['model_name'] = model_A_name
    model_B_validation_df['model_name'] = model_B_name

    df = pd.concat([model_A_validation_df, model_B_validation_df], sort=True)

    ################
    # AUC
    ################
    # fig, ax = plt.subplots()
    g = sns.catplot(data=df, x="model_name", y="mae_auc_percentage", hue="label", capsize=.2, height=6, aspect=.75,
                    kind="point")
    plt.xlabel('Model Name')
    plt.ylabel('AUC MAE Percentage')
    g.despine(left=True)
    plt.title('AUC Mean Absolute Error')

    fig, ax = plt.subplots()
    y = list(model_A_validation_df.mae_auc_percentage.values)
    x = list(model_A_validation_df.original_auc.values)
    labels = list(model_A_validation_df.label.values)

    ax.scatter(x, y, color='b', label=model_A_name)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))

    y = list(model_B_validation_df.mae_auc_percentage.values)
    x = list(model_B_validation_df.original_auc.values)
    labels = list(model_B_validation_df.label.values)

    ax.scatter(x, y, color='r', label=model_B_name)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))

    plt.xlabel('Area Under the Histogram')
    plt.ylabel('Mean Absolute Error (%)')
    plt.legend()

    ################
    # SSE
    ################
    # fig, ax = plt.subplots()
    g = sns.catplot(data=df, x="model_name", y="mae_sse_percentage", hue="label", capsize=.2, height=6, aspect=.75,
                    kind="point")
    plt.xlabel('Model Name')
    plt.ylabel('SSE MAE Percentage')
    g.despine(left=True)
    plt.title('SSE on fitted distribution Mean Absolute Error')

    fig, ax = plt.subplots()
    y = list(model_A_validation_df.mae_sse_percentage.values)
    x = list(model_A_validation_df.original_sse.values)
    labels = list(model_A_validation_df.label.values)

    ax.scatter(x, y, color='b', label=model_A_name)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))

    y = list(model_B_validation_df.mae_sse_percentage.values)
    x = list(model_B_validation_df.original_sse.values)
    labels = list(model_B_validation_df.label.values)

    ax.scatter(x, y, color='r', label=model_B_name)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))

    plt.xlabel('SSE on fitted distribution')
    plt.ylabel('Mean Absolute Error (%)')
    plt.legend()




def compute_dist_sse(dist_name, params, data, bins=100):
    """
    Compute Sum Squared Error of the data on the fitted distribution
    :param distribution:
    :param params:
    :param data:
    :return:
    """

    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    dist = getattr(st, dist_name)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
    sse = np.sum(np.power(y - pdf, 2.0))

    return sse

if __name__ == "__main__":
    main()