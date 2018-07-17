# -*- coding: utf-8 -*-
import datetime as dt
import glob
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from xED.Candidate_Study import modulo_datetime
from xED.Pattern_Discovery import pick_dataset, pick_custom_dataset

sns.set_style("whitegrid")
# plt.xkcd()

def main():
    dataset_name = 'aruba'
    # Original data
    original_dataset = pick_dataset(dataset_name)

    print("\n")
    print("###############################")
    print("EVALUATION OF THE MODEL on the {} HOUSE Dataset".format(dataset_name))
    print("##############################")
    print("\n")

    period = dt.timedelta(days=1)
    freq = dt.timedelta(minutes=5)

    #####################
    #  COMPARE MODELS   #
    ####################
    # model_A_lab = '15mn'
    # model_A_dirname = "./output/{}/Macro Activities Simulation results {}/".format(dataset_name, model_A_lab)
    # model_A_name = "{} Macro time step {}".format(dataset_name, model_A_lab)

    # model_A_lab = '15mn'
    # model_A_dirname = "./output/{}/Simple Model Simulation results {}/".format(dataset_name, model_A_lab)
    # model_A_name = "{} time step {}".format(dataset_name, model_A_lab)
    #
    # model_B_lab = '30mn'
    # model_B_dirname = "./output/{}/Simple Model Simulation results {}/".format(dataset_name, model_B_lab)
    # model_B_name = "{} time step {}".format(dataset_name, model_B_lab)
    #
    # compare_models(original_dataset, model_A_name=model_A_name, model_A_dir=model_A_dirname, model_B_name=model_B_name,
    #                model_B_dir=model_B_dirname, period=period, time_step=freq)

    activity = "leave_home"

    label = '15mn'

    # dirname = "./output/{}/Simple Model Simulation results {}/".format(dataset_name, label)
    dirname = "./output/{}/Macro Activities Simulation results {}/".format(dataset_name, label)

    confidence = 0.9
    # Occurrence time validation
    auc_original, mean_absolute_error, lower_auc, upper_auc = auc_validation(activity, original_dataset, dirname,
                                                                             period, freq, confidence, display=True)
    print("Original Dataset AUC = {}".format(round(auc_original, 4)))
    print("Absolute ERROR AUC = {} ({}% of the original AUC)".format(round(mean_absolute_error, 4),
                                                                     round(100 * mean_absolute_error / auc_original,
                                                                           4)))
    print("{}% Confidence Interval : [{} ; {}]".format(confidence * 100, round(lower_auc, 4), round(upper_auc, 4)))

    # Duration Validation
    activity_duration_validation(activity, original_dataset, dirname, dataset_name, confidence)




    plt.show()


def all_activities_validation(original_dataset, dirname, period, time_step, display=True):
    validation_df = pd.DataFrame(columns=['label', 'original_auc', 'mae_percentage'])

    labels = original_dataset.label.unique()

    for label in labels:
        original_auc, mean_absolute_error, lower_auc, upper_auc = auc_validation(label, original_dataset, dirname,
                                                                                 period,
                                                                                 time_step, display=False)

        mae_percentage = round(100 * mean_absolute_error / original_auc, 2)

        validation_df.loc[len(validation_df)] = [label, original_auc, mae_percentage]

    y = list(validation_df.mae_percentage.values)
    x = list(validation_df.original_auc.values)
    labels = list(validation_df.label.values)

    if display:
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='r')

        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))

        plt.xlabel('Area Under the Histogram')
        plt.ylabel('Mean Absolute Error (%)')
        plt.title('Activities Validation')

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


def area_under_hist(data, label, period=dt.timedelta(days=1), time_step=dt.timedelta(minutes=5), display=False):
    '''
    Compute the area under the histogram curve
    :param data:
    :param label:
    :param period:
    :param time_step:
    :return:
    '''

    occurrences = data[data.label == label].copy()
    occurrences['relative_date'] = occurrences.date.apply(lambda x: modulo_datetime(x.to_pydatetime(), period))
    occurrences['time_step_id'] = occurrences['relative_date'] / time_step.total_seconds()
    occurrences['time_step_id'] = occurrences['time_step_id'].apply(math.floor)


    hist = occurrences.groupby(['time_step_id']).count()['date']
    start_date = occurrences.date.min().to_pydatetime()
    end_date = occurrences.date.max().to_pydatetime()
    nb_periods = math.floor((end_date - start_date).total_seconds() / period.total_seconds())

    # Normalize the histogram
    hist = hist / nb_periods

    AUC = sum(hist * time_step.total_seconds() / period.total_seconds())
    if display:
        plt.figure()
        hist.plot(kind="bar")
        plt.title('Activity : {}'.format(label))
        plt.legend()

    return AUC


def auc_validation(label, original_dataset, replications_directory, period=dt.timedelta(days=1),
                   time_step=dt.timedelta(minutes=10), confidence=0.9, display=True):
    '''
    Validation of the simulation replications using the Area Under the Curve of the label distribution
    :param label:
    :param original_dataset:
    :param replication_directory:
    :param period:
    :param time_step:
    :return:
    '''

    auc_original = area_under_hist(data=original_dataset, label=label, period=period, time_step=time_step,
                                   display=display)

    list_files = glob.glob(replications_directory + '*.csv')

    if len(list_files) == 0:
        raise FileNotFoundError("'{}' does not contains csv files".format(replications_directory))

    auc_replications = []
    for filename in list_files:
        dataset = pick_custom_dataset(filename)
        auc_replications.append(area_under_hist(data=dataset, label=label, period=period, time_step=time_step))

    auc_replications = np.asarray(auc_replications)
    student_mean, student_error = compute_stochastic_error(auc_replications, confidence=confidence)
    upper_auc = student_mean + student_error
    lower_auc = student_mean - student_error

    mean_absolute_error = sum(abs(auc_replications - auc_original)) / len(auc_replications)

    return auc_original, mean_absolute_error, lower_auc, upper_auc


def activity_duration_validation(label, original_dataset, replications_directory, dataset_name, confidence=0.9,
                                 display=True):
    '''
    Validation of the simulation replications using the Duration of the activity throughout the data
    :param label:
    :param original_dataset:
    :param replications_directory:
    :param confidence:
    :return:
    '''

    list_files = glob.glob(replications_directory + '*.csv')

    if len(list_files) == 0:
        raise FileNotFoundError("'{}' does not contains csv files".format(replications_directory))

    evaluation_sim_results = {}
    for filename in list_files:
        dataset = dataset = pick_custom_dataset(filename)
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

    diff_df = model_B_validation_df.join(model_A_validation_df, lsuffix='_B', rsuffix='_A')
    diff_df['error_lost'] = diff_df.mae_percentage_B - diff_df.mae_percentage_A
    diff_df.sort_values(['error_lost'], ascending=False, inplace=True)

    fig, ax = plt.subplots()
    sns.set_color_codes("pastel")
    sns.barplot(x="error_lost", y="label_A", data=diff_df, label="Error lost", color="b")

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="", xlabel="Points of Errors lost")
    sns.despine(left=True, bottom=True)
    plt.title("From Model '{}' --> Model '{}'".format(model_B_name, model_A_name))

    fig, ax = plt.subplots()
    y = list(model_A_validation_df.mae_percentage.values)
    x = list(model_A_validation_df.original_auc.values)
    labels = list(model_A_validation_df.label.values)

    ax.scatter(x, y, color='b', label=model_A_name)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))

    y = list(model_B_validation_df.mae_percentage.values)
    x = list(model_B_validation_df.original_auc.values)
    labels = list(model_B_validation_df.label.values)

    ax.scatter(x, y, color='r', label=model_B_name)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))

    plt.xlabel('Area Under the Histogram')
    plt.ylabel('Mean Absolute Error (%)')
    plt.legend()



if __name__ == "__main__":
    main()