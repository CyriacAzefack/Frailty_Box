# -*- coding: utf-8 -*-
import glob
import warnings

import seaborn as sns
from Bio import pairwise2
# Import format_alignment method
from Bio.pairwise2 import format_alignment

from Data_Drift.Features_Extraction import *
from xED.Candidate_Study import *

# Import pairwise2 module

sns.set_style("darkgrid")
sns.set(font_scale=1.4)
plt.xkcd()

def main():

    period = dt.timedelta(days=1)
    time_step = dt.timedelta(minutes=5)

    #####################
    #  COMPARE MODELS   #
    ####################

    # models = []
    # models.append({
    #     'name': 'aruba',
    #     'label': 'Aruba',
    #     'sim_id': 2,
    #     'pattern_id': 0
    # })
    #
    # models.append({
    #     'name': 'hh101',
    #     'label': 'HH101',
    #     'sim_id': 2,
    #     'pattern_id': 0
    # })
    #
    # models.append({
    #     'name': 'KA',
    #     'label': 'KA',
    #     'sim_id': 2,
    #     'pattern_id': 0
    # })
    #
    # compare_models(models, period=period, time_step=time_step)

    dataset_name = 'aruba'
    # Original data
    original_dataset = pick_dataset(dataset_name)

    labels = original_dataset.label.unique()
    labels.sort()

    start_date = original_dataset.date.min().to_pydatetime()
    end_date = original_dataset.date.max().to_pydatetime()

    # Compute the number of periods
    nb_days = math.floor((end_date - start_date).total_seconds() / period.total_seconds())
    training_days = nb_days

    training_start_date = start_date
    training_end_date = start_date + dt.timedelta(days=training_days)

    testing_start_date = training_start_date
    testing_duration = dt.timedelta(days=nb_days)
    testing_end_date = testing_start_date + testing_duration

    training_dataset = original_dataset[
        (original_dataset.date > training_start_date) & (original_dataset.date < training_end_date)].copy()
    testing_dataset = original_dataset[
        (original_dataset.date > testing_start_date) & (original_dataset.date < testing_end_date)].copy()

    # original_amb = plot_ambulatogram(original_dataset, labels, start_date, nb_days)

    print("\n")
    print("########################################################################")
    print("# EVALUATION OF THE MODEL on the {} HOUSE Dataset #".format(dataset_name))
    print("##########################################################################")

    activity = "sleeping"

    simulation_id = 1
    pattern_folder_id = 0

    dirname = "./output/{}/Simulation/Simulation_X{}_Pattern_ID_{}/".format(dataset_name, simulation_id,
                                                                            pattern_folder_id)
    labels = original_dataset.label.unique()

    labels.sort()

    results_df = pd.DataFrame(columns=['hist_inters', 'density_area', 'rmse'])

    for label in labels:
        results_df.loc[label] = validation_periodic_time_distribution(label=label, real_dataset=testing_dataset,
                                                                      replications_directory=dirname,
                                                                      period=period, bin_width=5, display=True)

    results_df.to_csv(dirname + '../Aruba_label_validation_5mn_bin.csv', sep=";", index=True)

    print("###################################")
    print("#  SEQUENCE ALIGNMENT VALIDATION  #")
    print("###################################")

    # Compute the label similarity matrix with daily density
    training_dataset['relative_date'] = training_dataset.date.apply(
        lambda x: modulo_datetime(x.to_pydatetime(), dt.timedelta(days=1)))

    labels = training_dataset.label.unique()

    label_similarity_matrix = np.zeros((len(labels), len(labels)))

    for i in range(len(labels)):
        timestampA = training_dataset[training_dataset.label == labels[i]].relative_date.values
        for j in range(len(labels)):
            timestampB = training_dataset[training_dataset.label == labels[j]].relative_date.values
            label_similarity_matrix[i][j] = density_intersection_area(timestampA, timestampB)

    # Correspondance between activities and letter of the alphabet
    alphabet = {}

    for i in range(len(labels)):
        alphabet[labels[i]] = chr(ord('A') + i)
        print('\t{} : {}'.format(chr(ord('A') + i), labels[i]))

    letter_similarity = {}

    for i in range(len(labels)):
        for j in range(len(labels)):
            letter1 = alphabet[labels[i]]
            letter2 = alphabet[labels[j]]

            letter_similarity[(letter1, letter2)] = label_similarity_matrix[i][j]

    # sequence_alignement_validation(original_data=testing_dataset, directory=dirname, period=period, alphabet=alphabet,
    #                                letter_similarity=letter_similarity, display=True)


    # Duration Validation
    # confidence_error = 0.9
    labels_count_df = pd.DataFrame(columns=['duration'])

    for label in labels:
        labels_count_df.loc[label] = len(original_dataset[original_dataset.label == label])

    labels_count_df.sort_values(by=['duration'], ascending=False, inplace=True)

    explode = (0, 0, 0, 0, 0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    cmap = plt.get_cmap('tab20c')
    colors = cmap(np.linspace(0., 1., len(labels_count_df)))
    labels_count_df.duration.plot(kind='pie', fontsize=18, explode=explode, colors=colors, startangle=90)

    # wedges, texts = ax.pie(labels_count_df.duration, wedgeprops=dict(width=0.5), startangle=0, colors=colors)
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
    #     ax.annotate(labels_count_df.index[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
    #                 horizontalalignment=horizontalalignment, **kw)

    # plt.legend(labels=labels_count_df.index, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.ylabel('')

    plt.show()


def all_activities_validation(original_dataset, dirname, period, time_step, display=True):
    index = np.arange(int(period.total_seconds() / time_step.total_seconds()))
    validation_df = pd.DataFrame(index=index)

    all_validation_df = pd.DataFrame(columns=['rmse', 'cum_in'])
    labels_rmse = []


    labels = original_dataset.label.unique()

    labels.sort()
    for label in labels:
        intersect_area, den_area, rmse = validation_periodic_time_distribution(label, original_dataset, dirname,
                                                                               period, time_step,
                                                                               display=True)
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


def periodic_time_distribution(data, label, period, display=False):
    """
    Return the probability of the label occurring for each time step id
    :param data:
    :param label:
    :param period:
    :param time_step:
    :param display:
    :return: time_step_df, array of relative ts
    """

    occurrences = data[data.label == label].copy()

    if len(occurrences) == 0:
        return np.asarray([])

    if occurrences.empty:
        print('The label "{}" does not exist in the dataset'.format(label))
        return None
    occurrences['relative_date'] = occurrences.date.apply(lambda x: modulo_datetime(x.to_pydatetime(), period))

    ts_array = occurrences.relative_date.values

    if display:
        sns.kdeplot(ts_array, shade_lowest=False, shade=True, label=label)
        plt.title('Time distribution : {}'.format(label))
        plt.xlabel('Hour of the day')
        plt.ylabel('Density')
        plt.xlim(0, 24)
        plt.legend(loc="upper left")
        plt.show()

    return ts_array


def validation_periodic_time_distribution(label, real_dataset, replications_directory, period, bin_width=5,
                                          display=True):
    """
    Validation of the occurrence time of a specific label
    :param label:
    :param real_dataset:
    :param replications_directory:
    :param period:
    :param time_step:
    :param confidence:
    :param display:
    :return:
    """

    # Original Dataset
    original_ts_array = periodic_time_distribution(real_dataset, label, period, display=False)

    # Simulation replications
    list_files = glob.glob(replications_directory + '*.csv')
    if len(list_files) == 0:
        raise FileNotFoundError("'{}' does not contains csv files".format(replications_directory))

    bin_width *= 60  # Convert in seconsd
    repl_intersections_area = []
    repl_den_area = []
    repl_rmse = []
    for filename in list_files:
        dataset = pick_custom_dataset(filename)
        repl_ts_array = periodic_time_distribution(data=dataset, label=label, period=period,
                                                   display=False)
        intersect_area = histogram_intersection(original_ts_array, repl_ts_array, bin_width)
        den_area = density_intersection_area(original_ts_array, repl_ts_array)
        rmse = mse(original_ts_array, repl_ts_array, bin_width)

        repl_intersections_area.append(intersect_area)
        repl_rmse.append(rmse)
        repl_den_area.append(den_area)

        # bins = np.arange(0, 24.0, 5/60)
        #
        # sns.distplot(repl_ts_array/3600, bins=bins, norm_hist=True, kde=False, label='Simulation')
        # sns.distplot(original_ts_array/3600, bins=bins, norm_hist=True, kde=False, label='Real data')
        # plt.xlabel("Hour of the day")
        # plt.legend()
        # plt.show()

    min_bin = 0
    max_bin = 24 * 3600  # 24hours

    bins = np.arange(0, max_bin, bin_width)

    # bins = np.linspace(min_val, max_val, int(max_val / bin_minutes))

    hist_A, _ = np.histogram(original_ts_array, bins=bins, range=(min_bin, max_bin), density=True)

    hist_B, _ = np.histogram(repl_ts_array, bins=bins, range=(min_bin, max_bin), density=True)
    #
    # plt.plot(hist_A, label='Original data', linestyle="-", color='blue')
    # plt.plot(hist_B, label='Simulated data', linestyle="--", color='red')
    # plt.xticks(np.arange(0,24,1))
    # plt.show()

    intersect_area = np.mean(repl_intersections_area)

    rmse = np.mean(repl_rmse)

    den_area = np.mean(repl_den_area)

    if display:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title('Time distribution : {}'.format(label))
        # ax1.set_title('Time distribution : {}\nIntersect score={:.2f}'.format(label, intersect_area))
        sns.distplot(original_ts_array / 3600, bins=bins / 3600, ax=ax1, label='Real Data', kde=False)
        sns.distplot(repl_ts_array / 3600, bins=bins / 3600, ax=ax1, label='Simulated Data', kde=False)

        sns.kdeplot(original_ts_array / 3600, shade_lowest=False, shade=True, label='Real Data', ax=ax2)
        sns.kdeplot(repl_ts_array / 3600, shade_lowest=False, shade=True, label='Simulated Data', ax=ax2)

        ax1.set_ylabel('Number of occurrences')

        plt.xlabel('Hour of the day')
        plt.ylabel('Density')
        # ax1.set_xlim(0, 24)
        ax2.set_xlim(0, 24)
        ax1.set_xlim(0, 24)
        ax1.legend()
        plt.legend(loc="upper left")
        plt.show()

    return intersect_area, den_area, rmse


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

    return repl_durations


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

        label = "{}_Sim{}_Pattern{} : Mean_RMSE={:.2e}".format(model['label'], model['sim_id'], model['pattern_id'],
                                                               np.mean(rmse))

        # label = "{}\tMean RMSE={:.2e}".format(model['name'], np.mean(rmse))
        # label = model['label']

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


def patterns_validation(original_dataset, original_patterns, replications_directory, period, Tep, display=True):
    list_files = glob.glob(replications_directory + '*.csv')

    results = pd.DataFrame(
        columns=['episode', 'description', 'original_accuracy', 'mean_error', 'lower_error', 'upper_error'])

    if len(list_files) == 0:
        raise FileNotFoundError("'{}' does not contains csv files".format(replications_directory))

    for index, pattern in original_patterns.iterrows():  # Create one Macro/Single Activity per row

        episode = list(pattern['Episode'])
        description = pattern['Description']
        original_accuracy = pattern_accuracy(data=original_dataset, episode=episode, time_description=description,
                                             period=period, Tep=Tep)
        replications_accuracy = []
        for filename in list_files:
            dataset = pick_custom_dataset(filename)
            accuracy = pattern_accuracy(data=dataset, episode=episode, time_description=description,
                                        period=period, Tep=Tep)
            replications_accuracy.append(accuracy)

            evolution_percentage = 100 * (1 + list_files.index(filename)) / len(list_files)
            sys.stdout.write("\r{} %% of replications treated!".format(evolution_percentage))
            sys.stdout.flush()
        sys.stdout.write("\n")

        mean, error = compute_stochastic_error(replications_accuracy)
        min_val = mean - error
        max_val = mean + error

        natural_desc = {}
        for mean_time, std_time in description.items():
            natural_desc[str(dt.timedelta(seconds=mean_time))] = str(dt.timedelta(seconds=std_time))

        results.loc[len(results)] = [str(episode), natural_desc, round(original_accuracy, 3), round(mean, 3),
                                     round(min_val, 3), round(max_val, 3)]

        print([str(episode), natural_desc, round(original_accuracy, 3), round(mean, 3), round(min_val, 3),
               round(max_val, 3)])

        evolution_percentage = 100 * (index + 1) / len(original_patterns)
        print("{} %% of Pattern validated!!".format(round(evolution_percentage, 2)))

    results.to_csv(replications_directory + '/patterns_results_with_mean.csv', index=False, sep=";")

    results['error'] = results['original_accuracy'] - results['mean_error']

    sns.distplot(results['error'])
    plt.show()


def pattern_accuracy(data, episode, time_description, period, Tep):
    """
    Compute the accuracy of the pattern in the given data
    :param data:
    :param episode:
    :param time_description:
    :return: accuracy
    """
    occurrences = find_occurrences(data, episode, Tep)
    accuracy, expected_occurrences = compute_pattern_accuracy(occurrences=occurrences, period=period,
                                                              time_description=time_description)
    return accuracy


def plot_ambulatogram(data, labels, start_date=None, nb_days=1, display=False):
    if not start_date:
        start_date = data.date.min().to_pydatetime()

    end_date = start_date + dt.timedelta(days=nb_days)

    data = data[(data.date >= start_date) & (data.end_date < end_date)].copy()

    labels_df = pd.DataFrame(labels, columns=['label'])
    labels_df.sort_values(by=['label'], ascending=True, inplace=True)
    labels_df['label_id'] = 1 + np.arange(len(labels))

    def add_end_event(row):
        return [row.end_date, row.label]

    end_events = data.apply(add_end_event, axis=1)

    end_data = data[['date', 'label']].copy()
    end_data['date'] = end_events.apply(lambda x: x[0])
    end_data['label'] = end_events.apply(lambda x: x[1])

    data = data[['date', 'label']].append(end_data)

    data.sort_values(['date'], ascending=True, inplace=True)

    data = data.join(labels_df.set_index('label'), on='label')

    if display:
        fig, ax = plt.subplots(1, 1)
        ax.plot(data.date, data.label_id, linestyle="-")

        ax.set_yticks(labels_df.label_id.values)
        ax.set_yticklabels(labels_df.label.values, minor=False, rotation=45)
        plt.show()

    return data[['date', 'label_id']]




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


def sequence_alignement_validation(original_data, directory, period, alphabet, letter_similarity, display=True):
    """
    Compute the alignement between real life dataset and simulation replication event logs
    :param original_data:
    :param replications_directory:
    :param display:
    :return:
    """

    start_date = original_data.date.min().to_pydatetime()
    start_date = start_date - dt.timedelta(seconds=modulo_datetime(start_date, period=dt.timedelta(days=1)))
    end_date = original_data.end_date.max().to_pydatetime()

    repl_scores = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))

    list_files = glob.glob(directory + '*.csv')
    for filename in list_files:
        i = list_files.index(filename)

        dataset = pick_custom_dataset(filename)
        alignement_df = sequence_alignment(original_data=original_data, sim_data=dataset, alphabet=alphabet,
                                           letter_similarity=letter_similarity, period=period, start_date=start_date,
                                           end_date=end_date)

        alignement_df.columns = ["score_rep_{}".format(i)]
        repl_scores = pd.concat([repl_scores, alignement_df], axis=1)

        evolution_percentage = round(100 * (i + 1) / len(list_files), 2)
        sys.stdout.write("\r{} %% of replications evaluated!!".format(evolution_percentage))
        sys.stdout.flush()
    sys.stdout.write("\n")

    # plt.plot(alignement_df.day_date, alignement_df.score,
    #               label="Replication N°{}".format(list_files.index(filename)), linestyle="-")
    #
    # plt.title("Daily Sequence Alignement")
    # plt.xlabel("Days")
    # plt.ylabel("Alignment score")
    # plt.show()

    repl_scores['min_score'] = repl_scores.apply(lambda x: min(x), axis=1)
    repl_scores['max_score'] = repl_scores.apply(lambda x: max(x), axis=1)
    repl_scores['mean_score'] = repl_scores.apply(lambda x: np.mean(x), axis=1)

    repl_scores.drop(repl_scores.tail(1).index, inplace=True)  # drop last row

    if display:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(repl_scores)), repl_scores.mean_score, label="Alignment Mean", linestyle="-")

        ax.fill_between(np.arange(len(repl_scores)), repl_scores.min_score, repl_scores.max_score,
                        label='Alignment Min - Max', color='k', alpha=.2, hatch='//')

        alignment_mean = np.mean(repl_scores.mean_score)
        ax.axhline(alignment_mean, color='red', lw=2, linestyle='dashed')

        plt.draw()

        labels = [round(i, 2) for i in list(np.arange(0, 1, step=0.1)) + [1]]
        locs = list(np.arange(0, 1, step=0.1)) + [1]
        labels += [round(alignment_mean, 2)]
        locs += [alignment_mean]
        ax.set_yticklabels(labels)
        ax.set_yticks(locs)

        plt.xlabel("Days")
        plt.ylabel("Normalized alignment score")
        plt.ylim(0, 1)
        plt.legend(loc=1)
        plt.xticks(np.arange(0, len(repl_scores) + 10, step=10))

        plt.show()

    return None


def sequence_alignment(original_data, sim_data, alphabet, letter_similarity, period, start_date, end_date):
    """
    Compute the alignement score in each period of time
    :param original_data:
    :param simu_data:
    :param label_similarity_matrix:
    :param period:
    :return: a pandas Series with day_date as index and align score as column
    """

    # # SMITH-WATERMAN ALGORITHM
    # scoring_matrix = np.zeros((len(labels)+1, len(labels)+1))

    current_date = start_date

    align_df = pd.DataFrame(columns=['day_date', 'score'])

    while current_date < end_date:

        current_end_date = current_date + period

        seq1 = original_data[
            (original_data.date >= current_date) & (original_data.date < current_end_date)].label.values
        seq2 = sim_data[(sim_data.date >= current_date) & (sim_data.date < current_end_date)].label.values

        if len(seq2) == 0 or len(seq1) == 0:
            score = 0
            align_df.loc[len(align_df)] = [current_date, score]
            current_date = current_end_date
            continue

        X = "".join([alphabet[label] for label in seq1])
        Y = "".join([alphabet[label] for label in seq2])

        # alignments = pairwise2.align.globalds(X, Y, letter_similarity, -0.1, -0.1, one_alignment_only=True)

        alignments = pairwise2.align.globalms(X, Y, 8, -2, -2, -2, one_alignment_only=True)

        score = alignments[0][2]
        # perfect_score = pairwise2.align.globalds(X, X, letter_similarity, -0.1, -0.1, score_only=True)
        perfect_score = pairwise2.align.globalms(X, X, 8, -2, -2, -2, score_only=True)
        null_score = -2 * max(len(Y), len(X))

        ratio_score = (score - null_score) / (perfect_score - null_score)

        # ratio_score = score / perfect_score

        print('Ratio score={:.2f}'.format(ratio_score))
        print(format_alignment(*alignments[0]))

        align_df.loc[len(align_df)] = [current_date, ratio_score]

        current_date = current_end_date

    align_df.set_index(['day_date'], inplace=True)
    # align_df.drop(['day_date'], axis=1, inplace=True)
    return align_df


if __name__ == "__main__":
    main()