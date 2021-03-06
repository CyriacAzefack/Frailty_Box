import errno
import math
import multiprocessing as mp
import time as t

import seaborn as sns
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture
from statsmodels.nonparametric.kde import KDEUnivariate

from Pattern_Mining import FP_growth, Candidate_Study
from Utils import *

sns.set_style('darkgrid')


def main():
    """
    Check the distribution of frequency vs periodicity
    :return:
    """

    dataset_name = 'bped_ramon'
    output_folder = '../output/{}/'.format(dataset_name)
    dataset = pick_dataset(dataset_name, nb_days=-1)

    # TIME WINDOW PARAMETERS
    nb_days_per_window = 30
    time_window_duration = dt.timedelta(days=nb_days_per_window)
    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime() - time_window_duration

    # SIM_MODEL PARAMETERS
    period = dt.timedelta(days=1)
    tep = 30
    support_min = 10
    # support_min = 3

    nb_processes = 2 * mp.cpu_count()  # For parallel computing

    if not os.path.exists(os.path.dirname(output_folder)):
        try:
            os.makedirs(os.path.dirname(output_folder))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    start_time = t.time()
    print('##########################################')
    print('## MACRO ACTIVITIES EXTRACTION : {} ##'.format(dataset_name.upper()))
    print('##########################################')

    macro_activities = extract_macro_activities(dataset=dataset, support_min=support_min, tep=tep,
                                                display=True, verbose=True)

    # results_df.to_csv(f'../output/{dataset_name}/habits_results.csv', index=False)
    # monitoring_start_time = t.time()

    # nb_tw = math.floor((end_date - start_date) / period)  # Number of time windows available
    #
    # results = {}
    #
    # args = [(dataset, support_min, tep, period, time_window_duration, tw_id) for tw_id in range(nb_tw)]
    #
    # with mp.Pool(processes=nb_processes) as pool:
    #     mp_results = pool.starmap(extract_tw_macro_activities, args)
    #
    #     for result in mp_results:
    #         results[result[0]] = result[1]
    #
    # print("All Time Windows Treated")
    #
    # elapsed_time = dt.timedelta(seconds=round(t.time() - start_time, 1))
    #
    # print("###############################")
    # print("Elapsed time : {}".format(elapsed_time))
    #
    # pickle_out = open(output_folder + 'all_macro_activities', 'wb')
    # pickle.dump(results, pickle_out)
    # pickle_out.close()
    #
    # results = pickle.load(open(output_folder+'all_macro_activities', 'rb'))
    #
    # print(results)

    # plt.plot(np.arange(len(new_episodes_evol)), new_episodes_evol)
    # plt.title('Number of macro discovered')
    # plt.xlabel('Time Windows')
    # plt.ylabel('nb episodes')
    # plt.show()


def extract_tw_macro_activities(dataset, support_min, tep, period, window_duration, tw_id):
    """
    Extract macro activities from a time window
    :param dataset:
    :param support_min:
    :param tep:
    :param period:
    :param window_duration:
    :param tw_id:
    :return: tw_id (id of the time window), macro_activities_episode (A dict-like object with 'episode' as key and the
    tuple of dataframes (episode_occurrences, events) as value)
    """

    print('Time window {} Macro-Activities screening started...'.format(tw_id))
    start_date = dataset.date.min().to_pydatetime()

    window_start_date = start_date + tw_id * period
    window_end_date = window_start_date + window_duration

    tw_dataset = dataset.loc[(dataset.date >= window_start_date) & (dataset.date < window_end_date)].copy()

    tw_macro_activities_episodes = extract_macro_activities(dataset=tw_dataset, support_min=support_min, tep=tep,
                                                            display=False, verbose=False)

    print('Time window {} screening finished.'.format(tw_id))

    return tw_id, tw_macro_activities_episodes

    # results[tw_id] = tw_macro_activities
    #
    #


def extract_macro_activities(dataset, support_min, tep, verbose=False, display=False, singles_only=False):
    """
    :param display:
    :param dataset:
    :param support_min:
    :param tep:
    :param period:
    :param verbose:
    :param singles_only:
    :return:
    """

    macro_activities = {}

    results_df = pd.DataFrame(columns=['episode', 'description', 'nb_occ', 'accuracy', 'start_validity',
                                       'end_validity', 'nb_days_val'])

    i = 0
    while len(dataset) > 0 and not singles_only:
        i += 1
        # episode, nb_occ, ratio, score = find_best_episode(log_dataset=log_dataset, tep=tep, support_min=support_min,
        #                                                   period=period, display=display)
        # Most frequents episode

        best_episode, nb_occ, _, accuracy, description, start_val, end_val = find_best_episode(dataset=dataset, tep=tep,
                                                                                               support_min=support_min,
                                                                                               display=display,
                                                                                               verbose=verbose)

        if best_episode is None:
            print("**********No more best episode found*************")
            break

        episode_occurrences, events = compute_episode_occurrences(dataset=dataset, episode=best_episode, tep=tep)

        macro_activities[tuple(best_episode)] = (episode_occurrences, events)

        nb_days_val = int((end_val - start_val) / dt.timedelta(days=1))
        results_df.loc[len(results_df)] = [list(best_episode), description, nb_occ, accuracy, start_val, end_val,
                                           nb_days_val]
        if verbose:
            print("########################################")
            print("Run N°{}".format(i))
            print("Best episode found {}.".format(best_episode))
            print(f"Nb Occurrences : \t{nb_occ}")
            print(f"Accuracy : \t{accuracy:.2f}")
            print(f"Description: \t{description}")
            print(f"Validity Period : {str(start_val)} -- {str(end_val)}")
            print(f"Number Days of validity : {nb_days_val}")


        dataset = pd.concat([dataset, events]).drop_duplicates(keep=False)

    # Mining of the rest of the dataset by creating single-activities
    labels = dataset.label.unique()

    for label in labels:
        events = dataset[dataset.label == label].copy()
        episode_occurrences = events[['date', 'end_date']].copy()

        macro_activities[(label,)] = (episode_occurrences, events)

        if verbose:
            print("########################################")
            # print("Run N°{}".format(i))
            print("Best episode found {}.".format((label,)))
            print("Nb Occurrences : \t{}".format(len(events)))


    return macro_activities


def find_best_episode(dataset, tep, support_min, display=False, verbose=False):
    """
    Find the best episode in a event log
    :param verbose:
    :param dataset: Event log
    :param tep: duration max of an episode
    :param support_min: number of occurrences min in the event log
    :param period: periodicity of the analysis
    :param display : display the solutions
    :return: the best episode found
    """

    # Dataset to store the objective values of the solutions
    comparaison_df = pd.DataFrame(
        columns=['episode', 'nb_occ', 'nb_events', 'ratio_dataset', 'accuracy', 'power', 'start_val', 'end_val'])

    # Most frequents episode
    frequent_episodes = FP_growth.find_frequent_episodes(dataset, support_min, tep)

    if len(frequent_episodes) == 0:
        print("No frequent episodes found")
        return None, None, None, None, None, None, None

    GMM_descriptions = {}
    # Compute the sparsity of the episode occurrence time
    for episode, _ in frequent_episodes.items():
        episode = sorted(episode, reverse=True)

        periodicity = Candidate_Study.periodicity_search(data=dataset, episode=episode, delta_Tmax_ratio=3,
                                                         support_min=support_min, std_max=0.1, tolerance_ratio=2,
                                                         Tep=tep, display=False, verbose=False)

        if periodicity is not None:
            nb_occ = periodicity['nb_occ']
            ratio_dataset = len(episode) * nb_occ / len(dataset)  # ratio in the log_dataset
            comparaison_df.loc[len(comparaison_df)] = [list(episode), nb_occ, len(episode) * nb_occ, ratio_dataset,
                                                       periodicity['accuracy'], periodicity['compression_power'],
                                                       periodicity['delta_t'][0], periodicity['delta_t'][1]]
            GMM_descriptions[tuple(episode)] = periodicity['description']

    if len(comparaison_df) == 0:
        print("*****No periodicities Found******")
        return None, None, None, None, None, None, None

    scores = comparaison_df[["nb_events", "accuracy"]].values
    scores = np.asarray(scores)

    # Compute the pareto front
    pareto = identify_pareto(scores)
    pareto_front_df = comparaison_df.loc[pareto]

    pareto_front_df.sort_values(['ratio_dataset', 'accuracy'], ascending=False, inplace=True)

    ##############################################
    #       MOST INTERESTING EPISODE ??          #
    ##############################################

    max_distance = 0
    best_point = None

    for i, row in pareto_front_df.iterrows():
        # ratio_dataset = row['ratio_dataset']
        # score = row['score']
        dist = row['power']
        # dist = math.pow(point[0], point[1])

        if dist > max_distance:
            max_distance = dist
            best_point = row

    if display:
        sns.scatterplot(x='nb_events', y='accuracy', size='nb_occ', data=comparaison_df)
        plt.plot(pareto_front_df.nb_events, pareto_front_df.accuracy, color='r', label='Pareto Front')
        plt.plot([best_point['nb_events']], [best_point['accuracy']], marker='x', color="red")

        #
        for _, row in pareto_front_df.iterrows():
            plt.text(row['nb_events'], row['accuracy'], row['episode'], verticalalignment='top',
                     size='large', color='black', weight='semibold')

        plt.legend(loc=2)
        # plt.ylim((0,1))
        plt.xlabel('Nombre d\'évènements', fontsize=18)
        plt.ylabel('Précision de la périodicité', fontsize=18)
        plt.title('Selection du meilleur épisode', fontsize=20)
        plt.show()

    return best_point['episode'], best_point['nb_occ'], best_point['ratio_dataset'], best_point['accuracy'], \
           GMM_descriptions[tuple(best_point['episode'])], best_point['start_val'], best_point['end_val']


def compute_episode_description(dataset, episode, period, tep):
    """
    Compute the frequency description of the episode
    :param dataset:
    :param episode:
    :param tep:
    :return: a dict-like object {[mu] : sigma}
    """

    # find the episode occurrences
    occurrences = Candidate_Study.find_occurrences(dataset, episode, tep)

    # Relative timestamp in the period
    occurrences["relative_date"] = occurrences.date.apply(
        lambda x: Candidate_Study.modulo_datetime(x.to_pydatetime(), period))

    data_points = occurrences.relative_date.values

    # # For midnight-morning issue
    # data_points_2 = [x + period.total_seconds() for x in data_points]
    #
    # big_data_points = np.asarray(list(data_points) + list(data_points_2)).reshape(-1, 1)

    # Find the number of clusters
    kde_a = KDEUnivariate(data_points)
    kde_a.fit(bw="normal_reference")

    day_bins = np.linspace(0, period.total_seconds(), 1000)
    density_values = kde_a.evaluate(day_bins)

    mi, ma = argrelextrema(density_values, np.less)[0], argrelextrema(density_values, np.greater)[0]

    nb_clusters = len(day_bins[ma])
    #
    # Fit Gaussian Mixture Model
    GMM = GaussianMixture(n_components=nb_clusters, n_init=10).fit(data_points.reshape(-1, 1))

    # Compute the description
    GMM_descr = {}
    for i in range(len(GMM.means_)):
        mu = int(GMM.means_[i][0]) % period.total_seconds()
        sigma = int(math.ceil(np.sqrt(GMM.covariances_[i])))

        GMM_descr[mu] = sigma

    return GMM_descr


def compute_episode_occurrences(dataset, episode, tep):
    """
    Compute the episode occurrences in the log_dataset
    :param dataset:
    :param episode:
    :param tep:
    :return:
    """

    data = dataset[dataset.label.isin(episode)].copy()

    if len(episode) == 1:
        return data, data

    events = pd.DataFrame(columns=["date", "end_date", "label"])
    episode_occurrences = Candidate_Study.find_occurrences(data, episode, tep)

    for index, occurrence in episode_occurrences.iterrows():
        start_date = occurrence["date"]
        end_date = start_date + dt.timedelta(minutes=tep)
        mini_data = data.loc[(data.date >= start_date) & (data.date < end_date)].copy()

        mini_data.drop_duplicates(["label"], keep='first', inplace=True)
        events = events.append(mini_data, ignore_index=True)

    events.sort_values(["date"], ascending=True, inplace=True)

    return episode_occurrences, events


def identify_pareto(scores):
    """
    identify the pareto front from data
    :param scores: 2D numpy array representing solutions
    :return: the list of period_ts_index for non-dominated solutions
    """
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy period_ts_index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Pareto front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' point is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


if __name__ == "__main__":
    main()
