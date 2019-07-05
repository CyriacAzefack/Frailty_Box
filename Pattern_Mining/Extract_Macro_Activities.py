import errno
import multiprocessing as mp
import pickle
import time as t

import math
import matplotlib.pyplot as plt
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

    dataset_name = 'hh101'
    output_folder = '../output/{}/'.format(dataset_name)
    dataset = pick_dataset(dataset_name)

    # TIME WINDOW PARAMETERS
    nb_days_per_window = 30
    time_window_duration = dt.timedelta(days=nb_days_per_window)
    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime() - time_window_duration

    # SIM_MODEL PARAMETERS
    period = dt.timedelta(days=1)
    tep = 30
    support_min = nb_days_per_window




    nb_processes = 2 * mp.cpu_count()  # For parallel computing

    if not os.path.exists(os.path.dirname(output_folder)):
        try:
            os.makedirs(os.path.dirname(output_folder))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # tw_macro_activities = extract_macro_activities(dataset=dataset, support_min=support_min, tep=tep, period=period,
    #                                             verbose=True, display=False)
    # elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))
    #
    # print("###############################")
    # print("Elapsed time : {}".format(elapsed_time))

    print('##########################################')
    print('## MACRO ACTIVITIES EXTRACTION : {} ##'.format(dataset_name.upper()))
    print('##########################################')

    monitoring_start_time = t.time()


    nb_tw = math.floor((end_date - start_date) / period)  # Number of time windows available

    results = {}

    args = [(dataset, support_min, tep, period, time_window_duration, tw_id) for tw_id in range(nb_tw)]


    with mp.Pool(processes=nb_processes) as pool:
        mp_results = pool.starmap(extract_tw_macro_activities, args)

        for result in mp_results:
            results[result[0]] = result[1]


    print("All Time Windows Treated")

    elapsed_time = dt.timedelta(seconds=round(t.time() - monitoring_start_time, 1))

    print("###############################")
    print("Elapsed time : {}".format(elapsed_time))

    pickle_out = open(output_folder + 'all_macro_activities', 'wb')
    pickle.dump(results, pickle_out)
    pickle_out.close()

    # results = pickle.load(open(output_folder+'all_macro_activities', 'rb'))

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

    print('Time window {} started'.format(tw_id))
    start_date = dataset.date.min().to_pydatetime()

    window_start_date = start_date + tw_id * period
    window_end_date = window_start_date + window_duration

    tw_dataset = dataset.loc[(dataset.date >= window_start_date) & (dataset.date < window_end_date)].copy()

    tw_macro_activities_episodes = extract_macro_activities(dataset=tw_dataset, support_min=support_min, tep=tep,
                                                            period=period)

    print('Time window {} finished'.format(tw_id))

    return tw_id, tw_macro_activities_episodes

    # results[tw_id] = tw_macro_activities
    #
    #


def extract_macro_activities(dataset, support_min, tep, period, verbose=False):
    """
    Extract the tupe (episode, occurrences) from the input dataset
    :param dataset: input dataset
    :param support_min: Minimum number of episode occurrences
    :param tep: Duration max of an episode occurrence
    :param period: periodicity
    :return: A dict-like object with 'episode' as key and the tuple of dataframes (episode_occurrences, events) as value
    """

    macro_activities = {}

    i = 0
    while len(dataset) > 0:
        i += 1
        # episode, nb_occ, ratio, score = find_best_episode(dataset=dataset, tep=tep, support_min=support_min,
        #                                                   period=period, display=display)
        # Most frequents episode
        frequent_episodes = FP_growth.find_frequent_episodes(dataset, support_min, tep)

        if len(frequent_episodes) == 0:  # No more frequent episode present
            break

        # ordered_frequent_episodes = [(k, frequent_episodes[k]) for k in sorted(frequent_episodes,
        #                                                                        key=frequent_episodes.get, reverse=True)]

        # TODO : Set a GOOD episode selection policy
        ordered_frequent_episodes = sorted(frequent_episodes.items(), key=lambda t: (len(t[0]) - 1) * t[1],
                                           reverse=True)

        best_episode, nb_occ = ordered_frequent_episodes[0]

        episode_occurrences, events = compute_episode_occurrences(dataset=dataset, episode=best_episode, tep=tep)

        macro_activities[tuple(best_episode)] = (episode_occurrences, events)

        GMM_desc = compute_episode_description(dataset=dataset, episode=best_episode, period=period, tep=tep)

        if verbose:
            print("########################################")
            print("Run N°{}".format(i))
            print("Best episode found {}.".format(best_episode))
            print("Nb Occurrences : \t{}".format(nb_occ))
            # print("Ratio Dataset : \t{}".format(ratio))
            # print("Accuracy score : \t{:.2f}".format(score))
            for mu, sigma in GMM_desc.items():
                print('Mean : {} - Sigma : {}'.format(dt.timedelta(seconds=int(mu)), dt.timedelta(seconds=int(sigma))))

        dataset = pd.concat([dataset, events]).drop_duplicates(keep=False)

    return macro_activities


def find_best_episode(dataset, tep, support_min, period=dt.timedelta(days=1), display=False):
    """
    Find the best episode in a event log
    :param dataset: Event log
    :param tep: duration max of an episode
    :param support_min: number of occurrences min in the event log
    :param period: periodicity of the analysis
    :param display : display the solutions
    :return: the best episode found
    """

    # Dataset to store the objective values of the solutions
    comparaison_df = pd.DataFrame(columns=['episode', 'nb_occ', 'ratio_dataset', 'score'])

    # Most frequents episode
    frequent_episodes = FP_growth.find_frequent_episodes(dataset, support_min, tep)

    if len(frequent_episodes) == 0:
        return None, None, None, None

    # Compute the sparsity of the episode occurrence time
    for episode, nb_occ in frequent_episodes.items():
        ratio_dataset = len(episode) * nb_occ / len(dataset)  # ratio in the dataset

        # find the episode occurrences
        occurrences = Candidate_Study.find_occurrences(dataset, episode, tep)

        # Relative timestamp in the period
        occurrences["relative_date"] = occurrences.date.apply(
            lambda x: Candidate_Study.modulo_datetime(x.to_pydatetime(), period))

        data_points = occurrences.relative_date.values

        # For midnight-morning issue
        data_points_2 = [x + period.total_seconds() for x in data_points]

        big_data_points = np.asarray(list(data_points) + list(data_points_2)).reshape(-1, 1)

        # Find the number of clusters
        kde_a = KDEUnivariate(big_data_points)
        kde_a.fit(bw="normal_reference")

        day_bins = np.linspace(0, 2 * period.total_seconds(), 2000)
        density_values = kde_a.evaluate(day_bins)

        score = np.max(density_values) * period.total_seconds()

        comparaison_df.loc[len(comparaison_df)] = [list(episode), nb_occ, ratio_dataset, score]

    scores = comparaison_df[["ratio_dataset", "score"]].values
    scores = np.asarray(scores)

    # Compute the pareto front
    pareto = identify_pareto(scores)
    pareto_front_df = comparaison_df.loc[pareto]

    pareto_front_df.sort_values(['ratio_dataset'], ascending=True, inplace=True)

    ##############################################
    #       MOST INTERESTING EPISODE ??          #
    ##############################################

    max_distance = 0
    best_point = None

    for i, row in pareto_front_df.iterrows():
        # ratio_dataset = row['ratio_dataset']
        # score = row['score']
        dist = 2000 * len(row['episode']) + row['nb_occ']
        # dist = math.pow(point[0], point[1])

        if dist > max_distance:
            max_distance = dist
            best_point = row

    if display:
        sns.scatterplot(x='ratio_dataset', y='score', data=comparaison_df)
        plt.plot(pareto_front_df.nb_occ, pareto_front_df.score, color='r')
        plt.plot([best_point['ratio_dataset']], [best_point['score']], marker='o', markersize=8, color="red")

        #
        for _, row in pareto_front_df.iterrows():
            plt.text(row['ratio_dataset'] + 0.01, row['score'], row['episode'], horizontalalignment='left',
                     size='medium',
                     color='black', weight='semibold')

        plt.legend()
        plt.xlabel('Nb occurrences')
        plt.ylabel('Accuracy Score')
        plt.show()

    return best_point['episode'], best_point['nb_occ'], best_point['ratio_dataset'], best_point['score']


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
    Compute the episode occurrences in the dataset
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
    :return: the list of index for non-dominated solutions
    """
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
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
