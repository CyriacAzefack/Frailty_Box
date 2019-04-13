import sys

import math
import matplotlib.pyplot as plt
import seaborn as sns
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
    dataset = pick_dataset(dataset_name, nb_days=60)
    support_min = 10
    tep = 30
    period = dt.timedelta(days=1)

    # macro_activities = extract_macro_activities(dataset=dataset, support_min=support_min, tep=tep, period=period)

    time_window_duration = dt.timedelta(days=30)

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime() - time_window_duration

    nb_tw = math.ceil((end_date - start_date) / period)

    window_start_date = start_date

    new_episodes_evol = []

    known_episodes = []

    i = 0
    while window_start_date < end_date:
        i += 1
        window_end_date = window_start_date + time_window_duration
        tw_dataset = dataset.loc[(dataset.date >= window_start_date) & (dataset.date < window_end_date)].copy()

        macro_activities = extract_macro_activities(dataset=tw_dataset, support_min=support_min, tep=tep, period=period)

        episodes = list(macro_activities.keys())

        episodes = [frozenset(e) for e in episodes]
        old_episodes = list(set(episodes) & set(known_episodes))
        new_episodes = set(episodes) - set(old_episodes)
        nb_new_episodes = len(new_episodes)

        for episode in new_episodes:
            known_episodes.append(episode)

        new_episodes_evol.append(nb_new_episodes)

        window_start_date += period

        evolution_percentage = round(100 * (i) / nb_tw, 2)
        sys.stdout.write("\r{} %% of Time Windows treated!!".format(evolution_percentage))
        sys.stdout.flush()

    sys.stdout.write("\n")

    plt.plot(np.arange(len(new_episodes_evol)), new_episodes_evol)
    plt.title('Number of macro discovered')
    plt.xlabel('Time Windows')
    plt.ylabel('nb episodes')
    plt.show()


def extract_macro_activities(dataset, support_min, tep, period, display=False):
    """
    Extract the tupe (episode, occurrences) from the input dataset
    :param dataset: input dataset
    :param support_min: Minimum number of episode occurrences
    :param tep: Duration max of an episode occurrence
    :param period: periodicity
    :return: A dict-like object with 'episode' as key and 'episode occurrence' df as value
    """

    macro_activities = {}

    i = 0
    while len(dataset) > 0:
        i += 1
        episode, nb_occ_ratio, density_area = find_best_episode(dataset=dataset, tep=tep, support_min=support_min,
                                                                period=period, display=display)

        if episode is None:
            break

        episode_occurrences = compute_episode_occurrences(dataset=dataset, episode=episode, tep=tep)

        macro_activities[tuple(episode)] = episode_occurrences

        # dataset = pd.merge(dataset, episode_occurrences, how='inner', on=['date', 'end_date', 'label'])

        # if display:
        print("########################################")
        print("Run N°{}".format(i))
        print("Best episode found {}.".format(episode))
        print("Nb Occurrences : \t{}".format(int(nb_occ_ratio * len(dataset))))
        print("Density area in a day : \t{:.2f}".format(density_area))

        dataset = pd.concat([dataset, episode_occurrences]).drop_duplicates(keep=False)

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
    comparaison_df = pd.DataFrame(columns=['episode', 'nb_occ', 'score'])

    # Most frequents episode
    frequent_episodes = FP_growth.find_frequent_episodes(dataset, support_min, tep)

    if len(frequent_episodes) == 0:
        return None, None, None

    # Compute the sparsity of the episode occurrence time
    for episode, nb_occ in frequent_episodes.items():
        nb_occ *= len(episode)

        nb_occ /= len(dataset)  # ratio in the dataset

        # find the episode occurrences
        occurrences = Candidate_Study.find_occurrences(dataset, episode, tep)

        # Relative timestamp in the period
        occurrences["relative_date"] = occurrences.date.apply(
            lambda x: Candidate_Study.modulo_datetime(x.to_pydatetime(), period))

        data_points = occurrences.relative_date.values
        data_points = np.asarray(data_points).reshape(-1, 1)

        kde_a = KDEUnivariate(data_points)
        kde_a.fit(bw="scott")

        day_bins = np.linspace(0, 24 * 3600, 10000)

        density_values = kde_a.evaluate(day_bins)

        score = np.sum(density_values)  # Area under the curve for 24 hours

        comparaison_df.loc[len(comparaison_df)] = [list(episode), nb_occ, score]

    scores = comparaison_df[comparaison_df.columns[1:]].values
    scores = np.asarray(scores)

    # Compute the pareto front
    pareto = identify_pareto(scores)
    pareto_front_df = comparaison_df.loc[pareto]

    # Find the best point on the pareto front: where x & y are the closest
    max_distance = 0
    best_point = None

    for i, row in pareto_front_df.iterrows():
        nb_occ = row['nb_occ']
        score = row['score']
        dist = len(row['episode']) / abs(nb_occ - score)
        # dist = math.pow(point[0], point[1])

        if dist > max_distance:
            max_distance = dist
            best_point = row

    if display:
        plt.scatter(comparaison_df.nb_occ, comparaison_df.score)
        plt.plot(pareto_front_df.nb_occ, pareto_front_df.score, color='r')
        plt.plot([best_point['nb_occ']], [best_point['score']], marker='o', markersize=8, color="red")

        #
        for _, row in pareto_front_df.iterrows():
            plt.text(row['nb_occ'] + 0.01, row['score'], row['episode'], horizontalalignment='left', size='medium',
                     color='black', weight='semibold')

        plt.xlabel('Nb occurrences')
        plt.ylabel('Accuracy')
        plt.show()

    return best_point['episode'], best_point['nb_occ'], best_point['score']


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
        return data

    episode_occurrences = pd.DataFrame(columns=["date", "end_date", "label"])
    time_occurrences = Candidate_Study.find_occurrences(data, episode, tep)

    for index, occurrence in time_occurrences.iterrows():
        start_date = occurrence["date"]
        end_date = start_date + dt.timedelta(minutes=tep)
        mini_data = data.loc[(data.date >= start_date) & (data.date < end_date)].copy()

        mini_data.drop_duplicates(["label"], keep='first', inplace=True)
        episode_occurrences = episode_occurrences.append(mini_data, ignore_index=True)

    episode_occurrences.sort_values(["date"], ascending=True, inplace=True)

    return episode_occurrences


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
