import datetime as dt
import math
import pickle
import sys
from random import random

import numpy as np
import pandas as pd
import scipy.stats as st

from Graph_Model import Build_Graph
from Pattern_Discovery.Candidate_Study import find_occurrences, modulo_datetime
from Pattern_Discovery.Pattern_Discovery import pick_dataset


def main():
    # print('###########################')
    # print('# 1 - Find the patterns   #')
    # print('###########################')

    dataset_name = 'aruba'

    period = dt.timedelta(days=1)
    simulation_duration = 30 * period
    freq = dt.timedelta(minutes=10)

    dataset = pick_dataset(dataset_name, nb_days=300)

    # activities, matrix = compute_activity_compatibility_matrix(dataset)
    # Graph_Pattern.Graph.set_compatibility_matrix(activities, matrix)

    output = "./output/{}/ID_0".format(dataset_name)
    patterns = pickle.load(open(output + '/patterns.pickle', 'rb'))

    all_activities_tuples = []
    # for _, pattern in patterns.iterrows():
    #     pattern_activities = list(pattern['Episode'])
    #     all_activities_tuples.append(tuple(pattern_activities))

    # Unique activities

    single_activities = list(dataset.label.unique())
    for activity in single_activities:
        all_activities_tuples.append((activity,))

    # print('#########################################')
    # print('# 2 - Fitting duration distributions    #')
    # print('#########################################')

    law_activities = {}
    print('Starting Fit activities duration distributions')
    for activity in all_activities_tuples:
        episode_occurrences = find_occurrences(dataset, activity, Tep=60)
        if len(episode_occurrences) == 0:
            raise Exception
            continue

        episode_occurrences['activity_duration'] = episode_occurrences.end_date - episode_occurrences.date
        episode_occurrences['activity_duration'] = episode_occurrences['activity_duration'].apply(
            lambda x: x.total_seconds())
        durations = episode_occurrences.activity_duration.values
        dist_name, params = Build_Graph.best_fit_distribution(durations)
        law_activities[activity] = ((dist_name, params))

        evolution_percentage = round(100 * (all_activities_tuples.index(activity) + 1) / len(all_activities_tuples), 2)
        sys.stdout.write("\r{} %% of distribution fitting done".format(evolution_percentage))
        sys.stdout.flush()

    print()

    # print('###########################')
    # print('# 3 - Build Histograms    #')
    # print('###########################')

    # activities_list = activities_list[:2]

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    index = np.arange(int(period.total_seconds() / freq.total_seconds()) + 1)

    all_histograms = {}
    for episode in all_activities_tuples:
        # find episode occurrences
        df = find_occurrences(dataset, tuple(episode), Tep=60)

        df['relative_date'] = df.date.apply(lambda x: modulo_datetime(x.to_pydatetime(), period))

        df['group_id'] = df['relative_date'] / freq.total_seconds()
        df['group_id'] = df['group_id'].apply(math.floor)
        histogram = df.groupby(['group_id']).count()['date']
        histogram = histogram.reindex(index)
        histogram.fillna(0, inplace=True)

        # histogram.plot(kind="bar", label=' - '.join(episode), color=np.random.rand(3,1))

        all_histograms[episode] = histogram

        # (df_prime).plot(kind="bar", label=' - '.join(episode), color=np.random.rand(3, 1))
        # ax.set_facecolor('#eeeeee')
        # ax.set_xlabel("hour of the day")
        # ax.set_ylabel("count")
        # ax.set_title(episode)
    # plt.xlabel("hour of the day")
    # plt.ylabel("count")
    # plt.legend()
    # plt.show()

    # print('###################################')
    # print('# 4 - Start the loop Processus    #')
    # print('###################################')

    # current_sim_date = end_date + period - dt.timedelta(seconds=modulo_datetime(end_date, period))
    current_sim_date = start_date + period - dt.timedelta(seconds=modulo_datetime(start_date, period))
    simulation_result = pd.DataFrame(columns=['date', 'end_date', 'label'])

    # Compute the numbre of periods
    nb_periods = math.floor((end_date - start_date).total_seconds() / period.total_seconds())

    simulation_duration = nb_periods * period

    for replication in range(20):
        time_in_period = 0  # seconds
        simulation_result = pd.DataFrame(columns=['date', 'end_date', 'label'])
        print('###################################')
        print('# Simulation replication N°{}     #'.format(replication + 1))
        print('###################################')

        while time_in_period < simulation_duration.total_seconds():

            # Print evolution
            evolution_percentage = round(100 * (time_in_period / simulation_duration.total_seconds()), 2)
            sys.stdout.write("\r{} %% of Time evolution!!".format(evolution_percentage))
            sys.stdout.flush()

            # Pick the next group index according to time_in_period value
            group_index = math.ceil(time_in_period / freq.total_seconds()) % (max(index))

            # Search for the most likely to happen events
            candidate_episodes = []
            count_candidate_episodes = []
            for episode, histogram in all_histograms.items():

                count = histogram.loc[group_index]
                if count:
                    candidate_episodes.append(episode)
                    count_candidate_episodes.append(count)

            if len(candidate_episodes) == 0:
                time_in_period += freq.total_seconds()
                continue

            count_candidate_episodes = np.asarray(count_candidate_episodes)
            # normalize
            count_candidate_episodes = count_candidate_episodes / nb_periods
            count_candidate_episodes = np.cumsum(count_candidate_episodes)

            # Pick randomly a episode

            # Pick a random number between 0 -- 1
            rand = random()
            choosen_episode = None  # By default we take the first one

            for episode in candidate_episodes:
                if rand <= count_candidate_episodes[candidate_episodes.index(episode)]:
                    choosen_episode = episode
                    break

            if choosen_episode is None:
                time_in_period += freq.total_seconds()
                continue

            # TODO : Call the graph related to the choosen episode

            dist_name, params = law_activities[choosen_episode]

            duration_dist = getattr(st, dist_name)
            duration_arg = params[:-2]
            duration_loc = params[-2]
            duration_scale = params[-1]

            while True:
                generated_duration = -1
                while generated_duration < 0:
                    generated_duration = int(duration_dist.rvs(loc=duration_loc, scale=duration_scale, *duration_arg))

                try:
                    event_start_date = current_sim_date + dt.timedelta(seconds=time_in_period)
                    event_end_date = event_start_date + dt.timedelta(seconds=generated_duration)
                    simulation_result.loc[len(simulation_result)] = [event_start_date, event_end_date,
                                                                     ''.join(choosen_episode)]
                    break
                except ValueError as er:
                    print("OOOps ! Date Overflow. Let's try again...")

            time_in_period += generated_duration
        filename = "output/Simulation results/dataset_simulation_{}.csv".format(replication + 1)
        simulation_result.to_csv(filename, index=False, sep=';')


if __name__ == '__main__':
    main()
