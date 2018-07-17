import datetime as dt
import errno
import math
import os
import sys
import time as t
from random import random

import numpy as np
import pandas as pd

from DES import Activity, MacroActivity
from xED.Candidate_Study import find_occurrences, modulo_datetime
from xED.Pattern_Discovery import pick_dataset


def main():
    ####################
    # SIMPLE MODEL
    #
    # Characteristics :
    #   - Single Activity Implementation
    #   - No time evolution
    #   - Training Dataset : the Whole Original dataset
    #   - Test Dataset : The Whole Original dataset

    dataset_name = 'KA'

    period = dt.timedelta(days=1)
    freq = dt.timedelta(minutes=5)
    Tep = 30
    nb_replications = 20

    dataset = pick_dataset(dataset_name)

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    # Compute the numbre of periods
    nb_periods = math.floor((end_date - start_date).total_seconds() / period.total_seconds())

    all_activities = []

    output = "../output/{}/Macro Activities Simulation results 5mn/".format(dataset_name)

    train_dataset = dataset.copy()

    input = "../output/{}/ID_{}/patterns.pickle".format(dataset_name, 0)

    patterns = pd.read_pickle(input)

    for index, pattern in patterns.iterrows():
        episode = list(pattern['Episode'])

        occurrences = find_occurrences(data=train_dataset, episode=episode)

        if occurrences.empty:
            continue

        if len(episode) > 1:
            activity = MacroActivity.MacroActivity(episode=episode, dataset=train_dataset, occurrences=occurrences,
                                                   period=period, time_step=freq)
        else:
            activity = Activity.Activity(label=episode[0], occurrences=occurrences, period=period, time_step=freq)

        all_activities.append(activity)

        # Find the events corresponding to the expected occurrences
        mini_factorised_events = pd.DataFrame(columns=["date", "label"])
        for index, occurrence in occurrences.iterrows():
            occ_start_date = occurrence["date"]
            end_date = occ_start_date + dt.timedelta(minutes=Tep)
            mini_data = train_dataset.loc[(train_dataset.label.isin(episode))
                                          & (train_dataset.date >= occ_start_date)
                                          & (train_dataset.date < end_date)].copy()
            mini_data.sort_values(["date"], ascending=True, inplace=True)
            mini_data.drop_duplicates(["label"], keep='first', inplace=True)
            mini_factorised_events = mini_factorised_events.append(mini_data, ignore_index=True)

        train_dataset = pd.concat([train_dataset, mini_factorised_events], sort=False).drop_duplicates(keep=False)

    # Mining of Single activities

    single_episodes = list(train_dataset.label.unique())

    for activity in single_episodes:
        activity = (activity,)
        occurrences = find_occurrences(data=train_dataset, episode=activity)
        activity = Activity.Activity(label=activity, occurrences=occurrences, period=period, time_step=freq)
        all_activities.append(activity)

    print('All Activities Created !!')

    index = np.arange(int(period.total_seconds() / freq.total_seconds()) + 1)

    # print('###################################')
    # print('# 4 - Start the loop Processus    #')
    # print('###################################')

    # current_sim_date = end_date + period - dt.timedelta(seconds=modulo_datetime(end_date, period))
    current_sim_date = start_date + period - dt.timedelta(seconds=modulo_datetime(start_date, period))


    simulation_duration = nb_periods * period

    for replication in range(nb_replications):
        time_in_period = 0  # seconds
        simulation_result = pd.DataFrame(columns=['date', 'end_date', 'label'])
        print('###################################')
        print('# Simulation replication N°{}     #'.format(replication + 1))
        print('###################################')

        time_start = t.process_time()
        while time_in_period < simulation_duration.total_seconds():

            # Print evolution
            evolution_percentage = round(100 * (time_in_period / simulation_duration.total_seconds()), 2)
            sys.stdout.write("\r{} %% of Simulation done!!".format(evolution_percentage))
            sys.stdout.flush()

            # Pick the next group index according to time_in_period value
            time_step_id = math.ceil(time_in_period / freq.total_seconds()) % (max(index))

            # Search for the most likely to happen events
            candidate_activities = []
            count_candidate_activities = []

            current_date = current_sim_date + dt.timedelta(seconds=time_in_period)
            for activity in all_activities:
                # stats = activity.get_stats_from_date(date=current_date, time_step_id=time_step_id)
                stats = activity.get_stats(time_step_id=time_step_id)
                count = stats['hist_count']
                if count > 0:
                    candidate_activities.append(activity)
                    count_candidate_activities.append(count)

            if len(candidate_activities) == 0:  # No candidates found
                time_in_period += freq.total_seconds()
                continue

            count_candidate_activities = np.asarray(count_candidate_activities)
            # normalize
            # TODO : Allow Nothing to happen
            count_candidate_activities = count_candidate_activities / sum(
                count_candidate_activities)  # Not allowing 'Nothing' to happen
            # count_candidate_activities = count_candidate_activities / nb_periods # Allowing Nothing to happen

            count_candidate_activities = np.cumsum(count_candidate_activities)

            # Pick randomly an activity
            # Pick a random number between 0 -- 1
            rand = random()
            chosen_activity = None

            for activity in candidate_activities:
                if rand <= count_candidate_activities[candidate_activities.index(activity)]:
                    chosen_activity = activity
                    break

            if chosen_activity is None:
                time_in_period += freq.total_seconds()
                continue

            # Generate the activity
            activity_simulation, activity_duration = chosen_activity.simulate(current_date, time_step_id)

            simulation_result = pd.concat([simulation_result, activity_simulation], sort=True).drop_duplicates(
                keep='first')
            time_in_period += activity_duration

        sys.stdout.write("\n")
        filename = output + "dataset_simulation_rep_{}.csv".format(replication + 1)

        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        simulation_result.to_csv(filename, index=False, sep=';')
        elapsed_time = dt.timedelta(seconds=round(t.process_time() - time_start, 1))
        print("Time elapsed for the simulation : {}".format(elapsed_time))

if __name__ == '__main__':
    main()
