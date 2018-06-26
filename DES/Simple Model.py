import datetime as dt
import errno
import math
import os
import sys
from random import random

import numpy as np
import pandas as pd

from DES import Activity
from Pattern_Discovery.Candidate_Study import find_occurrences, modulo_datetime
from Pattern_Discovery.Pattern_Discovery import pick_dataset


def main():
    # print('###########################')
    # print('# 1 - Find the patterns   #')
    # print('###########################')

    dataset_name = 'aruba'

    period = dt.timedelta(days=7)
    freq = dt.timedelta(minutes=5)
    nb_replications = 20

    dataset = pick_dataset(dataset_name, nb_days=300)

    # activities, matrix = compute_activity_compatibility_matrix(dataset)
    # Graph_Pattern.Graph.set_compatibility_matrix(activities, matrix)

    output = "../output/{}/Simple Model Simulation results 1week 5mn/".format(dataset_name)

    all_activities = []

    # Unique activities

    single_activities = list(dataset.label.unique())

    # single_activities = single_activities[:2]

    for activity in single_activities:
        occurrences = find_occurrences(data=dataset, episode=(activity,))
        activity = Activity.Activity(activity=activity, occurrences=occurrences, period=period, time_step=freq)
        all_activities.append(activity)

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    index = np.arange(int(period.total_seconds() / freq.total_seconds()) + 1)

    # print('###################################')
    # print('# 4 - Start the loop Processus    #')
    # print('###################################')

    # current_sim_date = end_date + period - dt.timedelta(seconds=modulo_datetime(end_date, period))
    current_sim_date = start_date + period - dt.timedelta(seconds=modulo_datetime(start_date, period))
    simulation_result = pd.DataFrame(columns=['date', 'end_date', 'label'])

    # Compute the numbre of periods
    nb_periods = math.floor((end_date - start_date).total_seconds() / period.total_seconds())

    simulation_duration = nb_periods * period

    for replication in range(nb_replications):
        time_in_period = 0  # seconds
        simulation_result = pd.DataFrame(columns=['date', 'end_date', 'label'])
        print('###################################')
        print('# Simulation replication N°{}     #'.format(replication + 1))
        print('###################################')

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

            # stats = chosen_activity.get_stats_from_date(date=current_date, time_step_id=time_step_id, hist=False)
            stats = chosen_activity.get_stats(time_step_id=time_step_id)
            mean_duration = stats['mean_duration']
            std_duration = stats['std_duration']

            while True:
                generated_duration = -1
                while generated_duration < 0:
                    generated_duration = np.random.normal(mean_duration, std_duration)

                try:
                    event_start_date = current_sim_date + dt.timedelta(seconds=time_in_period)
                    event_end_date = event_start_date + dt.timedelta(seconds=generated_duration)
                    simulation_result.loc[len(simulation_result)] = [event_start_date, event_end_date,
                                                                     ''.join(chosen_activity.label)]
                    break
                except ValueError as er:
                    print("OOOps ! Date Overflow. Let's try again...")

            time_in_period += generated_duration
        sys.stdout.write("\n")
        filename = output + "dataset_simulation_rep_{}.csv".format(replication + 1)

        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        simulation_result.to_csv(filename, index=False, sep=';')


if __name__ == '__main__':
    main()
