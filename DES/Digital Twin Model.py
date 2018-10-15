import datetime as dt
import errno
import math
import os
import pickle
import sys
import time as t
from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DES import Activity
from xED.Candidate_Study import find_occurrences, modulo_datetime
from xED.Pattern_Discovery import pick_dataset


def main():
    ####################
    # Macro Activities MODEL
    #
    # Characteristics :
    #   - Single Activity Implementation
    #   - No time evolution
    #   - Training Dataset : the Whole Original dataset
    #   - Test Dataset : The Whole Original dataset

    dataset_name = 'KA'

    period = dt.timedelta(days=1)
    activities_generation_method = 'Macro'  # {'Simple', 'Macro'}
    duration_generation_method = 'Gaussian'  # {'Gaussian', 'Forecast Normal', 'TS Forecasting'}
    time_step_min = 120
    time_step = dt.timedelta(minutes=time_step_min)
    nb_replications = 10
    simmulation_id = 32
    Tep = 30  # For Macro Activities (Duration max of a macro activity)
    pattern_folder_id = 0


    dataset = pick_dataset(dataset_name)

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    # Compute the number of periods
    nb_periods = math.floor((end_date - start_date).total_seconds() / period.total_seconds())

    output = "../output/{}/Simulation/Simulation_X{}_Pattern_ID_{}/".format(dataset_name, simmulation_id,
                                                                            pattern_folder_id)

    # Create the folder if it does not exist yet
    if not os.path.exists(os.path.dirname(output)):
        try:
            os.makedirs(os.path.dirname(output))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    digital_twin_model = generate_all_activities(dataset_name, dataset, folder_id=pattern_folder_id, period=period,
                                                 time_step=time_step,
                                                 output=output, model=activities_generation_method,
                                                 duration_gen=duration_generation_method, Tep=Tep)

    # We can load them instead
    # all_activities = pickle.load(open(output + '/digital_twin_model.pkl', 'rb'))
    # print('Digital Twin Model  Loaded !!')

    index = np.arange(int(period.total_seconds() / time_step.total_seconds()) + 1)

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
            time_step_id = math.ceil(time_in_period / time_step.total_seconds()) % (max(index))

            # Search for the most likely to happen events
            candidate_activities = []
            count_candidate_activities = []

            current_date = current_sim_date + dt.timedelta(seconds=time_in_period)
            for activity in digital_twin_model:
                # stats = activity.get_stats_from_date(date=current_date, time_step_id=time_step_id)
                stats = activity.get_stats(time_step_id=time_step_id)
                count = stats['hist_count']
                if count > 0:
                    candidate_activities.append(activity)
                    count_candidate_activities.append(count)

            if len(candidate_activities) == 0:  # No candidates found
                time_in_period += time_step.total_seconds()
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
                time_in_period += time_step.total_seconds()
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


def generate_all_activities(dataset_name, dataset, period, time_step, output, folder_id, Tep, model='Simple',
                            duration_gen='Gaussian'):
    """
    Generate activities according to the method chosen
    :param dataset_name:
    :param dataset:
    :param period:
    :param time_step:
    :param output:
    :param model: 'Macro' pick the pattern episodes on their ranking and discover the temporality in the data
                  'Temporaral_Macro' pick the pattern episodes and their temporal occurrences on their ranking
    :param duration_gen:
    :param Tep:
    :return:
    """

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    all_activities = []
    train_dataset = dataset.copy()

    if model == 'Macro':  # Mining Macro activities first
        input = "../output/{}/ID_{}/patterns.pickle".format(dataset_name, folder_id)

        patterns = pd.read_pickle(input)

        patterns['Validity Duration'] = patterns['Validity Duration'].apply(lambda x: x.total_seconds())

        patterns['sort_key'] = patterns['Validity Duration'] * patterns['Accuracy']
        # patterns['sort_key'] = patterns['Episode'].apply(lambda x: len(x))  # * patterns['Accuracy']
        patterns.sort_values(['sort_key'], ascending=False, inplace=True)

        for index, pattern in patterns.iterrows():  # Create one Macro/Single Activity per row
            start_time = t.process_time()
            episode = list(pattern['Episode'])

            occurrences = find_occurrences(data=train_dataset, episode=episode)

            if occurrences.empty:
                continue

            if len(episode) > 1:
                activity = Activity.MacroActivity(episode=episode, dataset=train_dataset, occurrences=occurrences,
                                                  period=period, duration_gen=duration_gen, time_step=time_step,
                                                  start_date=start_date, end_date=end_date, display=False, Tep=Tep)
                plt.show()

            else:
                activity = Activity.Activity(label=episode, occurrences=occurrences, period=period, time_step=time_step,
                                             start_date=start_date, end_date=end_date)

            all_activities.append(activity)

            # Find the events corresponding to the expected occurrences
            mini_factorised_events = pd.DataFrame(columns=["date", "label"])
            for index, occurrence in occurrences.iterrows():
                occ_start_date = occurrence["date"]
                occ_end_date = occ_start_date + dt.timedelta(minutes=Tep)
                mini_data = train_dataset.loc[(train_dataset.label.isin(episode))
                                              & (train_dataset.date >= occ_start_date)
                                              & (train_dataset.date < occ_end_date)].copy()
                mini_data.sort_values(["date"], ascending=True, inplace=True)
                mini_data.drop_duplicates(["label"], keep='first', inplace=True)
                mini_factorised_events = mini_factorised_events.append(mini_data, ignore_index=True)

            train_dataset = pd.concat([train_dataset, mini_factorised_events], sort=False).drop_duplicates(keep=False)

            print(
                "Time spent for activity {}: {}".format(episode,
                                                        dt.timedelta(seconds=round(t.process_time() - start_time, 1))))

    # Mining of Single activities

    single_episodes = list(train_dataset.label.unique())

    for activity in single_episodes:
        start_time = t.process_time()
        episode = (activity,)
        occurrences = find_occurrences(data=train_dataset, episode=episode)
        activity = Activity.Activity(label=episode, occurrences=occurrences, period=period, time_step=time_step,
                                     start_date=start_date, end_date=end_date)
        all_activities.append(activity)
        print("Time spent for activity {}: {}".format(episode,
                                                      dt.timedelta(
                                                          seconds=round(t.process_time() - start_time, 1))))

    log_filename = output + "/parameters.txt"

    with open(log_filename, 'w+') as file:
        file.write("Parameters :\n")
        file.write("Pattern ID : {}\n".format(folder_id))
        file.write("Activities generation Method : {}\n".format(model))
        file.write("Duration generation Method : {}\n".format(duration_gen))
        file.write("Tep : {}\n".format(Tep))


    pickle.dump(all_activities, open(output + "/digital_twin_model.pkl", 'wb'))
    print('Digital Twin Model Created & Saved!!')

    return all_activities


if __name__ == '__main__':
    main()
