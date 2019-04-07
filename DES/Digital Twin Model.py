import datetime as dt
import errno
import os
import pickle
import random
import sys
import time as t
from optparse import OptionParser

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

from DES import Activity
from xED.Candidate_Study import find_occurrences, modulo_datetime
from xED.Pattern_Discovery import pick_dataset


# random.seed(1996)


def main():
    ####################
    # Macro Activities MODEL
    #
    # Characteristics :
    #   - Single Activity Implementation
    #   - No time evolution
    #   - Training Dataset : the Whole Original dataset
    #   - Test Dataset : The Whole Original dataset


    period = dt.timedelta(days=1)
    simulation_id = 1

    parser = OptionParser(usage='Usage of the Digital Twin Simulation model algorihtm: %prog <options>')
    parser.add_option('-n', '--dataset_name', help='Name of the Input event log', dest='dataset_name', action='store',
                      type='string')
    parser.add_option('--activity_gen', help='Method of Macro-Activity generation', dest='activity_gen',
                      action='store', type='choice', choices=['simple', 'macro'], default='macro')
    parser.add_option('--duration_model', help='Model for event duration', dest='duration_model',
                      action='store', type='choice', choices=['gaussian', 'forecast_norm', 'ts_forecast'],
                      default='gaussian')

    parser.add_option('--time_step', help='Number of minutes per time step', dest='time_step', action='store',
                      type=int, default=5)
    parser.add_option('--sim', help='Number of replications', dest='nb_sim', action='store',
                      type=int, default=5)
    parser.add_option('-t', '--tep', help='Duration max of an occurrence (in minutes)', dest='tep', action='store',
                      type=int, default=30)
    parser.add_option('--pattern_id', help='Identifier of the patterns used', dest='pattern_id', action='store',
                      type=int, default=0)
    parser.add_option('-d', '--debug', help='Display all the intermediate steps', dest='debug', action='store_true',
                      default=False)

    (options, args) = parser.parse_args()
    # Mandatory Options
    if options.dataset_name is None:
        print("The name of the Input event log is missing\n")
        parser.print_help()
        exit(-1)

    dataset_name = options.dataset_name
    activity_gen = options.activity_gen
    duration_model = options.duration_model
    time_step = dt.timedelta(minutes=options.time_step)
    nb_replications = options.nb_sim
    pattern_id = options.pattern_id
    Tep = options.tep
    debug = options.debug

    print('#' + 'PARAMETERS'.center(28, ' ') + '#')
    print("Dataset Name : {}".format(dataset_name.upper()))
    print("Pattern ID : {}".format(pattern_id))
    print("Tep (mn): {}".format(Tep))
    print("Activity Generation : {}".format(activity_gen))
    print("Event duration model : {}".format(duration_model))
    print("Time step duration (mn) : {}".format(int(time_step.total_seconds() / 60)))
    print("Number of replications : {}".format(nb_replications))
    print("Debug Mode: {}".format(debug))


    dataset = pick_dataset(dataset_name)

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    # Compute the number of periods
    nb_days = math.floor((end_date - start_date).total_seconds() / period.total_seconds())
    # training_days = int(math.floor(nb_days*0.9)) # 90% for training, 10% for test
    training_days = nb_days

    output = "../output/{}/Simulation/Simulation_X{}_Pattern_ID_{}/".format(dataset_name, simulation_id,
                                                                            pattern_id)

    # Create the folder if it does not exist yet
    if not os.path.exists(os.path.dirname(output)):
        try:
            os.makedirs(os.path.dirname(output))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    training_start_date = start_date
    training_end_date = start_date + dt.timedelta(days=training_days)

    simulation_start_date = training_start_date
    simulation_duration = dt.timedelta(days=training_days)

    training_dataset = dataset[(dataset.date >= training_start_date) & (dataset.date < training_end_date)].copy()

    digital_twin_model = build_Activities(dataset_name, dataset=training_dataset, folder_id=pattern_id, period=period,
                                          time_step=time_step, output=output, model=activity_gen,
                                          duration_model=duration_model, Tep=Tep, display=debug)

    # validate_simulation(digital_twin_model, original_data=dataset, period=period, time_step=time_step)

    # We can load them instead
    # all_activities = pickle.load(open(output + '/digital_twin_model.pkl', 'rb'))
    # print('Digital Twin Model  Loaded !!')


    # print('###################################')
    # print('# 4 - Start the loop Processus    #')
    # print('###################################')

    # current_sim_date = end_date + period - dt.timedelta(seconds=modulo_datetime(end_date, period))

    ###########################
    # SIMULATION ##############
    ###########################



    for replication in range(nb_replications):

        time_start = t.process_time()

        print('###################################')
        print('# Simulation replication N°{}     #'.format(replication + 1))
        print('###################################')

        simulated_dataset = launch_simulation(digital_twin_model, simulation_duration=simulation_duration,
                                              time_step=time_step, start_date=simulation_start_date, period=period)

        filename = output + "dataset_simulation_rep_{}.csv".format(replication + 1)

        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        simulated_dataset.to_csv(filename, index=False, sep=';')
        elapsed_time = dt.timedelta(seconds=round(t.process_time() - time_start, 1))
        print("Time elapsed for the simulation : {}".format(elapsed_time))


def build_Activities(dataset_name, dataset, period, time_step, output, folder_id, Tep, model='Macro',
                     duration_model='gaussian', display=False):
    """
    Generate activities according to the method chosen
    :param dataset_name:
    :param dataset:
    :param period:
    :param time_step:
    :param output:
    :param model: 'Macro' pick the pattern episodes on their ranking and discover the temporality in the data
                  'Temporaral_Macro' pick the pattern episodes and their temporal occurrences on their ranking
    :param duration_model:
    :param Tep:
    :return:
    """

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    all_activities = []
    train_dataset = dataset.copy()

    if model == 'macro':  # Mining Macro activities first
        input = "../output/{}/ID_{}/patterns.pickle".format(dataset_name, folder_id)

        patterns = pd.read_pickle(input)

        patterns['Validity Duration'] = patterns['Validity Duration'].apply(lambda x: x.total_seconds())

        # Rank the macro-activities
        patterns['sort_key'] = patterns['Compression Power'] * patterns['Accuracy']
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
                                                  period=period, duration_model=duration_model, time_step=time_step,
                                                  start_date=start_date, end_date=end_date, display=display, Tep=Tep)


            else:
                activity = Activity.Activity(label=episode, occurrences=occurrences, period=period, time_step=time_step,
                                             start_date=start_date, end_date=end_date, display=display)


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
        file.write("Duration generation Method : {}\n".format(duration_model))
        file.write("Time Step : {} min\n".format(time_step.total_seconds() / 60))
        file.write("Tep : {}\n".format(Tep))


    pickle.dump(all_activities, open(output + "/digital_twin_model.pkl", 'wb'))
    print('Digital Twin Model Created & Saved!!')

    return all_activities


def launch_simulation(digital_twin_model, simulation_duration, time_step, start_date, period=dt.timedelta(days=1),
                      allow_nothing_happen=False):
    """
    Launch the simulation on the digital twin model
    :param digital_twin_model:
    :param simulation_duration:
    :param time_step:
    :param start_date:
    :param period:
    :param allow_nothing_happen: If False, no idle time between macro-activities
    :return:
    """

    simulated_dataset = pd.DataFrame(columns=['date', 'end_date', 'label'])

    current_sim_date = start_date

    seconds_since_start = 0  # seconds

    time_step_index = np.arange(int(period.total_seconds() / time_step.total_seconds()) + 1)

    while seconds_since_start < simulation_duration.total_seconds():

        # Print evolution
        evolution_percentage = round(100 * (seconds_since_start / simulation_duration.total_seconds()), 2)
        sys.stdout.write("\r{} %% of Simulation done!!".format(evolution_percentage))
        sys.stdout.flush()

        # Pick the next group index according to seconds_since_start value
        time_step_id = math.ceil(seconds_since_start / time_step.total_seconds()) % (max(time_step_index))

        # Search for the events most likely to occur

        prob_activities = []

        current_date = current_sim_date + dt.timedelta(seconds=seconds_since_start)
        for activity in digital_twin_model:
            # stats = activity.get_stats_from_date(date=current_date, time_step_id=time_step_id)
            stats = activity.get_stats(time_step_id=time_step_id)
            prob_activities.append(stats['hist_prob'])

        prob_activities = np.asarray(prob_activities)

        if not allow_nothing_happen:
            prob_activities = prob_activities / sum(prob_activities)

        prob_activities = np.cumsum(prob_activities)

        # Pick randomly an activity
        # Pick a random number between 0 -- 1
        rand = random.random()
        chosen_activity = None

        for i in range(len(prob_activities)):
            if rand <= prob_activities[i]:
                chosen_activity = digital_twin_model[i]
                break

        if chosen_activity is None:  # Nothing happens
            seconds_since_start += time_step.total_seconds()
            continue

        # Generate the activity
        activity_simulation, activity_duration = chosen_activity.simulate(current_date, time_step_id)

        if activity_duration <= 0:
            continue

        simulated_dataset = pd.concat([simulated_dataset, activity_simulation], sort=True).drop_duplicates(
            keep='first')

        seconds_since_start += activity_duration
        pass

    sys.stdout.write("\n")

    return simulated_dataset


def validate_simulation(digital_twin_model, original_data, period, time_step):
    """
    Compute the probability to generate the original data sequence
    :param digital_twin_model:
    :param original_data:
    :return:
    """
    start_date = original_data.date.min().to_pydatetime()
    end_date = start_date + dt.timedelta(days=30)

    nb_events = len(original_data[(original_data.date >= start_date) & (original_data.date < end_date)])

    # nb_events = 200

    time_step_index = np.arange(int(period.total_seconds() / time_step.total_seconds()) + 1)

    data = original_data.head(nb_events).copy()

    probabilities = []
    current_prob = 1
    current_time = original_data.date.min().to_pydatetime()

    probabilities.append(current_prob)

    for index, event in data.iterrows():

        relative_ts = modulo_datetime(current_time, period)
        time_step_id = math.ceil(relative_ts / time_step.total_seconds()) % (max(time_step_index))

        #################
        # LABEL PROB
        #################
        # Compute the probability of an event starting at this moment and lasting for that long
        label = event['label']
        # event_duration = (event['end_date'] - event['date']).total_seconds()
        # prob_event = 0

        candidate_activities = []
        count_candidate_activities = []
        label_count = 0
        for activity in digital_twin_model:
            # stats = activity.get_stats_from_date(date=current_date, time_step_id=time_step_id)
            stats = activity.get_stats(time_step_id=time_step_id)
            count = stats['hist_count']
            if count > 0:
                candidate_activities.append(activity)
                count_candidate_activities.append(count)

            if label in activity.get_label():
                label_count += count

        count_candidate_activities = np.asarray(count_candidate_activities)
        # normalize

        prob_label = label_count / sum(count_candidate_activities)

        prob_event = prob_label * probabilities[-1]
        probabilities.append(prob_label)
        current_time = event['end_date'].to_pydatetime()

    y = signal.savgol_filter(probabilities, 11, 3)

    # y = probabilities

    plt.plot(np.arange(len(y)), y)
    plt.show()


if __name__ == '__main__':
    main()
