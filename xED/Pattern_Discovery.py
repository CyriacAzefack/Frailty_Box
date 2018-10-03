# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:22:14 2018

@author: cyriac.azefack
"""

import errno
import getopt
import inspect
import pickle
import sys
import time as t

import numpy as np

from Utils import *
from xED import Candidate_Study
from xED import FP_growth


def main(argv):
    ######################
    # DATA PREPROCESSING #
    ######################

    """
    The dataframe should have 1 index (date as datetime) and 1 feature (label)
    """
    support_dict = {
        'KA': 3,
        'KB': 2,
        'KC': 2,
        'aruba': 10
    }

    replication_id = None
    dataset_name = ''
    output_dir = None
    nb_days = -1
    support_min = None
    accuracy_min = 0.5
    Tep = 30  # 30 minutes

    try:
        opts, args = getopt.getopt(argv, "hn:r:o:",
                                   ["name=", "replication=", "output_dir=", "days=", "support_min=", "accuracy_min=",
                                    "Tep="])
    except getopt.GetoptError:
        print('Pattern_Discovery.py -n <dataset name> -n <replication index> -o <output dir> [--days <number of days>]'
              ' [--support_min <minimum support>] [--accuracy_min <minimum accuracy>] [--Tep <episode length parameter>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(
                'Pattern_Discovery.py -n <dataset name> -o <output dir> [--days <number of days>] '
                '[--support_min <minimum support>]')
            sys.exit()
        elif opt in ("-n", "--name"):
            dataset_name = arg
        elif opt in ("-r", "--replication"):
            replication_id = int(arg)
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt == '--days':
            nb_days = int(arg)
        elif opt == '--support_min':
            support_min = int(arg)
        elif opt == '--accuracy_min':
            accuracy_min = float(arg)
        elif opt == '--Tep':
            Tep = int(arg)

    # TODO : Find a way to compute the ideal support

    if not support_min:
        support_min = support_dict[dataset_name]

    if not output_dir:
        my_path = os.path.abspath(os.path.dirname(__file__))
        output_dir = os.path.join(my_path, "../output/{}/ID_{}".format(dataset_name, replication_id))

    print("Dataset Name : {}".format(dataset_name.upper()))
    print("Replication ID : {}".format(replication_id))
    print("Output Directory : {}".format(output_dir))
    print("Number of days selected : {}".format(nb_days))
    print("Support Minimum : {}".format(support_min))
    print("Accuracy Minimum : {}".format(accuracy_min))
    print("Tep : {}".format(Tep))


    dataset = pick_dataset(name=dataset_name, nb_days=nb_days)

    # Find the best case among different tries:

    start_time = t.process_time()
    patterns, patterns_string, data_left = pattern_discovery(data=dataset, Tep=Tep, accuracy_min=0.3,
                                                             support_min=support_min)

    p = inspect.signature(pattern_discovery)
    ratio_data_treated = round((1 - len(data_left) / len(dataset)) * 100, 2)

    elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))

    print("\n")
    print("###############################")
    print("Time to discover the patterns : {}".format(elapsed_time))
    print(
        "{}% of the {} dataset data explained by the patterns discovered".format(ratio_data_treated, dataset_name))
    print("##############################")
    print("\n")

    log_filename = output_dir + "/log.txt"
    if not os.path.exists(os.path.dirname(log_filename)):
        try:
            os.makedirs(os.path.dirname(log_filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(log_filename, 'w+') as file:
        file.write("Parameters :\n")
        file.write("Support Min : {}\n".format(support_min))
        file.write("Accuracy Min : {}\n".format(accuracy_min))
        file.write("Tep : {}\n".format(Tep))
        file.write("Time to process all the tries : {}\n".format(elapsed_time))
        file.write("{}% of the {} dataset explained by the patterns discovered\n".format(ratio_data_treated,
                                                                                         dataset_name))
        file.write("{} Patterns found\n".format(len(patterns)))

    # Dump the results in pickle files to re-use them later
    pickle.dump(patterns, open(output_dir + "/patterns.pickle", 'wb'))
    pickle.dump(data_left, open(output_dir + "/data_left.pickle", 'wb'))

    # Write readable results in csv file
    patterns_string.to_csv(output_dir + "/patterns.csv", sep=";", index=False)
    data_left.to_csv(output_dir + "/data_left.csv", sep=";", index=False)


def pattern_discovery(data, Tep=30, support_min=2, accuracy_min=0.5,
                      std_max=0.1, tolerance_ratio=2, delta_Tmax_ratio=3, verbose=True, weekly_habits=False):
    """
    Implementation of the extended Discovery Algorithm designed by Julie Soulas U{https://hal.archives-ouvertes.fr/tel-01356217/}

    :param data : Starting dataframe, date[datetime] as index and 1 column named "label"
    :param Tep : [in Minutes] Maximal time interval between events in an episode occurrence. Should correspond to the maximal duration of the ADLs.
    :param support_min : [greater than 1] Minimal number of occurrences of an episode, for that episode to be considered as frequent.
    :param accuracy_min : [between 0 and 1] Minimal accuracy for a periodicity description to be considered as interesting, and thus factorized
    :param std_max : Standard deviation max for a description (ratio of the period)
    :param tolerance_ratio : [greater than 0] An event expected to happen at time t (with standard deviation sigma) occurs as expected if it occurs in the interval [t - tolerance_ratio*sigma, t + tolerance_ratio*sigma]
    :param delta_Tmax_ratio : If there is a gap > delta_Tmax_ratio * Period between two occurrences of an episode, theoccurrences before and after the gap are split (different validity intervals).
    :return The compressed dataset
    """
    compressed = True

    final_periodicities = pd.DataFrame(columns=["Episode", "Period", "Description", "Validity Duration",
                                                "Start Time", "End Time", "Compression Power", "Accuracy"])
    final_periodicities_string = pd.DataFrame(columns=["Episode", "Period", "Description", "Validity Duration",
                                                       "Start Time", "End Time", "Compression Power", "Accuracy"])
    comp_iter = 0

    candidate_periods = [dt.timedelta(days=1), ]

    if weekly_habits:
        candidate_periods.append(dt.timedelta(days=7))
    while compressed:
        comp_iter += 1

        if verbose:
            print("\n")
            print("###############################")
            print("#    COMPRESSION N°%d START   #" % comp_iter)
            print("##############################")
            print("\n")

        compressed = False

        if verbose:
            print("  Finding frequent episodes candidates...  ".center(100, '*'))

        frequent_episodes = FP_growth.find_frequent_episodes(data, support_min, Tep)

        print(len(frequent_episodes), "candidate episodes found !!")

        periodicities = {}

        print(" Building candidates episodes periodicities... ".center(100, '*'))

        episode_index = 0
        for episode in frequent_episodes.keys():
            episode_index += 1
            # Build the description of the episode if interesting enough (Accuracy > Accuracy_min)
            description = Candidate_Study.periodicity_search(data, episode,
                                                             delta_Tmax_ratio=delta_Tmax_ratio,
                                                             support_min=support_min,
                                                             std_max=std_max,
                                                             accuracy_min=accuracy_min,
                                                             tolerance_ratio=tolerance_ratio, Tep=Tep,
                                                             candidate_periods=candidate_periods)
            if description is not None:
                # print("\nInteresting periodicity found for the episode", episode)
                periodicities[episode] = description

            sys.stdout.write("\r%.2f %% of episodes treated!!" % (100 * episode_index / len(frequent_episodes)))
            sys.stdout.flush()
        sys.stdout.write("\n")

        if len(periodicities) == 0:
            print("No more intersting periodicities found!!".center(100, '*'))
            break;
        else:
            print((str(len(periodicities)) + " interesting periodicities found!!").center(100, '*'))

        print("Sorting the interesting periodicities by compression power...".center(100, '*'))
        sorted_episode = sorted(periodicities.keys(), key=lambda x: periodicities[x]["compression_power"], reverse=True)

        print("Dataset Rewriting".center(100, '*'))

        factorised_events = pd.DataFrame(columns=["date", "label"])

        data_bis = data.copy()  # Data to handle the rewriting
        while True:

            episode = sorted_episode[0]
            periodicity = periodicities[episode]

            if periodicity["compression_power"] <= 1:
                print("### COMPRESSION FINISHED : Insufficient compression power reached...")
                break;

            expected_occurrences = periodicity["expected_occurrences"]

            mini_factorised_events = pd.DataFrame(columns=["date", "label"])

            # Find the events corresponding to the expected occurrences
            for index, occurrence in expected_occurrences.iterrows():
                start_date = occurrence["date"]
                end_date = start_date + dt.timedelta(minutes=Tep)
                mini_data = data_bis.loc[(data_bis.label.isin(episode))
                                         & (data_bis.date >= start_date)
                                         & (data_bis.date < end_date)].copy()
                mini_data.sort_values(["date"], ascending=True, inplace=True)
                mini_data.drop_duplicates(["label"], keep='first', inplace=True)
                mini_factorised_events = mini_factorised_events.append(mini_data, ignore_index=True)

            factorised_events = factorised_events.append(mini_factorised_events, ignore_index=True)
            factorised_events.sort_values(by=['date'], ascending=True, inplace=True)
            count_duplicates = factorised_events.duplicated(['date', 'label']).sum()
            if count_duplicates != 0:
                # Current periodicity involves events in factorized_events
                print("### COMPRESSION FINISHED : Overlapping factorized events reached...")
                break

            # Factorize DATA
            data = pd.concat([data, mini_factorised_events], sort=True).drop_duplicates(keep=False)

            # Add missing events
            # FIXME : Do we add missing events or not?
            if False:
                events_to_add = find_missing_events(data_bis, episode, expected_occurrences,
                                                    periodicity["description"], periodicity["period"], tolerance_ratio)
                data = pd.concat([data, events_to_add]).drop_duplicates(keep=False)

            data.reset_index(inplace=True, drop=True)

            # Add the periodicity to the results
            natural_periodicity = Candidate_Study.translate_description(periodicity)

            final_periodicities.loc[len(final_periodicities)] = [episode, periodicity["period"],
                                                                 periodicity["description"],
                                                                 periodicity["delta_t"][1] - periodicity["delta_t"][0],
                                                                 periodicity["delta_t"][0],
                                                                 periodicity["delta_t"][1],
                                                                 periodicity["compression_power"],
                                                                 periodicity["accuracy"]]

            final_periodicities_string.loc[len(final_periodicities_string)] = [episode, natural_periodicity["period"],
                                                                               natural_periodicity["description"],
                                                                               natural_periodicity["validity duration"],
                                                                               natural_periodicity["delta_t"][0],
                                                                               natural_periodicity["delta_t"][1],
                                                                               natural_periodicity["compression_power"],
                                                                               natural_periodicity["accuracy"]]

            # Remove periodicity from the list
            sorted_episode = sorted_episode[1:]
            if len(sorted_episode) == 0:
                print("### COMPRESSION FINISHED : No more frequent episodes...")
                break;

            compressed = True

    return final_periodicities, final_periodicities_string, data


def find_missing_events(data, episode, occurrences, description, period, tolerance_ratio):
    """
    Find missing occurrences of the episode description in the original dataset
    :return the missing events
    """

    data = data.loc[data.label.isin(episode)]

    missing_events_df = pd.DataFrame(columns=["date", "label"])

    occ_start_time = occurrences.date.min().to_pydatetime()
    start_period_date = occ_start_time + dt.timedelta(
        seconds=(period.total_seconds() - Candidate_Study.modulo_datetime(occ_start_time, period)))

    occ_end_time = occurrences.date.max().to_pydatetime()
    end_period_date = occ_end_time - dt.timedelta(
        seconds=Candidate_Study.modulo_datetime(occ_end_time, period))

    # Deal with the periods
    current_period_start_date = start_period_date
    while current_period_start_date < end_period_date:
        current_period_end_date = current_period_start_date + period
        for mu, sigma in description.items():
            comp_start_date = current_period_start_date + dt.timedelta(seconds=(mu - tolerance_ratio * sigma))
            comp_end_date = current_period_start_date + dt.timedelta(seconds=(mu + tolerance_ratio * sigma))

            occurrence_happenned = len(
                occurrences.loc[(occurrences.date >= comp_start_date) & (occurrences.date <= comp_end_date)]) > 0

            if occurrence_happenned:
                continue

            # If not happenned fill the missing events
            present_events = set(
                data.loc[(data.date >= comp_start_date) & (data.date <= comp_end_date), "label"].values)
            intersection = present_events.intersection(episode)
            missing_events = set(episode) - intersection

            for event in intersection:
                event_date = data.loc[
                    (data.date >= comp_start_date) & (data.date <= comp_end_date) & (data.label == event)].date.min()
                missing_events_df.loc[len(missing_events_df)] = [event_date, event]
            for event in missing_events:
                ts = int(mu + sigma * np.random.randn())
                event_date = current_period_start_date + dt.timedelta(seconds=ts)
                missing_events_df.loc[len(missing_events_df)] = [event_date, "MISSING " + event]

        current_period_start_date = current_period_end_date

    # Now the bord effects
    for mu, sigma in description.items():
        if mu < Candidate_Study.modulo_datetime(occ_start_time, period):
            continue
        if mu > Candidate_Study.modulo_datetime(occ_end_time, period):
            continue

        comp_start_date = current_period_start_date + dt.timedelta(seconds=(mu - tolerance_ratio * sigma))
        comp_end_date = current_period_start_date + dt.timedelta(seconds=(mu + tolerance_ratio * sigma))

        occurrence_happenned = len(
            occurrences.loc[(occurrences.date >= comp_start_date) & (occurrences.date <= comp_end_date)]) > 0

        if occurrence_happenned:
            continue

        # If not happenned fill the missing events
        present_events = set(data.loc[(data.date >= comp_start_date) & (data.date <= comp_end_date), "label"].values)
        missing_events = set(episode) - present_events.intersection(episode)

        for event in missing_events:
            ts = int(mu + sigma * np.random.randn())
            event_date = current_period_start_date + dt.timedelta(seconds=ts)
            missing_events_df.loc[len(missing_events_df)] = [event_date, "MISSING " + event]

    return missing_events_df





if __name__ == "__main__":
    main(sys.argv[1:])
