# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:22:14 2018

@author: cyriac.azefack
"""

import datetime as dt
import errno
import getopt
import os
import pickle
import sys
import time as t

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__)))
import Candidate_Study
import FP_growth


# matplotlib.style.use("seaborn")


def main(argv):
    # Default minimum supports
    support_dict = {
        'KA': 3,
        'KB': 2,
        'KC': 2,
        'aruba': 10
    }

    dataset_name = ''
    nb_days = -1
    support_min = None
    nb_tries = 1


    try:
        opts, args = getopt.getopt(argv, "hn:",
                                   ["name=", "days=", "support_min="])
    except getopt.GetoptError:
        print('Command Error :')
        print(
            'xED_Algorithm.py -n <dataset name> [--days <number of days to select>] [--support_min <minimum support>]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('How to use the command :')
            print('xED_Algorithm.py -n <dataset_name> [--days <number_days>] [--support_min <minimum support>')
            sys.exit()
        elif opt in ("-n", "--name"):
            dataset_name = arg
        elif opt in ("--days"):
            nb_days = int(arg)
        elif opt in ("--support_min"):
            support_min = int(arg)

    if support_min is None:
        support_min = support_dict[dataset_name]

    dataset = pick_dataset(name=dataset_name, nb_days=nb_days)

    output_dir = '../output/{}'.format(dataset_name)
    start_time = t.process_time()

    best_ratio_data_treated = 0
    best_patterns = None
    best_patterns_string = None
    best_data_left = None

    # Find the best case among different tries
    for _ in range(nb_tries):
        patterns, patterns_string, data_left = xED_algorithm(data=dataset, Tep=30,
                                                             support_min=support_min,
                                                             tolerance_ratio=2)
        ratio_data_treated = round((1 - len(data_left) / len(dataset)) * 100, 2)

        if ratio_data_treated > best_ratio_data_treated:
            best_ratio_data_treated = ratio_data_treated
            best_patterns = patterns
            best_patterns_string = patterns_string
            best_data_left = data_left

    elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))

    print("\n")
    print("###############################")
    print("Time to process all the tries : {}".format(elapsed_time))
    print("{}% of the {} dataset data explained by xED patterns".format(best_ratio_data_treated, dataset_name))
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
        file.write("Time to process all the tries : {}".format(elapsed_time))
        file.write("{}% of the {} dataset data explained by xED patterns".format(best_ratio_data_treated, dataset_name))
        file.write("{} Patterns found\n".format(len(best_patterns)))

    # Dump the results in pickle files to re-use them later
    pickle.dump(best_patterns, open(output_dir + "/patterns.pickle", 'wb'))
    pickle.dump(best_data_left, open(output_dir + "/data_left.pickle", 'wb'))

    # Write readable results in csv file
    best_patterns_string.to_csv(output_dir + "/patterns.csv", sep=";", index=False)
    best_data_left.to_csv(output_dir + "/data_left.csv", sep=";", index=False)

    # Write all the results in differents excel sheets
    # writer = pd.ExcelWriter(output_dir + "/all_results.xlsx")
    # dataset.to_excel(writer, sheet_name="Input Data", index=False)
    # best_patterns_string.to_excel(writer, sheet_name="Patterns", index=False)
    # best_data_left.to_excel(writer, sheet_name="Non treated Data", index=False)
    # writer.save()


def xED_algorithm(data, Tep=30, support_min=2, accuracy_min=0.5,
                  std_max=0.1, tolerance_ratio=2, delta_Tmax_ratio=3, verbose=True):
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
                                                             candidate_periods=[dt.timedelta(days=1),
                                                                                dt.timedelta(days=7)])
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
            data = pd.concat([data, mini_factorised_events]).drop_duplicates(keep=False)

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


def pick_dataset(name, nb_days=-1):
    my_path = os.path.abspath(os.path.dirname(__file__))

    dataset = None
    if name == 'toy':
        path = os.path.join(my_path, "../input/toy_dataset.csv")
        dataset = pd.read_csv(path, delimiter=';')
        date_format = '%Y-%d-%m %H:%M'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)

    elif name == 'aruba':
        path = os.path.join(my_path, "../input/aruba/dataset.csv")
        dataset = pd.read_csv(path, delimiter=';')
        date_format = '%Y-%m-%d %H:%M:%S.%f'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)

    else:
        filename = "../input/{} House/{}_label_dataset.csv".format(name, name)
        path = os.path.join(my_path, filename)
        dataset = pd.read_csv(path, delimiter=';')
        dataset['date'] = pd.to_datetime(dataset['date'])

    # We only take nb_days
    if nb_days > 0:
        start_date = dataset.date.min().to_pydatetime()
        end_date = start_date + dt.timedelta(days=nb_days)
        dataset = dataset.loc[(dataset.date >= start_date) & (dataset.date < end_date)].copy()

    return dataset


if __name__ == "__main__":
    main(sys.argv[1:])
