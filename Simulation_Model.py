# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:07 2018

@author: cyriac.azefack
"""
import datetime as dt
import errno
import getopt
import os
import sys
import time as t

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__)))
from Pattern_Discovery import Pattern_Discovery as pattern_discovery
from Graph_Model import Pattern2Graph as p2g


def main(argv):
    # Default Minimum supports
    support_dict = {
        'KA': 3,
        'KB': 2,
        'KC': 2,
        'aruba': 10
    }

    dataset_name = ''
    id_replication = ''
    nb_days = -1
    try:
        opts, args = getopt.getopt(argv, "hn:r:", ["name=", "replication=", "days="])
    except getopt.GetoptError:
        print('Command Error :')
        print('Simulation_Model.py -n <dataset_name> -r <replication_id> [--days <number_days>]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('How to use the command :')
            print('Simulation_Model.py -n <dataset name> -r <replication index> [--days <number of days>] '
                  '[--support_min <minimum support>]')
            sys.exit()
        elif opt in ("-n", "--name"):
            dataset_name = arg
        elif opt in ("-r", "--replication"):
            id_replication = int(arg)
        elif opt in ("--days"):
            nb_days = int(arg)

    print("Dataset Name : {}".format(dataset_name.upper()))
    print("ID Replication : {}".format(id_replication))
    print("Number of days selected : {}".format(nb_days))



    # READ THE INPUT DATASET
    dataset = pattern_discovery.pick_dataset(name=dataset_name, nb_days=nb_days)

    dirname = "output/{}".format(dataset_name)

    print("\n")
    print("###############################")
    print("SIMULATION REPLICATION N° {0:0=2d}".format(id_replication + 1))
    print("##############################")
    print("\n")

    start_time = t.process_time()
    # BUILD THE SIMULATION MODEL
    sim_model = build_simulation_model(data=dataset, output_directory=dirname)

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()


    # START THE SIMULATION
    simulated_data = simulation(data=dataset, simulation_model=sim_model, start_date=start_date,
                                end_date=end_date)

    elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))

    print("\n")
    print("###############################")
    print("REPLICATION N°{}  -  Time to process the dataset : {}".format(id_replication + 1, elapsed_time))
    print("##############################")
    print("\n")

    simulated_data_filename = dirname + "/Simulation Replications/dataset_simulation_{0:0=3d}.csv".format(
        id_replication + 1)
    if not os.path.exists(os.path.dirname(simulated_data_filename)):
        try:
            os.makedirs(os.path.dirname(simulated_data_filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    # SAVE THE SIMULATION RESULTS
    simulated_data.to_csv(simulated_data_filename, index=False, sep=';')


def build_simulation_model(data, output_directory='.'):
    '''
    Build the somulation model from scratch.
    :param data: Input sequence
    :param Tep: Duration Max between event in an occurrence
    :param output_directory: Directory to find the 'patterns.pickle' file
    :param verbose:
    :return:
    '''


    # Unpack the patterns from the dataset
    patterns = pd.read_pickle(output_directory + '/patterns.pickle')

    # Build All the graphs associated with the patterns
    simulation_model = []
    output = output_directory
    for _, pattern in patterns.iterrows():
        labels = list(pattern['Episode'])
        period = pattern['Period']
        validity_start_date = pattern['Start Time'].to_pydatetime()
        validity_end_date = pattern['End Time'].to_pydatetime()
        validity_duration = validity_end_date - validity_start_date
        nb_periods = validity_duration.total_seconds() / period.total_seconds()
        description = pattern['Description']
        output_directory = output + "/Patterns_Graph/" + "_".join(labels) + "/"

        if not os.path.exists(os.path.dirname(output_directory)):
            try:
                os.makedirs(os.path.dirname(output_directory))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        patterns_graph_list = p2g.pattern2graph(data=data, labels=labels, time_description=description, period=period,
                                                start_date=validity_start_date, end_date=validity_end_date,
                                                output_dir=output_directory, draw_graphs=False)

        simulation_model += patterns_graph_list

    return simulation_model


def simulation(data, simulation_model, start_date, end_date):
    '''
    Simulate the model from start_date to end_date and compare it to the original data
    :param data: Date to use for the model evaluation
    :param simulation_model: List of <i>Pattern Graphs</i>
    :param start_date: [datetime] Start date of the simulation
    :param end_date:  [datetime] End date of the simulation
    :return: Event sequence produced by the simulation
    '''

    simulated_data = pd.DataFrame(columns=["date", "label"])

    for pattern_graph in simulation_model:
        pattern_simulation_data = pattern_graph.simulate(start_date=start_date, end_date=end_date)
        simulated_data = pd.concat([simulated_data, pattern_simulation_data]).drop_duplicates(subset=['date', 'label'],
                                                                                              keep='first')
        simulated_data.reset_index(inplace=True, drop=True)

    simulated_data.sort_values(["date"], inplace=True, ascending=True)

    return simulated_data


if __name__ == "__main__":
    main(sys.argv[1:])
