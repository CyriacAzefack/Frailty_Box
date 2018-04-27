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

import xED_Algorithm.xED_Algorithm as xED
from Graph_Model import Pattern2Graph as p2g


def main(argv):
    # Default minimum supports
    support_dict = {
        'KA': 3,
        'KB': 2,
        'KC': 2,
        'aruba': 10
    }

    dataset_name = ''
    id_replication = ''
    nb_days = -1
    support_min = None
    nb_sim = 10
    graph_plot = False

    try:
        opts, args = getopt.getopt(argv, "hn:r:",
                                   ["name=", "replication_id=", "days=", "support_min=", "nbsim=", "graph_plot"])
    except getopt.GetoptError:
        print('Command Error :')
        print('Simulation_Model.py -n <dataset_name> -r <replication_id> [--days <number_days>] '
              '[--support_min <minimum support> [--nbsim <number of simulations>]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('How to use the command :')
            print('Simulation_Model.py -n <dataset_name> -r <replication_id> [--days <number_days>] '
                  '[--support_min <minimum support> [--nbsim <number of simulations>]')
            sys.exit()
        elif opt in ("-n", "--name"):
            dataset_name = arg
        elif opt in ("-r", "--replication"):
            id_replication = int(arg)
        elif opt in ("--days"):
            nb_days = int(arg)
        elif opt in ("--support_min"):
            support_min = int(arg)
        elif opt in ("--nbsim"):
            nb_sim = int(arg)
        elif opt in ("--graph_plot"):
            graph_plot = True

    # TODO : Automatically compute a better support min
    if not support_min:
        support_min = support_dict[dataset_name]

    print("Dataset Name : {}".format(dataset_name.upper()))
    print("ID Replication : {}".format(id_replication))
    print("Number of days selected : {}".format(nb_days))
    print("Support Minimum : {}".format(support_min))
    print("Number of simulations to launch : {}".format(nb_sim))
    print("Display Patterns Graphs : {}".format(graph_plot))


    # READ THE INPUT DATASET
    dataset = xED.pick_dataset(name=dataset_name, nb_days=nb_days)

    dirname = "./output/{}/Simulation Replications".format(dataset_name)

    print("\n")
    print("###############################")
    print("SIMULATION N° {0:0=2d}".format(id_replication + 1))
    print("##############################")
    print("\n")

    # BUILD THE SIMULATION MODEL
    start_time = t.process_time()

    sim_model = build_simulation_model(data=dataset, support_min=support_min, output_folder=dirname,
                                       display_graph=graph_plot)

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))

    print("\n")
    print("###############################")
    print("SIMULATION MODEL BUILT  -  Time for the build : {}".format(elapsed_time))
    print("##############################")
    print("\n")

    # START THE SIMULATION

    for sim_id in range(nb_sim):
        start_time = t.process_time()

        simulated_data = simulation(simulation_model=sim_model, start_date=start_date,
                                    end_date=end_date)

        # SAVE THE SIMULATION RESULTS
        filename = dirname + "/dataset_simulation_{}_{}.csv".format(id_replication + 1, sim_id)
        simulated_data.to_csv(filename, index=False, sep=';')
        elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))

        print("###############################")
        print("# Simulation REPLICATION N°{}  -  Time for the simulation {}".format(sim_id + 1, elapsed_time))
        print("# Results save in '{}'".format(filename))
        print("##############################")


def build_simulation_model(data, Tep=30, support_min=2, accuracy_min=0.5,
                           std_max=0.1, tolerance_ratio=2, delta_Tmax_ratio=3, output_folder='./',
                           verbose=True, display_graph=False):
    '''
    Build the somulation model from scratch.
    :param data: Input sequence
    :param Tep: Duration Max between event in an occurrence
    :param support_min:
    :param accuracy_min:
    :param std_max:
    :param tolerance_ratio:
    :param delta_Tmax_ratio:
    :param output_folder:
    :param verbose:
    :param nb_tries: Number of tries for the model. We take the one with the best data_explained_ratio
    :return:
    '''

    # TODO : Find a way to compute the ideal support

    # Unpack the patterns from the dataset
    patterns, patterns_string, data_left = xED.xED_algorithm(data=data, Tep=Tep, support_min=support_min,
                                                             accuracy_min=accuracy_min, std_max=std_max,
                                                             tolerance_ratio=tolerance_ratio,
                                                             delta_Tmax_ratio=delta_Tmax_ratio, verbose=verbose)

    ratio_data_treated = round((1 - len(data_left) / len(data)) * 100, 2)

    print("{}% of the dataset data explained by xED patterns".format(ratio_data_treated))

    # Build All the graphs associated with the patterns
    simulation_model = []
    output = output_folder
    for _, pattern in patterns.iterrows():
        labels = list(pattern['Episode'])
        period = pattern['Period']
        validity_start_date = pattern['Start Time'].to_pydatetime()
        validity_end_date = pattern['End Time'].to_pydatetime()
        validity_duration = validity_end_date - validity_start_date
        nb_periods = validity_duration.total_seconds() / period.total_seconds()
        time_description = pattern['Description']
        output_folder = output + "/Patterns_Graph/" + "_".join(labels) + "/"

        if not os.path.exists(os.path.dirname(output_folder)):
            try:
                os.makedirs(os.path.dirname(output_folder))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        patterns_graph_list = p2g.pattern2graph(data=data, labels=labels, time_description=time_description,
                                                period=period,
                                                start_date=validity_start_date, end_date=validity_end_date,
                                                output_directory=output_folder, debug=display_graph)

        simulation_model += patterns_graph_list

    return simulation_model


def simulation(simulation_model, start_date, end_date):
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
