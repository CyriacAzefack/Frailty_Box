# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:07 2018

@author: cyriac.azefack
"""
import datetime as dt
import errno
import os
import sys
import time as t
from optparse import OptionParser

import pandas as pd

import xED.Pattern_Discovery as pattern_discovery
from Graph_Model import Pattern2Graph as p2g


def main(argv):
    # Default minimum supports
    support_dict = {
        'KA': 3,
        'KB': 2,
        'KC': 2,
        'aruba': 10
    }

    parser = OptionParser(usage='Usage of the Pattern Discovery algorihtm: %prog <options>')
    parser.add_option('-n', '--dataset_name', help='Name of the Input event log', dest='dataset_name', action='store',
                      type='string')
    parser.add_option('-r', '--replication', help='Patterns replication ID', dest='replication', action='store',
                      type=int, default=0)
    # parser.add_option('-o', '--output_dir', help='Output directory', dest='output_dir', action='store', type='string')
    parser.add_option('-w', '--window_size', help='Number of the days used', dest='window_size', action='store',
                      type=int, default=-1)
    parser.add_option('-s', '--support', help='Minimum number of occurrences of a pattern', dest='support_min',
                      action='store', type=int, default=3)
    parser.add_option('--sim', help='Number of replications', dest='nb_sim', action='store',
                      type=int, default=5)
    parser.add_option('--display', help='Display graphs', dest='display', action='store_true', default=False)

    (options, args) = parser.parse_args()
    # Mandatory Options
    if options.dataset_name is None:
        print("The name of the Input event log is missing\n")
        parser.print_help()
        exit(-1)
    elif options.replication is None:
        print("The name of the Discovery Pattern ID is missing")
        parser.print_help()
        exit(-1)

    dataset_name = options.dataset_name
    id_replication = options.replication
    nb_days = options.window_size
    support_min = options.support_min
    nb_sim = options.nb_sim
    display = options.display

    print("Dataset Name : {}".format(dataset_name.upper()))
    print("ID Replication : {}".format(id_replication))
    print("Number of days selected : {}".format(nb_days))
    print("Support Minimum : {}".format(support_min))
    print("Number of simulations to launch : {}".format(nb_sim))
    print("Display Patterns Graphs : {}".format(display))


    # READ THE INPUT DATASET
    dataset = pattern_discovery.pick_dataset(name=dataset_name, nb_days=nb_days)

    my_path = os.path.abspath(os.path.dirname(__file__))
    dirname = os.path.join(my_path, "./output/{}/ID_{}".format(dataset_name, id_replication))
    # dirname = "./output/{}".format(dataset_name)

    print("\n")
    print("###############################")
    print("SIMULATION N° {0:0=2d}".format(id_replication + 1))
    print("##############################")
    print("\n")

    # BUILD THE SIMULATION MODEL
    start_time = t.process_time()

    simulation_model = build_simulation_model(data=dataset, output_directory=dirname, debug=display)

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))


    print("###############################")
    print("SIMULATION MODEL BUILT  -  Time for the build : {}".format(elapsed_time))
    print("##############################")
    print()

    # START THE SIMULATION
    dirname += "/Simulation Replications/"

    # Create the folder if it doesn't exist
    if not os.path.exists(os.path.dirname(dirname)):
        try:
            os.makedirs(os.path.dirname(dirname))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    for sim_id in range(nb_sim):
        start_time = t.process_time()

        simulated_data = simulation(simulation_model=simulation_model, start_date=start_date,
                                    end_date=end_date)

        # SAVE THE SIMULATION RESULTS
        filename = dirname + "dataset_simulation_{}_{}.csv".format(id_replication + 1, sim_id + 1)
        simulated_data.to_csv(filename, index=False, sep=';')
        elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))

        print("###############################")
        print("# Simulation REPLICATION N°{}/{} done -  Time for the simulation {}".format(sim_id + 1, nb_sim,
                                                                                           elapsed_time))
        print("# Results save in '{}'".format(filename))
        print("##############################")


def build_simulation_model(data, output_directory='.', debug=False):
    '''
    Build the simulation model from scratch.
    :param data: Input sequence
    :param output_directory: Location of the 'patterns.pickle' file
    :return:
    '''

    patterns = pd.read_pickle(output_directory + '/patterns.pickle')

    # Build All the graphs associated with the patterns
    simulation_model = []
    output = output_directory
    print("###############################")
    print("Start building Graphs from patterns")
    print("##############################")

    validity_start_date = data.date.min().to_pydatetime()
    validity_end_date = data.date.max().to_pydatetime()

    for index, pattern in patterns.iterrows():
        labels = list(pattern['Episode'])
        period = pattern['Period']
        # validity_start_date = pattern['Start Time'].to_pydatetime()
        # validity_end_date = pattern['End Time'].to_pydatetime()

        time_description = pattern['Description']
        output_folder = output + "/Patterns_Graph/" + "_".join(labels) + "/"

        if not os.path.exists(os.path.dirname(output_folder)):
            try:
                os.makedirs(os.path.dirname(output_folder))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        patterns_graph_list = p2g.pattern2graph(data=data, labels=labels, time_description=time_description,
                                                period=period, start_date=validity_start_date,
                                                end_date=validity_end_date, output_directory=output_folder,
                                                display_graph=debug)
        simulation_model += patterns_graph_list

        sys.stdout.write("\r%.2f %% of patterns converted to graphs!!" % (100 * (index + 1) / len(patterns)))
        sys.stdout.flush()
    sys.stdout.write("\n")

    print("\n")
    print("###############################")
    print("All Graphs Built  -  Starting Time evolution")
    print("##############################")
    print("\n")

    for pattern_graph in simulation_model:
        print("### Computing Time Evolution for Graph N° {}/{}".format(simulation_model.index(pattern_graph) + 1,
                                                                       len(simulation_model)))
        pattern_graph.compute_time_evolution(data, len(simulation_model))

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
        sys.stdout.write("\r%.2f %% of patterns simulated!!" %
                         (100 * (simulation_model.index(pattern_graph) + 1) / len(simulation_model)))
        sys.stdout.flush()
    sys.stdout.write("\n")

    simulated_data.sort_values(["date"], inplace=True, ascending=True)

    return simulated_data


if __name__ == "__main__":
    main(sys.argv[1:])
