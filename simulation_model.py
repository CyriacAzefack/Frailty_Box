# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:07 2018

@author: cyriac.azefack
"""
import errno
import os

import pandas as pd

import patterns2graph as p2g
import xED_algorithm as xED


def main():
    letters = ['A', 'B', 'C']
    dataset_type = 'label'

    support_dict = {
        'A': 3,
        'B': 2,
        'C': 2
    }


    for letter in letters:
        dataset = xED.pick_dataset(letter, dataset_type)

        dirname = "output/K{} House/{}".format(letter, dataset_type)

        sim_model = build_simulation_model(data=dataset, support_min=support_dict[letter], output_folder=dirname)

        start_date = dataset.date.min().to_pydatetime()
        end_date = dataset.date.max().to_pydatetime()

        simulated_data = simulation(data=dataset, simulation_model=sim_model, start_date=start_date, end_date=end_date)

        simulated_data.to_csv(dirname + "/simulated_data.csv", index=False, sep=';')


def build_simulation_model(data, Tep=30, support_min=2, accuracy_min=0.5,
                           std_max=0.1, tolerance_ratio=2, delta_Tmax_ratio=3, output_folder='./', verbose=True):
    '''
    Build the entire simulation model
    :param data:
    :param tolerance_ratio:
    :return:
    '''

    # TODO : Find a way to compute the ideal support

    # Unpack the patterns from the dataset
    patterns, patterns_string, data_left = xED.xED_algorithm(data=data, Tep=Tep, support_min=support_min,
                                                             accuracy_min=accuracy_min,
                                                             std_max=std_max, tolerance_ratio=tolerance_ratio,
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
        description = pattern['Description']
        output_folder = output + "/Patterns_Graph/" + "_".join(labels) + "/"

        if not os.path.exists(os.path.dirname(output_folder)):
            try:
                os.makedirs(os.path.dirname(output_folder))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        patterns_graph_list = p2g.patterns2graph(data=data, labels=labels, description=description, period=period,
                                                 start_date=validity_start_date, end_date=validity_end_date,
                                                 output_folder=output_folder)

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
    main()
