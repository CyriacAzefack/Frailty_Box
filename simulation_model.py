# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:07 2018

@author: cyriac.azefack
"""
from xED_algorithm import *


def main():
    letters = ['A']
    dataset_type = 'activity'

    for letter in letters:
        dataset = pick_dataset(letter, dataset_type)

        dirname = "output/K{} House/{}".format(letter, dataset_type)
        patterns = pickle.load(dirname + "/patterns.pickle")
        data_left = pickle.load(dirname + "/data_left.pickle")

        start_time = t.process_time()

        sim_model = build_simulation_model(data=dataset, patterns=patterns, tolerance_ratio=2)


def build_simulation_model(data, patterns, tolerance_ratio=2):
    return None
