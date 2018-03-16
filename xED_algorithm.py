# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:22:14 2018

@author: cyriac.azefack
"""

import sys 
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from dateutil.parser import parse
import seaborn as sns

matplotlib.style.use("seaborn")

import FP_growth
import candidate_study


def main(): 
    ########################################
    # DATA PREPROCESSING
    ########################################
    
    """
    The dataframe should have 1 index (date as datetime) and 1 feature (activity)
    """
    
    dataset = pick_dataset('toy')
    
    #dataset = dataset.set_index('date')
    results, data = xED_algorithm(data=dataset, Tep=30, support_min=3, 
                                tolerance_ratio=2)
    

def xED_algorithm(data, Tep = 30, support_min = 2, accuracy_min = 0.5, 
                  std_max = 0.1, tolerance_ratio = 2, delta_Tmax_ratio = 3, verbose = True):
    """
    Implementation of the extended Discovery Algorithm designed by Julie Soulas U{https://hal.archives-ouvertes.fr/tel-01356217/}
    
    :param df : Starting dataframe, date[datetime] as index and 1 column named "activity"
    :param Tep : [in Minutes] Maximal time interval between events in an episode occurrence. Should correspond to the maximal duration of the ADLs.
    :param support_treshold : [greater than 1] Minimal number of occurrences of an episode, for that episode to be considered as frequent.
    :param accuracy_min : [between 0 and 1] Minimal accuracy for a periodicity description to be considered as interesting, and thus factorized
    :param std_max : Standard deviation max for a description (ratio of the period)
    :param tolerance_ratio : [greater than 0] An event expected to happen at time t (with standard deviation sigma) occurs as expected if it occurs in the interval [t - tolerance_ratio*sigma, t + tolerance_ratio*sigma]
    :param delta_Tmax : If there is a gap > delta_Tmax between two occurrences of an episode, theoccurrences before and after the gap are split (different validity intervals).
    :return The compressed dataset
    """
    compressed = True
    
    comp_iter = 0
    while compressed :
        comp_iter += 1
        
        if verbose:
            print("\n")
            print("###############################")
            print("#    COMPRESSION N°%d START   #" % comp_iter)
            print("##############################")
            print("\n")
            
        compressed = False
        
        if verbose :
            print("  Finding frequent episodes candidates...  ".center(50, '*'))
        
        frequent_episodes = FP_growth.find_frequent_episodes(data, support_min, Tep)
        
        periodicities = []
        
        for episode in frequent_episodes :
            #Build the description of the episode if interesting enough (Accuracy > Accuracy_min)
            delta_t, descr = candidate_study.periodicity_search(data, episode, delta_Tmax_ratio, 
                                                                support_min, std_max, accuracy_min, 
                                                                tolerance_ratio, Tep)
            periodicity = {
                    'detla_t' : delta_t,
                    'descr' : descr
                    }
            periodicities.append(periodicity)
    
    return [], data



def pick_dataset(name) :
    dataset = None
    if name == 'toy':
        dataset = pd.read_csv("data/toy_dataset.txt", delimiter=';')
        date_format = '%Y-%d-%m %H:%M'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
        
    elif name == 'KA':
        dataset = pd.read_csv("data/KA_dataset.csv", delimiter=';')
        date_format = '%d-%b-%Y %H:%M:%S'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
        
    return dataset

if __name__ == "__main__":
    main()