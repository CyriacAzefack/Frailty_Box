# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:43:23 2017

@author: cyriac.azefack
"""
import sys 
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("./FP_Growth"))
import fp_growth

########################################
# DATA PREPROCESSING
########################################

"""
The dataframe should have 1 index (date as datetime) and 1 feature (activity)
"""
dataset = pd.rdf = pd.read_csv("toy_dataset.txt", delimiter=';')
date_format = '%Y-%d-%m %H:%M'
dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
dataset = dataset.set_index('date')

def xED_algorithm(data, Tep=30, support_treshold = 2, accuracy_min = 0.5, tolerance_ratio = 2, periods = [dt.timedelta(days=1)]):
    """
    Implementation of the extended Discovery Algorithm designed by Julie Soulas U{https://hal.archives-ouvertes.fr/tel-01356217/}
    
    :param data : Starting dataframe, date[datetime] as index and 1 column named "activity"
    :param Tep : [in Minutes] Maximal time interval between events in an episode occurrence. Should correspond to the maximal duration of the ADLs.
    :param support_treshold : [greater than 1] Minimal number of occurrences of an episode, for that episode to be considered as frequent.
    :param accuracy_min : [between 0 and 1] Minimal accuracy for a periodicity description to be considered as interesting, and thus factorized
    :param tolerance_ratio : [greater than 0] An event expected to happen at time t (with standard deviation sigma) occurs as expected if it occurs in the interval [t - tolerance_ratio*sigma, t + tolerance_ratio*sigma]
    :param periosd : a list of periods for the habits periodicity
    :return The compressed dataset
    """
    compressed = True
    
    
    while compressed:
        
        df, transactions = extract_transactions(data, Tep)
        
        frequent_episodes = fp_growth.find_frequent_patterns(transactions, support_treshold)
        
        frequent_episodes_description = build_descriptions(df, frequent_episodes, periods, 
                                                           support_treshold, tolerance_ratio, accuracy_min)
        
        print(frequent_episodes_description)
        return 
    


def extract_transactions(df, Tep ) :
    """
    Divide the dataframe in transactions subset with max lenth of Tep
    :param Tep: in minutes
    :return dataset [with a new 'trans_id' column], transactions
    """
    
    tep_hours = int(math.floor(Tep/60))
    tep_minutes = Tep - tep_hours*60
    start_time = min(df.index) #Start time of the dataset

    df['trans_id'] = 0

    current_start_time = start_time
    current_trans_id = 0

    transactions = []
    while True:
        current_trans_id += 1
        current_end_time = current_start_time + dt.timedelta(hours=tep_hours, minutes=tep_minutes)
        transactions.append(list(df.loc[(df.index >= current_start_time) & (df.index < current_end_time)].activity.values))
        df.loc[(df.index >= current_start_time) & (df.index < current_end_time), 'trans_id'] = current_trans_id

        if len(df.loc[df.index > current_end_time]) > 0 :
            current_start_time =  df.loc[df.index > current_end_time].index[0]
        else :
            break
    
    return df, transactions

def build_descriptions(df, episodes, candidate_periods, support_treshold, tolerance_ratio, accuracy_min):
    """
    Build the best description for all the episodes
    :param df : Starting dataframe, date[datetime] as index, columns named "activity" and "trans_id" (for all the transactions)
    :param episodes : a list of the frequent episodes
    :param support_treshold : [greater than 1] Minimal number of occurrences of an episode, for that episode to be considered as frequent.
    :param accuracy_min : [between 0 and 1] Minimal accuracy for a periodicity description to be considered as interesting, and thus factorized
    :param tolerance_ratio : [greater than 0] An event expected to happen at time t (with standard deviation sigma) occurs as expected if it occurs in the interval [t - tolerance_ratio*sigma, t + tolerance_ratio*sigma]
    :return A dataset of episodes with their best description
    """
    
    dataset_duration = (max(df.index) - min(df.index))/np.timedelta64(1, 's') #in seconds