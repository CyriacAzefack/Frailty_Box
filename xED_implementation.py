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
from dateutil.parser import parse

sys.path.append(os.path.abspath("./FP_Growth"))
import fp_growth




def xED_algorithm(df, Tep=60, support_treshold = 2, accuracy_min = 0.5, std_max = 0.1, tolerance_ratio = 2, candidate_periods = [dt.timedelta(days=1)]):
    """
    Implementation of the extended Discovery Algorithm designed by Julie Soulas U{https://hal.archives-ouvertes.fr/tel-01356217/}
    
    :param df : Starting dataframe, date[datetime] as index and 1 column named "activity"
    :param Tep : [in Minutes] Maximal time interval between events in an episode occurrence. Should correspond to the maximal duration of the ADLs.
    :param support_treshold : [greater than 1] Minimal number of occurrences of an episode, for that episode to be considered as frequent.
    :param accuracy_min : [between 0 and 1] Minimal accuracy for a periodicity description to be considered as interesting, and thus factorized
    :param std_max : Standard deviation max for a description
    :param tolerance_ratio : [greater than 0] An event expected to happen at time t (with standard deviation sigma) occurs as expected if it occurs in the interval [t - tolerance_ratio*sigma, t + tolerance_ratio*sigma]
    :param candidate_periods : a list of periods to check for habits periodicity
    :return The compressed dataset
    """
    compressed = True
    
    final_periodicities = []
    
    while compressed:
        
        compressed = False
        
        df, transactions = extract_transactions(df, Tep)
           
        frequent_episodes = fp_growth.find_frequent_patterns(transactions, support_treshold)
        
        if not frequent_episodes:
            break
        
        periodicities = {}
        
        for episode in frequent_episodes.keys():
             result = build_description(df, episode, candidate_periods, 
                                                           support_treshold, std_max, tolerance_ratio, accuracy_min)
             if result is None:
                 periodicities.pop(episode, None)
             else:
                 periodicities[episode] = result
                 
        for episode, periodicity in periodicities.items():
            if periodicity is not None:
                periodicity["nb_factorized_events"], periodicity["factorized_events_id"], periodicity["missing_events"] = find_factorized_events(df, episode, periodicity, tolerance_ratio)
        
        
        sorted_episodes_by_compression = sorted(periodicities.keys(), key=lambda x: periodicities[x]["nb_factorized_events"], reverse=True)
        
        factorized_events_id = []
        
        while set(factorized_events_id).isdisjoint(periodicities[sorted_episodes_by_compression[0]]["factorized_events_id"]) and (periodicities[sorted_episodes_by_compression[0]]["nb_factorized_events"] > 0) :
            
            df = df[["date", "activity"]]
            
            #Factorize data
            periodicities[sorted_episodes_by_compression[0]]["episode"] = sorted_episodes_by_compression[0]
            
            final_periodicities.append(periodicities[sorted_episodes_by_compression[0]])
            
            #add missing events
            for event in periodicities[sorted_episodes_by_compression[0]]["missing_events"] :
                df.loc[max(df.index) + 1] = [event["date"], event["activity"]]
            
            #drop described events
            drops_ids = periodicities[sorted_episodes_by_compression[0]]["factorized_events_id"]
            df.drop(drops_ids, inplace=True)
            
           
            factorized_events_id += drops_ids
            
            
            if (len(sorted_episodes_by_compression) < 2 ):
                break
            #Remove sorted_episodes_by_compression[0] from the list
            sorted_episodes_by_compression = sorted_episodes_by_compression[1:]
            
            
            compressed = True
    

    return final_periodicities, df
    


def extract_transactions(df, Tep = 30 ) :
    """
    Divide the dataframe in transactions subset with max lenth of Tep
    :param Tep: in minutes
    :return dataset [with a new 'trans_id' column], transactions
    """
    
    tep_hours = int(math.floor(Tep/60))
    tep_minutes = Tep - tep_hours*60
    start_time = min(df.date) #Start time of the dataset

    df['trans_id'] = 0

    current_start_time = start_time
    current_trans_id = 0

    transactions = []
    while True:
        current_trans_id += 1
        current_end_time = current_start_time + dt.timedelta(hours=tep_hours, minutes=tep_minutes)
        transactions.append(list(df.loc[(df.date >= current_start_time) & (df.date < current_end_time)].activity.values))
        df.loc[(df.date >= current_start_time) & (df.date < current_end_time), 'trans_id'] = current_trans_id
        
        if len(df.loc[df.date > current_end_time]) > 0 :
            current_start_time =  min(df.loc[df.date > current_end_time, "date"])
        else :
            break
    
    return df, transactions

def find_factorized_events(df, episode, periodicity, tolerance_ratio) :
    """
    Compute the number of factorized events 
    """
    
    dataset_duration = (max(df.date) - min(df.date))/np.timedelta64(1, 's') #in seconds
    nb_comp = len(periodicity["numeric"])
    nb_del_exp = nb_comp * math.ceil(dataset_duration / periodicity["period"].total_seconds()) * len(episode)
    
    missing_events = []
    
    df_copy = df[df.activity.isin(episode)].copy(deep=True)
    
    #Compute the start time of the occurence
    for trans_id in df_copy.trans_id.unique():
        df_copy.loc[df_copy.trans_id == trans_id, "occ_start_time"] = min(df_copy.loc[df_copy.trans_id == trans_id, "date"])
     
        
    df_copy.loc[:, "rel_start_time"] = df_copy["occ_start_time"].apply(lambda x : modulo_datetime(x.to_pydatetime(), periodicity["period"]))
    
    df_copy.loc[:, "expected"] = df_copy.rel_start_time.apply(lambda x : occurence_expected(x.total_seconds(), periodicity["numeric"], tolerance_ratio))
    
    #check for missing events
    for trans_id in df_copy.loc[df_copy.expected==True, "trans_id"].unique():
        events = list(df_copy.loc[(df_copy.expected == True) & (df_copy.trans_id == trans_id), "activity"])
        
        miss_events = set(list(episode)).symmetric_difference(set(events))
        
        if len(miss_events) == 0:
            continue
        #Find component mean
        dist = sys.maxsize
        mean_time = None
        rel_occ_time = df_copy.loc[(df_copy.activity == events[0]) & (df_copy.trans_id == trans_id), "rel_start_time"].astype('timedelta64[s]').values[0]
        occ_time = df_copy.loc[(df_copy.activity == events[0]) & (df_copy.trans_id == trans_id), "occ_start_time"].values[0]
        for mean in periodicity["numeric"].keys():
            if(abs(rel_occ_time - mean) < dist):
                dist = abs(rel_occ_time - mean)
                mean_time = mean
        
        occ_time = pd.Timestamp(occ_time).to_pydatetime()
        start_time = occ_time - dt.timedelta(seconds=rel_occ_time )+ dt.timedelta(seconds=mean_time)
        for event in miss_events:
            missing_events.append({"activity" : "[MISSING] " + event, "date" : start_time})    
    
    return 2*df_copy["expected"].sum() - nb_del_exp, list(df_copy.loc[df_copy.expected == True].index), missing_events
    
    
    
def build_description(df, episode, candidate_periods, support_treshold, std_max, tolerance_ratio, accuracy_min):
    """
    Build the best description for all the episodes
    :param df : Starting dataframe, date[datetime] as index, columns named "activity" and "trans_id" (for all the transactions)
    :param episode : frequent episode
    :param support_treshold : [greater than 1] Minimal number of occurrences of an episode, for that episode to be considered as frequent.
    :param std_max : standard deviation max for a period
    :param accuracy_min : [between 0 and 1] Minimal accuracy for a periodicity description to be considered as interesting, and thus factorized
    :param tolerance_ratio : [greater than 0] An event expected to happen at time t (with standard deviation sigma) occurs as expected if it occurs in the interval [t - tolerance_ratio*sigma, t + tolerance_ratio*sigma]
    :return A dataset of episodes with their best description
    """
    
    dataset_duration = (max(df.date) - min(df.date))/np.timedelta64(1, 's') #in seconds
    
    description = {
            "numeric": {}, #all the components of the description. mean_time as key and std_time as value
            "readable": {}, #A readable string for the description
            "accuracy" : 0, #Accuracy of the description
            "delta_t" : None, # Time duration when the description is valid
            }
    
    occurences = find_occurences(df, episode)
    
    for period in candidate_periods:
        delta_T_max = 3*period #Gap maximum between two consecutives occurences to be in the same group
        occ_period = occurences.copy(deep=True)
        occ_period.sort_values(["start_time", "end_time"], ascending=True, inplace=True)
        
        #Compute time between occurences
        occ_period.loc[:, "time_since_last_occ"] = occ_period['start_time'] - occ_period['end_time'].shift(1)
        
        #First row 'time_since_last_occ' is NaT so we replace by a duration of '0'
        occ_period.fillna(0, inplace=True)
        
        #Compute the relative start time
        occ_period.loc[:, "rel_start_time"] = occ_period["start_time"].apply(lambda x : modulo_datetime(x.to_pydatetime(), period))
       
        #Spit the occurrences in groups
        group_gap_bounds = [df.date.min(), df.date.max()]
        # [min_time, insertion of groups bound, max_time]
        group_gap_bounds[1:1] = sorted(list(occ_period[occ_period.time_since_last_occ > delta_T_max]['start_time']))
        
        for group_index in range(len(group_gap_bounds)-1):
            group_filter = (occ_period.start_time >= group_gap_bounds[group_index]) & (occ_period.start_time <= group_gap_bounds[group_index+1])
            occ_period.loc[group_filter, 'group_id'] = group_index+1

            #Find the CLUSTERS
            data_points = occ_period.loc[group_filter, "rel_start_time"].astype('timedelta64[s]').values
            if len(data_points) == 0:
                continue
            
            data_points = data_points.reshape(-1, 1)
            db = DBSCAN(eps=std_max*period.total_seconds(), min_samples=support_treshold).fit(data_points)
           
            N_comp = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0) # Noisy samples are given the label -1.
            
            if N_comp == 0:
                continue
            gmm = GaussianMixture(n_components = N_comp, covariance_type='full', init_params='random')
            gmm.fit(data_points)

            gmm_descr = {} # time_mean as key and std as value
            gmm_descr_str = {}
            
            for i in range(len(gmm.means_)):
                gmm_descr_str[str(dt.timedelta(seconds=gmm.means_[i][0]))] = str(dt.timedelta(
                        seconds=np.sqrt(gmm.covariances_)[i][0][0]))
                gmm_descr[gmm.means_[i][0]] = np.sqrt(gmm.covariances_)[i][0][0]
                
            
            # Tag the expected occurences
            occ_period.loc[group_filter, "expected"] = occ_period.loc[group_filter, "rel_start_time"].apply(
                    lambda x : occurence_expected(x.total_seconds(), gmm_descr, tolerance_ratio))
            
            #Compute description accuracy
            N_occ_exp = N_comp * math.ceil(dataset_duration / period.total_seconds())
            
            N_occ_exp_and_occ = len(occ_period[group_filter & occ_period.expected == True])
            accuracy = N_occ_exp_and_occ / N_occ_exp
            
            
                  
           
            if(accuracy >= accuracy_min) & (accuracy > description["accuracy"]):
                
                description["period"] = period
                description["accuracy"] = accuracy
                description["numeric"] = gmm_descr
                description["readable"] = gmm_descr_str
                description["delta_t"] = max(occ_period.loc[group_filter & occ_period.expected==True, "start_time"]) - min(list(occ_period.loc[group_filter & occ_period.expected==True, "start_time"]))
                
    
    if description['accuracy'] == 0:
        return None
    return description

def find_occurences(df, episode) :
    """
    return a dataframe of occurences of the episode in the dataframe df
    """
    
    occurences = pd.DataFrame(columns = ["trans_id", "start_time", "end_time"])
    
    occurences_trans_id_list = list(df.trans_id.unique())
    for item in episode :
        occurences_trans_id_list = list(set(occurences_trans_id_list).intersection(list(
                df[df.activity == item].trans_id.unique())))
        
    for id in occurences_trans_id_list :
        start_time = min(df[(df.trans_id == id) & (df.activity.isin(episode))].date) #First event of the episode
        end_time = max(df[(df.trans_id == id) & (df.activity.isin(episode))].date) #Last event of the episode
        
        occurences.loc[len(occurences)] = [id, start_time, end_time]
    
    
    return occurences

def modulo_datetime(date, period):
    """
    Compute the relative date in the period
    :param date: datetime.datetime object
    :param period : datetime.timedelta
    """
    seconds = int((date - dt.datetime.min).total_seconds())
    remainder = dt.timedelta(
        seconds = seconds % period.total_seconds(),
        microseconds = date.microsecond,
    )
    return remainder


def occurence_expected(start_time, description, tolerance_ratio = 1):
    """
    Return True if an occurence is expected to occur according to the description
    :param start_time : occurence start time (in seconds)
    :param description : description dict with mean times as keys and standard deviations as values
    :param tolerance_ratio : tolerable distance to the mean
    """
    for mean_time in description.keys():
        if abs(start_time - mean_time) < tolerance_ratio*description[mean_time]:
            return True
        
    return False


if __name__ == "__main__":
    ########################################
    # DATA PREPROCESSING
    ########################################
    
    """
    The dataframe should have 1 index (date as datetime) and 1 feature (activity)
    """
    dataset = pd.rdf = pd.read_csv("kaData.txt", delimiter=';')
    date_format = '%d-%b-%Y %H:%M:%S'
#    dataset = pd.rdf = pd.read_csv("toy_dataset.txt", delimiter=';')
#    date_format = '%Y-%d-%m %H:%M'
    
    dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
    #dataset = dataset.set_index('date')
    results, df = xED_algorithm(df=dataset, Tep=60, support_treshold=3, tolerance_ratio=2)
    
    dat = pd.DataFrame(columns= ["episode", "readable", "period", "accuracy", "delta_t", "nb_factorized_events"])
    for v in results :
        dat.loc[len(dat)] = [v['episode'], v["readable"], v["period"], v["accuracy"], v["delta_t"], v["nb_factorized_events"]]



    

