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
import seaborn as sns

sys.path.append(os.path.abspath("./FP_Growth"))
import fp_growth
import pyfpgrowth





def xED_algorithm(df, Tep=60, support_treshold = 2, accuracy_min = 0.5, std_max = 0.1, tolerance_ratio = 2, candidate_periods = [dt.timedelta(days=1)]):
    """
    Implementation of the extended Discovery Algorithm designed by Julie Soulas U{https://hal.archives-ouvertes.fr/tel-01356217/}
    
    :param df : Starting dataframe, date[datetime] as index and 1 column named "activity"
    :param Tep : [in Minutes] Maximal time interval between events in an episode occurrence. Should correspond to the maximal duration of the ADLs.
    :param support_treshold : [greater than 1] Minimal number of occurrences of an episode, for that episode to be considered as frequent.
    :param accuracy_min : [between 0 and 1] Minimal accuracy for a periodicity description to be considered as interesting, and thus factorized
    :param std_max : Standard deviation max for a description (ratio of the period)
    :param tolerance_ratio : [greater than 0] An event expected to happen at time t (with standard deviation sigma) occurs as expected if it occurs in the interval [t - tolerance_ratio*sigma, t + tolerance_ratio*sigma]
    :param candidate_periods : a list of periods to check for habits periodicity
    :return The compressed dataset
    """
    
    
    compressed = True
    
    final_periodicities = []
    
    comp_iter = 0
    while compressed:
        comp_iter += 1
        compressed = False
        
        transactions = extract_transactions(df, Tep)
        
        print("\n")
        print("###############################")
        print("#    COMPRESSION N°%d START   #" % comp_iter)
        print("##############################")
        print("\n")
        
        
        print("Finding frequent episodes candidates...".center(100, '*'))
        #frequent_episodes = fp_growth.find_frequent_patterns(transactions, support_treshold)
        frequent_episodes = pyfpgrowth.find_frequent_patterns(transactions, support_treshold)
        
        frequent_episodes = [("got to bed end", "use toilet start", "use toilet end", )]
        
        print(len(frequent_episodes), "episodes found !!")
        
        if not frequent_episodes:
            break
        
        periodicities = {}
        
        
        print("Building candidates episodes periodicities...".center(100, '*'))
        i = 0
        for episode in frequent_episodes:
            i += 1
            
            result = build_description(df, episode, Tep, candidate_periods, 
                                                           support_treshold, std_max, tolerance_ratio, accuracy_min)   
            if result is not None:
                print("\nInteresting periodicity found for the episode", episode)
                periodicities[episode] = result
                
            sys.stdout.write("\r%.2f %% of episodes treated!!" % (100*i/len(frequent_episodes)))
            sys.stdout.flush()
        sys.stdout.write("\n")
        
        
        for episode, periodicity in periodicities.items():
            if periodicity is not None:
                periodicity["nb_factorized_events"], periodicity["factorized_events_id"], periodicity["missing_events"] = find_factorized_events(df, episode, periodicity, tolerance_ratio, Tep)
        
        print("Sorting the candidate frequent episodes by compression power...".center(100, '*'))
        sorted_episodes_by_compression = sorted(periodicities.keys(), key=lambda x: periodicities[x]["nb_factorized_events"], reverse=True)
        
        factorized_events_id = []
        
        print("Dataset Rewriting".center(100, '*'))
        
        while set(factorized_events_id).isdisjoint(periodicities[sorted_episodes_by_compression[0]]["factorized_events_id"]) and (periodicities[sorted_episodes_by_compression[0]]["nb_factorized_events"] > 0) :
            
            df = df[["date", "activity"]]
            
            print("Factorize data by the episode", sorted_episodes_by_compression[0])
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
            
            
            if len(sorted_episodes_by_compression) < 2 :
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
    
    df.drop(['trans_id'], axis=1, inplace=True)
    
    return transactions

def find_factorized_events(df, episode, periodicity, tolerance_ratio, Tep) :
    """
    Compute the number of factorized events 
    """
    
    nb_deleted_events_expected = periodicity["nb_occurrences_expected"] * len(episode)
    
    missing_events = []
    factorized_events_id = []
    
    df_copy = df[df.activity.isin(episode)].copy(deep=True)
    df_copy.sort_values(["date"], ascending = True, inplace = True)
    
    
    occurences = find_occurences(df_copy, episode, Tep)
    occurences.loc[:, "rel_start_time"] = occurences["start_time"].apply(lambda x : modulo_datetime(x.to_pydatetime(), periodicity["period"]))
    occurences.loc[:, "expected"] = occurences.loc[:, "rel_start_time"].apply(
                    lambda x : occurence_expected(x.total_seconds(), periodicity["numeric"], tolerance_ratio))
       
    #Drop the unexpected occurrences
    occurences = pd.DataFrame(occurences.values[occurences.expected == True], columns=occurences.columns)
    
    for index, occurence in occurences.iterrows():
        events_id = df_copy.loc[(df_copy.date >= occurence["start_time"]) &
                                (df_copy.date <= occurence["end_time"])].index
        factorized_events_id = factorized_events_id + list(events_id)
    
    #drop the already analysed events
    df_copy.drop(factorized_events_id, inplace=True)
    
    
    #TRRACK MISING OCCURENCES
    
    missing_occurences = find_missing_occurences(episode, occurences, periodicity, tolerance_ratio)
    
    for index, missing_occurence in missing_occurences.iterrows():
        occured_events_filter = (df_copy.date >= missing_occurence["min_time"]) & (df_copy.date <= missing_occurence["max_time"])
        occured_events = list(df_copy.loc[occured_events_filter, "activity"].values)
        
        for event in occured_events:
            factorized_events_id.append(df_copy.loc[occured_events_filter & (df_copy.activity == event)].index[0])
            
        non_occurred_events = list(set(episode).symmetric_difference(set(occured_events)))
        
        for event in non_occurred_events:
            missing_events.append({"activity" : "[MISSING] " + event, "date" : missing_occurence["start_time"]})
    
    compression_power = len(factorized_events_id) - len(missing_events)
    return compression_power, factorized_events_id, missing_events
        
def find_missing_occurences(episode, occurences, periodicity, tolerance_ratio):
    """
    find all the missing occurrences of the episode
    """
    
    missing_occurences = pd.DataFrame(columns = ["min_time", "max_time", "start_time"])
    avg_occurence_duration = np.mean(occurences['end_time'] - occurences['start_time'])
    
    for mean_time, std_time in periodicity['numeric'].items():
        
        #Find all the occurrences of the current component
        occurences.loc[:, "expected"] = occurences.loc[:, "rel_start_time"].apply(
                    lambda x : occurence_expected(x.total_seconds(), dict([(mean_time, std_time)]), tolerance_ratio))
        comp_occurences = occurences[occurences.expected == True].copy(deep=True)
        
        #Compute time between occurences
        comp_occurences.loc[:, "time_since_last_occ"] = comp_occurences['start_time'] - comp_occurences['end_time'].shift(1)
        
        #First row 'time_since_last_occ' is NaT so we replace by a duration of '0'
        comp_occurences.fillna(0, inplace=True)
        
        #if the time_since_last_occ is more than a period, we might have a missing occurence in between
        comp_occurences = comp_occurences.drop(comp_occurences[comp_occurences['time_since_last_occ'] < periodicity['period']].index)        
        
        for index, comp_occurence in comp_occurences.iterrows():
            search_end_time = comp_occurence['start_time']
            search_start_time = comp_occurence['start_time'] - comp_occurence['time_since_last_occ']
            
            #Now we search for all the potentials occurences in this timespan
            current_start_time = search_start_time + periodicity["period"]
            
            #Since search_start_time is the end of a previous occurence
    
            while True:
                current_end_time = current_start_time + periodicity["period"]
                
                rel_current_start_time = modulo_datetime(current_start_time.to_pydatetime(), periodicity["period"]).total_seconds()
                

                if mean_time >= rel_current_start_time:
                    comp_mean_date = current_start_time + dt.timedelta(seconds=(mean_time - rel_current_start_time))
                else :
                    comp_mean_date = current_start_time + dt.timedelta(seconds = periodicity["period"].total_seconds() + mean_time - rel_current_start_time)
                    
                comp_start_date = comp_mean_date - dt.timedelta(seconds = tolerance_ratio * std_time)
                comp_end_date = comp_mean_date + dt.timedelta(seconds = tolerance_ratio * std_time)
                
                current_start_time = current_end_time
                
                if current_start_time > search_end_time:
                    break

                               
                missing_occurences.loc[len(missing_occurences)] = [comp_start_date, comp_end_date + avg_occurence_duration, comp_mean_date]
                
    return missing_occurences
            
        
        
        
    
    
def build_description(df, episode, Tep, candidate_periods, support_treshold, std_max, tolerance_ratio, accuracy_min):
    """
    Build the best description for all the episodes
    :param df : Starting dataframe, date[datetime] as index, columns named "activity" and "trans_id" (for all the transactions)
    :param episode : frequent episode
    :param Tep : [in Minutes] Maximal time interval between events in an episode occurrence. Should correspond to the maximal duration of the ADLs.
    :param support_treshold : [greater than 1] Minimal number of occurrences of an episode, for that episode to be considered as frequent.
    :param std_max : standard deviation max for a period
    :param accuracy_min : [between 0 and 1] Minimal accuracy for a periodicity description to be considered as interesting, and thus factorized
    :param tolerance_ratio : [greater than 0] An event expected to happen at time t (with standard deviation sigma) occurs as expected if it occurs in the interval [t - tolerance_ratio*sigma, t + tolerance_ratio*sigma]
    :return A dataset of episodes with their best description
    """
    
    

    
    description = {
            "numeric": {}, #all the components of the description. mean_time as key and std_time as value
            "readable": {}, #A readable string for the description
            "accuracy" : 0, #Accuracy of the description
            "nb_occurrences_expected" : 0, #Number of episode occurences expected
            "delta_t" : None # Time duration when the description is valid
            }
    
    occurences = find_occurences(df, episode, Tep)
    

    
    df_start_time = min(df.date)
    df_end_time = max(df.date)
    
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

        #Display start_time hours histogram
        #sns.distplot(occ_period["start_time"].dt.hour, norm_hist=False, rug=False, bins=20, kde=False)
        
        
        #Spit the occurrences in groups
        group_gap_bounds = [df.date.min(), df.date.max()]
        # [min_time, insertion of groups bound, max_time]
        group_gap_bounds[1:1] = sorted(list(occ_period[occ_period.time_since_last_occ > delta_T_max]['start_time']))
        
        for group_index in range(len(group_gap_bounds)-1):
            group_filter = (occ_period.start_time >= group_gap_bounds[group_index]) & (occ_period.start_time < group_gap_bounds[group_index+1])
            occ_period.loc[group_filter, 'group_id'] = group_index+1

            #Find the CLUSTERS
            data_points = occ_period.loc[group_filter, "rel_start_time"].astype('timedelta64[s]').values
            if len(data_points) == 0:
                continue
            
            data_points = data_points.reshape(-1, 1)
            db = DBSCAN(eps=std_max*period.total_seconds()/2, min_samples=support_treshold).fit(data_points)
           
            N_comp = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0) # Noisy samples are given the label -1.
            
            if N_comp == 0:
                continue
            gmm = GaussianMixture(n_components = N_comp, tol=0.01, covariance_type='spherical', n_init=10)
            gmm.fit(data_points)
            
            gmm_descr = {} # time_mean as key and std as value
            gmm_descr_str = {}
            
            for i in range(len(gmm.means_)):                
                gmm_descr_str[str(dt.timedelta(seconds=gmm.means_[i][0]))] = str(dt.timedelta(
                        seconds=math.ceil(np.sqrt(gmm.covariances_)[i])))
                gmm_descr[gmm.means_[i][0]] = math.ceil(np.sqrt(gmm.covariances_[i]))
                
            
            # Tag the expected occurences
            occ_period.loc[group_filter, "expected"] = occ_period.loc[group_filter, "rel_start_time"].apply(
                    lambda x : occurence_expected(x.total_seconds(), gmm_descr, tolerance_ratio))
            
            #Compute description accuracy
            rel_df_start_time = modulo_datetime(df_start_time.to_pydatetime(), period).total_seconds()
            rel_df_end_time = modulo_datetime(df_end_time.to_pydatetime(), period).total_seconds()
            
            #Push to the first 
            nb_full_periods_dataset = (df_end_time - df_start_time)/np.timedelta64(1, 's') 
            nb_full_periods_dataset = -1 + math.floor((nb_full_periods_dataset - rel_df_end_time + rel_df_start_time)/ period.total_seconds()) 
            nb_occurrences_expected  = N_comp * nb_full_periods_dataset #but that's not all

                
            for mean_time in gmm_descr.keys():
                if mean_time >= rel_df_start_time:
                    nb_occurrences_expected += 1
                
                if mean_time <= rel_df_end_time:
                    nb_occurrences_expected += 1
            
            nb_occurences_observed = len(occ_period[group_filter & occ_period.expected == True])
            accuracy = nb_occurences_observed / nb_occurrences_expected
            
            
                  
           
#            if accuracy > 1:
#                raise ValueError('The accuray should not exceed 1.00 !!', episode, nb_occurrences_expected, nb_occurences_observed) 
            if(accuracy >= accuracy_min) & (accuracy > description["accuracy"]):
                
                description["period"] = period
                description["accuracy"] = accuracy
                description["numeric"] = gmm_descr
                description["readable"] = gmm_descr_str
                description["nb_occurrences_expected"] = nb_occurrences_expected
                description["delta_t"] = max(occ_period.loc[group_filter & occ_period.expected==True, "start_time"]) - min(list(occ_period.loc[group_filter & occ_period.expected==True, "start_time"]))
                
    
    if description['accuracy'] == 0:
        return None
    return description

def find_occurences(df, episode, Tep) :
    """
    return a dataframe of occurences of the episode in the dataframe df
    """
    
    occurences = pd.DataFrame(columns = ["start_time", "end_time"])
    
    tep_hours = int(math.floor(Tep/60))
    tep_minutes = Tep - tep_hours*60

    episode_df = df.loc[df.activity.isin(episode)]
    
    current_start_time = min(episode_df.date)
    
    
    while True:
        current_end_time = current_start_time + dt.timedelta(hours=tep_hours, minutes=tep_minutes)
        
        #Find the episode events in the span time
        events_occured = set(episode_df.loc[(episode_df.date >= current_start_time) &
                                        (episode_df.date < current_end_time), "activity"].values)
        
        if events_occured == set(episode): #all the episode events occured in that time span

            current_end_time = max(episode_df.loc[(episode_df.date >= current_start_time) &
                                        (episode_df.date < current_end_time), "date"].values)
            occurences.loc[len(occurences)] = [current_start_time, current_end_time]
            
        
        if len(episode_df.loc[episode_df.date > current_end_time]) == 0:
            break
        
        current_start_time =  min(episode_df.loc[episode_df.date > current_end_time, "date"])
            
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
        if abs(start_time - mean_time) <= tolerance_ratio*description[mean_time]:
            return True
        
    return False


if __name__ == "__main__":
    ########################################
    # DATA PREPROCESSING
    ########################################
    
    """
    The dataframe should have 1 index (date as datetime) and 1 feature (activity)
    """
    dataset = pd.read_csv("KA_dataset.csv", delimiter=';')
    date_format = '%d-%b-%Y %H:%M:%S'
#    dataset = pd.read_csv("toy_dataset.txt", delimiter=';')
#    date_format = '%Y-%d-%m %H:%M'
    
    dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
    #dataset = dataset.set_index('date')
    results, df = xED_algorithm(df=dataset, Tep=30, support_treshold=2, 
                                tolerance_ratio=2, 
                                candidate_periods = [dt.timedelta(days=1)])
    
    dat = pd.DataFrame(columns= ["episode", "readable", "period", "accuracy", "delta_t", "nb_factorized_events"])
    for v in results :
        dat.loc[len(dat)] = [v['episode'], v["readable"], v["period"], v["accuracy"], v["delta_t"], v["nb_factorized_events"]]



    

