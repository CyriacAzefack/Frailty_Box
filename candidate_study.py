# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 09:41:33 2018

@author: cyriac.azefack
"""
import datetime as dt
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

import xED_algorithm


def main():
    data = xED_algorithm.pick_dataset('toy')
    episodes = []
    # episodes.append(["go to bed end", "use toilet start", "use toilet end"])
    # episodes.append(["take shower start", "take shower end", "leave house start"])
    # episodes.append(["use toilet start", "use toilet end", "go to bed start"])
    # episodes.append(["prepare Breakfast start", "prepare Breakfast end"])
    # episodes.append(["leave house end"])
    #episodes.append(['brush teeth start', 'go to bed start'])

    episodes.append(["breakfast"])
    # episode = ["kitchen", "lunch", "coffee"]

    with open('output_result.csv', 'w') as file:
        file.truncate()
        file.write("Episode;period;description;accuracy;start date;end date\n")
        for episode in episodes:
            descr = periodicity_search(data, episode, plot_graphs=True)

            nat = translate_description(descr)
            line = str(episode) + ";"
            line += str(nat["period"]) + ";"
            line += str(nat["description"]) + ";"
            line += str(nat["accuracy"]).replace('.', ',') + ";"
            line += str(nat["delta_t"][0]) + ";"
            line += str(nat["delta_t"][1]) + ";"
            line += "\n"

            file.write(line)
            print("Episode ", episode)
            print(nat)


def periodicity_search(data, episode, delta_Tmax_ratio=3, support_min=3, std_max=0.1,
                       accuracy_min=0.5, tolerance_ratio=2, Tep=30, plot_graphs=False):
    
    """
    Find the best description of an episode if it exists
    
    return delta_T : 
    """

    def is_occurence_expected(relative_date, GMM_descr, period):
        for mu, sigma in GMM_descr.items():
            if abs(relative_date - mu) <= tolerance_ratio * sigma:  # Normal cases
                return mu

            # Handle the bord effects cases
            lower_limit = mu - tolerance_ratio * sigma
            upper_limit = mu + tolerance_ratio * sigma

            mu2 = mu
            if lower_limit < 0:  # Midnight-morning issue (early in the period)
                mu2 = mu + period.total_seconds()
            elif upper_limit > period.total_seconds():
                mu2 = mu - period.total_seconds()

            if abs(relative_date - mu2) <= tolerance_ratio * sigma:  # Normal cases
                return mu2
            
        
        return None

    candidate_periods = [dt.timedelta(days=1), ]
    
    #Pick the episode events from the input sequence
    data = data.loc[data.activity.isin(episode)].copy()
    
    #find the episode occurences
    occurrences = find_occurrences(data, episode, Tep)
    
    
    best_descr = None
    
    best_accuracy = 0
    
    for T in candidate_periods:
        delta_Tmax = delta_Tmax_ratio * T
        period_occ = occurrences.copy()
        
        # Compute intervals between occurrences
        period_occ.loc[:, "time_since_last_occ"] = period_occ.date - period_occ.date.shift(1)
        
        #First row 'time_since_last_occ' is NaT so we replace by a duration of '0'
        period_occ.fillna(0, inplace=True)
        
        #Compute relative dates
        period_occ.loc[:, "relative_date"] = period_occ.date.apply(
                    lambda x : modulo_datetime(x.to_pydatetime(), T))
        
        #Display relative times histogram

        if plot_graphs:
            plt.figure()
            plt.title(episode)
            sns.distplot(period_occ.relative_date, norm_hist=False, rug=False, kde=True)
        
        #Spit the occurrences in groups
        group_gap_bounds = [data.date.min(), data.date.max()+dt.timedelta(seconds=10)]
        
        group_gap_bounds[1:1] = sorted(list(period_occ[period_occ.time_since_last_occ > delta_Tmax].date))
        
        for group_index in range(len(group_gap_bounds) - 1) :
            group_start_time = group_gap_bounds[group_index]
            group_end_time = group_gap_bounds[group_index + 1]
            
            group_occurrences = period_occ.loc[(period_occ.date >= group_start_time) & 
                                              (period_occ.date < group_end_time)].copy()
            
            if len(group_occurrences) == 0:
                continue
            
            group_start_time = group_occurrences.date.min().to_pydatetime()
            group_end_time = group_occurrences.date.max().to_pydatetime()
            
                       
            data_points = group_occurrences["relative_date"].values.reshape(-1, 1)
            #if no data then switch to the next group
            if len(data_points) == 0 :
                continue
            
            # For midnight-morning issue
            
            data_points_2 = [x + T.total_seconds() for x in data_points]
            
            big_data_points = np.asarray(list(data_points) + list(data_points_2)).reshape(-1, 1)

            Nb_clusters, interesting_points = find_number_clusters(big_data_points, eps = std_max*T.total_seconds(), min_samples = support_min)

            #if no clusters found then switch to the next group
            if Nb_clusters == 0:
                continue
            # Display points
#            if plot_graphs:
#                sns.distplot(interesting_points, norm_hist=False, rug=False, kde=True, bins=10)

            GMM = GaussianMixture(n_components=Nb_clusters, covariance_type='spherical', n_init=10)
            GMM.fit(interesting_points)
            
            GMM_descr = {} # mean_time (in seconds) as key and std_duration (in seconds) as value
            
            for i in range(len(GMM.means_)):
                mu = GMM.means_[i][0]
                sigma = math.ceil(np.sqrt(GMM.covariances_[0]))
                
                lower_limit = mu - tolerance_ratio*sigma
                upper_limit = mu + tolerance_ratio*sigma
                
                if (lower_limit < 0) or (lower_limit > T.total_seconds()) :
                    continue

                mu = mu % T.total_seconds()
                GMM_descr[mu] = sigma

                if plot_graphs:
                    lower_limit = mu - tolerance_ratio * sigma
                    upper_limit = mu + tolerance_ratio * sigma
                    # Plot the interval
                    c = np.random.rand(3, )
                    plt.plot([0, lower_limit], [lower_limit, 0], linewidth=2, color=c)
                    plt.plot([0, upper_limit], [upper_limit, 0], linewidth=2, color=c)

            #Compute the description accuracy
            
            #First, compute the number of full periods between "group_start_time" and "group_end_time"
            group_relative_start_time = modulo_datetime(group_start_time, T)
            group_relative_end_time = modulo_datetime(group_end_time, T)
            start_first_period = group_start_time - dt.timedelta(seconds=group_relative_start_time) + T
            end_last_period = group_end_time - dt.timedelta(seconds=group_relative_end_time)
            Nb_periods = (end_last_period - start_first_period).total_seconds()/T.total_seconds()
            
            #Now the bord effects
            bord_effects_expected_occ = 0
            if is_occurence_expected(group_relative_start_time, GMM_descr, T) is not None :
                bord_effects_expected_occ += 1
                
            if is_occurence_expected(group_relative_end_time, GMM_descr, T) is not None :
                bord_effects_expected_occ += 1
                
            for mean_time, std_time in GMM_descr.items():

                lower_limit = mean_time - tolerance_ratio * std_time
                upper_limit = mean_time + tolerance_ratio * std_time

                if group_relative_start_time < lower_limit:
                    bord_effects_expected_occ += 1

                if group_relative_end_time > upper_limit:
                    bord_effects_expected_occ += 1
               
            
            Nb_comp = len(GMM_descr)
            
            #Number of occurrences expected
            Nb_expected_occurrences = Nb_periods * Nb_comp + bord_effects_expected_occ

            # "Expected" is the relative mean time of the component where an occurrence happen           
            group_occurrences["expected"] = group_occurrences["relative_date"].apply(
                lambda x: is_occurence_expected(x, GMM_descr, T))

            # "diff_mean_time" is the time distance from the occurrence to the relative mean time of the component where an occurrence happen  
            group_occurrences["diff_mean_time"] = abs(group_occurrences["relative_date"] - group_occurrences["expected"])
            
            
            group_occurrences.fillna(0, inplace=True)

            # "component_absolute_mean_time" is the absolute date of the component where an occurrence happen
            group_occurrences["component_absolute_mean_time"] = group_occurrences.apply(
                    lambda row : relative2absolute_date(row["expected"], row["date"].to_pydatetime(), T), axis=1)

            group_occurrences.sort_values(['diff_mean_time'], ascending=True, inplace=True)
            
            #Drop extra occurrences
            group_occurrences.drop_duplicates(['component_absolute_mean_time'], keep='first', inplace=True)

            Nb_occurrences_happening_as_expected = len(group_occurrences.loc[group_occurrences.expected != 0])
            
            if Nb_expected_occurrences == 0:
                continue
            accuracy = Nb_occurrences_happening_as_expected / Nb_expected_occurrences
            
             # Raise an error if the accuracy is more than 1
            if accuracy > 1:
                raise ValueError('The accuray should not exceed 1.00 !!', episode, Nb_occurrences_happening_as_expected, Nb_expected_occurrences) 
            
            if (accuracy >= accuracy_min) & (accuracy > best_accuracy):
                #sns.distplot(data_points, norm_hist=False, rug=False, kde=True)
                best_accuracy = accuracy
                
    
                
                best_descr = {
                        "description" : GMM_descr,
                        "period" : T,
                        "accuracy" : accuracy,
                        "compression_power": Nb_occurrences_happening_as_expected * len(episode),
                        "expected_occurrences" : group_occurrences[["date"]],
                        "delta_t" : [group_start_time, group_end_time]
                        }
    
    
    
    return best_descr
    

def relative2absolute_date(relative_date, reference_date, period) :
    """
    Turn a relative date to an absolute date
    """
    
    date = reference_date - dt.timedelta(seconds = modulo_datetime(reference_date, period))
    
    date += dt.timedelta(seconds = relative_date)
    
    return date

def find_number_clusters(data_points, eps, min_samples):
    """
    return the number of clusters
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_points)
           
    Nb_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0) # Noisy samples are given the label -1.
    
    return Nb_clusters, db.components_
            
def modulo_datetime(date, period):
    """
    Compute the relative date in the period (time in seconds since the beginning of the corresponding period)
    :param date: datetime.datetime object
    :param period : datetime.timedelta
    """
    seconds = int((date - dt.datetime.min).total_seconds())
    remainder = dt.timedelta(
        seconds = seconds % period.total_seconds(),
        microseconds = date.microsecond,
    )
    return remainder.total_seconds()


def find_occurrences(data, episode, Tep):
    """
    Find the occurences of the  episode
    
    :param Tep : Maximum duration of an occurence
    """
    Tep = dt.timedelta(minutes=Tep)
    
    occurrences = pd.DataFrame(columns = ["date"])
    
    
    def sliding_window(row):
        """
        return true if there is an occurence of the episode starting at this timestamp
        """
        start_time = row.date
        end_time = row.date + Tep
        
        date_condition = (data.date >= start_time) & (data.date < end_time)
        
        next_activities = set(data.loc[date_condition, "activity"].values)
        
        return set(episode).issubset(next_activities)
        
    
    data.loc[:, "occurrence"] = data.apply(sliding_window, axis=1)
    
    while (len(data.loc[data.occurrence == True]) > 0):
        #Add a new occurrence
        occ_time = min(data.loc[data.occurrence == True].date)
        occurrences.loc[len(occurrences)] = [occ_time]
        
        #Marked the occurrences treated as "False"
        # TODO: can be improved
        indexes = []
        for s in episode :
            i = data.loc[(data.date >= occ_time) & (data.activity == s)].date.argmin()
            indexes.append(int(i))
        
        data.loc[data.index.isin(indexes), 'occurrence'] = False
    
    return occurrences

def translate_description(description) :
    """
    Translate the description in natural language
    """
    if description == None:
        return None
    natural_desc = {}
    natural_desc['period'] = str(description['period'])
    natural_desc['accuracy'] = round(description['accuracy'], 3)
    natural_desc['delta_t'] = [str(description['delta_t'][0]), str(description['delta_t'][1])]
    natural_desc['validity duration'] = str(description['delta_t'][1] - description['delta_t'][0])
    natural_desc["compression_power"] = description["compression_power"]
    natural_desc['description'] = {}
    for mean_time, std_time in description['description'].items():
        natural_desc['description'][str(dt.timedelta(seconds = mean_time))] = str(dt.timedelta(seconds = std_time))
        
    return natural_desc
    
    
if __name__ == "__main__":
    main()