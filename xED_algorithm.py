# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:22:14 2018

@author: cyriac.azefack
"""

import matplotlib
import pandas as pd
import sys
import time as t
import datetime as dt

#matplotlib.style.use("seaborn")

import FP_growth
import candidate_study


def main(): 
    ########################################
    # DATA PREPROCESSING
    ########################################
    
    """
    The dataframe should have 1 index (date as datetime) and 1 feature (activity)
    """
    
    letters = ['A']
    dataset_type = 'activity'
    
    for letter in letters :
        dataset = pick_dataset(letter, dataset_type)
        
        #dataset = dataset.set_index('date')
    
        start_time = t.process_time()
        results, data_left = xED_algorithm(data=dataset, Tep=30, support_min=2,
                                    tolerance_ratio=2)
    
        elapsed_time = dt.timedelta(seconds = round(t.process_time() - start_time, 3))
    
        print("\n")
        print("###############################")
        print("Time to process all the dataset : {}".format(elapsed_time))
        print("##############################")
        print("\n")
    
        
        results.to_csv("output/K{} House/K{}_{}_periodicities.csv".format(letter, letter, dataset_type), sep=";", index=False)
        
        data_left.to_csv("output/K{} House/K{}_{}_data_left.csv".format(letter, letter, dataset_type), sep=";", index=False)
        
        writer = pd.ExcelWriter("output/K{} House/K{}_{}_all.xlsx".format(letter, letter, dataset_type))
        dataset.to_excel(writer, sheet_name="Data input", index=False)       
        results.to_excel(writer, sheet_name="Periodicities", index=False)
        data_left.to_excel(writer, sheet_name="Data Left", index=False)
        writer.save()
        
        
    

def xED_algorithm(data, Tep = 30, support_min = 2, accuracy_min = 0.5, 
                  std_max = 0.1, tolerance_ratio = 2, delta_Tmax_ratio = 3, verbose = True):
    """
    Implementation of the extended Discovery Algorithm designed by Julie Soulas U{https://hal.archives-ouvertes.fr/tel-01356217/}
    
    :param data : Starting dataframe, date[datetime] as index and 1 column named "activity"
    :param Tep : [in Minutes] Maximal time interval between events in an episode occurrence. Should correspond to the maximal duration of the ADLs.
    :param support_min : [greater than 1] Minimal number of occurrences of an episode, for that episode to be considered as frequent.
    :param accuracy_min : [between 0 and 1] Minimal accuracy for a periodicity description to be considered as interesting, and thus factorized
    :param std_max : Standard deviation max for a description (ratio of the period)
    :param tolerance_ratio : [greater than 0] An event expected to happen at time t (with standard deviation sigma) occurs as expected if it occurs in the interval [t - tolerance_ratio*sigma, t + tolerance_ratio*sigma]
    :param delta_Tmax_ratio : If there is a gap > delta_Tmax_ratio * Period between two occurrences of an episode, theoccurrences before and after the gap are split (different validity intervals).
    :return The compressed dataset
    """
    compressed = True

    final_periodicities = pd.DataFrame(columns=["Episode", "Period", "Description", "Validity Duration",
                                                "Start Time", "End Time", "Compression Power", "Accuracy"])
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
            print("  Finding frequent episodes candidates...  ".center(100, '*'))
        
        frequent_episodes = FP_growth.find_frequent_episodes(data, support_min, Tep)
        
        print(len(frequent_episodes), "candidate episodes found !!")
        
        periodicities = {}
        
        print(" Building candidates episodes periodicities... ".center(100, '*'))
        
        episode_index = 0
        for episode in frequent_episodes.keys():
            episode_index += 1
            #Build the description of the episode if interesting enough (Accuracy > Accuracy_min)
            description = candidate_study.periodicity_search(data, episode,
                                                             delta_Tmax_ratio=delta_Tmax_ratio,
                                                             support_min=support_min,
                                                             std_max=std_max,
                                                             accuracy_min=accuracy_min,
                                                             tolerance_ratio=tolerance_ratio, Tep=Tep)
            if description is not None:
                #print("\nInteresting periodicity found for the episode", episode)
                periodicities[episode] = description
                
            sys.stdout.write("\r%.2f %% of episodes treated!!" % (100*episode_index/len(frequent_episodes)))
            sys.stdout.flush()
        sys.stdout.write("\n")
        
        if len(periodicities) == 0:
            print("No more intersting periodicities found!!".center(100, '*'))
            break;
        else :
            print((str(len(periodicities)) + " interesting periodicities found!!").center(100, '*'))

        print("Sorting the interesting periodicities by compression power...".center(100, '*'))
        sorted_episode = sorted(periodicities.keys(), key=lambda x: periodicities[x]["compression_power"], reverse=True)

        print("Dataset Rewriting".center(100, '*'))

        factorised_events = pd.DataFrame(columns=["date", "activity"])
        
        data_bis = data.copy() #Data to handle the rewriting
        while True:
            
            episode = sorted_episode[0]
            periodicity = periodicities[episode]

            if periodicity["compression_power"] <= 1:
                print("### COMPRESSION FINISHED : Insufficient compression power reached...")
                break;


            expected_occurrences = periodicity["expected_occurrences"]

            mini_factorised_events = pd.DataFrame(columns=["date", "activity"])

            #Find the events corresponding to the expected occurrences
            for index, occurrence in  expected_occurrences.iterrows() :
                start_date = occurrence["date"]
                end_date = start_date + dt.timedelta(minutes=Tep)
                mini_data = data_bis.loc[(data_bis.activity.isin(episode))
                                     & (data_bis.date >= start_date)
                                     & (data_bis.date < end_date)].copy()
                mini_data.sort_values(["date"], ascending=True, inplace=True)
                mini_data.drop_duplicates(["activity"], keep='first', inplace=True)
                mini_factorised_events = mini_factorised_events.append(mini_data, ignore_index=True)

            factorised_events = factorised_events.append(mini_factorised_events, ignore_index=True)
            count_duplicates =  factorised_events.duplicated(['date', 'activity']).sum()
            if count_duplicates != 0 :
                # Current periodicity involves events in factorized_events
                print("### COMPRESSION FINISHED : Overlapping factorized events reached...")
                break

            # Factorize DATA
            data = pd.concat([data, mini_factorised_events]).drop_duplicates(keep=False)
            data.reset_index(inplace=True, drop=True)


            # Add the periodicity to the results
            natural_periodicity = candidate_study.translate_description(periodicity)

            final_periodicities.loc[len(final_periodicities)] = [episode, natural_periodicity["period"],
                                                                 natural_periodicity["description"],
                                                                 natural_periodicity["validity duration"],
                                                                 natural_periodicity["delta_t"][0],
                                                                 natural_periodicity["delta_t"][1],
                                                                 natural_periodicity["compression_power"],
                                                                 natural_periodicity["accuracy"]]

            # Remove periodicity from the list
            sorted_episode = sorted_episode[1:]
            if len(sorted_episode) == 0:
                print("### COMPRESSION FINISHED : No more frequent episodes...")
                break;

            compressed = True
            

    return final_periodicities, data



def pick_dataset(letter, dataset_type = None) :
    dataset = None
    if letter == 'toy':
        dataset = pd.read_csv("input/toy_dataset.txt", delimiter=';')
        date_format = '%Y-%d-%m %H:%M'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
    
    else :
        dataset = pd.read_csv("input/K{} House/K{}_{}_dataset.csv".format(letter, letter, dataset_type), delimiter=';')
        dataset['date'] = pd.to_datetime(dataset['date'])
    
        
    return dataset

if __name__ == "__main__":
    main()