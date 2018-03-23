# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:48:16 2018

@author: cyriac.azefack
"""
import pandas as pd
import numpy as np


def main():
    l = ['C',]
    
    for letter in l :
        transform_data(letter)
        
def transform_data(letter = 'A'):
    
    activity_labels_df =  pd.read_csv("K{} House/K{}_activity_labels.csv".format(letter, letter), sep="\t");
    sensor_labels_df =  pd.read_csv("K{} House/K{}_sensor_labels.csv".format(letter, letter), sep="\t");
    
    as_df =  pd.read_csv("K{} House/K{}_as.csv".format(letter, letter), sep="\t");
    as_df.columns = [c.strip() for c in as_df.columns]
    ss_df =  pd.read_csv("K{} House/K{}_ss.csv".format(letter, letter), sep="\t");
    ss_df.columns = [c.strip() for c in ss_df.columns]
    ss_df.drop(['Val'], axis=1, inplace=True)
    
    
    
    # remove the quotes on the labels
    activity_labels_df["label"] = activity_labels_df["label"].str.replace('\'', '')
    sensor_labels_df["label"] = sensor_labels_df["label"].str.replace('\'', '')
    
    #merge
    
    activity_merge = pd.merge(as_df, activity_labels_df, on='ID', how='left')
    sensor_merge = pd.merge(ss_df, sensor_labels_df, on='ID', how='left')
    
    new_activity_df = pd.DataFrame(columns=["date", "activity"])

    date_format = '%d-%b-%Y %H:%M:%S'
    for index, row in activity_merge.iterrows():
        new_activity_df.loc[len(new_activity_df)] = [row["Start time"], row["label"]+" START"]
        new_activity_df.loc[len(new_activity_df)] = [row["End time"],  row["label"]+" END"]
    
    
    new_activity_df['date'] = pd.to_datetime(new_activity_df['date'], format=date_format)
    new_activity_df.sort_values(['date'], ascending=True, inplace=True)
    
    new_activity_df.to_csv("K{} House/K{}_activity_dataset.csv".format(letter, letter), sep=";", index=False)
    
    new_sensor_df = pd.DataFrame(columns=["date", "activity"])

    date_format = '%d-%b-%Y %H:%M:%S'
    for index, row in sensor_merge.iterrows():
        new_sensor_df.loc[len(new_sensor_df)] = [row["Start time"], str(row["label"])+" ON"]
        new_sensor_df.loc[len(new_sensor_df)] = [row["End time"],  str(row["label"])+" OFF"]
    
    
    new_sensor_df['date'] = pd.to_datetime(new_sensor_df['date'], format=date_format)
    new_sensor_df.sort_values(['date'], ascending=True, inplace=True)
    
    new_sensor_df.to_csv("K{} House/K{}_sensor_dataset.csv".format(letter, letter), sep=";", index=False)
    
if __name__ == '__main__':
    main()