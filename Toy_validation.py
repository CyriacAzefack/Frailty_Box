import glob
import os
import sys

from Behavior import *
from Utils import *

if __name__ == '__main__':

    window_size = 28

    behavior_type = Behavior.OCC_TIME

    drift_method = sys.argv[1]
    dfs = []

    dc_folder = "./input/Toy/Simulation/"
    for file in glob.glob(dc_folder + "*.csv"):
        dataset_name = os.path.basename(file)[:-4]  # to remove the csv extension

        print('####### {} begin --->'.format(dataset_name))

        df = data_drift(dataset_name, window_size, behavior_type, drift_method, False, False, labels=None)
        dfs.append(df)

    all_results_df = pd.concat(dfs, ignore_index=True)

    all_results_df.to_csv('./output/{}_Toy_Validation.csv'.format(drift_method), index=False, sep=';')
