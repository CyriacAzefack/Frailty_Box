from Behavior_Drift import *
from Data_Drift import Drift_Detector

main_folder = './input/Drift_Toy/'

labels = ['sleeping', 'work', 'eating']

toys_dict = []
text_files = [f for f in os.listdir(main_folder) if f.endswith('.csv')]
for file in text_files:
    nb_drifts, _, _, _, toy_id = file.split('.')[0].split('_')

    toy_dict = {
        'name': file,
        'nb_drifts': nb_drifts,
        'id': toy_id
    }

    toys_dict.append(toy_dict)

AE_folder_name = 'Notebook Drift Toys'
window_size = 7
time_step = dt.timedelta(minutes=5)
window_step = dt.timedelta(days=1)
latent_dim = 10
plot = False
debug = False

N = 450
results_df = pd.DataFrame(columns=['id', 'nb_drifts', 'nb_all_drifts_detected', 'nb_sleeping_drift_detected',
                                   'nb_work_drift_detected', 'nb_eating_drift_detected', ])

for toy_id in trange(len(toys_dict), desc='Toy Datasets'):
    toy = toys_dict[toy_id]

    nb_drifts = int(toy['nb_drifts'])

    n_per_beh = int(N / (nb_drifts + 1))

    drift_occ = [(i + 1) * n_per_beh for i in range(nb_drifts)]

    print("#############################")
    print(f"## Number of Behaviors = {int(toy['nb_drifts']) + 1} ###")
    print(f"## ID = {toy['id']} ###")
    print("#############################")

    path = main_folder + '/' + toy['name']
    data = pick_custom_dataset(path)

    time_window_size = dt.timedelta(days=window_size)

    # # # AUTO-ENCODER DRIFT MODEL
    print('\t###### MULTIPLE DRIFT DETECTION #######')
    behavior = AutoEncoderClustering(name=AE_folder_name, dataset=data, time_window_step=window_step,
                                     time_window_duration=time_window_size, time_step=time_step)

    # n_clusters = None
    clusters_indices, model_errors, silhouette = behavior.time_windows_clustering(display=plot, debug=debug,
                                                                                  latent_dim=latent_dim,
                                                                                  n_clusters=None)

    nb_all_drift_detected = len(clusters_indices)

    nb_label_drift_detected = []
    for label in labels:
        print(f"\t###### SINGLE DRIFT DETECTION : '{label}' #######")
        clusters, silhouette = Drift_Detector.activity_drift_detector(data, time_window_size, label, validation=False)
        nb_label_drift_detected.append(len(clusters))

    nb_sleep_drift_detected, nb_work_drift_detected, nb_eating_drift_detected = tuple(nb_label_drift_detected)

    results_df.loc[len(results_df)] = [toy['id'], toy['nb_drifts'], nb_all_drift_detected, nb_sleep_drift_detected,
                                       nb_work_drift_detected, nb_eating_drift_detected]

results_df.to_csv(f'./output/drift_toy_nb_drifts_results_w{window_size}_l{latent_dim}.csv', index=False)
