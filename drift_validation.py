from Behavior_Drift import *

main_folder = './input/Drift_Toy/'

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
window_size = 30
time_step = dt.timedelta(minutes=5)
window_step = dt.timedelta(days=1)
latent_dim = 10
plot = False
debug = False

N = 450
results_df = pd.DataFrame(columns=['id', 'nb_drifts', 'silhouette', 'id_drift', 'delay_days'])

for toy_id in trange(len(toys_dict), desc='Toy Datasets'):
    toy = toys_dict[toy_id]

    nb_drifts = int(toy['nb_drifts'])

    n_per_beh = int(N / (nb_drifts + 1))

    drift_occ = [(i + 1) * n_per_beh for i in range(nb_drifts)]

    print("#############################")
    print(f"## Number Drifts = {toy['nb_drifts']} ###")
    print(f"## ID = {toy['id']} ###")
    print("#############################")

    path = main_folder + '/' + toy['name']
    data = pick_custom_dataset(path)

    time_window_size = dt.timedelta(days=window_size)

    behavior = AutoEncoderClustering(name=AE_folder_name, dataset=data, time_window_step=window_step,
                                     time_window_duration=time_window_size, time_step=time_step)

    # n_clusters = None
    clusters_indices, model_errors, silhouette = behavior.time_windows_clustering(display=plot, debug=debug,
                                                                                  latent_dim=latent_dim,
                                                                                  n_clusters=nb_drifts + 1)
    for cluster_id, clusters_indices in clusters_indices.items():
        if cluster_id == 0:
            continue
        clusters_indices.sort()
        first_occ = clusters_indices[0]

        delay = first_occ - drift_occ[cluster_id - 1]

        results_df.loc[len(results_df)] = [toy['id'], toy['nb_drifts'], silhouette, cluster_id - 1, delay]

    results_df.to_csv(f'./output/drift_toy_delay_results_w{window_size}_l{latent_dim}.csv', index=False)
