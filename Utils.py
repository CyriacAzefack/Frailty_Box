import datetime as dt
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth


def pick_dataset(name, nb_days=-1, ):
    name = name.lower()
    my_path = os.path.abspath(os.path.dirname(__file__))

    dataset = None
    if 'toy' in name:
        path = os.path.join(my_path, "./input/Toy/Simulation/{}.csv".format(name))

        dataset = pd.read_csv(path, delimiter=';')
        date_format = '%Y-%m-%d %H:%M:%S.%f'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
        dataset['end_date'] = pd.to_datetime(dataset['end_date'], format=date_format)

    elif name == 'aruba':
        path = os.path.join(my_path, "./input/aruba/activity_dataset.csv")
        dataset = pd.read_csv(path, delimiter=';')
        date_format = '%Y-%m-%d %H:%M:%S.%f'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
        dataset['end_date'] = pd.to_datetime(dataset['end_date'], format=date_format)

    elif name == 'toulouse':
        path = os.path.join(my_path, "./input/Toulouse/toulouse_dataset.csv")
        dataset = pd.read_csv(path, delimiter=';')
        date_format = '%Y-%m-%d %H:%M:%S.%f'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
        dataset['end_date'] = pd.to_datetime(dataset['end_date'], format=date_format)

    elif name == 'mauricette':
        path = os.path.join(my_path, "./input/Mauricette/dataset.csv")
        dataset = pd.read_csv(path, delimiter=';')
        date_format = '%Y-%m-%d %H:%M:%S.%f'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
        dataset['end_date'] = pd.to_datetime(dataset['end_date'], format=date_format)


    elif name.startswith('hh'):
        path = os.path.join(my_path, "./input/HH/{}/dataset.csv".format(name))
        dataset = pd.read_csv(path, delimiter=',')
        date_format = '%Y-%m-%d %H:%M:%S.%f'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
        dataset['end_date'] = pd.to_datetime(dataset['end_date'], format=date_format)

    elif name == 'KA_events':
        filename = "./input/{} House/{}_dataset.csv".format('KA', 'KA_label')
        path = os.path.join(my_path, filename)
        dataset = pd.read_csv(path, delimiter=';')
        dataset['date'] = pd.to_datetime(dataset['date'])

    else:
        filename = "./input/{} House/{}_dataset.csv".format(name, name)
        path = os.path.join(my_path, filename)
        dataset = pd.read_csv(path, delimiter=';')
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset['end_date'] = pd.to_datetime(dataset['end_date'])

    # We only take nb_days
    if nb_days > 0:
        start_date = dataset.date.min().to_pydatetime()
        end_date = start_date + dt.timedelta(days=nb_days)
        dataset = dataset.loc[(dataset.date >= start_date) & (dataset.date < end_date)].copy()

    dataset.drop_duplicates(['date', 'label'], keep='last', inplace=True)

    # dataset['id_patient'] = dataset['date'].apply(lambda x : x.timetuple().tm_yday)
    # dataset['duree'] = 0
    # dataset['evt'] = dataset['label']
    # dataset['nbjours'] = dataset.date.apply(
    #         lambda x: int(Candidate_Study.modulo_datetime(x.to_pydatetime(), dt.timedelta(days=1))))
    #
    # dataset = dataset[['id_patient', 'duree', 'evt', 'nbjours']]
    #
    #
    # dataset.to_csv('./{}_hugo_dataset.csv'.format(name), period_ts_index=False, sep=";")

    return dataset


def pick_custom_dataset(path, nb_days=-1):
    dataset = pd.read_csv(path, delimiter=';')
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['end_date'] = pd.to_datetime(dataset['end_date'])

    # We only take nb_days
    if nb_days > 0:
        start_date = dataset.date.min().to_pydatetime()
        end_date = start_date + dt.timedelta(days=nb_days)
        dataset = dataset.loc[(dataset.date >= start_date) & (dataset.date < end_date)].copy()

    dataset.drop_duplicates(['date', 'label'], keep='last', inplace=True)

    return dataset


def empty_folder(path):
    """
    Delete all the files in the folder
    :param path:
    :return:
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))


def generate_random_color(n):
    """
    Generate n random colors
    :return:
    """

    colors = []
    # cmap = matplotlib.cm.get_cmap('nipy_spectral')
    # cmap = plt.get_cmap('Spectral')
    for i in range(n):
        color = matplotlib.cm.nipy_spectral(float(i) / n)
        colors.append(color)

    random.shuffle(colors)
    return colors


def stringify_keys(d):
    """
    Convert a dict's keys to strings if they are not.
    """
    for key in d.keys():

        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            try:
                d[str(key)] = value
            except Exception:
                try:
                    d[repr(key)] = value
                except Exception:
                    raise

            # delete old key
            del d[key]
    return d


def univariate_clustering(x):
    """
    1d - Clustering
    :param x:
    :param quantile:
    :return:
    """

    X = np.array(list(zip(x, np.zeros(len(x)))), dtype=np.int)

    if len(X) <= 2:
        X.reshape(-1, 1)
    bandwidth = estimate_bandwidth(X, quantile=0.3)
    if bandwidth == 0:
        return {}

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False)
    ms.fit(X)
    labels = ms.labels_
    # cluster_centers = ms.cluster_centers_

    X = X[labels >= 0]  # Filter '-1' label, outliers

    labels = labels[labels >= 0]  # Filter '-1' label, outliers

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    clusters = {}

    for k in range(n_clusters_):
        my_members = labels == k

        cluster_array = X[my_members, 0]

        # print("cluster {0}: {1}".format(k, cluster_array))

        clusters[k] = cluster_array

    sorted_clusters = dict(sorted(clusters.items(), key=lambda x: clusters[x[0]].mean()))

    return sorted_clusters


def find_occurrences(data, episode, tep=30):
    """
    Find the occurrences of the episode in the event log
    :param data: Event log data
    :param episode: list of labels
    :param tep : Maximum duration of an occurrence
    :return : A dataframe of occurrences with one date column
    """
    tep = dt.timedelta(minutes=tep)

    data = data[data.label.isin(episode)].copy()
    occurrences = pd.DataFrame(columns=["date", "end_date"])

    if len(data) == 0:
        return occurrences

    data.sort_values(by=['date'], inplace=True)

    if len(episode) == 1:
        return data[['date', 'end_date']]

    data['identical_next_label'] = data['label'].shift(-1) == data['label']

    data['enough_time'] = (data['date'].shift(-1) - data['date']) <= tep

    def sliding_window(row):
        """
        return true if there is an occurrence of the episode starting at this timestamp
        """
        start_time = row.date
        end_time = row.date + tep

        date_condition = (data.date >= start_time) & (data.date < end_time)

        next_labels = set(data[date_condition].label.values)

        return set(episode).issubset(next_labels)

    condition = (data.identical_next_label == False) & (data.enough_time == True)

    data.loc[condition, "occurrence"] = data[condition].apply(sliding_window, axis=1)

    data.fillna(False, inplace=True)

    while (len(data[data.occurrence == True]) > 0):
        # Add a new occurrence
        occ_time = data[data.occurrence == True].date.min().to_pydatetime()

        # Marked the occurrences treated as "False"
        # TODO: can be improved
        indexes = []
        for s in episode:
            i = data[(data.date >= occ_time) & (data.label == s)].date.idxmin()
            indexes.append(i)

        data.loc[indexes, 'occurrence'] = False

        end_occ_time = data.loc[indexes, "end_date"].max().to_pydatetime()
        # data.drop(indexes, inplace=True)
        occurrences.loc[len(occurrences)] = [occ_time, end_occ_time]

    occurrences.sort_values(by=['date'], ascending=True, inplace=True)
    return occurrences


def plot_graph(matrix, labels, plot=True):
    rows, cols = np.where(matrix > 0)

    gr = nx.Graph()

    for label in labels:
        gr.add_node(label)

    for i in range(len(rows)):
        row = rows[i]
        col = cols[i]

        gr.add_edge(labels[row], labels[col], weight=matrix[row][col])

    # gr.add_edges_from(edges)
    # nx.draw(gr, node_size=500, labels=labels, with_labels=True)

    if plot:
        nx.draw(gr, node_size=800, with_labels=True)
        plt.show()

    return gr


def convert_data_XES_log(name):
    data = pick_dataset(name)
    # ID for each day date : 'YYYYMMDD'
    data['case_id'] = data['date'].apply(lambda x: "{}{}{}".format(x.year, x.month, x.day))

    log = data[['case_id', 'date', 'end_date', 'label']]

    log.to_csv('./input/XES/XES_log_{}.csv'.format(name), index=False, sep=';')

# convert_data_XES_log('aruba')
