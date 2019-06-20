import datetime as dt
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth


def pick_dataset(name, nb_days=-1):
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
    # dataset.to_csv('./{}_hugo_dataset.csv'.format(name), index=False, sep=";")

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
    cmap = matplotlib.cm.get_cmap('Spectral')
    # cmap = plt.get_cmap('Spectral')
    for i in range(1, n + 1):
        rgb = cmap(1 / i)
        rgb = [int(256 * x) % 256 for x in rgb]
        color = '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])
        colors.append(color)

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


def univariate_clustering(x, quantile=0.5):
    """
    1d - Clustering
    :param x:
    :param quantile:
    :return:
    """

    X = np.array(list(zip(x, np.zeros(len(x)))), dtype=np.int)

    bandwidth = estimate_bandwidth(X, quantile=quantile)
    if bandwidth == 0:
        return {}
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

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
