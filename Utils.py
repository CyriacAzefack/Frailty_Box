import datetime as dt
import os

import matplotlib
import pandas as pd


def pick_dataset(name, nb_days=-1):
    my_path = os.path.abspath(os.path.dirname(__file__))

    dataset = None
    if name == 'toy':
        path = os.path.join(my_path, "./input/toy_dataset.csv")
        dataset = pd.read_csv(path, delimiter=';')
        date_format = '%Y-%d-%m %H:%M'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)

    elif name == 'aruba':
        path = os.path.join(my_path, "./input/aruba/activity_dataset.csv")
        dataset = pd.read_csv(path, delimiter=';')
        date_format = '%Y-%m-%d %H:%M:%S.%f'
        dataset['date'] = pd.to_datetime(dataset['date'], format=date_format)
        dataset['end_date'] = pd.to_datetime(dataset['end_date'], format=date_format)

    elif name.startswith('hh'):
        path = os.path.join(my_path, "./input/{}/dataset.csv".format(name))
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
    # ret = []
    # r = int(random.random() * 256)
    # g = int(random.random() * 256)
    # b = int(random.random() * 256)
    # step = 256 / n
    # for i in range(n):
    #     r += step
    #     g += 3 * step
    #     b += step
    #     r = int(r) % 256
    #     g = int(g) % 256
    #     b = int(b) % 256
    #     ret.append((r, g, b))
    # 256*rgb
    # colors = ['#%02x%02x%02x' % (c[0], c[1], c[2]) for c in ret]

    i = 0
    colors = []
    cmap = matplotlib.cm.get_cmap('Spectral')
    for i in range(1, n + 1):
        rgb = cmap(1 / i)
        rgb = [int(256 * x) for x in rgb]
        color = '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])
        colors.append(color)

    return colors