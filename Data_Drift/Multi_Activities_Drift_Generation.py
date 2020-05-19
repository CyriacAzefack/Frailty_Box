import copy
import random

import seaborn as sns
from tqdm import trange

from Utils import *

sns.set_style('darkgrid')


def main():
    output_folder = '../input/Drift_Toy/'

    init_behavior = init()

    labels = list(init_behavior.keys())
    N = 100
    NB_DAYS = 450

    for toy_id in trange(N, desc='All Synthetic data generation'):
        start_date = dt.datetime(year=2020, month=1, day=1)

        event_log = pd.DataFrame(columns=['date', 'end_date', 'label'])

        # Choose the number of drifts
        nb_drifts = random_nb_drift()
        # nb_drifts = 1

        # Split the date range
        nb_days_per_behavior = int(NB_DAYS / (nb_drifts + 1))

        current_behavior_start_date = start_date
        current_behavior = copy.deepcopy(init_behavior)

        for behavior_id in range(nb_drifts + 1):
            current_behavior_end_date = current_behavior_start_date + dt.timedelta(days=nb_days_per_behavior)

            current_date = current_behavior_start_date

            while current_date <= current_behavior_end_date:
                for label in labels:

                    label_behavior = current_behavior[label]

                    # rand = random.random()  # prob of the occurrence to happen
                    # if rand > label_behavior['accuracy']:
                    #     continue

                    timestamp = np.random.normal(label_behavior['mean_time'], label_behavior['std_time'])
                    duration = 0
                    while duration <= 0:
                        duration = np.random.normal(label_behavior['mean_duration'], label_behavior['std_duration'])
                    evt_start_date = current_date + dt.timedelta(seconds=timestamp)
                    evt_end_date = evt_start_date + dt.timedelta(seconds=duration)

                    # Add the event to the event log
                    event_log.loc[len(event_log)] = [evt_start_date, evt_end_date, label]

                current_date += dt.timedelta(days=1)

            # When finish, Apply some changes in the current_behavior
            current_behavior = apply_behavior_changes(current_behavior)
            current_behavior_start_date = current_date

        event_log.to_csv(output_folder + f'{nb_drifts}_drift_toy_data_{toy_id}.csv', index=False, sep=';')

        # print(f'{nb_drifts}_drift_toy_data_{toy_id} Generated')


def apply_behavior_changes(behavior):
    """
    Apply behavioral changes to a behavior
    :param behavior:
    :return:
    """

    new_behavior = copy.deepcopy(behavior)
    labels = list(behavior.keys())
    fields = ['mean_time', 'std_time', 'mean_duration', 'std_duration']

    nb_changing_labels = random.randint(1, len(labels))

    changing_labels = random.sample(labels, k=nb_changing_labels)

    # print(changing_labels)

    for label in changing_labels:
        nb_changing_fields = random.randint(3, len(fields))
        changing_fields = random.sample(fields, k=nb_changing_fields)

        # print(f'{label} Changing fields : {changing_fields}')

        for field in changing_fields:
            if field == 'mean_time':
                change_ratio = random.randint(-3, 3)  # Between -2 & 2
                new_behavior[label]['mean_time'] = new_behavior[label]['mean_time'] + \
                                                   change_ratio * new_behavior[label]['std_time']
            elif field == 'std_time':
                change_ratio = 1 + 2 * random.random()  # Between 0 and 2
                new_behavior[label]['std_time'] = change_ratio * new_behavior[label]['std_time']

            elif field == 'mean_duration':
                change_ratio = random.randint(-1, 3)  # Between -2 & 2
                new_behavior[label]['mean_duration'] = new_behavior[label]['mean_duration'] + \
                                                       change_ratio * new_behavior[label]['std_duration']

            elif field == 'std_duration':
                change_ratio = 1 + 2 * random.random()  # Between 0 and 2
                new_behavior[label]['std_duration'] = change_ratio * new_behavior[label]['std_duration']

            elif field == 'accuracy':
                change_ratio = 0.5 + 0.5 * random.random()
                new_behavior[label]['accuracy'] = change_ratio * new_behavior[label]['accuracy']

    # print(f'Old Behavior {behavior}')
    # print('#####"')
    # print(f'New Behavior {new_behavior}')
    return new_behavior


def random_nb_drift():
    """
    Return a randomly selected number of drift points
    :return:
    """
    prob_nb_drifts_range = [0.25, 0.25, 0.25, 0.25]
    prob_nb_drifts_range = np.cumsum(prob_nb_drifts_range)
    p = random.random()
    for i in range(len(prob_nb_drifts_range)):
        if p <= prob_nb_drifts_range[i]:
            return i + 1


def init():
    label = ['sleeping', 'work', 'lunch']

    # Initialisation
    # 3 occurrences per day

    original_behavior = {}

    work = {
        'mean_time': dt.timedelta(hours=9, minutes=30).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=10).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=17).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=21).total_seconds(),
        'accuracy': 1
    }

    original_behavior['work'] = work

    lunch = {
        'mean_time': dt.timedelta(hours=13, minutes=25).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=30).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=40).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=10).total_seconds(),
        'accuracy': 1
    }

    original_behavior['eating'] = lunch

    sleeping = {
        'mean_time': dt.timedelta(hours=21, minutes=0).total_seconds(),
        'std_time': dt.timedelta(hours=1, minutes=18).total_seconds(),
        'mean_duration': dt.timedelta(hours=6, minutes=40).total_seconds(),
        'std_duration': dt.timedelta(hours=1, minutes=20).total_seconds(),
        'accuracy': 1
    }
    original_behavior['sleeping'] = sleeping

    return original_behavior


if __name__ == '__main__':
    main()
