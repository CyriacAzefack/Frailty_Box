import datetime as dt
import math
import random

import numpy as np
import pandas as pd


def main():
    label = 'Eating'

    # Initialisation
    # 3 occurrences per day

    toy_name = 'progressive_change'

    breakfast = {
        'mean_time': dt.timedelta(hours=8, minutes=30).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=15).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=20).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=7, seconds=37).total_seconds(),
        'accuracy': .95
    }

    brunch = {
        'mean_time': dt.timedelta(hours=11, minutes=30).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=20).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=20).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=7, seconds=37).total_seconds(),
        'accuracy': .95
    }


    lunch = {
        'mean_time': dt.timedelta(hours=13, minutes=25).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=30).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=40).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=10, seconds=9).total_seconds(),
        'accuracy': .90
    }

    lunner = {
        'mean_time': dt.timedelta(hours=17, minutes=30).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=50).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=40).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=10, seconds=9).total_seconds(),
        'accuracy': .90
    }

    dinner = {
        'mean_time': dt.timedelta(hours=18, minutes=50).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=45).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=25).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=5, seconds=43).total_seconds(),
        'accuracy': 1
    }

    start_date = dt.datetime.strptime("01/01/19", "%d/%m/%y")

    behavior_A = [breakfast, lunch, dinner]
    behavior_B = [brunch, lunner]

    # Period 1

    event_log_1 = driftless_generation(label, behavior_A, start_date, nb_days=90)  # 3 months

    # Period 2
    start_date = event_log_1.end_date.max().to_pydatetime() + dt.timedelta(days=1)
    start_date = dt.datetime.combine(start_date, dt.datetime.min.time())

    event_log_2 = progressive_drift_generation(label, behavior_A, behavior_B, start_date, nb_days=150)  # 5 months

    # # Period 3
    # start_date = event_log_2.end_date.max().to_pydatetime() + dt.timedelta(days=1)
    # start_date = dt.datetime.combine(start_date, dt.datetime.min.time())
    #
    #
    # event_log_3 = driftless_generation(label, behavior_B, start_date, nb_days=180)  # 6 months

    # # Period 4
    # start_date = event_log_3.end_date.max().to_pydatetime() + dt.timedelta(days=1)
    # start_date = dt.datetime.combine(start_date, dt.datetime.min.time())
    #
    # behavior = [brunch, lunner]
    #
    # event_log_4 = driftless_generation(label, behavior, start_date, nb_days=90)  # 3 months

    event_logs = [event_log_1, event_log_2]

    event_log = pd.DataFrame(columns=['date', 'end_date', 'label'])
    for log in event_logs:
        event_log = event_log.append(log, ignore_index=True)

    event_log.to_csv('../input/Toy/toy_{}.csv'.format(toy_name), index=False, sep=';')

    # new_occurrences = [new_breakfast, lunch, dinner]
    #
    # start_date = event_log_2.end_date.max().to_pydatetime() + dt.timedelta(days=1)
    # start_date = dt.datetime.combine(start_date, dt.datetime.min.time())
    #
    # event_log_3 = driftless_generation(label, new_occurrences, start_date, nb_days=250)

    event_log = event_log_2.append(event_log_1, ignore_index=True)

    # event_log = event_log.append(event_log_3, ignore_index=True)

    event_log.to_csv('../input/Toy/toy_1_change_later_breakfast.csv', index=False, sep=';')


def driftless_generation(label, behavior, start_date, nb_days):
    """
    Generate an event log with no drift in the behavior
    :param label:
    :param behavior:
    :param start_date:
    :param nb_days: number of days
    :return:
    """

    end_date = start_date + dt.timedelta(days=nb_days)

    event_log = pd.DataFrame(columns=['date', 'end_date', 'label'])

    current_date = start_date
    while current_date < end_date:
        for occurrence in behavior:
            rand = random.random()
            if rand <= occurrence['accuracy']:  # prob of the occurrence to happen
                timestamp = np.random.normal(occurrence['mean_time'], occurrence['std_time'])
                nb_days = 0
                while nb_days <= 0:
                    nb_days = np.random.normal(occurrence['mean_duration'], occurrence['std_duration'])
                evt_start_date = current_date + dt.timedelta(seconds=timestamp)
                evt_end_date = evt_start_date + dt.timedelta(seconds=nb_days)

                event_log.loc[len(event_log)] = [evt_start_date, evt_end_date, label]

        current_date += dt.timedelta(days=1)

    return event_log


def progressive_drift_generation(label, behavior_A, behavior_B, start_date, nb_days):
    """
    Generate an event log with progressive drift between the start and end behavior
    :param label:
    :param start_behavior: Original behavior
    :param end_behavior:
    :param start_date:
    :param duration:
    :return:
    """

    end_date = start_date + dt.timedelta(days=nb_days)

    cutoff_value = 0.999
    d = int(nb_days / 2)

    alpha = math.log2(cutoff_value / (1 - cutoff_value)) / d

    def prob_sigmoid(x):
        return 1 / (1 + math.exp(alpha * (d - x)))


    event_log = pd.DataFrame(columns=['date', 'end_date', 'label'])

    current_date = start_date
    while current_date < end_date:
        x = (current_date - start_date).days

        # Choose the behavior
        behavior = behavior_A  # default behavior

        prob_B = prob_sigmoid(x)
        rand_beh = random.random()

        if rand_beh <= prob_B:
            behavior = behavior_B

        for occurrence in behavior:
            rand = random.random()
            if rand <= occurrence['accuracy']:  # prob of the occurrence to happen
                timestamp = np.random.normal(occurrence['mean_time'], occurrence['std_time'])
                nb_days = 0
                while nb_days <= 0:
                    nb_days = np.random.normal(occurrence['mean_duration'], occurrence['std_duration'])
                evt_start_date = current_date + dt.timedelta(seconds=timestamp)
                evt_end_date = evt_start_date + dt.timedelta(seconds=nb_days)

                event_log.loc[len(event_log)] = [evt_start_date, evt_end_date, label]

        current_date += dt.timedelta(days=1)

    return event_log


if __name__ == '__main__':
    main()
