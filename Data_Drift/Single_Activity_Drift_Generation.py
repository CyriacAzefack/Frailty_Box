import datetime as dt
import math
import random

import numpy as np
import pandas as pd


def main():
    label = 'Eating'

    # Initialisation
    # 3 occurrences per day

    output_folder = '../input/Toy/Simulation/'

    breakfast = {
        'mean_time': dt.timedelta(hours=8, minutes=30).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=15).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=20).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=8).total_seconds(),
        'accuracy': .95
    }

    brunch = {
        'mean_time': dt.timedelta(hours=11, minutes=30).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=15).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=20).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=7).total_seconds(),
        'accuracy': .95
    }

    lunch = {
        'mean_time': dt.timedelta(hours=13, minutes=25).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=30).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=40).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=10).total_seconds(),
        'accuracy': .95
    }

    lunner = {
        'mean_time': dt.timedelta(hours=17, minutes=30).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=25).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=40).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=10).total_seconds(),
        'accuracy': .95
    }

    dinner = {
        'mean_time': dt.timedelta(hours=18, minutes=50).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=45).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=25).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=5).total_seconds(),
        'accuracy': .95
    }

    behavior_A = [breakfast, lunch, dinner]
    behavior_B = [brunch, lunner]

    NB_DC = 100
    NB_DD = 100

    # DC Datasets

    for i in range(NB_DC):
        start_date = dt.datetime.strptime("01/01/19", "%d/%m/%y")

        pA = 0.5

        event_log = driftless_generation(label, behavior_A, behavior_B, pA, start_date, nb_days=365)

        event_log.to_csv()
        event_log.to_csv(output_folder + 'toy_DC_{}.csv'.format(i), index=False, sep=';')

        print("DC_{} Generated".format(i))

    # DD Datasets

    for i in range(NB_DD):
        start_date = dt.datetime.strptime("01/01/19", "%d/%m/%y")

        initial_pA = 1 - (0.5 / NB_DD) * i

        # 5 months - Beginning
        begin_event_log = driftless_generation(label, behavior_A, behavior_B, initial_pA, start_date, nb_days=5 * 30)

        # 2 months - transition
        start_date = begin_event_log.end_date.max().to_pydatetime() + dt.timedelta(days=1)
        start_date = dt.datetime.combine(start_date, dt.datetime.min.time())

        trans_event_log = progressive_drift_generation(label, behavior_A, behavior_B, initial_pA, start_date,
                                                       nb_days=60)

        # 5 months - Ending
        start_date = trans_event_log.end_date.max().to_pydatetime() + dt.timedelta(days=1)
        start_date = dt.datetime.combine(start_date, dt.datetime.min.time())

        end_event_log = driftless_generation(label, behavior_A, behavior_B, 1 - initial_pA, start_date, nb_days=5 * 30)

        event_logs = [begin_event_log, trans_event_log, end_event_log]

        event_log = pd.DataFrame(columns=['date', 'end_date', 'label'])
        for log in event_logs:
            event_log = event_log.append(log, ignore_index=True)

        event_log.to_csv(output_folder + 'toy_DD_{}.csv'.format(i), index=False, sep=';')

        print("DD_{} Generated".format(i))


def driftless_generation(label, behavior_A, behavior_B, prob_A, start_date, nb_days):
    """
    Generate an event log with no drift in the behavior
    :param label:
    :param behavior_A:
    :param behavior_B:
    :param prob_A:
    :param start_date:
    :param nb_days:
    :return:
    """

    end_date = start_date + dt.timedelta(days=nb_days)

    event_log = pd.DataFrame(columns=['date', 'end_date', 'label'])

    current_date = start_date
    while current_date < end_date:

        # Choose the behavior
        behavior = behavior_A  # default behavior

        rand_behavior = random.random()

        if rand_behavior > prob_A:
            behavior = behavior_B

        for occurrence in behavior:
            rand = random.random()
            if rand <= occurrence['accuracy']:  # prob of the occurrence to happen
                timestamp = np.random.normal(occurrence['mean_time'], occurrence['std_time'])
                duration = 0
                while duration <= 0:
                    duration = np.random.normal(occurrence['mean_duration'], occurrence['std_duration'])
                evt_start_date = current_date + dt.timedelta(seconds=timestamp)
                evt_end_date = evt_start_date + dt.timedelta(seconds=duration)

                event_log.loc[len(event_log)] = [evt_start_date, evt_end_date, label]

        current_date += dt.timedelta(days=1)

    return event_log


def progressive_drift_generation(label, behavior_A, behavior_B, initial_prob_A, start_date, nb_days):
    """
    Generate an event log with progressive drift between the start and end behavior
    :param label:
    :param behavior_A:
    :param behavior_B:
    :param initial_prob_A:
    :param start_date:
    :param nb_days:
    :return:
    """

    end_date = start_date + dt.timedelta(days=nb_days)

    cutoff_value = 0.999 * initial_prob_A
    d = int(nb_days / 2)

    alpha = math.log2(cutoff_value / (initial_prob_A - cutoff_value)) / d

    def prob_sigmoid(x):
        """
        Rising sigmoïd for prob_b
        :param x:
        :return:
        """
        return initial_prob_A / (1 + math.exp(alpha * (d - x)))

    event_log = pd.DataFrame(columns=['date', 'end_date', 'label'])

    current_date = start_date
    while current_date < end_date:
        x = (current_date - start_date).days

        # Choose the behavior
        behavior = behavior_A  # default behavior

        prob_B = prob_sigmoid(x)
        rand_behavior = random.random()

        if rand_behavior <= prob_B:
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
