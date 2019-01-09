import datetime as dt
import random

import numpy as np
import pandas as pd


def main():
    label = 'Eating'

    # Initialisation
    # 3 occurrences per day

    std_factor = 2

    breakfast = {
        'mean_time': dt.timedelta(hours=8, minutes=30).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=50).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=20).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=7, seconds=37).total_seconds(),
        'accuracy': 0.85
    }

    lunch = {
        'mean_time': dt.timedelta(hours=13, minutes=25).total_seconds(),
        'std_time': dt.timedelta(hours=1, minutes=30).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=40).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=10, seconds=9).total_seconds(),
        'accuracy': 0.9
    }

    dinner = {
        'mean_time': dt.timedelta(hours=18, minutes=50).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=45).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=25).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=5, seconds=43).total_seconds(),
        'accuracy': 0.85
    }

    occurrences = [breakfast, lunch, dinner]

    start_date = dt.datetime.strptime("01/01/19", "%d/%m/%y")

    event_log_1 = driftless_generation(label, occurrences, start_date, nb_days=120)

    new_breakfast = {
        'mean_time': dt.timedelta(hours=10, minutes=30).total_seconds(),
        'std_time': dt.timedelta(hours=0, minutes=50).total_seconds(),
        'mean_duration': dt.timedelta(hours=0, minutes=20).total_seconds(),
        'std_duration': dt.timedelta(hours=0, minutes=7, seconds=37).total_seconds(),
        'accuracy': 0.7
    }

    new_occurrences = [new_breakfast, lunch, dinner]

    start_date = event_log_1.end_date.max().to_pydatetime() + dt.timedelta(days=1)
    start_date = dt.datetime.combine(start_date, dt.datetime.min.time())

    event_log_2 = driftless_generation(label, new_occurrences, start_date, nb_days=245)

    event_log = event_log_1.append(event_log_2, ignore_index=True)

    event_log.to_csv('../input/Toy/toy_1_change_later_breakfast.csv', index=False, sep=';')


def driftless_generation(label, occurrences, start_date, nb_days):
    """
    Generate an event log with no drift in the behavior
    :param occurrences:
    :param start_date:
    :param duration:
    :return:
    """

    end_date = start_date + dt.timedelta(days=nb_days)  # 1 year of data

    event_log = pd.DataFrame(columns=['date', 'end_date', 'label'])

    current_date = start_date
    while current_date < end_date:
        for occurrence in occurrences:
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


if __name__ == '__main__':
    main()
