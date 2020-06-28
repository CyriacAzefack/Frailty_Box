# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 09:41:33 2018

@author: cyriac.azefack
"""
import math
import os
import sys

import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

sys.path.append(os.path.join(os.path.dirname(__file__)))

from Utils import *

def main():
    data = pick_dataset('KA')
    episodes = []
    # episodes.append(("go to bed END", "use toilet START", "use toilet START"))
    # episodes.append(["take shower start", "take shower end", "leave house start"])
    # episodes.append(["use toilet start", "use toilet end", "go to bed start"])
    # episodes.append(["prepare Breakfast start", "prepare Breakfast end"])
    # episodes.append(["leave house end"])
    episodes.append(['brush teeth start', 'go to bed start'])

    # episodes.append(["breakfast"])
    # episodes.append(("prepare breakfast",))

    with open('output/output_candidate_study_step.csv', 'w') as file:
        file.truncate()
        file.write("Episode;period;description;accuracy;start date;end date\n")
        for episode in episodes:
            descr = periodicity_search(data, episode, display=False)

            nat = translate_description(descr)
            line = str(episode) + ";"
            line += str(nat["period"]) + ";"
            line += str(nat["description"]) + ";"
            line += str(nat["accuracy"]).replace('.', ',') + ";"
            line += str(nat["delta_t"][0]) + ";"
            line += str(nat["delta_t"][1]) + ";"
            line += "\n"

            file.write(line)
            print("Episode ", episode)
            print(nat)


def periodicity_search(data, episode, delta_Tmax_ratio=3, support_min=3, std_max=0.1, tolerance_ratio=2,
                       Tep=30, period_T=dt.timedelta(days=1), display=False, verbose=False):
    """
    Find the best time description of an episode if it exists
    
    return delta_T : 
    """

    # Pick the episode events from the input sequence
    data = data.loc[data.label.isin(episode)].copy()

    if verbose:
        print("######################################")
        print("Periodicity episode :", episode)

    if len(data) == 0:
        if verbose:
            print(f'No data for the episode :{episode}')
        return None

    # find the episode occurences
    occurrences = find_occurrences(data, episode, Tep)

    if len(occurrences) < support_min:
        return None

    best_periodicity = None

    best_accuracy = 0

    # Compute intervals between occurrences

    # First row 'time_since_last_occ' is NaT so we replace by a duration of '0'
    occurrences.fillna(pd.Timestamp(0), inplace=True)

    # Compute relative dates
    occurrences.loc[:, "relative_date"] = occurrences.date.apply(
        lambda x: modulo_datetime(x.to_pydatetime(), period_T))

    # Display relative times count_histogram

    if display:
        plt.figure()
        plt.title(episode)
        sns.distplot(occurrences.relative_date / 3600, bins=20, norm_hist=False, rug=False, kde=False)
        plt.xlim((0, 24))
        plt.show()

    start_time = occurrences.date.min().to_pydatetime()
    end_time = occurrences.date.max().to_pydatetime()

    data_points = occurrences["relative_date"].values.reshape(-1, 1)

    cut_threshold = dt.timedelta(hours=3).total_seconds()
    m = np.ma.where(data_points < cut_threshold)
    data_points[m] = data_points[m] + period_T.total_seconds()
    # if no data then switch to the next group
    if len(data_points) < 2:
        if verbose:
            print('Not enough data points')
        return None

    cut_treshold = dt.timedelta(hours=3).total_seconds()

    if verbose:
        print("EPSILON DBSCAN :", str(dt.timedelta(seconds=std_max * period_T.total_seconds())))

    epsilon = 3600
    Nb_clusters, interesting_points = find_number_clusters(data_points, eps=epsilon, min_samples=int(support_min / 2),
                                                           display=display)

    # if no clusters found then switch to the next group
    if Nb_clusters == 0:
        if verbose:
            print('No clusters found')
        return None
    # Display points
    #            if display:
    #                sns.distplot(interesting_points, norm_hist=False, rug=False, kde=True, bins=10)

    GMM = GaussianMixture(n_components=Nb_clusters, n_init=10)
    GMM.fit(data_points)

    GMM_descr = {}  # mean_time (in seconds) as key and std_duration (in seconds) as value

    for i in range(len(GMM.means_)):
        mu = int(GMM.means_[i][0])
        sigma = int(math.ceil(np.sqrt(GMM.covariances_[i])))

        if sigma > std_max * period_T.total_seconds():
            continue
        lower_limit = mu - tolerance_ratio * sigma
        upper_limit = mu + tolerance_ratio * sigma

        # if (lower_limit < 0) or (lower_limit > period_T.total_seconds()):
        #     continue

        mu = mu % period_T.total_seconds()
        GMM_descr[mu] = sigma

        if verbose:
            print(f'Component {i} : mu={str(dt.timedelta(seconds=mu))}, sigma={str(dt.timedelta(seconds=sigma))}')

        # if display:
        #     lower_limit = mu - tolerance_ratio * sigma
        #     upper_limit = mu + tolerance_ratio * sigma
        #     # Plot the interval
        #     c = np.random.rand(3, )
        #     plt.plot([0, lower_limit], [lower_limit, 0], linewidth=2, color=c)
        #     plt.plot([0, upper_limit], [upper_limit, 0], linewidth=2, color=c)

        # Compute the time description accuracy

    accuracy, expected_occurrences = compute_pattern_accuracy(occurrences=occurrences, period=period_T,
                                                              time_description=GMM_descr)

    if not accuracy:
        if verbose:
            print("No accuracy found")
        return None

    str_GMM = {}
    for key, value in GMM_descr.items():
        str_GMM[str(dt.timedelta(seconds=key))] = str(dt.timedelta(seconds=value))

    best_periodicity = {
        "description": str_GMM,
        "period": period_T,
        "accuracy": accuracy,
        "nb_occ": len(occurrences),
        "compression_power": len(expected_occurrences) * len(episode),
        # "expected_occurrences": expected_occurrences,
        "delta_t": [start_time, end_time]
    }

    if verbose:
        print("Periodicity episode :", episode)
        print(f"\tAccuracy: {accuracy}")
        print(f"\tCompression Power: {len(expected_occurrences) * len(episode)}")
        print("######################################")

    return best_periodicity


def relative2absolute_date(relative_date, reference_date, period):
    """
    Turn a relative date to an absolute date
    """

    date = reference_date - dt.timedelta(seconds=modulo_datetime(reference_date, period))

    date += dt.timedelta(seconds=relative_date)

    return date


def find_number_clusters(X, eps, min_samples, display=False):
    """
    return the number of clusters
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    Nb_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)

    # #############################################################################
    # PLOT THE RESULTS
    ##############
    if display:
        angles = []

        for x in X:
            angles.append(2 * np.pi * x / (24 * 3600))

        X = [[math.sin(alpha), math.cos(alpha)] for alpha in angles]
        X = np.asarray(X)
        fig, ax = plt.subplots(1)
        theta = np.linspace(0, 2 * np.pi, 100)
        r = 1
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        ax.plot(x1, x2, linestyle='--')
        ax.set_aspect(1)
        plt.grid(linestyle='--')

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

            xy = X[class_member_mask & ~core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=3)

        # plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
    # Noisy samples are given the label -1.

    return Nb_clusters, db.components_


def modulo_datetime(date, period):
    """
    Compute the relative date in the period (time in seconds since the beginning of the corresponding period)
    :param date: datetime.datetime object
    :param period : datetime.timedelta
    """
    seconds = int((date - dt.datetime.min).total_seconds())
    remainder = dt.timedelta(
        seconds=seconds % period.total_seconds(),
        microseconds=date.microsecond,
    )
    return remainder.total_seconds()

def find_occurrences_fast(data, episode, Tep=30):
    """
    Fetch the occurrences of the episode in the log_dataset
    :param data:
    :param episode:
    :param Tep:
    :return:
    """
    Tep = dt.timedelta(minutes=Tep)

    data = data.loc[data.label.isin(episode)].copy()
    data.sort_values(by=['date'], inplace=True)

    if len(episode) == 1:
        return data[['date', 'end_date']]

    occurrences = {}

    def occurrence_exist(row):
        start_time = row.date
        end_time = row.date + Tep

        date_condition = (data.date >= start_time) & (data.date < end_time)

        next_labels = set(data.loc[date_condition, "label"].values)


def translate_description(description):
    """
    Translate the description in natural language
    """
    if not description:
        return None
    natural_desc = {}
    natural_desc['period'] = str(description['period'])
    natural_desc['accuracy'] = round(description['accuracy'], 3)
    natural_desc['delta_t'] = [str(description['delta_t'][0]), str(description['delta_t'][1])]
    natural_desc['validity duration'] = str(description['delta_t'][1] - description['delta_t'][0])
    natural_desc["compression_power"] = description["compression_power"]
    natural_desc['description'] = {}
    for mean_time, std_time in description['description'].items():
        natural_desc['description'][str(dt.timedelta(seconds=mean_time))] = str(dt.timedelta(seconds=std_time))

    return natural_desc


def is_occurence_expected(relative_date, GMM_descr, period, tolerance_ratio):
    for mu, sigma in GMM_descr.items():
        if abs(relative_date - mu) <= tolerance_ratio * sigma:  # Normal cases
            return mu

        # Handle the bord effects cases
        lower_limit = mu - tolerance_ratio * sigma
        upper_limit = mu + tolerance_ratio * sigma

        mu2 = mu
        if lower_limit < 0:  # Midnight-morning issue (early in the period)
            mu2 = mu + period.total_seconds()
        elif upper_limit > period.total_seconds():
            mu2 = mu - period.total_seconds()

        if abs(relative_date - mu2) <= tolerance_ratio * sigma:  # Normal cases
            return mu2

    return None


def compute_pattern_accuracy(occurrences, period, time_description, start_date=None, end_date=None, tolerance_ratio=2):
    '''
    Compute the accuracy of a pattern
    :param occurrences: Occurrences of the pattern
    :param period: Periodicity
    :param time_description: Time Description of the pattern
    :param start_date: Start date of the time period where we want to compute the accuracy
    :param end_date: End date of the period where we want to compute the accuracy
    :param tolerance_ratio:
    :return:
    '''

    if not start_date:
        start_date = occurrences.date.min().to_pydatetime()

    if not end_date:
        end_date = occurrences.date.max().to_pydatetime()

    occurrences = occurrences[(occurrences.date >= start_date) & (occurrences.date <= end_date)].copy()
    # First, compute the number of full periods between "start_date" and "end_date"
    relative_start_date = modulo_datetime(start_date, period)
    start_first_period = start_date
    if relative_start_date != 0:
        start_first_period = start_date - dt.timedelta(seconds=relative_start_date) + period

    relative_end_date = modulo_datetime(end_date, period)
    end_last_period = end_date
    if relative_end_date != 0:
        end_last_period = end_date - dt.timedelta(seconds=relative_end_date)

    nb_periods = (end_last_period - start_first_period).total_seconds() / period.total_seconds()

    nb_description_components = len(time_description)

    # Number of occurrences expected
    nb_occurrences_expected = nb_periods * nb_description_components

    # Now the bord effects
    for mean_time, std_time in time_description.items():
        if (mean_time > relative_start_date):
            nb_occurrences_expected += 1
        if (mean_time < relative_end_date):
            nb_occurrences_expected += 1

    if nb_occurrences_expected == 0 or len(occurrences) == 0:
        return None, None

    # Compute the relative date of the occurrences
    occurrences.loc[:, "relative_date"] = occurrences.date.apply(
        lambda x: modulo_datetime(x.to_pydatetime(), period))

    # "Expected" is the relative mean time of the component where an occurrence happen
    occurrences["expected"] = occurrences["relative_date"].apply(
        lambda x: is_occurence_expected(x, time_description, period, tolerance_ratio))

    # We need to drop the occurrences happening around the same time called extra occurrences

    # "diff_mean_time" is the time distanceDTW from the occurrence to the relative mean time of the component where an
    # occurrence happen
    occurrences["diff_mean_time"] = abs(occurrences["relative_date"] - occurrences["expected"])
    occurrences.fillna(0, inplace=True)

    # "component_absolute_mean_time" is the absolute date of the component where an occurrence happen
    occurrences["component_absolute_mean_time"] = occurrences.apply(
        lambda row: relative2absolute_date(row["expected"], row["date"].to_pydatetime(), period), axis=1)

    occurrences.sort_values(['diff_mean_time'], ascending=True, inplace=True)
    # Drop extra occurrences
    occurrences.drop_duplicates(['component_absolute_mean_time'], keep='first', inplace=True)

    Nb_occurrences_happening_as_expected = len(occurrences.loc[occurrences.expected != 0])

    accuracy = Nb_occurrences_happening_as_expected / nb_occurrences_expected

    accuracy = min(1, accuracy)
    # if accuracy > 1:
    #     raise ValueError('The accuracy should not exceed 1.00 !!',
    #                      Nb_occurrences_happening_as_expected, nb_occurrences_expected)

    return accuracy, occurrences[occurrences.expected != 0][['date']]


def plot_time_circle(data_points, period, plot=False):
    """
    plot time data points into a circle
    :param data_points:
    :return:
    """

    degrees = []
    x = []
    y = []

    for point in data_points:
        alpha = 2 * np.pi * point / period
        x.append(math.sin(alpha))
        y.append(math.cos(alpha))

        degrees.append(360 * point / period)

    if plot:
        radians = np.deg2rad(degrees)

        bin_size = 2
        a, b = np.histogram(degrees, bins=np.arange(0, 360 + bin_size, bin_size))
        centers = np.deg2rad(np.ediff1d(b) // 2 + b[:-1])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticklabels(['12am', '3am', '6am', '9am', '12pm', '3pm', '6pm', '9pm'])
        ax.tick_params(direction='out', length=6, width=6, colors='r', grid_alpha=1, labelsize=14)

        plt.show()

    return x, y


if __name__ == "__main__":
    main()
