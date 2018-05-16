# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:22:58 2018

@author: cyriac.azefack
"""
import os
import sys

import scipy.stats as st

sys.path.append(os.path.join(os.path.dirname(__file__)))

import Graph_Pattern

from Pattern_Discovery.Candidate_Study import *
from Pattern_Discovery.Pattern_Discovery import *


def main():
    dataset_name = 'aruba'

    dataset = pick_dataset(dataset_name, nb_days=120)

    output = "../output/{}".format(dataset_name)
    patterns = pickle.load(open(output + '/patterns.pickle', 'rb'))

    pattern_graph_list = []
    for _, pattern in patterns.iterrows():

        labels = list(pattern['Episode'])
        period = pattern['Period']
        validity_start_date = pattern['Start Time'].to_pydatetime()
        validity_end_date = pattern['End Time'].to_pydatetime()
        validity_duration = validity_end_date - validity_start_date
        nb_periods = validity_duration.total_seconds() / period.total_seconds()
        description = pattern['Description']
        output_folder = output + "/Patterns_Graph/" + "_".join(labels) + "/"

        if not os.path.exists(os.path.dirname(output_folder)):
            try:
                os.makedirs(os.path.dirname(output_folder))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        mini_list = pattern2graph(data=dataset, labels=labels, time_description=description, period=period,
                                  start_date=validity_start_date, end_date=validity_end_date,
                                  output_directory=output_folder, display_graph=True)

        pattern_graph_list += mini_list


def pattern2graph(data, labels, time_description, period, start_date, end_date, tolerance_ratio=2, Tep=30,
                  output_directory='./', display_graph=False):
    '''
    Turn a pattern to a graph
    :param data: Input dataset
    :param labels: list of labels included in the pattern
    :param time_description: description of the pattern {mu1 : sigma1, mu2 : sigma2, ...}
    :param start_date: 
    :param tolerance_ratio: tolerance ratio to get the expectect occurrences
    :param Tep : [in Minutes] Maximal time interval between events in an episode occurrence. Should correspond to the maximal duration of the ADLs.
    :param output_directory Output Directory to save graphs images
    :return: A transition probability matrix and a transition waiting time matrix for each component of the description
    '''

    pattern_graph_list = []

    for mu, sigma in time_description.items():
        # Find pattern occurrences
        occurrences = find_occurrences(data, tuple(labels), Tep)
        occurrences = occurrences.loc[(occurrences.date >= start_date) & (occurrences.date <= end_date)].copy()

        # Compute relative dates
        occurrences.loc[:, "relative_date"] = occurrences.date.apply(
            lambda x: modulo_datetime(x.to_pydatetime(), period))

        # Drop unexpected occurrences
        occurrences["expected"] = occurrences["relative_date"].apply(
            lambda x: is_occurence_expected(x, {mu: sigma}, period, tolerance_ratio))
        occurrences.dropna(inplace=True, axis=0)

        if len(occurrences) == 0:
            continue

        events = find_events_occurrences(data, labels, occurrences, period, Tep)

        # Find the numbers of columns needed for the graphs
        # Find the number max of events in a occurrence

        period_ids = events['period_id'].unique()

        # Build a list of occurrences events list to build the graph
        events_occurrences_lists = []

        graph_nodes_labels = []
        for period_id in period_ids:
            period_list = []
            period_df = events[events.period_id == period_id]
            i = 0
            for index, event_row in period_df.iterrows():
                new_label = event_row['label'] + '_' + str(i)
                events.at[index, 'label'] = new_label
                period_list.append(new_label)
                i += 1
            graph_nodes_labels += period_list
            events_occurrences_lists.append(period_list)

        # Set of nodes for the graphs
        graph_nodes_labels = set(graph_nodes_labels)

        for period_id in period_ids:
            events_occurrences_lists.append(events.loc[events.period_id == period_id, 'label'].tolist())

        nodes, prob_matrix = build_probability_acyclic_graph(graph_nodes_labels, events_occurrences_lists)

        # TODO : Add the correlation between the time series
        # events['ts'] = events['date'].apply(lambda x: x.timestamp())
        events['next_label'] = events['label'].shift(-1)
        events['next_date'] = events['date'].shift(-1)
        events['duration'] = events['next_date'] - events['date']
        events['duration'] = events['duration'].apply(lambda x: x.total_seconds())

        n = len(nodes)

        # n-1 x n-1 edges for waiting time transition laws (no wait time to END NODE)
        time_matrix = [[[] for j in range(n)] for i in range(n)]  # Empty lists, [[mean_time, std_time], ...]

        for i in range(n):
            for j in range(n):
                if prob_matrix[i][j] != 0:  # Useless to compute time for never happening transition
                    start_node = nodes[i]
                    end_node = nodes[j]
                    time_df = events.loc[(events.label == start_node) & (events.next_label == end_node)]
                    if len(time_df) == 0 and end_node.endswith('_0'):
                        time_df = events.loc[events.next_label == end_node]
                    durations = time_df.duration.values
                    if len(durations) < 3:
                        time_matrix[i][j] = ('norm', [np.mean(durations), np.std(durations)])
                    else:
                        time_matrix[i][j] = best_fit_distribution(durations)

        # Fill the time_matrix for the first edges


        pattern_graph = Graph_Pattern.Graph_Pattern(nodes, period, mu, sigma, prob_matrix, time_matrix)
        pattern_graph_list.append(pattern_graph)

        if display_graph:
            pattern_graph.display(output_folder=output_directory, debug=True)

    return pattern_graph_list


def find_events_occurrences(data, labels, occurrences, period, Tep):
    '''
    Find the events included in the pattern occurrences
    :param data: Input Sequence
    :param labels: labels of the pattern
    :param occurrences: Occurrences of the pattern
    :param period: Frequency of the pattern
    :param Tep: is the time duration max between labels in the same occurrence
    :return: A Dataframe of events included in the occurrences. Columns : ['date', 'label', 'period_id']
    '''

    Tep = dt.timedelta(minutes=Tep)
    events = pd.DataFrame(columns=["date", "label", "period_id"])
    start_time = occurrences.date.min().to_pydatetime()
    start_date_first_period = start_time - dt.timedelta(
        seconds=modulo_datetime(start_time, period))

    end_time = occurrences.date.max().to_pydatetime()
    start_date_last_period = end_time - dt.timedelta(
        seconds=modulo_datetime(end_time, period))

    data = data.loc[data.label.isin(labels)]

    start_date_current_period = start_date_first_period

    period_id = 0
    while start_date_current_period <= start_date_last_period:
        end_date_current_period = start_date_current_period + period

        date_filter = (occurrences.date > start_date_current_period) \
                      & (occurrences.date < end_date_current_period)

        occurrence_happened = len(occurrences.loc[date_filter]) > 0
        if occurrence_happened:  # Occurrence happened
            # Fill events Dataframe
            occ_date = occurrences.loc[date_filter].date.min().to_pydatetime()
            occ_end_date = occ_date + Tep
            occ_events = data.loc[(data.date >= occ_date) & (data.date <= occ_end_date)].copy()
            occ_events['period_id'] = period_id
            events = pd.concat([events, occ_events]).drop_duplicates(keep=False)
            events.reset_index(inplace=True, drop=True)

        period_id += 1
        start_date_current_period = end_date_current_period

    return events


def best_fit_distribution(data, bins=200, ax=None):
    dist_list = ['norm', 'expon', 'lognorm', 'triang', 'beta']

    y, x = np.histogram(data, bins=200, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    best_distribution = 'norm'
    best_params = (0.0, 1.0)
    best_sse = np.inf

    for dist_name in dist_list:
        dist = getattr(st, dist_name)
        param = dist.fit(data)  # distribution fitting

        # Separate parts of parameters
        arg = param[:-2]
        loc = param[-2]
        scale = param[-1]

        param = list(param)

        # Calculate fitted PDF and error with fit in distribution
        pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.sum(np.power(y - pdf, 2.0))

        # if axis pass in add to plot
        try:
            if ax:
                pd.Series(pdf, x).plot(ax=ax, legend=True, label=dist_name)
        except Exception:
            pass

        # identify if this distribution is better
        if best_sse > sse > 0:
            best_distribution = dist_name
            best_params = param
            best_sse = sse

    return (best_distribution, best_params)


def build_probability_acyclic_graph(graph_nodes_labels, occurrence_list):
    '''
    Build the acyclic graph
    :param graph_nodes_labels: Pattern labels
    :param occurrence_list: List of list of ordered events per occurrence
    :return: List of the graph nodes and the probability transition matrix
    '''

    list_length = [len(l) for l in occurrence_list]
    nb_max_events = max(list_length)

    nodes = [Graph_Pattern.Graph_Pattern.START_NODE]
    nodes += graph_nodes_labels

    n = len(nodes)  # Size of the transition matrix

    prob_matrix = np.zeros((n, n))

    # Deal with the beginning of the graph
    single_list = []
    for list in occurrence_list:
        single_list.append(list[0])

    for label in set(single_list):
        prob_matrix[0][nodes.index(label)] = single_list.count(label) / len(single_list)

    for i in range(n - 2):
        tuple_list = []
        single_list = []
        for list in occurrence_list:
            if len(list) > i + 1:
                tuple_list.append(list[i:i + 2])
                single_list.append(list[i])

        for label_1 in set(single_list):
            # Count the number of tuple_list starting by 'label_1'
            nb_max = sum([1 if list[0] == label_1 else 0 for list in tuple_list])
            for label_2 in graph_nodes_labels:
                # Count the number of tuple_list starting by 'label_1' and finishing by 'label_2'
                nb = sum([1 if list == [label_1, label_2] else 0 for list in tuple_list])
                prob_matrix[nodes.index(label_1)][nodes.index(label_2)] = nb / nb_max

    # Checking the validity of the transition (sum output = 1 OR 0)

    tol = 0.001  # Error tolerance
    for i in range(n):
        # Row
        s_row = prob_matrix[i, :].sum()
        if abs(s_row - 1) > tol and s_row != 0:
            raise ValueError(
                'The sum of the probabilities transition from {} is neither 1 nor 0: {}'.format(nodes[i], s_row))

    return nodes, prob_matrix

if __name__ == '__main__':
    main()
