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
    dataset_name = 'KA'

    dataset = pick_dataset(dataset_name, nb_days=20)

    output = "../output/{}".format(dataset_name)
    patterns = pickle.load(open(output + '/patterns.pickle', 'rb'))

    patterns = patterns[:1]

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    pattern_graph_list = []
    for _, pattern in patterns.iterrows():

        labels = list(pattern['Episode'])
        period = pattern['Period']
        description = pattern['Description']
        output_folder = output + "/Patterns_Graph/" + "_".join(labels) + "/"

        if not os.path.exists(os.path.dirname(output_folder)):
            try:
                os.makedirs(os.path.dirname(output_folder))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        start_time = t.process_time()
        mini_list = pattern2graph(data=dataset, labels=labels, time_description=description, period=period,
                                  start_date=start_date, end_date=end_date, output_directory=output_folder,
                                  display_graph=True)
        elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))

        print("\n")
        print("###############################")
        print("Patterns turned into graphs. Elapsed time : {}".format(elapsed_time))
        print("##############################")
        print("\n")

        # end_date = start_date + dt.timedelta(days=5)
        # Compute Time Evolution
        start_time = t.process_time()
        for pattern_graph in mini_list:
            pattern_graph.compute_time_evolution(dataset, len(mini_list))
            sim = pattern_graph.simulate(start_date, end_date)
            filename = output + "/dataset_simulation.csv"
            sim.to_csv(filename, index=False, sep=';')

        elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))
        print("\n")
        print("###############################")
        print("Patterns Simulated. Elapsed time : {}".format(elapsed_time))
        print("##############################")
        print("\n")

        pattern_graph_list += mini_list


def pattern2graph(data, labels, time_description, period, start_date, end_date, tolerance_ratio=2, Tep=30,
                  output_directory='./', display_graph=False):
    '''
    Turn a pattern to a graph
    :param data: Input dataset
    :param labels: list of labels included in the pattern
    :param time_description: time description of the pattern {mu1 : sigma1, mu2 : sigma2, ...}
    :param tolerance_ratio: tolerance ratio to get the expected occurrences
    :param period : Periodicity of the time description
    :param start_date :
    :param end_date :
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

        accuracy = len(period_ids) / (max(period_ids) + 1)

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

        nodes, prob_matrix = build_probability_acyclic_graph(labels, graph_nodes_labels, events_occurrences_lists,
                                                             accuracy)

        # TODO : Add the correlation between the time series
        # events['ts'] = events['date'].apply(lambda x: x.timestamp())
        events['next_label'] = events['label'].shift(-1).fillna('_nan').apply(Graph_Pattern.Graph.node2label)
        events['next_date'] = events['date'].shift(-1)
        events['duration'] = events['next_date'] - events['date']
        events['duration'] = events['duration'].apply(lambda x: x.total_seconds())

        n = len(nodes)  # Nb rows of the prob matrix
        l = len(labels)  # Nb columns of the prob matrix

        # n x l edges for waiting time transition laws
        time_matrix = [[[] for j in range(l)] for i in range(n)]  # Empty lists, [[mean_time, std_time], ...]

        for i in range(n):
            for j in range(l):
                if prob_matrix[i][j] != 0:  # Useless to compute time for never happening transition
                    start_node = nodes[i]
                    end_node = labels[j]

                    if start_node == Graph_Pattern.Graph.START_NODE:  # START_NODE transitions
                        time_matrix[i][j] = ('norm', [mu, sigma])
                        continue

                    time_df = events.loc[(events.label == start_node) & (events.next_label == end_node)]
                    durations = time_df.duration.values
                    # TODO : Manage with other distribution than 'norm'
                    if len(durations) > 0:
                        time_matrix[i][j] = ('norm', [np.mean(durations), np.std(durations)])

        # Fill the time_matrix for the first edges

        pattern_graph = Graph_Pattern.Graph(nodes, labels, period, mu, sigma, prob_matrix, time_matrix)
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

    # Result dataframe
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

        date_filter = (occurrences.date >= start_date_current_period) \
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

    return best_distribution, best_params


def build_probability_acyclic_graph(labels, graph_nodes_labels, occurrence_list, accuracy):
    '''
    Build the acyclic graph
    :param labels: Labels of the pattern
    :param graph_nodes_labels: Pattern labels
    :param occurrence_list: List of list of ordered events per occurrence
    :return: Probability transition matrix size = nb(nodes) x nb(labels)
    '''

    list_length = [len(l) for l in occurrence_list]
    nb_max_events = max(list_length)

    nodes = [Graph_Pattern.Graph.START_NODE]
    nodes += graph_nodes_labels

    n = len(nodes)  # Size of the transition matrix
    l = len(labels)

    prob_matrix = np.zeros((n, l))

    # Deal with the beginning of the graph
    single_list = []
    for list in occurrence_list:
        single_list.append(list[0])

    for node in set(single_list):
        label = Graph_Pattern.Graph.node2label(node)
        prob_matrix[0][labels.index(label)] = accuracy * single_list.count(node) / len(single_list)

    for i in range(n - 2):
        tuple_list = []
        single_list = []
        for list in occurrence_list:
            if len(list) > i + 1:
                tuple_list.append(list[i: i + 2])
                single_list.append(list[i])

        for node_1 in set(single_list):
            # Count the number of tuple_list starting by 'label_1'
            nb_max = sum([1 if list[0] == node_1 else 0 for list in tuple_list])
            for node_2 in graph_nodes_labels:
                # Count the number of tuple_list starting by 'label_1' and finishing by 'label_2'
                nb = sum([1 if list == [node_1, node_2] else 0 for list in tuple_list])
                label_2 = Graph_Pattern.Graph.node2label(node_2)
                p = nb / nb_max
                if p != 0:
                    prob_matrix[nodes.index(node_1)][labels.index(label_2)] = p

    # Checking the validity of the transition (sum output = 1 OR 0)

    tol = 0.001  # Error tolerance
    for i in range(1, n):
        # Row
        s_row = prob_matrix[i, :].sum()
        if abs(s_row - 1) > tol and s_row != 0:
            raise ValueError(
                'The sum of the probabilities transition from {} is neither 1 nor 0: {}'.format(nodes[i], s_row))

    return nodes, prob_matrix

if __name__ == '__main__':
    main()
