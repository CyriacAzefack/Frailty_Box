# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:22:58 2018

@author: cyriac.azefack
"""
import base64
from collections import defaultdict
from pprint import pprint
from subprocess import check_call

import matplotlib.image as mpimg
import networkx as nx
import scipy

from candidate_study import *
from xED_algorithm import *


def main():
    letter = 'C'
    dataset_type = 'label'
    dataset = pick_dataset(letter, dataset_type)

    output = "output/K{} House/{}".format(letter, dataset_type)
    patterns = pickle.load(open(output + '/patterns.pickle', 'rb'))

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

        Mp_dict, Mwait_dict = patterns2graph(data=dataset, labels=labels, description=description, period=period,
                                             start_date=validity_start_date, end_date=validity_end_date,
                                             output_folder=output_folder)


def patterns2graph(data, labels, description, period, start_date, end_date, tolerance_ratio=2, Tep=30,
                   output_folder='./'):
    '''
    :param data: Input dataset
    :param labels: list of labels included in the pattern
    :param description: description of the pattern {mu1 : sigma1, mu2 : sigma2, ...}
    :param tolerance_ratio: tolerance ratio to get the expectect occurrences
    :param Tep : [in Minutes] Maximal time interval between events in an episode occurrence. Should correspond to the maximal duration of the ADLs.
    :return: A transition probability matrix and a transition waiting time matrix for each component of the description
    '''

    Mp_dict = {}
    Mwait_dict = {}
    nodes = ['START PERIOD'] + labels + ['END PERIOD']
    n = len(nodes)
    for mu, sigma in description.items():
        # n x n edges for probabilities transition
        Mp = np.zeros((n, n))

        # n-1 x n-1 edges for waiting time transition laws (no wait time to END NODE)
        Mwait = defaultdict(lambda: defaultdict(list))
        for i in range(n - 1):
            for j in range(n - 1):
                Mwait[i][j] = []

        # Find pattern occurrences
        occurrences = find_occurrences(data, tuple(labels), Tep)
        occurrences = occurrences.loc[(occurrences.date >= start_date) & (occurrences.date <= end_date)].copy()

        # Compute relative dates
        occurrences.loc[:, "relative_date"] = occurrences.date.apply(
            lambda x: modulo_datetime(x.to_pydatetime(), period))

        occurrences["expected"] = occurrences["relative_date"].apply(
            lambda x: is_occurence_expected(x, {mu: sigma}, period, tolerance_ratio))
        occurrences.dropna(inplace=True, axis=0)

        if len(occurrences) == 0:
            continue

        events = find_events_occurrences(data, labels, occurrences, period, Tep)

        nb_periods = events.period_id.max() + 1

        nb_occurrences_per_label = np.zeros(n)
        # START Node
        nb_occurrences_per_label[0] = nb_periods
        nb_occurrences_per_label[n - 1] = nb_periods
        for i in range(n - 2):
            label = nodes[i + 1]
            # count the label occurrences
            nb_occurrences_per_label[i + 1] = len(events.loc[events.label == label])

        start_date = occurrences.date.min().to_pydatetime()
        first_period_start_date = start_date - dt.timedelta(
            seconds=modulo_datetime(start_date, period))
        # Build the transition probability matrix
        # We always have the EDGE END --> START
        # TODO : EDGE END ----> START is putted as 1 (for check purporses)
        Mp[n - 1, 0] = 1


        # Missing occurrences, #START --> END directly (waiting time = Period)
        # TODO : Totally missing occurrences
        Mp[0, n - 1] += (nb_periods - len(events.period_id.unique())) / nb_periods
        # Mwait[0][n - 1].append(period.total_seconds())
        for period_id in events.period_id.unique():
            period_start_date = first_period_start_date + period_id * period
            period_end_date = period_start_date + period
            date_condition = (events.date >= period_start_date) \
                             & (events.date < period_end_date)

            period_events = events.loc[date_condition].copy()

            for label in labels:
                # Entering edge
                label_events = period_events.loc[period_events.label == label]
                nb_label = len(label_events)

                for _, row in label_events.iterrows():
                    # SORTING EDGES
                    if len(period_events.loc[period_events.date > row['date']]) > 0:
                        sorting_id = period_events.loc[period_events.date > row['date']].date.argmin()
                        sorting_label = period_events.loc[[sorting_id]].label.values[0]
                        sorting_label_date = period_events.loc[[sorting_id]].date.min().to_pydatetime()
                        Mp[nodes.index(label), nodes.index(sorting_label)] += 1 / nb_occurrences_per_label[
                            nodes.index(label)]
                        Mwait[nodes.index(label)][nodes.index(sorting_label)].append(
                            modulo_datetime(sorting_label_date, period) - modulo_datetime(row['date'].to_pydatetime(),
                                                                                          period))
                    else:
                        # last label of the occurrence, Label --> END PERIOD
                        Mp[nodes.index(label), n - 1] += 1 / nb_occurrences_per_label[nodes.index(label)]


            # First label
            first_id = period_events.date.argmin()
            first_label = period_events.loc[[first_id]].label.values[0]
            first_label_date = period_events.loc[[first_id]].date.min().to_pydatetime()
            Mp[0, nodes.index(first_label)] += 1 / nb_periods
            Mwait[0][nodes.index(first_label)].append(modulo_datetime(first_label_date, period))

        # Checking of all the rows and columns
        tol = 0.0001
        for i in range(n):
            # Row
            s_row = Mp[i, :].sum()
            if abs(s_row - 1) > tol:
                raise ValueError('The sum of the row {} is not 1 : {}'.format(i, s_row))
        Mp_dict[mu] = Mp

        # Find the best fitting probability distribution law for all the waiting times
        for i in range(n - 1):
            for j in range(n - 1):
                array = Mwait[i][j]
                if len(array) > 4:
                    array = np.asarray(array)
                    # dist_law = find_best_fitting_dist (array)

        # FIXME : The file should have a readable name including the time description
        filename = 'directed_graph_'
        filename = base64.urlsafe_b64encode(filename.encode('ascii', 'strict')).decode('ascii', 'strict')
        draw_directed_graph(nodes=nodes, matrix=Mp, filename=output_folder + filename, show_image=False)

    return Mp_dict, Mwait_dict


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


def draw_directed_graph(nodes, matrix, filename, show_image=False):
    '''
    Draw the directed graph corresponding and save the image
    :param nodes: Nodes of the graph
    :param matrix: Transition matrix
    :param filename: File path where the graph image is saved
    :param show_image: If True, plot the image
    :return:
    '''
    Q = pd.DataFrame(matrix)
    Q.columns = nodes
    Q.index = nodes
    edges_wts = _get_markov_edges(Q)

    # create graph object
    # TODO : Add a title to the graph
    G = nx.MultiDiGraph(label='qsdqsd')

    # nodes correspond to states
    G.add_nodes_from(nodes)
    print(f'Nodes:\n{G.nodes()}\n')

    # edges represent transition probabilities
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        if tmp_origin == "START PERIOD" or tmp_origin == "END PERIOD":
            G.add_node(tmp_origin, color='black', style='filled', fillcolor='red')
        else:
            G.add_node(tmp_origin, color='green')

        if tmp_destination == "START PERIOD" or tmp_destination == "END PERIOD":
            G.add_node(tmp_destination, color='black', style='filled', fillcolor='red')
        else:
            G.add_node(tmp_destination, color='green')
        G.add_edge(tmp_origin, tmp_destination, weight=v, penwidth=2 if v > 0.5 else 1, label=v,
                   color='blue' if v > 0.5 else 'black')
    print(f'Edges:')
    pprint(G.edges(data=True))

    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    nx.draw_networkx(G, pos)

    # create edge labels for jupyter plot but is not necessary
    edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.drawing.nx_pydot.write_dot(G, filename + '.dot')
    check_call(['dot', '-Tpng', filename + '.dot', '-o', filename + '.png'])
    if show_image:
        plt.clf()
        plt.axis('off')
        img = mpimg.imread(filename + '.png')
        plt.imshow(img)
        plt.show()


def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            if Q.loc[idx, col] != 0:
                edges[(idx, col)] = round(Q.loc[idx, col], 3)
    return edges


def find_best_fitting_dist(array):
    dist_names = ['norm']
    plt.figure()
    plt.hist(array)
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(array)
        pdf_fitted = dist.pdf(array, np.mean(array), np.std(array))
        plt.plot(array, pdf_fitted, '-o', label=dist_name)
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()
