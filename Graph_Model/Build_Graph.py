import os
import sys

import scipy.stats as st

from Graph_Model import Acyclic_Graph

sys.path.append(os.path.join(os.path.dirname(__file__)))

from xED.Candidate_Study import *
from xED.Pattern_Discovery import *


def main():
    dataset_name = 'aruba'

    dataset = pick_dataset(dataset_name, nb_days=120)

    output = "../output/{}/ID_0".format(dataset_name)
    patterns = pickle.load(open(output + '/patterns.pickle', 'rb'))

    # Drop duplicates episodes
    patterns.drop_duplicates(subset=['Episode'], keep='first', inplace=True)

    # patterns = patterns[:2]

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    simulation_result = pd.DataFrame(columns=['date', 'end_date', 'label'])
    all_graphs = []

    # Add graphs coming from patterns
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

        start_time = t.process_time()  # To compute time spent building the graph

        pattern_graph = build_graph(data=dataset, labels=labels, period=period,
                                    start_date=start_date, end_date=end_date, output_directory=output_folder,
                                    display_graph=True)
        elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))

        print("\n")
        print("###############################")
        print("Patterns turned into graphs. Elapsed time : {}".format(elapsed_time))
        print("##############################")
        print("\n")

        all_graphs.append(pattern_graph)

    # Add graphs coming from unusual events

    data_left = pickle.load(open(output + '/data_left.pickle', 'rb'))

    activities = data_left.label.unique()


def build_graph(data, labels, period, start_date, end_date, Tep=30,
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

    data = data.loc[(data.date >= start_date) & (data.date <= end_date)].copy()

    # Find pattern occurrences
    occurrences = find_occurrences(data, tuple(labels), Tep)
    occurrences = occurrences.loc[(occurrences.date >= start_date) & (occurrences.date <= end_date)].copy()

    # Compute relative dates
    # occurrences.loc[:, "relative_date"] = occurrences.date.apply(
    #     lambda x: modulo_datetime(x.to_pydatetime(), period))

    # # Drop unexpected occurrences
    # occurrences["expected"] = occurrences["relative_date"].apply(
    #     lambda x: is_occurence_expected(x, {mu_time: sigma_time}, period, tolerance_ratio))
    # occurrences.dropna(inplace=True, axis=0)

    if len(occurrences) == 0:
        return None

    events = find_events_occurrences(data, labels, occurrences, period, Tep)

    # Find the numbers of columns needed for the graphs
    # Find the number max of events in a occurrence

    period_ids = events['period_id'].unique().astype(int)

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

    # Set of graph_nodes for the graphs
    graph_nodes_labels = set(graph_nodes_labels)

    for period_id in range(min(period_ids), max(period_ids) + 1):
        events_occurrences_lists.append(events.loc[events.period_id == period_id, 'label'].tolist())

    graph_nodes, graph_labels, prob_matrix = build_probability_acyclic_graph(labels, graph_nodes_labels,
                                                                             events_occurrences_lists)

    # Build the time matrix
    events.loc[:, "relative_date"] = events.date.apply(lambda x: modulo_datetime(x.to_pydatetime(), period))
    events['is_last_event'] = events['period_id'] != events['period_id'].shift(-1)
    events['is_first_event'] = events['period_id'] != events['period_id'].shift(1)
    events['next_label'] = events['label'].shift(-1).fillna('_nan').apply(Acyclic_Graph.Acyclic_Graph.node2label)
    events['next_date'] = events['date'].shift(-1)
    events['inter_event_duration'] = events['next_date'] - events['end_date']
    events['inter_event_duration'] = events['inter_event_duration'].apply(lambda x: x.total_seconds())
    # events = events[events.is_last_event == False]

    n = len(graph_nodes)  # Nb rows of the prob matrix
    l = len(graph_labels)  # Nb columns of the prob matrix

    # n x l edges for waiting time transition laws
    time_matrix = [[[] for j in range(l)] for i in
                   range(n)]  # Empty lists, [[mean_time, std_time], ...] transition durations

    for i in range(n):
        for j in range(l - 1):  # We dont need the "END NODE"
            if prob_matrix[i][j] != 0:  # Useless to compute time for never happening transition
                from_node = graph_nodes[i]
                to_label = graph_labels[j]

                if from_node == Acyclic_Graph.Acyclic_Graph.START_NODE:  # START_NODE transitions
                    time_df = events.loc[(events.next_label == to_label) & (events.is_first_event == True)]

                    inter_events_durations = time_df.relative_date.values

                else:
                    time_df = events.loc[(events.label == from_node) & (events.next_label == to_label)]
                    inter_events_durations = time_df.inter_event_duration.values

                # We remove NaN from the values
                inter_events_durations = inter_events_durations[~np.isnan(inter_events_durations)]
                inter_events_durations = clean_data_arrays(inter_events_durations)
                time_matrix[i][j] = ('norm', [np.mean(inter_events_durations), np.std(inter_events_durations)])

    events['activity_duration'] = events['end_date'] - events['date']
    events['activity_duration'] = events['activity_duration'].apply(lambda x: x.total_seconds())

    duration_matrix = [[] for i in range(n)]  # Empty lists, [[mean_time, std_time], ...] Activity duration
    for i in range(n):
        node = graph_nodes[i]
        if node != Acyclic_Graph.Acyclic_Graph.START_NODE:
            time_df = events.loc[events.label == node]
            activity_durations = time_df.activity_duration.values
            # We remove NaN from the values
            activity_durations = activity_durations[~np.isnan(activity_durations)]
            if len(activity_durations) > 0:
                activity_durations = clean_data_arrays(activity_durations)
                # plt.figure()
                # sns.distplot(activity_durations)
                # plt.show()
                duration_matrix[i] = ('norm', [np.mean(activity_durations), np.std(activity_durations)])

    acyclic_graph = Acyclic_Graph.Acyclic_Graph(graph_nodes, labels, period, prob_matrix, time_matrix, duration_matrix)

    if display_graph:
        acyclic_graph.display(output_folder=output_directory, debug=True)

    return acyclic_graph


def build_probability_acyclic_graph(labels, graph_nodes_labels, occurrence_list):
    '''
    Build the acyclic graph
    :param graph_labels: Labels of the pattern
    :param graph_nodes_labels: Pattern labels
    :param occurrence_list: List of list of ordered events per occurrence
    :return: Probability transition matrix size = nb(graph_nodes) x nb(labels)
    '''

    list_length = [len(l) for l in occurrence_list]

    nodes = [Acyclic_Graph.Acyclic_Graph.START_NODE]
    nodes += graph_nodes_labels

    graph_labels = labels + [Acyclic_Graph.Acyclic_Graph.NONE_NODE]

    n = len(nodes)  # Size of the transition matrix
    l = len(graph_labels)

    prob_matrix = np.zeros((n, l))

    # Deal with the beginning of the graph
    single_list = []
    non_occurrences = 0
    for list in occurrence_list:
        if list:
            single_list.append(list[0])
        else:
            non_occurrences += 1

    for node in set(single_list):
        label = Acyclic_Graph.Acyclic_Graph.node2label(node)
        prob_matrix[0][graph_labels.index(label)] = single_list.count(node) / len(occurrence_list)

    prob_matrix[0][l - 1] = non_occurrences / len(occurrence_list)

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
                label_2 = Acyclic_Graph.Acyclic_Graph.node2label(node_2)
                p = nb / nb_max
                if p != 0:
                    prob_matrix[nodes.index(node_1)][graph_labels.index(label_2)] = p

    # Checking the validity of the transition (sum output = 1 OR 0)

    tol = 0.001  # Error tolerance
    for i in range(n):
        # Row
        s_row = prob_matrix[i, :].sum()
        if abs(s_row - 1) > tol and s_row != 0:
            raise ValueError(
                'The sum of the probabilities transition from {} is neither 1 nor 0: {}'.format(nodes[i], s_row))

    return nodes, graph_labels, prob_matrix


def clean_data_arrays(data_array):
    """
    Clean the data by removing the outliers
    :param data_array:
    :return:
    """
    if len(data_array) < 2:
        return data_array

    eps = np.std(data_array) / 2
    db = DBSCAN(eps=eps, min_samples=2, p=1).fit(
        np.asarray(data_array).reshape(-1, 1))
    if len(db.components_) > 0:
        data_array = db.components_

    return data_array


def compute_activity_compatibility_matrix(data):
    activities = list(data.label.unique())
    n = len(activities)

    compatibility_matrix = np.zeros(shape=(n, n))

    for activity in activities:
        activ_df = data.loc[data.label == activity]
        non_activ_df = data.loc[data.label != activity]
        for _, activ_row in activ_df.iterrows():
            start_date = activ_row.date
            end_date = activ_row.end_date
            date_filter = ((non_activ_df.date < start_date) & (non_activ_df.end_date > start_date)) | (
                    (non_activ_df.end_date > end_date) & (non_activ_df.date < end_date)) | (
                                  (non_activ_df.date > start_date) & (non_activ_df.end_date < end_date))

            result = non_activ_df.loc[date_filter]
            if not result.empty:
                result_activities = result.label.unique()
                for result_activity in result_activities:
                    compatibility_matrix[activities.index(activity)][activities.index(result_activity)] += 1

    for i in range(n):
        for j in range(n):
            if compatibility_matrix[i][j] < 5:
                compatibility_matrix[i][j] = 0
            else:
                compatibility_matrix[i][j] = 1
    return activities, compatibility_matrix


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
    dist_list = ['norm', 'expon', 'lognorm', 'triang', 'beta', 'gaussian_kde']

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


if __name__ == "__main__":
    main()