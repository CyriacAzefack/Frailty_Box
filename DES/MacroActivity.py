from DES.Activity import Activity
from Graph_Model.Acyclic_Graph import Acyclic_Graph
from Graph_Model.Pattern2Graph import *
from xED.Candidate_Study import modulo_datetime, find_occurrences
from xED.Pattern_Discovery import pick_dataset


def main():
    dataset = pick_dataset('aruba')

    # path = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/Simulation results 1/dataset_simulation_10.csv"
    # dataset = pick_custom_dataset(path)

    episode = ('bed_to_toilet', 'sleeping',)
    period = dt.timedelta(days=1)
    time_step = dt.timedelta(minutes=15)
    occurrences = find_occurrences(data=dataset, episode=episode)

    print('{} occurrences of the episode {}'.format(len(occurrences), episode))
    activity = MacroActivity(episode=episode, dataset=dataset, occurrences=occurrences, period=period,
                             time_step=time_step, display=True)

    date = dataset.date.min().to_pydatetime()
    simulation, duration = activity.simulate(date, 8)
    print(simulation)
    pass


class MacroActivity(Activity):

    def __init__(self, episode, dataset, occurrences, period, time_step, display=False, Tep=30):
        '''
        Create a Macro Activity
        :param labels:
        :param occurrences:
        :param period:
        :param time_step:
        :param display_histogram:
        '''
        Activity.__init__(self, episode, occurrences, period, time_step, display_histogram=display)

        # Find the events corresponding to the occurrences
        events = pd.DataFrame(columns=["date", "label", 'occ_id'])
        for index, occurrence in occurrences.iterrows():
            occ_start_date = occurrence["date"]
            end_date = occ_start_date + dt.timedelta(minutes=Tep)
            mini_data = dataset.loc[(dataset.label.isin(episode))
                                    & (dataset.date >= occ_start_date)
                                    & (dataset.date < end_date)].copy()
            mini_data.sort_values(["date"], ascending=True, inplace=True)
            mini_data.drop_duplicates(["label"], keep='first', inplace=True)
            mini_data['occ_id'] = index
            events = events.append(mini_data, ignore_index=True)

        self.activities = {}  # key: label, value: Activity

        for label in episode:
            label_events = events.loc[events.label == label].copy()
            label_activity = Activity(label=(label,), occurrences=label_events, period=period,
                                      time_step=time_step)
            self.activities[label] = label_activity

        # TODO : Build a graph for every time_step_id
        self.graph = self.build_activities_graph(episode=episode, events=events, period=period, display=display)

    def simulate(self, date, time_step_id):
        '''
        Simulate the activities on the macro-activity
        :param date:
        :param time_step_id:
        :return: simulations results, macro-activity duration
        '''
        simulation_result = self.graph.simulate(date, date + self.period / 2)

        duration = (simulation_result.end_date.max().to_pydatetime() - date).total_seconds()

        return simulation_result, duration

    def build_activities_graph(self, episode, events, period, display=False):
        '''
        Create a graph for the Macro_Activities
        :param episode:
        :param events:
        :param p:
        '''

        occurrence_ids = events['occ_id'].unique()

        # Build a list of occurrences events list to build the graph
        events_occurrences_lists = []

        graph_nodes_labels = []
        for occ_id in occurrence_ids:
            occ_list = []
            occ_df = events[events.occ_id == occ_id]
            i = 0
            for index, event_row in occ_df.iterrows():
                new_label = event_row['label'] + '_' + str(i)
                events.at[index, 'label'] = new_label
                occ_list.append(new_label)
                i += 1
            graph_nodes_labels += occ_list
            events_occurrences_lists.append(occ_list)

        # Set of graph_nodes for the graphs
        graph_nodes_labels = set(graph_nodes_labels)

        for occ_id in range(min(occurrence_ids), max(occurrence_ids) + 1):
            events_occurrences_lists.append(events.loc[events.occ_id == occ_id, 'label'].tolist())

        graph_nodes, graph_labels, prob_matrix = build_probability_acyclic_graph(list(episode), graph_nodes_labels,
                                                                                 events_occurrences_lists)

        # Build the time matrix
        events.loc[:, "relative_date"] = events.date.apply(
            lambda x: modulo_datetime(x.to_pydatetime(), period))
        events['is_last_event'] = events['occ_id'] != events['occ_id'].shift(-1)
        events['is_first_event'] = events['occ_id'] != events['occ_id'].shift(1)
        events['next_label'] = events['label'].shift(-1).fillna('_nan').apply(Acyclic_Graph.node2label)
        events['next_date'] = events['date'].shift(-1)
        events['inter_event_duration'] = events['next_date'] - events['date']
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

                    if from_node == Acyclic_Graph.START_NODE:  # START_NODE transitions
                        time_matrix[i][j] = ('norm', [0, 0])
                        continue

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
            if node != Acyclic_Graph.START_NODE:
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

        acyclic_graph = Acyclic_Graph(nodes=graph_nodes, labels=list(episode), period=period, prob_matrix=prob_matrix,
                                      wait_matrix=time_matrix, activities_duration=duration_matrix)

        if display:
            acyclic_graph.display(output_folder='./', debug=True)

        return acyclic_graph


if __name__ == '__main__':
    main()
