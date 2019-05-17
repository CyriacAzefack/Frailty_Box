import seaborn as sns

from Graph_Model import Acyclic_Graph
from Graph_Model.Pattern2Graph import *
from Pattern_Mining.Candidate_Study import modulo_datetime
from Pattern_Mining.Extract_Macro_Activities import compute_episode_occurrences
from Pattern_Mining.Pattern_Discovery import pick_dataset

sns.set_style('darkgrid')


# np.random.seed(1996)

def main():
    dataset = pick_dataset('hh101')

    # path = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/Simulation results 1/dataset_simulation_10.csv"
    # dataset = pick_custom_dataset(path)

    episode = ('dress', 'personal_hygiene', 'sleep')
    period = dt.timedelta(days=1)
    time_step = dt.timedelta(minutes=10)
    tep = 30

    occurrences, events = compute_episode_occurrences(dataset=dataset, episode=episode, tep=tep)

    print('{} occurrences of the episode {}'.format(len(occurrences), episode))
    activity = MacroActivity(episode=episode, occurrences=occurrences, events=events, period=period,
                             time_step=time_step, start_time_window_id=0, tep=tep, display=True)

    # start_date = occurrences.date.min().to_pydatetime()
    # future_date = start_date + dt.timedelta(days=30)
    #
    # stats = activity.get_stats(8)
    #
    # print(stats)
    # pickle.dump(activity.occurrences, open("occurrences.pickle", 'wb'))


class MacroActivity:
    ID = 0

    def __init__(self, episode, occurrences, events, period, time_step, start_time_window_id, tep=30, display=False):
        '''
        Creation of a Macro-Activity
        :param episode:
        :param occurrences:
        :param events:
        :param period:
        :param time_step:
        :param start_time_window_id:
        :param tep:
        :param display:
        '''

        print('\n')
        print("##################################################")
        print("## Creation of the Macro-Activity '{}' ##".format(episode))
        print("####################################################")
        print('\n')
        self.episode = episode
        self.period = period
        self.time_step = time_step
        self.index = np.arange(int(period.total_seconds() / time_step.total_seconds()) + 1)
        self.tep = dt.timedelta(minutes=30)

        # Initialize the count_histogram
        colums = ['tw_id'] + ['ts_{}'.format(ts_id) for ts_id in self.index]
        self.count_histogram = pd.DataFrame(columns=colums)  # For daily profiles

        self.duration_distrib = {}  # For activity duration laws

        for label in self.episode:
            # Stored as Gaussian Distribution
            self.duration_distrib[label] = pd.DataFrame(columns=['tw_id', 'mean', 'std'])

        self.occurrence_order = pd.DataFrame(
            columns=['tw_id'])  # For execution orders, the columns of the df are like '0132'

        self.add_time_window(occurrences, events, start_time_window_id, display=display)

        # self.build_histogram(occurrences, display=display)

        MacroActivity.ID += 1

    def preprocessing(self, occurrences, events):
        '''
        Preprocessing the occurrences and the events
        :param occurrences:
        :return:
        '''
        occurrences['relative_date'] = occurrences.date.apply(
            lambda x: modulo_datetime(x.to_pydatetime(), self.period))
        occurrences['time_step_id'] = occurrences['relative_date'] / self.time_step.total_seconds()
        occurrences['time_step_id'] = occurrences['time_step_id'].apply(math.floor)
        occurrences['activity_duration'] = occurrences.end_date - occurrences.date
        occurrences['activity_duration'] = occurrences['activity_duration'].apply(lambda x: x.total_seconds())

        events['activity_duration'] = events.end_date - events.date
        events['activity_duration'] = events['activity_duration'].apply(lambda x: x.total_seconds())

        return occurrences, events

    def build_histogram(self, occurrences, display=False):
        '''
        Build the Time distribution count_histogram on occurrences
        :param occurrences:
        :param display:
        :return:
        '''

        hist = occurrences.groupby(['time_step_id']).count()['date']

        # Create an index to have every time steps in the period

        hist = hist.reindex(self.index)

        hist.fillna(0, inplace=True)

        # hist = hist/len(hist)  # normalize

        if display:
            plt.bar(hist.index, hist.values)
            plt.title(
                '--'.join(self.episode) + '\nTime step : {} min'.format(round(self.time_step.total_seconds() / 60, 1)))
            plt.ylabel('Probability')
            plt.show()

        return hist

    def add_time_window(self, occurrences, events, time_window_id, display=False):
        """
        Update the ocurrence time Histogram and the duration laws history with the new time window data
        :param occurrences:
        :return:
        """
        occurrences, events = self.preprocessing(occurrences, events)

        # Update histogram count history
        hist = self.build_histogram(occurrences, display=display)
        self.count_histogram.at[time_window_id] = [time_window_id] + list(hist.values.T)

        print('Histogram count [UPDATED]')

        # Update duration laws
        for label in self.episode:
            label_df = events[events.label == label]
            duration_df = self.duration_distrib[label]

            mean_duration = np.mean(label_df.activity_duration)
            std_duration = np.std(label_df.activity_duration)
            duration_df.at[time_window_id] = [time_window_id, mean_duration, std_duration]

        print('Duration Gaussian Distribution [UPDATED]')

        # Update execution order

        occ_order_probability_dict = {}

        # Replace labels by their alphabetic identifier
        for label in self.episode:
            events.loc[events.label == label, 'alph_id'] = sorted(self.episode).index(label)

        for id, occ_row in occurrences.iterrows():
            start_date = occ_row.date
            end_date = start_date + self.tep
            arr = list(events.loc[(events.date >= start_date) & (events.date < end_date), 'alph_id'].values)
            # remove duplicates
            arr = list(dict.fromkeys(arr))
            occ_order_str = ''.join([str(int(x)) for x in arr])

            if occ_order_str in occ_order_probability_dict:
                occ_order_probability_dict[occ_order_str] += 1 / len(occurrences)
            else:
                occ_order_probability_dict[occ_order_str] = 1 / len(occurrences)

        self.occurrence_order.at[time_window_id, 'tw_id'] = time_window_id
        for occ_order_str, prob in occ_order_probability_dict.items():
            self.occurrence_order.at[time_window_id, occ_order_str] = prob

        self.occurrence_order.fillna(0, inplace=True)

        print('Execution Order Probability [UPDATED]')

    def simulate(self, date, time_step_id):
        '''
        Simulate the activities on the macro-activity
        :param date:
        :param time_step_id:
        :return: simulations results, macro-activity duration
        '''

        # TODO : Fix this graph MESSSS
        graph_sim = self.graph.simulate(date, date + self.Tep)

        # graph_sim['time_before_next_event'] = graph_sim['date'].shift(-1) - graph_sim['date']
        # graph_sim['time_before_next_event'] = graph_sim['time_before_next_event'].apply(lambda x: x.total_seconds())
        # graph_sim.fillna(0, inplace=True)

        # simulation_results = pd.DataFrame(columns=['label', 'date', 'end_date'])
        #
        # for _, row in graph_sim.iterrows():
        #     activity = self.activities[row.label]
        #     _, activity_duration = activity.simulate(date, time_step_id)
        #     end_date = date + dt.timedelta(seconds=activity_duration)
        #     simulation_results.loc[len(simulation_results)] = [row.label, date, end_date]
        #     date = date + dt.timedelta(seconds=row.time_before_next_event)
        #
        # duration = (simulation_results.end_date.max().to_pydatetime() - date).total_seconds()

        duration = (graph_sim.end_date.max().to_pydatetime() - date).total_seconds()

        return graph_sim, duration

    def build_activities_graph(self, episode, events, period, display=False):
        '''
        Create a graph for the Macro_Activities
        :param episode:
        :param events:
        :param period:
        :param display:
        '''

        occurrence_ids = events['occ_id'].unique()

        # Build a list of occurrences events list to build the graph
        events_occurrences_lists = []

        graph_nodes_labels = []
        for occ_id in occurrence_ids:
            occ_list = []
            occ_df = events[events.occ_id == occ_id]

            node_id = ''
            for index, event_row in occ_df.iterrows():
                node_id = node_id + str(episode.index(event_row['label']))
                new_label = event_row['label'] + '_' + node_id
                events.at[index, 'label'] = new_label
                occ_list.append(new_label)

            graph_nodes_labels += occ_list
            events_occurrences_lists.append(occ_list)

        # Set of graph_nodes for the graphs (the branches of the graph)
        graph_nodes_labels = set(graph_nodes_labels)

        # for occ_id in range(min(occurrence_ids), max(occurrence_ids) + 1):
        #     events_occurrences_lists.append(events.loc[events.occ_id == occ_id, 'label'].tolist())

        graph_nodes, graph_labels, prob_matrix = build_probability_acyclic_graph(list(episode), graph_nodes_labels,
                                                                                 events_occurrences_lists)

        # Build the time matrix
        events.loc[:, "relative_date"] = events.date.apply(
            lambda x: modulo_datetime(x.to_pydatetime(), period))
        events['is_last_event'] = events['occ_id'] != events['occ_id'].shift(-1)
        events['is_first_event'] = events['occ_id'] != events['occ_id'].shift(1)
        events['next_label'] = events['label'].shift(-1).fillna('_nan').apply(Acyclic_Graph.Acyclic_Graph.node2label)
        events['next_date'] = events['date'].shift(-1)
        events['inter_event_duration'] = events['next_date'] - events['date']
        events['inter_event_duration'] = events['inter_event_duration'].apply(lambda x: x.total_seconds())
        # events = events[events.is_last_event == False]

        n = len(graph_nodes)  # Nb rows of the prob matrix
        l = len(graph_labels)  # Nb columns of the prob matrix

        # n x l edges for waiting time transition laws
        time_matrix = [[[] for j in range(l)] for i in
                       range(n)]  # Empty lists, [[mean_time, std_time], ...] transition durations

        for position_index in range(n):
            for j in range(l - 1):  # We dont need the "END NODE"
                if prob_matrix[position_index][j] != 0:  # Useless to compute time for never happening transition
                    from_node = graph_nodes[position_index]
                    to_label = graph_labels[j]

                    if from_node == Acyclic_Graph.Acyclic_Graph.START_NODE:  # START_NODE transitions
                        time_matrix[position_index][j] = ('norm', [0, 0])
                        continue

                    time_df = events.loc[(events.label == from_node) & (events.next_label == to_label)]
                    inter_events_durations = time_df.inter_event_duration.values

                    # We remove NaN from the values
                    inter_events_durations = inter_events_durations[~np.isnan(inter_events_durations)]
                    inter_events_durations = clean_data_arrays(inter_events_durations)
                    time_matrix[position_index][j] = (
                    'norm', [np.mean(inter_events_durations), np.std(inter_events_durations)])

        events['activity_duration'] = events['end_date'] - events['date']
        events['activity_duration'] = events['activity_duration'].apply(lambda x: x.total_seconds())

        duration_matrix = [[] for i in range(n)]  # Empty lists, [[mean_time, std_time], ...] Activity duration
        for position_index in range(n):
            node = graph_nodes[position_index]
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
                    duration_matrix[position_index] = (
                    'norm', [np.mean(activity_durations), np.std(activity_durations)])

        acyclic_graph = Acyclic_Graph.Acyclic_Graph(nodes=graph_nodes, labels=list(episode), period=period,
                                                    prob_matrix=prob_matrix,
                                                    wait_matrix=time_matrix, activities_duration=duration_matrix)

        if display:
            acyclic_graph.display(output_folder='./', debug=True)

        return acyclic_graph


class ActivityObjectManager:
    """
    Manage the macro-activities created in the time windows
    """

    def __init__(self, name, period, time_step):
        """
        Initialisation of the Manager
        :param name:
        :param period:
        :param time_step:
        """
        self.name = name
        self.period = period
        self.time_step = time_step
        self.discovered_episodes = []  # All the episodes discovered until then
        self.activity_objects = {}  # The Activity/MacroActivity objects

    def update(self, episode, occurrences, events, time_window_id):
        """
        Update the Macro-Activity Object related to the macro-activity discovered if they already exist OR create a new Object
        :param macro_activities_list:
        :param time_window_id:
        :return:
        """

        set_episode = frozenset(episode)

        if set_episode not in self.discovered_episodes:  # Create a new Macro-Activity Object
            activity_object = MacroActivity(episode, occurrences, events, self.period, self.time_step, display=False)
            self.discovered_episodes.append(set_episode)
            self.activity_objects[set_episode] = activity_object
        else:
            activity_object = self.activity_objects[set_episode]
            activity_object.add_time_window(episode, occurrences, events, time_window_id)


if __name__ == '__main__':
    main()
