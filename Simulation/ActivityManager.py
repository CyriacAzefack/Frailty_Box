import datetime as dt
import random
import sys
from subprocess import check_call

import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from Simulation.MacroActivity import MacroActivity


class ActivityManager:
    """
    Manage the macro-activities created in the time windows
    """

    def __init__(self, name, period, time_step, tep):
        """
        Initialisation of the Manager
        :param name:
        :param period:
        :param time_step:
        """
        self.name = name
        self.period = period
        self.tep = tep
        self.time_step = time_step
        self.discovered_episodes = []  # All the episodes discovered until then
        self.activity_objects = {}  # The Activity/MacroActivity objects
        self.mixed_occurrences = pd.DataFrame(columns=['date', 'end_date', 'label', 'tw_id'])
        self.last_time_window_id = 0

    def update(self, episode, occurrences, events, time_window_id=0, display=False):
        """
        Update the Macro-Activity Object related to the macro-activity discovered if they already exist OR create a
        new Object
        :param episode:
        :param occurrences:
        :param events:
        :param time_window_id:
        :return:
        """

        set_episode = frozenset(episode)

        if set_episode not in self.discovered_episodes:  # Create a new Macro-Activity Object
            activity_object = MacroActivity(episode=episode, period=self.period, time_step=self.time_step, tep=self.tep)

            self.discovered_episodes.append(set_episode)
            self.activity_objects[set_episode] = activity_object

        activity_object = self.activity_objects[set_episode]
        activity_object.add_time_window(occurrences=occurrences, events=events,
                                        time_window_id=time_window_id, display=display)

        occurrences['label'] = [str(set_episode) for _ in range(len(occurrences))]
        occurrences['tw_id'] = time_window_id

        self.mixed_occurrences = pd.concat(
            [self.mixed_occurrences, occurrences[['date', 'end_date', 'label', 'tw_id']]]).drop_duplicates(keep=False)

        self.mixed_occurrences.sort_values(['date'], inplace=True)

        self.last_time_window_id = time_window_id



    def get_macro_activity_from_name(self, set_episode):
        """
        return the MacroActivity Object related to the episode
        :param set_episode:
        :return:
        """

        # set_episode = frozenset(set_episode)

        return self.activity_objects[set_episode]

    def build_transition_matrix(self, time_window_id=0, display=False):
        """
        Build a transition matrix for all the available activities
        :return:
        """

        data = self.mixed_occurrences[self.mixed_occurrences.tw_id == time_window_id].copy()
        labels = data.label.unique()

        data['next_label'] = data['label'].shift(-1)
        data.dropna(inplace=True)

        matrix = pd.DataFrame(columns=labels, index=labels)

        for i_label in labels:
            i_data = data[data.label == i_label]
            nb_i_occ = len(i_data)
            for j_label in labels:
                nb_j_occ = len(i_data[i_data.next_label == j_label])
                matrix.loc[i_label, j_label] = nb_j_occ / nb_i_occ

        if display:
            plot_markov_chain(matrix, labels, threshold=0.0)

        return matrix

    def build_forecasting_models(self, train_ratio, display=False):
        """
        Build forecasting models for Macro-Activity parameters
        :param train_ratio: ratio of data used for training
        :param method: method used for the forecasting
        :param display:
        :return:
        """

        error_df = pd.DataFrame(columns=['episode', 'error'])
        i = 0
        for set_episode, macro_activity_object in self.activity_objects.items():
            i += 1
            # print('Forecasting Model for : {}!!'.format(set_episode))
            error = macro_activity_object.fit_history_count_forecasting_model(train_ratio=train_ratio,
                                                                              last_time_window_id=self.last_time_window_id,
                                                                              display=display)
            if error is None:
                error = -10
            error_df.at[len(error_df)] = [tuple(set_episode), error]

            sys.stdout.write(
                "\r{}/{} Macro-Activities Forecasting models done...".format(i, len(self.activity_objects)))
            sys.stdout.flush()
        sys.stdout.write("\n")


        plt.hist(list(error_df.error.values))
        plt.title('NMSE Distribution for all macro_activities forecasting models')
        plt.show()

        return error_df

    def get_activity_daily_profiles(self, time_window_id=0):
        """
        :param time_window_id:
        :return: a dict-like object of Macro-activities ADP
        """

        macro_ADPs = {}

        activities_count_histogram = pd.DataFrame()

        for set_episode, macro_activity in self.activity_objects.items():
            count_histogram = macro_activity.get_count_histogram(time_window_id=time_window_id)
            count_histogram.drop(['tw_id'], axis=1, inplace=True)
            count_histogram.index = [set_episode]
            activities_count_histogram = activities_count_histogram.append(count_histogram)

        activities_count_histogram = activities_count_histogram.div(activities_count_histogram.sum(axis=0), axis=1)
        activities_count_histogram.fillna(0, inplace=True)

        for index, row in activities_count_histogram.iterrows():
            macro_ADPs[index] = row.values

        return macro_ADPs

    def simulate(self, start_date, end_date, time_window_id=0):
        """
        Generate data between two dates using the model parameters for the selected time_window_id
        :param start_date: Start date of the simulation
        :param end_date: end date
        :param time_window_id: selected time window id
        :return:
        """

        simulated_dataset = pd.DataFrame(columns=['date', 'end_date', 'label'])
        previous_event = None
        current_date = start_date

        simulation_duration = (end_date - start_date).total_seconds()

        macro_ADPs = self.get_activity_daily_profiles(time_window_id=time_window_id)
        # transition_matrix = self.build_transition_matrix(time_window_id=time_window_id)

        while current_date < end_date:

            evolution_percentage = round(100 * ((current_date - start_date).total_seconds() / simulation_duration), 2)
            sys.stdout.write("\r{} %% of Simulation done!!".format(evolution_percentage))
            sys.stdout.flush()

            # Compute the time step id

            day_date = dt.datetime.combine(pd.to_datetime(current_date).date(), dt.datetime.min.time())

            time_step_id = math.ceil((current_date - day_date).total_seconds() \
                                     / self.time_step.total_seconds()) % int(self.period / self.time_step)

            # Choose the next Activity

            set_episodes = []
            scores_episodes = []

            for set_episode, macro_activity in self.activity_objects.items():
                # # Transition probability
                # if previous_event == None:
                #     prob_score = 1
                # else:
                #     prob_score = transition_matrix.loc[str(previous_event.get_set_episode())][str(set_episode)]

                # ADP score
                ADP_value = macro_ADPs[set_episode][time_step_id]
                # ADP_score = ADP_value if rand > ADP_value else 1

                # TODO : Linear function for Activity selection
                score = ADP_value

                set_episodes.append(set_episode)
                scores_episodes.append(score)

            scores_episodes = np.asarray(scores_episodes)

            scores_episodes = scores_episodes / sum(scores_episodes)

            scores_episodes = np.cumsum(scores_episodes)

            rand = random.random()

            chosen_set_episode = None
            for i in range(len(scores_episodes)):
                if rand <= scores_episodes[i]:
                    chosen_set_episode = set_episodes[i]
                    break

            # # Pick the episode with the max score
            # chosen_set_episode = max(scores.items(), key=operator.itemgetter(1))[0]

            if chosen_set_episode is None:  # Nothing happens
                current_date += self.time_step
                continue

            chosen_macro_activity = self.get_macro_activity_from_name(chosen_set_episode)

            # print("Time step ID : {}".format(time_step_id))
            # print("Chosen Activity : {} ".format(chosen_macro_activity))

            # Simulate the MACRO-ACTIVITY
            macro_activity_events = chosen_macro_activity.simulate(start_date=current_date, time_step_id=time_step_id,
                                                                   time_window_id=time_window_id)
            simulated_dataset = simulated_dataset.append(macro_activity_events)

            current_date = simulated_dataset.end_date.max().to_pydatetime()

            # previous_event = chosen_macro_activity

        sys.stdout.write("\n")

        return simulated_dataset


def plot_markov_chain(matrix, labels, threshold=0.1):
    """
    Plot the markov chain corresponding to the matrix
    :param matrix:
    :param labels:
    :return:
    """
    G = nx.MultiDiGraph()
    G.add_nodes_from(labels)
    print(f'Nodes:\n{G.nodes()}\n')

    edges_wts = {}
    for col in matrix.columns:
        for idx in matrix.index:
            if matrix.loc[idx, col] > threshold:
                edges_wts[(idx, col)] = round(matrix.loc[idx, col], 2)

    # edges represent transition probabilities
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    nx.draw_networkx(G, pos)

    edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    filename = 'markov'
    nx.drawing.nx_pydot.write_dot(G, filename + '.dot')
    check_call(['dot', '-Tpng', filename + '.dot', '-o', filename + '.png'])

    plt.clf()
    plt.axis('off')
    img = mpimg.imread(filename + '.png')
    plt.imshow(img)
    plt.title('Markov Chain between activities')
    plt.show()
