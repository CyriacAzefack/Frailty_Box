import datetime as dt
import sys
from pprint import pprint
from random import random
from subprocess import check_call

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st

from xED.Candidate_Study import modulo_datetime


# from Graph_Model import Build_Graph

class Acyclic_Graph:
    ID = 0
    NB_PERIODS_SLIDING_WINDOW = dt.timedelta(days=30)
    START_NODE = "START PERIOD"
    NONE_NODE = "END"

    def __init__(self, nodes, labels, period, prob_matrix, wait_matrix, activities_duration):
        '''
        Initialization of one Pattern_Graph
        :param nodes: starts with "START_PERIOD"
        :param labels: labels of the Pattern
        :param period:
        :param mu:
        :param sigma:
        :param prob_matrix:
        :param wait_matrix:
        '''

        Acyclic_Graph.ID += 1

        self.graph_nodes = nodes  # len(graph_nodes) = n
        self.labels = labels  # len(labels) = l
        self.graph_labels = labels + [Acyclic_Graph.NONE_NODE]  # len (graph_labels) = l + 1
        self.period = period
        self.sliding_time_window = Acyclic_Graph.NB_PERIODS_SLIDING_WINDOW
        self.prob_matrix = prob_matrix  # size = (n x l+1)
        self.time_matrix = wait_matrix  # size = (n x l+1)
        self.activities_duration = activities_duration
        self.ID = Acyclic_Graph.ID
        self.graph = None
        self.time_evo_prob_matrix = None
        self.time_evo_time_matrix = None
        self.time_evo_activties_duration = None

    def display(self, output_folder, debug=False):

        filename = 'graph'

        title = 'Period : ' + str(self.period) + '\n'
        title += 'Activities : ' + '-'.join(self.labels) + '\n'

        self.draw_directed_graph(filename=output_folder + filename,
                                 title=title, debug=debug)

    def draw_directed_graph(self, filename, title, debug=False):
        '''
        Draw the directed graph corresponding and save the image
        :param graph_nodes: Nodes of the graph
        :param prob_matrix: Transition matrix
        :param filename: File path where the graph image is saved
        :param debug: If True, plot the image
        :return:
        '''
        Q = pd.DataFrame(self.prob_matrix)
        Q.columns = self.graph_labels
        Q.index = self.graph_nodes
        edges_wts = self.get_markov_edges(Q)

        # create graph object
        G = nx.MultiDiGraph()

        # graph_nodes correspond to states
        # G.add_nodes_from(self.graph_nodes)

        G.graph['graph'] = {'label': title, 'labelloc': 't', 'fontsize': '20 ', 'fontcolor': 'blue',
                            'fontname': 'times-bold'}  # default
        # edges represent transition probabilities
        for k, prob in edges_wts.items():
            tmp_origin, tmp_destination = k[0], k[1]

            # Add Origin Node
            if tmp_origin == Acyclic_Graph.START_NODE:
                G.add_node(tmp_origin, color='black', style='filled', fillcolor='red')
            else:
                G.add_node(tmp_origin, color='green')

            # Add Destination Node
            if tmp_destination == Acyclic_Graph.NONE_NODE:
                G.add_node(tmp_destination, color='black', style='filled', fillcolor='blue')
            else:

                G.add_node(tmp_destination, color='green')
                # G.add_node(tmp_destination, color='green')

            # Add Edge

            G.add_edge(tmp_origin, tmp_destination, weight=prob, penwidth=2 if prob > 0.5 else 1, label=prob,
                       color='blue' if prob > 0.5 else 'black')

            # Add Edge with waiting time label
            # G.add_edge(tmp_origin, tmp_destination, weight=v, penwidth=2 if v > 0.5 else 1, label=v,
            #                 color='blue' if v > 0.5 else 'black',
            #                 headlabel=self.time_matrix[self.graph_nodes.index(tmp_origin)][self.labels.index(destination_label)])
        if debug:
            print('Edges:')
            pprint(G.edges(data=True))
        # pprint(G.graph.get('graph', {}))

        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos)

        # create edge labels for jupyter plot but is not necessary
        edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        nx.drawing.nx_pydot.write_dot(G, filename + '.dot')
        check_call(['dot', '-Tpng', filename + '.dot', '-o', filename + '.png'])
        self.graph = G
        if debug:
            plt.clf()
            plt.axis('off')
            img = mpimg.imread(filename + '.png')
            plt.imshow(img)
            plt.show()

    def get_markov_edges(self, Q):
        '''
        Return the edges of the graph
        :param Q:
        :return:
        '''
        edges = {}
        for col in Q.columns:
            for idx in Q.index:
                if Q.loc[idx, col] != 0:
                    # Build the name of the transition
                    lvl = 0
                    if idx != Acyclic_Graph.START_NODE:
                        lvl = int(idx[idx.rindex('_') + 1:]) + 1

                    node_name = col
                    if col != Acyclic_Graph.NONE_NODE:
                        node_name = col + '_' + str(lvl)
                    edges[(idx, node_name)] = round(Q.loc[idx, col], 3)
        return edges

    def get_prob_matrix(self):
        return self.prob_matrix

    def get_time_matrix(self):
        return self.time_matrix

    def get_activities_duration(self):
        return self.activities_duration

    def get_nodes(self):
        return self.graph_nodes

    def simulate(self, start_date, end_date):
        """
        Simulate the current graph for the given period of time
        :param start_date:
        :param end_date:
        :return: simulation results
        """
        # self.extrapolate_time_evolution(start_date)
        # prob_matrix, time_matrix = self.get_date_status(start_date)
        # time_matrix = self.time_matrix

        prob_matrix, time_matrix, activities_duration = self.prob_matrix, self.time_matrix, self.activities_duration

        l = len(self.graph_labels)
        n = len(self.graph_nodes)

        simulation_results = pd.DataFrame(columns=["date", "end_date", "label"])

        current_state = Acyclic_Graph.START_NODE
        current_date = start_date

        while current_date < end_date:
            state_index = self.graph_nodes.index(current_state)
            row = prob_matrix[state_index, :]
            cs_row = np.cumsum(row)

            # Pick a random number between 0 -- 1
            rand = random()
            destination_index = None

            for index in range(l):
                if rand <= cs_row[index]:
                    destination_index = index
                    break

            if destination_index is None or destination_index == l - 1:  # We reached one end of the graph
                destination_state = Acyclic_Graph.START_NODE  # Come back to the beginning

                # We move to the start of the next period
                destination_date = current_date + dt.timedelta(
                    seconds=self.period.total_seconds() - modulo_datetime(current_date, self.period))

            else:
                # Turn the column label to a node name
                lvl = 0
                if current_state != Acyclic_Graph.START_NODE:
                    lvl = int(current_state[current_state.rindex('_') + 1:]) + 1
                destination_state = self.graph_labels[destination_index] + '_' + str(lvl)

                destination_label = destination_state[
                                    0: destination_state.rindex('_')]  # We remove everything after the last '_'

                # Waiting Time distribution
                waiting_tuple = time_matrix[state_index][destination_index]
                waiting_dist_name = waiting_tuple[0]
                waiting_param = waiting_tuple[1:][0]
                waiting_dist = getattr(st, waiting_dist_name)
                waiting_arg = waiting_param[:-2]
                waiting_loc = waiting_param[-2]
                waiting_scale = waiting_param[-1]

                # Activity duration Time distribution
                activity_duration = self.activities_duration[self.graph_nodes.index(destination_state)]
                duration_dist_name = activity_duration[0]
                duration_param = activity_duration[1:][0]
                duration_dist = getattr(st, duration_dist_name)
                duration_arg = duration_param[:-2]
                duration_loc = duration_param[-2]
                duration_scale = duration_param[-1]

                # Compute time intervals to avoid concurrency

                while True:

                    waiting_time = int(waiting_dist.rvs(loc=waiting_loc, scale=waiting_scale, *waiting_arg))
                    duration_time = int(duration_dist.rvs(loc=duration_loc, scale=duration_scale, *duration_arg))

                    try:
                        activity_start_date = current_date + dt.timedelta(seconds=waiting_time)
                        activity_end_date = activity_start_date + dt.timedelta(seconds=duration_time)

                        break
                    except ValueError as er:
                        print("OOOps ! Date Overflow. Let's try again...")

                simulation_results.loc[len(simulation_results)] = [activity_start_date, activity_end_date,
                                                                   destination_label]
                destination_date = activity_end_date

            current_state = destination_state
            current_date = destination_date

        return simulation_results

    def compute_time_evolution(self, data, nb_patterns):
        '''
        Compute the pattern time evolution by sliding a time window through the original data
        :param data:
        :return:
        '''

        n = len(self.graph_nodes)
        l = len(self.graph_labels)
        nb_days_per_periods = self.period.days

        start_date = data.date.min().to_pydatetime()
        # We start at the beginning of the first period
        start_date = start_date - dt.timedelta(seconds=modulo_datetime(start_date, self.period))
        end_date = data.date.max().to_pydatetime()

        time_evo_prob_matrix = [
            [pd.DataFrame(index=pd.date_range(start_date, end_date, freq=str(nb_days_per_periods) + 'D'),
                          columns=['probability']).fillna(0) for j in range(l)] for i in range(n)]  # Date as index

        time_evo_time_matrix = [
            [pd.DataFrame(index=pd.date_range(start_date, end_date, freq=str(nb_days_per_periods) + 'D'),
                          columns=['mean_time', 'sigma_time']).fillna(0) for j in range(l)] for i in
            range(n)]  # Date as index

        time_evo_activties_duration = [
            pd.DataFrame(index=pd.date_range(start_date, end_date, freq=str(nb_days_per_periods) + 'D'),
                         columns=['mean_time', 'sigma_time']).fillna(0) for i in range(n)]

        # We take the time window into account for the end_date
        end_date = end_date - self.sliding_time_window

        nb_periods = int((end_date - start_date).total_seconds() / self.period.total_seconds()) + 1

        period_index = 0
        current_start_date = start_date

        while current_start_date < end_date:
            current_end_date = current_start_date + self.sliding_time_window
            sub_graph = Build_Graph.build_graph(data, labels=self.labels, period=self.period,
                                                start_date=current_start_date, end_date=current_end_date,
                                                display_graph=False)

            if len(sub_graph) == 0:
                current_start_date += self.period
                continue

            sub_graph = sub_graph[0]  # Since we have one time_description, we have one graph to fetch
            sub_graph_prob_matrix = sub_graph.get_prob_matrix()
            sub_graph_time_matrix = sub_graph.get_time_matrix()
            sub_graph_activities_duration = sub_graph.get_activities_duration()
            sub_graph_nodes = sub_graph.get_nodes()

            # Fill time_evo_prob_matrix and time_evo_time_matrix
            for sub_graph_node_i in sub_graph_nodes:
                sub_i = sub_graph_nodes.index(sub_graph_node_i)
                i = self.graph_nodes.index(sub_graph_node_i)

                act_duration_df = time_evo_activties_duration[i]
                act_duration_df.loc[current_start_date] = sub_graph_activities_duration[sub_i]
                for label in self.graph_labels:

                    j = self.graph_labels.index(label)

                    prob_df = time_evo_prob_matrix[i][j]
                    prob_df.loc[current_start_date] = sub_graph_prob_matrix[sub_i, j]

                    if sub_graph_time_matrix[sub_i][j]:
                        time_df = time_evo_time_matrix[i][j]

                        # TODO : Manage with other distribution than 'norm'
                        time_df.loc[current_start_date] = [sub_graph_time_matrix[sub_i][j][1][0],
                                                           sub_graph_time_matrix[sub_i][j][1][1]]

            current_start_date += self.period
            evolution_percentage = round(100 * (period_index + 1) / nb_periods, 2)
            sys.stdout.write("\r{} %% of time evolution computed for the GRAPH N°{}/{}!!".format(evolution_percentage,
                                                                                                 self.ID, nb_patterns))
            sys.stdout.flush()
            period_index += 1
        sys.stdout.write("\n")

        self.time_evo_prob_matrix = time_evo_prob_matrix
        self.time_evo_time_matrix = time_evo_time_matrix
        self.time_evo_activties_duration = time_evo_activties_duration

    def get_date_status(self, start_date):
        # We start at the beginning of the first period
        start_date = start_date - dt.timedelta(seconds=modulo_datetime(start_date, self.period))

        n = len(self.graph_nodes)
        l = len(self.graph_labels)

        prob_matrix = [[self.time_evo_prob_matrix[i][j].loc[start_date, "probability"] for j in range(l)] for i in
                       range(n)]
        prob_matrix = np.array(prob_matrix)
        prob_matrix.reshape((n, l))

        time_matrix = [[self.time_evo_time_matrix[i][j].loc[start_date, ["mean_time", "sigma_time"]].values
                        for j in range(l)] for i in range(n)]
        for i in range(n):
            for j in range(l):
                time_matrix[i][j] = ('norm', time_matrix[i][j])

        activities_duration = [self.time_evo_activties_duration[i]]

        return prob_matrix, time_matrix

    @staticmethod
    def node2label(txt):
        '''
        turn a node name to the label name
        :param str:
        :return:
        '''
        return txt[0: txt.rindex('_')]  # We remove everything after the last '_'
