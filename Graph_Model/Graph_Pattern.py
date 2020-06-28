# -*- coding: utf-8 -*-
import datetime as dt
import os
import sys
from pprint import pprint
from random import random
from subprocess import check_call

import Pattern2Graph
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns

sns.set_style("darkgrid")


sys.path.append(os.path.join(os.path.dirname(__file__)))
from Pattern_Mining.Candidate_Study import modulo_datetime


class Graph:
    ID = 0
    NB_PERIODS_SLIDING_WINDOW = dt.timedelta(days=5)
    START_NODE = "START PERIOD"
    NONE_NODE = "END"
    COMPATIBILITY_MATRIX = None
    ACTIVITIES = None

    def __init__(self, nodes, labels, period, mu, sigma, prob_matrix, wait_matrix, activities_duration):
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

        Graph.ID += 1

        self.graph_nodes = nodes  # len(graph_nodes) = n
        self.labels = labels # len(labels) = l
        self.graph_labels = labels + [Graph.NONE_NODE]  # len (graph_labels) = l + 1
        self.period = period
        self.sliding_time_window = Graph.NB_PERIODS_SLIDING_WINDOW
        self.mu = mu
        self.sigma = sigma
        self.prob_matrix = prob_matrix  # size = (n x l+1)
        self.time_matrix = wait_matrix  # size = (n x l+1)
        self.activities_duration = activities_duration
        self.ID = Graph.ID
        self.graph = None
        self.time_evo_prob_matrix = None
        self.time_evo_time_matrix = None



    def __repr__(self):
        '''
        :return: Text description of a graph pattern
        '''
        txt = 'Graph Pattern ID N°{}\n'.format(self.ID)
        txt += 'Labels : {}\n'.format(self.labels)
        txt += 'Period : {}\n'.format(self.period)
        txt += 'Mean Time : {}\n'.format(dt.timedelta(seconds=self.mu))
        txt += 'Std Time : {}\n'.format(dt.timedelta(seconds=self.sigma))

        return txt

    def get_prob_matrix(self):
        return self.prob_matrix

    def get_time_matrix(self):
        return self.time_matrix

    def get_nodes(self):
        return self.graph_nodes

    def display(self, output_folder, debug=False):
        td = dt.timedelta(seconds=self.mu)
        filename = 'directed_graph_{}_{}_{}'.format(td.days, td.seconds // 3600, (td.seconds // 60) % 60)

        title = 'Period : ' + str(self.period) + '\n'
        title += 'Mean Time : ' + str(td) + '  ---   Std Time : ' + str(dt.timedelta(seconds=self.sigma))

        self.draw_directed_graph(filename=output_folder + filename,
                                 title=title, debug=debug)

    def get_index(self):
        return self.ID

    def draw_directed_graph(self, filename, title, debug=False):
        '''
        Draw the directed graph corresponding and save the image
        :param graph_nodes: Nodes of the graph
        :param prob_matrix: Transition distance_matrix
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
            if tmp_origin == Graph.START_NODE:
                G.add_node(tmp_origin, color='black', style='filled', fillcolor='red')
            else:
                G.add_node(tmp_origin, color='green')

            # Add Destination Node
            if tmp_destination == Graph.NONE_NODE:
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
            #                 headlabel=self.time_matrix[self.graph_nodes.period_ts_index(tmp_origin)][self.labels.period_ts_index(destination_label)])
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
                    if idx != Graph.START_NODE:
                        lvl = int(idx[idx.rindex('_')+1:]) + 1

                    node_name = col
                    if col != Graph.NONE_NODE:
                        node_name = col + '_' + str(lvl)
                    edges[(idx, node_name)] = round(Q.loc[idx, col], 3)
        return edges

    def simulate(self, previous_data, start_date, end_date):
        """
        Simulate the current graph for the given period of time
        :param start_date:
        :param end_date:
        :return:
        """
        # self.extrapolate_time_evolution(start_date)
        # prob_matrix, time_matrix = self.get_date_status(start_date)
        # time_matrix = self.time_matrix

        prob_matrix, time_matrix = self.prob_matrix, self.time_matrix

        l = len(self.graph_labels)

        result = pd.DataFrame(columns=["date", "end_date", "label"])

        # The real 'start_date' is the beginning of the next period
        start_date_rel = modulo_datetime(start_date, self.period)
        if start_date_rel != 0:
            start_date = start_date + dt.timedelta(seconds=self.period.total_seconds() - start_date_rel)

        current_state = Graph.START_NODE
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
                destination_state = Graph.START_NODE # Come back to the beginning

                # We move to the start of the next period
                destination_date = current_date + dt.timedelta(
                    seconds=self.period.total_seconds() - modulo_datetime(current_date, self.period))

            else:

                # Turn the column label to a node name
                lvl = 0
                if current_state != Graph.START_NODE:
                    lvl = int(current_state[current_state.rindex('_') + 1:]) + 1
                destination_state = self.graph_labels[destination_index] + '_' + str(lvl)

                destination_label = destination_state[
                                    0: destination_state.rindex('_')]  # We remove everything after the last '_'

                tuple = time_matrix[state_index][destination_index]
                dist_name = tuple[0]
                param = tuple[1:][0]
                dist = getattr(st, dist_name)

                activity_duration = self.activities_duration[self.graph_nodes.index(destination_state)]
                duration_dist_name = activity_duration[0]
                duration_param = activity_duration[1:][0]
                duration_dist = getattr(st, duration_dist_name)
                # Separate parts of parameters

                compatibabilty_respected = False

                nb_ite = 10
                activity_start_date, activity_end_date = None, None
                while not compatibabilty_respected & nb_ite > 0:
                    nb_ite -= 1
                    activity_start_date, activity_end_date = self.generate_activity(current_date, waiting_dist=dist,
                                                                                    waiting_param=param,
                                                                                    duration_dist=duration_dist,
                                                                                    duration_param=duration_param)
                    compatibabilty_respected = True

                    date_filter = ((previous_data.date < activity_start_date) & (
                            previous_data.end_date > activity_start_date)) | (
                                          (previous_data.end_date > activity_end_date) & (
                                          previous_data.date < activity_end_date)) | (
                                          (previous_data.date > activity_start_date) & (
                                          previous_data.end_date < activity_end_date))

                    filtered_data = previous_data.loc[date_filter]
                    if not filtered_data.empty:
                        filtered_activities = filtered_data.label.unique()
                        for filtered_activity in filtered_activities:
                            if Graph.COMPATIBILITY_MATRIX[Graph.ACTIVITIES.index(destination_label)][
                                Graph.ACTIVITIES.index(filtered_activity)] == 0:
                                compatibabilty_respected = False

                if nb_ite > 0:
                    result.loc[len(result)] = [activity_start_date, activity_end_date, destination_label]
                destination_date = activity_start_date


            current_state = destination_state
            current_date = destination_date

        return result

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
                          columns=['probability']).fillna(0) for j in range(l)] for i in
            range(n)]  # Date as period_ts_index

        time_evo_time_matrix = [
            [pd.DataFrame(index=pd.date_range(start_date, end_date, freq=str(nb_days_per_periods) + 'D'),
                          columns=['mean_time', 'sigma_time']).fillna(0) for j in range(l)] for i in
            range(n)]  # Date as period_ts_index

        # We take the time window into account for the end_date
        end_date = end_date - self.sliding_time_window

        nb_periods = int((end_date - start_date).total_seconds() / self.period.total_seconds()) + 1

        period_index = 0
        current_start_date = start_date

        while current_start_date < end_date:
            current_end_date = current_start_date + self.sliding_time_window
            time_description = {self.mu: self.sigma}
            mini_graph_pattern = Pattern2Graph.pattern2graph(data, labels=self.labels, time_description=time_description,
                                                             period=self.period, start_date=current_start_date,
                                                             end_date=current_end_date, display_graph=False)

            if len(mini_graph_pattern) == 0:
                current_start_date += self.period
                continue

            mini_graph_pattern = mini_graph_pattern[0] # Since we have one time_description, we have one graph to fetch
            mini_prob_matrix = mini_graph_pattern.get_prob_matrix()
            mini_time_matrix = mini_graph_pattern.get_time_matrix()
            mini_nodes = mini_graph_pattern.get_nodes()

            # Fill time_evo_prob_matrix and time_evo_time_matrix
            for mini_node_i in mini_nodes:
                for label in self.graph_labels:
                    mini_i = mini_nodes.index(mini_node_i)
                    big_i = self.graph_nodes.index(mini_node_i)
                    j = self.graph_labels.index(label)

                    prob_df = time_evo_prob_matrix[big_i][j]
                    prob_df.loc[current_start_date] = mini_prob_matrix[mini_i, j]

                    if mini_time_matrix[mini_i][j]:
                        time_df = time_evo_time_matrix[big_i][j]

                        # TODO : Manage with other distribution than 'norm'
                        time_df.loc[current_start_date] = [mini_time_matrix[mini_i][j][1][0],
                                                           mini_time_matrix[mini_i][j][1][1]]

            current_start_date += self.period
            evolution_percentage = round(100 * (period_index + 1) / nb_periods, 2)
            sys.stdout.write("\r{} %% of time evolution computed for the GRAPH N°{}/{}!!".format(evolution_percentage,
                                                                                                 self.ID, nb_patterns))
            sys.stdout.flush()
            period_index += 1
        sys.stdout.write("\n")

        self.time_evo_prob_matrix = time_evo_prob_matrix
        self.time_evo_time_matrix = time_evo_time_matrix

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

        return prob_matrix, time_matrix

    def extrapolate_time_evolution(self, start_date):
        '''
        Compute the probability transition distance_matrix and the waiting time transition distance_matrix by extrapolating time evolution
        :param start_date:
        :param end_date:
        :return: (prob_matrix, time_matrix)
        '''

        n = len(self.graph_nodes)
        l = len(self.graph_labels)
        extr_prob_matrix = np.zeros((n, l))
        extr_time_matrix = [[[] for j in range(l)] for i in range(n)]  # Empty lists

        for i in range(n):
            fig, (ax1, ax2) = plt.subplots(2)
            for j in range(l):
                txt = "--> [{}]".format(self.graph_labels[j])
                df = self.time_evo_prob_matrix[i][j]
                ax1.plot_date(df.index, df.probability, label=txt, linestyle="-")
                df = self.time_evo_time_matrix[i][j]
                ax2.plot_date(df.index, df.mean_time / 60, label=txt, linestyle="-")
            ax1.title.set_text('From Node [{}]\nProbability transition'.format(self.graph_nodes[i]))
            ax1.set_ylabel('Transition probability')
            ax2.set_ylabel('Mean Time (min)')
            ax2.set_xlabel('Date')
            ax2.set_title('Waiting Time transition')
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper left")
            plt.gcf().autofmt_xdate()
            plt.show()

    def generate_activity(self, date, waiting_dist, waiting_param, duration_dist, duration_param):
        """
        Generate a date interval respecting the constraints
        :param date:
        :param waiting_dist:
        :param waiting_param:
        :param duration_dist:
        :param duration_param:
        :return:
        """

        waiting_arg = waiting_param[:-2]
        waiting_loc = waiting_param[-2]
        waiting_scale = waiting_param[-1]

        duration_arg = duration_param[:-2]
        duration_loc = duration_param[-2]
        duration_scale = duration_param[-1]

        start_date = None
        end_date = None
        while True:

            waiting_time = int(waiting_dist.rvs(loc=waiting_loc, scale=waiting_scale, *waiting_arg))

            duration = -1
            while duration < 0:
                duration = int(duration_dist.rvs(loc=duration_loc, scale=duration_scale, *duration_arg))

            try:
                start_date = date + dt.timedelta(seconds=waiting_time)
                end_date = start_date + dt.timedelta(seconds=duration)
                break;
            except ValueError as e:
                print(e)
                print("OOOps ! Date Overflow. Let's try again...")

        return start_date, end_date

    @staticmethod
    def node2label(txt):
        '''
        turn a node name to the label name
        :param str:
        :return:
        '''
        return txt[0: txt.rindex('_')]  # We remove everything after the last '_'

    @staticmethod
    def set_compatibility_matrix(activities, matrix):
        Graph.ACTIVITIES = activities
        Graph.COMPATIBILITY_MATRIX = matrix
