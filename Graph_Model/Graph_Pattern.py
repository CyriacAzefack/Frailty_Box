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

sys.path.append(os.path.join(os.path.dirname(__file__)))
from Pattern_Discovery.Candidate_Study import modulo_datetime

sns.set_style("darkgrid")

class Graph:
    ID = 0
    NB_PERIODS_SLIDING_WINDOW = 5
    START_NODE = "START PERIOD"

    def __init__(self, nodes, labels, period, mu, sigma, prob_matrix, wait_matrix):
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
        
        self.nodes = nodes  # len(nodes) = n
        self.labels = labels # len(labels) = l
        self.period = period
        self.sliding_time_window = Graph.NB_PERIODS_SLIDING_WINDOW * period
        self.mu = mu
        self.sigma = sigma
        self.prob_matrix = prob_matrix  # size = (n x l)
        self.time_matrix = wait_matrix  # size = (n x l)
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
        return self.nodes

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
        :param nodes: Nodes of the graph
        :param prob_matrix: Transition matrix
        :param filename: File path where the graph image is saved
        :param debug: If True, plot the image
        :return:
        '''
        Q = pd.DataFrame(self.prob_matrix)
        Q.columns = self.labels
        Q.index = self.nodes
        edges_wts = self.get_markov_edges(Q)

        # create graph object
        G = nx.MultiDiGraph()

        # nodes correspond to states
        # G.add_nodes_from(self.nodes)

        G.graph['graph'] = {'label': title, 'labelloc': 't', 'fontsize': '20 ', 'fontcolor': 'blue',
                            'fontname': 'times-bold'}  # default
        # edges represent transition probabilities
        for k, v in edges_wts.items():
            tmp_origin, tmp_destination = k[0], k[1]

            # Add Origin Node
            if tmp_origin == Graph.START_NODE:
                G.add_node(tmp_origin, color='black', style='filled', fillcolor='red')
            else:
                G.add_node(tmp_origin, color='green')

            # Add Destination Node
            G.add_node(tmp_destination, color='green')

            # if tmp_origin != "END_PERIOD" or tmp_destination != "END_PERIOD":
            #     G.add_edge(tmp_origin, tmp_destination, weight=v, penwidth=2 if v > 0.5 else 1, label=v,
            #                color='blue' if v > 0.5 else 'black',
            #                headlabel=self.wait_matrix[self.nodes.index(tmp_origin)][self.nodes.index(tmp_destination)])
            # else:

            # Add Edge
            G.add_edge(tmp_origin, tmp_destination, weight=v, penwidth=2 if v > 0.5 else 1, label=v,
                           color='blue' if v > 0.5 else 'black')
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
                    node_name = col + '_' + str(lvl)
                    edges[(idx, node_name)] = round(Q.loc[idx, col], 3)
        return edges

    def simulate(self, start_date, end_date):

        self.extrapolate_time_evolution(start_date, end_date)

        result = pd.DataFrame(columns=["date", "label"])

        # The real 'start_date' is the beginning of the next period
        start_date_rel = modulo_datetime(start_date, self.period)
        if start_date_rel != 0:
            start_date = start_date + dt.timedelta(seconds=self.period.total_seconds() - start_date_rel)

        current_state = Graph.START_NODE
        current_date = start_date

        while current_date < end_date:
            state_index = self.nodes.index(current_state)
            row = self.prob_matrix[state_index, :]
            cs_row = np.cumsum(row)

            # Pick a random number between 0 -- 1
            rand = random()
            destination_index = None

            for index in range(len(self.nodes)):
                if rand <= cs_row[index]:
                    destination_index = index
                    break

            if not destination_index: # We reached one end of the graph
                destination_state = Graph.START_NODE # Come back to the beginning

                # We move to the start of the next period
                destination_date = current_date + dt.timedelta(
                    seconds=self.period.total_seconds() - modulo_datetime(current_date, self.period))

            else :
                destination_state = self.nodes[destination_index]

                # Now we have to compute the waiting time to get to the destination state
                tuple = self.time_matrix[state_index][destination_index]
                dist_name = tuple[0]
                param = tuple[1:][0]
                dist = getattr(st, dist_name)
                # Separate parts of parameters
                arg = param[:-2]
                loc = param[-2]
                scale = param[-1]

                while True:
                    try:
                        waiting_time = int(dist.rvs(loc=loc, scale=scale, *arg))
                        # Compute the destination date
                        destination_date = current_date + dt.timedelta(seconds=waiting_time)
                        # Add the event to the result,
                        destination_label = destination_state[0 : destination_state.rindex('_')] # We remove everything after the last '_'
                        result.loc[len(result)] = [destination_date, destination_label]
                        break
                    except:
                        print("OOOps ! Date Overflow. Let's try again...")

            current_state = destination_state
            current_date = destination_date

        return result

    def compute_time_evolution(self, data, nb_patterns):
        '''
        Compute the pattern time evolution by sliding a time window through the original data
        :param data:
        :return:
        '''

        n = len(self.nodes)
        time_evo_prob_matrix = [[pd.DataFrame(columns=['probability']) for j in range(n)] for i in
                                range(n)]  # Date as index
        time_evo_time_matrix = [[pd.DataFrame(columns=['mean_time', 'sigma_time']) for j in range(n)] for i in
                                range(n)]  # Date as index

        start_date = data.date.min().to_pydatetime()
        # We start at the beginning of the first period
        start_date = start_date - dt.timedelta(seconds=modulo_datetime(start_date, self.period))
        end_date = data.date.max().to_pydatetime()
        # We take the time window into account for the end_date
        end_date = end_date - self.sliding_time_window

        nb_periods = int((end_date - start_date).total_seconds() / self.period.total_seconds()) + 1

        period_index = 0
        current_start_date = start_date

        # Build the list of Pattern labels from the nodes
        labels = self.nodes[:]
        labels.remove(Graph.START_NODE)

        for i in range(len(labels)) :
            labels[i] = labels[i][0: labels[i].rindex('_')]  # We remove everything after the last '_'
        labels = list(set(labels))  # Remove duplicates

        while current_start_date < end_date:
            current_end_date = current_start_date + self.sliding_time_window
            time_description = {self.mu: self.sigma}
            mini_graph_pattern = Pattern2Graph.pattern2graph(data, labels=labels, time_description=time_description,
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
                for label in self.labels:
                    mini_i = mini_nodes.index(mini_node_i)
                    big_i = self.nodes.index(mini_node_i)
                    j = self.labels.index(label)

                    prob_df = time_evo_prob_matrix[big_i][j]
                    prob_df.loc[current_start_date] = [mini_prob_matrix[mini_i, j]]

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

    def extrapolate_time_evolution(self, start_date, end_date):
        '''
        Compute the probability transition matrix and the waiting time transition matrix by extrapolating time evolution
        :param start_date:
        :param end_date:
        :return: (prob_matrix, time_matrix)
        '''

        n = len(self.nodes)
        extr_prob_matrix = np.zeros((n, n))
        extr_time_matrix = [[[] for j in range(n - 1)] for i in range(n - 1)]  # Empty lists

        for i in range(n):
            fig, ax = plt.subplots()
            for j in range(n):
                df = self.time_evo_prob_matrix[i][j]
                txt = "--> [{}]".format(self.nodes[j])
                ax.plot_date(df.index, df.probability, label=txt, linestyle="-")
            ax.set_ylabel('Transition probability')
            ax.set_xlabel('Date')
            ax.set_title('From Node [{}]'.format(self.nodes[i]))
            ax.legend()
            plt.gcf().autofmt_xdate()
            plt.show()

    @staticmethod
    def node2label(txt):
        '''
        turn a node name to the label name
        :param str:
        :return:
        '''
        return txt[0: txt.rindex('_')]  # We remove everything after the last '_'
