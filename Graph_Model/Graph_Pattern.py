# -*- coding: utf-8 -*-
import datetime as dt
from pprint import pprint
from random import uniform
from subprocess import check_call

import Pattern2Graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from xED_Algorithm.Candidate_Study import modulo_datetime

sns.set_style("darkgrid")

class Graph_Pattern:
    ID = 0
    NB_PERIODS_SLIDING_WINDOW = 5
    START_NODE = "START PERIOD"
    END_NODE = "END PERIOD"

    def __init__(self, nodes, period, mu, sigma, prob_matrix, wait_matrix):
        '''
        Initializatoin of one Pattern_Graph
        :param nodes: starts with "START_PERIOD" and ends with "END_PERIOD"
        :param period:
        :param mu:
        :param sigma:
        :param prob_matrix:
        :param wait_matrix:
        '''

        self.nodes = nodes  # len(nodes) = n
        self.period = period
        self.sliding_time_window = Graph_Pattern.NB_PERIODS_SLIDING_WINDOW * period
        self.mu = mu
        self.sigma = sigma
        self.prob_matrix = prob_matrix  # size = (n x n)
        self.time_matrix = wait_matrix  # size = (n-1 x n-1)
        self.ID = Graph_Pattern.ID
        self.graph = None
        self.time_evo_prob_matrix = None
        self.time_evo_time_matrix = None

        Graph_Pattern.ID += 1

    def __repr__(self):
        '''
        :return: Text description of a graph pattern
        '''
        txt = 'Graph Pattern ID N°{}\n'.format(self.ID)
        txt += 'Nodes : {}\n'.format(self.nodes)
        txt += 'Period : {}\n'.format(self.period)
        txt += 'Mean Time : {}\n'.format(dt.timedelta(seconds=self.mu))
        txt += 'Std Time : {}\n'.format(dt.timedelta(seconds=self.sigma))

        return txt

    def get_prob_matrix(self):
        return self.prob_matrix

    def get_time_matrix(self):
        return self.time_matrix

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
        Q.columns = self.nodes
        Q.index = self.nodes
        edges_wts = self.get_markov_edges(Q)

        # create graph object
        G = nx.MultiDiGraph()

        # nodes correspond to states
        G.add_nodes_from(self.nodes)
        if debug:
            print('Nodes:\n{}\n'.format(G.nodes()))

        G.graph['graph'] = {'label': title, 'labelloc': 't', 'fontsize': '20 ', 'fontcolor': 'blue',
                            'fontname': 'times-bold'}  # default
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

            if tmp_origin != "END_PERIOD" or tmp_destination != "END_PERIOD":
                G.add_edge(tmp_origin, tmp_destination, weight=v, penwidth=2 if v > 0.5 else 1, label=v,
                           color='blue' if v > 0.5 else 'black',
                           headlabel=self.time_matrix[self.nodes.index(tmp_origin)][self.nodes.index(tmp_destination)])
            else:
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

    def get_markov_edges(self, Q):
        edges = {}
        for col in Q.columns:
            for idx in Q.index:
                if Q.loc[idx, col] != 0:
                    edges[(idx, col)] = round(Q.loc[idx, col], 3)
        return edges

    def simulate(self, start_date, end_date):

        self.extrapolate_time_evolution(start_date, end_date)

        result = pd.DataFrame(columns=["date", "label"])

        # The real 'start_date' is the beginning of the next period
        start_date_rel = modulo_datetime(start_date, self.period)
        if start_date_rel != 0:
            start_date = start_date + dt.timedelta(seconds=self.period.total_seconds() - start_date_rel)

        current_state = Graph_Pattern.START_NODE
        current_date = start_date

        while current_date < end_date:
            state_index = self.nodes.index(current_state)
            row = self.prob_matrix[state_index, :]
            cs_row = np.cumsum(row)

            # Pick a random number between 0 -- 1

            destination_index = None

            rand = uniform(0, cs_row[-1])

            for index in range(len(self.nodes)):
                if rand <= cs_row[index]:
                    destination_index = index
                    break

            destination_state = self.nodes[destination_index]

            if destination_state == Graph_Pattern.END_NODE:
                # We go to the end of the current period
                destination_date = current_date + dt.timedelta(
                    seconds=self.period.total_seconds() - modulo_datetime(current_date, self.period))
            elif destination_state == Graph_Pattern.START_NODE:
                destination_date = current_date
            else:
                # Now we have to compute the waiting time to get to the destination state
                tuple = self.time_matrix[state_index][destination_index]
                mu = tuple[0]
                sigma = tuple[1]

                while True:
                    try:
                        waiting_time = int(sigma * np.random.randn() + mu)
                        # Compute the destination date
                        destination_date = current_date + dt.timedelta(seconds=waiting_time)
                        # Add the event to the result
                        result.loc[len(result)] = [destination_date, destination_state]
                        break
                    except:
                        print("OOOps ! Date Overflow. Let's try again...")

            current_state = destination_state
            current_date = destination_date

        return result

    def compute_time_evolution(self, data):
        '''
        Compute the pattern time evolution by sliding a time window through the original data
        :param data:
        :return:
        '''

        n = len(self.nodes)
        time_evo_prob_matrix = [[pd.DataFrame(columns=['probability']) for j in range(n)] for i in
                                range(n)]  # Date as index
        time_evo_time_matrix = [[pd.DataFrame(columns=['mean_time', 'sigma_time']) for j in range(n - 1)] for i in
                                range(n - 1)]  # Date as index

        start_date = data.date.min().to_pydatetime()
        # We start at the beginning of the first period
        start_date = start_date - dt.timedelta(seconds=modulo_datetime(start_date, self.period))
        end_date = data.date.max().to_pydatetime()
        # We take the time window into account for the end_date
        end_date = end_date - self.sliding_time_window

        current_start_date = start_date

        nodes = self.nodes[1:-1]  # We remove 'START_NODE' and 'END_NODE'
        while current_start_date < end_date:
            current_end_date = current_start_date + self.sliding_time_window
            time_description = {self.mu: self.sigma}
            graph_pattern = Pattern2Graph.pattern2graph(data, labels=nodes, time_description=time_description,
                                                        period=self.period, start_date=current_start_date,
                                                        end_date=current_end_date, display_graph=False)

            if len(graph_pattern) == 0:
                current_start_date += self.period
                continue
            graph_pattern = graph_pattern[0]
            prob_matrix = graph_pattern.get_prob_matrix()
            time_matrix = graph_pattern.get_time_matrix()

            # Fill time_evo_prob_matrix
            for i in range(n):
                for j in range(n):
                    prob_df = time_evo_prob_matrix[i][j]
                    prob_df.loc[current_start_date] = [prob_matrix[i, j]]

                    if i < n - 1 and j < n - 1:
                        if time_matrix[i][j]:
                            time_df = time_evo_time_matrix[i][j]
                            time_df.loc[current_start_date] = [time_matrix[i][j][0], time_matrix[i][j][1]]

            current_start_date += self.period

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
