# -*- coding: utf-8 -*-
import datetime as dt
import os
import sys
from pprint import pprint
from random import random
from subprocess import check_call

import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.append(os.path.join(os.path.dirname(__file__)))
from Pattern_Discovery.Candidate_Study import modulo_datetime


class Graph_Pattern:
    ID = 0
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
        self.nodes = nodes
        self.period = period
        self.mu = mu
        self.sigma = sigma
        self.prob_matrix = prob_matrix
        self.wait_matrix = wait_matrix
        self.ID = Graph_Pattern.ID
        Graph_Pattern.ID += 1

    def display(self, output_folder, debug=False):
        td = dt.timedelta(seconds=self.mu)
        filename = 'directed_graph_{}_{}_{}'.format(td.days, td.seconds // 3600, (td.seconds // 60) % 60)

        title = 'Period : ' + str(self.period) + '\n'
        title += 'Mean Time : ' + str(td) + '  ---   Std Time : ' + str(dt.timedelta(seconds=self.sigma))

        self.draw_directed_graph(filename=output_folder + filename,
                                 title=title, debug=debug)

    def getIndex(self):
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
                           headlabel=self.wait_matrix[self.nodes.index(tmp_origin)][self.nodes.index(tmp_destination)])
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
        if debug:
            plt.clf()
            plt.axis('off')
            img = mpimg.imread(filename + '.png')
            plt.imshow(img)
            plt.show()

    def get_markov_edges(self, Q):
        edges = {}
        for col in Q.columns:
            for idx in Q.index:
                if Q.loc[idx, col] != 0:
                    edges[(idx, col)] = round(Q.loc[idx, col], 3)
        return edges

    def simulate(self, start_date, end_date):
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
            rand = random()
            destination_index = None

            for index in range(len(self.nodes)):
                if rand < cs_row[index]:
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
                tuple = self.wait_matrix[state_index][destination_index]
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
                        # Add the event to the result
                        result.loc[len(result)] = [destination_date, destination_state]
                        break
                    except:
                        print("OOOps ! Date Overflow. Let's try again...")




            current_state = destination_state
            current_date = destination_date

        return result
