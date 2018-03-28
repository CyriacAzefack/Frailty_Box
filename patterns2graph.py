# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:22:58 2018

@author: cyriac.azefack
"""
from candidate_study import *


def patterns2graph(data, labels, description, tolerance_ratio=2, Tep=30):
    '''
    :param data: Input dataset
    :param labels: list of labels included in the pattern
    :param description: description of the pattern {mu1 : sigma1, mu2 : sigma2, ...}
    :param tolerance_ratio: tolerance ratio to get the expectect occurrences
    :param Tep : [in Minutes] Maximal time interval between events in an episode occurrence. Should correspond to the maximal duration of the ADLs.
    :return: A transition probability matrix and a transition waiting time matrix for each component of the description
    '''

    Mp_dict = {}
    Mw_dict = {}
    nodes = ['START NODE'] + labels + ['END NOE']
    n = len(nodes)

    occurrences = find_occurrences(data, labels, Tep)
    for mu, sigma in description.items():
        Mp = np.zeros((n, n))

    return None
