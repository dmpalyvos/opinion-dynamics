# -*- coding: utf-8 -*-
# pylint: disable=E1101
'''
Models of Opinion Formation
'''
import numpy as np
from numpy.linalg import norm
from viz import plotOpinions


def deGroot(A, s, max_rounds, eps=1e-6, plot=False, conv_stop=True):
    '''Simulates the DeGroot Model.

    Runs a maximum of max_rounds rounds of the DeGroot model. If the model
    converges sooner, the function returns. Chosing not to plot the opinions
    allows for lower memory consumption and faster runtime.

    Args:

        A (NxN numpy array): Adjacency matrix

        s (1xN numpy array): Initial opinions vector

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        plot (bool): Plot preference (default: False)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

    Returns:

        A txN vector of the opinions of the nodes over time

    '''
    max_rounds = int(max_rounds)

    N = len(s)
    max_rounds += 1  # Round 0 contains the initial opinions
    z = s
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in range(max_rounds):
        z = np.dot(A, z)
        opinions[t, :] = z
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('DeGroot converged after {t} rounds'.format(t=t))
            break

    if plot:
        plotOpinions(opinions[0:t, :], 'DeGroot')

    return opinions


def friedkinJohnsen(A, s, max_rounds, eps=1e-6, plot=False, conv_stop=True):
    '''Simulates the Friedkin-Johnsen (Kleinberg) Model.

    Runs a maximum of max_rounds rounds of the Friedkin-Jonsen model. If the
    model converges sooner, the function returns. Chosing not to plot the
    opinions allows for lower memory consumption and faster runtime. The
    stubborness matrix of the model is extracted from the diagonal of
    matrix A.

    Args:

        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        plot (bool): Plot preference (default: False)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

    Returns:

        A txN vector of the opinions of the nodes over time

    '''
    max_rounds = int(max_rounds)

    B = np.diag(np.diag(A))  # Stubborness matrix of the model
    A_model = A - B  # Adjacency matrix of the model

    N = len(s)
    max_rounds += 1  # Round 0 contains the initial opinions
    z = s
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in range(max_rounds):
        z = np.dot(A_model, z) + np.dot(B, s)
        opinions[t, :] = z
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Friedkin-Johnsen converged after {t} rounds'.format(t=t))
            break

    if plot:
        plotOpinions(opinions[0:t, :], 'Friedkin-Johnsen')

    return opinions[0:t, :]


def hk(s, op_eps, max_rounds, eps, plot=False, conv_stop=True):
    '''Simulates the model of Hegselmann-Krause

    This model does nto require an adjacency matrix. Connections between
    nodes are calculated depending on the proximity of their opinions.

    Args:

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        op_eps: ε parameter of the model

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        plot (bool): Plot preference (default: False)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

    Returns:

        A txN vector of the opinions of the nodes over time

    '''

    max_rounds = int(max_rounds)
    N = len(s)
    max_rounds += 1  # Round 0 contains the initial opinions
    z = s
    z_prev = s
    opinions = np.zeros((max_rounds, N))
    for t in range(max_rounds):
        Q = np.zeros((N, N))
        for i in range(N):
            neighbors_i = np.abs(z_prev - z_prev[i]) <= op_eps
            Q[i, neighbors_i] = 1
            z[i] = np.mean(z_prev[neighbors_i])
        opinions[t, :] = z
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Hegselmann-Krause converged after {t} rounds'.format(t=t))
            break

    if plot:
        plotOpinions(opinions[0:t, :], 'Hegselmann-Krause', dcolor=True)

    return opinions[0:t, :]
