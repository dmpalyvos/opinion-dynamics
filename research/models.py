# -*- coding: utf-8 -*-
# pylint: disable=E1101
'''
Models of Opinion Formation
'''

# TODO: numpy copy

from __future__ import division

import numpy as np
from numpy.linalg import norm
from viz import plotOpinions
from util import rchoice


def deGroot(A, s, max_rounds, eps=1e-6, plot=False, conv_stop=True):
    '''Simulates the DeGroot Model.

    Runs a maximum of max_rounds rounds of the DeGroot model. If the model
    converges sooner, the function returns.

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

    N = np.size(s)
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
    model converges sooner, the function returns. The stubborness matrix of 
    the model is extracted from the diagonal of matrix A.

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

    N = np.size(s)
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


def meetFriend(A, s, max_rounds, eps=1e-6, plot=False, conv_stop=True):
    '''Simulates the Friedkin-Johnsen (Kleinberg) Model.

    Runs a maximum of max_rounds rounds of the "Meeting a Friend" model. If the
    model converges sooner, the function returns. The stubborness matrix of 
    the model is extracted from the diagonal of matrix A.

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

    N = np.size(s)
    max_rounds += 1  # Round 0 contains the initial opinions
    z = s
    z_prev = s
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    if np.size(np.nonzero(A.sum(axis=1))) != N:
        raise ValueError("Matrix A has one or more zero rows")

    for t in range(max_rounds):
        # Update the opinion for each node
        for i in range(N):
            r_i = rchoice(A[i, :])
            if r_i == i:
                op = s[i]
            else:
                op = z_prev[r_i]
            z[i] = (op + t*z_prev[i]) / (t+1)
        z_prev = z
        opinions[t, :] = z
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Meet a Friend converged after {t} rounds'.format(t=t))
            break

    if plot:
        plotOpinions(opinions[0:t, :], 'Meet a friend')

    return opinions[0:t, :]


def meetFriend_nomem(A, s, max_rounds, eps=1e-6, conv_stop=True):
    '''Simulates the Friedkin-Johnsen (Kleinberg) Model.

    Runs a maximum of max_rounds rounds of the "Meeting a Friend" model. If the
    model converges sooner, the function returns. The stubborness matrix of
    the model is extracted from the diagonal of matrix A. This function does
    not save the opinions overtime and cannot generate a plot. However it uses
    very little memory and is useful for determining the final opinions and
    the convergence time of the model.

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
        A tuple (t,z) where t is the convergence time and z the vector of the
        final opinions.

    '''

    max_rounds = int(max_rounds)

    N = np.size(s)
    max_rounds += 1  # Round 0 contains the initial opinions
    z = np.copy(s)
    z_prev = np.copy(s)

    if np.size(np.nonzero(A.sum(axis=1))) != N:
        raise ValueError("Matrix A has one or more zero rows")

    for t in range(max_rounds):
        # Update the opinion for each node
        for i in range(N):
            r_i = rchoice(A[i, :])
            if r_i == i:
                op = s[i]
            else:
                op = z_prev[r_i]
            z[i] = (op + t*z_prev[i]) / (t+1)
        if conv_stop and \
           norm(z - z_prev, np.inf) < eps:
            print('Meet a Friend converged after {t} rounds'.format(t=t))
            break
        z_prev = np.copy(z)

    return t, z


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
    N = np.size(s)
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
