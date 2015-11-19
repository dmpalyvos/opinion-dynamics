# -*- coding: utf-8 -*-
# pylint: disable=E1101
'''
Models of Opinion Formation
'''


from __future__ import division, print_function

import numpy as np
from numpy.linalg import norm
from viz import plotOpinions
from util import rchoice, rowStochastic


def preprocessArgs(s, max_rounds):
    '''Argument processing common for most models.

    Returns:
        N, z, max_rounds
    '''

    N = np.size(s)
    max_rounds = int(max_rounds) + 1  # Round 0 contains the initial opinions
    z = s.copy()

    return N, z, max_rounds


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

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in range(1, max_rounds):
        z = np.dot(A, z)
        opinions[t, :] = z
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('DeGroot converged after {t} rounds'.format(t=t))
            break

    if plot:
        plotOpinions(opinions[0:t+1, :], 'DeGroot')

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

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    B = np.diag(np.diag(A))  # Stubborness matrix of the model
    A_model = A - B  # Adjacency matrix of the model

    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = z

    for t in range(1, max_rounds):
        z = np.dot(A_model, z) + np.dot(B, s)
        opinions[t, :] = z
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Friedkin-Johnsen converged after {t} rounds'.format(t=t))
            break

    if plot:
        plotOpinions(opinions[0:t+1, :], 'Friedkin-Johnsen')

    return opinions[0:t+1, :]


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

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    z_prev = z.copy()
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    # Cannot allow zero rows because rchoice() will fail
    if np.size(np.nonzero(A.sum(axis=1))) != N:
        raise ValueError("Matrix A has one or more zero rows")

    for t in range(1, max_rounds):
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
        plotOpinions(opinions[0:t+1, :], 'Meet a friend')

    return opinions[0:t+1, :]


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

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    z_prev = z.copy()

    if np.size(np.nonzero(A.sum(axis=1))) != N:
        raise ValueError("Matrix A has one or more zero rows")

    for t in range(1, max_rounds):
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
        z_prev = z.copy()

    return t, z


def dynamic_weights(A, s, z, c, eps, p):
    '''Creates weighted edges based on the differences of opinions.

    The Generalized Asymmetric model works by using a dynamic weight matrix
    which is generated in each round. These weights are analogous to the
    proximity of the intrinsic belief of each node to the opinions of
    its neighbors.

    Args:
        A (NxN numpy array): Adjacency matrix (non-weighted)

        s (1xN numpy array): Intrinsic beliefs vector

        z (1xN numpy array): Current opinions vector

        c (string): Choose c function for the model. Possible choices are
        'simple', 'log', 'pow'.

        eps: Used in 'pow' func only

        p: Used in 'pow' func only

    Returns:
        The NxN matrix representing the weighted graph of the network.

    '''

    N = np.size(s)
    Q = np.zeros((N, N))

    functionDict = {
        'linear': lambda dist: 1 - dist,
        'log': lambda dist: 1 / np.log(dist + np.e),
        'pow': lambda dist: 1 / np.power(dist+eps, p)
    }

    cFunc = functionDict[c]

    for i in range(N):
        dist = np.abs(z - s[i])
        cResult = cFunc(dist)
        q = np.zeros(N)
        neighbors_i = A[i, :] > 0
        for node in np.flatnonzero(A[i, neighbors_i]):
            q[node] = cResult[node]/np.sum(cResult[neighbors_i])
        Q[i, :] = q

    return Q


def ga(A, B, s, max_rounds, eps=1e-6, plot=False, conv_stop=True, **kwargs):
    '''Simulates the Generalized Asymmetric Coevolutionary Game.

    This model does nto require an adjacency matrix. Connections between
    nodes are calculated depending on the proximity of their opinions.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)
        
        B (NxN numpy array): The stubborness of each node
        
        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        op_eps: ε parameter of the model

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        plot (bool): Plot preference (default: False)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        **kargs: Arguments c, eps, and p for dynamic_weights function (eps and
        p need to be specified only if c='pow') (default: c='linear')

    Returns:
        A txN vector of the opinions of the nodes over time

    '''

    # Check if c function was specified
    if kwargs:
        c = kwargs['c']
        # Extra parameters for pow function
        eps_c = kwargs.get('eps', 0.1)
        p_c = kwargs.get('eps', 2)
    else:
        # Otherwise use linear as default
        c = 'linear'
        eps_c = None
        p_c = None

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in range(1, max_rounds):
        Q = dynamic_weights(A, s, z, c, eps_c, p_c) + B
        Q = rowStochastic(Q)
        B_temp = np.diag(np.diag(Q))
        Q = Q - B_temp
        z = np.dot(Q, z) + np.dot(B_temp, s)
        opinions[t, :] = z
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('G-A converged after {t} rounds'.format(t=t))
            break

    if plot:
        plotOpinions(opinions[0:t+1, :], 'Hegselmann-Krause', dcolor=True)

    return opinions[0:t+1, :]


def hk(s, op_eps, max_rounds, eps=1e-6, plot=False, conv_stop=True):
    '''Simulates the model of Hegselmann-Krause.

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

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    z_prev = z.copy()
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in range(1, max_rounds):
        for i in range(N):
            # The node chooses only those with a close enough opinion
            friends_i = np.abs(z_prev - z_prev[i]) <= op_eps
            z[i] = np.mean(z_prev[friends_i])
        opinions[t, :] = z
        z_prev = z.copy()
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Hegselmann-Krause converged after {t} rounds'.format(t=t))
            break

    if plot:
        plotOpinions(opinions[0:t+1, :], 'Hegselmann-Krause', dcolor=True)

    return opinions[0:t+1, :]


def hk_local(A, s, op_eps, max_rounds, eps=1e-6, plot=False, conv_stop=True):
    '''Simulates the model of Hegselmann-Krause with an Adjacency Matrix

    Contraray to the standard Hegselmann-Krause Model, here we make use of
    an adjacency matrix that represents an underlying social structure
    independent of the opinions held by the members of the society.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

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

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    # All nodes must listen to themselves for the averaging to work
    A_model = A + np.eye(N)
    z_prev = z.copy()
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in range(1, max_rounds):
        for i in range(N):
            # Neighbors in the underlying social network
            neighbor_i = A_model[i, :] > 0
            opinion_close = np.abs(z_prev - z_prev[i]) <= op_eps
            # The node listens to those who share a connection with him
            # in the underlying network and also have an opinion
            # which is close to his own
            friends_i = np.logical_and(neighbor_i, opinion_close)
            z[i] = np.mean(z_prev[friends_i])
        opinions[t, :] = z
        z_prev = z.copy()
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Hegselmann-Krause (Local Knowledge) converged after {t} \
            rounds'.format(t=t))
            break

    if plot:
        plotOpinions(opinions[0:t+1, :], 'Hegselmann-Krause', dcolor=True)

    return opinions[0:t+1, :]
