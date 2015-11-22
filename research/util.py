# -*- coding: utf-8 -*-
# pylint: disable=E1101

'''
Helper functions
'''
from __future__ import division, print_function

import numpy as np
import numpy.random as rand
import json
import os

from numpy.linalg import inv
from datetime import datetime


def rchoice(weights):
    '''Makes a (weighted) random choice.

    Given a vector of probabilities with a total sum of 1, this function
    returns the index of one element of the list with probability equal to
    this element's value. For example, given the vector [0.2, 0.5, 0.3], the
    probability that the function returns 0 is 20%, the probability that
    the functions returns 1 is 50% and the probability that it returns 2
    is 30%.

    Args:
        weights (1xN array): The vector with the probability of each index

    Returns:
        The randomly chosen index
    '''

    positive_probs = np.nonzero(weights)[0]
    s = 0.0
    r = rand.random()

    for i in positive_probs:
        s += weights[i]
        if r <= s:
            return i

    raise RuntimeError('Failed to make a random choice. Check input vector.')


def rowStochastic(A):
    '''Makes a matrix row (right) stochastic.

    Given a real square matrix, returns a new matrix which is right
    stochastic, meaning that each of its rows sums to 1.

    Args:
        A (NxN numpy array): The matrix to be converted

    Returns:
        A NxN numpy array which is row stochastic.
    '''

    return A / A.sum(axis=1, keepdims=True)


def randomSpanningTree(N, rand_weights=False):
    '''Creats a graph of N nodes connected by a random spanning tree.

    Args:
        N (int): Number of nodes

    Returns:
        A NxN numpy array representing the adjacency matrix of the graph.

    '''

    nodes = rand.permutation(N)
    A = np.zeros((N, N))

    for i in range(1, N):
        w = rand.random() if rand_weights else 1
        A[nodes[i - 1], nodes[i]] = w
        A[nodes[i], nodes[i - 1]] = w

    return A


def meanDegree(A):
    '''Calculates the mean degree of a graph.

    Args:
        A (NxN numpy array): The adjacency matrix of the graph

    Returns:
        The mean degree of the graph.

    '''
    B = np.empty_like(A)
    np.copyto(B, A)
    B[B > 0] = 1
    degrees = B.sum(axis=1)
    return np.mean(degrees)


def gnp(N, p, rand_weights=False, stochastic=False, verbose=False):
    '''Constructs an undirected connected G(N, p) network with random weights.

    Args:
        N (int): Number of nodes

        p (double): The probability that each vertice is created

        verbose (bool): Choose whether to print the size and the mean
        degree of the network

    Returns:
        A NxN numpy array representing the adjacency matrix of the graph.
    '''

    A = randomSpanningTree(N)
    for i in range(N):
        for j in range(N):
            r = rand.random()
            if r < p:
                w = rand.random() if rand_weights else 1
                A[i, j] = w
                A[j, i] = w

    if verbose:
        print('G(N,p) Network Created: N = {N}, Mean Degree = {deg}'.format(
              N=N, deg=meanDegree(A)))

    if stochastic:
        A = rowStochastic(A)

    return A


def expectedEquilibrium(A, s):
    '''Calculates the equilibrium of the Friedkin-Johnsen Model

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        s (1xN numpy array): Intrinsic beliefs vector

    Returns:
        ((I-A)^-1)Bs

    '''

    N = np.shape(A)[0]
    B = np.diag(np.diag(A))

    return np.dot(np.dot(inv(np.eye(N) - (A - B)), B), s)


def saveData(simid, N, max_rounds, eps, conv_stop, **kwargs):
    '''Save the initial conditions and the results of a simulation

    Args:
        simid (string): Unique simulation id starting with the name
        of the model and followed by a unique number

        N (int): Number of nodes

        max_rounds (int): Maximum number of rounds

        eps (float): Limit of convergence for the simulation

        conv_stop (bool): Specify whether the simulation stopped when the
        model converged

        **kwargs: Important arrays of the simulation that need to be saved.
        Those depend on the model but generally should contain the initial
        opinions, the opinions over time, the adjacency matrix etc. The names
        of the files are determined by the name of each dictionary entry so
        try to keep these consistent.

    '''

    # Various non-essential info about the simulation
    metadata = {
        'N': N,
        'max_rounds': max_rounds,
        'eps': eps,
        'conv_stop': conv_stop,
        'time': str(datetime.now())
    }

    filename = '{0}_metadata.txt'.format(simid)
    if os.path.isfile(filename):
        print('Files for simulation {0} already exist. Will'.format(simid),
              'append "_duplicate" string to results. Please change file'
              'names by hand.')
        simid += '_duplicate'
    with open(filename, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    # Save the arrays used in the simulation
    for name, array in kwargs.iteritems():
        np.savetxt('{simid}_{name}.txt'.format(simid=simid, name=name), array,
                   fmt='%6.4f')
