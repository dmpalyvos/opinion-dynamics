# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as rand

def rowStochastic(A):
    """Makes a matrix row (right) stochastic.
    
    Given a real square matrix, returns a new matrix which is right 
    stochastic, meaning that each of its rows sums to 1.
    
    Args: 
        A (NxN numpy array): The matrix to be converted
    
    Returns:
        A NxN numpy array which is row stochastic.
    """
    
    return A/A.sum(axis=1, keepdims = True)
    
    
def randomSpanningTree(N):
    """Creats a graph of N nodes connected by a random spanning tree.
    
    Args:
        N (int): Number of nodes
    
    Returns:
        A NxN numpy array representing the adjacency matrix of the graph.
    """

    nodes = rand.permutation(N)
    A = np.zeros((N, N))
    
    for i in range(1, N):
        w = rand.random()
        A[nodes[i-1],nodes[i]] = w
        A[nodes[i],nodes[i-1]] = w
        
    return A
    
def gnp(N, p, connected = True):
    """Constructs an undirected connected G(N, p) network with random weights.
    
    Args:
        N (int): Number of nodes
        p (double): The probability that each vertice is created
    
    Returns:
        A NxN numpy array representing the adjacency matrix of the graph.
    """
    
    A  = np.zeros((N,N))
    A += randomSpanningTree(N)
    for i in range(N):
        for j in range(N):
            r = rand.random()
            if r < p:
                w = rand.random()
                A[i, j] = w
                A[j, i] = w
                
    
    return A
    

def plotOpinions(opinions, title=''):
    """Creates a plot of the opinions over time
    
    Args:
        opinions (txN vector): Vector of the opinions over time
        title (string): Optional title of the plot (default: '')
    
    """
    maxRounds = np.shape(opinions)[0]
    opinionNumber = np.shape(opinions)[1]
    for t in range(0,opinionNumber):
        plt.plot(range(0,maxRounds),opinions[:,t], linewidth = 0.5)
    plt.ylabel('Opinion')
    plt.xlabel('t')
    plt.title(title)
    plt.show()