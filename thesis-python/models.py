# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
from util import *

def deGroot(A, s, maxRounds, eps = 1e-6, plot = False):
    """Simulates the DeGroot Model.
    
    Runs a maximum of maxRounds rounds of the DeGroot model. If the model 
    converges sooner, the function returns. Chosing not to plot the opinions
    allows for lower memory consumption and faster runtime.
    
    Args:
        A (NxN numpy array): Adjacency matrix
        s (1xN numpy array): Initial opinions vector
        maxRounds (int): Maximum number of rounds to simulate 
        eps (double): Maximum difference between rounds before we assume that 
            the model has converged (default: 1e-6)
        plot (bool): Plot preference (default: False)
        
    Returns:
        A txN vector of the opinions of the nodes over time
        
    """
    
    # Preprocess A
    A = rowStochastic(A)
    
    N = len(s)
    maxRounds += 1 # Round 0 contains the initial opinions
    x = s
    opinions = np.zeros((maxRounds, N))
    opinions[0,:] = s
    
    for t in range(0,maxRounds):
        x = np.dot(A, x)
        opinions[t,:] = x
        if norm(opinions[t-1,:] - opinions[t,:], np.inf) < eps:
            print('DeGroot converged after {t} rounds'.format(t=t))
            break
    
    if plot:
        plotOpinions(opinions[0:t,:],'DeGroot')
    
    return opinions
    

    
def friedkinJohnsen(A, s, maxRounds, eps = 1e-6, plot = False):
    """Simulates the Friedkin-Johnsen (Kleinberg) Model.
    
    Runs a maximum of maxRounds rounds of the Friedkin-Jonsen model. If the 
    model converges sooner, the function returns. Chosing not to plot the 
    opinions allows for lower memory consumption and faster runtime. The 
    stubborness matrix of the model is extracted from the diagonal of 
    matrix A.
    
    Args:
        A (NxN numpy array): Adjacency matrix
        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector
        maxRounds (int): Maximum number of rounds to simulate 
        eps (double): Maximum difference between rounds before we assume that 
            the model has converged (default: 1e-6)
        plot (bool): Plot preference (default: False)
        
    Returns:
        A txN vector of the opinions of the nodes over time
        
    """

    # Preprocess A and extract stubborness matrix B
    A = rowStochastic(A)
    B = np.diag(np.diag(A))
    A = A - B
    N = len(s)
    maxRounds += 1 # Round 0 contains the initial opinions
    x = s
    opinions = np.zeros((maxRounds, N))
    opinions[0,:] = s
    
    for t in range(0, maxRounds):
        x = np.dot(A, x) + np.dot(B, s)
        opinions[t,:] = x
        if norm(opinions[t-1,:] - opinions[t,:], np.inf) < eps:
            print('Friedkin-Johnsen converged after {t} rounds'.format(t=t))
            break
    
    if plot:
        plotOpinions(opinions[0:t,:],'DeGroot')
    
    return opinions
