# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
import util

def deGroot(A, s, maxRounds, eps = 1e-6, plot = False):
    """Simulates the DeGroot Model.
    
    Runs a maximum of maxRounds rounds of the DeGroot model. If the model 
    converges sooner, the function returns. Chosing not to plot the opinions
    allows for lower memory consumption and faster runtime.
    
    Args:
        A (NxN numpy array): Adjacency matrix.
        s (Nx1 numpy array): Initial opinions vector.
        eps (double): Maximum difference between rounds before we assume that 
            the model has converged.
        maxRounds (int): Maximum number of rounds to simulate.
        plot (bool): Plot preference
        
    Returns:
        A vector containing the final opinions of the nodes.
        
    """

    N = len(s)
    maxRounds += 1 # Round 0 contains the initial opinions
    x = s
    opinions = np.zeros((maxRounds,N)) if plot else None
    opinions[0,:] = s
    
    for t in range(0,maxRounds):
        x = np.dot(A,x)
        if plot:
            opinions[t,:] = x
        if norm(opinions[t-1,:] - opinions[t,:],np.inf) < eps:
            print('DeGroot converged after {t} rounds'.format(t=t))
            break
    
    if plot:
        util.plotOpinions(opinions[0:t,:],'DeGroot')
    
    return opinions[t,:]