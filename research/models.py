# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm, inv
from util import *

def deGroot(A, s, max_rounds, eps=1e-6, plot=False, conv_stop=True):
    """Simulates the DeGroot Model.
    
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
        
    """
    max_rounds = int(max_rounds)
    
    N = len(s)
    max_rounds += 1 # Round 0 contains the initial opinions
    x = s
    opinions = np.zeros((max_rounds, N))
    opinions[0,:] = s
    
    for t in range(0,max_rounds):
        x = np.dot(A, x)
        opinions[t,:] = x
        if conv_stop and norm(opinions[t - 1,:] - opinions[t,:], np.inf) < eps:
            print('DeGroot converged after {t} rounds'.format(t=t))
            break
    
    if plot:
        plotOpinions(opinions[0:t,:], 'DeGroot')
    
    return opinions
    

    
def friedkinJohnsen(A, s, max_rounds, eps=1e-6, plot=False, conv_stop=True):
    """Simulates the Friedkin-Johnsen (Kleinberg) Model.
    
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
        
    """
    max_rounds = int(max_rounds)

    # Preprocess A and extract stubborness matrix B
    B = np.diag(np.diag(A)) # Stubborness matrix of the model
    A_model = A - B # Adjacency matrix of the model
    
    N = len(s)
    max_rounds += 1 # Round 0 contains the initial opinions
    x = s
    opinions = np.zeros((max_rounds, N))
    opinions[0,:] = s
    
    for t in range(0, max_rounds):
        x = np.dot(A_model, x) + np.dot(B, s)
        opinions[t,:] = x
        if conv_stop and norm(opinions[t - 1,:] - opinions[t,:], np.inf) < eps:
            print('Friedkin-Johnsen converged after {t} rounds'.format(t=t))
            break
    
    if plot:
        plotOpinions(opinions[0:t,:], 'Friedkin-Johnsen')
    
    return opinions[0:t,:]
