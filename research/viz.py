# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from util import *

def plotOpinions(opinions, title='', dcolor=False):
    '''Creates a plot of the opinions over time
    
    Args:
    
        opinions (txN vector): Vector of the opinions over time
        
        title (string): Optional title of the plot (default: '')
        
        dcolor (bool): Color the plot lines depending on the value of 
        each opinion (default: False)
    
    '''
    
    max_rounds = np.shape(opinions)[0]
    opinion_number = np.shape(opinions)[1]
    for t in range(0, opinion_number):
        x = range(0, max_rounds)
        y = opinions[:,t]
        if dcolor:
            colorline(range(max_rounds), opinions[:,t], z=opinions[:,t])
            pass
        else:
            plt.plot(range(max_rounds), opinions[:,t])
    plt.ylabel('Opinion')
    plt.xlabel('t')
    plt.title(title)
    plt.axis((0, max_rounds, opinions.min() - 0.1, opinions.max() + 0.1))
    plt.show()
    
        
def plotDistance(A, s, opinions):
    '''Plot the distance of the opinions from the expected equilibrium
    
    Creates a plot of the distance from the expected equilibrium of the
    Friedkin-Johnsen model over time.
    
    Args:
    
        A (NxN numpy array): Adjacency Matrix
        
        s (1xN numpy array): Intrinsic beliefs vector
        
        opinions (txN vector): Vector of the opinions over time
    
    '''

    eq = expectedEquilibrium(A, s)
    dist = norm(opinions - eq, axis=1)
    plt.plot(range(dist.size),dist)
    plt.xlim(0,dist.size)
    plt.ylim(np.min(dist),np.max(dist))
    

'''
Begin colorline 
Based on the work of David P. Sanders
Source: https://github.com/dpsanders/matplotlib-examples
'''
     

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments



def colorline(x, y, z=None, cmap=plt.get_cmap('cool'), norm=plt.Normalize(0.0, 1.0), linewidth=1, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc
        
    
def clear_frame(ax=None): 
    # Taken from a post by Tony S Yu
    if ax is None: 
        ax = plt.gca() 
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    for spine in ax.spines.itervalues(): 
        spine.set_visible(False) 

'''
End Colorline
'''

