# -*- coding: utf-8 -*-
# pylint: disable=E1101

'''
Visualization functions
'''

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numpy.linalg import norm
from util import expectedEquilibrium
import networkx as nx


def plotNetwork(A, s, k=0.2, node_size=20):
    '''Plot the network graph. Not final yet.
    '''
    graph = nx.Graph()
    N = A.shape[0]
    graph.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if A[i, j] > 0:
                graph.add_edge(i, j, weight=100*A[i, j])

    pos = nx.spring_layout(graph, k=k, scale=1.0, iterations=500)
    # Draw the nodes and edges
    print(s.min(), s.max())
    nx.draw_networkx_nodes(graph, pos, node_color=s, vmin=0, vmax=1,
                           node_size=node_size, alpha=0.8, cmap=plt.cm.cool)

    nx.draw_networkx_edges(graph, pos, width=0.3, alpha=0.4)
    plt.show()


def plotOpinions(opinions, title='', dcolor=False, interp=True):
    '''Creates a plot of the opinions over time

    Args:
        opinions (txN vector): Vector of the opinions over time

        title (string): Optional title of the plot (default: '')

        dcolor (bool): Color the plot lines depending on the value of
        each opinion (default: False)

        interp (bool): Interpolate the points to get smoother color transitions
        if dcolor is enabled If dcolor is disabled, no action is taken
        (default: True)

    '''

    max_rounds = np.shape(opinions)[0]
    opinion_number = np.shape(opinions)[1]
    for t in range(opinion_number):
        x = range(max_rounds)
        y = opinions[:, t]
        if dcolor:
            # If small number of rounds, increase the points
            # for a smoother plot
            if interp and max_rounds < 100:
                (x, y) = interpolatePoints(x, y, factor=4)
            colorline(x, y, z=y)
        else:
            plt.plot(x, y)
    plt.ylabel('Opinion')
    plt.xlabel('t')
    plt.title(title)
    plt.axis((0, max_rounds, opinions.min() - 0.1, opinions.max() + 0.1))
    plt.show()


def plotDistance(A, s, opinions):
    '''Plots the distance of the opinions from the expected equilibrium

    Creates a plot of the distance from the expected equilibrium of the
    Friedkin-Johnsen model over time.

    Args:
        A (NxN numpy array): Adjacency Matrix

        s (1xN numpy array): Intrinsic beliefs vector

        opinions (txN vector): Vector of the opinions over time

    '''

    eq = expectedEquilibrium(A, s)
    dist = norm(opinions - eq, axis=1)
    plt.plot(range(dist.size), dist)
    plt.xlim(0, dist.size)
    plt.title('Distance from Friedkin-Johnsen Equilibrium')


def colorline(x, y, z=None, cmap=plt.get_cmap('cool'),
              norm=plt.Normalize(0.0, 1.0), linewidth=1, alpha=1.0):
    '''Plot a colored line with coordinates x and y

    Plots a line the color of which changes according to parameter z. For a
    small number of points it it suggested that you interpolate them using
    interpolatePoints() before calling this, in order to have smooth color
    transititons.

    Args:
        x: The x-coordinates of each point

        y: The y-coordinates of each point

        z: The color value of each point. If the norm parameter is not
        specified, minimum color is 0.0 and maximum color is 1.0. If value is
        None, the line will change color as the points progress (default: None)

        cmap: The prefered colormap (default: cool)

        norm: The normalization of the colors. (default: Normalize(0.0,1.0))

        linewidth: The width of the line (default 1)

        alpha: The opacity of the line (default: 1.0)


    Source:
        Based on the work of David P. Sanders
        https://github.com/dpsanders/matplotlib-examples

    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap,
                        norm=norm, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates,
    in the correct format for LineCollection: an array of the form
    numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def clear_frame(ax=None):
    # Taken from a post by Tony S Yu
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.itervalues():
        spine.set_visible(False)


#
def interpolatePoints(x, y, factor=10):
    '''
    Take points listed in two vectors and return them at a higher
    resultion. Create at least factor*len(x) new points that include the
    original points and those spaced in between.

    Returns new x and y arrays as a tuple (x,y).

    Based on this post: http://stackoverflow.com/a/8505774
    '''

    NPOINTS = np.size(x)
    # r is the distance spanned between pairs of points
    r = [0]
    for i in range(1, len(x)):
        dx = x[i]-x[i-1]
        dy = y[i]-y[i-1]
        r.append(np.sqrt(dx*dx+dy*dy))
    r = np.array(r)

    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    dr = rtot[-1]/(NPOINTS*factor-1)
    xmod = [x[0]]
    ymod = [y[0]]
    rPos = 0  # current point on walk along data
    rcount = 1
    while rPos < r.sum():
        x1, x2 = x[rcount-1], x[rcount]
        y1, y2 = y[rcount-1], y[rcount]
        dpos = rPos-rtot[rcount]
        theta = np.arctan2((x2-x1), (y2-y1))
        rx = np.sin(theta)*dpos+x1
        ry = np.cos(theta)*dpos+y1
        xmod.append(rx)
        ymod.append(ry)
        rPos += dr
        while rPos > rtot[rcount+1]:
            rPos = rtot[rcount+1]
            rcount += 1
            if rcount > rtot[-1]:
                break

    return xmod, ymod
