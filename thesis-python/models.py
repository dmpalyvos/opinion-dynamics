# -*- coding: utf-8 -*-
import numpy as np

import util

def deGroot(A, s, maxRounds, plot=False):
    x = s
    N = len(s)
    plotData = np.zeros((maxRounds,N)) if plot else None
    for i in range(0,maxRounds):
        if plot:
            plotData[i,:] = x
        x = np.dot(A,x)
    if plot:
        util.plotOpinions(plotData,'DeGroot')
