# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plotOpinions(x,title=''):
    maxRounds = np.shape(x)[0]
    opinionNumber = np.shape(x)[1]
    for i in range(0,opinionNumber):
        plt.plot(range(0,maxRounds),x[:,i])
    plt.ylabel('Opinion')
    plt.xlabel('t')
    plt.title(title)
    plt.show()

