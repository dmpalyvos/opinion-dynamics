# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
import seaborn as sns

import models
from util import *

N = 10000000
weights = np.array([0.2, 0.1, 0.3, 0.15, 0.1])
results = np.empty(N)
for i in range(N):
    results[i] = rchoice(weights)

plt.hist(results,bins=range(len(weights)+1),normed=True)
