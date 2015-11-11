# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
import seaborn as sns

import models
from util import *


N = 2000
rand.seed(123)
s = rand.random(N)
A = gnp(N,0.2,True)
#models.deGroot(A, s, 10000, plot = False)


