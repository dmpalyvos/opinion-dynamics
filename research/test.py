# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
import seaborn as sns

import models
from util import *

#rand.seed(223)
N = 20
A = gnp(N,0.7,rand_weights=True)
A = A - 0.*np.diag(np.diag(A))
s = rand.rand(N)
models.friedkinJohnsen(A,s,300,plot=True)