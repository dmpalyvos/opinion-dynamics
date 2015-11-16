# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
from numpy.linalg import norm, inv
import seaborn

from models import *
from util import *
from viz import plotDistance

N = 150
A = gnp(N, 0.05, rand_weights=True, stochastic=True, verbose=True)
A = A + np.diag(np.diag(A))
A = rowStochastic(A)
A = A + np.diag(np.diag(A))
A = rowStochastic(A)
A = A + np.diag(np.diag(A))
A = rowStochastic(A)
s = rand.rand(N)

opinions = friedkinJohnsen(A, s, 100)
fj_eq = expectedEquilibrium(A, s)
plotOpinions(opinions,dcolor=True)

plotDistance(A, s, opinions)