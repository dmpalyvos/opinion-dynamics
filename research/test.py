# -*- coding: utf-8 -*-
import numpy as np
from numpy import diag
import numpy.random as rand
from numpy.linalg import norm, inv
import seaborn

import models
from util import gnp
from viz import plotNetwork


rand.seed(1233)
N = 128
max_rounds = 1000
s = rand.rand(N)
A = gnp(N, 0.4, rand_weights=True)
B = diag(rand.rand(N)) * 0.5
#models.deGroot(A, s, max_rounds, plot=True)
#op = models.friedkinJohnsen(A, s, max_rounds, plot=True)
#models.meetFriend(A, s, 1e5, conv_stop=False)
#models.hk(s, 0.07, max_rounds, eps=1e-8, plot=True)
#models.hk_local(A, s, 0.07, max_rounds, eps=1e-8, plot=True, save=True)
#models.ga(A, B, s, max_rounds, plot=True, save=True)
models.kNN(A, s, 10, max_rounds, conv_stop=True)
models.kNN_nomem(A, s, 10, max_rounds, conv_stop=True)
#plotNetwork(A, op[-1,:])