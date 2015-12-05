# -*- coding: utf-8 -*-
import numpy as np
from numpy import diag
import numpy.random as rand
from numpy.linalg import norm, inv
import seaborn

import models
from viz import plotNetwork
from util import gnp, barabasi_albert, from_edgelist
import matplotlib.pyplot as plt

rand.seed(1233)
A, N = from_edgelist('./networks/facebook_combined.txt')
s = rand.rand(N)
max_rounds = 101
#A = gnp(N, 0.12, rand_weights=True)
B = diag(rand.rand(N)) * 0.5
#models.deGroot(A, s, max_rounds, plot=True)
#op = models.friedkinJohnsen(A, s, max_rounds, plot=True)
#models.meetFriend(A, s, 1e5, conv_stop=False)
#models.hk(s, 0.07, max_rounds, eps=1e-8, plot=True)
#models.hk_local(A, s, 0.07, max_rounds, eps=1e-8, plot=True)
#models.ga(A, B, s, max_rounds, plot=True, save=True)
#models.kNN_static(A, s, 5, max_rounds, conv_stop=True, plot=True)
#t, z, Q = models.kNN_dynamic_nomem(A, s, 4, max_rounds)
t, z = models.kNN_static_nomem(A, s, 4, max_rounds)
#plotNetwork(Q, z, node_size=10, iterations=50)
#plotNetwork(A, op[-1,:])
plt.hist(z)