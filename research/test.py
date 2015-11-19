# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
from numpy.linalg import norm, inv
import seaborn

import models
from util import gnp

rand.seed(1233)
N = 128
max_rounds = 100
s = rand.rand(N)
A = gnp(N, 0.3, rand_weights=True, stochastic=True)

models.deGroot(A, s, max_rounds, plot=True)
models.friedkinJohnsen(A, s, max_rounds, plot=True)
models.meetFriend(A, s, max_rounds, plot=True)
models.hk(s, 0.07, max_rounds, eps=1e-8, plot=True)
models.hk_local(A, s, 0.07, max_rounds, eps=1e-8, plot=True)
