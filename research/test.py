# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
from numpy.linalg import norm, inv
import seaborn

from models import *
from util import *
from viz import *

rand.seed(1233)
N = 128
max_rounds = 100
s = rand.rand(N)
A = gnp(N, 0.3, rand_weights=True,stochastic=True)

deGroot(A, s, max_rounds, plot=True)
friedkinJohnsen(A, s, max_rounds, plot=True)
meetFriend(A, s, max_rounds, plot=True)
hk(s, 0.07, max_rounds, eps=1e-8, plot=True)
