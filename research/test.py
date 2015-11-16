# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
import seaborn as sns
from numpy.linalg import norm, inv

from models import *
from util import *

rand.seed(100)
N = 8
A = gnp(N, 0.5, rand_weights=True, stochastic=True)
s = rand.rand(N)

opinions = friedkinJohnsen(A, s, 1e3,plot=True)
fj_eq = expectedEquilibrium(A, s)

plotDistance(A, s, opinions)