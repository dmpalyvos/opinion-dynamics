# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
from numpy.linalg import norm, inv
import seaborn

from models import *
from util import *
from viz import *


N = 30
s = rand.rand(N)
A = gnp(N, 0, rand_weights=True, stochastic=True)
meetFriend_nomem(A, s, 1e3, conv_stop=True)
