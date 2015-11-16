# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
import seaborn as sns
from numpy.linalg import norm, inv

from models import *
from util import *


N = 8
A = gnp(N, 0.5, rand_weights=True)
s = rand.rand(N)

opinions = friedkinJohnsen(A, s, 1e3,plot = True)
fj_eq = fjEquilibrium(A,s)

dist = norm(opinions - fj_eq, ord = np.inf, axis = 1)
plt.plot(range(len(dist)),dist)

for i in fj_eq:
    plt.axhline(y=i)
print opinions[-1,:] - fj_eq