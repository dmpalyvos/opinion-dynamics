# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
from numpy.linalg import norm, inv
import seaborn

from models import *
from util import *
from viz import plotDistance

N = 1000
s = rand.rand(N)
hk(s,0.1,1e3,1e-5,plot=True)