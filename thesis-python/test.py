# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rand
import seaborn as sns

import models
from util import *
    
def main():

    N = 15
    s = rand.random(N)
    sns.distplot(s,bins=10,kde=False)
    plt.show()
    A = gnp(N,0.3)

    models.friedkinJohnsen(A, s, 100, plot = True)


if __name__ == "__main__":
    main()

