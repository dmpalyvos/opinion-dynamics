# -*- coding: utf-8 -*-
'''
Test toolkits for parallel processing
'''
from IPython.parallel import Client

import numpy as np
import numpy.random as rand
from util import gnp


def f1(args):
    A, s = args
    from models import meetFriend_nomem
    t, z = meetFriend_nomem(A, s, 1e6, eps=1e-8)
    return t


def main():
    rc = Client()
    dview = rc[:]
    networks = [gnp(40, p) for p in np.arange(0, 0.9, 0.1)]
    opinions = [rand.rand(40) for p in np.arange(0, 0.9, 0.1)]
    asyncs = dview.map_sync(f1, zip(networks, opinions))
    print [res for res in asyncs]

if __name__ == '__main__':
    main()
