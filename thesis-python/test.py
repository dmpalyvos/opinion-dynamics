# -*- coding: utf-8 -*-
import numpy as np
import models
    
def main():
    A = np.array([[0.2, 0.6, 0.2],
                  [0.7, 0.1, 0.2],
                  [0.4, 0.3, 0.3]])
                  
    x = np.array([0.1, 0.5, 1])

    models.deGroot(A, x, 100, plot = True)


if __name__ == "__main__":
    main()

