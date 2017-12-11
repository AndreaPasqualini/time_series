"""
Author: Andrea Pasqualini
        Bocconi University, Milan
Date created:  29 march 2017
"""
import numpy as np
import tspy as ts
import scipy.linalg as la

x = np.array([1, 2, 3, 4])
x = np.diag(x)
l, v = la.eig(x)

y = ts.lag(x, 1)        # [OK] testing 'lag'
x_lead = ts.lag(x, -1)  # [OK] testing 'lag' with leading result
z = ts.lagcat(x, 3)     # [OK] testing 'lagcat'
