"""
Author: Andrea Pasqualini
        Bocconi University, Milan
Date created:  29 march 2017
"""
import numpy as np
import myfuns as my
import scipy.linalg as la

x = np.array([1, 2, 3, 4])
x = np.diag(x)
l, v = la.eig(x)

y = my.lag(x, 1)        # [OK] testing 'lag'
x_lead = my.lag(x, -1)  # [OK] testing 'lag' with leading result
z = my.lagcat(x, 3)     # [OK] testing 'lagcat'
