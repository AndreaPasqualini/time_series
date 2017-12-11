"""
Author: Andrea Pasqualini
        Bocconi University, Milan
Date created:  29 march 2017
"""
import numpy as np
import tspy as ts
import matplotlib.pyplot as plt

T = 1000
y = np.zeros((T, 1))
for t in range(1, T):
    y[t] = 0.75 * y[t-1] + np.random.normal()

results = ts.ols(y[1:T, :], y[0:T-1, :], everything=True)

yhat = results['fitted']

fig, ax = plt.subplots()
ax.plot(y[0:T-1], y[1:T],
        'o',
        color='black',
        label='Data')
ax.plot(y[0:T-1], yhat,
        linewidth=2,
        color='red',
        label='Fitted values')
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()
