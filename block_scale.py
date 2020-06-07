import numpy as np
import matplotlib.pyplot as plt

dim = np.array([50, 100, 200, 500, 1000, 2000])
time_naive = np.array([0.000381, 0.00298, 0.0234, 0.359, 2.854, 22.9])
time_block = np.array([0.000338, 0.00275, 0.0217, 0.337, 2.698, 21.75])

plt.figure(figsize=[7,7])
plt.plot(dim,time_naive, label="Naive multiplication")
plt.scatter(dim,time_naive)
plt.plot(dim, time_block, label="Block multiplication")
plt.scatter(dim, time_block)
plt.xlabel('Matrices row and column size')
plt.ylabel('Execution time (s)')
plt.legend(loc='upper left')
plt.savefig('Block_scaling.png')