import numpy as np
import matplotlib.pyplot as plt

dim = 1080
process = np.array([9, 16, 25, 36, 64, 100, 144])
time_single = 3.625
time_non_block = np.array([0.442, 0.2575, 0.1933, 0.1602, 0.1392, 0.1096, 0.094])
time_block = np.array([0.421, 0.247, 0.1844, 0.150, 0.132, 0.103, 0.0850])

plt.figure(figsize=[7,10])
plt.plot(process,time_single/time_non_block, label="Fox algorithm")
plt.scatter(process,time_single/time_non_block)
plt.plot(process, time_single/time_block, label="Fox algorithm with block multiply")
plt.scatter(process, time_single/time_block)
plt.plot(process, process, label="Linear speed up")
plt.xlabel('Number of processes')
plt.ylabel('Speed up ratio (T_naive/T_fox)')
plt.legend(loc='upper left')
plt.savefig('Speed_up_block.png')