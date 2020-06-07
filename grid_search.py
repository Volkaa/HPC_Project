import numpy as np
import matplotlib.pyplot as plt

dim = 1080
nb_proc = 9
stride = np.array([5,10,15,20,25,30,35,40,45,50,60,70])
time_block = np.array([0.5934, 0.496, 0.4572, 0.434, 0.425, 0.4215, 0.4368, 0.4376, 0.435, 0.433, 0.431, 0.431])

plt.plot(stride,time_block)
plt.scatter(stride,time_block)
plt.xlabel('Stride value (block dimension)')
plt.ylabel('Execution time (s)')
plt.savefig('Grid_search.png')