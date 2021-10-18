import numpy as np
import matplotlib.pyplot as plt

t=np.arange(0, 10, 0.1)
x_t=np.exp(-2*t)+1
h_t=np.concatenate(np.ones((1, 10)))*0.1

plt.figure(1)
plt.stem(x_t)
plt.figure(2)
plt.stem(h_t)
plt.show()