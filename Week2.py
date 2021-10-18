import numpy as np
import matplotlib.pyplot as plt

time_step=0.01
t=np.arange(0, 2, time_step)
print(t)
x_t1=(np.exp(-t))
u_t=np.concatenate(np.ones((1,200)))*time_step

x_t2=np.exp(-t)


y_t1=np.convolve(x_t1, u_t) + 1


plt.stem(y_t1)
plt.show()