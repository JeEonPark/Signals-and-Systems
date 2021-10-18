import numpy as np
import matplotlib.pyplot as plt

'''
x = np.arange(0, 10, 0.001) #10000개가 10초, 1000개가 1초
y = np.sin(2.0*np.pi*10.0*x) + 0.5*np.sin(2.0*np.pi*10.0*x)

plt.figure(1)
plt.plot(y[0:1000])
plt.grid()
plt.show()

yf = np.fft.fft(y)
mid = int(len(yf)/2) #5000
end = int(len(yf)) #10000
yf = np.concatenate(yf[mid:end], yf[0:mid])

plt.figure(2)
plt.stem(np.abs(yf))
plt.grid()
plt.show()
'''


'''
x = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
xf = np.fft.fft(x)
mid = int(len(xf)/2)
end = int(len(xf))
xf = np.concatenate((xf[mid:end], xf[0:mid]))
plt.figure(2)
plt.plot(np.abs(xf))
plt.show()
'''