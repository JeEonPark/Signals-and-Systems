import numpy as np
import matplotlib.pyplot as plt

'''
time_step = 0.01
t = np.arange(0, 1, time_step).reshape(1,100) #100개의 데이터

fs = np.zeros((1,100))
rec = np.concatenate(((np.ones((1,50))), -np.ones((1,50))), axis=1)
count = 1

for f in range(1, 12, 2):
    fs += np.sin(2*np.pi*(f)*t)/count #예를들어 10가지 더했을 경우 sin값이 10까지 올라갈 수 있기 때문에 count로 나눠줌
    plt.subplot(6,1,count)
    count += 1
    plt.plot(fs.T)
    plt.plot(rec.T)
plt.show()
'''

''' 3-1번
time_step = 0.01
t = np.arange(0, 2*np.pi, time_step).reshape(1, int(100*2*np.pi)+1) #100개의 데이터

fs = np.zeros((1,629))
rec = np.concatenate(((np.ones((1,int(100*np.pi)))), -np.ones((1,int(100*np.pi)))), axis=1)
count = 1

for f in range(1, 12, 2):
    fs += np.sin(2*np.pi*(f)*(1/(2*np.pi))*t)/count #예를들어 10가지 더했을 경우 sin값이 10까지 올라갈 수 있기 때문에 count로 나눠줌
    plt.subplot(6,1,count)
    count += 1
    plt.plot(fs.T)
    plt.plot(rec.T)
plt.show()
'''

''' 3-2번
time_step = 0.01
t = np.arange(0, 2*np.pi, time_step).reshape(1, int(100*2*np.pi)+1) #100개의 데이터

fs = np.zeros((1,629))
rec = 4*np.concatenate(np.ones((1,int(100*2*np.pi))))-(4*np.concatenate(((np.ones((1,int(100*1.5*(np.pi))))), 0*np.ones((1,int(100*0.5*np.pi)))), axis=1))+(4*np.concatenate(((np.ones((1,int(100*0.5*(np.pi))))), 0*np.ones((1,int(100*1.5*np.pi)))), axis=1))
count = 1

for f in range(1, 600, 100):
    fs += np.cos(2*np.pi*(f)*(1/(2*np.pi))*t)/(count) #예를들어 10가지 더했을 경우 sin값이 10까지 올라갈 수 있기 때문에 count로 나눠줌
    plt.subplot(6,1,count)
    count += 1
    plt.plot(2+fs.T)
    plt.plot(rec.T)
plt.show()
'''

time_step = 0.01
t = np.arange(0, 4*np.pi, time_step) #100개의 데이터

rec = np.sin(2*np.pi*(t-6))/(np.pi*(t-6))

plt.plot(rec)
plt.show()