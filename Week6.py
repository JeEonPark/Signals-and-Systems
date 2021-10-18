from matplotlib import pyplot as plt
import numpy as np

step = 0.001 #0.004, 0.008... #변경점1
t = np.arange(step, 1+step, step)
c_f=100

#1. 변조하고 싶은 신호
sig = np.sin(2*np.pi*20*t)-2*np.sin(2*np.pi*40*t)+1.5*np.sin(2*np.pi*60*t)
plt.subplot(811) #7행 1열 짜리 1번째꺼
plt.plot(t,sig)

#2. 1번의 그래프를 주파수 영역으로 푸리에 변환
freq_sig=np.abs(np.fft.fftshift(np.fft.fft(sig)))
plt.subplot(812)
plt.stem(freq_sig+1000)

asdf = np.abs(np.fft.ifftshift(np.fft.ifft(freq_sig+1000)))
plt.subplot(813)
plt.stem(asdf)

# carrier=np.cos(2*np.pi*c_f*t)
# am_sig=sig*carrier
# plt.subplot(813)
# plt.plot(t, am_sig)

# f_carrier = np.abs(np.fft.fftshift(np.fft.fft(carrier)))
# plt.subplot(814)
# plt.stem(f_carrier)
#
# freq_am_sig = np.abs(np.fft.fftshift(np.fft.fft(am_sig)))
# plt.subplot(815)
# plt.stem(freq_am_sig)
#
# rcv_sig = am_sig*carrier
# plt.subplot(816)
# plt.plot(t, rcv_sig)

# freq_am_sig = np.abs(np.fft.fftshift(np.fft.fft(rcv_sig)))
# plt.subplot(817)
# plt.stem(freq_am_sig)
#
# LPF_f=100 #이게 뭐야
# t_lpf=np.arange(-0.5, 0.5, step)
# LPF=[]
# for t2 in t_lpf:
#     if t2==0:
#         tmp_LPF=1
#     else:
#         tmp_LPF=np.sin(2 * np.pi * LPF_f * t2) / (2 * np.pi * LPF_f * t2)
#     LPF.append(tmp_LPF)
# LPF_sig=np.convolve(rcv_sig, LPF, 'same')
# filtered_sig=(LPF_sig/np.sum(LPF))*2
# plt.subplot(818)
# plt.plot(t, filtered_sig)

plt.show()

plt.figure(2)
plt.subplot(211)
plt.plot(t, LPF)
f_LPF = np.abs(np.fft.fftshift(np.fft.fft(LPF)))
plt.subplot(212)
plt.plot(f_LPF)

plt.show()

'''
#신호및 시스템 과제
#1. 시간 step을 0.001로 바꾸시오.
#2. sig의 주파수를 각각 20,40,60으로 바꾸시오(기존엔 2,3,4 였음)
#3. 최종 LPF한 신호가 원래 신호와 동일하게 복원되도록 프로그램을 수정하고 이유를 쓰시오.
#제출방법: Pyplot으로 그린 그림1(수업과 같은 8개 그림)과 2그림2(LPF의 시간과 주파수 모양)을 캡쳐해서 붙이고
#신호가 복원되도록 하기 위해서 한 이론적 배경과 프로그램 수정사항에 대해서 설명하여 보고서를 1페이지로 작성하고
#작성된 문서를 제출하시오.

from matplotlib import pyplot as plt
import numpy as np

step = 0.001 #0.004, 0.008 ....
t = np.arange(step, 1 + step, step)
c_f = 50

sig = np.sin(2 * np.pi * 20 * t) - 2 * np.sin(2 * np.pi * 40 * t) + 1.5 * np.sin(2 * np.pi * 60 * t)
plt.subplot(811)
plt.plot(t, sig)

freq_sig = np.abs(np.fft.fftshift(np.fft.fft(sig)))
plt.subplot(812)
plt.stem(freq_sig)

carrier = np.cos(2 * np.pi * c_f * t)
am_sig = sig * carrier
plt.subplot(813)
plt.plot(t, am_sig)

f_carrier = np.abs(np.fft.fftshift(np.fft.fft(carrier)))
plt.subplot(814)
plt.stem(f_carrier)

freq_sig = np.abs(np.fft.fftshift(np.fft.fft(am_sig)))
plt.subplot(815)
plt.stem(freq_sig)

rcv_sig = am_sig * carrier
plt.subplot(816)
plt.plot(t, rcv_sig)

freq_am_sig = np.abs(np.fft.fftshift(np.fft.fft(rcv_sig)))
plt.subplot(817)
plt.stem(freq_am_sig)

LPF_f = 150
t_lpf = np.arange(-0.5, 0.5, step)
LPF = []
for t2 in t_lpf:
    if t2 == 0:
        tmp_LPF = 1
    else:
        tmp_LPF = np.sin(2 * np.pi * LPF_f * t2) / (2 * np.pi * LPF_f * t2)
    LPF.append(tmp_LPF)
LPF_sig = np.convolve(rcv_sig, LPF, 'same')
filtered_sig = LPF_sig/np.sum(LPF) * 2
plt.subplot(818)
plt.plot(t, filtered_sig)
plt.show()

plt.figure(2)
plt.subplot(211)
plt.plot(t, LPF)
f_LPF = np.abs(np.fft.fftshift(np.fft.fft(LPF)))
plt.subplot(212)
plt.plot(f_LPF)

plt.show()
'''