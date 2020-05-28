#https://makersportal.com/blog/2018/9/13/audio-processing-in-python-part-i-sampling-and-the-fast-fourier-transform
from scipy.io import wavfile
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
plt.style.use('ggplot')

seed_audio_name = 'bruh2.wav'
rate, data = sio.wavfile.read(seed_audio_name)
data_ch1 = data[:, 0]
print(data_ch1.min())
print(data_ch1.max())
# sampling information
#Fs = rate # sample rate
#T = 1/Fs # sampling period
#t = 1 # seconds of sampling
#N = Fs*t # total points in signal

Fs = rate
T = 1/Fs
N = len(data_ch1)
t = N/Fs

# fourier transform and frequency domain
ft = np.fft.fft(data_ch1)
index = np.arange(0,int(N/2),500)
Y_k = ft[index] # FFT function from numpy
Y_k[1:] = Y_k[1:] # need to take the single-sided spectrum only
Pxx = np.abs(Y_k) # be sure to get rid of imaginary part

f = np.fft.fftfreq(N, T) # frequency vector
f = f[index]

# plotting
#fig,ax = plt.subplots()
#plt.plot(f,Pxx,linewidth=5)
#ax.set_xscale('log')
#ax.set_yscale('log')
#plt.ylabel('Amplitude')
#plt.xlabel('Frequency [Hz]')
#plt.show()

new_data_ch1 = np.zeros(shape=data_ch1.shape,dtype=data_ch1.dtype)

t =  np.linspace(0.,1., N)
amplitude = 1
#print(t)
#print(t.shape)
#print(f)
#print(f.shape)
#print(np.multiply.outer(t,f).shape)
#print(np.multiply.outer(t,f))
#print((amplitude * np.sin(2. * np.pi * np.multiply.outer(t,f))).shape)
#print(Pxx.shape)
sin_sig = np.dot(amplitude * (np.sin(2. * np.pi * np.multiply.outer(t,f))), Pxx.T)

print(sin_sig.shape)
print(sin_sig.max())
print(sin_sig.min())
sio.wavfile.write("FFT_test.wav", 44100, sin_sig)

#print(list(sin_sig))
print("stop")

## fourier transform and frequency domain
#ft = np.fft.fft(data_ch1)
#Y_k = ft[0:int(N/2)]/N # FFT function from numpy
#Y_k[1:] = 2*Y_k[1:] # need to take the single-sided spectrum only
#Pxx = np.abs(Y_k) # be sure to get rid of imaginary part

#f = np.fft.fftfreq(N, T) # frequency vector
#f = f[0:int(N/2)]

## plotting
#fig,ax = plt.subplots()
#plt.plot(f,Pxx,linewidth=5)
#ax.set_xscale('log')
#ax.set_yscale('log')
#plt.ylabel('Amplitude')
#plt.xlabel('Frequency [Hz]')
#plt.show()
