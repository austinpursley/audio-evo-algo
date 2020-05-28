import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import random

def equal_tempered_scale():
    f0 = 440
    a = 1.059463
    n = np.arange(-53,68,1)
    fn = f0*np.power(a, n)
    return fn

def make_sin_waves(freq_arr, amplitude, N):
    t =  np.linspace(0.,1., N)
    t_freq = np.multiply.outer(t,freq_arr)
   
    return amplitude * np.sin(2. * np.pi * np.multiply.outer(freq_arr,t)) 

def norm_0to1(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

fs = equal_tempered_scale()

dt = np.int16
#amp = np.iinfo(dt).max/5
amp = 1.0
sample = 44100
t = 5 # 1 sec
N = sample*t
sin_wavs = make_sin_waves(fs, amp, N)
test1 = sin_wavs
print(type(test1))
print(test1.shape)
print(len(test1))

sample_sin = sin_wavs[np.random.choice(sin_wavs.shape[0], 10, replace=False)]

rfft = np.fft.rfft(sample_sin)
print(rfft.shape)
weights_r = np.array(np.random.uniform(low=0.0, high=1.0, size = rfft.shape[0]))
weights_c = np.array(np.random.uniform(low=0.0, high=1.0, size = rfft.shape[0]))
rfft_mix_real = np.matmul(rfft.real.T, weights_r)
rfft_mix_imag = np.matmul(rfft.imag.T, weights_c)
rfft_mix = rfft_mix_real + 1j*rfft_mix_imag
mix = np.fft.irfft(rfft_mix)
mix = (np.iinfo(dt).max/5) * norm_0to1(mix)
sio.wavfile.write("test.wav", sample, np.array(mix,np.int16))

# plotting 
x = np.arange(0,44100,100)
y = mix[0:44100:100]
fig,ax = plt.subplots()
plt.plot(x, y,linewidth=5)
plt.show()
