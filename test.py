import scipy
import scipy.io as sio
from scipy.io import wavfile
import numpy as np

rate, data = sio.wavfile.read('bruh2.wav')
rate2, data2 = sio.wavfile.read('cartoon_run.wav')
print("rate", rate)
print("rate2", rate2)
#data = np.sin(data)
#data2 = np.sin(data2)

print(len(data))
print(len(data2))

smaller = min(len(data), len(data2)) - 1

data = data[:smaller, :]
data2 = data2[:smaller, :]


#avg = np.average(data, axis=1)
#ch1 = data[:, 0]
#ch2 = data[:, 1]



ch1_a = data[:, 0]
ch1_b = data2[:, 0]


print(ch1_a)
print(ch1_b)

avg = (ch1_a + ch1_b)/2.0
print(avg)
sio.wavfile.write("example.wav", rate, avg)

#sio.wavfile.write("example_a.wav", rate, ch1_a)
#sio.wavfile.write("example_b.wav", rate, ch1_b)


#comb_mat = np.concatenate(ch1_a, ch1_b)

#comb_mat[:, 0] = data[:, 0]
#comb_mat[:, 1] = data2[:, 0]

#avg = np.average(comb_mat, axis=1)
