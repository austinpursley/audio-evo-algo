import scipy
import scipy.io as sio
from scipy.io import wavfile
import numpy as np
import evolvetools as et

rate, data = sio.wavfile.read('bruh2.wav')
ch1 = data[:, 0]
print(rate)
#for d in data:
#    for i in d:
#        print(i)
mutate = np.zeros(shape=ch1.shape,dtype=np.int16)
for i, s in enumerate(ch1):
    mutate[i] = et.mutation(ch1[i], 0.005)
#    if ch1[i] != mutate[i]:
#        print("ch1")
#        print(ch1[i])
#        print("mutate")    
#        print(mutate[i])

print(np.array_equal(ch1, mutate))
print(ch1[0].dtype)
print(mutate[0].dtype)

print(ch1.shape)
print(mutate.shape)

#for i, c in enumerate((ch1 == mutate)):
#    if c == False:
#        print("*************")
#        print("FALSE")
#        print(c)
#        print(i)
#        print("VALUES")
#        print(ch1[i])
#        print(mutate[i])
#        print("*************")
sio.wavfile.write("bruh2_ch1.wav", rate, ch1)
sio.wavfile.write("mutate.wav", rate, mutate)

