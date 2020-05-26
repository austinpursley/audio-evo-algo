from deap import algorithms 
from deap import base
from deap import creator
from deap import tools

import random
import scipy.io as sio
from scipy.io import wavfile
from scipy import spatial
from scipy import signal
import numpy as np
import glob

import matplotlib.pyplot as plt

def equalTemperedScale():
    f0 = 440
    a = 1.059463
    n = np.arange(-53,68,1)
    fn = f0*np.power(a, n)
    return fn

def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2

def mutComplex(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            real = individual[i].real
            imag = individual[i].imag
#            a = 1.5*real.max() 
#            b = 1.5*imag.max()
            a = 1.5
            b = 1.5
            rand_real = random.uniform(-1*a,a)    
            rand_imag = random.uniform(-1*b,b)
            individual[i] = np.complex128(rand_real*real + rand_imag*imag*1j)
    return individual,

def selBestAndSeed(individuals, k, fit_attr="fitness", seed=np.array):
    individuals += seed
    return tools.selBest(individuals, k, fit_attr)

# define sounds for "building blocks"
#fs = equalTemperedScale()
#samplerate = 44100
#N = 44100
#amplitude = np.iinfo(np.int16).max/5
#print(amplitude)
#t =  np.linspace(0.,1., N)
#data = np.array(amplitude * np.sin(2. * np.pi * np.multiply.outer(t,fs)), np.int16)
#print(data.shape)

#d1 = data[:,50]
#d2 = data[:,51]
#d1_ft = np.fft.rfft(d1)
#d2_ft = np.fft.rfft(d2)
#print(d1_ft.shape)
#print(d2_ft.shape)
##test = np.array(cxTwoPointCopy(d1_ft, d2_ft)).reshape(-1)
#test = cxTwoPointCopy(d1_ft, d2_ft)[0]
#print(test)
## plotting 
#x = np.arange(0,44100,100)
#y = test[0:44100:100]
#fig,ax = plt.subplots()
#plt.plot(x, y,linewidth=5)
#plt.show()

#for i, d in enumerate(data.T):
#    i = "%.2d" % i
#    print(i)
#    print(d.dtype)
#    sio.wavfile.write("blocks/block" + str(i) + "_sin.wav", samplerate, d)

#data = np.array(amplitude * np.cos(2. * np.pi * np.multiply.outer(t,fs)), np.int16)
#print(data.shape)
#for i, d in enumerate(data.T):
#    i = "%.2d" % i
#    print(i)
#    print(d.dtype)
#    sio.wavfile.write("blocks/block" + str(i) + "_cos.wav", samplerate, d)


#data = np.array(amplitude*signal.sawtooth(2 * np.pi * np.multiply.outer(t,fs)), np.int16)
#print(data.shape)
#for i, d in enumerate(data.T):
#    i = "%.2d" % i
#    print(i)
#    print(d.dtype)
#    sio.wavfile.write("blocks/block" + str(i) + "_saw.wav", samplerate, d)


def evalData2Data(indiv_ft, target_ft):
    t_len = len(target_ft)
    i_len = len(indiv_ft)
    if t_len > i_len:
        length = t_len - i_len
        indiv_ft = np.pad(indiv_ft, (0, length), mode='constant')
    elif t_len < i_len:
        length = i_len - t_len
        target_ft = np.pad(target_ft, (0, length), mode='constant')
    real = indiv_ft.real.flatten()
    imag = indiv_ft.imag.flatten()
    indiv_flat = np.concatenate((real, imag))
    real = target_ft.real.flatten()
    imag = target_ft.imag.flatten()
    target_flat = np.concatenate((real, imag))
    result = spatial.distance.cosine(indiv_flat, target_flat) #distance
    return result,

def initIndividual(icls, audio_filename):
    rate, data = sio.wavfile.read(audio_filename)
#    x = np.arange(0,44100,100)
#    y = data[0:44100:100]
#    fig,ax = plt.subplots()
#    plt.plot(x, y,linewidth=5)
#    plt.show()
    if data.ndim == 2:
        data = data[:, 0]
    t_len = len(data)
    i_len = 44100
    if t_len > i_len:
        data = data[:i_len]
    elif t_len < i_len:
        length = i_len - t_len
        data = np.pad(data, (0, length), mode='constant')

    indiv_ft = np.fft.rfft(data)
    return icls(indiv_ft)

def initPopulation(pcls, ind_init, audio_fn_list):
    return pcls(ind_init(fn) for fn in audio_fn_list)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("individual", initIndividual, creator.Individual)
audio_fn_list = sorted(glob.glob("blocks/*.wav"))
#print(audio_fn_list)
toolbox.register("population", initPopulation, list, toolbox.individual, audio_fn_list)

target_audio_name = 'minecraft_oof.wav'
rate_t, data_t = sio.wavfile.read(target_audio_name)
target_ch1 = data_t[:, 0]
t_len = len(target_ch1)
i_len = 44100
if t_len > i_len:
    target_ch1 = target_ch1[:i_len]
elif t_len < i_len:
    length = i_len - t_len
    target_ch1 = np.pad(target_ch1, (0, length), mode='constant')
target_ft = np.fft.rfft(target_ch1)

## plotting 
#x = np.arange(0,44100,100)
#y = target_ch1[0:44100:100]
#fig,ax = plt.subplots()
#plt.plot(x, y,linewidth=5)
#plt.show()

#test = np.fft.ifft(target_ft)

## plotting
#fig,ax = plt.subplots()
#plt.plot(test,linewidth=5)
#plt.show()

toolbox.register("evaluate", evalData2Data, target_ft=target_ft)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", mutComplex, indpb=0.05)
toolbox.register("select", selBestAndSeed, seed=toolbox.population())

def main():
    random.seed(64)
    pop = toolbox.population()
#    samplerate = 44100
#    for i, d in enumerate(pop):
#        print(d)
#        	i = "%.2d" % i
#        sio.wavfile.write("test/block" + str(i) + ".wav", samplerate, d)

    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
#    stats.register("std", np.std)
#    stats.register("min", np.min)
#    stats.register("max", np.max)
    algorithms.eaMuPlusLambda(pop, toolbox, 10, 50, cxpb=0.9, mutpb=0.1, ngen=300, stats=stats,
                        halloffame=hof)
#    data = np.fft.ifft(hof)
#    normalized = np.array((rd-min(rd))/(max(rd)-min(rd))*25000, np.int16).reshape(-1)
#    print(normalized)
    hof_inv = np.fft.irfft(hof)
    hof_inv = np.array(hof_inv,np.float).reshape(-1)
    sio.wavfile.write("inverse_FFT_test.wav", 44100, np.array(hof_inv,np.int16))    
#    hof_arr = np.array(hof,np.float).reshape(-1)
    print(hof_inv.shape)
    print(hof_inv.dtype)
    print(hof_inv.min())
    print(hof_inv.max())

#    # plotting
#    fig,ax = plt.subplots()
#    plt.plot(hof_inv,linewidth=5)
#    plt.show()    
    
    normalized = 2*(hof_inv-min(hof_inv)) / (max(hof_inv)-min(hof_inv))-1
    normalized = 32000*normalized
    sio.wavfile.write("inverse_FFT_test_norm.wav", 44100, np.array(normalized,np.int16))
    print(normalized.shape)
    print(normalized.dtype)
    print(normalized.min())
    print(normalized.max())
    # plotting 
    x = np.arange(0,44100,100)
    y = normalized[0:44100:100]
    fig,ax = plt.subplots()
    plt.plot(x, y,linewidth=5)
    plt.show()

#    rate, test =sio.wavfile.read("inverse_FFT_test_norm.wav")
#    
#    # plotting 
#    x = np.arange(0,44100,100)
#    y = test[0:44100:100]
#    fig,ax = plt.subplots()
#    plt.plot(x, y,linewidth=5)
    plt.show()

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()
