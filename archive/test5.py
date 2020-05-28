from deap import algorithms 
from deap import base
from deap import creator
from deap import tools

import random
import scipy.io as sio
from scipy.io import wavfile
from scipy import spatial
import numpy as np
import glob

import matplotlib.pyplot as plt

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

## define sounds for "building blocks"
#fs = np.array([440.00, 466.16, 493.88, 523.25, 554.37, 
#      587.33, 622.25, 659.25, 698.46, 739.99, 
#      783.99, 830.61])
#samplerate = 44100
#N = 44100
#amplitude = np.iinfo(np.int16).max/5
#print(amplitude)
#t =  np.linspace(0.,1., N)
#data = np.array(amplitude * np.sin(2. * np.pi * np.multiply.outer(t,fs)), np.int16)
#print(data.shape)
#for i, d in enumerate(data.T):
#    i = "%.2d" % i
#    print(i)
#    print(d.dtype)
#    sio.wavfile.write("blocks/block" + str(i) + ".wav", samplerate, d)


def evalData2Data(individual, target):
    t_len = len(target)
    i_len = len(individual)
    if t_len > i_len:
        length = t_len - i_len
        individual = np.pad(individual, (0, length), mode='constant')
    elif t_len < i_len:
        length = i_len - t_len
        target = np.pad(target, (0, length), mode='constant')
    real = individual.real.flatten()
    imag = individual.imag.flatten()
    individual = np.concatenate((real, imag))
    real = target.real.flatten()
    imag = target.imag.flatten()
    target = np.concatenate((real, imag))
    result = spatial.distance.cosine(individual, target) #distance
    return result,

def initIndividual(icls, audio_filename):
    rate, data = sio.wavfile.read(audio_filename)
    ft = np.fft.fft(data)
    return icls(ft)

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
target_ft = np.fft.fft(target_ch1)

## plotting
#fig,ax = plt.subplots()
#plt.plot(target_ch1,linewidth=5)
#plt.show()

toolbox.register("evaluate", evalData2Data, target=target_ft)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

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
    algorithms.eaMuPlusLambda(pop, toolbox, 500, 1000, cxpb=0.9, mutpb=0.1, ngen=1000, stats=stats,
                        halloffame=hof)
    data = np.fft.ifft(hof)
    rd = np.array(data.real, np.int16).reshape(-1)
    normalized = (rd-min(rd))/(max(rd)-min(rd))*25000
    print(normalized)
    sio.wavfile.write("inverse_FFT_test.wav", 44100, normalized)

     # plotting
    fig,ax = plt.subplots()
    plt.plot(rd,linewidth=5)
    plt.show()

    # plotting
    fig,ax = plt.subplots()
    plt.plot(normalized,linewidth=5)
    plt.show()
    
    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()
