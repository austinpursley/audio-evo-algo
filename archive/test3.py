#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import numpy as np

from deap import algorithms 
from deap import base
from deap import creator
from deap import tools

import scipy
import scipy.io as sio
from scipy.io import wavfile
from scipy import spatial
import evolvetools as et

from playsound import playsound

import difflib

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def mutate_arr(arr):
    #for d in data:
    #    for i in d:
    #        print(i)
    mutant = np.zeros(shape=arr.shape,dtype=arr.dtype)
    for i, s in enumerate(arr):
        mutant[i] = et.mutation(s,0.5)
    return mutant

def initIndividual(icls, data):
    mutant = mutate_arr(data)
    return icls(mutant)

def initPopulation(pcls, ind_init, data, n):
    return pcls(ind_init(data) for i in range(0,n))

def evalSound(individual):
    rate = 44100
    sio.wavfile.write("indiv.wav", rate, individual)
    playsound("indiv.wav")
    print("How did that sound?")
    score = input("Enter integer between 0-100)")
    if RepresentsInt(score) == True:
        score = int(score)
    else:
        print("score not number, setting to 0 lmao")
        score = 0
    return score,

def evalData2Data(individual, target):
    result = 1 - spatial.distance.cosine(individual, target) #similarity
    if not 0 <= result <= 1:
        result = 0
    return result,
    
#    return np.sum(individual == target),


#    return np.linalg.norm((individual - target), ord=1),
#    sm = difflib.SequenceMatcher(None, individual, target)
#    print(sm.ratio)
#    return(sm.ratio),

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

seed_audio_name = 'bruh2.wav'
rate, data = sio.wavfile.read(seed_audio_name)

target_audio_name = 'minecraft_oof.wav'
rate_t, data_t = sio.wavfile.read(target_audio_name)

smaller = min(len(data), len(data_t)) - 1
data = data[:smaller, :]
data_t = data_t[:smaller, :]

data_ch1 = data[:, 0]
target_ch1 = data_t[:, 0]

print(len(data_ch1))

#sio.wavfile.write("test1.wav", 44100, data_ch1)
#sio.wavfile.write("test2.wav", 44100, target_ch1)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("individual", initIndividual, creator.Individual)
toolbox.register("population", initPopulation, list, toolbox.individual, data_ch1)
    
toolbox.register("evaluate", evalData2Data, target=target_ch1)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    
    pop = toolbox.population(n=1000)
    
    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaMuPlusLambda(pop, toolbox, 10, 1000, cxpb=0.5, mutpb=0.5, ngen=50, stats=stats,
                        halloffame=hof)
    
    sio.wavfile.write("hof.wav", 44100, hof[0])

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()

