import benchmark_functions as bf
import numpy as np

from inspect import signature
from pso import Swarm
from firefly import Firefly
TEST_NUM = 100
"""
Benchmark test for switching between pso and firefly
"""
def line():
  print(u'\u2500' * 50)

results = np.empty([len(bf.BenchmarkFunction.__subclasses__()), 100])

#swarm parameters
pop = 30
#pso parameters
vmax=0.6
beta=0.6
c1=2.8
c2=1.3
#firefly parameters
beta_0=1.5
gamma=0.8
alpha=1

for index, func in enumerate(bf.BenchmarkFunction.__subclasses__()):
    func_args = signature(func).parameters
    type(func_args)
    print(func_args)
    print(func_args.get('n_dimensions'))

    if func_args.get('n_dimensions') != None:
      instance = func(n_dimensions=30)

    else: instance = func()

    print(instance.name()) 
    bounds = instance.suggested_bounds()
    bounds = np.array(bounds)
    bounds = bounds.T
    print(bounds)

    swarm = Swarm(function=instance, population=pop, bounds=bounds, vmax=vmax, beta=beta, c1=c1, c2=c2)
    firefly = Firefly(function=instance, population=pop, bounds=bounds, beta_0=beta_0, light_absorption=gamma, randomisation_param=alpha)

    for i in range(100):
      swarm.initialise_swarm()
      swarm.step(steps=70)
      firefly.position = swarm.position
      firefly.update_attractiveness()
      firefly.step(steps=30)

      results[index, i] = firefly.g_fitness


print(results)
average = np.mean(results, axis=1)
minimum = results.min(axis=1)
maximum = results.max(axis=1)
std = np.std(results, axis=1)

print("average: ", average)
print("min: ", minimum)
print("max: ", maximum)
print("std: ", std)

print("swarm size={}".format(pop))
print("pso: vmax={}, beta={}, c1={}, c2={}".format(vmax, beta, c1, c2))
print("firefly: beta_0={}, gamma={}, alpha={}".format(beta_0, gamma, alpha))

print(" Func | avrg | min  | max  | std  |")
for index, func in enumerate(bf.BenchmarkFunction.__subclasses__()):
    instance=func()
    print("{} & {} & {} & {} & {} ".format(instance.name(), average[index], minimum[index], maximum[index], std[index]))


