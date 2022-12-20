import benchmark_functions as bf
import numpy as np

from inspect import signature
from firefly import Firefly

TEST_NUM = 100

def line():
  print(u'\u2500' * 50)

results = np.empty([len(bf.BenchmarkFunction.__subclasses__()), 100])

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

    swarm = Firefly(function=instance, population=30, bounds=bounds, beta_0=1, light_absorption=0.8, randomisation_param = 1)
    
    for i in range(100):
      swarm.initialise_swarm()
      swarm.step(steps=100)
      results[index, i] = swarm.g_fitness


print(results)
average = np.mean(results, axis=1)
minimum = results.min(axis=1)
maximum = results.max(axis=1)
std = np.std(results, axis=1)
print("mean: ", average)
print("min: ", minimum)
print("max: ",  maximum)
print("std: ", std)


