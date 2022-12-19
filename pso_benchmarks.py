import benchmark_functions as bf
import numpy as np

from inspect import signature
from pso import Swarm

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

    swarm = Swarm(function=instance, population=30, bounds=bounds, vmax=0.6, beta=0.6, c1=2.8, c2=1.3)
    
    for i in range(100):
      swarm.initialise_swarm()
      swarm.step(steps=100)
      results[index, i] = swarm.g_fitness


print(results)
average = np.mean(results, axis=1)
print(average)
