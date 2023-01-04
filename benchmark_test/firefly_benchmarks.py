import benchmark_functions as bf
import numpy as np

from inspect import signature
from firefly import Firefly

#test parameters
TEST_NUM = 100
pop=30
beta_0=1.5
gamma=0.8
alpha=1

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

    swarm = Firefly(function=instance, population=pop, bounds=bounds, beta_0=beta_0, light_absorption=gamma, randomisation_param = alpha)
    
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

print("swarm size={}".format(pop))
print("firefly: beta_0={}, gamma={}, alpha={}".format(beta_0, gamma, alpha))
print(" Func | avrg | min  | max  | std  |")

for index, func in enumerate(bf.BenchmarkFunction.__subclasses__()):
  instance=func()
  iavg = np.format_float_scientific(average[index], precision=4)
  imin = np.format_float_scientific(minimum[index], precision=4)
  imax = np.format_float_scientific(maximum[index], precision=4)
  istd = np.format_float_scientific(std[index], precision=4)
  print("{} & {} & {} & {} & {} ".format(instance.name(), iavg, imin, imax, istd))

