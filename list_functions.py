import benchmark_functions as bf
import numpy as np

from inspect import signature

TEST_NUM = 100

def line():
  print(u'\u2500' * 50)

results = np.empty([len(bf.BenchmarkFunction.__subclasses__()), 100])

for index, func in enumerate(bf.BenchmarkFunction.__subclasses__()):
    func_args = signature(func).parameters

    if func_args.get('n_dimensions') != None:
      instance = func(n_dimensions=30)

    else: instance = func()

    print(instance.name()) 
