import benchmark_functions as bf

TEST_NUM = 100

def line():
  print(u'\u2500' * 50)

for func in bf.BenchmarkFunction.__subclasses__():
    instance = func()
    print(instance.name()) 


