from pso import Swarm

def line():
  print("--------------------------------------------------------------------------------------------------")

def simple_fun(x, y):
  return(x - (x * y))

def test_init():
  line()
  print("__init__ Test")
  line()
  print("Testing object and attribute initialisation...")
  swarm = Swarm(function=simple_fun, population = 50, bounds = [[0,1], [1,10]], vmax=1)
  if (swarm.population==50 and swarm.bounds[1]==[1,10] and swarm.vmax==1 and swarm.c1==2.8 and swarm.c2==1.3 and swarm.beta==1 and swarm.max_iterations==200):

    print("Test Passed!")
    line()
    return True

  print("Test Failed :( ")
  line()
  return False

tests = [test_init]
total_tests = len(tests)
passes = 0

for test in tests:
  if test():
    passes += 1

line()
print("Passed ",passes," out of",total_tests," tests!")
  
