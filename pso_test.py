from pso import Swarm
import numpy as np

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
  if (swarm.population==50 and swarm.bounds[1][1] == 10 and swarm.vmax==1 and swarm.c1==2.8 and swarm.c2==1.3 and swarm.beta==1 and swarm.max_iterations==200):

    print("Test Passed!")
    line()
    return True

  print("Test Failed :( ")
  line()
  return False

def test_initial_vel():
  line()
  print("Initial Velocity Test")
  line()
  bounds = np.array([[0,1], [1,10]])
  population = 50
  swarm = Swarm(function=simple_fun, population = population, bounds=bounds, vmax=1)
  swarm.initialise_swarm()

  if(swarm.velocity[0] > 1 or swarm.velocity[1] > 1):
    line()
    return "Test failed, initial velocity is too high :("

  if(swarm.velocity[0] < -1 or swarm.velocity[1] < -1):
    line()
    return "Initial Velocity Test failed, initial velocity is too Low :("
    return False

  print("Test passed!")
  line()
  return True

def test_initial_pos():
  line()
  print("Initial Position Test")
  line()
  bounds = np.array([[0,1], [1,10]])
  population = 50
  swarm = Swarm(function=simple_fun, population=population, bounds=bounds, vmax=1)
  swarm.initialise_swarm()
  
  if(swarm.position[0] < 0 or swarm.position[1] < 1 or swarm.position[0] > 1 or swarm.position[1] > 10):
    line()
    return "Inital Position Test Failed, postion is outside of given bounds"

  return True

tests = [test_init, test_initial_vel, test_initial_pos]
total_tests = len(tests)
passes = 0

for test in tests:
  passed = test()
  if passed == True:
    passes += 1
  else: print(passed)

line()
print("Passed ",passes," out of",total_tests," tests!")
  
