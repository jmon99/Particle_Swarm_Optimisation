from pso import Swarm
import numpy as np

def simple_fun(x, y):
  return(x - (x * y))

def test_init():
  swarm = Swarm(function=simple_fun, population = 50, bounds = [[0,1], [1,10]], vmax=1)
  if (swarm.population==50 and swarm.bounds[1][1] == 10 and swarm.vmax==1 and swarm.c1==2.8 and swarm.c2==1.3 and swarm.beta==1 and swarm.max_iterations==200):

    return True

  return "Test Failed, expected attributes either not stored, or are inaccuate to given input parameters."

def test_initial_velocity():
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

  return True

def test_initial_position():
  bounds = np.array([[0,1], [1,10]])
  population = 50
  swarm = Swarm(function=simple_fun, population=population, bounds=bounds, vmax=1)
  swarm.initialise_swarm()
  
  if(swarm.position[0] < 0 or swarm.position[1] < 1 or swarm.position[0] > 1 or swarm.position[1] > 10):
    line()
    return "Inital Position Test Failed, postion is outside of given bounds"

  return True

def test_dimension_check_for_bounds():
  bounds = np.array([[1,1,1], [2,4,5], [1,6,7]])
  population = 50

  try:
    swarm = Swarm(function=simple_fun, population=population, bounds=bounds, vmax=1)
    swarm.initialise_swarm()
    return "Failed to raise Exception"
  
  except:
    return True 

def test_check_data_type_for_bounds1():
  bounds = [[0,1],[5,10],[2,6]]
  population = 50

  try:
    swarm = Swarm(function=simple_fun, population=population, bounds=bounds, vmax=1)
    swarm.initialise_swarm()
    return True
  except:
    return "Failed to convert python list"

def test_check_data_type_for_bounds2():
  bounds = 10
  population = 50

  try:
    swarm = Swarm(function=simple_fun, population=population, bounds=bounds, vmax=1)
    swarm.initialise_swarm()
    return "Failed to through error when wrong type given for bounds"
  except:
    return True
     
   
  
  
