from pso import Swarm
import numpy as np

def simple_fun(x, y):
  return(x - (x * y))

def simple_fun_list(param):
  return simple_fun(param[0], param[1])

def test_init():
  swarm = Swarm(function=simple_fun, population = 50, bounds = [[0,1], [1,10]], vmax=1)
  if (swarm.population==50 and swarm.bounds[1][1] == 10 and swarm.vmax==1 and swarm.c1==2.8 and swarm.c2==1.3 and swarm.beta==1 and swarm.max_iterations==200):

    return True

  return "Test Failed, expected attributes either not stored, or are inaccuate to given input parameters."

def test_initial_velocity():
  bounds = np.array([[0,1], [1,10]])
  population = 50
  swarm = Swarm(function=simple_fun_list, population = population, bounds=bounds, vmax=1)
  swarm.initialise_swarm()

  if swarm.velocity.shape != (population, 2):
    return "Test failed, velocity should be a [pop, param] in this case [50,2] array, instead was {}".format(swarm.velocity.shape)

  if(swarm.velocity[0][0] > 1 or swarm.velocity[1][1] > 1):
    line()
    return "Test failed, initial velocity is too high :("

  if(swarm.velocity[0][1] < -1 or swarm.velocity[1][0] < -1):
    line()
    return "Initial Velocity Test failed, initial velocity is too Low :("
    return False

  return True

def test_initial_position():
  bounds = np.array([[0,1], [1,10]])
  population = 50
  swarm = Swarm(function=simple_fun_list, population=population, bounds=bounds, vmax=1)
  swarm.initialise_swarm()

  if(swarm.position.shape != swarm.velocity.shape):
    return "swarm.postion initialised with wrong shape"
  
  if(swarm.position[0][0] < 0 or swarm.position[1][1] < 1 or swarm.position[1][0] > 1 or swarm.position[0][1] > 10):
    return "Inital Position Test Failed, postion {} is outside of given bounds".format(swarm.position)

  return True

def test_dimension_check_for_bounds():
  bounds = np.array([[1,1,1], [2,4,5], [1,6,7]])
  population = 50

  try:
    swarm = Swarm(function=simple_fun_list, population=population, bounds=bounds, vmax=1)
    swarm.initialise_swarm()
    return "Failed to raise Exception"
  
  except:
    return True 

def test_check_data_type_for_bounds1():
  bounds = [[0,1],[5,10],[2,6]]
  population = 50

  try:
    swarm = Swarm(function=simple_fun_list, population=population, bounds=bounds, vmax=1)
    swarm.initialise_swarm()
    return True
  except:
    return "Failed to convert python list"

def test_check_data_type_for_bounds2():
  bounds = 10
  population = 50
  
  try:
    swarm = Swarm(function=simple_fun_list, population=population, bounds=bounds, vmax=1)
    swarm.initialise_swarm()
    return "Failed to through error when wrong type given for bounds"
  except:
    return True
     
def test_update_velocity():
  bounds = [[0,1]]
  population = 3
  swarm = Swarm(function=simple_fun_list, population=population, bounds=bounds, vmax=10)
  swarm.position = [[0],[0],[0]]
  swarm.velocity = [[1],[1],[1]]
  swarm.p_best = [[1],[0],[2]]
  swarm.g_best = 1
  swarm.velocity = swarm.update_velocity()
  expected_vol = [[3.62],[1.01],[3.54]]
  round_res = np.round_(swarm.velocity, decimals = 2)

  for i in range(len(expected_vol)):
    if round_res[i][0] != expected_vol[i][0]:
      return "Updated velocities incorrect. Recieved {}, expected {}".format(round_res, expected_vol)
  
  return True

def test_update_position():
  bounds = [[0,1], [-1,2], [-2,2]]
  population = 3
  swarm = Swarm(function=simple_fun_list, population=population, bounds=bounds, vmax=1)
  swarm.position = np.array([[0, 1, 1], [1, 1, 1], [2, 2, 1]])
  swarm.velocity = np.array([[0.1, 0.5, 0],[0.1, -0.4, 0.2],[-1, -0.5, 0.1]])
  swarm.position = swarm.update_position()
  expected = [[0.1, 1.5, 1], [1., 0.6, 1.2], [1., 1.5, 1.1]]
  
  for i in range(swarm.position.shape[0]):
    for j in range(swarm.position.shape[1]):
      if swarm.position[i][j] != expected[i][j]:
        return "Position update incorrect, recieved {} , expected {}".format(swarm.position,expected)

  return True

def test_step():
  bounds = [[-1, 1], [-1, 1]]
  population = 10
  swarm = Swarm(function=simple_fun_list, population=population, bounds=bounds, vmax = 0.1)
  swarm.initialise_swarm()
  swarm.step(steps = 100, tol = 0.01)
  
  for pos in swarm.position:

    if pos[0] < bounds[0][0] or pos[0] > bounds[0][1] or pos[1] < bounds[1][0] or pos[1] > bounds[1][1]:
      return "swarm member(s) out of bounds" + str(swarm.g_best) + str(swarm.g_fitness)

    if swarm.g_fitness == 2:
      print("Global Optima found!")
      return True

  return "No explicit error, but search did not find global optima"
  

