from pso import Swarm, convert_to_bounds, find_neighbours
import numpy as np

def simple_fun(x, y):
  return(x - (x * y))

def simple_fun_list(param):
  return simple_fun(param[0], param[1])

def rosenbrock(param):
  f = 0

  for i in range(len(param) - 1):
    f += (100 * (param[i + 1] - param[i]**2)**2 + (1 - param[i])**2)

  return f

def test_init():
  swarm = Swarm(function=simple_fun, population = 50, bounds = [[0,1], [1,10]], vmax=1)
  if (swarm.population==50 and swarm.limits[1][1]==10 and swarm.vmax==1 and swarm.c1==2.8 and swarm.c2==1.3 and swarm.beta==1):

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
  
  if(swarm.position[0][0] < 0 or swarm.position[1][1] < 0 or swarm.position[1][0] > 1 or swarm.position[0][1] > 1):
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
  swarm.position = np.array([[0.],[0.],[0.]])
  swarm.velocity = np.array([[1.],[1.],[1.]])
  swarm.p_best = np.array([[1.],[0.],[2.]])
  swarm.g_best = 1
  np.random.seed(100)
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
  swarm.position = np.array([[0, .1, .1], [.1, .1, .1], [.2, .2, .1]])
  swarm.velocity = np.array([[0.1, 0.5, 0],[0.1, -0.4, 0.1],[-.1, -0.5, 0.1]])
  swarm.position = swarm.update_position()
  expected = [[0.1, .6, .1], [.2, 0, .2], [.1, 0, .2]]
  
  for i in range(swarm.position.shape[0]):
    for j in range(swarm.position.shape[1]):
      if swarm.position[i][j] != expected[i][j]:
        return "Position update incorrect, recieved {} , expected {}".format(swarm.position,expected)

  return True

def test_step():
  bounds = [[-1, 1], [-1, 1]]
  population = 10
  swarm = Swarm(function=simple_fun_list, population=population, bounds=bounds, vmax = 2)
  swarm.initialise_swarm()
  swarm.step(steps = 10)
  
  for pos in swarm.position:

    if pos[0] < bounds[0][0] or pos[0] > bounds[0][1] or pos[1] < bounds[1][0] or pos[1] > bounds[1][1]:
      return "swarm member(s) out of bounds" + str(swarm.g_best) + str(swarm.g_fitness)

    if swarm.g_fitness == -2:
      print("Global Optima found!")
      return True

  return "No explicit error, but search did not find global optima {}".format(swarm.g_fitness)
  
def test_rosenbrock_10_steps():
  bounds = [[-5,5]] * 30
  population = 30
  swarm = Swarm(function=rosenbrock, population=population, bounds=bounds, vmax = 0.75, beta = 0.79681, c1=2.8, c2=1.3)
  swarm.initialise_swarm()
  swarm.step(steps = 10)
  print(swarm.g_best)
  print("params: {}".format(convert_to_bounds(bounds, [swarm.g_best])))
  print(swarm.g_fitness)
  return True

def opt_test_fit_rosenbrock():
  bounds = [[-5,5]] * 30
  population = 30
  swarm = Swarm(function=rosenbrock, population=population, bounds=bounds, vmax = 0.5, beta = 0.41681, c1 = 3)
  swarm.initialise_swarm()
  swarm.fit(tol=0.000000000000001)
  print(swarm.g_best)
  print(swarm.g_fitness)
  return True

def test_convert_to_bounds():
  limits = np.array([[0.0001, 0.0009], [0.1, 0.9], [1, 5]])
  positions = np.array([[0.23, 0.12, 0.812], [0.46, 0.06, 0.406]])
  parameters = convert_to_bounds(limits, positions)
  expected = [[2.84e-4, 0.196, 4.248]]
  
  for i in range(len(expected)):
    if parameters[0][i] != expected[0][i]:
      return "Incorrect conversion, expected {}, but received {}".format(expected, parameters)

  return True 

def test_rosenbrock_50_dynamic_steps():
  bounds = [[-5,5]] * 30
  population = 30
  swarm = Swarm(function=rosenbrock, population=population, bounds=bounds, vmax = 0.5, beta = 0.6, c1=5.6, c2=2.6)
  swarm.initialise_swarm()
  swarm.step(steps = 10, dynamic_acc=True)
  print(swarm.g_best)
  print("params: {}".format(convert_to_bounds(bounds, [swarm.g_best])))
  print(swarm.g_fitness)
  return True

def test_find_neighbours():
  positions = np.array([[0,0],[0,1],[1,0],[1,1]])
  neighbourhoods = find_neighbours(positions, 3)

  if np.isin(3, neighbourhoods[0]) or np.isin(0, neighbourhoods[3]):
    return "neighbourhood calculated incorrectly, should return indicies of neighbourhood"

  return True

