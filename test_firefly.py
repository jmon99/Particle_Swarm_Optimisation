import numpy as np

from firefly import Firefly, distance

def sum_fun(param):
  sum  = 0

  for num in param:
    sum += num

  return sum

def rosenbrock(param):
  f = 0

  for i in range(len(param) - 1):
    f += (100 * (param[i + 1] - param[i]**2)**2 + (1 - param[i])**2)

  return f


def test_init():
  swarm = Firefly(function = rosenbrock, population = 10, bounds= np.array([[-5,5],[-5,5]]), beta_0 = 1 ,light_absorption = 0.5, randomisation_param = 0.5)
  expected_search_bounds = np.array([[0, 1], [0, 1]])
  expected_parameter_bounds = np.array([[-5, 5], [-5, 5]])

  for i in range(len(expected_search_bounds)):
    if expected_search_bounds[i][0] != swarm.bounds[i][0] or swarm.bounds[i][1] != swarm.bounds[i][1]:
      return "incorrect bounds on search space. recieved {}, expected {}".format(swarm.bounds, expected_search_bounds)

  for i in range(len(expected_parameter_bounds)):
    if expected_parameter_bounds[i][0] !=  swarm.limits[i][0] or expected_parameter_bounds[i][1] != swarm.limits[i][1]:
      return "incorrect bounds on parameter space. recieved {}, expected{}".format(swarm.limits, expected_parameter_bounds)

  return True

def test_initialise_swarm():
  swarm = Firefly(function=sum_fun, population = 1, bounds = np.array([[0,10], [0,10]]), beta_0 = 1, light_absorption = 0.5, randomisation_param = 0.5)
  swarm.initialise_swarm()

  for i, pos in enumerate(swarm.position):

    print(swarm.attractiveness)
    if (swarm.attractiveness[i], 2) != ((pos[0] + pos[1]) * 10, 2):
      return "Attractiveness calculated incorectly, should use fitness function. recieved {}, expected {}".format((pos[0] + pos[1])*10, swarm.attractiveness[i])

  print("pos: ", swarm.position)
  return True

def test_distance():
  i = np.array([0.5, 1, 0.3])
  j = np.array([0.2, 0.9, 0.4])
  expected = 0.3316625
  result = distance(i, j)
  result = np.round(result, 7)

  if expected != result:
    return "incorrect distance, ecpected {}, but recieved {}".format(expected, result)

  return True

def test_step():
  swarm = Firefly(function=sum_fun, population = 30, bounds = np.array([[0,1],[0,1]]),beta_0 = 1 ,light_absorption=0.01, randomisation_param = 1)
  swarm.position = np.array([[0,0],[0.5,0.5]])
  swarm.attractiveness = np.array([0,1])
  swarm.step(steps = 2)
  print(swarm.position)
  print(swarm.attractiveness)
  return True

def test_rosenbrock():
  bounds = np.array([[-5, 5]]*30) 
  population = 30
  swarm = Firefly(function = rosenbrock, population = population, bounds = bounds, beta_0 = 1, light_absorption=0.01, randomisation_param=1)
  swarm.initialise_swarm()
  swarm.step(steps = 500)

