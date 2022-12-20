import numpy as np


"""
PSO class.
"""

def convert_to_bounds(bounds, positions):
  values = np.zeros_like(positions)

  for i, pos in enumerate(positions):
    for j in range(len(bounds)):
      values[i][j] = bounds[j][0] + ((bounds[j][1] - bounds[j][0]) * pos[j])

  return values


def find_neighbours(position, k):
  """
  Returns neighbourhoood of size k for each particle in an array of positions. Each neighbourhood consisting of the particle
  and its k-1 nearest neighbours in the search space

  param position -> numpy array containing positions of swarm particles.
  param k -> size of the neighbourhood to be returned
  returns numpy array of neighbourhoods index positions in given position array
  """

  population = len(position)
  distance_matrix =  np.empty([population, population])
  neighbourhoods = np.empty([population, population])
  distance_matrix = np.sqrt((position**2).sum(axis=1)[:, np.newaxis] + (position**2).sum(axis=1) - 2 * position.dot(position.T))
  indicies = np.argpartition(distance_matrix, k, axis=1)
  indicies = indicies[:,:k]

  return indicies

class Swarm:

  def __init__(self,
    function,
    population,
    bounds,
    vmax,
    beta=1,
    c1=2.8,
    c2=1.3,
    cost_bound=None):

    """
    Initailise Swarm object.

    Parameters:
    function -> the fitness function to be optimised
    population -> the size of the swarm
    bounds -> numpy array of pairs of the lower and upper bound of each parameter
    	      given in the format [[x1_lower, x1_upper][x2_lower, x2_upper],...]
    vmax -> maximum velocity value
    beta, c1, c2 -> paramaters of the PSO algorithm
    max_iterations -> Stop condition, algorithm will stop after given number of iterations
    cost_bound -> Stop condition, algorithm will stop when fitness function passes cost_bound 
    """

    #check bounds is a numpy array of pairs
    if type(bounds) != 'numpy.ndarray':
      bounds = np.array(bounds)
    if len(bounds.shape) != 2:
      raise Exception("bounds should be a numpy  array of pairs of bounds in the format np.array([[x1_lower, x1_upper], [x2_lower, x2_upper], ...])")
    if bounds.shape[1] != 2:
      raise Exception("bounds should be a numpy  array of pairs of bounds in the format np.array([[x1_lower, x1_upper], [x2_lower, x2_upper], ...])")

    self.function = function
    self.population = population
    self.bounds = np.array([[0,1]] * len(bounds))#bounds used during search
    self.limits = bounds #value bounds used for parameter assignment of fitness function
    self.beta = beta
    self.c1 = c1
    self.c2 = c2
    self.cost_bound = cost_bound
    self.vmax = vmax
    self.velocity = None
    self.position = None
    self.p_best = None
    self.g_best = None
    self.best_fitness = None
    self.g_fitness = None

  def initialise_swarm(self, k=None):
    """
    Initialises particles with random velocities and positions
    """
    lower_bounds = self.bounds[:,0]
    upper_bounds = self.bounds[:,1]
    self.velocity = np.random.uniform(0 - self.vmax, self.vmax, [self.population, len(lower_bounds)])
    self.position = np.random.uniform(lower_bounds, upper_bounds, [self.population, len(lower_bounds)])
    self.p_best = self.position
    params = convert_to_bounds(self.limits, self.position)
    self.best_fitness = np.fromiter(map(self.function, params), dtype=np.float32)
    g_index = np.argmin(self.best_fitness)
    self.swarm_fitness = self.g_fitness = self.best_fitness[g_index]
    self.g_best = self.position[g_index]
    
    if k != None:
      neighbourhoods = find_neighbours(self.position, k)
      self.n_best = np.empty_like(self.positions)

      for i, neighbourhood in neighbourhoods:
        n_fitness = np.take(self.best_fitness, neighbourhoods)
        self.n_best[i] = n_fitness.max()

  def update_velocity(self):
    """
    Updates the array of velocity values self.velocity . Each element represents the velocity of a particle.
    If any of the new velocity values exceed vmax in magnitude they are reduced to match it. 
    """
    r1, r2 = np.random.uniform(0,1,[2,self.population])
    term1 = self.beta * self.velocity
    term2 = self.c1 * np.multiply(r1[:,np.newaxis],(self.p_best - self.position))
    term3 = self.c2 * r2[:,np.newaxis] * (self.g_best - self.position)
    self.velocity = np.add(np.add(term1, term2), term3)
    norms = np.linalg.norm(self.velocity, axis = 1)
    
    for i, norm in enumerate(norms):
      if norm > self.vmax:
        self.velocity[i] = self.velocity[i]/norm
        self.velocity[i] = self.velocity[i] * self.vmax
     
    return self.velocity

  def update_position(self):
    """
    Updates the position of each particle, in the array self.position
    """
    position = np.add(self.position, self.velocity)
    
    for i,pos in enumerate(position):
      for j,bound in enumerate(self.bounds):
        if pos[j] < bound[0]:
          pos[j] = bound[0]
        if pos[j] > bound[1]:
          pos[j] = bound[1]

    return position

  def step(self, steps = 1, dynamic_acc = False):
    """
    Performs a single step of the PSO algoritm, or the number of steps given in the optional step parameter
    """
    delta = 2

    for i in range(steps):
      self.c1 = self.c1 / (1  + ((i)/steps))
      self.c2 = self.c2 * (((delta * i) + steps)/((delta * steps) + i))
      params = convert_to_bounds(self.limits, self.position)
      cur_fitness = np.fromiter(map(self.function, params), dtype=np.float32)

      for j in range(len(cur_fitness)):

        if cur_fitness[j] < self.best_fitness[j]:
          self.best_fitness[j] = cur_fitness[j]
          self.p_best[j] = self.position[j]
    
      self.swarm_fitness = cur_fitness.min()

      if self.swarm_fitness < self.g_fitness:
        self.g_fitness = self.swarm_fitness
        g_index = np.argmin(self.best_fitness)
        self.g_best = self.position[g_index]

      self.velocity = self.update_velocity()
      self.position = self.update_position()
      print("step{} Current swarm fitness: {}".format(i, self.swarm_fitness))

  def fit(self, tol = 0.00001, max_iter = None):
    """
    Performs PSO on the function self.function given as an attribute when __init__ was called.
    PSO continues until the change in best value(current best swarm value, not historuc best) is
    less than the tolerance value.

    tol -> optional parameter to set the tolerance, if not provided the default is 0.00001 
    """
    old_fit = self.swarm_fitness
    self.step(steps = 2)

    while np.abs(self.swarm_fitness - old_fit) > tol:
      old_fit = self.swarm_fitness
      self.step()


