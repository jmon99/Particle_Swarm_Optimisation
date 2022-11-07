import numpy as np

"""
PSO class.
"""

class Swarm:

  def __init__(self,
    function,
    population,
    bounds,
    vmax,
    beta=1,
    c1=2.8,
    c2=1.3,
    max_iterations=200,
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
    self.bounds = bounds
    self.beta = beta
    self.c1 = c1
    self.c2 = c2
    self.max_iterations = max_iterations
    self.cost_bound = cost_bound
    self.vmax = vmax
    self.velocity = None
    self.position = None
    self.p_best = None
    self.g_best = None
    self.best_fitness = None
    self.g_fitness = None

  def initialise_swarm(self):
    """
    Initialises particles with random velocities and positions
    """
    lower_bounds = self.bounds[:,0]
    upper_bounds = self.bounds[:,1]
    self.velocity = np.random.uniform(0 - self.vmax, self.vmax, [self.population, len(lower_bounds)])
    self.position = np.random.uniform(lower_bounds, upper_bounds, [self.population, len(lower_bounds)])
    self.p_best = self.position
    self.best_fitness = np.array(list(map(self.function, self.position)))
    g_index = np.argmax(self.best_fitness)
    self.g_fitness = self.best_fitness[g_index]
    self.g_best = self.position[g_index]


  def update_velocity(self):
    np.random.seed(100)
    r1, r2 = np.random.uniform(0,1,[2,self.population])
    term1 = self.beta * self.velocity
    term2 = self.c1 * np.multiply(r1[:,np.newaxis],self.p_best)
    term3 = self.c2 * r2[:,np.newaxis] * self.g_best
    return np.add(np.add(term1, term2), term3)
    
  def update_position(self):
    return np.add(self.position, self.velocity)

  def step(self, steps = 1, tol = 0):

    for i in range(steps):
      self.velocity = self.update_velocity()
      print("vel", self.velocity)
      self.position = self.update_position()
      print("pos", self.position)
      cur_fitness = np.array(list(map(self.function, self.position)))

      for i in range(len(cur_fitness)):

        if cur_fitness[i] > self.best_fitness[i]:
          self.best_fitness[i] = cur_fitness[i]
          self.p_best[i] = self.position[i]
    
      if cur_fitness.max() > self.g_fitness:
        self.g_fitness = cur_fitness.max()

      print(self.best_fitness)


