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

    #check bounds is a numpy array of pairs
    if type(bounds) != 'numpy.ndarray':
      bounds = np.array(bounds)
    if bounds.shape[1] != 2:
      print("bounds should be a numpy  array of pairs of bounds in the format np.array([[x1_lower, x1_upper], [x2_lower, x2_upper], ...])")

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

    
  def initialise_swarm(self):
    self.velocity = np.random.uniform(0 - self.vmax, self.vmax, self.population)
    lower_bounds = self.bounds[:,0]
    upper_bounds = self.bounds[:,1]
    self.position = np.random.uniform(lower_bounds, upper_bounds)

  
