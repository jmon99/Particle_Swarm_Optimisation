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

    self.function = function
    self.population = population
    self.bounds = bounds
    self.beta = beta
    self.c1 = c1
    self.c2 = c2
    self.max_iterations = max_iterations
    self.cost_bound = cost_bound
    self.vmax = vmax

    



