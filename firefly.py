import numpy as np

def convert_to_bounds(bounds, positions):
  values = np.zeros_like(positions)
  
  for i, pos in enumerate(positions):
    for j in range(len(bounds)):
      values[i][j] = bounds[j][0] + ((bounds[j][1] - bounds[j][0]) * pos[j])
  
  return values

def distance(i, j):
  return np.linalg.norm(i - j)

class Firefly:

  def __init__(self,
    function,
    population,
    bounds,
    beta_0,
    light_absorption,
    randomisation_param,
    final_randomisation_param = None):

    self.function = function
    self.population = population
    self.limits = bounds
    self.bounds = np.array([[0,1]] * len(bounds))
    self.beta_0 = beta_0
    self.gamma = light_absorption
    self.alpha = randomisation_param
    self.alpha_inf = final_randomisation_param
    self.position = None
    self.attractiveness = None
    self.g_fitness = None
    self.g_best = None
    self.cur_fitness = None
   
  def initialise_swarm(self):
    lower_bounds = self.bounds[:,0]
    upper_bounds = self.bounds[:,1]
    self.position = np.random.uniform(lower_bounds, upper_bounds, [self.population, len(lower_bounds)])
    params = convert_to_bounds(self.limits, self.position)
    self.attractiveness = np.fromiter(map(self.function, params), dtype=np.float32)
    self.g_fitness = self.attractiveness.min()
    self.g_best = self.position[np.argmin(self.attractiveness)]

  def step(self, steps=1):

    for step in range(steps):
      #update randomisation parameter
      self.alpha *= 0.98

      for i,pos in enumerate(self.position):

        for j,firefly in enumerate(self.position):

          if self.attractiveness[j] < self.attractiveness[i]:
            exponent = -self.gamma * distance(i,j)**2
            add = np.array(self.beta_0 * np.exp(exponent) * (firefly - pos) + self.alpha*np.random.uniform(-0.5,0.5,len(self.bounds)))
            self.position[i] = self.position[i] + add 

      for i,pos in enumerate(self.position):

       for j,bound in enumerate(self.bounds):

         if pos[j] < bound[0]:
           pos[j] = bound[0]

         if pos[j] > bound[1]:
           pos[j] = bound[1]

      params = convert_to_bounds(self.limits, self.position)
      self.attractiveness = np.fromiter(map(self.function, params), dtype=np.float32)
      g_fitness = self.attractiveness.min()
      if g_fitness < self.g_fitness:
        self.g_fitness = g_fitness

      print("step{}: swarm fitness = {}".format(step, g_fitness))


