from firefly import Firefly
from pso import Swarm

def rosenbrock(param):
  f = 0

  for i in range(len(param) - 1):
    f += (100 * (param[i + 1] - param[i]**2)**2 + (1 - param[i])**2)

  return f

bounds = [[-5,10]] * 30
population = 100
step_num = 30
pso_sum = 0

for i in range(100):
  swarm = Swarm(function=rosenbrock, population=population, bounds=bounds, vmax = 0.6, beta = 0.5, c1=1.5, c2=1.5)
  swarm.initialise_swarm()
  swarm.step(steps = step_num)
  pso_sum += swarm.g_fitness

pso = pso_sum/100

pso_dynamic_sum = 0

for i in range(100):
  swarm = Swarm(function=rosenbrock, population=population, bounds=bounds, vmax = 0.5, beta = 0.6, c1=5.6, c2=2.6)
  swarm.initialise_swarm()
  swarm.step(steps = step_num, dynamic_acc=True)
  pso_dynamic_sum += swarm.g_fitness

pso_dynamic = pso_dynamic_sum / 100

firefly_sum = 0

for i in range(100):
  swarm = Firefly(function = rosenbrock, population = population, bounds = bounds, beta_0 = 1, light_absorption=1, randomisation_param=1)
  swarm.initialise_swarm()
  swarm.step(steps = step_num)
  firefly_sum += swarm.g_fitness

firefly = firefly_sum/100

print("pso: ", pso)
print("dynamic pso: ", pso_dynamic)
print("firefly: ", firefly)


