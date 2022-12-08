import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda

from torch.optim import SGD,Adam,RMSprop
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from pso import Swarm

for i in range(torch.cuda.device_count()):
  print("device: {}".format(torch.cuda.get_device_name(i)))

dtype = torch.float
device = "cuda:3" if torch.cuda.is_available() else "cpu"
device = torch.device("mps")
print(f"Using {device} device")
print(device)

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

class Cnn(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    return x


def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.to(device)
  model.train(mode=True)

  for batch, (X,y) in enumerate(dataloader):
    X,y = X.to(device), y.to(device)
    
    #forward pass
    pred = model(X)
    loss = loss_fn(pred, y)
    
    #back propagation
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.to(device)
  model.eval()
  test_loss, correct = 0, 0

  with torch.no_grad():

    for X, y in dataloader:
      X,y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  return test_loss

training_data = datasets.FashionMNIST(
  root="data",
  train=True,
  download=True,
  transform=ToTensor()
)

test_data = datasets.FashionMNIST(
  root="data",
  train=False,
  download=True,
  transform=ToTensor()
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)
model = Cnn().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 5

"""
for t in range(epochs):
  print(f"Epoch {t+1}\n-------------------------------")
  train(train_dataloader, model, loss_fn, optimizer)
  loss = test(test_dataloader, model, loss_fn)
print("Done!")
"""

def pso_wrapper(params):
  print(params)
  model = Cnn().to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr=params[0], momentum=params[1])
  
  for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    loss = test(test_dataloader, model, loss_fn)
    
  torch.cuda.empty_cache(),
  
  return(loss)

swarm = Swarm(pso_wrapper, 5, [[0.00000001,0.1], [0, 1]], 0.75, beta=0.8)
swarm.initialise_swarm()
swarm.fit(tol=0.001)
print(swarm.g_best)
print(swarm.g_fitness)
