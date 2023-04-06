import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

import os

# CNN model
class CNN(nn.Module):
  # ((n-f+2p)/s)+1

  # where n is the pixels of the image i.e. 32
  # f is the number of kernels, in our case it is 5*5 kernel which mean f = 5
  # p is the padding, p = 0
  # s is the stride, s = 0
  def __init__(self, state_size_1d, options_count):
    super().__init__()
    self.kernal_size = 3
    self.out_channels = 20
    self.hidden_layer_ratio = 5
    self.conv_output_size = ((state_size_1d - self.kernal_size + 1) ** 2) * self.out_channels
    self.hidden_layer_size = self.conv_output_size * self.hidden_layer_ratio
    self.output_size = state_size_1d ** 2

    self.cnn1 = nn.Conv2d(options_count, self.out_channels, self.kernal_size)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(self.conv_output_size, self.hidden_layer_size)
    self.fc2 = nn.Linear(self.hidden_layer_size, self.output_size)

  def forward(self, x, random_action_batch=None):
    x = F.relu(self.cnn1(x))
    x = self.flatten(x)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    # x = F.log_softmax(x, dim=1)
    # if random_action_batch is None:
    #   random_action_batch = x
    # x = x * random_action_batch
    return x

# Used to create model with desired parameters
def initialize_model(b_load_saved_model, state_size_1d, options_count, optimizer, learning_rate, device):
  # Create model
  model = CNN(state_size_1d, options_count).to(device)

  # Load previously saved model if desired
  if os.path.isfile("model.pth") and b_load_saved_model:
    model.load_state_dict(torch.load("model.pth"))
    print("Loaded PyTorch Model State from model.pth")

  # Define loss function, optimizer, and learning rate decay
  optimizer = optimizer(model.parameters(), lr=learning_rate)
  return model, optimizer