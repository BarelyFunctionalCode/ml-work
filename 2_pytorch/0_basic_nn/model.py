import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

import os


# Standard nn model
class NeuralNetwork(nn.Module):
  def __init__(self, inputs, hidden_layers, hidden_layer_size, outputs):
    super().__init__()
    self.flatten = nn.Flatten()

    hidden_layer_objects = []
    for i in range(hidden_layers):
      hidden_layer_objects.append(nn.Linear(hidden_layer_size, hidden_layer_size))
      hidden_layer_objects.append(nn.ReLU())

    self.linear_relu_stack = nn.Sequential(
      nn.Linear(inputs, hidden_layer_size),
      nn.ReLU(),
      *hidden_layer_objects,
      nn.Linear(hidden_layer_size, outputs)
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

# Used to create model with desired parameters
def initialize_model(b_load_saved_model, input_count, hidden_layer_count, hidden_layer_size, output_count, loss_fn, optimizer, learning_rate, lr_decay_gamma, device):
  print("Initializing Model:")
  # Create model
  model = NeuralNetwork(input_count, hidden_layer_count, hidden_layer_size, output_count).to(device)

  # Load previously saved model if desired
  if os.path.isfile("model.pth") and b_load_saved_model:
    model.load_state_dict(torch.load("model.pth"))
    print("Loaded PyTorch Model State from model.pth")

  # Define loss function, optimizer, and learning rate decay
  loss_fn = loss_fn()
  optimizer = optimizer(model.parameters(), lr=learning_rate)
  scheduler = ExponentialLR(optimizer, gamma=lr_decay_gamma)
  print(model)
  return model, loss_fn, optimizer, scheduler