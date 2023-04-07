import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn

import math

from envs.ImaBust import ImaBustEnv
from agent import Agent
from test_games.ImaBust import ImaBust



# model hyperparameters
b_load_saved_agent = False           # Load previously trained agent
batch_size = 1000                    # Random sample size from replay memory when optimizing policy
num_episodes = 20000                 # Max episodes for training
success_threshold = 20               # Metric to determine solved model and end training early i.e game wins
state_size_1d = 4                    # 1D size of state used for env and agent
loss_func = nn.CrossEntropyLoss      # Loss function used in agent's model optimization
optimizer_option = torch.optim.AdamW # Optimizer used in agent's model optimization
learning_rate = 1e-2                 # Optimizer's learning rate
gamma = 0.99                         # Discount future action reward factor
eps_decay = 0.9994                   # Decay rate of whether to choose a random action or learned action
tau = 0.001                          # Soft update ratio from policy network to fixed target network



# Get cpu or gpu device for training.
# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


min_value = 0 # Minimum value observed in env state
max_value = 8 # Maximum value observed in env state
options_count = max_value - min_value + 1 # Number of options available to agent

# TODO: replace program interface with a window handle and utilities for keyboard/mouse inputs to the program
# initialize Program Interface
print("Initializing Program Interface...")
program_interface = ImaBust(state_size_1d, min_value, max_value, 512, 60)

# initialize Environment
print("Initializing Env...")
env = ImaBustEnv(state_size_1d, min_value, max_value, program_interface)

# Initialize agent
print("Initializing Agent...")
estimate_max_duration = (state_size_1d ** 2) * (max_value - min_value)
agent = Agent(
      b_load_saved_agent,
      state_size_1d,
      min_value,
      max_value,
      options_count,
      batch_size,
      loss_func,
      optimizer_option,
      learning_rate,
      gamma,
      tau,
      eps_decay,
      estimate_max_duration,
      success_threshold,
      device
    )

# Determine episodes to train for based on if GPU is available
if torch.cuda.is_available():
  num_episodes_for_device = num_episodes
else:
  num_episodes_for_device = num_episodes / 5

# Start training loop
print("Training")
for i in range(num_episodes_for_device):
  current_success_threshold = agent.train(env)
  if current_success_threshold >= success_threshold: break

# Display and save final agent data
print("Saving model and showing final results")
agent.finish_and_save()
