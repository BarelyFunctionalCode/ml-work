import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn

from model import initialize_model
from utils import train, test, generate_random_float_array_dataset


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


# model hyperparameters
b_load_saved_model = False

hidden_layer_count = 1
hidden_layer_size = 25

loss_fn = nn.L1Loss
optimizer = torch.optim.SGD
learning_rate = 1.0
lr_decay_gamma = 0.999

epochs = 10000

# debug settings
print_frequency = 1000
print_results_limit = 10

# training data generation
batch_size = 150
batch_count = 5
input_count = 10
output_count = 2
min_output = -1
max_output = 1

# generating dataset
train_dataloader, test_dataloader = generate_random_float_array_dataset(
                                      batch_size,
                                      batch_count,
                                      input_count,
                                      output_count,
                                      min_output,
                                      max_output
                                    )

# initialize model
model, loss_fn, optimizer, scheduler = initialize_model(
                                          b_load_saved_model,
                                          input_count,
                                          hidden_layer_count,
                                          hidden_layer_size,
                                          output_count,
                                          loss_fn,
                                          optimizer,
                                          learning_rate,
                                          lr_decay_gamma,
                                          device
                                        )

# run model training
print("Start Training:")
for t in range(epochs):
  print_results = True if t % print_frequency == 0 or t == epochs else False
  if print_results: print(f"Epoch {t+1}\n-------------------------------")
  train(device, train_dataloader, model, loss_fn, optimizer, scheduler, print_results)
  test_pred, test_y = test(device, test_dataloader, model, loss_fn, print_results)

# print final test results
print("\n\nTesting Results:")
i = 0
for t_p, t_y in zip(test_pred, test_y):
  if i >= print_results_limit: break
  print(f"y: {t_y.round(decimals=3)}\tpred: {t_p.round(decimals=3)}")
  i += 1

torch.save(model.state_dict(), "model.pth")
print("\n\nSaved PyTorch Model State to model.pth")